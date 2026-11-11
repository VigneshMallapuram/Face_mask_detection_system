"""Face-mask inference and webcam streaming helpers."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class MaskDetector:
    """Run face detection and mask classification on webcam frames."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        cascade_path: str | os.PathLike[str] | None = None,
        image_size: tuple[int, int] = (100, 100),
        alert_cooldown: float = 3.0,
        min_face_size: int = 70,
        min_neighbors: int = 8,
        confidence_threshold: float = 0.75,
    ) -> None:
        try:
            from tensorflow.keras.models import load_model
        except ImportError as error:
            raise RuntimeError(
                "TensorFlow could not load its native Windows DLLs. "
                "This is commonly caused by an Application Control policy "
                "or an unsupported Python/TensorFlow combination. "
                "Use a supported Python version in a fresh virtual environment "
                "or ask your administrator to allow the TensorFlow DLLs."
            ) from error

        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"Mask model not found at '{model_file}'. "
                "Train it first with: python train.py --data-dir data"
            )

        cascade_file = Path(cascade_path) if cascade_path else Path(
            cv2.data.haarcascades
        ) / "haarcascade_frontalface_default.xml"
        if not cascade_file.is_file():
            raise FileNotFoundError(f"Face cascade not found at '{cascade_file}'.")

        self.model = load_model(model_file)
        self.face_cascade = cv2.CascadeClassifier(str(cascade_file))
        if self.face_cascade.empty():
            raise RuntimeError(f"Unable to load face cascade from '{cascade_file}'.")
        self.image_size = image_size
        self.min_face_size = min_face_size
        self.min_neighbors = min_neighbors
        self.confidence_threshold = confidence_threshold
        self.alert_cooldown = alert_cooldown
        self._last_alert = 0.0
        self._alert_lock = threading.Lock()

    def _predict_mask(self, face: np.ndarray) -> tuple[str, float]:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.image_size)
        sample = resized.astype("float32") / 255.0
        sample = sample.reshape(1, self.image_size[1], self.image_size[0], 1)
        prediction = np.asarray(self.model.predict(sample, verbose=0))

        if prediction.shape[-1] == 1:
            mask_probability = float(prediction.reshape(-1)[0])
        elif prediction.shape[-1] >= 2:
            mask_probability = float(prediction.reshape(-1, prediction.shape[-1])[0][1])
        else:
            raise ValueError("The mask model returned an empty prediction.")

        mask_probability = float(np.clip(mask_probability, 0.0, 1.0))
        if mask_probability >= 0.5:
            label, confidence = "Mask", mask_probability
        else:
            label, confidence = "No Mask", 1.0 - mask_probability
        if confidence < self.confidence_threshold:
            label = "Uncertain"
        return label, confidence

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Annotate a BGR frame and return it with whether an alert is active."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
        )
        no_mask_seen = False

        for x, y, width, height in faces:
            face = frame[y : y + height, x : x + width]
            label, confidence = self._predict_mask(face)
            no_mask = label == "No Mask"
            no_mask_seen = no_mask_seen or no_mask
            color = (0, 0, 255) if no_mask else (0, 180, 180)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(
                frame,
                f"{label}: {confidence:.0%}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        if no_mask_seen:
            cv2.putText(
                frame,
                "ALERT: MASK REQUIRED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
        return frame, no_mask_seen

    def should_alert(self, no_mask_seen: bool) -> bool:
        """Rate-limit alert events so a continuous violation is not noisy."""
        if not no_mask_seen:
            return False
        with self._alert_lock:
            now = time.monotonic()
            if now - self._last_alert < self.alert_cooldown:
                return False
            self._last_alert = now
            return True


def webcam_frames(
    detector: MaskDetector, camera_index: int = 0
) -> Iterator[tuple[bytes, bool]]:
    """Yield JPEG-encoded annotated frames from a webcam."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if os.name == "nt" else [cv2.CAP_ANY]
    camera = None
    for backend in backends:
        candidate = cv2.VideoCapture(camera_index, backend)
        if candidate.isOpened():
            camera = candidate
            break
        candidate.release()
    if camera is None:
        raise RuntimeError(
            f"Unable to open webcam {camera_index}. Check camera permissions, "
            "whether another application is using it, and try CAMERA_INDEX=1."
        )

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError(
                    "The webcam opened but returned no frame. Close other camera "
                    "apps, check Windows camera permissions, or try CAMERA_INDEX=1."
                )
            annotated, no_mask_seen = detector.process_frame(frame)
            encoded, buffer = cv2.imencode(".jpg", annotated)
            if not encoded:
                raise RuntimeError("Could not encode a webcam frame as JPEG.")
            yield buffer.tobytes(), detector.should_alert(no_mask_seen)
    finally:
        camera.release()
