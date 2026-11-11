"""Flask entry point for the live face-mask alert stream."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, Response, jsonify, render_template

app = Flask(__name__)
MODEL_PATH = Path(os.environ.get("MASK_MODEL", "models/mask_detector.keras"))
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
detector = None
webcam_frames = None
startup_error: str | None = None

try:
    from mask_detector import MaskDetector, webcam_frames

    detector = MaskDetector(MODEL_PATH)
except (FileNotFoundError, ImportError, RuntimeError, OSError, ValueError) as error:
    startup_error = str(error)


@app.get("/")
def index() -> str:
    return render_template("index.html", startup_error=startup_error)


@app.get("/status")
def status() -> tuple[Response, int]:
    if startup_error:
        return jsonify({"ready": False, "error": startup_error}), 503
    return jsonify({"ready": True}), 200


@app.get("/video_feed")
def video_feed() -> Response:
    if detector is None:
        return Response(startup_error or "Detector is unavailable.", status=503)

    def stream():
        for frame, alert in webcam_frames(detector, CAMERA_INDEX):
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                b"X-Mask-Alert: " + (b"1" if alert else b"0") + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )

    return Response(
        stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
