"""Train a grayscale CNN from data/mask and data/no_mask directories."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_model(input_shape: tuple[int, int, int] = (100, 100, 1)):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        MaxPooling2D,
        Rescaling,
    )

    model = Sequential(
        [
            Rescaling(1.0 / 255, input_shape=input_shape),
            Conv2D(32, 3, activation="relu"),
            MaxPooling2D(),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(),
            Conv2D(128, 3, activation="relu"),
            MaxPooling2D(),
            Flatten(),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def train(data_dir: Path, output: Path, epochs: int, batch_size: int) -> None:
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.utils import image_dataset_from_directory
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow could not load its native Windows DLLs. "
            "This is commonly caused by an Application Control policy "
            "or an unsupported Python/TensorFlow combination. "
            "Use a supported Python version in a fresh virtual environment "
            "or ask your administrator to allow the TensorFlow DLLs."
        ) from error

    if not (data_dir / "mask").is_dir() or not (data_dir / "no_mask").is_dir():
        raise FileNotFoundError(
            f"Expected '{data_dir / 'mask'}' and '{data_dir / 'no_mask'}' directories."
        )

    train_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=(100, 100),
        batch_size=batch_size,
        class_names=["no_mask", "mask"],
        validation_split=0.2,
        subset="training",
        seed=42,
    )
    validation_ds = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=(100, 100),
        batch_size=batch_size,
        class_names=["no_mask", "mask"],
        validation_split=0.2,
        subset="validation",
        seed=42,
    )
    model = build_model()
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    )
    validation_accuracy = history.history["val_accuracy"][-1]
    print(f"Final validation accuracy: {validation_accuracy:.1%}")
    if validation_accuracy < 0.70:
        raise RuntimeError(
            "The model did not reach 70% validation accuracy. "
            "Check that data/mask contains masked faces and data/no_mask "
            "contains unmasked faces, then retrain."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/mask_detector.keras"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.data_dir, args.output, args.epochs, args.batch_size)
