"""Training pipeline for personalized EEG state classification."""

from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import train_test_split

from eeg_toolkit.artifacts import load_calibration_dataset, save_model_artifact
from eeg_toolkit.modeling import (
    build_model_artifact,
    evaluate_accuracy,
    train_logreg,
)


DEFAULT_DATASET_PATH = "python_package/examples/tests/artifacts/eeg_calibration_dataset.npz"
DEFAULT_MODEL_PATH = "python_package/examples/tests/artifacts/eeg_logreg_model.joblib"


def _input_with_default(prompt_text, default_value):
    try:
        value = input(prompt_text).strip()
    except EOFError:
        value = ""
    return value if value else str(default_value)


def _validate_dataset(X: np.ndarray, y_idx: np.ndarray, label_names: np.ndarray) -> None:
    if X.ndim != 2 or X.shape[0] == 0:
        raise RuntimeError("Dataset has no usable feature rows.")
    if y_idx.shape[0] != X.shape[0]:
        raise RuntimeError("Mismatch between number of samples in X and labels.")
    if len(label_names) < 2:
        raise RuntimeError("Need at least two label classes for supervised training.")


def run():
    print("Training workflow for personalized EEG state classification.")

    dataset_path = _input_with_default(
        f"Calibration dataset path [{DEFAULT_DATASET_PATH}]: ",
        DEFAULT_DATASET_PATH,
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = load_calibration_dataset(dataset_path)
    X = data["X"]
    y_idx = data["y_idx"]
    y = data["y"]
    label_names = data["label_names"]

    _validate_dataset(X, y_idx, label_names)

    print()
    print("--- Dataset summary ---")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Label names:", label_names.tolist())
    label_counts = {}
    min_class_count = X.shape[0]
    for label_name in label_names:
        class_count = int(np.sum(y == label_name))
        label_counts[label_name] = class_count
        min_class_count = min(min_class_count, class_count)
    print("Label counts:", label_counts)

    stratify = y_idx if min_class_count >= 2 else None
    if stratify is None:
        print("Warning: at least one class has fewer than 2 samples; split is not stratified.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_idx,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    model = train_logreg(X_train, y_train, random_state=42)
    accuracy = evaluate_accuracy(model, X_test, y_test)

    print()
    print("--- Evaluation ---")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print(f"Accuracy: {accuracy:.4f}")
    print(
        "Interpretation note: performance depends on the calibration protocol, "
        "label quality, and synthetic/state transform design."
    )

    metadata = {
        "sampling_rate": int(data["sampling_rate"]),
        "window_size_samples": int(data["window_size_samples"]),
        "window_duration_sec": int(data["window_duration_sec"]),
        "apply_preprocessing": bool(int(data["apply_preprocessing"])),
        "preprocessing_bandpass_low_hz": float(data["preprocessing_bandpass_low_hz"]),
        "preprocessing_bandpass_high_hz": float(data["preprocessing_bandpass_high_hz"]),
        "preprocessing_filter_order": int(data["preprocessing_filter_order"]),
        "feature_description": str(data["feature_description"]),
        "eeg_channel_names": data["eeg_channel_names"].tolist(),
    }
    model_artifact = build_model_artifact(model, label_names, metadata)

    model_path = _input_with_default(
        f"Model output path [{DEFAULT_MODEL_PATH}]: ",
        DEFAULT_MODEL_PATH,
    )
    save_model_artifact(model_path, model_artifact)

    print()
    print("--- Model saved ---")
    print("Model path:", model_path)
    print("Stored label names:", model_artifact["label_names"])


def main():
    run()


if __name__ == "__main__":
    main()

