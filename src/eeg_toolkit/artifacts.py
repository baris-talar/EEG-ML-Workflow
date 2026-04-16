"""Persistence helpers for dataset and model artifacts."""

from __future__ import annotations

import os

import numpy as np
from joblib import dump, load


def save_calibration_dataset(path: str, dataset: dict) -> None:
    """Save a calibration dataset as compressed NPZ."""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(path, **dataset)


def load_calibration_dataset(path: str):
    """Load a calibration dataset from NPZ."""
    return np.load(path)


def save_model_artifact(path: str, artifact: dict) -> None:
    """Save model artifact to joblib format."""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    dump(artifact, path)


def load_model_artifact(path: str):
    """Load model artifact from joblib format."""
    return load(path)

