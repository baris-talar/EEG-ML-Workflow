"""Feature extraction and label encoding helpers."""

from __future__ import annotations

import numpy as np


def extract_mean_variance_features(windows) -> np.ndarray:
    """Extract per-channel mean and variance from each window."""
    feature_vectors = []
    num_channels = None

    for window in windows:
        if num_channels is None:
            num_channels = window.shape[0]
        means = np.mean(window, axis=1)
        variances = np.var(window, axis=1)
        feature_vectors.append(np.concatenate([means, variances]))

    if not feature_vectors:
        if num_channels is None:
            return np.empty((0, 0), dtype=float)
        return np.empty((0, num_channels * 2), dtype=float)
    return np.array(feature_vectors, dtype=float)


def encode_labels(labels) -> tuple[np.ndarray, np.ndarray]:
    """Encode string labels into sorted label names and integer indices."""
    y = np.array(labels, dtype=str)
    label_names = np.array(sorted(set(y.tolist())), dtype=str)
    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    y_idx = np.array([label_to_index[label] for label in y], dtype=int)
    return label_names, y_idx

