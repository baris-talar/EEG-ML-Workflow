"""Model training, evaluation, and prediction helpers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_logreg(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """Train logistic regression with the project default configuration."""
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_accuracy(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate a classifier using accuracy."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def predict_labels(model, X: np.ndarray, label_names: np.ndarray) -> list[str]:
    """Predict class indices and map them to string labels."""
    predicted_idx = model.predict(X)
    return [str(label_names[idx]) for idx in predicted_idx]


def build_model_artifact(model, label_names: np.ndarray, metadata: dict) -> dict:
    """Build a persisted model artifact payload."""
    artifact = {
        "model": model,
        "label_names": label_names.tolist(),
    }
    artifact.update(metadata)
    return artifact

