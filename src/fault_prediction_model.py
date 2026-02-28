"""Simple dependency-free ML pipeline for smart-grid fault prediction.

This module implements a lightweight binary classifier (logistic regression with SGD)
and synthetic data generation aligned with the README framework.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


FEATURE_NAMES = [
    "harmonics",
    "phase_angle_drift",
    "voltage_sag_index",
    "weather_severity",
    "lightning_density",
    "asset_health_risk",
    "breaker_operation_stress",
    "line_centrality",
    "redundancy_score",
    "load_transfer_flexibility",
]


@dataclass
class Dataset:
    features: List[List[float]]
    labels: List[int]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def generate_smart_grid_dataset(samples: int = 4000, seed: int = 42) -> Dataset:
    """Create synthetic grid telemetry samples and fault labels.

    The feature space mirrors README sections: electrical, contextual,
    asset-health, and topological indicators.
    """

    rng = random.Random(seed)
    x_data: List[List[float]] = []
    y_data: List[int] = []

    for _ in range(samples):
        # Electrical signals
        harmonics = _clamp(rng.gauss(0.45, 0.18))
        phase_angle_drift = _clamp(rng.gauss(0.40, 0.20))
        voltage_sag_index = _clamp(rng.gauss(0.35, 0.22))

        # Environmental/contextual
        weather_severity = _clamp(rng.betavariate(2.0, 5.0) + rng.random() * 0.2)
        lightning_density = _clamp(rng.betavariate(2.5, 4.0))

        # Asset health
        asset_health_risk = _clamp(rng.gauss(0.42, 0.21))
        breaker_operation_stress = _clamp(rng.gauss(0.38, 0.19))

        # Topology
        line_centrality = _clamp(rng.gauss(0.55, 0.20))
        redundancy_score = _clamp(rng.gauss(0.48, 0.22))
        load_transfer_flexibility = _clamp(rng.gauss(0.50, 0.20))

        features = [
            harmonics,
            phase_angle_drift,
            voltage_sag_index,
            weather_severity,
            lightning_density,
            asset_health_risk,
            breaker_operation_stress,
            line_centrality,
            redundancy_score,
            load_transfer_flexibility,
        ]

        # Hidden risk model used to construct labels.
        risk_score = (
            1.20 * harmonics
            + 1.10 * phase_angle_drift
            + 1.35 * voltage_sag_index
            + 1.15 * weather_severity
            + 0.90 * lightning_density
            + 1.05 * asset_health_risk
            + 0.80 * breaker_operation_stress
            + 0.45 * line_centrality
            - 0.95 * redundancy_score
            - 0.85 * load_transfer_flexibility
            - 2.20
        )

        # Convert risk to probability with noise.
        noise = rng.gauss(0.0, 0.20)
        prob_fault = 1.0 / (1.0 + math.exp(-(risk_score + noise)))
        label = 1 if rng.random() < prob_fault else 0

        x_data.append(features)
        y_data.append(label)

    return Dataset(features=x_data, labels=y_data)


class LogisticRegressionSGD:
    """Minimal binary logistic regression using SGD and L2 regularization."""

    def __init__(self, n_features: int, learning_rate: float = 0.1, l2: float = 1e-3):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def predict_proba_one(self, x: Sequence[float]) -> float:
        z = self.bias
        for w, xi in zip(self.weights, x):
            z += w * xi
        return self._sigmoid(z)

    def predict_one(self, x: Sequence[float], threshold: float = 0.5) -> int:
        return 1 if self.predict_proba_one(x) >= threshold else 0

    def fit(self, x_train: Sequence[Sequence[float]], y_train: Sequence[int], epochs: int = 60) -> None:
        if not x_train:
            raise ValueError("x_train is empty")

        rng = random.Random(7)
        indices = list(range(len(x_train)))

        for _ in range(epochs):
            rng.shuffle(indices)
            for idx in indices:
                x = x_train[idx]
                y = y_train[idx]
                pred = self.predict_proba_one(x)
                error = pred - y

                for j in range(self.n_features):
                    grad = error * x[j] + self.l2 * self.weights[j]
                    self.weights[j] -= self.learning_rate * grad

                self.bias -= self.learning_rate * error

    def evaluate(self, x_test: Sequence[Sequence[float]], y_test: Sequence[int]) -> dict:
        tp = tn = fp = fn = 0

        for x, y in zip(x_test, y_test):
            pred = self.predict_one(x)
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 0 and y == 0:
                tn += 1
            elif pred == 1 and y == 0:
                fp += 1
            else:
                fn += 1

        total = max(1, len(y_test))
        accuracy = (tp + tn) / total
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "model": "LogisticRegressionSGD",
            "feature_names": FEATURE_NAMES,
            "weights": self.weights,
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_test_split(dataset: Dataset, test_ratio: float = 0.2, seed: int = 42) -> Tuple[Dataset, Dataset]:
    indices = list(range(len(dataset.labels)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split_idx = int(len(indices) * (1 - test_ratio))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    def collect(idx_list: Sequence[int]) -> Dataset:
        return Dataset(
            features=[dataset.features[i] for i in idx_list],
            labels=[dataset.labels[i] for i in idx_list],
        )

    return collect(train_indices), collect(test_indices)
