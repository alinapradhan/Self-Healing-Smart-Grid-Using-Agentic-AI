"""Train and evaluate a dependency-free smart-grid fault prediction model."""

from __future__ import annotations

from pathlib import Path

from src.fault_prediction_model import (
    FEATURE_NAMES,
    LogisticRegressionSGD,
    generate_smart_grid_dataset,
    train_test_split,
)


def main() -> None:
    dataset = generate_smart_grid_dataset(samples=5000, seed=11)
    train_set, test_set = train_test_split(dataset, test_ratio=0.2, seed=11)

    model = LogisticRegressionSGD(n_features=len(FEATURE_NAMES), learning_rate=0.12, l2=8e-4)
    model.fit(train_set.features, train_set.labels, epochs=80)

    metrics = model.evaluate(test_set.features, test_set.labels)

    model_dir = Path("artifacts")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fault_predictor_model.json"
    model.save(model_path)

    print("=== Smart-Grid Fault Prediction Model ===")
    print(f"Train samples: {len(train_set.labels)} | Test samples: {len(test_set.labels)}")
    print(f"Accuracy : {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall   : {metrics['recall']:.3f}")
    print(f"F1 score : {metrics['f1']:.3f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Saved model artifact: {model_path}")


if __name__ == "__main__":
    main()
