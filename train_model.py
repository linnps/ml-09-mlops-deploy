"""
Train the model that the FastAPI service will serve, persist it to disk
with feature schema and reference statistics for drift monitoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from generate_data import DataConfig, generate


def main() -> None:
    cfg = DataConfig()
    splits = generate(cfg)
    Xtr, ytr = splits["train"]
    Xte, yte = splits["test"]

    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    model.fit(Xtr, ytr)

    train_acc = accuracy_score(ytr, model.predict(Xtr))
    test_acc = accuracy_score(yte, model.predict(Xte))
    test_auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])

    artifact_dir = Path("models")
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump(model, artifact_dir / "classifier.joblib")
    feature_names = [f"x{i+1}" for i in range(Xtr.shape[1])]
    schema = {
        "feature_names": feature_names,
        "model_type": "GradientBoostingClassifier",
        "metrics": {"train_acc": float(train_acc), "test_acc": float(test_acc),
                    "test_auc": float(test_auc)},
        "reference_stats": {
            f: {"mean": float(Xtr[:, i].mean()), "std": float(Xtr[:, i].std())}
            for i, f in enumerate(feature_names)
        },
    }
    with open(artifact_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    np.save(artifact_dir / "reference_X.npy", Xtr)

    print(f"Trained model — train acc {train_acc:.3f}, test acc {test_acc:.3f}, "
          f"test AUC {test_auc:.3f}")
    print(f"Saved to: {artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
