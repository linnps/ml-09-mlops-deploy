"""
Synthetic tabular classification data with a known generative process,
for the MLOps demo. We also generate two variants of the *live* data:
- a "stable" stream that looks like the training distribution
- a "drifted" stream where two features have shifted means

The drift variant lets the monitoring code (`monitor.py`) demonstrate
that PSI / KS-test pick up the shift.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    n_samples: int = 4000
    n_features: int = 6
    drift_shift_size: float = 1.6     # std-deviations of mean shift on drifted stream
    seed: int = 42


def _make_xy(rng: np.random.Generator, n: int, n_features: int,
             drift: bool = False, drift_shift_size: float = 1.6) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(0, 1, size=(n, n_features))
    if drift:
        X[:, 0] += drift_shift_size
        X[:, 1] -= drift_shift_size * 0.7
    # Decision rule: y = 1 iff x0 + x1*x2 > 0  (non-linear interaction)
    score = X[:, 0] + X[:, 1] * X[:, 2]
    y = (score > 0).astype(int)
    return X, y


def generate(cfg: DataConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    Xtr, ytr = _make_xy(rng, cfg.n_samples, cfg.n_features)
    Xte, yte = _make_xy(rng, cfg.n_samples // 4, cfg.n_features)
    Xstable, _ = _make_xy(rng, cfg.n_samples // 2, cfg.n_features)
    Xdrift,  _ = _make_xy(rng, cfg.n_samples // 2, cfg.n_features,
                          drift=True, drift_shift_size=cfg.drift_shift_size)
    return {"train": (Xtr, ytr), "test": (Xte, yte),
            "live_stable": (Xstable, None), "live_drifted": (Xdrift, None)}


def save(out_dir: Path, splits: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [f"x{i+1}" for i in range(splits["train"][0].shape[1])]
    for k, (X, y) in splits.items():
        df = pd.DataFrame(X, columns=cols)
        if y is not None:
            df["y"] = y
        df.to_csv(out_dir / f"{k}.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()
    cfg = DataConfig()
    splits = generate(cfg)
    save(args.out_dir, splits)
    for k, (X, _) in splits.items():
        print(f"  {k}: {X.shape}")
    print(f"Saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
