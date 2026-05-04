"""
Drift monitoring for the deployed classifier.

Compute Population Stability Index (PSI) and Kolmogorov–Smirnov
statistic for each feature, comparing a *reference* sample (training
distribution) against a *current* sample (live data window).

PSI reading guide (industry rule of thumb):
    < 0.10   no significant change
    0.10–0.25 moderate shift, investigate
    > 0.25   significant shift, alert
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI = sum_i (current_i - reference_i) * log(current_i / reference_i)

    Bin edges are quantiles of the reference distribution.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_p = ref_hist / max(len(reference), 1)
    cur_p = cur_hist / max(len(current), 1)

    eps = 1e-6
    ref_p = np.where(ref_p == 0, eps, ref_p)
    cur_p = np.where(cur_p == 0, eps, cur_p)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def feature_report(reference: np.ndarray, current: np.ndarray,
                   feature_names: list[str]) -> list[dict]:
    out = []
    for i, name in enumerate(feature_names):
        ks_stat, ks_p = ks_2samp(reference[:, i], current[:, i])
        out.append({
            "feature": name,
            "psi": psi(reference[:, i], current[:, i]),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
        })
    return out
