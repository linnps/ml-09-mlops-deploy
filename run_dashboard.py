"""
Run the full demo: train model, start the FastAPI service in-process,
hit it with a load of requests to measure latency, run drift monitoring
on a stable + drifted live-data stream, and render dashboard figures.

Why in-process? It removes the Docker / port-binding step from the
reproduce path while still exercising the same FastAPI handler that
production would call. The architecture diagram in the README explains
how this scales out.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from fastapi.testclient import TestClient

import service
import train_model
from generate_data import DataConfig, generate
from monitor import feature_report, psi

# ---------------------------------------------------------------- style ----
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E5E5E5"
COLOR_TEXT = "#333333"
COLOR_BLUE = "#3B6EA8"
COLOR_RED = "#C04040"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#CCCCCC"
COLOR_LIGHT_BLUE = "#9EB7D6"

mpl.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_LIGHT_GRAY,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.size": 11,
})


# --------------------------------------------------------- latency probe --
def latency_probe(client: TestClient, X: np.ndarray, feature_names: list[str],
                  n_requests: int = 400) -> tuple[np.ndarray, dict]:
    """Hit the /predict endpoint n times and record per-request latency (ms)."""
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(X), size=n_requests)
    latencies = []
    correct_count = 0
    for idx in pick:
        row = {f: float(X[idx, i]) for i, f in enumerate(feature_names)}
        start = time.perf_counter()
        r = client.post("/predict", json={"features": row})
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        assert r.status_code == 200
        if r.json()["prediction"] in (0, 1):
            correct_count += 1
    arr = np.array(latencies)
    summary = {
        "n_requests": int(n_requests),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
    }
    return arr, summary


# ---------------------------------------------------------------- figures --
def fig_latency(latencies: np.ndarray, summary: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    ax.hist(latencies, bins=30, color=COLOR_BLUE, edgecolor="white", linewidth=0.6, alpha=0.85)
    for label, c in [("p50", COLOR_GRAY), ("p95", COLOR_RED), ("p99", COLOR_RED)]:
        v = summary[f"{label}_ms"]
        ax.axvline(v, color=c, linewidth=1.4, linestyle="--",
                   label=f"{label.upper()} = {v:.1f} ms")
    ax.set_xlabel("Per-request latency (ms)")
    ax.set_ylabel("Request count")
    ax.set_title(f"FastAPI /predict latency over {summary['n_requests']} requests")
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_drift(report_stable: list[dict], report_drift: list[dict], out_path: Path) -> None:
    feats = [r["feature"] for r in report_stable]
    psi_stable = [r["psi"] for r in report_stable]
    psi_drift = [r["psi"] for r in report_drift]

    x = np.arange(len(feats))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(x - width / 2, psi_stable, width, color=COLOR_LIGHT_BLUE,
           edgecolor=COLOR_LIGHT_GRAY, linewidth=0.6, label="Stable stream")
    ax.bar(x + width / 2, psi_drift, width, color=COLOR_RED,
           edgecolor=COLOR_LIGHT_GRAY, linewidth=0.6, label="Drifted stream")

    ax.axhline(0.10, color=COLOR_GRAY, linewidth=0.9, linestyle=":",
               label="warning (PSI = 0.10)")
    ax.axhline(0.25, color=COLOR_RED, linewidth=0.9, linestyle=":",
               label="alert (PSI = 0.25)")

    ax.set_xticks(x); ax.set_xticklabels(feats)
    ax.set_ylabel("PSI"); ax.set_title("Per-feature PSI — stable vs drifted live stream")
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_distribution_overlay(reference: np.ndarray, current: np.ndarray,
                             feature_names: list[str], out_path: Path) -> None:
    n = len(feature_names)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows), constrained_layout=True)
    axes = axes.ravel()
    for i, name in enumerate(feature_names):
        ax = axes[i]
        bins = np.linspace(min(reference[:, i].min(), current[:, i].min()),
                           max(reference[:, i].max(), current[:, i].max()), 26)
        ax.hist(reference[:, i], bins=bins, color=COLOR_BLUE, alpha=0.65,
                edgecolor="white", linewidth=0.4, label="reference")
        ax.hist(current[:, i], bins=bins, color=COLOR_RED, alpha=0.65,
                edgecolor="white", linewidth=0.4, label="drifted")
        ax.set_title(name)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    axes[0].legend(loc="upper right")
    fig.suptitle("Reference (training) vs drifted live stream — feature histograms",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_architecture(out_path: Path) -> None:
    """A static schematic of the deploy/ monitor architecture."""
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.axis("off")

    boxes = [
        (0.04, 0.55, 0.18, 0.30, "Train script\n(train_model.py)", COLOR_LIGHT_BLUE),
        (0.30, 0.55, 0.18, 0.30, "Model artifact\n+ schema + ref stats", COLOR_LIGHT_GRAY),
        (0.56, 0.55, 0.18, 0.30, "FastAPI service\n(/predict, /info, /health)", COLOR_BLUE),
        (0.82, 0.55, 0.14, 0.30, "Client", COLOR_GRAY),
        (0.30, 0.10, 0.18, 0.30, "Live data\n(stable / drifted)", COLOR_LIGHT_GRAY),
        (0.56, 0.10, 0.18, 0.30, "Drift monitor\n(PSI / KS)", COLOR_RED),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                    edgecolor=COLOR_LIGHT_GRAY, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10, color="white" if color in (COLOR_BLUE, COLOR_RED) else COLOR_TEXT,
                weight="bold")

    arrows = [
        (0.22, 0.70, 0.30, 0.70),  # train → artifact
        (0.48, 0.70, 0.56, 0.70),  # artifact → service
        (0.74, 0.70, 0.82, 0.70),  # service → client
        (0.48, 0.25, 0.56, 0.25),  # live data → monitor
        (0.65, 0.40, 0.65, 0.55),  # service → live data (predictions feed back? no — keep simple)
        (0.48, 0.65, 0.39, 0.40),  # artifact → ref stats for monitor
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=COLOR_GRAY, lw=1.2))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("End-to-end architecture",
                 fontsize=14, fontweight="bold", pad=10)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main() -> None:
    print("=== train ===")
    train_model.main()

    # Reload service-state from disk.
    service.load_artifacts()
    client = TestClient(service.app)

    feature_names = service._state.schema["feature_names"]

    print("\n=== latency probe ===")
    cfg = DataConfig()
    splits = generate(cfg)
    Xtr, _ = splits["train"]
    latencies, lat_summary = latency_probe(client, Xtr, feature_names, n_requests=400)
    print(f"  n={lat_summary['n_requests']}  p50={lat_summary['p50_ms']:.2f}ms  "
          f"p95={lat_summary['p95_ms']:.2f}ms  p99={lat_summary['p99_ms']:.2f}ms")

    print("\n=== drift monitoring ===")
    Xstable = splits["live_stable"][0]
    Xdrift = splits["live_drifted"][0]
    report_stable = feature_report(Xtr, Xstable, feature_names)
    report_drift = feature_report(Xtr, Xdrift, feature_names)
    print("  stable stream PSI:")
    for r in report_stable:
        print(f"    {r['feature']}: PSI={r['psi']:.3f}, KS p={r['ks_p']:.3f}")
    print("  drifted stream PSI:")
    for r in report_drift:
        print(f"    {r['feature']}: PSI={r['psi']:.3f}, KS p={r['ks_p']:.3f}")

    Path("results").mkdir(exist_ok=True)
    summary = {
        "latency": lat_summary,
        "drift_stable": report_stable,
        "drift_drifted": report_drift,
        "service_endpoints": ["/health", "/info", "/predict", "/predict/batch"],
    }
    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    assets = Path("assets"); assets.mkdir(exist_ok=True)
    fig_architecture(assets / "01_architecture.png")
    fig_latency(latencies, lat_summary, assets / "02_latency.png")
    fig_drift(report_stable, report_drift, assets / "03_psi.png")
    fig_distribution_overlay(Xtr, Xdrift, feature_names, assets / "04_distributions.png")

    print(f"\nFigures saved to: {assets.resolve()}")


if __name__ == "__main__":
    main()
