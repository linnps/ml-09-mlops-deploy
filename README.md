<div align="center">

# MLOps — Deploy a Model and Watch It Drift

**Train a classifier, serve it as a FastAPI service, load-test it, then catch a synthetic data drift with PSI.**

![status](https://img.shields.io/badge/status-complete-3B6EA8?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-3B6EA8?style=flat-square)
![framework](https://img.shields.io/badge/framework-FastAPI-3B6EA8?style=flat-square)
![docker](https://img.shields.io/badge/docker-ready-7A7A7A?style=flat-square)
![data](https://img.shields.io/badge/data-self--generated-7A7A7A?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-7A7A7A?style=flat-square)

</div>

---

## At a glance

> A miniature production-shaped pipeline: train → persist → serve → load-test → monitor for drift. Each stage is small enough to fit in one file, and the entire run is `python run_dashboard.py`.

<table>
<tr>
<td align="center" width="33%">
<sub>Test AUC</sub><br>
<b style="font-size:1.6em; color:#3B6EA8;">0.986</b><br>
<sub>GradientBoosting on synthetic data</sub>
</td>
<td align="center" width="33%">
<sub>p99 latency</sub><br>
<b style="font-size:1.6em; color:#3B6EA8;">1.65 ms</b><br>
<sub>per /predict request, in-process</sub>
</td>
<td align="center" width="33%">
<sub>Drift PSI on x1</sub><br>
<b style="font-size:1.6em; color:#C04040;">2.15</b><br>
<sub>(threshold 0.25 — strong alert)</sub>
</td>
</tr>
</table>

| Stage | What we're measuring | Headline number |
|---|---|---|
| Train | Test accuracy / AUC of the classifier | **acc 0.940, AUC 0.986** |
| Serve | p50 / p95 / p99 latency of `/predict` over 400 requests | **0.98 / 1.25 / 1.65 ms** |
| Monitor (stable stream) | PSI per feature, no drift expected | **all PSI < 0.01** ✅ |
| Monitor (drifted stream) | PSI per feature, drift injected on x1 and x2 | **x1 = 2.15, x2 = 1.19** 🚨 |

<sub>**Headline finding:** the *shape* of production ML — a model artifact that ships with its schema and reference statistics, an HTTP service that validates inputs, and a monitor that compares each live batch's distribution to training — is small. Less than 400 lines of Python total. The discipline is in *enforcing* that shape, not in the algorithms.</sub>

---

## Dashboard

### 1. End-to-end architecture

![architecture](assets/01_architecture.png)

Five components, three flows:

- **Train** produces an artifact bundle: the fitted model + schema (feature names, model type, training metrics) + reference distribution statistics.
- **Serve** loads the artifact at startup and exposes `/health`, `/info`, `/predict`, `/predict/batch`. Inputs are validated against the schema; missing features return HTTP 400, not a silent NaN.
- **Monitor** compares live data batches against the reference statistics using PSI and KS — flagging the operator when the distribution shifts.

### 2. Latency profile

![latency](assets/02_latency.png)

400 requests through the FastAPI handler (in-process via `fastapi.testclient.TestClient` so we don't depend on a port-binding step). The histogram is tight: p50 of 0.98 ms, p99 of 1.65 ms. That's the lower bound — a real server adds network latency, request parsing, and TLS — but it tells us the *model evaluation* is not the bottleneck. For a GradientBoosting model on 6 features, that's expected.

### 3. PSI — drift detection on stable vs drifted streams

![psi](assets/03_psi.png)

Per-feature PSI on two live streams:

- **Stable stream (light blue)**: drawn from the same distribution as training. All PSIs sit well below the warning threshold of 0.10.
- **Drifted stream (red)**: same generator, but with `x1` shifted by +1.6σ and `x2` shifted by −1.1σ. PSI for x1 explodes to **2.15**, and x2 to **1.19** — both far above the alert threshold of 0.25. The unaffected features (x3–x6) stay quiet, so the alert is feature-level actionable, not just "something is wrong."

This is the operational story: a monitor that *only* fires when the data actually shifts, and tells you *which feature* shifted.

### 4. Distribution overlays — what the drift actually looks like

![distributions](assets/04_distributions.png)

Reference (blue) vs drifted (red) histograms, one panel per feature. The mean shifts on x1 and x2 are visually obvious; x3–x6 sit on top of each other. Visualization isn't strictly necessary — PSI does the math — but it's the artifact you want when an alert fires and someone asks "what does that mean?"

---

## What's actually happening

### The artifact, not just the model

```
models/
├── classifier.joblib       # the fitted estimator
├── schema.json             # feature names, training metrics
└── reference_X.npy         # training-set features for drift comparison
```

A *model* is a file. A *production-ready model* is a bundle: the file plus everything you need to call it correctly, plus everything you need to detect when calling it has stopped being a good idea. Without the schema, the service has no idea what fields to expect. Without the reference stats, the monitor has nothing to compare against.

### The service contract

```python
POST /predict
{
  "features": {
    "x1": 0.42, "x2": -1.10, "x3": 0.07,
    "x4": 1.30, "x5": -0.20, "x6": 0.66
  }
}
```

Pydantic validates the request shape. The handler then ensures every required feature from the schema is present (HTTP 400 if not), assembles them in the schema's declared order, and runs `predict_proba`. The full classifier path is twelve lines of code in `service.py`.

### What PSI actually measures

```
PSI = sum_i (p_i_current − p_i_ref) · log(p_i_current / p_i_ref)
```

Bin the reference distribution into deciles. For each bin, compare the fraction of *current* observations falling in that bin against the *reference* fraction. Penalize the difference logarithmically. The result is symmetric in spirit (large in either direction) and bounded only by your numerical floor.

Industry rule of thumb:

| PSI | Meaning |
|---|---|
| < 0.10 | No significant change — keep serving. |
| 0.10 – 0.25 | Moderate shift — investigate, don't necessarily retrain. |
| > 0.25 | Significant shift — alert, consider retraining. |

We pair it with the Kolmogorov–Smirnov two-sample test (`scipy.stats.ks_2samp`) for a complementary p-value-flavored perspective.

### What's deliberately not in scope

- Real Kubernetes / cloud deploy. Would distract from the operational shape.
- ML feature flags, A/B between models. One model, one path.
- Online retraining. Drift is detected, not auto-corrected.
- A model registry beyond joblib + schema.json. Plenty for one model; you'd reach for MLflow at 10+.

The goal of this project is "I understand the shape of production ML and can implement it from scratch." Not "I have rebuilt SageMaker."

---

## Reproduce

### Quick (no Docker, all in-process)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_dashboard.py    # train + serve in-process + load-test + drift report + figs
```

Wall-time ~5 seconds. All four dashboard PNGs land in `assets/` and the metrics summary in `results/metrics.json`.

### Real service

```bash
python train_model.py
uvicorn service:app --port 8000 --reload
# then:
curl http://localhost:8000/health
curl http://localhost:8000/info
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features":{"x1":0.4,"x2":-1.1,"x3":0.07,"x4":1.3,"x5":-0.2,"x6":0.66}}'
```

### Containerized

```bash
docker build -t ml-09-classifier .
docker run -p 8000:8000 ml-09-classifier
```

The Dockerfile is intentionally simple: install requirements, train so the artifact is baked into the image, then start uvicorn. A real one would split build/run, use a non-root user, and pull the model from a registry rather than training during build.

---

## Project layout

```
09-mlops-deploy/
├── README.md              ← this dashboard
├── requirements.txt
├── Dockerfile             ← single-stage build that bakes the model
├── generate_data.py       ← stable + drifted synthetic streams
├── train_model.py         ← persist model + schema + reference stats
├── service.py             ← FastAPI service (predict / health / info)
├── monitor.py             ← PSI + KS drift metrics
├── run_dashboard.py       ← orchestrates train + serve + probe + monitor + figs
├── models/                ← (regenerated) classifier.joblib + schema.json
├── assets/                ← 4 dashboard PNGs
└── results/metrics.json
```

---

## What I learned

- **The "model" is the smallest artifact in the system.** The schema, the reference statistics, the service contract, the monitoring window definitions, the alert thresholds — those are the parts that determine whether the model behaves correctly six months from now. Allocating effort to *those* rather than to the model itself is what separates research code from production code.
- **PSI is much sharper than its formula suggests.** A mean shift of 1.6σ on a single feature pushes PSI from 0.01 to 2.15 — a 200×+ change on a metric that nominally only requires a 25× change to alert. That sensitivity is exactly what you want from a drift signal.
- **Latency budgets are a forcing function for architecture.** When p99 of `predict_proba` is 1.6 ms but real production needs 50 ms p99 end-to-end, every other component (request validation, feature lookup, network) has 48 ms to do its work. That's a useful framing for what "deploying ML" actually requires.
- **Building this without Docker first was a feature, not a bug.** `TestClient` lets the same FastAPI handler that production uses run inside the test process, which means the dashboard run is reproducible without spinning up containers. Docker is then layered on top once the contract is right.

---

<div align="center">
<sub>Part of a hands-on machine-learning portfolio. Data is fully synthetic and self-generated.</sub>
</div>
