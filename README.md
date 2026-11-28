# AgriShield

> Food security forecasting and early-warning system for climate-aware price and shortage-risk analysis.

## Overview

AgriShield is a machine-learning project focused on forecasting food-price pressure and shortage risk using weather, macroeconomic, and regional indicators. It was developed as a research-driven system for early warning in food-security planning, with emphasis on applied forecasting and operational risk detection.

## Problem statement

Food systems are sensitive to climate variability, inflation pressure, transport cost changes, and broader regional risk factors. AgriShield explores whether these signals can be modeled to anticipate food-price stress and shortage risk early enough to support intervention or monitoring workflows.

## System architecture

```mermaid
flowchart LR
    A[Weather data\nOpen-Meteo] --> B[Data ingestion & cleaning]
    C[CPI / inflation / macro indicators] --> B
    D[Regional feature engineering] --> B
    B --> E[Forecasting pipeline]
    E --> F[Random Forest baseline]
    E --> G[PyTorch / neural model]
    F --> H[Price & risk forecasts]
    G --> H
    H --> I[Threshold alerting]
    I --> J[Dashboard / reporting]
```

## Data pipeline

The notebooks build `food_security_data.csv` (180 monthly rows: US, Ukraine, India, 2021 to 2025). **Most of it is synthetic, including both prediction targets.**

| Columns | Origin in the committed CSV |
| --- | --- |
| `food_price_index`, `shortage_risk_score` (targets) | **Synthetic.** Hand-written formulas over the other columns, plus random noise. They are not observed ground truth. |
| `crop_yield`, `fuel_price`, `transport_cost`, `geopolitical_risk`, `news_sentiment` | **Synthetic.** Region and year profiles, simulated shocks and random noise. |
| `temperature`, `precipitation` | **US: synthetic** seasonal profile (it matches the notebook's fallback formula). Ukraine and India do not match the fallback formula, consistent with the notebook's Open-Meteo fetch. |
| `inflation_rate` | Annual values from the notebook's World Bank CPI fetch, held constant within each year. US 2025 equals the notebook's 2.5 default. |

The committed CSV carries no per-value provenance, so the origin above is inferred from the notebook code and the values. The models therefore mostly learn to reproduce a simulation. Do not read the scores below as real-world forecasting skill.

## Machine learning approach

The repository evaluates multiple modeling strategies for two forecasting objectives:

1. future food-price index prediction
2. future shortage-risk score prediction

The project includes a Random Forest baseline and a neural-network approach, with the objective of comparing standard and advanced forecasting methods under the same feature pipeline.

## Model results

Values from `dashboard_v2/train_results.json`: unweighted means over four grouped walk-forward folds, on the largely synthetic data described above.

| Model | Target | RMSE | R² |
| --- | --- | ---: | ---: |
| Random Forest | Price forecast | 25.20 | 0.586 |
| PyTorch / MLP | Price forecast | 25.65 | 0.576 |
| Random Forest | Shortage risk | 7.99 | 0.535 |
| PyTorch / MLP | Shortage risk | 6.97 | 0.571 |

The neural network is slightly worse than the Random Forest on price and better on shortage risk. The committed run used the CPU (`device: cpu`). The dashboard reads these values from `/v1/metrics`.

## Tech stack

- Python
- Jupyter notebooks
- pandas / NumPy
- scikit-learn
- PyTorch
- React, TypeScript, Vite, Recharts
- matplotlib
- Open-Meteo and World Bank data ingestion

## Project status

This project is best understood as a research and prototype system focused on applied ML for food-security analytics and decision support.

## Running locally

1. Open the notebooks in the repository root to inspect the data pipeline and modeling workflow.
2. Install the required Python packages for the notebook and training environment.
3. Run the forecasting pipeline or dashboard notebook to reproduce the analysis.

### Dashboard frontend

The React app lives in `dashboard_v2/frontend` and needs Node.js 22.12+. Install the Python dependencies first (`pip install -r dashboard_v2/requirements.txt`), then:

```powershell
cd dashboard_v2/frontend
npm ci
npm run dev        # starts the API on :8000 and Vite on :5173
npm run build      # writes dashboard_v2/dist, which the API server serves
npm run typecheck
npm run lint
npm test           # unit tests (Vitest)
```

For the browser check, install Playwright (`pip install playwright && playwright install chromium`), build the frontend, then run `python dashboard_v2/tests/e2e_dashboard.py` from the repository root. It is not part of CI.

The scenario simulator starts from the selected region's latest three months. Temperature is a shift in °C and precipitation a percentage change, both applied to the latest reading; the other sliders start at the latest values. Unchanged sliders therefore send real history to the model. Because that history ends one month later than the inputs behind the March 2026 outlook, the simulator forecasts April 2026.

### Dashboard ML package

The shared Python implementation lives in `dashboard_v2/agri_shield_ml`. From `dashboard_v2`, run
`python train_deep_learning.py` to preserve the dashboard's `train_results.json`
output and write immutable, version-named weight/manifest pairs
(`pytorch_food_security_model.<version>.pth` and
`pytorch_food_security_model.<version>.manifest.json`). The manifest identifies
the artifact schema/version, ordered feature names, fitted feature/target
scalers, evaluation metrics, feature/output population baselines, calibrated
prediction interval widths, and SHA-256 digest of the weights. Use
`python smoke_test_ml_package.py` for a small feature/artifact round-trip check.
Training reports four grouped walk-forward folds (six validation months per
fold, with a three-month forecast-horizon gap) in the JSON `validation`
section. Run `python train_deep_learning.py --validation-only --epochs 1` for
a short validation-only evaluation without production training or artifact
writes.

### Versioned dashboard API

The dashboard can run through the Python standard-library HTTP server (no web
framework dependency). From `dashboard_v2`, start it with:

```powershell
python api_server.py
```

Open `http://127.0.0.1:8000/`. The versioned API exposes `GET /v1/health`,
`GET /v1/metrics`, `POST /v1/predict`, `GET /v1/models`, `GET /v1/audit`, and `GET /v1/monitoring`.
The dashboard uses
the API for startup forecasts, custom-region recalculation, and simulator
predictions. If prediction requests fail, it labels itself "Static fallback"
and uses simple in-browser estimates. The built app must be served over HTTP;
opening the HTML file directly is not supported.

`model_registry.json` persistently selects the active model version and binds
its artifact manifest by SHA-256. With no `AGRISHIELD_ARTIFACT_KEY`, this
manifest digest detects accidental changes but is not an authenticity
signature. For authenticated verification, set a private key and explicitly
sign the current manifest before starting the server:

```powershell
$env:AGRISHIELD_ARTIFACT_KEY = "use-a-secret-from-your-secret-manager"
python api_server.py --sign-registry
python api_server.py
```

When the key is set, the registry must contain a matching HMAC-SHA256
signature; the server refuses unsigned or modified manifests. Do not commit
the key. Prediction audit entries include model version, ordered feature
names, UTC timestamp, request/response SHA-256 hashes, and a chained audit
hash, persisted to the ignored `audit.jsonl` file. API and training events are
emitted as one-record-per-line JSON logs. Prediction monitoring statistics are
persisted without raw feature values in the ignored `monitoring.jsonl` file.
The monitor compares request feature/output populations to the baseline in each
model manifest, reporting mean/std z-scores and PSI (drift thresholds: z-score
3 or PSI 0.2). Newly trained artifacts include a 90% marginal interval
calibrated from walk-forward validation absolute errors. Older artifacts without
that calibration use a labeled normal approximation based on validation RMSE;
intervals are not joint or conditional coverage guarantees. Existing manifests
without prediction-population baselines report feature z-scores from their
scalers; prediction drift comparisons begin after a model is trained with the
new monitoring metadata.

#### Configure and deploy

The API accepts these environment variables; matching CLI flags override them:

| Environment variable | CLI flag | Default | Purpose |
| --- | --- | --- | --- |
| `AGRISHIELD_HOST` | `--host` | `127.0.0.1` | Bind address (`0.0.0.0` inside a container) |
| `AGRISHIELD_PORT` | `--port` | `8000` | Listen port, from 0 to 65535 |
| `AGRISHIELD_ARTIFACT_KEY` | `--artifact-key` | unset | Require HMAC verification of the active manifest |
| `AGRISHIELD_AUDIT_PATH` | `--audit-path` | `audit.jsonl` under dashboard root | Chained prediction audit log |
| `AGRISHIELD_REGISTRY_PATH` | `--registry-path` | `model_registry.json` under dashboard root | Active model registry |
| `AGRISHIELD_DASHBOARD_ROOT` | `--dashboard-root` | API directory | Static dashboard and default artifact directory |
| `AGRISHIELD_LOG_LEVEL` | `--log-level` | `INFO` | `CRITICAL`, `ERROR`, `WARNING`, `INFO`, or `DEBUG` |

Relative audit and registry paths are resolved from the dashboard root. A
configured artifact key must be non-empty and match an HMAC signature already
stored in the registry; the API does not sign artifacts during startup. Missing,
invalid, or mismatched registries/manifests, unreadable model artifacts,
unwritable audit storage, invalid configuration, and port binding failures stop
startup with an `AgriShield startup failed` message. Check `GET /v1/health` for
the loaded API and model version. Invalid JSON, schema, and feature requests
return HTTP 400; oversized prediction bodies return 413; unknown `/v1` routes
return 404; unexpected prediction failures return a generic HTTP 500 and are
recorded in the structured server logs.

Run with Docker Compose from the repository root:

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/v1/health
docker compose logs -f agrishield-api
```

Set `AGRISHIELD_PORT` in `.env` to change both the container listen port and
published host port. Keep `AGRISHIELD_ARTIFACT_KEY` unset when using the
unsigned default registry. To enforce signed-manifest verification, set the
key in the deployment environment and use a registry signed with that same
key; never commit `.env` or the secret. Compose persists audit/monitoring logs
in the `agrishield-data` named volume. Model registry and artifacts are baked
into the image, so rebuild after changing them. The container healthcheck polls
`/v1/health`; an unhealthy state means startup did not complete or the API is
not responding. This API has no client authentication; do not expose it directly
to untrusted networks.

#### Registering and rolling back models

Training writes immutable, version-named weight/manifest pairs rather than
overwriting the active model. From `dashboard_v2`, train a candidate and inspect
the registry:

```powershell
python train_deep_learning.py --model-version candidate-20261126
python model_registry.py --list
```

Register the resulting manifest after validation. If
`AGRISHIELD_ARTIFACT_KEY` is configured, keep it available in this shell so
registration signs the candidate manifest:

```powershell
python model_registry.py --register candidate-20261126 `
  --manifest pytorch_food_security_model.candidate-20261126.manifest.json
python model_registry.py --activate candidate-20261126 --reason "validation approved"
python api_server.py
```

Each model version remains in the registry, and `rollback_metadata` records
activation and rollback history. The server loads the active model at startup,
so restart it after activation. To revert, select a previously registered
version and restart the API process:

```powershell
python model_registry.py --list
python model_registry.py --rollback 1.0.0 --reason "candidate drifted"
python api_server.py
```

Rollback verifies the selected manifest digest (and HMAC when configured)
before atomically switching the active registry pointer. Keep candidate and
previous model weight/manifest files together in `dashboard_v2`; do not delete
registered artifacts needed for recovery.

Run the API unit tests from `dashboard_v2`:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Limitations and future work

- This is a prototype data-science workflow rather than a production operational platform.
- Forecast accuracy depends on data quality and region coverage.
- The system can be extended with live ingestion, stronger evaluation pipelines, and more robust deployment workflows.
