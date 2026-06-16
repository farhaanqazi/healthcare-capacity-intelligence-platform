# Healthcare Capacity Intelligence Platform

End-to-end analytics platform for **NHS Referral-to-Treatment (RTT) waiting lists**: a
medallion data pipeline (Bronze → Silver → Gold star schema) and three machine-learning
models — demand forecasting, waiting-time regression, and breach-risk scoring — built on
real public NHS data, with a production-grade test suite. Local-first and cloud-ready
(SQLite → PostgreSQL), with a Gold layer ready to connect to Power BI.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![ML](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![BI](https://img.shields.io/badge/BI-Power_BI-yellow.svg)
![Tests](https://img.shields.io/badge/tests-14_gates-success.svg)

---

## Why this project

NHS hospitals must keep ≥92% of patients within the 18-week RTT standard. This platform
turns the monthly public waiting-list extracts into:

- a **clean, conformed dataset** at a single analytical grain (hospital × specialty × month);
- a **dimensional warehouse** (star schema) for BI and reporting;
- **forecasts and risk scores** to flag where the standard is likely to be breached next month.

It is built as a portfolio-grade demonstration of data engineering, data warehousing,
and honest, baseline-driven machine learning.

---

## Architecture

```
NHS RTT CSVs ──► Bronze (raw, immutable)
                   │
                   ▼
              Silver  ── rtt_unified.parquet      (cleaned, conformed, derived measures)
                   │
        ┌──────────┴───────────┐
        ▼                      ▼
   Gold star schema       Feature store
   (facts + dims)         rtt_features.parquet
   SQLite / PostgreSQL          │
        │                       ▼
        ▼                  ML models (demand · waiting time · breach risk)
   Power BI                     │
                                ▼
                          Test suite (14 gates)
```

The full nine-layer target design (including the FastAPI service, model registry,
monitoring, and AWS deployment) is documented in [architecture.md](architecture.md).

### Data grain

A single row is one **provider (hospital) × treatment-function (specialty) × month**.
The national figure reconciles end-to-end (e.g. ~7.05M waiting in Mar-2026). Two source
gotchas are handled explicitly: the `Total` column is empty on detail rows (the total is
re-derived from the wait-band counts), and the `C_999` treatment function is an
all-specialties summary line that would double-count if included, so it is excluded.

---

## Machine-learning models

All models use **time-based splits** (never random) and are measured against an honest
**persistence baseline** (next month = this month). A model is only claimed as useful
where it beats that baseline.

| Model | Target | Algorithm | Result |
|---|---|---|---|
| **Breach risk** | Will the specialty breach the 18-week standard next month? | XGBoost classifier | **Clear win** — ROC AUC ≈ 0.98; score = P(breach) × 100 |
| **Demand** | Next month's total waiting list | XGBoost (FVA hybrid) | Persistence-dominated at 1-month horizon → routed champion-challenger holds parity, never worse |
| **Waiting time** | Next month's % within 18 weeks | XGBoost / LightGBM | Persistence-dominated (MAE parity, marginally better RMSE) |

The key, deliberately reported finding: **level series are near-random-walk at a one-month
horizon, so they are persistence-dominated, while the binary breach outcome carries strong
signal.** That contrast — and a Forecast-Value-Added wrapper that guarantees the model is
never worse than the baseline — is the methodological core of the project.

---

## Repository layout

```
.
├── architecture.md          # Full nine-layer solution design
├── notebooks/               # The visible, step-by-step build (run in order 01 → 07)
│   ├── 01_explore_raw_data.ipynb
│   ├── 02_unify_extracts.ipynb        # Bronze → Silver
│   ├── 03_feature_engineering.ipynb   # Silver → feature store
│   ├── 04_demand_model.ipynb
│   ├── 05_breach_model.ipynb
│   ├── 06_waiting_time_model.ipynb
│   └── 07_gold_star_schema.ipynb      # Silver → Gold (star schema)
├── src/hcip/                # Reusable library shared by notebooks, tests, and serving
│   ├── modeling.py          # Feature defs, model constructors, evaluation (FVA, ECE, Brier)
│   └── gold.py              # Star-schema builder + warehouse loader (SQLAlchemy)
├── tests/                   # Production-grade test suite (14 gates)
│   ├── test_models.py             # Leakage, baseline-beat, AUC floor, overfit, determinism
│   ├── test_calibration_slices.py # Calibration (ECE/Brier) + per-segment AUC floors
│   └── test_data_validation.py    # Pandera schema, grain uniqueness, reconciliation
├── conftest.py              # Shared fixtures (walk-forward windows, holdouts)
├── docs/                    # Documentation including the Power BI integration guide
├── viz/                     # Power BI Dashboards (hcip.pbix)
└── pyproject.toml
```

> **Data is not committed.** `data/` (raw 1.9 GB + derived) is git-ignored; the notebooks
> regenerate the Silver, feature, and Gold layers from the raw extracts. See
> [Getting the data](#getting-the-data).

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) and Python 3.12 (ML wheels are not yet
reliably available on 3.14).

```bash
uv venv --python 3.12
uv pip install -e ".[notebooks,dev]"
```

Register the Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name hcip --display-name "Python (HCIP)"
```

### Getting the data

1. Download the monthly **Consultant-led RTT Waiting Times** CSVs from
   [NHS England](https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/)
   (the "Full CSV data file" for each month) into `data/raw/`.
2. Open the notebooks and run them in order **01 → 07** with the `Python (HCIP)` kernel.
   This rebuilds `data/interim/`, `data/processed/`, and `data/gold/`.

---

## Running the tests

```bash
uv run pytest -q
```

The suite trains models on time-based splits and fails the build on: feature leakage, a
model losing to its persistence baseline, AUC below floor, train/test overfitting,
miscalibration, per-segment AUC drops, schema violations, or grain duplication.

---

## Business intelligence (Power BI)

The Gold layer writes both a SQLite database (`data/gold/hcip_gold.db`) and Parquet exports
(`data/gold/*.parquet`).

A complete, pre-built Power BI dashboard is available at `viz/hcip.pbix`. This dashboard includes the full semantic model (Star Schema) linking the `fact_predictions` and `fact_waiting_list` tables to the hospital, specialty, and date dimensions. It features heatmaps for breach risk, demand forecasting trendlines, and scatter plots for worst-offenders.

For instructions on how to connect your own Power BI environment from scratch or refresh the data connections, see the [Power BI Integration Guide](docs/powerbi_integration.md).

In a production environment, the same model can be re-pointed at the PostgreSQL warehouse.

---

## Roadmap

- [x] Bronze → Silver → feature store → Gold pipeline (notebooks 01–07)
- [x] Three ML models with honest baselines and a 14-gate test suite
- [x] End-to-end Power BI Integration (`viz/hcip.pbix`)
- [ ] Training script + model registry (persisted models, versioned metrics, `fact_predictions`)
- [ ] FastAPI prediction service (`/predict-demand`, `/predict-wait-time`, `/predict-breach-risk`, `/health`, `/model-metrics`)
- [ ] CI: run the test suite in GitHub Actions

---

## License

[MIT](LICENSE) © 2026 Farhaan Qazi.

Data © NHS England, published under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
