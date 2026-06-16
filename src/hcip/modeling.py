"""Reusable modelling logic for the Healthcare Capacity Intelligence Platform.

Centralises feature definitions, model constructors, and evaluation routines so
that notebooks, the test suite, and the training/serving code all share one
implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
import pickle

SEED = 42
STANDARD = 0.92  # NHS 18-week standard: >=92% of pathways within 18 weeks
SERIES = ["provider_code", "specialty_code"]

FEATURE_COLS = [
    "total_waiting", "pct_within_18wk", "breach_rate", "over_52_share", "over_104_share",
    "month", "quarter", "month_sin", "month_cos",
    "lag1_total", "lag2_total", "lag3_total", "lag12_total", "lag1_breach", "lag12_breach",
    "roll3_total", "roll6_total", "roll12_total", "roll3_std_total",
    "mom_change_total", "mom_pct_total", "yoy_pct_total",
]


def features_path() -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "data" / "processed" / "rtt_features.parquet"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("rtt_features.parquet not found under any parent of the cwd.")


def load_features() -> pd.DataFrame:
    return pd.read_parquet(features_path())


def regressor(n_estimators: int = 300, n_jobs: int = -1) -> XGBRegressor:
    return XGBRegressor(n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=n_jobs)


def classifier(n_estimators: int = 300, n_jobs: int = -1) -> XGBClassifier:
    return XGBClassifier(n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=n_jobs,
                         eval_metric="logloss")


def valid_target_months(df: pd.DataFrame, target: str = "target_breach_next") -> list:
    """Feature months that have a defined target (i.e. a next month exists)."""
    return sorted(df.dropna(subset=[target])["period_date"].unique())


def breach_fit_eval(df: pd.DataFrame, test_month, n_estimators: int = 200) -> dict:
    """Train breach classifier on all months before `test_month`, evaluate on it."""
    d = df.dropna(subset=["target_breach_next", "pct_within_18wk"]).copy()
    d["y"] = d["target_breach_next"].astype(int)
    tr = d[d["period_date"] < test_month]
    te = d[d["period_date"] == test_month]
    clf = classifier(n_estimators).fit(tr[FEATURE_COLS], tr["y"])
    p_tr = clf.predict_proba(tr[FEATURE_COLS])[:, 1]
    p_te = clf.predict_proba(te[FEATURE_COLS])[:, 1]
    base = (te["pct_within_18wk"] < STANDARD).astype(int).to_numpy()  # breach-persistence baseline
    two_classes = te["y"].nunique() > 1
    return {
        "test_month": pd.Timestamp(test_month),
        "n_test": len(te),
        "train_auc": roc_auc_score(tr["y"], p_tr),
        "test_auc": roc_auc_score(te["y"], p_te) if two_classes else np.nan,
        "baseline_auc": roc_auc_score(te["y"], base) if two_classes else np.nan,
        "proba": p_te,
        "y": te["y"].to_numpy(),
        "test_df": te,
    }


def breach_holdout(df: pd.DataFrame, test_from=pd.Timestamp("2025-12-01"), n_estimators: int = 200) -> dict:
    """Single train/test holdout for calibration and slice analysis."""
    d = df.dropna(subset=["target_breach_next", "pct_within_18wk"]).copy()
    d["y"] = d["target_breach_next"].astype(int)
    tr = d[d["period_date"] < test_from]
    te = d[d["period_date"] >= test_from]
    clf = classifier(n_estimators).fit(tr[FEATURE_COLS], tr["y"])
    proba = clf.predict_proba(te[FEATURE_COLS])[:, 1]
    return {"clf": clf, "test_df": te, "proba": proba, "y": te["y"].to_numpy()}


def fva_forecast(df: pd.DataFrame, level_col: str, current_col: str,
                 train_end=pd.Timestamp("2025-09-01"),
                 valid_end=pd.Timestamp("2025-12-01"),
                 test_start=pd.Timestamp("2025-12-01"),
                 n_estimators: int = 300, clip01: bool = False) -> dict:
    """Forecast Value Added champion-challenger forecast for a near-persistent series.

    Champion = persistence (current_col). Challenger = XGBoost on the delta.
    Routing decided on the validation window only. Returns level predictions on test.
    """
    d = df.dropna(subset=[level_col, current_col]).copy()
    tr = d[d["period_date"] < train_end]
    va = d[(d["period_date"] >= train_end) & (d["period_date"] < valid_end)].copy()
    te = d[d["period_date"] >= test_start].copy()

    model = regressor(n_estimators).fit(tr[FEATURE_COLS], tr[level_col] - tr[current_col])

    def ml_level(part: pd.DataFrame) -> np.ndarray:
        out = part[current_col].to_numpy() + model.predict(part[FEATURE_COLS])
        return np.clip(out, 0, 1) if clip01 else out

    va["e_persist"] = (va[level_col] - va[current_col]).abs()
    va["e_ml"] = (va[level_col] - ml_level(va)).abs()
    g = va.groupby(SERIES)[["e_persist", "e_ml"]].mean()
    ml_wins = set(g.index[g["e_ml"] < g["e_persist"]])

    keys = list(zip(te["provider_code"], te["specialty_code"]))
    route = np.array([k in ml_wins for k in keys])
    hybrid = np.where(route, ml_level(te), te[current_col].to_numpy())
    y = te[level_col].to_numpy()
    return {
        "y": y,
        "persistence": te[current_col].to_numpy(),
        "hybrid": hybrid,
        "persistence_mae": mean_absolute_error(y, te[current_col]),
        "hybrid_mae": mean_absolute_error(y, hybrid),
        "routed_fraction": float(route.mean()),
    }


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Mean gap between predicted confidence and observed frequency across probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(proba, edges[1:-1]), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(proba[mask].mean() - y_true[mask].mean())
    return float(ece)


def brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(brier_score_loss(np.asarray(y_true, dtype=int), np.asarray(proba, dtype=float)))


def save_model(model: Any, path: Path | str) -> None:
    """Serialize a trained model to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path | str) -> Any:
    """Load a serialized model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
