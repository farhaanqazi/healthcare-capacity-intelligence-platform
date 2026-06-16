"""Training script for the ML models.

Trains the demand, wait-time, and breach-risk models using the feature store,
saves the artifacts to the model registry, and writes fact_predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from hcip.modeling import (FEATURE_COLS, classifier, features_path,
                           load_features, regressor, save_model)
from hcip.gold import write_gold

def train_and_evaluate() -> None:
    print("Loading features...")
    df = load_features()
    
    # Identify the last available month
    months = sorted(df["period_date"].unique())
    last_month = months[-1]
    
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = []
    
    print("Training Breach Risk Model...")
    # Target: target_breach_next
    breach_df = df.dropna(subset=["target_breach_next", "pct_within_18wk"]).copy()
    breach_df["y"] = breach_df["target_breach_next"].astype(int)
    
    clf = classifier(n_estimators=300).fit(breach_df[FEATURE_COLS], breach_df["y"])
    save_model(clf, models_dir / "breach_model.pkl")
    
    # Predict for the last month to simulate 'next month' predictions
    curr = df[df["period_date"] == last_month].copy()
    curr["pred_breach_prob"] = clf.predict_proba(curr[FEATURE_COLS])[:, 1]
    
    print("Training Demand Model (Level)...")
    # Using simple XGBoost for demand as a baseline for the registry
    demand_df = df.dropna(subset=["target_total_next", "total_waiting"]).copy()
    demand_df["delta"] = demand_df["target_total_next"] - demand_df["total_waiting"]
    
    dem_reg = regressor(n_estimators=300).fit(demand_df[FEATURE_COLS], demand_df["delta"])
    save_model(dem_reg, models_dir / "demand_model.pkl")
    
    curr["pred_total_next"] = curr["total_waiting"] + dem_reg.predict(curr[FEATURE_COLS])
    curr["pred_total_next"] = curr["pred_total_next"].clip(lower=0)
    
    print("Training Wait Time Model (Pct within 18wk)...")
    wait_df = df.dropna(subset=["target_pct_within_18_next", "pct_within_18wk"]).copy()
    wait_df["delta"] = wait_df["target_pct_within_18_next"] - wait_df["pct_within_18wk"]
    
    wait_reg = regressor(n_estimators=300).fit(wait_df[FEATURE_COLS], wait_df["delta"])
    save_model(wait_reg, models_dir / "wait_time_model.pkl")
    
    curr["pred_pct_next"] = curr["pct_within_18wk"] + wait_reg.predict(curr[FEATURE_COLS])
    curr["pred_pct_next"] = curr["pred_pct_next"].clip(0, 1)
    
    # Create fact_predictions
    fact_preds = curr[["provider_code", "specialty_code", "icb_code", "period_date", 
                       "pred_breach_prob", "pred_total_next", "pred_pct_next"]].copy()
    
    # Write to local parquet
    pred_path = Path("data/processed/fact_predictions.parquet")
    fact_preds.to_parquet(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")
    
    # Save a metadata file
    meta = {
        "training_date": datetime.now().isoformat(),
        "latest_data_month": str(last_month),
        "models": ["breach_model.pkl", "demand_model.pkl", "wait_time_model.pkl"]
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print("Training complete! Models saved to data/models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HCIP models and save to registry.")
    args = parser.parse_args()
    train_and_evaluate()
