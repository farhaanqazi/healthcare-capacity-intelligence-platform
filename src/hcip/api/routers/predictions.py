from fastapi import APIRouter, Depends, HTTPException
import pandas as pd

from hcip.api.schemas import (
    BreachRiskResponse, DemandResponse, PredictionRequest, WaitTimeResponse
)
from hcip.api.deps import ModelRegistry, get_model_registry
from hcip.modeling import FEATURE_COLS

router = APIRouter(tags=["predictions"])

def _prepare_features(req: PredictionRequest) -> pd.DataFrame:
    # Ensure all required features are present
    missing = [f for f in FEATURE_COLS if f not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    
    # Create a single-row DataFrame in the exact order
    df = pd.DataFrame([req.features])
    return df[FEATURE_COLS]

@router.post("/predict-demand", response_model=DemandResponse)
def predict_demand(req: PredictionRequest, registry: ModelRegistry = Depends(get_model_registry)):
    if registry.demand_model is None:
        raise HTTPException(status_code=503, detail="Demand model not loaded")
    
    features_df = _prepare_features(req)
    # The demand model in train.py predicted the delta. We need the current "total_waiting" to add to it.
    current_total = req.features.get("total_waiting", 0.0)
    delta = registry.demand_model.predict(features_df)[0]
    pred_total = max(0.0, current_total + delta)
    
    return DemandResponse(
        pred_total_next=float(pred_total),
        provider_code=req.provider_code,
        specialty_code=req.specialty_code
    )

@router.post("/predict-wait-time", response_model=WaitTimeResponse)
def predict_wait_time(req: PredictionRequest, registry: ModelRegistry = Depends(get_model_registry)):
    if registry.wait_time_model is None:
        raise HTTPException(status_code=503, detail="Wait time model not loaded")
    
    features_df = _prepare_features(req)
    # The wait time model predicted the delta. We need "pct_within_18wk"
    current_pct = req.features.get("pct_within_18wk", 0.0)
    delta = registry.wait_time_model.predict(features_df)[0]
    pred_pct = max(0.0, min(1.0, current_pct + delta))
    
    return WaitTimeResponse(
        pred_pct_next=float(pred_pct),
        provider_code=req.provider_code,
        specialty_code=req.specialty_code
    )

@router.post("/predict-breach-risk", response_model=BreachRiskResponse)
def predict_breach_risk(req: PredictionRequest, registry: ModelRegistry = Depends(get_model_registry)):
    if registry.breach_model is None:
        raise HTTPException(status_code=503, detail="Breach model not loaded")
    
    features_df = _prepare_features(req)
    prob = registry.breach_model.predict_proba(features_df)[0, 1]
    
    return BreachRiskResponse(
        pred_breach_prob=float(prob),
        provider_code=req.provider_code,
        specialty_code=req.specialty_code
    )
