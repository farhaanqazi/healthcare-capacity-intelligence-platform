from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    provider_code: str = Field(..., description="Hospital Provider Code")
    specialty_code: str = Field(..., description="Treatment Function Code")
    features: dict[str, float] = Field(..., description="Dictionary of engineered feature values required by the model")

class DemandResponse(BaseModel):
    pred_total_next: float
    provider_code: str
    specialty_code: str

class WaitTimeResponse(BaseModel):
    pred_pct_next: float
    provider_code: str
    specialty_code: str

class BreachRiskResponse(BaseModel):
    pred_breach_prob: float
    provider_code: str
    specialty_code: str

class ModelMetricsResponse(BaseModel):
    training_date: str
    latest_data_month: str
    models: list[str]
