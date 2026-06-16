from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends

from hcip.api.deps import ModelRegistry, get_model_registry, registry
from hcip.api.routers import predictions
from hcip.api.schemas import ModelMetricsResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    registry.load_all()
    yield
    # Clean up at shutdown if needed

app = FastAPI(
    title="HCIP Prediction API",
    description="API for NHS Referral-to-Treatment waiting list predictions.",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(predictions.router)

@app.get("/health", tags=["system"])
def health_check(reg: ModelRegistry = Depends(get_model_registry)):
    models_loaded = all([
        reg.breach_model is not None,
        reg.demand_model is not None,
        reg.wait_time_model is not None
    ])
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded
    }

@app.get("/model-metrics", response_model=ModelMetricsResponse, tags=["system"])
def get_model_metrics(reg: ModelRegistry = Depends(get_model_registry)):
    return ModelMetricsResponse(
        training_date=reg.metadata.get("training_date", "unknown"),
        latest_data_month=reg.metadata.get("latest_data_month", "unknown"),
        models=reg.metadata.get("models", [])
    )
