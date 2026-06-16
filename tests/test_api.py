import pytest
from fastapi.testclient import TestClient
from hcip.api.main import app

client = TestClient(app)

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True

def test_predict_breach_risk():
    with TestClient(app) as client:
        payload = {
            "provider_code": "R1H",
            "specialty_code": "C_100",
            "features": {
                "total_waiting": 1000.0, "pct_within_18wk": 0.85, "breach_rate": 0.15,
                "over_52_share": 0.05, "over_104_share": 0.0, "month": 6.0, "quarter": 2.0,
                "month_sin": 0.5, "month_cos": 0.5, "lag1_total": 950.0, "lag2_total": 900.0,
                "lag3_total": 850.0, "lag12_total": 800.0, "lag1_breach": 0.10, "lag12_breach": 0.05,
                "roll3_total": 900.0, "roll6_total": 850.0, "roll12_total": 800.0, "roll3_std_total": 50.0,
                "mom_change_total": 50.0, "mom_pct_total": 0.05, "yoy_pct_total": 0.20
            }
        }
        response = client.post("/predict-breach-risk", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "pred_breach_prob" in data
        assert data["provider_code"] == "R1H"
