import sys
import os
import pytest
from fastapi.testclient import TestClient

# Agregar src/api al path para que pytest encuentre los módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'api'))

from main import app

client = TestClient(app)


def test_health_ok():
    """El endpoint /health debe responder 200 con status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["modelo"] == "lightgbm_scoring"
    assert data["features"] == 65


def test_model_info_ok():
    """El endpoint /model-info debe responder 200 con metadata correcta."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["nombre"] == "lightgbm_scoring"
    assert data["framework"] == "LightGBM"
    assert data["metricas"]["auc"] == 0.768


def test_predict_default_values():
    """El endpoint /predict debe responder 200 con valores default."""
    payload = {
        "EXT_SOURCE_2": 0.5,
        "DAYS_BIRTH": -15000,
        "DAYS_EMPLOYED": -2000,
        "AMT_CREDIT": 500000,
        "AMT_INCOME_TOTAL": 150000
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probabilidad_incumplimiento" in data
    assert "score_crediticio" in data
    assert "decision" in data
    assert "nivel_riesgo" in data
    assert "top_variables" in data
    assert len(data["top_variables"]) == 3
    assert 300 <= data["score_crediticio"] <= 850
    assert 0.0 <= data["probabilidad_incumplimiento"] <= 1.0


def test_predict_perfil_bajo_riesgo():
    """Perfil de bajo riesgo debe producir score alto y decisión de aprobar."""
    payload = {
        "EXT_SOURCE_1": 0.85,
        "EXT_SOURCE_2": 0.90,
        "EXT_SOURCE_3": 0.88,
        "DAYS_BIRTH": -18000,
        "DAYS_EMPLOYED": -5000,
        "AMT_CREDIT": 200000,
        "AMT_INCOME_TOTAL": 300000,
        "AMT_ANNUITY": 15000
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["probabilidad_incumplimiento"] < 0.35
    assert data["score_crediticio"] > 600


def test_predict_campos_invalidos():
    """Campos con tipos incorrectos deben devolver error 422."""
    payload = {
        "EXT_SOURCE_2": "no_es_numero",
        "DAYS_BIRTH": "texto"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422