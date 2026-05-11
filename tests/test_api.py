import sys
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'api'))


def make_mock_model():
    """Crea un modelo simulado que imita LGBMClassifier."""
    mock = MagicMock()
    mock.feature_name_ = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_CREDIT',
        'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION',
        'DAYS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
        'HOUR_APPR_PROCESS_START', 'OWN_CAR_AGE', 'CNT_CHILDREN',
        'CNT_FAM_MEMBERS', 'TOTALAREA_MODE', 'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
        'FLAG_EMAIL', 'REG_CITY_NOT_LIVE_CITY',
        'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
    ]
    mock.predict.return_value = np.array([0.15])
    return mock


def make_mock_explainer(n_features):
    """Crea un SHAP explainer simulado."""
    mock = MagicMock()
    shap_vals = np.zeros(n_features)
    shap_vals[0] = 0.3
    shap_vals[1] = -0.2
    shap_vals[2] = 0.1
    mock.shap_values.return_value = [shap_vals]
    return mock


@pytest.fixture(autouse=True)
def mock_model_and_shap():
    """Fixture que reemplaza el modelo real y SHAP en todos los tests."""
    mock_model = make_mock_model()
    n_features = len(mock_model.feature_name_)
    mock_explainer = make_mock_explainer(n_features)

    with patch('model_loader._model', mock_model), \
         patch('model_loader._feature_names',
               mock_model.feature_name_), \
         patch('shap.TreeExplainer', return_value=mock_explainer):
        yield


from main import app
client = TestClient(app)


def test_health_ok():
    """El endpoint /health debe responder 200 con status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["modelo"] == "lightgbm_scoring"


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
    """Con PD=0.15 el score debe ser mayor a 600."""
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
    assert data["probabilidad_incumplimiento"] == 0.15
    assert data["score_crediticio"] > 600


def test_predict_campos_invalidos():
    """Campos con tipos incorrectos deben devolver error 422."""
    payload = {
        "EXT_SOURCE_2": "no_es_numero",
        "DAYS_BIRTH": "texto"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422