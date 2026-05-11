from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import shap
import warnings
warnings.filterwarnings('ignore')

from schemas import SolicitudCredito, RespuestaCredito
from model_loader import get_model

app = FastAPI(
    title="Credit Risk Scoring API",
    description="API REST para evaluación de riesgo crediticio. "
                "Modelo LightGBM con explicabilidad SHAP.",
    version="1.0.0"
)

# Nombres legibles de variables (mismo diccionario que el dashboard)
NOMBRES_LEGIBLES = {
    'EXT_SOURCE_1': 'Score externo 1',
    'EXT_SOURCE_2': 'Score externo 2',
    'EXT_SOURCE_3': 'Score externo 3',
    'DAYS_BIRTH': 'Edad del solicitante',
    'DAYS_EMPLOYED': 'Antigüedad laboral',
    'AMT_CREDIT': 'Monto del crédito',
    'AMT_INCOME_TOTAL': 'Ingreso anual',
    'AMT_ANNUITY': 'Cuota mensual',
    'AMT_GOODS_PRICE': 'Precio del bien',
    'DAYS_ID_PUBLISH': 'Antigüedad del ID',
    'DAYS_REGISTRATION': 'Antigüedad de registro',
    'DAYS_LAST_PHONE_CHANGE': 'Último cambio de teléfono',
    'REGION_POPULATION_RELATIVE': 'Densidad de región',
    'HOUR_APPR_PROCESS_START': 'Hora de solicitud',
    'OWN_CAR_AGE': 'Antigüedad del auto',
    'CNT_CHILDREN': 'Número de hijos',
    'CNT_FAM_MEMBERS': 'Miembros del hogar',
    'FLAG_OWN_CAR': 'Tiene auto',
    'FLAG_OWN_REALTY': 'Tiene propiedad',
    'FLAG_WORK_PHONE': 'Teléfono de trabajo',
    'FLAG_PHONE': 'Teléfono fijo',
    'FLAG_EMAIL': 'Tiene email',
    'REG_CITY_NOT_LIVE_CITY': 'Ciudad registro ≠ residencia',
    'REG_CITY_NOT_WORK_CITY': 'Ciudad registro ≠ trabajo',
    'LIVE_CITY_NOT_WORK_CITY': 'Ciudad residencia ≠ trabajo',
    'TOTALAREA_MODE': 'Área total vivienda',
}


def calcular_score(pd_value: float) -> int:
    """Convierte probabilidad de incumplimiento a score (300-850)."""
    import math
    if pd_value <= 0:
        pd_value = 0.001
    if pd_value >= 1:
        pd_value = 0.999
    odds = (1 - pd_value) / pd_value
    score = 600 + 20 * math.log2(odds)
    return int(max(300, min(850, score)))


def clasificar_riesgo(pd_value: float) -> tuple:
    """Retorna (decision, nivel_riesgo) según la PD."""
    if pd_value < 0.10:
        return "APROBAR", "Bajo"
    elif pd_value < 0.20:
        return "APROBAR CON SEGUIMIENTO", "Medio-Bajo"
    elif pd_value < 0.35:
        return "REVISAR MANUALMENTE", "Medio"
    else:
        return "RECHAZAR", "Alto"


@app.get("/health")
def health():
    """Verifica que el servicio y el modelo están activos."""
    try:
        model, features = get_model()
        return {
            "status": "ok",
            "modelo": "lightgbm_scoring",
            "alias": "production",
            "features": len(features)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/model-info")
def model_info():
    """Retorna metadata del modelo en producción."""
    model, feature_names = get_model()
    return {
        "nombre": "lightgbm_scoring",
        "version": "3",
        "alias": "production",
        "framework": "LightGBM",
        "features": len(feature_names),
        "descripcion": "LightGBM entrenado en Home Credit Default Risk dataset",
        "metricas": {
            "auc": 0.768,
            "umbral_optimo": 0.59
        }
    }


@app.post("/predict", response_model=RespuestaCredito)
def predict(solicitud: SolicitudCredito):
    """
    Evalúa una solicitud de crédito.

    Retorna probabilidad de incumplimiento, score crediticio,
    decisión y top 3 variables más influyentes (SHAP).
    """
    try:
        model, feature_names = get_model()

        # Construir DataFrame con las features del modelo
        input_data = solicitud.model_dump()
        df = pd.DataFrame([input_data])

        # Alinear columnas al orden exacto del modelo
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
        df = df.fillna(0)

        # Predicción
        pd_value = float(model.predict(df)[0])
        score = calcular_score(pd_value)
        decision, nivel_riesgo = clasificar_riesgo(pd_value)

        # SHAP - top 3 variables
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        shap_array = np.array(shap_values[0])

        indices_top = np.argsort(np.abs(shap_array))[::-1][:3]
        top_variables = []
        for idx in indices_top:
            nombre_tecnico = feature_names[idx]
            nombre_legible = NOMBRES_LEGIBLES.get(
                nombre_tecnico, nombre_tecnico
            )
            top_variables.append({
                "variable": nombre_legible,
                "contribucion": round(float(shap_array[idx]), 4),
                "direccion": "aumenta riesgo" if shap_array[idx] > 0
                             else "reduce riesgo"
            })

        mensaje = (
            f"Solicitud evaluada. PD={pd_value:.1%}, "
            f"Score={score} pts. Decisión: {decision}."
        )

        return RespuestaCredito(
            probabilidad_incumplimiento=round(pd_value, 4),
            score_crediticio=score,
            decision=decision,
            nivel_riesgo=nivel_riesgo,
            top_variables=top_variables,
            mensaje=mensaje
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))