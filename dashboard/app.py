import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────────
# CARGAR MODELO
# ─────────────────────────────────────────────

@st.cache_resource
# cache_resource evita recargar el modelo en cada
# interacción del usuario — mejora rendimiento significativamente
def load_model():
    model_path = Path("../mlruns/521656391598123434/models/m-4b3e2786ebbf4f6982ed89b1009b26b8/artifacts/model.pkl")
    if not model_path.exists():
        return None
    return joblib.load(model_path)

model = load_model()

# ─────────────────────────────────────────────
# CARGAR DATOS DE REFERENCIA
# ─────────────────────────────────────────────

@st.cache_data
def load_reference_data():
    df = pd.read_csv("../data/processed/df_lgbm.csv")
    return df

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🏦 Credit Risk Scoring Dashboard")
st.markdown("""
Sistema de scoring crediticio basado en LightGBM.
Predice la probabilidad de incumplimiento de solicitantes de crédito.
""")
st.divider()

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🎯 Scoring en Tiempo Real",
    "📊 Análisis del Portfolio",
    "🔍 Métricas del Modelo"
])

# ─────────────────────────────────────────────
# TAB 1 — SCORING EN TIEMPO REAL
# ─────────────────────────────────────────────

with tab1:
    st.header("Evaluación de Solicitante")
    st.markdown("Ingresa los datos del solicitante para obtener su score de riesgo.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Datos Financieros")
        amt_income = st.number_input(
            "Ingreso anual ($)",
            min_value=10000,
            max_value=1000000,
            value=150000,
            step=10000
        )
        amt_credit = st.number_input(
            "Monto del crédito ($)",
            min_value=10000,
            max_value=4000000,
            value=500000,
            step=10000
        )
        amt_annuity = st.number_input(
            "Anualidad ($)",
            min_value=1000,
            max_value=200000,
            value=25000,
            step=1000
        )

    with col2:
        st.subheader("👤 Datos del Solicitante")
        days_birth = st.slider(
            "Edad (años)",
            min_value=18,
            max_value=70,
            value=35
        )
        days_employed = st.slider(
            "Antigüedad laboral (años)",
            min_value=0,
            max_value=40,
            value=5
        )
        ext_source_2 = st.slider(
            "Score externo (Buró)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )

    with col3:
        st.subheader("📈 Historial Crediticio")
        ext_source_3 = st.slider(
            "Score externo 2",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
        bureau_active = st.number_input(
            "Créditos activos en Buró",
            min_value=0,
            max_value=20,
            value=2
        )
        bureau_overdue = st.number_input(
            "Créditos vencidos en Buró",
            min_value=0,
            max_value=10,
            value=0
        )

    st.divider()

    if st.button("🔮 Calcular Score de Riesgo", type="primary"):
        if model is None:
            st.error("⚠️ Modelo no disponible. Verifica la ruta del archivo.")
        else:
            # Construir vector de features con valores por defecto
            # para features no capturadas en el formulario
            try:
                df_ref = load_reference_data()
                feature_cols = [c for c in df_ref.columns
                               if c not in ["TARGET", "SK_ID_CURR"]]

                # Crear fila con medianas como base
                input_data = df_ref[feature_cols].median().to_frame().T

                # Sobreescribir con valores del formulario
                feature_map = {
                    "AMT_INCOME_TOTAL": amt_income,
                    "AMT_CREDIT": amt_credit,
                    "AMT_ANNUITY": amt_annuity,
                    "DAYS_BIRTH": -days_birth * 365,
                    "DAYS_EMPLOYED": -days_employed * 365,
                    "EXT_SOURCE_2": ext_source_2,
                    "EXT_SOURCE_3": ext_source_3,
                }

                for feature, value in feature_map.items():
                    if feature in input_data.columns:
                        input_data[feature] = value

                # Predicción
                pd_prob = model.predict_proba(input_data)[0][1]
                score = int((1 - pd_prob) * 1000)

                # Mostrar resultado
                col_res1, col_res2, col_res3 = st.columns(3)

                with col_res1:
                    st.metric(
                        label="Probabilidad de Incumplimiento",
                        value=f"{pd_prob:.1%}"
                    )

                with col_res2:
                    st.metric(
                        label="Score de Crédito",
                        value=f"{score} pts"
                    )

                with col_res3:
                    if pd_prob < 0.10:
                        decision = "✅ APROBAR"
                        color = "success"
                    elif pd_prob < 0.20:
                        decision = "⚠️ REVISAR"
                        color = "warning"
                    else:
                        decision = "❌ RECHAZAR"
                        color = "error"

                    st.metric(label="Decisión Recomendada", value=decision)

                # Barra de riesgo
                st.progress(pd_prob, text=f"Nivel de riesgo: {pd_prob:.1%}")

            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")

# ─────────────────────────────────────────────
# TAB 2 — ANÁLISIS DEL PORTFOLIO
# ─────────────────────────────────────────────

with tab2:
    st.header("Análisis del Portfolio")

    try:
        df = load_reference_data()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Solicitudes", f"{len(df):,}")
        with col2:
            mora_rate = df["TARGET"].mean()
            st.metric("Tasa de Mora", f"{mora_rate:.1%}")
        with col3:
            st.metric("Features del Modelo", "65")
        with col4:
            st.metric("AUC del Modelo", "0.768")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribución del Target")
            target_counts = df["TARGET"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(
                target_counts,
                labels=["Sin mora", "Con mora"],
                autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c"]
            )
            st.pyplot(fig)

        with col2:
            st.subheader("Distribución de Ingresos")
            fig, ax = plt.subplots()
            ax.hist(
                df["AMT_INCOME_TOTAL"].clip(upper=500000),
                bins=50,
                color="#3498db",
                alpha=0.7
            )
            ax.set_xlabel("Ingreso anual ($)")
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)

    except Exception as e:
        st.warning(f"No se pudieron cargar los datos de referencia: {str(e)}")

# ─────────────────────────────────────────────
# TAB 3 — MÉTRICAS DEL MODELO
# ─────────────────────────────────────────────

with tab3:
    st.header("Métricas del Modelo en Producción")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Información del Modelo")
        st.table(pd.DataFrame({
            "Parámetro": [
                "Algoritmo",
                "Versión",
                "AUC ROC",
                "Dataset",
                "Features",
                "Tasa de mora",
                "Umbral de aprobación",
                "Entorno"
            ],
            "Valor": [
                "LightGBM",
                "lightgbm-credit-risk v1",
                "0.768",
                "Home Credit Default Risk",
                "65",
                "8.1%",
                "PD < 10%",
                "Azure ML"
            ]
        }))

    with col2:
        st.subheader("🎯 Criterios de Decisión")
        st.table(pd.DataFrame({
            "Segmento": ["Bajo riesgo", "Riesgo medio", "Alto riesgo"],
            "PD": ["< 10%", "10% - 20%", "> 20%"],
            "Decisión": ["✅ Aprobar", "⚠️ Revisar", "❌ Rechazar"],
            "Score": ["> 900 pts", "800-900 pts", "< 800 pts"]
        }))

        st.divider()
        st.subheader("📊 Pipeline MLOps")
        st.success("✅ Prep datos — Completado")
        st.success("✅ Entrenamiento — AUC 0.7675")
        st.success("✅ Evaluación — APROBADO (umbral 0.75)")
        st.success("✅ Registro — Azure ML Model Registry")