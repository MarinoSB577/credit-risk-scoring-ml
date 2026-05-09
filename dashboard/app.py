import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import anthropic
import os
import matplotlib.pyplot as plt
from pathlib import Path

from pathlib import Path
from dotenv import load_dotenv

# Cargar .env desde la raíz del proyecto
load_dotenv(Path(__file__).parent.parent / ".env")

# Limpiar variables SSL de Conda que interfieren con httpx
os.environ.pop('SSL_CERT_FILE', None)
os.environ.pop('SSL_CERT_DIR', None)


# ─────────────────────────────────────────────
# DICCIONARIO DE NOMBRES LEGIBLES PARA FEATURES
# Traduce nombres técnicos del dataset a términos
# comprensibles para analistas y clientes
# ─────────────────────────────────────────────

FEATURE_NAMES_ES = {
    "EXT_SOURCE_1": "Score externo Buró (fuente 1)",
    "EXT_SOURCE_2": "Score externo Buró (fuente 2)",
    "EXT_SOURCE_3": "Score externo Buró (fuente 3)",
    "AMT_GOODS_PRICE": "Precio del bien a financiar",
    "AMT_CREDIT": "Monto del crédito solicitado",
    "AMT_INCOME_TOTAL": "Ingreso anual del solicitante",
    "AMT_ANNUITY": "Pago mensual del crédito",
    "DAYS_BIRTH": "Edad del solicitante",
    "DAYS_EMPLOYED": "Antigüedad laboral",
    "DAYS_REGISTRATION": "Antigüedad de registro de identidad",
    "DAYS_ID_PUBLISH": "Antigüedad de documento de identidad",
    "PLAZO_MESES": "Plazo del crédito (meses)",
    "OWN_CAR_AGE": "Antigüedad del vehículo propio",
    "NAME_EDUCATION_TYPE": "Nivel de escolaridad",
    "NAME_INCOME_TYPE": "Tipo de ingreso",
    "NAME_FAMILY_STATUS": "Estado civil",
    "NAME_HOUSING_TYPE": "Tipo de vivienda",
    "OCCUPATION_TYPE": "Tipo de ocupación",
    "ORGANIZATION_TYPE": "Tipo de organización empleadora",
    "CNT_CHILDREN": "Número de hijos",
    "CNT_FAM_MEMBERS": "Número de miembros del hogar",
    "REGION_POPULATION_RELATIVE": "Densidad poblacional de la región",
    "REGION_RATING_CLIENT": "Calificación de riesgo de la región",
    "DTI_RATIO": "Ratio deuda / ingreso",
    "CREDIT_INCOME_RATIO": "Ratio crédito / ingreso",
    "FLAG_OWN_CAR": "Tiene vehículo propio",
    "FLAG_OWN_REALTY": "Tiene propiedad inmueble",
    "FLAG_WORK_PHONE": "Tiene teléfono de trabajo",
    "FLAG_PHONE": "Tiene teléfono fijo",
    "FLAG_EMAIL": "Tiene correo electrónico registrado",
}

def get_feature_name(feature):
    """Devuelve nombre legible de la feature o el nombre técnico si no está en el diccionario"""
    return FEATURE_NAMES_ES.get(feature, feature)

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
def load_model():
    model_path = Path(__file__).parent / "models" / "model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)

model = load_model()

# ─────────────────────────────────────────────
# CARGAR DATOS DE REFERENCIA
# ─────────────────────────────────────────────

@st.cache_data
def load_reference_data():
    df = pd.read_csv(Path(__file__).parent / "data" / "df_sample.csv")
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

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Scoring en Tiempo Real",
    "📊 Análisis del Portfolio",
    "🔍 Métricas del Modelo",
    "🤖 Explicación del Agente"
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

        # ─────────────────────────────────────────────
# TAB 4 — EXPLICACIÓN DEL AGENTE
# ─────────────────────────────────────────────

with tab4:
    st.header("🤖 Agente Explicador de Decisiones Crediticias")
    st.markdown("""
    El agente analiza la predicción del modelo usando SHAP y genera una 
    justificación en lenguaje natural citando normativa CNBV aplicable.
    """)

    # ─────────────────────────────────────────────
    # FUNCIÓN: CALCULAR SHAP
    # ─────────────────────────────────────────────

    @st.cache_data
    def get_shap_values(_model, _input_data):
        """
        Calcula valores SHAP para una predicción.
        El guión bajo en los parámetros evita que Streamlit
        intente hashear objetos no serializables (modelo, DataFrame)
        """
      
        explainer = shap.TreeExplainer(_model)
        # LightGBM es compatible directamente con TreeExplainer
        # sin necesidad de extraer el booster como XGBoost

        shap_values = explainer.shap_values(_input_data)
        return shap_values

    # ─────────────────────────────────────────────
    # FUNCIÓN: GENERAR EXPLICACIÓN CON CLAUDE
    # ─────────────────────────────────────────────

    def generar_explicacion_agente(pd_prob, decision, top_features, input_data):
        """
        Genera explicación en lenguaje natural usando Claude API.
        
        Args:
            pd_prob: probabilidad de incumplimiento (float)
            decision: APROBAR / REVISAR / RECHAZAR (str)
            top_features: lista de (feature, shap_value, valor_real)
            input_data: DataFrame con datos del solicitante
        """

        # Construir descripción de features para el prompt
        features_texto = "\n".join([
        f"  - {get_feature_name(feat)}: valor={val_real:.3f}, contribución SHAP={shap_val:+.4f} "
        f"({'aumenta' if shap_val > 0 else 'reduce'} el riesgo)"
        for feat, shap_val, val_real in top_features
        ])

        # ─────────────────────────────────────────────
        # PROMPT ENGINEERING
        # El prompt incluye:
        # 1. Rol del agente
        # 2. Datos de la solicitud
        # 3. Valores SHAP explicados
        # 4. Normativa CNBV relevante embebida
        # 5. Instrucciones de formato
        # ─────────────────────────────────────────────

        prompt = f"""Eres un analista experto en riesgo crediticio de una institución 
financiera mexicana regulada por la CNBV. Tu rol es explicar las decisiones 
del modelo de scoring de forma clara, profesional y regulatoriamente válida.

DATOS DE LA SOLICITUD:
- Probabilidad de Incumplimiento (PD): {pd_prob:.1%}
- Decisión del modelo: {decision}
- Ingreso anual: {int(input_data['AMT_INCOME_TOTAL'].values[0])} pesos
- Monto solicitado: {int(input_data['AMT_CREDIT'].values[0])} pesos
- Edad: {abs(int(input_data['DAYS_BIRTH'].values[0])) // 365} años
- Antigüedad laboral: {abs(int(input_data['DAYS_EMPLOYED'].values[0])) // 365} años

FACTORES DETERMINANTES (análisis SHAP):
{features_texto}

NORMATIVA CNBV APLICABLE:
- Circular Única de Bancos, Art. 92: las instituciones deben evaluar 
  la capacidad de pago del acreditado considerando ingresos, deudas 
  existentes y flujo de efectivo disponible.
- Disposiciones CNBV sobre transparencia: el cliente tiene derecho a 
  conocer las razones de rechazo de su solicitud en términos comprensibles.
- Criterios de calificación de cartera: la PD debe reflejar la probabilidad 
  real de incumplimiento basada en características del acreditado y del crédito.
- Basilea III: las instituciones deben mantener capital proporcional al 
  riesgo de sus carteras, lo que justifica umbrales de aprobación basados en PD.

INSTRUCCIONES:
1. Redacta una explicación profesional de máximo 200 palabras
2. Menciona los 2-3 factores principales que determinaron la decisión
3. Cita la normativa CNBV relevante de forma natural en el texto
4. Si la decisión es RECHAZAR, incluye una recomendación constructiva para el solicitante
5. Si la decisión es REVISAR, indica qué información adicional podría ayudar
6. Usa lenguaje claro que pueda entender tanto el analista como el cliente
7. NO uses markdown, cursivas, negritas, asteriscos ni ningún formato especial
8. NO uses paréntesis con números — escribe los números directamente en el texto
9. Redacta únicamente en párrafos continuos con texto plano"""

        # Llamar a Claude API
        os.environ.pop('SSL_CERT_FILE', None)
        os.environ.pop('SSL_CERT_DIR', None)

        # En Streamlit Cloud los secrets se leen con st.secrets
        # Localmente se leen desde el .env con os.environ
        api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-6",
            # claude-sonnet-4-6: balance óptimo entre calidad y velocidad
            # para explicaciones de texto de longitud media
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text

    # ─────────────────────────────────────────────
    # FORMULARIO DE EVALUACIÓN
    # ─────────────────────────────────────────────

    st.subheader("Datos del Solicitante")

    col1, col2, col3 = st.columns(3)

    with col1:
        amt_income_ag = st.number_input(
            "Ingreso anual ($)", min_value=10000, max_value=1000000,
            value=150000, step=10000, key="ag_income"
        )
        amt_credit_ag = st.number_input(
            "Monto del crédito ($)", min_value=10000, max_value=4000000,
            value=500000, step=10000, key="ag_credit"
        )

    with col2:
        days_birth_ag = st.slider(
            "Edad (años)", min_value=18, max_value=70, value=35, key="ag_age"
        )
        days_employed_ag = st.slider(
            "Antigüedad laboral (años)", min_value=0, max_value=40,
            value=5, key="ag_employed"
        )

    with col3:
        ext_source_2_ag = st.slider(
            "Score externo (Buró)", min_value=0.0, max_value=1.0,
            value=0.5, step=0.01, key="ag_ext2"
        )
        ext_source_3_ag = st.slider(
            "Score externo 2", min_value=0.0, max_value=1.0,
            value=0.5, step=0.01, key="ag_ext3"
        )

    st.divider()

    if st.button("🤖 Generar Análisis del Agente", type="primary"):
        if model is None:
            st.error("⚠️ Modelo no disponible.")
        else:
            try:
                # Preparar input con medianas como base
                df_ref = load_reference_data()
                feature_cols = [c for c in df_ref.columns
                               if c not in ["TARGET", "SK_ID_CURR"]]
                input_data = df_ref[feature_cols].median().to_frame().T

                # Sobreescribir con valores del formulario
                feature_map = {
                    "AMT_INCOME_TOTAL": amt_income_ag,
                    "AMT_CREDIT": amt_credit_ag,
                    "DAYS_BIRTH": -days_birth_ag * 365,
                    "DAYS_EMPLOYED": -days_employed_ag * 365,
                    "EXT_SOURCE_2": ext_source_2_ag,
                    "EXT_SOURCE_3": ext_source_3_ag,
                }
                for feat, val in feature_map.items():
                    if feat in input_data.columns:
                        input_data[feat] = val

                # ── Predicción ──
                pd_prob = model.predict_proba(input_data)[0][1]
                score = int((1 - pd_prob) * 1000)

                if pd_prob < 0.10:
                    decision = "APROBAR"
                elif pd_prob < 0.20:
                    decision = "REVISAR"
                else:
                    decision = "RECHAZAR"

                # ── Métricas principales ──
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("PD (Prob. Incumplimiento)", f"{pd_prob:.1%}")
                with col_r2:
                    st.metric("Score", f"{score} pts")
                with col_r3:
                    st.metric("Decisión", decision)

                st.divider()

                # ── Calcular SHAP ──
                with st.spinner("Calculando importancia de variables (SHAP)..."):
                    # Obtener modelo base de LightGBM desde el wrapper de joblib
                    lgbm_model = model
                    shap_vals = get_shap_values(lgbm_model, input_data)

                    # Para clasificación binaria, SHAP devuelve array
                    # Tomar valores para la clase positiva (incumplimiento)
                    if isinstance(shap_vals, list):
                        sv = shap_vals[1][0]
                    else:
                        sv = shap_vals[0]

                    # Top 5 features por valor absoluto de SHAP
                    feature_names = input_data.columns.tolist()
                    shap_df = sorted(
                        zip(feature_names, sv, input_data.values[0]),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5]

                # ── Gráfica SHAP ──
                st.subheader("📊 Variables que determinaron la decisión")
                fig, ax = plt.subplots(figsize=(8, 3))
                feats = [get_feature_name(x[0]) for x in shap_df]
                vals = [x[1] for x in shap_df]
                colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
                # Rojo = aumenta riesgo, Verde = reduce riesgo
                ax.barh(feats[::-1], vals[::-1], color=colors[::-1])
                ax.axvline(x=0, color="black", linewidth=0.8)
                ax.set_xlabel("Contribución SHAP (+ aumenta riesgo, - reduce riesgo)")
                ax.set_title("Importancia de variables en esta solicitud")
                plt.tight_layout()
                st.pyplot(fig)

                st.divider()

                # ── Generar explicación con Claude ──
                st.subheader("📋 Justificación del Agente")
                with st.spinner("Generando explicación regulatoria..."):
                    explicacion = generar_explicacion_agente(
                        pd_prob, decision, shap_df[:3], input_data
                    )

                # Mostrar con estilo según decisión
                if decision == "APROBAR":
                    st.success(explicacion)
                elif decision == "REVISAR":
                    st.warning(explicacion)
                else:
                    st.error(explicacion)

                # Pie de página regulatorio
                st.caption(
                    "⚖️ Esta explicación es generada por IA con base en el modelo "
                    "LightGBM y normativa CNBV vigente. No sustituye el criterio "
                    "del analista de crédito ni constituye resolución definitiva."
                )

            except Exception as e:
                st.error(f"Error en el análisis: {str(e)}")
                st.exception(e)