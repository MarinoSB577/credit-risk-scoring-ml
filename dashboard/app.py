import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import anthropic
import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env desde la raíz del proyecto
load_dotenv(Path(__file__).parent.parent / ".env")

# Limpiar variables SSL de Conda que interfieren con httpx
os.environ.pop('SSL_CERT_FILE', None)
os.environ.pop('SSL_CERT_DIR', None)

# ─────────────────────────────────────────────
# DICCIONARIO DE NOMBRES LEGIBLES PARA FEATURES
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
    "EXT_SOURCE_PROMEDIO": "Promedio de scores externos (Buró)",
    "EXT_SOURCE_MIN": "Score externo mínimo (Buró)",
    "EXT_SOURCE_MAX": "Score externo máximo (Buró)",
    "RIESGO_EDAD_SCORE": "Score de riesgo por edad",
    "REGION_RATING_CLIENT_W_CITY": "Calificación de riesgo región y ciudad",
    "DAYS_LAST_PHONE_CHANGE": "Días desde último cambio de teléfono",
    "DAYS_EMPLOYED_PERC": "Proporción antigüedad laboral / edad",
    "INCOME_CREDIT_PERC": "Proporción ingreso / crédito",
    "INCOME_PER_PERSON": "Ingreso por miembro del hogar",
    "ANNUITY_INCOME_PERC": "Proporción pago mensual / ingreso",
    "PAYMENT_RATE": "Tasa de pago mensual",
    "BURO_DAYS_CREDIT_MAX": "Máximo días de crédito en Buró",
    "BURO_DAYS_CREDIT_MEAN": "Promedio días de crédito en Buró",
    "BURO_AMT_CREDIT_SUM": "Suma total créditos en Buró",
    "PREV_AMT_ANNUITY_MEAN": "Promedio pago mensual créditos anteriores",
    "PREV_DAYS_DECISION_MEAN": "Promedio días desde decisiones anteriores",
}

def get_feature_name(feature):
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

@st.cache_resource
def load_scorecard():
    scorecard_path = Path(__file__).parent / "models" / "scorecard_woe.pkl"
    if not scorecard_path.exists():
        return None
    return joblib.load(scorecard_path)

scorecard = load_scorecard()

# ─────────────────────────────────────────────
# CARGAR DATOS DE REFERENCIA
# ─────────────────────────────────────────────

@st.cache_data
def load_reference_data():
    df = pd.read_csv(Path(__file__).parent / "data" / "df_sample.csv")
    return df

# ─────────────────────────────────────────────
# CARGAR ESTADÍSTICAS DEL PORTAFOLIO
# ─────────────────────────────────────────────

@st.cache_data
def load_portfolio_stats():
    stats_path = Path(__file__).parent / "models" / "portfolio_stats.json"
    if not stats_path.exists():
        return {}
    with open(stats_path, "r") as f:
        return json.load(f)

# ─────────────────────────────────────────────
# RAG ORIGINACIÓN — CAMBIO 2
# ─────────────────────────────────────────────

@st.cache_resource
def cargar_rag_originacion():
    """
    Carga ChromaDB y modelo de embeddings para RAG de originación.
    Mismo patrón que cargar_rag() en tab_cobranza.py.
    Si la colección no existe, retorna (None, None, None) y la función
    generar_explicacion_agente() usa el fallback hardcodeado original.
    """
    try:
        os.environ.pop('SSL_CERT_FILE', None)
        os.environ.pop('SSL_CERT_DIR', None)

        import chromadb
        from sentence_transformers import SentenceTransformer

        DASHBOARD_PATH   = Path(__file__).parent
        BASE_PATH        = DASHBOARD_PATH.parent
        MODELS_PATH      = DASHBOARD_PATH / 'models'
        COLLECTIONS_PATH = BASE_PATH / 'src' / 'collections'

        # Agregar src/collections al path
        sys.path.insert(0, str(COLLECTIONS_PATH))

        config_path = MODELS_PATH / 'chroma_config_originacion.pkl'
        if not config_path.exists():
            st.warning(
                "⚠️ chroma_config_originacion.pkl no encontrado. "
                "Ejecuta el notebook 12_indexacion_rag_originacion.ipynb primero."
            )
            return None, None, None

        config      = joblib.load(config_path)
        chroma_path = config['chroma_path']

        chroma_client   = chromadb.PersistentClient(path=chroma_path)
        modelo_emb      = SentenceTransformer(config['modelo_embeddings'])
        COLLECTION_NAME = config['collection_name']

        colecciones = [c.name for c in chroma_client.list_collections()]

        if COLLECTION_NAME not in colecciones or \
           chroma_client.get_collection(COLLECTION_NAME).count() == 0:

            st.info("⚙️ Inicializando base de conocimiento RAG de originación...")

            if COLLECTION_NAME in colecciones:
                chroma_client.delete_collection(COLLECTION_NAME)

            coleccion = chroma_client.create_collection(
                name     = COLLECTION_NAME,
                metadata = {"version": "1.0"}
            )

            # Cargar e indexar documentos automáticamente
            docs_path  = Path(config['docs_path'])
            sys.path.insert(0, str(docs_path.parent))

            from document_loader import load_documents_from_folder
            from sentence_transformers import SentenceTransformer

            chunks     = load_documents_from_folder(str(docs_path), verbose=False)
            modelo_emb = SentenceTransformer(config['modelo_embeddings'])

            BATCH_SIZE = 10
            for i in range(0, len(chunks), BATCH_SIZE):
                lote = chunks[i:i + BATCH_SIZE]
                coleccion.add(
                    ids        = [c['chunk_id'] for c in lote],
                    documents  = [c['text']     for c in lote],
                    embeddings = modelo_emb.encode(
                        [c['text'] for c in lote],
                        normalize_embeddings=True
                    ).tolist(),
                    metadatas  = [{'source': c['source'],
                                   'page'  : str(c['page'])} for c in lote],
                )

            st.success(
                f"✅ Base de conocimiento inicializada: "
                f"{len(chunks)} fragmentos indexados"
            )

        else:
            coleccion = chroma_client.get_collection(COLLECTION_NAME)


# ─────────────────────────────────────────────
# RAG ORIGINACIÓN — CAMBIO 3
# ─────────────────────────────────────────────

def consultar_rag_originacion(decision, pd_prob, top_features,
                               coleccion, modelo_emb,
                               n_resultados=3):
    """
    Búsqueda semántica en la base de conocimiento de originación.
    Mismo patrón que consultar_rag_dashboard() en tab_cobranza.py.

    Query: decisión + rango PD + variables SHAP principales.
    Documento prioritario: según la decisión (aprobar/revisar/rechazar).
    """
    # ── Construir query semántica ────────────────────────────
    variable_principal  = get_feature_name(top_features[0][0]) \
                          if top_features else "historial crediticio"
    variable_secundaria = get_feature_name(top_features[1][0]) \
                          if len(top_features) > 1 else ""

    rango_pd = (
        "bajo riesgo aprobación criterios otorgamiento"
        if pd_prob < 0.10 else
        "riesgo medio revisión análisis adicional capacidad pago"
        if pd_prob < 0.30 else
        "alto riesgo rechazo incumplimiento reservas cartera"
    )

    query = (
        f"decisión crédito {decision.lower()} "
        f"{rango_pd} "
        f"{variable_principal} {variable_secundaria} "
        f"normativa CNBV explicabilidad"
    )

    # ── Documento prioritario según decisión ────────────────
    prioridad_map = {
        "APROBAR" : "criterios_otorgamiento.txt",
        "REVISAR" : "criterios_otorgamiento.txt",
        "RECHAZAR": "calificacion_cartera_CUB.pdf",
    }
    archivo_prio = prioridad_map.get(decision, "criterios_otorgamiento.txt")

    embedding_q = modelo_emb.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    # ── Búsqueda prioritaria ─────────────────────────────────
    try:
        res_prio = coleccion.query(
            query_embeddings = embedding_q,
            n_results        = 1,
            where            = {"source": archivo_prio}
        )
    except Exception:
        res_prio = {"documents": [[]], "metadatas": [[]]}

    # ── Búsqueda complementaria ──────────────────────────────
    res_comp = coleccion.query(
        query_embeddings = embedding_q,
        n_results        = n_resultados,
        where            = {"source": {"$ne": archivo_prio}}
    )

    # ── Combinar y deduplicar ────────────────────────────────
    docs = []
    if res_prio['documents'][0]:
        docs.append({
            'source'   : res_prio['metadatas'][0][0]['source'],
            'contenido': res_prio['documents'][0][0],
            'tipo'     : 'prioritario'
        })

    archivos_vistos = {d['source'] for d in docs}
    for doc, meta in zip(
        res_comp['documents'][0],
        res_comp['metadatas'][0]
    ):
        if meta['source'] not in archivos_vistos:
            docs.append({
                'source'   : meta['source'],
                'contenido': doc,
                'tipo'     : 'complementario'
            })
            archivos_vistos.add(meta['source'])

    # ── Formatear contexto para el prompt ───────────────────
    partes = []
    for doc in docs:
        nombre = (doc['source']
                  .replace('.txt', '').replace('.pdf', '')
                  .replace('.docx', '').replace('.xlsx', '')
                  .replace('_', ' ').title())
        tipo   = "PRIORITARIO" if doc['tipo'] == 'prioritario' \
                 else "Complementario"
        partes.append(f"[{tipo} — {nombre}]\n{doc['contenido']}")

    return {
        'contexto_rag': "\n\n".join(partes),
        'docs_usados' : [d['source'] for d in docs]
    }


# ─────────────────────────────────────────────
# FUNCIONES SHAP Y AGENTE — CAMBIO 4
# ─────────────────────────────────────────────

@st.cache_data
def get_shap_values(_model, _input_data):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_input_data)
    return shap_values


def generar_explicacion_agente(pd_prob, decision, top_features,
                                input_data,
                                coleccion=None, modelo_emb=None):
    """
    Genera explicación regulatoria con valores reales del cliente,
    referencia del portafolio y contexto RAG de normativa CNBV.
    Si coleccion/modelo_emb son None, usa normativa hardcodeada (fallback).
    """
    # ── Cargar estadísticas del portafolio ───────────────────
    portfolio_stats = load_portfolio_stats()

    # ── Función para interpretar cada variable ───────────────
    def interpretar_valor(feat, val_real, shap_val):
        nombre    = get_feature_name(feat)
        direccion = "aumenta" if shap_val > 0 else "reduce"

        if feat == 'DAYS_BIRTH':
            val_interp  = f"{abs(int(val_real)) // 365} años"
            ref         = portfolio_stats.get(feat, {})
            mediana     = abs(ref.get('mediana', 0)) / 365
            comparacion = "por encima" if abs(val_real) / 365 > mediana else "por debajo"
            return (f"{nombre}: {val_interp} "
                    f"({comparacion} de la mediana del portafolio: {mediana:.0f} años) "
                    f"→ {direccion} el riesgo (SHAP: {shap_val:+.4f})")

        elif feat == 'DAYS_EMPLOYED':
            val_interp  = f"{abs(int(val_real)) // 365} años de antigüedad"
            ref         = portfolio_stats.get(feat, {})
            mediana     = abs(ref.get('mediana', 0)) / 365
            comparacion = "por encima" if abs(val_real) / 365 > mediana else "por debajo"
            return (f"{nombre}: {val_interp} "
                    f"({comparacion} de la mediana: {mediana:.0f} años) "
                    f"→ {direccion} el riesgo (SHAP: {shap_val:+.4f})")

        elif feat == 'AMT_CREDIT':
            ref         = portfolio_stats.get(feat, {})
            mediana     = ref.get('mediana', 0)
            comparacion = "por encima" if val_real > mediana else "por debajo"
            return (f"{nombre}: ${val_real:,.0f} MXN "
                    f"({comparacion} de la mediana del portafolio: ${mediana:,.0f} MXN) "
                    f"→ {direccion} el riesgo (SHAP: {shap_val:+.4f})")

        elif feat == 'AMT_INCOME_TOTAL':
            ref         = portfolio_stats.get(feat, {})
            mediana     = ref.get('mediana', 0)
            comparacion = "por encima" if val_real > mediana else "por debajo"
            return (f"{nombre}: ${val_real:,.0f} MXN anuales "
                    f"({comparacion} de la mediana: ${mediana:,.0f} MXN) "
                    f"→ {direccion} el riesgo (SHAP: {shap_val:+.4f})")

        elif feat == 'AMT_ANNUITY':
            ref         = portfolio_stats.get(feat, {})
            mediana     = ref.get('mediana', 0)
            comparacion = "por encima" if val_real > mediana else "por debajo"
            return (f"{nombre}: ${val_real:,.0f} MXN/mes "
                    f"({comparacion} de la mediana: ${mediana:,.0f} MXN/mes) "
                    f"→ {direccion} el riesgo (SHAP: {shap_val:+.4f})")

        else:
            ref = portfolio_stats.get(feat, {})
            if ref.get('tipo') == 'numerica':
                mediana = ref.get('mediana', None)
                if mediana is not None:
                    comparacion = "por encima" if val_real > mediana else "por debajo"
                    return (f"{nombre}: {val_real:.4f} "
                            f"({comparacion} de la mediana: {mediana:.4f}) "
                            f"→ {direccion} el riesgo (SHAP: {shap_val:+.4f})")
            return (f"{nombre}: {val_real:.4f} "
                    f"→ {direccion} el riesgo (SHAP: {shap_val:+.4f})")

    # ── Construir texto de variables enriquecido ─────────────
    features_texto = "\n".join([
        f"  {i+1}. {interpretar_valor(feat, val_real, shap_val)}"
        for i, (feat, shap_val, val_real) in enumerate(top_features)
    ])

    # ── Calcular carga financiera ─────────────────────────────
    try:
        ingreso_mensual = input_data['AMT_INCOME_TOTAL'].values[0] / 12
        cuota_mensual   = input_data['AMT_ANNUITY'].values[0]
        carga_fin       = cuota_mensual / ingreso_mensual
        carga_texto     = f"{carga_fin:.1%} del ingreso mensual"
    except Exception:
        carga_texto = "no calculada"

    # ── Consultar RAG o usar normativa fallback ───────────────
    if coleccion is not None and modelo_emb is not None:
        resultado_rag    = consultar_rag_originacion(
            decision, pd_prob, top_features, coleccion, modelo_emb
        )
        normativa_texto  = resultado_rag['contexto_rag']
        fuente_normativa = "base de conocimiento RAG (normativa CNBV indexada)"
    else:
        normativa_texto = (
            "Circular Única de Bancos, Art. 92: las instituciones deben evaluar\n"
            "la capacidad de pago del acreditado considerando ingresos, deudas\n"
            "existentes y flujo de efectivo disponible.\n"
            "Disposiciones CNBV sobre transparencia: el cliente tiene derecho a\n"
            "conocer las razones de rechazo de su solicitud en términos comprensibles.\n"
            "Criterios de calificación de cartera: la PD debe reflejar la probabilidad\n"
            "real de incumplimiento basada en características del acreditado y del crédito.\n"
            "Basilea III: las instituciones deben mantener capital proporcional al\n"
            "riesgo de sus carteras, lo que justifica umbrales de aprobación basados en PD."
        )
        fuente_normativa = "normativa CNBV de referencia"

    # ── Prompt enriquecido ────────────────────────────────────
    prompt = f"""Eres un analista experto en riesgo crediticio de una institución
financiera mexicana regulada por la CNBV. Tu rol es explicar las decisiones
del modelo de scoring de forma clara, profesional y regulatoriamente válida.

DATOS DE LA SOLICITUD:
- Probabilidad de Incumplimiento (PD): {pd_prob:.1%}
- Decisión del modelo: {decision}
- Ingreso anual: ${int(input_data['AMT_INCOME_TOTAL'].values[0]):,} MXN
- Monto solicitado: ${int(input_data['AMT_CREDIT'].values[0]):,} MXN
- Cuota mensual: ${int(input_data['AMT_ANNUITY'].values[0]):,} MXN
- Carga financiera: {carga_texto}
- Edad: {abs(int(input_data['DAYS_BIRTH'].values[0])) // 365} años
- Antigüedad laboral: {abs(int(input_data['DAYS_EMPLOYED'].values[0])) // 365} años

FACTORES DETERMINANTES (análisis SHAP con referencia al portafolio):
{features_texto}

NORMATIVA CNBV APLICABLE ({fuente_normativa}):
{normativa_texto}

INSTRUCCIONES:
1. Redacta una explicación profesional de máximo 200 palabras
2. Menciona los 2-3 factores principales usando los valores REALES del cliente
   y su comparación con el portafolio (por encima/por debajo de la mediana)
3. Incluye la carga financiera ({carga_texto}) en el análisis si es relevante
4. Cita la normativa CNBV relevante de forma natural en el texto
5. Si la decisión es RECHAZAR, incluye una recomendación constructiva
6. Si la decisión es REVISAR, indica qué información adicional podría ayudar
7. Usa lenguaje claro para analista y cliente
8. NO uses markdown, cursivas, negritas ni asteriscos
9. Redacta en párrafos continuos con texto plano"""

    os.environ.pop('SSL_CERT_FILE', None)
    os.environ.pop('SSL_CERT_DIR', None)
    api_key = st.secrets.get(
        "ANTHROPIC_API_KEY",
        os.environ.get("ANTHROPIC_API_KEY")
    )
    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model      = "claude-sonnet-4-20250514",
        max_tokens = 500,
        messages   = [{"role": "user", "content": prompt}]
    )
    return message.content[0].text


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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎯 Scoring en Tiempo Real",
    "📊 Análisis del Portfolio",
    "🔍 Métricas del Modelo",
    "🤖 Explicación del Agente",
    "📋 Scorecard WoE",
    "🚨 Agente de Cobranza",
    "🎯 Agente Ejecutivo",
    "📡 Métricas Vivas"
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
            "Ingreso anual ($)", min_value=10000, max_value=1000000,
            value=150000, step=10000
        )
        amt_credit = st.number_input(
            "Monto del crédito ($)", min_value=10000, max_value=4000000,
            value=500000, step=10000
        )
        amt_annuity = st.number_input(
            "Anualidad ($)", min_value=1000, max_value=200000,
            value=25000, step=1000
        )

    with col2:
        st.subheader("👤 Datos del Solicitante")
        days_birth = st.slider(
            "Edad (años)", min_value=18, max_value=70, value=35
        )
        days_employed = st.slider(
            "Antigüedad laboral (años)", min_value=0, max_value=40, value=5
        )
        ext_source_2 = st.slider(
            "Score externo (Buró)", min_value=0.0, max_value=1.0,
            value=0.5, step=0.01
        )

    with col3:
        st.subheader("📈 Historial Crediticio")
        ext_source_3 = st.slider(
            "Score externo 2", min_value=0.0, max_value=1.0,
            value=0.5, step=0.01
        )
        bureau_active = st.number_input(
            "Créditos activos en Buró", min_value=0, max_value=20, value=2
        )
        bureau_overdue = st.number_input(
            "Créditos vencidos en Buró", min_value=0, max_value=10, value=0
        )

    st.divider()

    if st.button("🔮 Calcular Score de Riesgo", type="primary"):
        if model is None:
            st.error("⚠️ Modelo no disponible. Verifica la ruta del archivo.")
        else:
            try:
                df_ref = load_reference_data()
                feature_cols = [c for c in df_ref.columns
                               if c not in ["TARGET", "SK_ID_CURR"]]
                input_data = df_ref[feature_cols].median().to_frame().T

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

                pd_prob = model.predict_proba(input_data)[0][1]
                score = int((1 - pd_prob) * 1000)

                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("Probabilidad de Incumplimiento", f"{pd_prob:.1%}")
                with col_res2:
                    st.metric("Score de Crédito", f"{score} pts")
                with col_res3:
                    if pd_prob < 0.10:
                        decision = "✅ APROBAR"
                    elif pd_prob < 0.20:
                        decision = "⚠️ REVISAR"
                    else:
                        decision = "❌ RECHAZAR"
                    st.metric("Decisión Recomendada", decision)

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
                bins=50, color="#3498db", alpha=0.7
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
                "Algoritmo", "Versión", "AUC ROC", "Dataset",
                "Features", "Tasa de mora", "Umbral de aprobación", "Entorno"
            ],
            "Valor": [
                "LightGBM", "lightgbm-credit-risk v1", "0.768",
                "Home Credit Default Risk", "65", "8.1%", "PD < 10%", "Azure ML"
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

    # ── CAMBIO 5: cargar RAG al inicio del tab ───────────────
    col_rag, modelo_emb_rag, config_rag = cargar_rag_originacion()

    # ─────────────────────────────────────────────
    # SELECTOR DE PERFIL BASE
    # ─────────────────────────────────────────────

    st.subheader("1️⃣ Perfil base del solicitante")
    df_ref_perfiles = load_reference_data()

    perfil_opciones = {
        "🟢 Bajo riesgo — cliente con buen historial":
            df_ref_perfiles[df_ref_perfiles['TARGET']==0].nsmallest(1, 'EXT_SOURCE_2'),
        "🟡 Riesgo medio — cliente promedio del dataset":
            df_ref_perfiles.sample(1, random_state=42),
        "🔴 Alto riesgo — cliente con historial negativo":
            df_ref_perfiles[df_ref_perfiles['TARGET']==1].nlargest(1, 'EXT_SOURCE_2'),
    }

    perfil_seleccionado = st.selectbox(
        "Selecciona un perfil base para la evaluación:",
        options=list(perfil_opciones.keys()),
        index=0
    )
    st.divider()

    # ─────────────────────────────────────────────
    # FORMULARIO DE AJUSTE
    # ─────────────────────────────────────────────

    st.subheader("2️⃣ Ajusta los datos del solicitante")

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

    # ─────────────────────────────────────────────
    # POLÍTICA CREDITICIA
    # ─────────────────────────────────────────────

    st.subheader("3️⃣ Política crediticia")
    umbral_aprobacion = st.slider(
        "Umbral de aprobación (PD máxima para aprobar)",
        min_value=0.05, max_value=0.40, value=0.10, step=0.01,
        format="%.0f%%",
        help="Conservador: 10% | Moderado: 20% | Agresivo: 30%",
        key="umbral_aprobacion_tab4"
    )

    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        st.metric("Umbral actual", f"{umbral_aprobacion:.0%}")
    with col_u2:
        if umbral_aprobacion <= 0.10:
            st.metric("Política", "🔵 Conservadora")
        elif umbral_aprobacion <= 0.20:
            st.metric("Política", "🟡 Moderada")
        else:
            st.metric("Política", "🔴 Agresiva")
    with col_u3:
        try:
            df_tmp = load_reference_data()
            feature_cols_tmp = [c for c in df_tmp.columns
                               if c not in ["TARGET", "SK_ID_CURR"]]
            preds_tmp = model.predict_proba(df_tmp[feature_cols_tmp])[:, 1]
            pct_aprueba = (preds_tmp < umbral_aprobacion).mean()
            st.metric("% cartera que aprobaría", f"{pct_aprueba:.1%}")
        except Exception:
            st.metric("% cartera que aprobaría", "N/A")

    st.divider()

    # ─────────────────────────────────────────────
    # BOTÓN Y RESULTADOS
    # ─────────────────────────────────────────────

    if st.button("🤖 Generar Análisis del Agente", type="primary"):
        if model is None:
            st.error("⚠️ Modelo no disponible.")
        else:
            try:
                perfil_base  = perfil_opciones[perfil_seleccionado]
                feature_cols = [c for c in perfil_base.columns
                               if c not in ["TARGET", "SK_ID_CURR"]]
                input_data   = perfil_base[feature_cols].copy()

                feature_map = {
                    "AMT_INCOME_TOTAL": amt_income_ag,
                    "AMT_CREDIT"      : amt_credit_ag,
                    "DAYS_BIRTH"      : -days_birth_ag * 365,
                    "DAYS_EMPLOYED"   : -days_employed_ag * 365,
                    "EXT_SOURCE_2"    : ext_source_2_ag,
                    "EXT_SOURCE_3"    : ext_source_3_ag,
                }
                for feat, val in feature_map.items():
                    if feat in input_data.columns:
                        input_data[feat] = val

                pd_prob = model.predict_proba(input_data)[0][1]
                score   = int((1 - pd_prob) * 1000)

                if pd_prob < umbral_aprobacion:
                    decision = "APROBAR"
                elif pd_prob < umbral_aprobacion * 2:
                    decision = "REVISAR"
                else:
                    decision = "RECHAZAR"

                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("PD (Prob. Incumplimiento)", f"{pd_prob:.1%}")
                with col_r2:
                    st.metric("Score", f"{score} pts")
                with col_r3:
                    st.metric("Decisión", decision)

                st.divider()

                with st.spinner("Calculando importancia de variables (SHAP)..."):
                    shap_vals = get_shap_values(model, input_data)
                    if isinstance(shap_vals, list):
                        sv = shap_vals[1][0]
                    else:
                        sv = shap_vals[0]

                    feature_names = input_data.columns.tolist()
                    shap_df = sorted(
                        zip(feature_names, sv, input_data.values[0]),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5]

                st.subheader("📊 Variables que determinaron la decisión")
                fig, ax = plt.subplots(figsize=(8, 3))
                feats  = [get_feature_name(x[0]) for x in shap_df]
                vals   = [x[1] for x in shap_df]
                colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
                ax.barh(feats[::-1], vals[::-1], color=colors[::-1])
                ax.axvline(x=0, color="black", linewidth=0.8)
                ax.set_xlabel("Contribución SHAP (+ aumenta riesgo, - reduce riesgo)")
                ax.set_title("Importancia de variables en esta solicitud")
                plt.tight_layout()
                st.pyplot(fig)

                st.divider()

                st.subheader("📋 Justificación del Agente")
                with st.spinner("Generando explicación regulatoria..."):
                    # ── CAMBIO 6: pasar col_rag y modelo_emb_rag ─
                    explicacion = generar_explicacion_agente(
                        pd_prob, decision, shap_df[:3], input_data,
                        coleccion  = col_rag,
                        modelo_emb = modelo_emb_rag
                    )

                if decision == "APROBAR":
                    st.success(explicacion)
                elif decision == "REVISAR":
                    st.warning(explicacion)
                else:
                    st.error(explicacion)

                st.caption(
                    "⚖️ Esta explicación es generada por IA con base en el modelo "
                    "LightGBM y normativa CNBV vigente. No sustituye el criterio "
                    "del analista de crédito ni constituye resolución definitiva."
                )

                # Guardar contexto para el chat conversacional
                shap_resumen = ", ".join([
                    f"{get_feature_name(f[0])}({f[1]:+.3f})"
                    for f in shap_df[:3]
                ])
                st.session_state.contexto_evaluacion = {
                    "perfil"    : perfil_seleccionado,
                    "pd_prob"   : f"{pd_prob:.1%}",
                    "score"     : score,
                    "decision"  : decision,
                    "umbral"    : f"{umbral_aprobacion:.0%}",
                    "ingreso"   : int(amt_income_ag),
                    "monto"     : int(amt_credit_ag),
                    "edad"      : days_birth_ag,
                    "antiguedad": days_employed_ag,
                    "shap_resumen": shap_resumen
                }

            except Exception as e:
                st.error(f"Error en el análisis: {str(e)}")
                st.exception(e)

    # ─────────────────────────────────────────────
    # CHAT CONVERSACIONAL CON EL AGENTE
    # ─────────────────────────────────────────────

    st.divider()
    st.subheader("💬 Conversa con el Agente")
    st.markdown("""
    Haz preguntas sobre la evaluación. El agente recuerda el contexto
    del análisis anterior y puede explorar escenarios hipotéticos.
    """)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "contexto_evaluacion" not in st.session_state:
        st.session_state.contexto_evaluacion = None

    if st.button("🗑️ Limpiar conversación", key="limpiar_chat"):
        st.session_state.chat_history = []
        st.session_state.contexto_evaluacion = None
        st.rerun()

    for mensaje in st.session_state.chat_history:
        with st.chat_message(mensaje["role"]):
            st.write(mensaje["content"])

    pregunta = st.chat_input(
        "Pregunta al agente: ¿qué pasaría si...? ¿por qué...? ¿cómo mejorar...?"
    )

    if pregunta:
        st.session_state.chat_history.append({
            "role": "user", "content": pregunta
        })

        with st.chat_message("user"):
            st.write(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("El agente está analizando..."):
                try:
                    contexto = ""
                    if st.session_state.contexto_evaluacion:
                        ctx = st.session_state.contexto_evaluacion
                        contexto = f"""
CONTEXTO DEL ANÁLISIS ACTUAL:
- Perfil base: {ctx.get('perfil', 'No disponible')}
- PD calculada: {ctx.get('pd_prob', 'N/D')}
- Score: {ctx.get('score', 'N/D')} pts
- Decisión: {ctx.get('decision', 'N/D')}
- Umbral de aprobación: {ctx.get('umbral', 'N/D')}
- Ingreso anual: {ctx.get('ingreso', 'N/D')} pesos
- Monto solicitado: {ctx.get('monto', 'N/D')} pesos
- Edad: {ctx.get('edad', 'N/D')} años
- Antigüedad laboral: {ctx.get('antiguedad', 'N/D')} años
- Top variables SHAP: {ctx.get('shap_resumen', 'N/D')}
"""
                    else:
                        contexto = "No hay análisis previo."

                    mensajes_api = []
                    for msg in st.session_state.chat_history[:-1]:
                        mensajes_api.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    mensajes_api.append({"role": "user", "content": pregunta})

                    system_prompt = f"""Eres un agente experto en riesgo crediticio de una institución
financiera mexicana regulada por la CNBV.

{contexto}

INSTRUCCIONES:
1. Responde de forma concisa y profesional (máximo 150 palabras)
2. Si te preguntan sobre escenarios hipotéticos, razona sobre
   cómo cambiarían las variables y el riesgo
3. Cita normativa CNBV cuando sea relevante
4. NO uses markdown, bullets ni formato especial
5. Responde en párrafos continuos con texto plano"""

                    os.environ.pop('SSL_CERT_FILE', None)
                    os.environ.pop('SSL_CERT_DIR', None)
                    api_key = st.secrets.get(
                        "ANTHROPIC_API_KEY",
                        os.environ.get("ANTHROPIC_API_KEY")
                    )
                    client = anthropic.Anthropic(api_key=api_key)

                    respuesta = client.messages.create(
                        model      = "claude-sonnet-4-20250514",
                        max_tokens = 400,
                        system     = system_prompt,
                        messages   = mensajes_api
                    )

                    texto_respuesta = respuesta.content[0].text
                    st.write(texto_respuesta)

                    st.session_state.chat_history.append({
                        "role": "assistant", "content": texto_respuesta
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ─────────────────────────────────────────────
# TAB 5 — SCORECARD WoE
# ─────────────────────────────────────────────

with tab5:
    st.header("📋 Scorecard de Crédito con WoE")
    st.markdown("""
    Modelo regulatorio basado en Weight of Evidence (WoE).
    Produce un score en puntos interpretable y auditable por la CNBV.
    """)

    if scorecard is None:
        st.error("⚠️ Scorecard no disponible. Verifica la ruta del archivo.")
    else:
        st.subheader("Datos del Solicitante")
        col1, col2, col3 = st.columns(3)

        with col1:
            sc_income = st.number_input(
                "Ingreso anual ($)", min_value=10000, max_value=1000000,
                value=150000, step=10000, key="sc_income"
            )
            sc_credit = st.number_input(
                "Monto del crédito ($)", min_value=10000, max_value=4000000,
                value=500000, step=10000, key="sc_credit"
            )
            sc_goods = st.number_input(
                "Precio del bien ($)", min_value=10000, max_value=4000000,
                value=450000, step=10000, key="sc_goods"
            )

        with col2:
            sc_age = st.slider(
                "Edad (años)", min_value=18, max_value=70,
                value=35, key="sc_age"
            )
            sc_employed = st.slider(
                "Antigüedad laboral (años)", min_value=0, max_value=40,
                value=5, key="sc_employed"
            )

        with col3:
            sc_ext2 = st.slider(
                "Score externo Buró (fuente 2)", min_value=0.0, max_value=1.0,
                value=0.5, step=0.01, key="sc_ext2"
            )
            sc_ext3 = st.slider(
                "Score externo Buró (fuente 3)", min_value=0.0, max_value=1.0,
                value=0.5, step=0.01, key="sc_ext3"
            )

        st.divider()

        if st.button("📋 Calcular Scorecard", type="primary", key="sc_button"):
            try:
                df_ref = load_reference_data()
                feature_cols = [c for c in df_ref.columns
                               if c not in ["TARGET", "SK_ID_CURR"]]
                input_sc = df_ref[feature_cols].median().to_frame().T

                input_sc["AMT_INCOME_TOTAL"]  = sc_income
                input_sc["AMT_CREDIT"]        = sc_credit
                input_sc["AMT_GOODS_PRICE"]   = sc_goods
                input_sc["DAYS_BIRTH"]        = -sc_age * 365
                input_sc["DAYS_EMPLOYED"]     = -sc_employed * 365
                input_sc["EXT_SOURCE_2"]      = sc_ext2
                input_sc["EXT_SOURCE_3"]      = sc_ext3

                input_sc["EXT_SOURCE_PROMEDIO"] = (
                    input_sc["EXT_SOURCE_2"] + input_sc["EXT_SOURCE_3"]
                ) / 2
                input_sc["RIESGO_EDAD_SCORE"] = (
                    1 - input_sc["EXT_SOURCE_PROMEDIO"]
                ) * (1 + abs(input_sc["DAYS_BIRTH"]) / (70 * 365))

                VARIABLES_SCORECARD = [
                    "EXT_SOURCE_PROMEDIO", "RIESGO_EDAD_SCORE",
                    "EXT_SOURCE_3", "EXT_SOURCE_2", "DAYS_EMPLOYED",
                    "AMT_GOODS_PRICE", "DAYS_BIRTH", "AMT_CREDIT"
                ]

                input_scorecard = input_sc[VARIABLES_SCORECARD]
                score_pts = scorecard.score(input_scorecard)[0]
                proba     = scorecard.predict_proba(input_scorecard)[0][1]

                if score_pts >= 570:
                    decision      = "APROBAR"
                    mora_esperada = "3.3%"
                elif score_pts >= 545:
                    decision      = "REVISAR"
                    mora_esperada = "8.1%"
                else:
                    decision      = "RECHAZAR"
                    mora_esperada = "19.0%"

                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                with col_r1:
                    st.metric("Score", f"{score_pts:.0f} pts")
                with col_r2:
                    st.metric("PD", f"{proba:.1%}")
                with col_r3:
                    st.metric("Decisión", decision)
                with col_r4:
                    st.metric("Mora esperada en segmento", mora_esperada)

                st.divider()
                st.subheader("📊 Posición en la escala de score")

                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh(["Score"], [627-489], left=489, color="#f0f0f0", height=0.4)
                ax.barh(["Score"], [545-489], left=489, color="#e74c3c",
                        height=0.4, alpha=0.6, label="RECHAZAR (<545)")
                ax.barh(["Score"], [570-545], left=545, color="#f39c12",
                        height=0.4, alpha=0.6, label="REVISAR (545-570)")
                ax.barh(["Score"], [627-570], left=570, color="#2ecc71",
                        height=0.4, alpha=0.6, label="APROBAR (≥570)")
                ax.axvline(x=score_pts, color="black",
                           linewidth=3, label=f"Tu score: {score_pts:.0f}")
                ax.set_xlim(489, 627)
                ax.set_xlabel("Score (puntos)")
                ax.set_title("Escala de score — mayor puntaje = menor riesgo")
                ax.legend(loc="upper left", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)

                st.divider()
                st.subheader("📋 Contribución por variable")
                st.markdown("Cuántos puntos aporta cada variable a tu score total:")

                tabla        = scorecard.table(style="detailed")
                contrib_data = []

                for var in VARIABLES_SCORECARD:
                    subtabla = tabla[tabla['Variable'] == var]
                    if len(subtabla) == 0:
                        continue
                    val = input_scorecard[var].values[0]
                    for _, row in subtabla.iterrows():
                        bin_str = str(row['Bin'])
                        if 'Special' in bin_str or 'Missing' in bin_str:
                            continue
                        try:
                            bin_str_clean = (bin_str.replace('(', '')
                                                    .replace(')', '')
                                                    .replace('[', '')
                                                    .replace(']', ''))
                            parts = bin_str_clean.split(',')
                            lb = float(parts[0].strip().replace('-inf', str(-np.inf)))
                            ub = float(parts[1].strip().replace('inf', str(np.inf)))
                            if lb <= val < ub:
                                puntos = row.get('Points', row.get('Score', 0))
                                contrib_data.append({
                                    "Variable": FEATURE_NAMES_ES.get(var, var),
                                    "Valor"   : f"{val:.3f}",
                                    "Rango"   : row['Bin'],
                                    "Puntos"  : f"{puntos:.1f}"
                                })
                                break
                        except Exception:
                            continue

                if contrib_data:
                    df_contrib = pd.DataFrame(contrib_data)
                    st.dataframe(df_contrib, use_container_width=True,
                                hide_index=True)

                st.caption(
                    "⚖️ Scorecard basado en Weight of Evidence (WoE). "
                    "Metodología auditable conforme a Circular Única de Bancos CNBV. "
                    "AUC=0.728 | KS=0.340 | Variables: 8"
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

# ─────────────────────────────────────────────
# TAB 6 — AGENTE DE COBRANZA
# ─────────────────────────────────────────────
from tab_cobranza import render_tab_cobranza

with tab6:
    render_tab_cobranza()

# ─────────────────────────────────────────────
# TAB 7 — AGENTE EJECUTIVO DE RIESGO
# ─────────────────────────────────────────────
from tab_agente_ejecutivo import render_tab_agente_ejecutivo

with tab7:
    render_tab_agente_ejecutivo()

# ─────────────────────────────────────────────
# TAB 8 — MÉTRICAS VIVAS
# ─────────────────────────────────────────────
from tab_metricas_vivas import render_tab_metricas_vivas

with tab8:
    render_tab_metricas_vivas()
