# ============================================================
# TAB 7 — AGENTE EJECUTIVO DE RIESGO
# ============================================================
# Propósito  : Interfaz del Agente Ejecutivo para el dashboard.
#              Orquesta los agentes de Originación y Cobranza
#              mediante LangGraph para generar reportes
#              ejecutivos integrados.
#
# Audiencia  : Director de Riesgo / Dirección General
# ============================================================

import streamlit as st
import anthropic
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional
import operator

# ── Rutas ────────────────────────────────────────────────────
DASHBOARD_PATH = Path(__file__).parent
BASE_PATH      = DASHBOARD_PATH.parent
MODELS_PATH    = DASHBOARD_PATH / 'models'

# Fix SSL para entorno conda en Windows
os.environ.pop('SSL_CERT_FILE', None)
os.environ.pop('SSL_CERT_DIR', None)


def obtener_api_key():
    """Lee la API key desde secrets o .env"""
    try:
        return st.secrets['ANTHROPIC_API_KEY']
    except Exception:
        pass
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if api_key:
        return api_key
    env_path = BASE_PATH / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for linea in f:
                if linea.startswith('ANTHROPIC_API_KEY'):
                    return linea.split('=')[1].strip() \
                                .strip('"').strip("'")
    return None


# ── Funciones de datos ───────────────────────────────────────

@st.cache_data
def obtener_metricas_originacion():
    """Métricas del Agente de Originación."""
    try:
        import pandas as pd
        import numpy as np
        df = pd.read_csv(
            BASE_PATH / 'data' / 'processed' / 'df_collections.csv',
            usecols=['SK_ID_CURR', 'TARGET', 'AMT_CREDIT',
                     'AMT_ANNUITY', 'AMT_INCOME_TOTAL',
                     'NAME_CONTRACT_TYPE']
        )
        carga = (df['AMT_ANNUITY'] /
                 (df['AMT_INCOME_TOTAL'] / 12)).mean()
        return {
            'n_solicitudes' : len(df),
            'n_mora'        : int(df['TARGET'].sum()),
            'tasa_mora'     : df['TARGET'].mean(),
            'monto_promedio': df['AMT_CREDIT'].mean(),
            'cartera_total' : df['AMT_CREDIT'].sum(),
            'carga_fin_prom': carga,
            'auc'           : 0.768,
        }
    except Exception as e:
        return {'error': str(e)}


@st.cache_data
def obtener_metricas_cobranza():
    """Métricas del Agente de Cobranza (EWS)."""
    try:
        import pandas as pd
        df = pd.read_csv(
            BASE_PATH / 'data' / 'processed' / 'df_ews.csv',
            usecols=['SK_ID_CURR', 'TARGET_EWS',
                     'max_dias_atraso', 'carga_financiera',
                     'es_cliente_recurrente']
        )
        df_alerta = df[df['TARGET_EWS'] == 1]
        return {
            'n_activos'      : len(df),
            'n_alerta'       : int(df['TARGET_EWS'].sum()),
            'tasa_alerta'    : df['TARGET_EWS'].mean(),
            'pct_recurrentes': df_alerta[
                'es_cliente_recurrente'].mean(),
            'carga_alerta'   : df_alerta[
                'carga_financiera'].mean(),
            'auc'            : 0.8712,
            'recall'         : 0.796,
        }
    except Exception as e:
        return {'error': str(e)}


def clasificar_pregunta_claude(pregunta: str,
                                api_key: str) -> str:
    """Usa Claude para clasificar la intención."""
    try:
        cliente = anthropic.Anthropic(api_key=api_key)
        resp = cliente.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 10,
            messages   = [{
                "role"   : "user",
                "content": f"""Clasifica esta pregunta en UNA palabra.
Opciones: originacion, cobranza, ambos

- originacion: solicitudes, score, PD, nuevos créditos
- cobranza: mora, atrasos, EWS, alertas, recuperación
- ambos: reporte general, cartera completa, situación global

Pregunta: {pregunta}

Responde SOLO con una palabra."""
            }]
        )
        tipo = resp.content[0].text.strip().lower()
        if tipo not in {'originacion', 'cobranza', 'ambos'}:
            return 'ambos'
        return tipo
    except Exception:
        return 'ambos'


def generar_reporte_ejecutivo(pregunta: str,
                               tipo: str,
                               orig: dict,
                               cob: dict,
                               api_key: str) -> str:
    """Genera el reporte ejecutivo con Claude."""
    contexto = ""

    if orig and 'error' not in orig and tipo in ('originacion', 'ambos'):
        contexto += f"""
DATOS DE ORIGINACIÓN:
  Solicitudes totales    : {orig.get('n_solicitudes', 0):,}
  Clientes en mora       : {orig.get('n_mora', 0):,}
  Tasa de mora           : {orig.get('tasa_mora', 0):.1%}
  Monto promedio crédito : ${orig.get('monto_promedio', 0):,.0f} MXN
  Cartera total          : ${orig.get('cartera_total', 0):,.0f} MXN
  Carga financiera prom. : {orig.get('carga_fin_prom', 0):.1%}
  AUC modelo             : {orig.get('auc', 0)}
        """.strip()

    if cob and 'error' not in cob and tipo in ('cobranza', 'ambos'):
        contexto += f"""

DATOS DE COBRANZA (EWS):
  Clientes activos       : {cob.get('n_activos', 0):,}
  Alertas EWS activas    : {cob.get('n_alerta', 0):,}
  Tasa de alerta         : {cob.get('tasa_alerta', 0):.1%}
  % recurrentes en alerta: {cob.get('pct_recurrentes', 0):.1%}
  Carga fin. en alerta   : {cob.get('carga_alerta', 0):.1%}
  AUC modelo EWS         : {cob.get('auc', 0)}
  Recall EWS             : {cob.get('recall', 0):.1%}
        """.strip()

    try:
        cliente = anthropic.Anthropic(api_key=api_key)
        resp = cliente.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 600,
            system     = """Eres el Agente Ejecutivo de Riesgo Crediticio.
Generas reportes ejecutivos concisos para la Dirección General
de una institución financiera mexicana.
- Usa solo los datos proporcionados
- Máximo 300 palabras
- Estructura: SITUACIÓN ACTUAL / ALERTAS / RECOMENDACIONES
- Lenguaje ejecutivo, no técnico
- Cumple con terminología regulatoria CNBV""",
            messages   = [{
                "role"   : "user",
                "content": f"""PREGUNTA: {pregunta}

DATOS:
{'='*50}
{contexto}
{'='*50}

Genera el reporte ejecutivo."""
            }]
        )
        return resp.content[0].text
    except Exception as e:
        return f"Error al generar reporte: {e}"


# ============================================================
# FUNCIÓN PRINCIPAL DE LA TAB
# ============================================================

def render_tab_agente_ejecutivo():
    """Renderiza la Tab 7 — Agente Ejecutivo de Riesgo."""

    st.header("🎯 Agente Ejecutivo de Riesgo")
    st.markdown(
        "Orquesta el **Agente de Originación** y el "
        "**Agente de Cobranza** para generar reportes "
        "ejecutivos integrados. Clasificación de intenciones "
        "con **Claude API + LangGraph**."
    )

    api_key = obtener_api_key()
    if not api_key:
        st.error("❌ ANTHROPIC_API_KEY no configurada.")
        st.stop()

    # ── Métricas de los sub-agentes ──────────────────────────
    st.subheader("📊 Estado de los Sub-Agentes")

    orig = obtener_metricas_originacion()
    cob  = obtener_metricas_cobranza()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🏦 Tasa de mora",
            f"{orig.get('tasa_mora', 0):.1%}",
            help="Agente de Originación"
        )
    with col2:
        st.metric(
            "💰 Cartera total",
            f"${orig.get('cartera_total', 0)/1e9:.1f}B MXN",
            help="Cartera total originada"
        )
    with col3:
        st.metric(
            "🚨 Alertas EWS",
            f"{cob.get('n_alerta', 0):,}",
            help="Clientes en riesgo detectados por EWS"
        )
    with col4:
        st.metric(
            "⚡ Tasa de alerta",
            f"{cob.get('tasa_alerta', 0):.1%}",
            help="% de cartera activa con alerta EWS"
        )

    st.divider()

    # ── Arquitectura del sistema ─────────────────────────────
    with st.expander("🔧 Arquitectura del Agente Ejecutivo"):
        # De:
        st.markdown("""
PREGUNTA DEL ANALISTA
│
...
""")

# A:
        st.code("""
PREGUNTA DEL ANALISTA
        │
        ▼
[Claude API — Clasificador de intenciones]
        │
   ─────┼─────────────────────
   │         │              │
   ▼         ▼              ▼
orig.    cobranza        ambos
   │         │              │
   ▼         ▼              ▼
Agente   Agente EWS   Ambos agentes
Orig.    + RAG        en secuencia
   │         │              │
   └─────────┴──────────────┘
                │
                ▼
   [Claude API — Reporte Ejecutivo]
""", language=None)
        st.caption(
            "Implementado con LangGraph. "
            "Clasificación con Claude API (Opción B)."
        )

    st.divider()

    # ── Chat con el Agente Ejecutivo ─────────────────────────
    st.subheader("💬 Consulta al Agente Ejecutivo")

    # Preguntas de ejemplo
    st.markdown("**Preguntas sugeridas:**")
    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        if st.button("📋 Reporte semanal completo",
                     use_container_width=True):
            st.session_state['pregunta_ejecutivo'] = (
                "Dame el reporte semanal de riesgo crediticio — "
                "necesito saber cómo está la cartera completa."
            )
    with col_e2:
        if st.button("🚨 Estado de cobranza",
                     use_container_width=True):
            st.session_state['pregunta_ejecutivo'] = (
                "¿Cuántos clientes están en alerta EWS "
                "y cuál es la estrategia de cobranza recomendada?"
            )
    with col_e3:
        if st.button("🏦 Calidad de originación",
                     use_container_width=True):
            st.session_state['pregunta_ejecutivo'] = (
                "¿Cómo está la calidad de la originación "
                "y cuál es la tasa de mora actual?"
            )

    # Inicializar historial
    if 'historial_ejecutivo' not in st.session_state:
        st.session_state['historial_ejecutivo'] = []

    # Input de pregunta
    pregunta_default = st.session_state.get(
        'pregunta_ejecutivo', ''
    )

    pregunta = st.chat_input(
        "Haz una pregunta al Agente Ejecutivo de Riesgo...",
        key="chat_ejecutivo"
    )

    # Si se presionó un botón de ejemplo
    if 'pregunta_ejecutivo' in st.session_state and \
       st.session_state['pregunta_ejecutivo']:
        pregunta = st.session_state.pop('pregunta_ejecutivo')

    # Procesar pregunta
    if pregunta:
        # Agregar al historial
        st.session_state['historial_ejecutivo'].append({
            'role'   : 'user',
            'content': pregunta
        })

        with st.spinner(
            "🔄 Clasificando pregunta → consultando "
            "agentes → generando reporte..."
        ):
            # Paso 1: Clasificar con Claude
            tipo = clasificar_pregunta_claude(
                pregunta, api_key
            )

            # Paso 2: Generar reporte
            reporte = generar_reporte_ejecutivo(
                pregunta, tipo, orig, cob, api_key
            )

        # Agregar reporte al historial
        st.session_state['historial_ejecutivo'].append({
            'role'   : 'assistant',
            'content': reporte,
            'tipo'   : tipo
        })

        st.rerun()

    # Mostrar historial
    if st.session_state['historial_ejecutivo']:
        for msg in st.session_state['historial_ejecutivo']:
            with st.chat_message(msg['role']):
                if msg['role'] == 'assistant' and \
                   'tipo' in msg:
                    st.caption(
                        f"🔀 Enrutado a: **{msg['tipo']}** "
                        f"| Clasificado por Claude API"
                    )
                st.markdown(msg['content'])

        # Botón para nueva consulta
        if st.button("🔄 Limpiar conversación"):
            st.session_state['historial_ejecutivo'] = []
            st.rerun()
    else:
        st.info(
            "👆 Escribe una pregunta o usa los botones "
            "de ejemplo para consultar al Agente Ejecutivo."
        )