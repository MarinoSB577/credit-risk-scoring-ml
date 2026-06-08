# ============================================================
# TAB 6 — AGENTE DE COBRANZA
# ============================================================
# Propósito  : Interfaz del Agente de Cobranza para el
#              dashboard de Credit Risk Scoring.
#
# Componentes:
#   1. Lista de clientes en riesgo detectados por EWS
#   2. Detalle del cliente seleccionado con SHAP
#   3. Generación de estrategia RAG + Claude API
#   4. Chat conversacional con el agente
#
# Dependencias:
#   - dashboard/models/modelo_ews.pkl
#   - dashboard/models/shap_ews_explainer.pkl
#   - dashboard/models/ews_metadata.pkl
#   - dashboard/models/chroma_config.pkl
#   - src/collections/documentos/ (base de conocimiento)
#   - src/collections/chroma_db/ (vector store)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import anthropic
import os
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


# ── Agente de Cobranza Level 3 — importación opcional ───────────────────────
_LEVEL3_OK = False
try:
    import sys as _sys
    from pathlib import Path as _Path
    _crm_path = str(_Path(__file__).parent.parent / "src")
    if _crm_path not in _sys.path:
        _sys.path.insert(0, _crm_path)
    from crm import (
        inicializar_crm  as _inicializar_crm,
        ejecutar_agente  as _ejecutar_agente,
        ejecutar_lote_ews as _ejecutar_lote_ews,
        get_db_path      as _get_db_path,
    )
    _LEVEL3_OK = True
except ImportError:
    pass

# ── Rutas del proyecto ───────────────────────────────────────
DASHBOARD_PATH   = Path(__file__).parent
BASE_PATH        = DASHBOARD_PATH.parent
MODELS_PATH      = DASHBOARD_PATH / 'models'
COLLECTIONS_PATH = BASE_PATH / 'src' / 'collections'

# Agregar src/ al path para importar módulos del proyecto
sys.path.insert(0, str(BASE_PATH / 'src'))

# ── Configuración del EWS ────────────────────────────────────
UMBRAL_EWS       = 0.20   # Umbral de decisión del modelo
UMBRAL_MORA_DIAS = 5      # Días de atraso para definir mora


# ── Funciones de carga con caché ─────────────────────────────

@st.cache_resource
def cargar_modelo_ews():
    """Carga el modelo LightGBM del EWS."""
    ruta = MODELS_PATH / 'modelo_ews.pkl'
    if not ruta.exists():
        return None
    return joblib.load(ruta)


@st.cache_resource
def cargar_shap_ews():
    """Carga el SHAP explainer del EWS."""
    ruta = MODELS_PATH / 'shap_ews_explainer.pkl'
    if not ruta.exists():
        return None
    return joblib.load(ruta)


@st.cache_resource
def cargar_metadata_ews():
    """Carga la metadata del modelo EWS."""
    ruta = MODELS_PATH / 'ews_metadata.pkl'
    if not ruta.exists():
        return None
    return joblib.load(ruta)


@st.cache_resource
def cargar_rag():
    """
    Carga ChromaDB y el modelo de embeddings para RAG.
    Si ChromaDB no tiene datos indexados, re-indexa
    automáticamente desde los documentos en disco.
    Se cachea para no recargar en cada interacción.
    """
    try:
        # Fix SSL para entorno conda en Windows
        os.environ.pop('SSL_CERT_FILE', None)
        os.environ.pop('SSL_CERT_DIR', None)

        import chromadb
        from sentence_transformers import SentenceTransformer
        from docx import Document as DocxDocument
        import openpyxl

        # Cargar configuración
        config_path = MODELS_PATH / 'chroma_config.pkl'
        if not config_path.exists():
            st.error("chroma_config.pkl no encontrado")
            return None, None, None

        config   = joblib.load(config_path)
        chroma_path = config['chroma_path']

        # Inicializar ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=chroma_path
        )

        COLLECTION_NAME = config['collection_name']

        # Cargar modelo de embeddings
        modelo_emb = SentenceTransformer(
            config['modelo_embeddings']
        )

        # Verificar si la colección existe y tiene datos
        colecciones = [
            c.name for c in chroma_client.list_collections()
        ]
        necesita_indexar = (
            COLLECTION_NAME not in colecciones or
            chroma_client.get_collection(
                COLLECTION_NAME
            ).count() == 0
        )

        if necesita_indexar:
            # Re-indexar documentos automáticamente
            st.info("⚙️ Inicializando base de conocimiento RAG...")

            if COLLECTION_NAME in colecciones:
                chroma_client.delete_collection(COLLECTION_NAME)

            coleccion = chroma_client.create_collection(
                name     = COLLECTION_NAME,
                metadata = {"version": "1.0"}
            )

            # Cargar y procesar documentos
            docs_path = COLLECTIONS_PATH / 'documentos'
            if not docs_path.exists():
                st.error(
                    f"Carpeta de documentos no encontrada: "
                    f"{docs_path}"
                )
                return None, None, None

            chunks_total = []
            EXTENSIONES  = {'.docx', '.xlsx', '.txt'}

            for archivo in sorted(os.listdir(docs_path)):
                ext  = Path(archivo).suffix.lower()
                if ext not in EXTENSIONES:
                    continue

                ruta_archivo = docs_path / archivo

                try:
                    if ext == '.docx':
                        doc   = DocxDocument(str(ruta_archivo))
                        texto = '\n'.join([
                            p.text.strip()
                            for p in doc.paragraphs
                            if p.text.strip()
                        ])
                    elif ext == '.xlsx':
                        wb    = openpyxl.load_workbook(
                            str(ruta_archivo)
                        )
                        lineas = []
                        for hoja in wb.sheetnames:
                            ws      = wb[hoja]
                            headers = []
                            for i, fila in enumerate(
                                ws.iter_rows(values_only=True)
                            ):
                                if i == 0:
                                    headers = [
                                        str(h) for h in fila if h
                                    ]
                                    continue
                                if any(c for c in fila):
                                    partes = [
                                        f"{headers[j]}: {v}"
                                        for j, v in enumerate(fila)
                                        if v and j < len(headers)
                                    ]
                                    if partes:
                                        lineas.append(
                                            ' | '.join(partes)
                                        )
                        texto = '\n'.join(lineas)
                    elif ext == '.txt':
                        with open(
                            ruta_archivo, 'r',
                            encoding='utf-8', errors='ignore'
                        ) as f:
                            texto = f.read()

                    # Dividir en chunks
                    CHUNK_SIZE    = 800
                    CHUNK_OVERLAP = 100
                    inicio = 0

                    while inicio < len(texto):
                        fin = inicio + CHUNK_SIZE
                        if fin < len(texto):
                            ultimo = max(
                                texto.rfind('.', inicio, fin),
                                texto.rfind('\n', inicio, fin)
                            )
                            if ultimo > inicio:
                                fin = ultimo + 1

                        chunk = texto[inicio:fin].strip()
                        if chunk:
                            chunks_total.append({
                                'id'      : f"{archivo}__chunk_{len(chunks_total):03d}",
                                'texto'   : chunk,
                                'archivo' : archivo,
                            })
                        inicio = fin - CHUNK_OVERLAP

                except Exception as e:
                    st.warning(f"Error procesando {archivo}: {e}")

            # Indexar en ChromaDB
            BATCH_SIZE = 10
            for i in range(0, len(chunks_total), BATCH_SIZE):
                lote       = chunks_total[i:i+BATCH_SIZE]
                textos     = [c['texto']   for c in lote]
                ids        = [c['id']      for c in lote]
                metadatas  = [
                    {'archivo': c['archivo']} for c in lote
                ]
                embeddings = modelo_emb.encode(
                    textos, show_progress_bar=False
                ).tolist()

                coleccion.add(
                    ids        = ids,
                    documents  = textos,
                    embeddings = embeddings,
                    metadatas  = metadatas
                )

            st.success(
                f"✅ Base de conocimiento inicializada: "
                f"{len(chunks_total)} fragmentos indexados"
            )

        else:
            coleccion = chroma_client.get_collection(
                COLLECTION_NAME
            )

        return coleccion, modelo_emb, config

    except Exception as e:
        st.error(f"Error cargando RAG: {e}")
        return None, None, None



def obtener_api_key():
    """Lee la API key de Anthropic desde secrets o .env"""
    # Streamlit Cloud secrets
    try:
        return st.secrets['ANTHROPIC_API_KEY']
    except Exception:
        pass

    # Variable de entorno
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if api_key:
        return api_key

    # Archivo .env
    env_path = BASE_PATH / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for linea in f:
                if linea.startswith('ANTHROPIC_API_KEY'):
                    return linea.split('=')[1].strip() \
                                .strip('"').strip("'")

    return None


@st.cache_data
def generar_clientes_riesgo(n_clientes: int = 200):
    """
    Genera una muestra simulada de clientes en riesgo
    para demostración del dashboard.

    En producción esto vendría de la base de datos
    de clientes activos evaluados por el EWS.
    """
    np.random.seed(42)

    clientes = []
    for i in range(n_clientes):
        # Probabilidad de mora distribuida entre 20% y 95%
        prob = np.random.beta(2, 3) * 0.8 + 0.20

        # Días de atraso correlacionados con la probabilidad
        dias_base = int(prob * 100)
        dias = max(0, int(np.random.normal(dias_base, 15)))

        # Perfil del cliente
        carga = np.random.uniform(0.25, 0.85)
        rec   = np.random.choice([0, 1], p=[0.3, 0.7])
        cuotas = np.random.randint(3, 36)
        dpd    = max(0, int(np.random.normal(dias * 0.8, 5)))

        clientes.append({
            'cliente_id'           : f"CLI-{100000 + i:06d}",
            'probabilidad_mora'    : round(prob, 3),
            'max_dias_atraso'      : dias,
            'max_dpd'              : dpd,
            'carga_financiera'     : round(carga, 2),
            'es_cliente_recurrente': rec,
            'n_cuotas_total'       : cuotas,
            'monto_credito'        : np.random.randint(
                50000, 800000
            ),
        })

    df = pd.DataFrame(clientes)

    # Solo clientes con alerta EWS activa
    df = df[df['probabilidad_mora'] >= UMBRAL_EWS].copy()
    df = df.sort_values(
        'probabilidad_mora', ascending=False
    ).reset_index(drop=True)

    return df


def clasificar_nivel_mora(dias: int) -> tuple:
    """Retorna (nivel, color, emoji) según días de atraso."""
    if dias == 0:
        return "Preventiva", "#FFA500", "🟡"
    elif dias <= 7:
        return "Alerta temprana", "#FF8C00", "🟠"
    elif dias <= 30:
        return "Mora leve", "#FF4500", "🔴"
    elif dias <= 90:
        return "Mora media", "#DC143C", "🔴"
    else:
        return "Mora grave", "#8B0000", "🆘"


def consultar_rag_dashboard(cliente_info: dict,
                             coleccion,
                             modelo_emb,
                             n_resultados: int = 3) -> dict:
    """
    Versión del RAG adaptada para el dashboard.
    Igual que en el notebook pero sin dependencias
    de variables globales del notebook.
    """
    dias = cliente_info.get('max_dias_atraso', 0)

    # Construir query semántica
    if dias == 0:
        nivel = "preventiva sin atraso alerta temprana"
    elif dias <= 7:
        nivel = "preventiva 1 a 7 días atraso"
    elif dias <= 30:
        nivel = "temprana 8 a 30 días atraso"
    elif dias <= 90:
        nivel = "media 31 a 90 días reestructura"
    else:
        nivel = "tardía más de 90 días quita"

    rec   = cliente_info.get('es_cliente_recurrente', 0)
    carga = cliente_info.get('carga_financiera', 0)

    perfil = "cliente recurrente" if rec \
             else "cliente nuevo"
    if carga > 0.5:
        perfil += " carga financiera alta"

    query = (
        f"estrategia cobranza {nivel} {perfil} "
        f"acciones protocolo contacto"
    )

    # Determinar documento prioritario
    if dias <= 7:
        archivo_prio = "01_estrategia_preventiva.docx"
    elif dias <= 30:
        archivo_prio = "02_estrategia_temprana.docx"
    elif dias <= 90:
        archivo_prio = "03_estrategia_media.docx"
    else:
        archivo_prio = "04_estrategia_tardia.docx"

    embedding_q = modelo_emb.encode([query]).tolist()

    # Búsqueda prioritaria
    res_prio = coleccion.query(
        query_embeddings = embedding_q,
        n_results        = 1,
        where            = {"archivo": archivo_prio}
    )

    # Búsqueda complementaria
    res_comp = coleccion.query(
        query_embeddings = embedding_q,
        n_results        = n_resultados,
        where            = {"archivo": {"$ne": archivo_prio}}
    )

    # Combinar y formatear
    docs = []
    if res_prio['documents'][0]:
        docs.append({
            'archivo'  : res_prio['metadatas'][0][0]['archivo'],
            'contenido': res_prio['documents'][0][0],
            'tipo'     : 'prioritario'
        })

    # Deduplicar archivos complementarios
    archivos_vistos = {d['archivo'] for d in docs}
    for doc, meta in zip(
        res_comp['documents'][0],
        res_comp['metadatas'][0]
    ):
        if meta['archivo'] not in archivos_vistos:
            docs.append({
                'archivo'  : meta['archivo'],
                'contenido': doc,
                'tipo'     : 'complementario'
            })
            archivos_vistos.add(meta['archivo'])

    # Formatear contexto
    partes = []
    for i, doc in enumerate(docs, 1):
        nombre = doc['archivo'].replace('.docx', '') \
                               .replace('.xlsx', '') \
                               .replace('_', ' ').title()
        tipo   = "📌 PRIORITARIO" \
                 if doc['tipo'] == 'prioritario' \
                 else "📄 Complementario"
        partes.append(
            f"[{tipo} — {nombre}]\n{doc['contenido']}"
        )

    return {
        'contexto_rag': "\n\n".join(partes),
        'docs_usados' : [d['archivo'] for d in docs]
    }


def generar_estrategia(cliente_info: dict,
                        coleccion,
                        modelo_emb,
                        historial_chat: list = None) -> str:
    """
    Genera estrategia de cobranza con RAG + Claude API.
    Soporta historial de chat para conversación continua.
    """
    api_key = obtener_api_key()
    if not api_key:
        return "❌ API key de Anthropic no configurada."

    # Consultar RAG
    resultado_rag = consultar_rag_dashboard(
        cliente_info, coleccion, modelo_emb
    )

    # Formatear perfil
    dias  = cliente_info.get('max_dias_atraso', 0)
    prob  = cliente_info.get('probabilidad_mora', 0)
    carga = cliente_info.get('carga_financiera', 0)
    rec   = cliente_info.get('es_cliente_recurrente', 0)
    nivel, _, _ = clasificar_nivel_mora(dias)

    perfil_texto = f"""
CLIENTE EN RIESGO:
  ID                  : {cliente_info.get('cliente_id', 'N/A')}
  Probabilidad mora   : {prob:.1%}
  Días de atraso      : {dias} días ({nivel})
  Carga financiera    : {carga:.1%} del ingreso
  Cliente recurrente  : {'Sí' if rec else 'No'}
  Total cuotas        : {cliente_info.get('n_cuotas_total', 0)}
  Monto crédito       : ${cliente_info.get('monto_credito', 0):,}
    """.strip()

    prompt_sistema = """Eres el Agente de Cobranza del sistema
de inteligencia crediticia. Generas estrategias personalizadas
y accionables para gestores de cobranza.

REGLAS:
- Basa tu respuesta ÚNICAMENTE en los documentos proporcionados
- Sé específico: el gestor debe saber qué hacer HOY
- Respeta siempre el marco legal CONDUSEF
- Máximo 250 palabras por respuesta
- Si te preguntan algo fuera del contexto de cobranza,
  redirige amablemente al tema de cobranza"""

    # Construir mensajes con historial si existe
    messages = []

    if historial_chat:
        messages.extend(historial_chat)
    else:
        # Primera vez — estrategia inicial
        messages.append({
            "role"   : "user",
            "content": f"""
{perfil_texto}

DOCUMENTOS DE REFERENCIA:
{'='*50}
{resultado_rag['contexto_rag']}
{'='*50}

Genera la estrategia de cobranza para este cliente:
1. Acción inmediata (HOY)
2. Guión para la llamada (2-3 líneas)
3. Oferta a presentar
4. Escalamiento si no hay respuesta en 48h
5. Restricciones CONDUSEF relevantes
"""
        })

    try:
        cliente_anthropic = anthropic.Anthropic(
            api_key=api_key
        )
        respuesta = cliente_anthropic.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 800,
            system     = prompt_sistema,
            messages   = messages
        )
        return respuesta.content[0].text

    except Exception as e:
        return f"❌ Error al generar estrategia: {e}"




# ────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares — Agente Level 3
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def _crm_clientes():
    """Lee tabla de clientes del CRM (se refresca cada 10s)."""
    if not _LEVEL3_OK:
        return pd.DataFrame()
    import sqlite3
    con = sqlite3.connect(str(_get_db_path()))
    df = pd.read_sql(
        "SELECT cliente_id, nombre, etapa_cobranza, "
        "ROUND(ews_score,3) as ews_score, dias_mora, "
        "ROUND(monto_adeudado,2) as monto_adeudado, "
        "gestionado_humano "
        "FROM clientes ORDER BY ews_score DESC",
        con
    )
    con.close()
    df['gestionado_humano'] = df['gestionado_humano'].apply(
        lambda x: "🔒 Bloqueado" if x else "✅ Activo"
    )
    return df


@st.cache_data(ttl=10)
def _crm_escalaciones_pendientes():
    """Lee escalaciones pendientes del CRM."""
    if not _LEVEL3_OK:
        return pd.DataFrame()
    import sqlite3
    con = sqlite3.connect(str(_get_db_path()))
    df = pd.read_sql(
        """SELECT e.escalacion_id, e.cliente_id, c.nombre,
           e.nivel, SUBSTR(e.motivo,1,70) as motivo, e.timestamp
           FROM escalaciones e
           JOIN clientes c ON e.cliente_id = c.cliente_id
           WHERE e.estado = 'pendiente'
           ORDER BY e.timestamp DESC""",
        con
    )
    con.close()
    return df


@st.cache_data(ttl=10)
def _crm_log_decisiones(limite: int = 20):
    """Lee las últimas N decisiones del log."""
    if not _LEVEL3_OK:
        return pd.DataFrame()
    import sqlite3
    con = sqlite3.connect(str(_get_db_path()))
    df = pd.read_sql(
        f"""SELECT l.log_id, l.cliente_id, c.nombre,
            l.tool_llamada, l.status,
            SUBSTR(l.razonamiento,1,90) as razonamiento,
            l.timestamp
            FROM log_decisiones l
            JOIN clientes c ON l.cliente_id = c.cliente_id
            ORDER BY l.timestamp DESC LIMIT {limite}""",
        con
    )
    con.close()
    return df


def _resolver_escalacion(escalacion_id: str,
                          resuelto_por: str = "Supervisor") -> None:
    """
    Marca una escalación como resuelta.
    Si no quedan más escalaciones pendientes para el cliente,
    libera gestionado_humano=0 para que el trigger autónomo vuelva a actuar.
    """
    if not _LEVEL3_OK:
        return
    import sqlite3
    from datetime import datetime
    con = sqlite3.connect(str(_get_db_path()))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with con:
        row = con.execute(
            "SELECT cliente_id FROM escalaciones WHERE escalacion_id=?",
            (escalacion_id,)
        ).fetchone()
        con.execute(
            "UPDATE escalaciones SET estado='resuelto', "
            "resuelto_por=?, fecha_resolucion=? "
            "WHERE escalacion_id=?",
            (resuelto_por, ts, escalacion_id)
        )
        if row:
            cid = row[0]
            n_pend = con.execute(
                "SELECT COUNT(*) FROM escalaciones "
                "WHERE cliente_id=? AND estado='pendiente'",
                (cid,)
            ).fetchone()[0]
            if n_pend == 0:
                con.execute(
                    "UPDATE clientes SET gestionado_humano=0 "
                    "WHERE cliente_id=?", (cid,)
                )
    con.close()

# ============================================================
# FUNCIÓN PRINCIPAL DE LA TAB
# ============================================================

def render_tab_cobranza():
    """
    Renderiza la Tab 6 — Agente de Cobranza.
    Esta función es llamada desde app.py.
    """
    st.header("🚨 Agente de Cobranza — Early Warning System")
    st.markdown(
        "Clientes con **alerta activa** detectados por el "
        "modelo EWS (LightGBM AUC=0.8712). "
        "Genera estrategias personalizadas con RAG + Claude API."
    )

    # ── Cargar recursos ──────────────────────────────────────
    modelo_ews = cargar_modelo_ews()
    metadata   = cargar_metadata_ews()
    coleccion, modelo_emb, config = cargar_rag()

    # Verificar que los recursos están disponibles
    recursos_ok = True

    if modelo_ews is None:
        st.warning("⚠️ Modelo EWS no encontrado en "
                   "dashboard/models/modelo_ews.pkl")
        recursos_ok = False

    if coleccion is None:
        st.warning("⚠️ ChromaDB no encontrado. "
                   "Ejecuta el notebook 10 primero.")
        recursos_ok = False

    if not recursos_ok:
        st.stop()

    # ── Métricas del modelo ──────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("AUC-ROC", "0.8712",
                  help="Discriminación del modelo EWS")
    with col2:
        st.metric("KS Statistic", "0.5719",
                  help="Separación buenos/malos pagadores")
    with col3:
        st.metric("Recall", "79.6%",
                  help="% de moras detectadas correctamente")
    with col4:
        st.metric("Umbral decisión", "0.20",
                  help="Justificado por análisis costo-beneficio")

    st.divider()

    # ── Lista de clientes en riesgo ──────────────────────────
    st.subheader("📋 Clientes con Alerta EWS Activa")

    df_riesgo = generar_clientes_riesgo(n_clientes=200)

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        umbral_filtro = st.slider(
            "Probabilidad mínima",
            min_value  = 0.20,
            max_value  = 0.95,
            value      = 0.20,
            step       = 0.05,
            format     = "%.0f%%"
        )
    with col_f2:
        nivel_filtro = st.selectbox(
            "Nivel de mora",
            options = ["Todos", "Preventiva",
                       "Mora leve", "Mora media", "Mora grave"]
        )
    with col_f3:
        recurrente_filtro = st.selectbox(
            "Tipo de cliente",
            options = ["Todos", "Recurrente", "Nuevo"]
        )

    # Aplicar filtros
    df_filtrado = df_riesgo[
        df_riesgo['probabilidad_mora'] >= umbral_filtro
    ].copy()

    if nivel_filtro != "Todos":
        df_filtrado = df_filtrado[
            df_filtrado['max_dias_atraso'].apply(
                lambda d: clasificar_nivel_mora(d)[0]
            ) == nivel_filtro
        ]

    if recurrente_filtro == "Recurrente":
        df_filtrado = df_filtrado[
            df_filtrado['es_cliente_recurrente'] == 1
        ]
    elif recurrente_filtro == "Nuevo":
        df_filtrado = df_filtrado[
            df_filtrado['es_cliente_recurrente'] == 0
        ]

    # Mostrar resumen
    st.caption(
        f"**{len(df_filtrado):,}** clientes con alerta activa "
        f"de {len(df_riesgo):,} evaluados"
    )

    # Preparar tabla para mostrar
    df_display = df_filtrado.head(50).copy()
    df_display['Nivel mora'] = df_display[
        'max_dias_atraso'
    ].apply(lambda d: clasificar_nivel_mora(d)[2] + " " +
                       clasificar_nivel_mora(d)[0])
    df_display['Prob. mora'] = df_display[
        'probabilidad_mora'
    ].apply(lambda x: f"{x:.1%}")
    df_display['Días atraso'] = df_display['max_dias_atraso']
    df_display['Recurrente'] = df_display[
        'es_cliente_recurrente'
    ].apply(lambda x: "✅ Sí" if x else "❌ No")
    df_display['Carga fin.'] = df_display[
        'carga_financiera'
    ].apply(lambda x: f"{x:.0%}")

    # Tabla interactiva
    seleccion = st.dataframe(
        df_display[[
            'cliente_id', 'Prob. mora', 'Días atraso',
            'Nivel mora', 'Recurrente', 'Carga fin.'
        ]].rename(columns={'cliente_id': 'Cliente ID'}),
        use_container_width = True,
        hide_index          = True,
        on_select           = "rerun",
        selection_mode      = "single-row",
        key                 = "tabla_clientes_riesgo"
    )


    # ═══════════════════════════════════════════════════════════════════════
    # AGENTE DE COBRANZA AUTÓNOMO — LEVEL 3
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🤖 Agente de Cobranza Autónomo — Level 3")

    if not _LEVEL3_OK:
        st.warning(
            "⚠️ Módulo `src/crm` no encontrado. "
            "Completa las Fases 1-2 del Agente Level 3 primero."
        )
    else:
        # Fix SSL (necesario para langchain_anthropic en Windows con conda)
        os.environ.pop('SSL_CERT_FILE', None)

        # Inicializar CRM — idempotente, seguro en cada render
        _inicializar_crm()

        # Descripción + métricas de diseño
        col_desc, col_m1, col_m2, col_m3 = st.columns([3, 1, 1, 1])
        with col_desc:
            st.markdown(
                "Agente que **ejecuta acciones** directamente en el CRM sin "
                "aprobación por acción. El supervisor audita el `log_decisiones`. "
                "Casos críticos (judicial · monto >$50k · 5+ intentos fallidos) "
                "requieren confirmación humana antes de ejecutar."
            )
        with col_m1:
            st.metric("Nodos grafo", "7",
                      help="LangGraph StateGraph compilado")
        with col_m2:
            st.metric("Tools activas", "4",
                      help="registrar_contacto · actualizar_estado · "
                           "escalar_caso · generar_propuesta")
        with col_m3:
            st.metric("CRM tables", "5",
                      help="clientes · contactos · escalaciones · "
                           "propuestas · log_decisiones")

        # Sub-tabs de la sección Level 3
        tab_ej, tab_esc, tab_log_d = st.tabs([
            "▶️ Ejecutar agente",
            "🚨 Escalaciones pendientes",
            "📊 Log de decisiones",
        ])

        # ── Tab: Ejecutar ────────────────────────────────────────
        with tab_ej:
            col_ctrl, col_tabla = st.columns([1, 2])

            with col_ctrl:
                modo_autonomo = st.toggle(
                    "⚡ Modo autónomo",
                    value=False,
                    help=(
                        "OFF → modo asistido (agente ejecuta, sin reglas de pausa). "
                        "ON → modo autónomo con pausa obligatoria en casos críticos."
                    ),
                    key="l3_modo_autonomo"
                )
                st.caption(
                    "🟢 Autónomo activo" if modo_autonomo
                    else "🔵 Modo asistido"
                )
                st.divider()

                # Selector de cliente del CRM
                df_clientes = _crm_clientes()
                if df_clientes.empty:
                    st.warning("CRM vacío. Ejecuta init_crm.py primero.")
                else:
                    etiquetas_l3 = [
                        f"{row.cliente_id} — {row.nombre[:18]} "
                        f"({row.etapa_cobranza})"
                        for row in df_clientes.itertuples()
                    ]
                    idx_l3 = st.selectbox(
                        "Selecciona cliente CRM",
                        range(len(df_clientes)),
                        format_func=lambda i: etiquetas_l3[i],
                        key="l3_cliente_sel"
                    )
                    cliente_id_l3 = df_clientes.iloc[idx_l3]['cliente_id']

                    st.divider()

                    # Botón: ejecución individual
                    if st.button(
                        "▶️ Ejecutar agente",
                        type="primary",
                        use_container_width=True,
                        key="l3_btn_individual"
                    ):
                        with st.spinner(f"Procesando {cliente_id_l3}..."):
                            resultado_l3 = _ejecutar_agente(
                                cliente_id_l3,
                                modo_autonomo=modo_autonomo
                            )
                        st.session_state["l3_resultado"]  = resultado_l3
                        st.session_state["l3_cliente_id"] = cliente_id_l3
                        st.cache_data.clear()
                        st.rerun()

                    # Botón: lote EWS
                    if st.button(
                        "⚡ Lote EWS — 5 clientes",
                        use_container_width=True,
                        key="l3_btn_lote"
                    ):
                        df_activos = df_clientes[
                            df_clientes['gestionado_humano'] == "✅ Activo"
                        ].head(5)
                        ids_lote = df_activos['cliente_id'].tolist()

                        if not ids_lote:
                            st.warning("No hay clientes activos en el CRM.")
                        else:
                            barra_l3 = st.progress(
                                0, text="Iniciando lote EWS..."
                            )
                            lote_resultados = []
                            for i_l, cid_l in enumerate(ids_lote):
                                barra_l3.progress(
                                    (i_l + 1) / len(ids_lote),
                                    text=f"Procesando {cid_l} "
                                         f"({i_l+1}/{len(ids_lote)})..."
                                )
                                r_l = _ejecutar_agente(
                                    cid_l, modo_autonomo=modo_autonomo
                                )
                                lote_resultados.append(r_l.to_dict())
                            st.session_state["l3_lote"] = lote_resultados
                            st.cache_data.clear()
                            st.rerun()

            with col_tabla:
                st.markdown("##### Clientes en el CRM")
                st.dataframe(
                    df_clientes.rename(columns={
                        "cliente_id":       "ID",
                        "nombre":           "Nombre",
                        "etapa_cobranza":   "Etapa",
                        "ews_score":        "EWS",
                        "dias_mora":        "Mora (días)",
                        "monto_adeudado":   "Monto ($)",
                        "gestionado_humano":"Estado",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=380,
                )

            # ── Resultado individual ──────────────────────────────
            if "l3_resultado" in st.session_state:
                r_i = st.session_state["l3_resultado"]
                st.divider()
                st.markdown(
                    f"#### Resultado — "
                    f"{st.session_state.get('l3_cliente_id', '')}"
                )

                if r_i.requiere_confirmacion and modo_autonomo:
                    st.warning(
                        "⚠️ **Requiere confirmación humana.** "
                        "Motivo: etapa judicial, monto >$50,000 "
                        "o 5+ intentos fallidos."
                    )
                    col_ap, col_can = st.columns(2)
                    with col_ap:
                        if st.button(
                            "✅ Confirmar y ejecutar",
                            type="primary",
                            key="l3_confirmar"
                        ):
                            with st.spinner("Ejecutando..."):
                                r_conf = _ejecutar_agente(
                                    st.session_state["l3_cliente_id"],
                                    modo_autonomo=False
                                )
                            st.session_state["l3_resultado"] = r_conf
                            st.cache_data.clear()
                            st.rerun()
                    with col_can:
                        if st.button("❌ Cancelar", key="l3_cancelar"):
                            del st.session_state["l3_resultado"]
                            st.rerun()
                else:
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("Acción", r_i.accion_tomada)
                    with col_r2:
                        st.metric("Urgencia", r_i.urgencia or "—")
                    with col_r3:
                        if r_i.error:
                            est = "❌ Error"
                        elif r_i.resultado_ok:
                            est = "✅ OK"
                        else:
                            est = "ℹ️ Sin acción"
                        st.metric("Estado", est)

                    if r_i.mensaje_resultado:
                        st.success(r_i.mensaje_resultado)
                    if r_i.error:
                        st.error(f"Error: {r_i.error}")
                    if r_i.razonamiento:
                        with st.expander("📝 Razonamiento del agente"):
                            st.write(r_i.razonamiento)

            # ── Resultado lote ────────────────────────────────────
            if "l3_lote" in st.session_state:
                st.divider()
                st.markdown("#### Resultado del lote EWS")
                df_lt = pd.DataFrame(st.session_state["l3_lote"])
                cols_lt = [c for c in [
                    "cliente_id", "accion_tomada", "urgencia",
                    "resultado_ok", "requiere_confirmacion",
                    "mensaje_resultado",
                ] if c in df_lt.columns]
                st.dataframe(
                    df_lt[cols_lt].rename(columns={
                        "cliente_id":            "Cliente",
                        "accion_tomada":         "Acción",
                        "urgencia":              "Urgencia",
                        "resultado_ok":          "OK",
                        "requiere_confirmacion": "Pausa",
                        "mensaje_resultado":     "Resultado",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                col_lt1, col_lt2, col_lt3 = st.columns(3)
                with col_lt1:
                    n_ok_lt = sum(
                        1 for r in st.session_state["l3_lote"]
                        if r.get("resultado_ok")
                    )
                    st.metric("Ejecutadas", f"{n_ok_lt}/{len(df_lt)}")
                with col_lt2:
                    n_err_lt = sum(
                        1 for r in st.session_state["l3_lote"]
                        if r.get("error")
                    )
                    st.metric("Errores", n_err_lt)
                with col_lt3:
                    n_pau_lt = sum(
                        1 for r in st.session_state["l3_lote"]
                        if r.get("requiere_confirmacion")
                    )
                    st.metric("Requieren pausa", n_pau_lt)

        # ── Tab: Escalaciones ─────────────────────────────────────
        with tab_esc:
            st.markdown("##### Escalaciones pendientes de resolución humana")
            st.caption(
                "Casos escalados por el agente que requieren intervención. "
                "Al resolver, se libera el bloqueo automático del cliente."
            )
            df_esc_tab = _crm_escalaciones_pendientes()

            if df_esc_tab.empty:
                st.info("✅ No hay escalaciones pendientes en este momento.")
            else:
                st.caption(f"**{len(df_esc_tab)}** casos pendientes")
                for _, row_esc in df_esc_tab.iterrows():
                    with st.container(border=True):
                        ce1, ce2, ce3, ce4 = st.columns([1, 1, 3, 1])
                        with ce1:
                            st.markdown(f"**{row_esc['cliente_id']}**")
                            st.caption(row_esc['nombre'])
                        with ce2:
                            n_col = {
                                "legal": "🔴", "supervisor": "🟠",
                                "castigo": "⚫"
                            }.get(row_esc['nivel'], "🔵")
                            st.markdown(f"{n_col} {row_esc['nivel'].upper()}")
                            st.caption(str(row_esc['timestamp'])[:16])
                        with ce3:
                            st.markdown(row_esc['motivo'])
                        with ce4:
                            if st.button(
                                "✅ Resolver",
                                key=f"resolver_{row_esc['escalacion_id']}",
                                use_container_width=True
                            ):
                                _resolver_escalacion(
                                    row_esc['escalacion_id'],
                                    resuelto_por="Supervisor_Dashboard"
                                )
                                st.success(
                                    f"Caso {row_esc['escalacion_id']} resuelto. "
                                    f"Cliente {row_esc['cliente_id']} reactivado."
                                )
                                st.cache_data.clear()
                                st.rerun()

        # ── Tab: Log de decisiones ────────────────────────────────
        with tab_log_d:
            st.markdown("##### Log de decisiones del agente")
            st.caption(
                "Registro auditable completo: qué tool eligió el agente, "
                "con qué argumentos, qué resultado obtuvo y por qué lo decidió."
            )
            df_log_tab = _crm_log_decisiones(limite=20)

            if df_log_tab.empty:
                st.info("El log de decisiones está vacío.")
            else:
                col_lg1, col_lg2, col_lg3 = st.columns(3)
                with col_lg1:
                    st.metric(
                        "Decisiones exitosas",
                        len(df_log_tab[df_log_tab['status'] == 'ok'])
                    )
                with col_lg2:
                    st.metric(
                        "Errores registrados",
                        len(df_log_tab[df_log_tab['status'] == 'error'])
                    )
                with col_lg3:
                    tool_top_log = (
                        df_log_tab[df_log_tab['tool_llamada'].notna()]
                        ['tool_llamada'].value_counts().idxmax()
                        if df_log_tab['tool_llamada'].notna().any()
                        else "—"
                    )
                    st.metric("Tool más usada", tool_top_log)

                st.dataframe(
                    df_log_tab.rename(columns={
                        "log_id":       "ID",
                        "cliente_id":   "Cliente",
                        "nombre":       "Nombre",
                        "tool_llamada": "Tool",
                        "status":       "Status",
                        "razonamiento": "Razonamiento",
                        "timestamp":    "Timestamp",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )


    # ── Panel de estrategia ──────────────────────────────────
    st.subheader("🎯 Estrategia de Cobranza")

    # Verificar si hay una fila seleccionada
    filas_sel = seleccion.selection.rows \
                if hasattr(seleccion, 'selection') else []

    if not filas_sel:
        st.info(
            "👆 Selecciona un cliente de la tabla "
            "para generar su estrategia de cobranza."
        )
        return

    # Cliente seleccionado
    idx_sel    = filas_sel[0]
    cliente    = df_filtrado.iloc[idx_sel].to_dict()
    nivel, color, emoji = clasificar_nivel_mora(
        cliente['max_dias_atraso']
    )

    # Mostrar datos del cliente seleccionado
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric(
            "Cliente",
            cliente['cliente_id']
        )
    with col_b:
        st.metric(
            "Probabilidad mora",
            f"{cliente['probabilidad_mora']:.1%}",
            delta     = f"{cliente['probabilidad_mora']-UMBRAL_EWS:.1%} sobre umbral",
            delta_color = "inverse"
        )
    with col_c:
        st.metric(
            "Días de atraso",
            f"{cliente['max_dias_atraso']} días",
            help = f"Nivel: {nivel}"
        )
    with col_d:
        st.metric(
            "Carga financiera",
            f"{cliente['carga_financiera']:.1%}",
            help = "% del ingreso mensual destinado a la cuota"
        )

    # ── Inicializar historial de chat ────────────────────────
    chat_key = f"chat_{cliente['cliente_id']}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # ── Botón para generar estrategia inicial ────────────────
    col_btn1, col_btn2 = st.columns([1, 4])

    with col_btn1:
        generar = st.button(
            "🤖 Generar Estrategia",
            type            = "primary",
            use_container_width = True,
            key             = f"btn_gen_{cliente['cliente_id']}"
        )

    with col_btn2:
        if st.button(
            "🔄 Nueva consulta",
            use_container_width = True,
            key = f"btn_reset_{cliente['cliente_id']}"
        ):
            st.session_state[chat_key] = []
            st.rerun()

    # Generar estrategia inicial
    if generar and not st.session_state[chat_key]:
        with st.spinner(
            "🔍 Consultando base de conocimiento RAG + "
            "generando estrategia con Claude API..."
        ):
            estrategia = generar_estrategia(
                cliente_info  = cliente,
                coleccion     = coleccion,
                modelo_emb    = modelo_emb,
            )

            # Guardar en historial
            st.session_state[chat_key] = [
                {
                    "role"   : "user",
                    "content": f"Genera estrategia para cliente "
                               f"{cliente['cliente_id']} con "
                               f"{cliente['max_dias_atraso']} días "
                               f"de atraso y probabilidad de mora "
                               f"{cliente['probabilidad_mora']:.1%}"
                },
                {
                    "role"   : "assistant",
                    "content": estrategia
                }
            ]
            st.rerun()

    # ── Mostrar historial de chat ────────────────────────────
    if st.session_state[chat_key]:
        st.markdown("#### 💬 Conversación con el Agente")

        for mensaje in st.session_state[chat_key]:
            with st.chat_message(mensaje['role']):
                st.markdown(mensaje['content'])

        # ── Input para preguntas de seguimiento ──────────────
        pregunta = st.chat_input(
            "Haz una pregunta al agente de cobranza...",
            key = f"chat_input_{cliente['cliente_id']}"
        )

        if pregunta:
            # Agregar pregunta al historial
            st.session_state[chat_key].append({
                "role"   : "user",
                "content": pregunta
            })

            with st.spinner("Generando respuesta..."):
                # Pasar historial completo para contexto
                respuesta = generar_estrategia(
                    cliente_info  = cliente,
                    coleccion     = coleccion,
                    modelo_emb    = modelo_emb,
                    historial_chat= st.session_state[chat_key]
                )

            st.session_state[chat_key].append({
                "role"   : "assistant",
                "content": respuesta
            })
            st.rerun()
