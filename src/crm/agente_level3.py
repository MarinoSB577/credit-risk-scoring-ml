"""
agente_level3.py — Grafo LangGraph del Agente de Cobranza Autónomo Level 3.

Arquitectura: 6 nodos funcionales + END con separación estricta de responsabilidades.

  1. enriquecer_contexto  → CRM → ClienteContextResumen en estado
  2. evaluar_riesgo       → LLM + RAG cobranza → EvaluacionRiesgo estructurada
  3. decidir_accion       → LLM tool calling → ToolCall | None
  4. ejecutar_accion      → ejecuta la tool elegida contra el CRM
  5. registrar_decision   → INSERT log_decisiones (resultado operativo o de negocio)
  6. registrar_error      → INSERT log_decisiones (error técnico)
  7. END                  → resultado serializado para Tab 6

Edges condicionales:
  enriquecer_contexto → error          : registrar_error
  enriquecer_contexto → ok             : evaluar_riesgo
  decidir_accion      → tool_call=None : END  (sin acción procedente)
  decidir_accion      → error          : registrar_error
  decidir_accion      → tool_call      : ejecutar_accion
  ejecutar_accion     → error técnico  : registrar_error
  ejecutar_accion     → resultado      : registrar_decision  (incluso si ok=False)

API pública:
  ejecutar_agente(cliente_id, modo_autonomo) → ResultadoAgente
  ejecutar_lote_ews(cliente_ids, modo_autonomo) → List[ResultadoAgente]
"""

from __future__ import annotations

import json
import operator
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .crm_tools import TOOLS_AGENTE
from .init_crm import get_db_path
from .schemas import (
    ClienteContextResumen,
    EvaluacionRiesgo,
    evaluar_pausa,
)
from .crm_tools import consultar_cliente


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Modelo LLM — ajustar según el requirements.txt del proyecto
MODELO_LLM = "claude-haiku-4-5-20251001"   # rápido para demo; cambiar a claude-sonnet-4-6 para producción

# Path de ChromaDB — relativo al root del proyecto (donde se ejecuta streamlit)
_ROOT = Path(__file__).parent.parent.parent
CHROMA_PATH = str(_ROOT / "chroma_db")
COLECCION_COBRANZA = "cobranza_knowledge_base"


# ---------------------------------------------------------------------------
# LLM — dos instancias con roles distintos
# ---------------------------------------------------------------------------

_llm_base = ChatAnthropic(model=MODELO_LLM, temperature=0)

# Decisor: tool calling explícito con las 4 tools del agente
_llm_decisor = _llm_base.bind_tools(TOOLS_AGENTE)

# Nota: with_structured_output eliminado — se usa JSON directo en nodo_evaluar_riesgo
# para evitar dependencias de versión de langchain_anthropic


# ---------------------------------------------------------------------------
# RAG de cobranza — standalone, sin dependencia de Streamlit
# ---------------------------------------------------------------------------

def _consultar_rag_cobranza(query: str, n_resultados: int = 3) -> List[str]:
    """
    Consulta la colección ChromaDB de cobranza.
    Degradación elegante: si la colección no está disponible, retorna lista vacía
    y el agente continúa sin contexto normativo (menos óptimo pero funcional).

    Args:
        query:       Texto de búsqueda semántica.
        n_resultados: Número de fragmentos a recuperar.

    Returns:
        Lista de strings con los fragmentos más relevantes.
    """
    try:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        coleccion = client.get_collection(COLECCION_COBRANZA)
        n = min(n_resultados, coleccion.count())
        if n == 0:
            return []
        resultados = coleccion.query(query_texts=[query], n_results=n)
        return resultados["documents"][0] if resultados["documents"] else []
    except Exception:
        # Colección no encontrada, ChromaDB no disponible, o colección vacía
        return []


# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------

class EstadoCobranza(TypedDict):
    """
    Estado completo que viaja entre nodos del grafo.

    Convenciones:
      - Campos Optional empiezan en None y se pueblan a medida que avanza el grafo.
      - historial_tools usa Annotated + operator.add para acumulación automática.
      - requiere_confirmacion: si True en modo autónomo, Tab 6 debe pedir aprobación
        antes de invocar el grafo (el grafo no verifica esto internamente).
      - error: cualquier nodo puede escribir aquí; el router lo detecta y desvía.
    """
    cliente_id:            str
    contexto_resumen:      Optional[ClienteContextResumen]
    evaluacion:            Optional[EvaluacionRiesgo]
    tool_call:             Optional[dict]            # {"name": str, "args": dict, "id": str}
    resultado_tool:        Optional[dict]            # output de la tool ejecutada
    historial_tools:       Annotated[List[dict], operator.add]  # acumulativo en sesión
    error:                 Optional[str]
    requiere_confirmacion: bool
    modo_autonomo:         bool


# ---------------------------------------------------------------------------
# Helpers de prompt
# ---------------------------------------------------------------------------

def _fmt_contactos(contactos: List) -> str:
    if not contactos:
        return "  Sin contactos registrados."
    lineas = []
    for c in contactos:
        lineas.append(f"  [{c.timestamp[:10]}] {c.canal} → {c.resultado}"
                      + (f": {c.notas}" if c.notas else ""))
    return "\n".join(lineas)


def _fmt_escalaciones(escalaciones: List) -> str:
    if not escalaciones:
        return "  Sin escalaciones activas."
    return "\n".join(
        f"  [{e.nivel.upper()}] {e.motivo[:80]} (estado: {e.estado})"
        for e in escalaciones
    )


def _fmt_propuestas(propuestas: List) -> str:
    if not propuestas:
        return "  Sin propuestas activas."
    return "\n".join(
        f"  {p.tipo} ${p.monto_propuesto:,.0f} / {p.plazo_meses}m → {p.estado}"
        for p in propuestas
    )


def _construir_prompt_evaluacion(
    ctx: ClienteContextResumen,
    fragmentos_rag: List[str],
) -> List:
    rag_texto = (
        "\n".join(f"  • {f[:200]}" for f in fragmentos_rag)
        if fragmentos_rag
        else "  No disponible — opera sin contexto normativo."
    )

    # Esquema JSON que el LLM debe respetar (campos y tipos exactos)
    _SCHEMA = (
        '{"urgencia": "alta|media|baja", '
        '"etapa_recomendada": "preventiva|temprana|administrativa|judicial|castigo o null si mantiene etapa", '
        '"accion_sugerida": "descripción concreta de la acción a tomar", '
        '"restricciones_normativas": ["lista de restricciones CNBV aplicables o lista vacía"], '
        '"razonamiento": "explicación detallada del análisis"}'
    )

    sistema = (
        "Eres un experto en cobranza de microfinanzas en México con conocimiento "
        "profundo de la normativa CNBV. Analiza el perfil del cliente y determina "
        "la estrategia de cobranza óptima. Sé preciso y orientado a acción.\n\n"
        "IMPORTANTE: Responde ÚNICAMENTE con un objeto JSON válido con esta estructura:\n"
        + _SCHEMA +
        "\nNo incluyas texto, explicaciones ni bloques de markdown antes o después del JSON."
    )

    humano = f"""
PERFIL DEL CLIENTE
  ID: {ctx.cliente_id} | Nombre: {ctx.nombre}
  EWS Score: {ctx.ews_score:.3f} | Días de mora: {ctx.dias_mora}
  Monto adeudado: ${ctx.monto_adeudado:,.2f} | Límite de crédito: ${ctx.limite_credito:,.2f}
  Etapa actual: {ctx.etapa_cobranza.upper()}
  Gestionado por humano: {ctx.gestionado_humano}
  Intentos de contacto fallidos acumulados: {ctx.intentos_fallidos}
  Última acción del agente: {ctx.ultima_accion_agente or 'Ninguna'}

ÚLTIMOS CONTACTOS (máx. 3):
{_fmt_contactos(ctx.ultimos_contactos)}

ESCALACIONES ACTIVAS:
{_fmt_escalaciones(ctx.escalaciones_activas)}

PROPUESTAS ACTIVAS:
{_fmt_propuestas(ctx.propuestas_activas)}

NORMATIVA CNBV APLICABLE (RAG):
{rag_texto}

Evalúa la situación y determina la estrategia de cobranza más apropiada,
considerando la normativa aplicable y el perfil de riesgo del cliente.
""".strip()

    return [SystemMessage(content=sistema), HumanMessage(content=humano)]


def _construir_prompt_decision(
    ctx: ClienteContextResumen,
    evaluacion: EvaluacionRiesgo,
) -> List:
    sistema = """Eres un agente de cobranza autónomo. Tu tarea es elegir UNA herramienta
y ejecutarla con los argumentos correctos según la evaluación de riesgo recibida.

REGLAS ESTRICTAS:
1. Si gestionado_humano=True: NO uses ninguna herramienta. Responde explicando por qué.
2. Si ya existe una propuesta activa del mismo tipo: NO generes otra. Registra contacto en su lugar.
3. Si ya existe una escalación pendiente al mismo nivel: NO escales de nuevo.
4. Elige SOLO una herramienta por ejecución.
5. Usa argumentos precisos y descriptivos (notas y motivos deben ser informativos).
6. Si no hay acción procedente, responde SIN usar herramientas e indica el motivo."""

    humano = f"""
CONTEXTO DEL CLIENTE:
  {ctx.nombre} (ID: {ctx.cliente_id})
  Etapa: {ctx.etapa_cobranza} | Mora: {ctx.dias_mora} días | Deuda: ${ctx.monto_adeudado:,.2f}
  Intentos fallidos: {ctx.intentos_fallidos} | Gestionado por humano: {ctx.gestionado_humano}
  Escalaciones activas: {len(ctx.escalaciones_activas)}
  Propuestas activas: {len(ctx.propuestas_activas)}

EVALUACIÓN DE RIESGO:
  Urgencia: {evaluacion.urgencia.upper()}
  Acción sugerida: {evaluacion.accion_sugerida}
  Etapa recomendada: {evaluacion.etapa_recomendada or 'Mantener etapa actual'}
  Razonamiento: {evaluacion.razonamiento[:300]}

Elige la herramienta apropiada y proporciona todos los argumentos necesarios.
""".strip()

    return [SystemMessage(content=sistema), HumanMessage(content=humano)]


# ---------------------------------------------------------------------------
# Helpers de persistencia en log_decisiones
# ---------------------------------------------------------------------------

def _uid(prefix: str = "") -> str:
    return prefix + str(uuid.uuid4())[:8].upper()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _insertar_log(
    cliente_id: str,
    nodo_origen: str,
    tool_llamada: Optional[str],
    argumentos_json: Optional[str],
    resultado_json: Optional[str],
    razonamiento: Optional[str],
    status: str,  # "ok" | "error"
) -> None:
    """INSERT atómico en log_decisiones. Falla silenciosa — no propaga excepciones."""
    try:
        con = sqlite3.connect(get_db_path())
        con.execute("PRAGMA foreign_keys = ON")
        with con:
            con.execute("""
                INSERT INTO log_decisiones
                    (log_id, cliente_id, nodo_origen, tool_llamada, argumentos_json,
                     resultado_json, razonamiento, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                _uid("LOG"), cliente_id, nodo_origen,
                tool_llamada, argumentos_json, resultado_json,
                razonamiento, status, _now(),
            ))
        con.close()
    except Exception:
        pass  # El log no debe bloquear el flujo


# ---------------------------------------------------------------------------
# Nodos del grafo
# ---------------------------------------------------------------------------

def nodo_enriquecer_contexto(state: EstadoCobranza) -> dict:
    """
    Nodo 1 — Carga el contexto comprimido del cliente desde el CRM.
    Evalúa si el caso requiere confirmación humana (modo autónomo).
    """
    ctx = consultar_cliente(state["cliente_id"])

    if ctx is None:
        return {
            "error": f"Cliente '{state['cliente_id']}' no encontrado en el CRM.",
            "requiere_confirmacion": False,
        }

    # Si el cliente ya tiene escalación activa y está marcado como gestionado_humano,
    # el agente no debe actuar — el campo se propagará al nodo decidir_accion
    requiere_pausa = evaluar_pausa(ctx) and state.get("modo_autonomo", False)

    return {
        "contexto_resumen": ctx,
        "requiere_confirmacion": requiere_pausa,
        "error": None,
    }


def nodo_evaluar_riesgo(state: EstadoCobranza) -> dict:
    """
    Nodo 2 — Evalúa el riesgo del cliente combinando contexto CRM + RAG normativo.
    Produce un objeto EvaluacionRiesgo parseando JSON directamente del LLM.
    Approach robusto: evita dependencias de versión de with_structured_output.
    """
    ctx = state["contexto_resumen"]

    # Construir query semántica para el RAG basada en el perfil
    query_rag = (
        f"estrategia cobranza {ctx.etapa_cobranza} "
        f"mora {ctx.dias_mora} dias "
        f"monto {int(ctx.monto_adeudado)} pesos"
    )
    fragmentos = _consultar_rag_cobranza(query_rag)

    mensajes = _construir_prompt_evaluacion(ctx, fragmentos)

    try:
        respuesta = _llm_base.invoke(mensajes)
        content = respuesta.content if hasattr(respuesta, "content") else str(respuesta)

        # Limpiar posibles bloques de markdown que el LLM pueda añadir
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        else:
            content = content.strip()

        data = json.loads(content)
        evaluacion = EvaluacionRiesgo(**data)
        return {"evaluacion": evaluacion, "error": None}

    except json.JSONDecodeError as e:
        return {"error": f"Error al parsear JSON de evaluar_riesgo: {e}. Respuesta: {content[:200]}"}
    except Exception as e:
        import traceback
        return {"error": f"Error en evaluar_riesgo: {type(e).__name__}: {e}"}


def nodo_decidir_accion(state: EstadoCobranza) -> dict:
    """
    Nodo 3 — El LLM elige qué tool ejecutar (o decide no actuar).
    Produce tool_call con {name, args, id} o None si no hay acción procedente.
    """
    ctx = state["contexto_resumen"]
    evaluacion = state.get("evaluacion")

    # Guardia defensiva — no debería llegar aquí con evaluacion=None
    # (el edge condicional de evaluar_riesgo lo previene), pero por robustez:
    if evaluacion is None:
        return {"error": "evaluacion es None en decidir_accion — evaluar_riesgo falló silenciosamente"}

    # Guardia: cliente gestionado por humano → no actuar
    if ctx.gestionado_humano:
        return {
            "tool_call": None,
            "error": None,
        }

    mensajes = _construir_prompt_decision(ctx, evaluacion)

    try:
        respuesta = _llm_decisor.invoke(mensajes)

        if respuesta.tool_calls:
            tc = respuesta.tool_calls[0]  # una sola acción por ejecución
            tool_call = {
                "name": tc["name"],
                "args": tc["args"],
                "id":   tc.get("id", _uid()),
            }
            return {
                "tool_call":      tool_call,
                "historial_tools": [{"name": tc["name"], "args": tc["args"]}],
                "error":          None,
            }
        else:
            # El LLM decidió que no hay acción procedente
            return {
                "tool_call":      None,
                "historial_tools": [],
                "error":          None,
            }

    except Exception as e:
        return {"error": f"Error en decidir_accion: {e}"}


def nodo_ejecutar_accion(state: EstadoCobranza) -> dict:
    """
    Nodo 4 — Ejecuta la tool elegida por el LLM en el nodo anterior.
    Diferencia errores técnicos (excepción) de resultados de negocio (ok=False).
    Solo los errores técnicos activan el edge hacia registrar_error.
    """
    tc = state["tool_call"]
    tool_map = {t.name: t for t in TOOLS_AGENTE}
    tool_fn = tool_map.get(tc["name"])

    if tool_fn is None:
        return {
            "error": f"Tool desconocida: '{tc['name']}'. Tools disponibles: {list(tool_map.keys())}",
            "resultado_tool": None,
        }

    try:
        resultado = tool_fn.invoke(tc["args"])
        # ok=False es un resultado de negocio válido (validación fallida, constraint),
        # no un error técnico — pasa igual por registrar_decision para dejar el log.
        return {"resultado_tool": resultado, "error": None}

    except Exception as e:
        # Error técnico: excepción no controlada dentro de la tool
        return {
            "error": f"Error técnico en '{tc['name']}': {e}",
            "resultado_tool": None,
        }


def nodo_registrar_decision(state: EstadoCobranza) -> dict:
    """
    Nodo 5 — Registra en log_decisiones el resultado de la acción ejecutada.
    Se ejecuta tanto para resultados exitosos (ok=True) como de negocio (ok=False).
    """
    tc = state.get("tool_call") or {}
    resultado = state.get("resultado_tool") or {}
    evaluacion = state.get("evaluacion")

    _insertar_log(
        cliente_id=state["cliente_id"],
        nodo_origen="ejecutar_accion",
        tool_llamada=tc.get("name"),
        argumentos_json=json.dumps(tc.get("args", {})),
        resultado_json=json.dumps(resultado),
        razonamiento=evaluacion.razonamiento if evaluacion else None,
        status="ok",
    )
    return {}


def nodo_registrar_error(state: EstadoCobranza) -> dict:
    """
    Nodo 6 — Registra en log_decisiones los errores técnicos del grafo.
    No propaga excepciones — el grafo siempre termina en END.
    """
    error = state.get("error", "Error no especificado")
    tc = state.get("tool_call") or {}

    _insertar_log(
        cliente_id=state["cliente_id"],
        nodo_origen="registrar_error",
        tool_llamada=tc.get("name"),
        argumentos_json=json.dumps(tc.get("args", {})) if tc else None,
        resultado_json=json.dumps({"error": error}),
        razonamiento=None,
        status="error",
    )
    return {}


# ---------------------------------------------------------------------------
# Funciones de routing (edges condicionales)
# ---------------------------------------------------------------------------

def _route_enriquecer_contexto(state: EstadoCobranza) -> str:
    if state.get("error"):
        return "registrar_error"
    return "evaluar_riesgo"


def _route_evaluar_riesgo(state: EstadoCobranza) -> str:
    # Si evaluar_riesgo capturó una excepción, error está en el estado
    # y evaluacion sigue en None — no debe llegar a decidir_accion
    if state.get("error") or state.get("evaluacion") is None:
        return "registrar_error"
    return "decidir_accion"


def _route_decidir_accion(state: EstadoCobranza) -> str:
    if state.get("error"):
        return "registrar_error"
    if state.get("tool_call") is None:
        return END           # sin acción procedente — termina limpiamente
    return "ejecutar_accion"


def _route_ejecutar_accion(state: EstadoCobranza) -> str:
    # Solo errores técnicos (resultado_tool=None + error presente) van a registrar_error
    if state.get("error") and state.get("resultado_tool") is None:
        return "registrar_error"
    return "registrar_decision"


# ---------------------------------------------------------------------------
# Construcción y compilación del grafo
# ---------------------------------------------------------------------------

def _construir_grafo() -> StateGraph:
    g = StateGraph(EstadoCobranza)

    # Nodos
    g.add_node("enriquecer_contexto", nodo_enriquecer_contexto)
    g.add_node("evaluar_riesgo",      nodo_evaluar_riesgo)
    g.add_node("decidir_accion",      nodo_decidir_accion)
    g.add_node("ejecutar_accion",     nodo_ejecutar_accion)
    g.add_node("registrar_decision",  nodo_registrar_decision)
    g.add_node("registrar_error",     nodo_registrar_error)

    # Entry point
    g.set_entry_point("enriquecer_contexto")

    # Edges lineales
    g.add_edge("registrar_decision", END)
    g.add_edge("registrar_error",    END)

    # Edges condicionales
    g.add_conditional_edges(
        "enriquecer_contexto",
        _route_enriquecer_contexto,
        {"evaluar_riesgo": "evaluar_riesgo", "registrar_error": "registrar_error"},
    )
    g.add_conditional_edges(
        "evaluar_riesgo",
        _route_evaluar_riesgo,
        {"decidir_accion": "decidir_accion", "registrar_error": "registrar_error"},
    )
    g.add_conditional_edges(
        "decidir_accion",
        _route_decidir_accion,
        {"ejecutar_accion": "ejecutar_accion",
         "registrar_error": "registrar_error",
         END: END},
    )
    g.add_conditional_edges(
        "ejecutar_accion",
        _route_ejecutar_accion,
        {"registrar_decision": "registrar_decision",
         "registrar_error":    "registrar_error"},
    )

    return g.compile()


# Grafo compilado — singleton del módulo
grafo_cobranza = _construir_grafo()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

class ResultadoAgente:
    """Resultado serializable de una ejecución del agente para un cliente."""

    def __init__(self, estado_final: EstadoCobranza) -> None:
        tc = estado_final.get("tool_call") or {}
        resultado = estado_final.get("resultado_tool") or {}
        evaluacion = estado_final.get("evaluacion")

        self.cliente_id:            str            = estado_final["cliente_id"]
        self.accion_tomada:         str            = tc.get("name", "sin_accion")
        self.argumentos:            dict           = tc.get("args", {})
        self.resultado_ok:          bool           = resultado.get("ok", False) if resultado else False
        self.mensaje_resultado:     str            = resultado.get("mensaje", "") if resultado else ""
        self.urgencia:              str            = evaluacion.urgencia if evaluacion else ""
        self.razonamiento:          str            = evaluacion.razonamiento if evaluacion else ""
        self.requiere_confirmacion: bool           = estado_final.get("requiere_confirmacion", False)
        self.error:                 Optional[str]  = estado_final.get("error")
        self.historial_tools:       List[dict]     = estado_final.get("historial_tools", [])

    def to_dict(self) -> dict:
        return {
            "cliente_id":            self.cliente_id,
            "accion_tomada":         self.accion_tomada,
            "argumentos":            self.argumentos,
            "resultado_ok":          self.resultado_ok,
            "mensaje_resultado":     self.mensaje_resultado,
            "urgencia":              self.urgencia,
            "razonamiento":          self.razonamiento,
            "requiere_confirmacion": self.requiere_confirmacion,
            "error":                 self.error,
            "historial_tools":       self.historial_tools,
        }

    def __repr__(self) -> str:
        estado = "✓" if self.resultado_ok else ("⚠" if not self.error else "✗")
        return (
            f"ResultadoAgente({estado} {self.cliente_id} | "
            f"accion={self.accion_tomada} | urgencia={self.urgencia})"
        )


def ejecutar_agente(
    cliente_id: str,
    modo_autonomo: bool = False,
) -> ResultadoAgente:
    """
    Ejecuta el grafo de cobranza para un cliente.

    En modo_autonomo=True, el grafo ejecuta la acción directamente.
    Si requiere_confirmacion=True en el resultado, Tab 6 debe pedir aprobación
    al usuario antes de llamar a esta función — la verificación es responsabilidad
    de la capa de presentación, no del grafo.

    Args:
        cliente_id:    ID del cliente en el CRM (ej. "CLI006").
        modo_autonomo: Si True, el agente actúa sin aprobación por acción.

    Returns:
        ResultadoAgente con la acción tomada, el resultado y el razonamiento.
    """
    estado_inicial: EstadoCobranza = {
        "cliente_id":            cliente_id,
        "contexto_resumen":      None,
        "evaluacion":            None,
        "tool_call":             None,
        "resultado_tool":        None,
        "historial_tools":       [],
        "error":                 None,
        "requiere_confirmacion": False,
        "modo_autonomo":         modo_autonomo,
    }

    estado_final = grafo_cobranza.invoke(estado_inicial)
    return ResultadoAgente(estado_final)


def ejecutar_lote_ews(
    cliente_ids: List[str],
    modo_autonomo: bool = False,
) -> List[ResultadoAgente]:
    """
    Procesa una lista de clientes en serie (output del batch EWS diario).

    Diseño serial deliberado: más simple, más fácil de auditar, suficiente para
    portfolios de demo y producción de escala pequeña.
    Mejora futura documentada: asyncio para portfolios >500 clientes concurrentes.

    Args:
        cliente_ids:   Lista de IDs a procesar (ej. output del EWS trigger).
        modo_autonomo: Aplica a todos los clientes del lote.

    Returns:
        Lista de ResultadoAgente en el mismo orden que cliente_ids.
    """
    resultados = []
    for cid in cliente_ids:
        resultado = ejecutar_agente(cid, modo_autonomo=modo_autonomo)
        resultados.append(resultado)
    return resultados
