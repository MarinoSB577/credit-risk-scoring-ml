"""
crm_tools.py — Tools del Agente de Cobranza Level 3.

Separación de roles (crítica para el correcto funcionamiento de LangGraph):

  consultar_cliente()  → Función de servicio de datos. La llama SIEMPRE
                         el nodo enriquecer_contexto. NO está en el menú
                         de tools del LLM. No la decoramos con @tool.

  registrar_contacto() → Tool del agente. El LLM la elige en decidir_accion.
  actualizar_estado()  → Tool del agente.
  escalar_caso()       → Tool del agente.
  generar_propuesta()  → Tool del agente.

  TOOLS_AGENTE         → Lista lista para pasar a model.bind_tools() en el grafo.

Todas las operaciones de escritura usan transacciones SQLite explícitas.
Los errores de SQLite se capturan y devuelven como resultados estructurados
(nunca propagan excepciones al grafo — el nodo registrar_error las gestiona).
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from .schemas import (
    ActualizarEstadoInput,
    ClienteContextResumen,
    ContactoReciente,
    ContactoResult,
    EscalacionActiva,
    EscalacionResult,
    EtapaCobranza,
    EscalarCasoInput,
    GenerarPropuestaInput,
    ORDEN_ETAPAS,
    PropuestaActiva,
    PropuestaResult,
    RegistrarContactoInput,
    UpdateResult,
)
from .init_crm import get_db_path


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    """Conexión con foreign keys activadas y row_factory para acceso por nombre."""
    con = sqlite3.connect(get_db_path())
    con.execute("PRAGMA foreign_keys = ON")
    con.row_factory = sqlite3.Row
    return con


def _uid(prefix: str = "") -> str:
    return prefix + str(uuid.uuid4())[:8].upper()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# FUNCIÓN DE SERVICIO — consultar_cliente
# No decorada con @tool. No aparece en el JSON Schema del LLM.
# ---------------------------------------------------------------------------

def consultar_cliente(cliente_id: str) -> Optional[ClienteContextResumen]:
    """
    Recupera el contexto comprimido del cliente para el estado del grafo.

    Devuelve:
      - Datos base del cliente (EWS, mora, monto, etapa)
      - Últimos 3 contactos ordenados por timestamp desc
      - Conteo total de intentos fallidos (para evaluar_pausa)
      - Última acción registrada por el agente (resumen)
      - Escalaciones con estado='pendiente'
      - Propuestas con estado='enviada'

    El historial completo permanece en SQLite para auditoría.
    Retorna None si el cliente_id no existe.
    """
    con = _conn()
    try:
        row = con.execute(
            "SELECT * FROM clientes WHERE cliente_id = ?", (cliente_id,)
        ).fetchone()

        if row is None:
            return None

        cliente = dict(row)

        # Últimos 3 contactos
        contactos_raw = con.execute("""
            SELECT contacto_id, canal, resultado, timestamp, notas
            FROM contactos
            WHERE cliente_id = ?
            ORDER BY timestamp DESC
            LIMIT 3
        """, (cliente_id,)).fetchall()
        ultimos_contactos = [ContactoReciente(**dict(r)) for r in contactos_raw]

        # Total de intentos fallidos (todos los registros, no solo los últimos 3)
        (intentos_fallidos,) = con.execute("""
            SELECT COUNT(*) FROM contactos
            WHERE cliente_id = ? AND resultado = 'no_contesto'
        """, (cliente_id,)).fetchone()

        # Última acción del agente (resumen legible)
        ultima_row = con.execute("""
            SELECT tool_llamada, razonamiento FROM log_decisiones
            WHERE cliente_id = ? AND status = 'ok' AND tool_llamada IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
        """, (cliente_id,)).fetchone()
        ultima_accion: Optional[str] = None
        if ultima_row:
            razon = (ultima_row["razonamiento"] or "")[:100]
            ultima_accion = f"{ultima_row['tool_llamada']}: {razon}"

        # Escalaciones pendientes
        esc_rows = con.execute("""
            SELECT escalacion_id, nivel, motivo, estado, timestamp
            FROM escalaciones
            WHERE cliente_id = ? AND estado = 'pendiente'
        """, (cliente_id,)).fetchall()
        escalaciones_activas = [EscalacionActiva(**dict(r)) for r in esc_rows]

        # Propuestas activas (sin respuesta del cliente)
        prop_rows = con.execute("""
            SELECT propuesta_id, tipo, monto_propuesto, plazo_meses, estado, timestamp
            FROM propuestas
            WHERE cliente_id = ? AND estado = 'enviada'
        """, (cliente_id,)).fetchall()
        propuestas_activas = [PropuestaActiva(**dict(r)) for r in prop_rows]

        return ClienteContextResumen(
            cliente_id=cliente["cliente_id"],
            nombre=cliente["nombre"],
            ews_score=cliente["ews_score"],
            dias_mora=cliente["dias_mora"],
            monto_adeudado=cliente["monto_adeudado"],
            limite_credito=cliente["limite_credito"],
            etapa_cobranza=cliente["etapa_cobranza"],
            gestor_asignado=cliente["gestor_asignado"],
            gestionado_humano=bool(cliente["gestionado_humano"]),
            intentos_fallidos=intentos_fallidos,
            ultimos_contactos=ultimos_contactos,
            ultima_accion_agente=ultima_accion,
            escalaciones_activas=escalaciones_activas,
            propuestas_activas=propuestas_activas,
        )
    finally:
        con.close()


# ---------------------------------------------------------------------------
# TOOLS DEL AGENTE — expuestas al LLM via LangGraph
# ---------------------------------------------------------------------------

@tool(args_schema=RegistrarContactoInput)
def registrar_contacto(
    cliente_id: str,
    canal: str,
    resultado: str,
    notas: str = "",
) -> dict:
    """
    Registra un intento de contacto con el cliente en el CRM.

    Usa esta herramienta cuando necesites documentar una gestión de cobranza
    (llamada, SMS, email o WhatsApp) y su resultado. Actualiza también la
    fecha de último contacto del cliente.

    Canales disponibles: sms | llamada | email | whatsapp
    Resultados válidos: contesto | no_contesto | promesa_pago | rechazo
    """
    con = _conn()
    try:
        # Normalizar Enums a string puro (compatible Python 3.10 y 3.11+)
        canal_val     = canal.value    if hasattr(canal,    "value") else canal
        resultado_val = resultado.value if hasattr(resultado, "value") else resultado

        contacto_id = _uid("CON")
        ts = _now()

        with con:
            con.execute("""
                INSERT INTO contactos
                    (contacto_id, cliente_id, canal, resultado, notas, agente, timestamp)
                VALUES (?, ?, ?, ?, ?, 'Agente_L3', ?)
            """, (contacto_id, cliente_id, canal_val, resultado_val, notas, ts))

            con.execute("""
                UPDATE clientes SET ultimo_contacto = ?
                WHERE cliente_id = ?
            """, (ts, cliente_id))

        return ContactoResult(
            ok=True,
            contacto_id=contacto_id,
            mensaje=f"Contacto registrado. Canal: {canal_val} | Resultado: {resultado_val}.",
        ).model_dump()

    except sqlite3.IntegrityError as e:
        return ContactoResult(
            ok=False, contacto_id="",
            mensaje=f"Error de integridad: {e}. Verifica que el cliente_id sea válido.",
        ).model_dump()
    except sqlite3.Error as e:
        return ContactoResult(
            ok=False, contacto_id="",
            mensaje=f"Error de base de datos: {e}",
        ).model_dump()
    finally:
        con.close()


@tool(args_schema=ActualizarEstadoInput)
def actualizar_estado(
    cliente_id: str,
    nueva_etapa: str,
    motivo: str,
) -> dict:
    """
    Avanza la etapa de cobranza del cliente en el CRM.

    IMPORTANTE: Solo permite avanzar etapa, nunca retroceder. El orden es:
    preventiva → temprana → administrativa → judicial → castigo.

    Usa esta herramienta cuando la mora o el comportamiento del cliente justifiquen
    una gestión más intensa. Proporciona siempre un motivo claro y documentado.

    Etapas válidas: preventiva | temprana | administrativa | judicial | castigo
    """
    con = _conn()
    try:
        row = con.execute(
            "SELECT etapa_cobranza FROM clientes WHERE cliente_id = ?",
            (cliente_id,)
        ).fetchone()

        if row is None:
            return UpdateResult(
                ok=False, etapa_anterior="", etapa_nueva=nueva_etapa,
                mensaje=f"Cliente '{cliente_id}' no encontrado en el CRM.",
            ).model_dump()

        etapa_actual = row["etapa_cobranza"]

        # Validación con Enum ordenado — nunca comparar strings directamente
        try:
            ea = EtapaCobranza(etapa_actual)
            en = EtapaCobranza(nueva_etapa)
        except ValueError as e:
            return UpdateResult(
                ok=False, etapa_anterior=etapa_actual, etapa_nueva=nueva_etapa,
                mensaje=f"Etapa no reconocida: {e}",
            ).model_dump()

        # Valor string puro (compatible Python 3.10 y 3.11+)
        nueva_etapa_val = en.value

        if ORDEN_ETAPAS[en] <= ORDEN_ETAPAS[ea]:
            return UpdateResult(
                ok=False, etapa_anterior=etapa_actual, etapa_nueva=nueva_etapa_val,
                mensaje=(
                    f"Operación rechazada: no se puede mover de '{etapa_actual}' "
                    f"a '{nueva_etapa_val}'. Solo se permiten avances."
                ),
            ).model_dump()

        with con:
            con.execute(
                "UPDATE clientes SET etapa_cobranza = ? WHERE cliente_id = ?",
                (nueva_etapa_val, cliente_id)
            )

        return UpdateResult(
            ok=True,
            etapa_anterior=etapa_actual,
            etapa_nueva=nueva_etapa_val,
            mensaje=f"Etapa actualizada: {etapa_actual} → {nueva_etapa_val}. Motivo: {motivo}",
        ).model_dump()

    except sqlite3.Error as e:
        return UpdateResult(
            ok=False, etapa_anterior="", etapa_nueva=nueva_etapa,
            mensaje=f"Error de base de datos: {e}",
        ).model_dump()
    finally:
        con.close()


@tool(args_schema=EscalarCasoInput)
def escalar_caso(
    cliente_id: str,
    nivel: str,
    motivo: str,
) -> dict:
    """
    Escala el caso del cliente a supervisión humana, área legal o castigo contable.

    Al escalar, el cliente queda marcado como gestionado_humano=True, lo que
    suspende los triggers automáticos del agente hasta que un humano resuelva
    la escalación desde el panel de Tab 6.

    Usa esta herramienta cuando el caso supere la capacidad de gestión autónoma:
    monto muy alto, proceso legal iniciado, o patrón persistente de no-contacto.

    Niveles: supervisor | legal | castigo
    """
    con = _conn()
    try:
        nivel_val = nivel.value if hasattr(nivel, "value") else nivel
        escalacion_id = _uid("ESC")

        with con:
            con.execute("""
                INSERT INTO escalaciones
                    (escalacion_id, cliente_id, nivel, motivo, estado, timestamp)
                VALUES (?, ?, ?, ?, 'pendiente', ?)
            """, (escalacion_id, cliente_id, nivel_val, motivo, _now()))

            con.execute(
                "UPDATE clientes SET gestionado_humano = 1 WHERE cliente_id = ?",
                (cliente_id,)
            )

        return EscalacionResult(
            ok=True,
            escalacion_id=escalacion_id,
            gestionado_humano=True,
            mensaje=(
                f"Caso escalado a {nivel_val}. Trigger autónomo suspendido. "
                f"Requiere resolución humana desde Tab 6."
            ),
        ).model_dump()

    except sqlite3.IntegrityError as e:
        return EscalacionResult(
            ok=False, escalacion_id="", gestionado_humano=False,
            mensaje=f"Error de integridad: {e}",
        ).model_dump()
    except sqlite3.Error as e:
        return EscalacionResult(
            ok=False, escalacion_id="", gestionado_humano=False,
            mensaje=f"Error de base de datos: {e}",
        ).model_dump()
    finally:
        con.close()


@tool(args_schema=GenerarPropuestaInput)
def generar_propuesta(
    cliente_id: str,
    tipo: str,
    monto_propuesto: float,
    plazo_meses: int,
) -> dict:
    """
    Genera una propuesta de solución para regularizar la deuda del cliente.

    La propuesta queda registrada con estado='enviada' hasta que el cliente
    responda (aceptada | rechazada). La validación de monto la hace esta
    herramienta, no el LLM: el monto propuesto no puede superar la deuda
    vigente en más del 10%.

    Usa esta herramienta cuando el cliente ha demostrado disposición de pago
    pero necesita condiciones especiales para regularizarse.

    Tipos: quita | plan_pagos | refinanciamiento
    Plazo: 1-60 meses
    """
    con = _conn()
    try:
        tipo_val = tipo.value if hasattr(tipo, "value") else tipo
        row = con.execute(
            "SELECT monto_adeudado FROM clientes WHERE cliente_id = ?",
            (cliente_id,)
        ).fetchone()

        if row is None:
            return PropuestaResult(
                ok=False, propuesta_id="",
                mensaje=f"Cliente '{cliente_id}' no encontrado en el CRM.",
            ).model_dump()

        monto_adeudado = row["monto_adeudado"]

        # Validación de negocio: la propuesta no puede exceder la deuda en +10%
        if monto_propuesto > monto_adeudado * 1.10:
            return PropuestaResult(
                ok=False, propuesta_id="",
                mensaje=(
                    f"Monto propuesto (${monto_propuesto:,.2f}) supera la deuda "
                    f"vigente (${monto_adeudado:,.2f}) en más del 10%. Ajusta el monto."
                ),
            ).model_dump()

        propuesta_id = _uid("PRO")

        with con:
            con.execute("""
                INSERT INTO propuestas
                    (propuesta_id, cliente_id, tipo, monto_propuesto,
                     plazo_meses, estado, timestamp)
                VALUES (?, ?, ?, ?, ?, 'enviada', ?)
            """, (propuesta_id, cliente_id, tipo_val, monto_propuesto, plazo_meses, _now()))

        return PropuestaResult(
            ok=True,
            propuesta_id=propuesta_id,
            mensaje=(
                f"Propuesta generada: {tipo_val} por ${monto_propuesto:,.2f} "
                f"en {plazo_meses} meses. Estado: enviada."
            ),
        ).model_dump()

    except sqlite3.IntegrityError as e:
        return PropuestaResult(
            ok=False, propuesta_id="",
            mensaje=f"Error de integridad: {e}",
        ).model_dump()
    except sqlite3.Error as e:
        return PropuestaResult(
            ok=False, propuesta_id="",
            mensaje=f"Error de base de datos: {e}",
        ).model_dump()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Exportación para el grafo LangGraph
# ---------------------------------------------------------------------------

TOOLS_AGENTE = [
    registrar_contacto,
    actualizar_estado,
    escalar_caso,
    generar_propuesta,
]
