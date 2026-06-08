"""
schemas.py — Modelos Pydantic para el CRM del Agente de Cobranza Level 3.

Separado de crm_tools.py para que el grafo LangGraph pueda importar tipos
sin cargar la lógica de base de datos.

Contiene:
  - Enums con orden explícito (EtapaCobranza, canales, resultados)
  - Modelos de contexto que viajan en el estado del grafo
  - Inputs/outputs de cada tool (JSON Schema para LangGraph)
  - Reglas de pausa para modo autónomo (sin floats inventados)
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EtapaCobranza(str, Enum):
    PREVENTIVA     = "preventiva"
    TEMPRANA       = "temprana"
    ADMINISTRATIVA = "administrativa"
    JUDICIAL       = "judicial"
    CASTIGO        = "castigo"


# Orden explícito — única fuente de verdad para validar avances de etapa.
# Clave de diseño: usar este dict, nunca comparar strings directamente.
ORDEN_ETAPAS: dict[EtapaCobranza, int] = {
    EtapaCobranza.PREVENTIVA:     0,
    EtapaCobranza.TEMPRANA:       1,
    EtapaCobranza.ADMINISTRATIVA: 2,
    EtapaCobranza.JUDICIAL:       3,
    EtapaCobranza.CASTIGO:        4,
}


class CanalContacto(str, Enum):
    SMS      = "sms"
    LLAMADA  = "llamada"
    EMAIL    = "email"
    WHATSAPP = "whatsapp"


class ResultadoContacto(str, Enum):
    CONTESTO     = "contesto"
    NO_CONTESTO  = "no_contesto"
    PROMESA_PAGO = "promesa_pago"
    RECHAZO      = "rechazo"


class NivelEscalacion(str, Enum):
    SUPERVISOR = "supervisor"
    LEGAL      = "legal"
    CASTIGO    = "castigo"


class TipoPropuesta(str, Enum):
    QUITA            = "quita"
    PLAN_PAGOS       = "plan_pagos"
    REFINANCIAMIENTO = "refinanciamiento"


class EstadoPropuesta(str, Enum):
    ENVIADA   = "enviada"
    ACEPTADA  = "aceptada"
    RECHAZADA = "rechazada"


# ---------------------------------------------------------------------------
# Modelos de contexto — viajan en el estado del grafo
# ---------------------------------------------------------------------------

class ContactoReciente(BaseModel):
    contacto_id: str
    canal:       str
    resultado:   str
    timestamp:   str
    notas:       str = ""


class EscalacionActiva(BaseModel):
    escalacion_id: str
    nivel:         str
    motivo:        str
    estado:        str
    timestamp:     str


class PropuestaActiva(BaseModel):
    propuesta_id:    str
    tipo:            str
    monto_propuesto: float
    plazo_meses:     int
    estado:          str
    timestamp:       str


class ClienteContextResumen(BaseModel):
    """
    Versión comprimida del contexto que viaja en el estado del grafo LangGraph.

    Diseño deliberado:
      - Últimos 3 contactos, no el historial completo → controla tokens al LLM.
      - intentos_fallidos como int calculado → alimenta reglas_pausa sin string parsing.
      - El historial completo permanece en SQLite para auditoría.
    """
    cliente_id:       str
    nombre:           str
    ews_score:        float
    dias_mora:        int
    monto_adeudado:   float
    limite_credito:   float
    etapa_cobranza:   str
    gestor_asignado:  str
    gestionado_humano: bool
    intentos_fallidos: int

    ultimos_contactos:    List[ContactoReciente] = Field(default_factory=list)
    ultima_accion_agente: Optional[str]          = None
    escalaciones_activas: List[EscalacionActiva] = Field(default_factory=list)
    propuestas_activas:   List[PropuestaActiva]  = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Inputs de tools — generan el JSON Schema que LangGraph expone al LLM
# ---------------------------------------------------------------------------

class RegistrarContactoInput(BaseModel):
    cliente_id: str             = Field(..., description="ID del cliente en el CRM (ej. CLI006)")
    canal:      CanalContacto   = Field(..., description="Canal usado: sms | llamada | email | whatsapp")
    resultado:  ResultadoContacto = Field(..., description="Resultado: contesto | no_contesto | promesa_pago | rechazo")
    notas:      str             = Field(default="", description="Notas adicionales del contacto (opcional)")


class ActualizarEstadoInput(BaseModel):
    cliente_id:  str            = Field(..., description="ID del cliente en el CRM")
    nueva_etapa: EtapaCobranza  = Field(..., description="Nueva etapa (solo avance): temprana | administrativa | judicial | castigo")
    motivo:      str            = Field(..., description="Justificación del cambio de etapa, mínimo 10 caracteres")


class EscalarCasoInput(BaseModel):
    cliente_id: str             = Field(..., description="ID del cliente en el CRM")
    nivel:      NivelEscalacion = Field(..., description="Nivel de escalación: supervisor | legal | castigo")
    motivo:     str             = Field(..., description="Motivo detallado que justifica la escalación")


class GenerarPropuestaInput(BaseModel):
    cliente_id:      str          = Field(..., description="ID del cliente en el CRM")
    tipo:            TipoPropuesta = Field(..., description="Tipo: quita | plan_pagos | refinanciamiento")
    monto_propuesto: float        = Field(..., gt=0, description="Monto en MXN. No debe superar la deuda vigente en más de 10%.")
    plazo_meses:     int          = Field(..., ge=1, le=60, description="Plazo en meses (1-60)")


# ---------------------------------------------------------------------------
# Outputs de tools
# ---------------------------------------------------------------------------

class ContactoResult(BaseModel):
    ok:          bool
    contacto_id: str
    mensaje:     str


class UpdateResult(BaseModel):
    ok:             bool
    etapa_anterior: str
    etapa_nueva:    str
    mensaje:        str


class EscalacionResult(BaseModel):
    ok:               bool
    escalacion_id:    str
    gestionado_humano: bool
    mensaje:          str


class PropuestaResult(BaseModel):
    ok:           bool
    propuesta_id: str
    mensaje:      str


# ---------------------------------------------------------------------------
# Output estructurado del nodo evaluar_riesgo
# ---------------------------------------------------------------------------

class EvaluacionRiesgo(BaseModel):
    """
    Output estructurado del nodo evaluar_riesgo.
    El LLM lo produce vía with_structured_output — garantiza tipado correcto
    sin parsing manual de JSON.
    """
    urgencia: str = Field(
        ...,
        description="Nivel de urgencia de la gestión: 'alta', 'media' o 'baja'",
    )
    etapa_recomendada: Optional[str] = Field(
        default=None,
        description="Etapa a la que debería avanzar el cliente, o null si debe mantenerse",
    )
    accion_sugerida: str = Field(
        ...,
        description="Descripción concreta de la acción recomendada para el agente",
    )
    restricciones_normativas: List[str] = Field(
        default_factory=list,
        description="Restricciones normativas CNBV aplicables al caso",
    )
    razonamiento: str = Field(
        ...,
        description="Explicación detallada del análisis y la recomendación",
    )


# ---------------------------------------------------------------------------
# Reglas de pausa para modo autónomo
# ---------------------------------------------------------------------------
# Diseño: reglas explícitas en lugar de un float de "confianza" fabricado
# por el LLM. Más transparente, más fácil de auditar y defender.
#
# Si cualquier regla se cumple, el agente genera la recomendación pero
# establece requiere_confirmacion=True en el estado del grafo. Tab 6
# muestra un botón de aprobación antes de ejecutar la acción.

_REGLAS: List[Callable[[ClienteContextResumen], bool]] = [
    lambda ctx: ctx.etapa_cobranza == EtapaCobranza.JUDICIAL.value,
    lambda ctx: ctx.monto_adeudado > 50_000,
    lambda ctx: ctx.intentos_fallidos >= 5,
]


def evaluar_pausa(contexto: ClienteContextResumen) -> bool:
    """
    Retorna True si el agente en modo autónomo debe pausar y pedir
    confirmación humana antes de ejecutar la acción decidida.

    Reglas activas:
      1. Etapa judicial (riesgo legal)
      2. Monto adeudado > $50,000 MXN (impacto financiero alto)
      3. 5+ intentos de contacto fallidos (patrón de no-contacto)
    """
    return any(regla(contexto) for regla in _REGLAS)
