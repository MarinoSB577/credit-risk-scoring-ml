"""
src/crm — Paquete del CRM SQLite para el Agente de Cobranza Level 3.

Interfaz pública:
    inicializar_crm()       → crea tablas y siembra datos sintéticos
    consultar_cliente()     → función de servicio para enriquecer_contexto
    TOOLS_AGENTE            → lista de tools para registrar en LangGraph
    evaluar_pausa()         → reglas explícitas de confirmación en modo autónomo
    ejecutar_agente()       → invoca el grafo para un cliente
    ejecutar_lote_ews()     → procesa una lista de clientes en serie
"""

from .init_crm import inicializar_crm, get_db_path
from .crm_tools import consultar_cliente, TOOLS_AGENTE
from .schemas import (
    EtapaCobranza,
    ORDEN_ETAPAS,
    ClienteContextResumen,
    EvaluacionRiesgo,
    evaluar_pausa,
)
from .agente_level3 import ejecutar_agente, ejecutar_lote_ews, ResultadoAgente, grafo_cobranza

__all__ = [
    "inicializar_crm",
    "get_db_path",
    "consultar_cliente",
    "TOOLS_AGENTE",
    "EtapaCobranza",
    "ORDEN_ETAPAS",
    "ClienteContextResumen",
    "EvaluacionRiesgo",
    "evaluar_pausa",
    "ejecutar_agente",
    "ejecutar_lote_ews",
    "ResultadoAgente",
    "grafo_cobranza",
]
