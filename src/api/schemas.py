from pydantic import BaseModel
from typing import Optional


class SolicitudCredito(BaseModel):
    """Datos de entrada para evaluar una solicitud de crédito."""
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = 0.5
    EXT_SOURCE_3: Optional[float] = None
    DAYS_BIRTH: float = -15000
    DAYS_EMPLOYED: float = -2000
    AMT_CREDIT: float = 500000
    AMT_INCOME_TOTAL: float = 150000
    AMT_ANNUITY: Optional[float] = 25000
    AMT_GOODS_PRICE: Optional[float] = None
    DAYS_ID_PUBLISH: float = -2000
    DAYS_REGISTRATION: float = -5000
    DAYS_LAST_PHONE_CHANGE: float = -500
    REGION_POPULATION_RELATIVE: float = 0.02
    HOUR_APPR_PROCESS_START: float = 12.0
    OWN_CAR_AGE: Optional[float] = None
    CNT_CHILDREN: float = 0
    CNT_FAM_MEMBERS: Optional[float] = 2
    TOTALAREA_MODE: Optional[float] = None
    FLAG_OWN_CAR: float = 0
    FLAG_OWN_REALTY: float = 1
    FLAG_WORK_PHONE: float = 0
    FLAG_PHONE: float = 0
    FLAG_EMAIL: float = 0
    REG_CITY_NOT_LIVE_CITY: float = 0
    REG_CITY_NOT_WORK_CITY: float = 0
    LIVE_CITY_NOT_WORK_CITY: float = 0


class RespuestaCredito(BaseModel):
    """Resultado de la evaluación crediticia."""
    probabilidad_incumplimiento: float
    score_crediticio: int
    decision: str
    nivel_riesgo: str
    top_variables: list
    mensaje: str