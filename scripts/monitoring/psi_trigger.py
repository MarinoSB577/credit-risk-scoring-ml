"""
psi_trigger.py
─────────────────────────────────────────────────────────────
Conecta el monitoreo PSI con el pipeline de reentrenamiento
de Azure ML. Cierra el Gap 1 del ciclo MLOps.

Flujo:
  1. Calcula PSI entre datos de referencia y producción
  2. Evalúa si hay drift severo (PSI ≥ 0.25 en N variables)
  3. Si hay drift → dispara el pipeline de Azure ML
  4. Registra el evento en log
  5. Genera archivo de alerta

Uso:
  python scripts/monitoring/psi_trigger.py --escenario leve
  python scripts/monitoring/psi_trigger.py --escenario severo
  python scripts/monitoring/psi_trigger.py --escenario severo --dry-run

Parámetros:
  --escenario : leve | severo
  --dry-run   : simula el trigger sin ejecutar el pipeline real
  --umbral-psi: umbral de PSI para considerar drift severo (default: 0.25)
  --min-vars  : mínimo de variables con drift para disparar (default: 3)

Autor: Marín Serrato Barrios
Fecha: Mayo 2026
─────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────
RUTA_BASE      = Path(__file__).parent.parent.parent
RUTA_REF       = RUTA_BASE / 'data' / 'processed' / 'df_lgbm.csv'
RUTA_PROD      = RUTA_BASE / 'data' / 'production'
RUTA_REPORTES  = RUTA_BASE / 'reports' / 'drift'
RUTA_LOGS      = RUTA_BASE / 'reports' / 'logs'
RUTA_ALERTAS   = RUTA_BASE / 'reports' / 'alertas'

# ── Variables numéricas del modelo ────────────────────────────
FEATURES_MODELO = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_CREDIT',
    'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION',
    'DAYS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
    'OWN_CAR_AGE', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
    'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
]

# ── Configurar logging ────────────────────────────────────────
def configurar_logging():
    RUTA_LOGS.mkdir(parents=True, exist_ok=True)
    fecha  = datetime.now().strftime('%Y%m%d')
    log_path = RUTA_LOGS / f'psi_trigger_{fecha}.log'

    logging.basicConfig(
        level   = logging.INFO,
        format  = '%(asctime)s | %(levelname)s | %(message)s',
        handlers= [
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('psi_trigger')


# ── Función PSI ───────────────────────────────────────────────
def calcular_psi(ref: pd.Series,
                 prod: pd.Series,
                 bins: int = 10) -> float:
    """Calcula PSI entre distribución de referencia y producción."""
    breakpoints = np.nanpercentile(
        ref.dropna(), np.linspace(0, 100, bins + 1)
    )
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts  = np.histogram(ref.dropna(),  bins=breakpoints)[0]
    prod_counts = np.histogram(prod.dropna(), bins=breakpoints)[0]

    ref_pct  = ref_counts  / max(len(ref.dropna()),  1)
    prod_pct = prod_counts / max(len(prod.dropna()), 1)

    ref_pct  = np.where(ref_pct  == 0, 0.0001, ref_pct)
    prod_pct = np.where(prod_pct == 0, 0.0001, prod_pct)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi), 4)


def clasificar_psi(psi: float) -> str:
    if psi < 0.10:  return "ESTABLE"
    elif psi < 0.25: return "MODERADO"
    else:            return "DRIFT_SEVERO"


# ── Trigger Azure ML ──────────────────────────────────────────
def disparar_pipeline_azure(logger, dry_run: bool = False) -> bool:
    """
    Dispara el pipeline de reentrenamiento en Azure ML.

    En modo dry_run simula el trigger sin ejecutar nada real.
    En modo real usa el SDK de Azure ML para lanzar el pipeline.

    Retorna True si el trigger fue exitoso.
    """
    if dry_run:
        logger.info("🔵 [DRY-RUN] Simulando trigger de Azure ML Pipeline")
        logger.info("🔵 [DRY-RUN] Pipeline que se dispararía:")
        logger.info("🔵 [DRY-RUN]   prep_data → train → evaluate → register")
        logger.info("🔵 [DRY-RUN]   Gate: AUC ≥ 0.75")
        logger.info("🔵 [DRY-RUN] En producción real esto ejecutaría:")
        logger.info("🔵 [DRY-RUN]   ml_client.jobs.create_or_update(pipeline_job)")
        return True

    # ── Intento de trigger real con Azure ML SDK ──────────────
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        logger.info("Conectando a Azure ML Workspace...")

        # Credenciales — en producción usar Managed Identity
        credential = DefaultAzureCredential()

        # Configuración del workspace
        # En producción estos valores vendrían de variables de entorno
        subscription_id  = os.environ.get('AZURE_SUBSCRIPTION_ID', '')
        resource_group   = os.environ.get('AZURE_RESOURCE_GROUP', '')
        workspace_name   = os.environ.get('AZURE_WORKSPACE_NAME',
                                          'ws-credit-risk-ml')

        if not all([subscription_id, resource_group]):
            logger.warning(
                "Variables de entorno Azure no configuradas. "
                "Usando modo simulado."
            )
            return disparar_pipeline_azure(logger, dry_run=True)

        ml_client = MLClient(
            credential      = credential,
            subscription_id = subscription_id,
            resource_group_name = resource_group,
            workspace_name  = workspace_name
        )

        # Obtener el pipeline registrado
        pipeline_name = "credit-risk-retraining-pipeline"
        logger.info(f"Disparando pipeline: {pipeline_name}")

        # Crear job del pipeline
        pipeline_job = ml_client.pipelines.get(pipeline_name)
        submitted_job = ml_client.jobs.create_or_update(pipeline_job)

        logger.info(
            f"✅ Pipeline disparado exitosamente — "
            f"Job ID: {submitted_job.name}"
        )
        return True

    except ImportError:
        logger.warning(
            "azure-ai-ml no instalado. "
            "Usando modo simulado (dry-run)."
        )
        return disparar_pipeline_azure(logger, dry_run=True)

    except Exception as e:
        logger.error(f"Error disparando pipeline Azure ML: {e}")
        return False


# ── Generar alerta ────────────────────────────────────────────
def generar_alerta(logger,
                   resultado_psi: dict,
                   trigger_exitoso: bool,
                   dry_run: bool):
    """
    Genera un archivo JSON de alerta con el resumen del evento.
    En producción este archivo sería leído por n8n o un webhook
    para enviar notificaciones por email o Slack.
    """
    RUTA_ALERTAS.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    alerta = {
        'timestamp'        : datetime.now().isoformat(),
        'tipo'             : 'DRIFT_DETECTADO',
        'severity'         : 'HIGH' if resultado_psi['n_criticas'] >= 3
                             else 'MEDIUM',
        'variables_criticas': resultado_psi['n_criticas'],
        'variables_moderadas': resultado_psi['n_moderadas'],
        'psi_maximo'       : resultado_psi['psi_maximo'],
        'variable_mayor_drift': resultado_psi['variable_mayor_drift'],
        'trigger_pipeline' : trigger_exitoso,
        'dry_run'          : dry_run,
        'recomendacion'    : (
            'Pipeline de reentrenamiento disparado automáticamente.'
            if trigger_exitoso and not dry_run
            else 'Revisión manual requerida.'
        ),
        'notificacion'     : {
            'email'  : 'riesgo@institucion.com.mx',
            'slack'  : '#alertas-mlops',
            'mensaje': (
                f"⚠️ DRIFT DETECTADO en modelo de credit scoring. "
                f"{resultado_psi['n_criticas']} variables con PSI crítico. "
                f"Variable con mayor drift: "
                f"{resultado_psi['variable_mayor_drift']} "
                f"(PSI={resultado_psi['psi_maximo']:.4f}). "
                f"Pipeline de reentrenamiento: "
                f"{'DISPARADO' if trigger_exitoso else 'PENDIENTE'}."
            )
        }
    }

    ruta_alerta = RUTA_ALERTAS / f'alerta_{timestamp}.json'
    with open(ruta_alerta, 'w', encoding='utf-8') as f:
        json.dump(alerta, f, ensure_ascii=False, indent=2)

    logger.info(f"📋 Alerta guardada en: {ruta_alerta}")
    return alerta


# ── Función principal ─────────────────────────────────────────
def main(escenario: str,
         dry_run: bool,
         umbral_psi: float,
         min_vars: int):

    logger = configurar_logging()

    logger.info("=" * 60)
    logger.info("PSI TRIGGER — SISTEMA DE MONITOREO AUTOMÁTICO")
    logger.info("=" * 60)
    logger.info(f"Escenario    : {escenario.upper()}")
    logger.info(f"Dry-run      : {dry_run}")
    logger.info(f"Umbral PSI   : {umbral_psi}")
    logger.info(f"Min variables: {min_vars}")

    # ── Cargar datos ──────────────────────────────────────────
    logger.info("\n[1/4] Cargando datos...")

    if not RUTA_REF.exists():
        logger.error(f"Datos de referencia no encontrados: {RUTA_REF}")
        sys.exit(1)

    archivo_prod = f'produccion_drift_{escenario}.csv'
    ruta_prod    = RUTA_PROD / archivo_prod

    if not ruta_prod.exists():
        logger.error(f"Datos de producción no encontrados: {ruta_prod}")
        logger.error("Ejecuta primero: python scripts/monitoring/simulate_production_data.py")
        sys.exit(1)

    df_ref  = pd.read_csv(RUTA_REF)
    df_prod = pd.read_csv(ruta_prod)

    logger.info(f"  Referencia : {df_ref.shape[0]:,} filas")
    logger.info(f"  Producción : {df_prod.shape[0]:,} filas")

    # ── Calcular PSI ──────────────────────────────────────────
    logger.info("\n[2/4] Calculando PSI por variable...")

    variables_disponibles = [
        f for f in FEATURES_MODELO
        if f in df_ref.columns and f in df_prod.columns
        and df_ref[f].dtype in ['float64', 'int64']
    ]

    resultados = []
    for var in variables_disponibles:
        psi    = calcular_psi(df_ref[var], df_prod[var])
        estado = clasificar_psi(psi)
        resultados.append({
            'Variable': var, 'PSI': psi, 'Estado': estado
        })
        nivel = ("❌" if estado == "DRIFT_SEVERO"
                 else "⚠️" if estado == "MODERADO"
                 else "✅")
        logger.info(f"  {nivel} {var:<35} PSI={psi:.4f} [{estado}]")

    df_psi = pd.DataFrame(resultados).sort_values(
        'PSI', ascending=False
    )

    # Guardar CSV
    RUTA_REPORTES.mkdir(parents=True, exist_ok=True)
    ruta_csv = RUTA_REPORTES / f'psi_trigger_{escenario}.csv'
    df_psi.to_csv(ruta_csv, index=False)

    # ── Evaluar drift ─────────────────────────────────────────
    logger.info("\n[3/4] Evaluando nivel de drift...")

    n_criticas  = (df_psi['PSI'] >= umbral_psi).sum()
    n_moderadas = (
        (df_psi['PSI'] >= 0.10) & (df_psi['PSI'] < umbral_psi)
    ).sum()
    n_estables  = (df_psi['PSI'] < 0.10).sum()
    psi_maximo  = df_psi['PSI'].max()
    var_max     = df_psi.iloc[0]['Variable']

    resultado_psi = {
        'n_criticas'         : int(n_criticas),
        'n_moderadas'        : int(n_moderadas),
        'n_estables'         : int(n_estables),
        'psi_maximo'         : float(psi_maximo),
        'variable_mayor_drift': var_max
    }

    logger.info(f"  ❌ Drift severo  (PSI ≥ {umbral_psi}): "
                f"{n_criticas} variables")
    logger.info(f"  ⚠️  Moderado     (PSI 0.10-{umbral_psi}): "
                f"{n_moderadas} variables")
    logger.info(f"  ✅ Estable       (PSI < 0.10): "
                f"{n_estables} variables")
    logger.info(f"  📊 PSI máximo   : {psi_maximo:.4f} "
                f"({var_max})")

    # ── Decisión de trigger ───────────────────────────────────
    logger.info("\n[4/4] Evaluando trigger de reentrenamiento...")

    trigger_exitoso = False
    hay_drift_critico = n_criticas >= min_vars

    if hay_drift_critico:
        logger.warning(
            f"🔴 DRIFT CRÍTICO DETECTADO — "
            f"{n_criticas} variables superan PSI={umbral_psi}"
        )
        logger.info("   Disparando pipeline de reentrenamiento...")
        trigger_exitoso = disparar_pipeline_azure(logger, dry_run)

        if trigger_exitoso:
            logger.info(
                "✅ Pipeline disparado — el modelo se reentrenará "
                "automáticamente"
            )
        else:
            logger.error(
                "❌ Error al disparar el pipeline — "
                "revisión manual requerida"
            )
    elif n_criticas >= 1 or n_moderadas >= 5:
        logger.warning(
            f"🟡 DRIFT MODERADO — monitorear de cerca. "
            f"No se dispara el pipeline automáticamente."
        )
    else:
        logger.info(
            "🟢 Modelo VÁLIDO — distribución estable. "
            "No se requiere reentrenamiento."
        )

    # ── Generar alerta ────────────────────────────────────────
    if hay_drift_critico:
        alerta = generar_alerta(
            logger, resultado_psi, trigger_exitoso, dry_run
        )
        logger.info(
            f"\n📢 RESUMEN DE ALERTA:\n"
            f"   {alerta['notificacion']['mensaje']}"
        )

    # ── Resumen final ─────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 60)
    logger.info(f"  Variables analizadas : {len(resultados)}")
    logger.info(f"  Drift severo         : {n_criticas}")
    logger.info(f"  Drift moderado       : {n_moderadas}")
    logger.info(f"  Estables             : {n_estables}")
    logger.info(f"  Trigger pipeline     : "
                f"{'✅ SÍ' if trigger_exitoso else '❌ NO'}")
    logger.info(f"  CSV guardado en      : {ruta_csv}")

    return {
        'drift_critico'  : hay_drift_critico,
        'trigger_exitoso': trigger_exitoso,
        'n_criticas'     : int(n_criticas),
        'psi_maximo'     : float(psi_maximo)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PSI Trigger — Monitoreo automático con trigger MLOps'
    )
    parser.add_argument(
        '--escenario',
        choices=['leve', 'severo'],
        default='leve',
        help='Escenario de datos de producción a usar'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Simula el trigger sin ejecutar el pipeline real'
    )
    parser.add_argument(
        '--umbral-psi',
        type=float,
        default=0.25,
        help='Umbral PSI para considerar drift severo (default: 0.25)'
    )
    parser.add_argument(
        '--min-vars',
        type=int,
        default=3,
        help='Mínimo de variables con drift para disparar pipeline (default: 3)'
    )

    args = parser.parse_args()
    main(
        escenario  = args.escenario,
        dry_run    = args.dry_run,
        umbral_psi = args.umbral_psi,
        min_vars   = args.min_vars
    )