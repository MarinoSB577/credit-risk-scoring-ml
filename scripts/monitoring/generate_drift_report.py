"""
generate_drift_report.py
Calcula PSI por variable y genera reporte visual con Evidently 0.7.x.
Uso:
  python scripts/monitoring/generate_drift_report.py --escenario leve
  python scripts/monitoring/generate_drift_report.py --escenario severo
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Rutas
RUTA_BASE = Path(__file__).parent.parent.parent
RUTA_REF = RUTA_BASE / 'data' / 'processed' / 'df_lgbm.csv'
RUTA_PROD = RUTA_BASE / 'data' / 'production'
RUTA_REPORTES = RUTA_BASE / 'reports' / 'drift'

# Variables numéricas del modelo
FEATURES_MODELO = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_CREDIT',
    'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION',
    'DAYS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
    'HOUR_APPR_PROCESS_START', 'OWN_CAR_AGE', 'CNT_CHILDREN',
    'CNT_FAM_MEMBERS', 'TOTALAREA_MODE', 'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
    'FLAG_EMAIL', 'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
]


def calcular_psi(ref: pd.Series,
                 prod: pd.Series,
                 bins: int = 10) -> float:
    """Calcula PSI entre distribución de referencia y producción."""
    breakpoints = np.nanpercentile(ref.dropna(),
                                   np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    ref_counts = np.histogram(ref.dropna(), bins=breakpoints)[0]
    prod_counts = np.histogram(prod.dropna(), bins=breakpoints)[0]

    ref_pct = ref_counts / len(ref.dropna())
    prod_pct = prod_counts / len(prod.dropna())

    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    prod_pct = np.where(prod_pct == 0, 0.0001, prod_pct)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi), 4)


def clasificar_psi(psi: float) -> str:
    if psi < 0.10:
        return "✅ Estable"
    elif psi < 0.25:
        return "⚠️  Moderado"
    else:
        return "❌ Drift severo"


def generar_reporte_evidently(df_ref, df_prod, cols, escenario):
    """Genera reporte HTML con Evidently 0.7.x."""
    try:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset

        definition = DataDefinition(numerical_columns=cols)

        ref_dataset = Dataset.from_pandas(
            df_ref[cols].sample(5000, random_state=42),
            data_definition=definition
        )
        prod_dataset = Dataset.from_pandas(
            df_prod[cols],
            data_definition=definition
        )

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(ref_dataset, prod_dataset)

        RUTA_REPORTES.mkdir(parents=True, exist_ok=True)
        ruta_html = RUTA_REPORTES / f'drift_report_{escenario}.html'
        result.save_html(str(ruta_html))
        print(f"✅ Reporte Evidently guardado en: {ruta_html}")

    except Exception as e:
        print(f"⚠️  Reporte Evidently no generado: {e}")
        print("    El análisis PSI es válido independientemente.")


def main(escenario: str):
    print(f"\n{'='*55}")
    print(f"  REPORTE DE MONITOREO — Escenario: {escenario.upper()}")
    print(f"{'='*55}\n")

    df_ref = pd.read_csv(RUTA_REF)
    archivo_prod = f'produccion_drift_{escenario}.csv'
    df_prod = pd.read_csv(RUTA_PROD / archivo_prod)

    print(f"Referencia:  {df_ref.shape[0]:,} filas")
    print(f"Producción:  {df_prod.shape[0]:,} filas\n")

    variables_disponibles = [
        f for f in FEATURES_MODELO
        if f in df_ref.columns and f in df_prod.columns
        and df_ref[f].dtype in ['float64', 'int64']
    ]

    resultados = []
    for var in variables_disponibles:
        psi = calcular_psi(df_ref[var], df_prod[var])
        resultados.append({
            'Variable': var,
            'PSI': psi,
            'Estado': clasificar_psi(psi)
        })

    df_psi = pd.DataFrame(resultados).sort_values(
        'PSI', ascending=False
    )

    print("PSI por variable (top 15):")
    print("-" * 50)
    for _, row in df_psi.head(15).iterrows():
        print(
            f"  {row['Variable']:<35} "
            f"{row['PSI']:.4f}  {row['Estado']}"
        )

    psi_criticos = (df_psi['PSI'] >= 0.25).sum()
    psi_moderados = (
        (df_psi['PSI'] >= 0.10) & (df_psi['PSI'] < 0.25)
    ).sum()
    psi_estables = (df_psi['PSI'] < 0.10).sum()

    print(f"\nResumen:")
    print(f"  ❌ Drift severo  (PSI ≥ 0.25): {psi_criticos} variables")
    print(f"  ⚠️  Moderado     (PSI 0.10-0.25): {psi_moderados} variables")
    print(f"  ✅ Estable       (PSI < 0.10): {psi_estables} variables")

    print(f"\nRecomendación:")
    if psi_criticos >= 3:
        print("  🔴 REENTRENAR el modelo — drift severo detectado")
    elif psi_criticos >= 1 or psi_moderados >= 5:
        print("  🟡 MONITOREAR de cerca — cambios moderados detectados")
    else:
        print("  🟢 Modelo VÁLIDO — distribución estable")

    # Guardar CSV con resultados
    RUTA_REPORTES.mkdir(parents=True, exist_ok=True)
    ruta_csv = RUTA_REPORTES / f'psi_report_{escenario}.csv'
    df_psi.to_csv(ruta_csv, index=False)
    print(f"\n✅ Tabla PSI guardada en: {ruta_csv}")

    # Generar reporte Evidently
    print(f"\nGenerando reporte visual Evidently...")
    generar_reporte_evidently(
        df_ref, df_prod, variables_disponibles, escenario
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--escenario',
        choices=['leve', 'severo'],
        default='leve'
    )
    args = parser.parse_args()
    main(args.escenario)