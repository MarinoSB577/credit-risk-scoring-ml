"""
simulate_production_data.py
Simula datos de producción con drift controlado para testing de monitoreo.
Genera dos escenarios:
  - Escenario A: drift leve (PSI ~0.08) — modelo debería seguir válido
  - Escenario B: drift severo (PSI ~0.30) — modelo debería reentrenarse
"""

import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
np.random.seed(SEED)

RUTA_DATOS = Path(__file__).parent.parent.parent / 'data' / 'processed'
RUTA_SALIDA = Path(__file__).parent.parent.parent / 'data' / 'production'


def simular_produccion(df_ref: pd.DataFrame,
                       n: int = 5000,
                       drift_factor: float = 0.0) -> pd.DataFrame:
    """
    Genera datos de producción simulados.

    drift_factor=0.0  → sin drift (igual al entrenamiento)
    drift_factor=0.5  → drift leve
    drift_factor=2.0  → drift severo
    """
    df_prod = df_ref.sample(n=n, random_state=SEED).copy()

    # Aplicar drift a las variables más importantes
    if drift_factor > 0:

        # EXT_SOURCE_2 — score externo baja (peores solicitantes)
        if 'EXT_SOURCE_2' in df_prod.columns:
            ruido = np.random.normal(
                loc=-0.08 * drift_factor,
                scale=0.05 * drift_factor,
                size=len(df_prod)
            )
            df_prod['EXT_SOURCE_2'] = (
                df_prod['EXT_SOURCE_2'] + ruido
            ).clip(0, 1)

        # EXT_SOURCE_3 — score externo baja
        if 'EXT_SOURCE_3' in df_prod.columns:
            ruido = np.random.normal(
                loc=-0.06 * drift_factor,
                scale=0.04 * drift_factor,
                size=len(df_prod)
            )
            df_prod['EXT_SOURCE_3'] = (
                df_prod['EXT_SOURCE_3'] + ruido
            ).clip(0, 1)

        # DAYS_BIRTH — solicitantes más jóvenes
        if 'DAYS_BIRTH' in df_prod.columns:
            ruido = np.random.normal(
                loc=500 * drift_factor,
                scale=200 * drift_factor,
                size=len(df_prod)
            )
            df_prod['DAYS_BIRTH'] = df_prod['DAYS_BIRTH'] + ruido

        # AMT_INCOME_TOTAL — ingresos más bajos
        if 'AMT_INCOME_TOTAL' in df_prod.columns:
            factor = 1 - (0.10 * drift_factor)
            df_prod['AMT_INCOME_TOTAL'] = (
                df_prod['AMT_INCOME_TOTAL'] * factor
            )

        # DAYS_EMPLOYED — menos antigüedad laboral
        if 'DAYS_EMPLOYED' in df_prod.columns:
            mask = df_prod['DAYS_EMPLOYED'] < 0
            ruido = np.random.normal(
                loc=300 * drift_factor,
                scale=100 * drift_factor,
                size=mask.sum()
            )
            df_prod.loc[mask, 'DAYS_EMPLOYED'] = (
                df_prod.loc[mask, 'DAYS_EMPLOYED'] + ruido
            )

    return df_prod


if __name__ == '__main__':
    print("Cargando datos de referencia...")
    df_ref = pd.read_csv(RUTA_DATOS / 'df_lgbm.csv')
    print(f"Referencia: {df_ref.shape}")

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    # Escenario A — drift leve
    df_leve = simular_produccion(df_ref, n=5000, drift_factor=0.5)
    df_leve.to_csv(RUTA_SALIDA / 'produccion_drift_leve.csv', index=False)
    print("✅ Escenario A (drift leve) guardado")

    # Escenario B — drift severo
    df_severo = simular_produccion(df_ref, n=5000, drift_factor=2.0)
    df_severo.to_csv(
        RUTA_SALIDA / 'produccion_drift_severo.csv', index=False
    )
    print("✅ Escenario B (drift severo) guardado")

    print(f"\nArchivos en: {RUTA_SALIDA}")