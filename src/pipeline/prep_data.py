import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# ARGUMENTOS
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str, help="Ruta al dataset de entrada")
parser.add_argument("--output_data", type=str, help="Ruta donde guardar datos preparados")
args = parser.parse_args()
# Azure ML pasa los argumentos automáticamente cuando
# ejecuta este script dentro del pipeline

# ─────────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────────

print("Cargando datos...")
df = pd.read_csv(Path(args.input_data) / "df_lgbm.csv")
print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

# ─────────────────────────────────────────────
# LIMPIEZA BÁSICA
# ─────────────────────────────────────────────

print("Aplicando limpieza...")

# Eliminar duplicados
n_before = len(df)
df = df.drop_duplicates()
n_after = len(df)
print(f"Duplicados eliminados: {n_before - n_after:,}")

# Eliminar filas con target nulo
df = df.dropna(subset=["TARGET"])
print(f"Filas con TARGET nulo eliminadas")

# Rellenar nulos numéricos con mediana
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "TARGET"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print(f"Nulos numéricos rellenados con mediana")

# ─────────────────────────────────────────────
# GUARDAR DATOS PREPARADOS
# ─────────────────────────────────────────────

output_path = Path(args.output_data)
output_path.mkdir(parents=True, exist_ok=True)
# Crea el directorio de salida si no existe

output_file = output_path / "df_prepared.csv"
df.to_csv(output_file, index=False)

print(f"Datos preparados guardados en: {output_file}")
print(f"Shape final: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"Distribución TARGET:\n{df['TARGET'].value_counts(normalize=True).round(3)}")