import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ─────────────────────────────────────────────
# ARGUMENTOS
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str, help="Ruta a los datos preparados")
parser.add_argument("--output_model", type=str, help="Ruta donde guardar el modelo entrenado")
parser.add_argument("--output_metrics", type=str, help="Ruta donde guardar métricas")
parser.add_argument("--n_estimators", type=int, default=568)
parser.add_argument("--learning_rate", type=float, default=0.05)
parser.add_argument("--num_leaves", type=int, default=31)
args = parser.parse_args()

# ─────────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────────

print("Cargando datos preparados...")
df = pd.read_csv(Path(args.input_data) / "df_prepared.csv")
print(f"Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")

# ─────────────────────────────────────────────
# PREPARAR FEATURES Y TARGET
# ─────────────────────────────────────────────

TARGET = "TARGET"
EXCLUDE = ["SK_ID_CURR", TARGET]
features = [c for c in df.columns if c not in EXCLUDE]
X = df[features]
y = df[TARGET]

print(f"Features: {len(features)}")
print(f"Positivos (mora): {y.sum():,} ({y.mean():.1%})")

# ─────────────────────────────────────────────
# SPLIT TRAIN/TEST
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# ENTRENAR MODELO
# ─────────────────────────────────────────────

params = {
    "n_estimators": args.n_estimators,
    "learning_rate": args.learning_rate,
    "num_leaves": args.num_leaves,
    "objective": "binary",
    "metric": "auc",
    "is_unbalance": True,
    "random_state": 42,
    "n_jobs": -1
}

print(f"Entrenando LightGBM...")
model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# EVALUAR
# ─────────────────────────────────────────────

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC en test: {auc:.4f}")

# ─────────────────────────────────────────────
# GUARDAR MODELO Y MÉTRICAS
# ─────────────────────────────────────────────

output_model_path = Path(args.output_model)
output_model_path.mkdir(parents=True, exist_ok=True)

# Guardar modelo con joblib — sin dependencia de MLflow
import joblib
joblib.dump(model, output_model_path / "model.pkl")
print(f"Modelo guardado en: {output_model_path / 'model.pkl'}")

metrics = {
    "auc_test": auc,
    "n_train": len(X_train),
    "n_test": len(X_test)
}
output_metrics_path = Path(args.output_metrics)
output_metrics_path.mkdir(parents=True, exist_ok=True)
with open(output_metrics_path / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Métricas guardadas: {metrics}")