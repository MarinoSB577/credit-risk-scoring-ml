import argparse
import json
from pathlib import Path

# ─────────────────────────────────────────────
# ARGUMENTOS
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--input_metrics", type=str, help="Ruta a las métricas del modelo nuevo")
parser.add_argument("--output_evaluation", type=str, help="Ruta donde guardar resultado de evaluación")
parser.add_argument("--auc_threshold", type=float, default=0.75,
                    help="AUC mínimo para aprobar el modelo")
# auc_threshold es el gate de calidad — si el modelo nuevo
# no supera este umbral, no se registra en Azure ML.
# Usamos 0.75 como mínimo aceptable (nuestro modelo actual tiene 0.768)
args = parser.parse_args()

# ─────────────────────────────────────────────
# CARGAR MÉTRICAS
# ─────────────────────────────────────────────

metrics_file = Path(args.input_metrics) / "metrics.json"
with open(metrics_file, "r") as f:
    metrics = json.load(f)

auc_new = metrics["auc_test"]
print(f"AUC modelo nuevo: {auc_new:.4f}")
print(f"AUC umbral mínimo: {args.auc_threshold:.4f}")

# ─────────────────────────────────────────────
# EVALUAR SI EL MODELO APRUEBA
# ─────────────────────────────────────────────

approved = auc_new >= args.auc_threshold
# True si el modelo nuevo supera el umbral de calidad

result = {
    "auc_new": auc_new,
    "auc_threshold": args.auc_threshold,
    "approved": approved,
    "reason": (
        f"AUC {auc_new:.4f} >= umbral {args.auc_threshold:.4f} — APROBADO"
        if approved else
        f"AUC {auc_new:.4f} < umbral {args.auc_threshold:.4f} — RECHAZADO"
    )
}

print(f"Resultado: {result['reason']}")

# ─────────────────────────────────────────────
# GUARDAR RESULTADO
# ─────────────────────────────────────────────

output_path = Path(args.output_evaluation)
output_path.mkdir(parents=True, exist_ok=True)

with open(output_path / "evaluation.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Evaluación guardada en: {output_path / 'evaluation.json'}")

# Exit code 1 si el modelo no aprueba —
# Azure ML Pipeline detecta esto y detiene los pasos siguientes
if not approved:
    raise SystemExit(f"Modelo rechazado: {result['reason']}")