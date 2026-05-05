import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_model", type=str)
parser.add_argument("--input_evaluation", type=str)
parser.add_argument("--model_name", type=str, default="lightgbm-credit-risk")
parser.add_argument("--subscription_id", type=str)
parser.add_argument("--resource_group", type=str)
parser.add_argument("--workspace_name", type=str)
args = parser.parse_args()

# Leer evaluación
evaluation_file = Path(args.input_evaluation) / "evaluation.json"
with open(evaluation_file, "r") as f:
    evaluation = json.load(f)

print(f"Resultado: {evaluation['reason']}")

# Guardar señal para registro posterior desde notebook
output_path = Path(args.input_model).parent / "register_signal"
output_path.mkdir(parents=True, exist_ok=True)

signal = {
    "model_path": str(Path(args.input_model) / "model.pkl"),
    "model_name": args.model_name,
    "evaluation": evaluation,
    "approved": evaluation["approved"]
}

with open(output_path / "signal.json", "w") as f:
    json.dump(signal, f, indent=2)

print(f"✅ Señal de registro guardada")
print(f"Modelo aprobado: {evaluation['approved']}")
print(f"AUC: {evaluation['auc_new']:.4f}")