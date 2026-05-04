import os
import json
import mlflow
import pandas as pd

# ─────────────────────────────────────────────
# INIT — se ejecuta UNA vez al iniciar el endpoint
# ─────────────────────────────────────────────

def init():
    """
    Carga el modelo desde el directorio asignado por Azure ML.
    Azure ML inyecta la ruta del modelo en la variable de entorno
    AZUREML_MODEL_DIR al momento de iniciar el contenedor.
    """
    global model
    # global permite que run() acceda al modelo cargado

    model_path = os.path.join(
        os.environ["AZUREML_MODEL_DIR"],
        "model"
    )
    # AZUREML_MODEL_DIR apunta al directorio donde Azure ML
    # descargó los artefactos del modelo registrado.
    # La subcarpeta "model" contiene el modelo MLflow.

    model = mlflow.pyfunc.load_model(model_path)
    # mlflow.pyfunc proporciona una interfaz unificada
    # para cargar modelos independientemente del framework.
    # Funciona con LightGBM, sklearn, XGBoost, etc.

    print("Modelo cargado correctamente")


# ─────────────────────────────────────────────
# RUN — se ejecuta en CADA llamada a la API
# ─────────────────────────────────────────────

def run(raw_data):
    """
    Recibe datos JSON, ejecuta predicción y devuelve resultado.

    Args:
        raw_data: string JSON con los datos del solicitante
    Returns:
        JSON con probabilidad de incumplimiento
    """
    try:
        data = json.loads(raw_data)
        # Convierte el string JSON a diccionario Python

        df = pd.DataFrame(data)
        # Convierte el diccionario a DataFrame —
        # formato que espera el modelo LightGBM

        predictions = model.predict(df)
        # Ejecuta la predicción sobre el DataFrame.
        # Devuelve probabilidades de incumplimiento (0-1)

        return json.dumps({
            "probabilidad_incumplimiento": predictions.tolist(),
            "status": "ok"
        })
        # Serializa el resultado como JSON para la respuesta HTTP

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error"
        })
        # Captura cualquier error y lo devuelve estructurado
        # en lugar de dejar que el endpoint falle silenciosamente