import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Ruta al modelo dentro del contenedor (/app/mlruns/...)
# En el contenedor, mlruns/ se copia a /app/mlruns/
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'mlruns',
    '521656391598123434',
    'models',
    'm-4b3e2786ebbf4f6982ed89b1009b26b8',
    'artifacts',
    'model.pkl'
)

_model = None
_feature_names = None


def get_model():
    """Carga el modelo LightGBM desde joblib. Singleton."""
    global _model, _feature_names

    if _model is not None:
        return _model, _feature_names

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo no encontrado en: {MODEL_PATH}"
        )

    _model = joblib.load(MODEL_PATH)
    _feature_names = list(_model.feature_name_)

    return _model, _feature_names