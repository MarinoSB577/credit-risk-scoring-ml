import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# La ruta se puede sobreescribir con variable de entorno MODEL_PATH
# Local: dashboard/models/model.pkl (relativo a la raíz del proyecto)
# Docker: /app/dashboard/models/model.pkl
_DEFAULT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..',
    'dashboard', 'models', 'model.pkl'
))

MODEL_PATH = os.environ.get('MODEL_PATH', _DEFAULT_PATH)

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