from pathlib import Path
import joblib

def load_model(model_name: str):
    # Ruta ABSOLUTA al directorio de modelos
    PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Subir 4 niveles desde backend/app/models/
    MODELS_DIR = PROJECT_ROOT / "ml_model/saved_models"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    
    return joblib.load(model_path)