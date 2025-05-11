from flask import Blueprint, request, jsonify, send_from_directory
import os
import pandas as pd
from .models.load_models import load_model

bp = Blueprint('api', __name__)

# Ruta al frontend
FRONTEND_FOLDER = os.path.join(os.path.dirname(__file__), '../../frontend')

@bp.route('/', methods=['GET'])
def serve_frontend():
    return send_from_directory(FRONTEND_FOLDER, 'index.html')

@bp.route('/<path:path>', methods=['GET'])
def serve_static(path):
    return send_from_directory(FRONTEND_FOLDER, path)

@bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 1. Cargar el pipeline completo (incluye preprocesador)
        model_name = data.get("model", "xgboost")
        pipeline = load_model(model_name)
        
        # 2. Obtener características esperadas del pipeline
        expected_features = pipeline.named_steps['preprocessor'].feature_names_in_
        
        # 3. Validar campos faltantes
        missing = [feat for feat in expected_features if feat not in data]
        if missing:
            return jsonify({"error": f"Campos faltantes: {missing}"}), 400
        
        # 4. Crear DataFrame con orden correcto
        df = pd.DataFrame([data], columns=expected_features)
        
        # 5. Predecir usando el pipeline (preprocesa automáticamente)
        probability = pipeline.predict_proba(df)[0][1]
        
        return jsonify({
            "risk": float(round(probability, 4)),
            "model": model_name,
            "message": "Success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@bp.route('/')
def home():
    return "API de Predicción de Deserción Académica - Usa /predict"