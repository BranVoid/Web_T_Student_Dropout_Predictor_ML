from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
import sys

# Añade la ruta al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend/app')))

from models.load_models import load_model  # Importación absoluta

app = Flask(__name__, 
            static_folder='frontend/static', 
            template_folder='frontend/templates')


# Configuración de rutas principales
@app.route('/')
def index():
    return render_template('index.html')  # Ahora busca en frontend/templates

# Ruta para archivos estáticos (CSS/JS)
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('frontend/static', path)

# ========================
# RUTAS DE PREDICCIÓN
# ========================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 1. Cargar el pipeline con preprocesador
        model_name = data.get("model", "xgboost")
        pipeline = load_model(model_name)
        
        # 2. Obtener características esperadas
        expected_features = pipeline.named_steps['preprocessor'].feature_names_in_
        
        # 3. Validar campos
        missing = [feat for feat in expected_features if feat not in data]
        if missing:
            return jsonify({"error": f"Campos faltantes: {missing}"}), 400
        
        # 4. Crear DataFrame con orden correcto
        df = pd.DataFrame([data], columns=expected_features)
        
        # 5. Realizar predicción
        probability = pipeline.predict_proba(df)[0][1]
        
        return jsonify({
            "risk": float(round(probability, 4)),
            "model": model_name,
            "message": "Success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================
# RUTAS DE VISTAS
# ========================

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/material')
def material():
    return render_template('material.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)