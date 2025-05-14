from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
import joblib
import sys

# Añade la ruta al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend/app')))

app = Flask(__name__,
            static_folder='frontend/static',
            template_folder='frontend/templates')

# Cargar el modelo una vez al iniciar
MODEL_PATH = "./ml_model/saved_models/xgboost.pkl"
pipeline = joblib.load(MODEL_PATH)
expected_features = pipeline.named_steps['preprocessor'].feature_names_in_

# Configuración de rutas principales
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Validar entrada
        data = request.get_json()
        field_mapping = {
        'genero': 'gender',
        'edad': 'age',
        'estado_civil': 'marital_status',
        'discapacidad': 'disability',
        'distrito_residencia': 'district',
        'nivel_educativo_padre': 'father_education',
        'nivel_educativo_madre': 'mother_education',
        'miembros_hogar': 'household_members',
        'dependencia_economica': 'economic_dependency',
        'tipo_seguro': 'insurance_type',
        'enfermedad_cronica': 'chronic_illness',
        'programa_academico': 'academic_program',
        'ciclo_actual': 'academic_cycle',
        'cursos_matriculados': 'enrolled_courses',
        'promedio_ponderado': 'last_semester_gpa',
        'curso_reprobado': 'failed_courses',
        'asistencia_clases': 'class_attendance',
        'satisfaccion_carrera': 'career_satisfaction',
        'motivacion_personal': 'motivation_level',
        'motivacion_profesores': 'professor_motivation',
        'considero_abandono': 'consider_dropout',
        'transporte': 'transportation',
        'tiempo_traslado': 'commute_time',
        'acceso_internet': 'internet_access',
        'espacio_estudio': 'study_space',
        'uso_biblioteca': 'library_use',
        'tutoria_academica': 'tutoring',
        'apoyo_universitario': 'university_support',
        'tipo_vivienda': 'housing_type',
        'ingreso_personal': 'personal_income',
        'ingreso_familiar': 'family_income',
        'beca': 'scholarship',
        'trabajo_actual': 'working',
        'horas_trabajo': 'work_hours',
        'afecta_estudios': 'work_affects_studies',
        'horas_estudio': 'study_hours',
        'acoso_universitario': 'discrimination',
        'desmotivacion': 'emotional_exhaustion',
        'estres_academico': 'academic_stress',
        'apoyo_emocional': 'emotional_support'
    }
        if not data:
            return jsonify({"error": "Datos no proporcionados"}), 400
        
       # Convertir nombres de campos
        formatted_data = {field_mapping.get(k, k): v for k, v in data.items()}
        
        # 1. Cargar el pipeline
        pipeline = joblib.load('ml_model/saved_models/xgboost.pkl')  # Ajusta la ruta
        
        # 2. Obtener características esperadas
        expected_features = pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        # 3. Validar campos (versión simplificada)
        missing = [feat for feat in expected_features if feat not in formatted_data]
        if missing:
            return jsonify({"error": f"Faltan algunos campos. Complete todos los campos del formulario"}), 400
        
        # 4. Crear DataFrame
        df = pd.DataFrame([formatted_data])
        
        # 5. Realizar predicción
        try:
            probability = pipeline.predict_proba(df)[0][1]
        except Exception as model_error:
            return jsonify({"error": f"Error en el modelo: {str(model_error)}"}), 500

        # 6. Formatear respuesta
        return jsonify({
            "risk": float(probability),
            "model": "xgboost",
            "features_used": list(expected_features)
        })

    except Exception as e:
        return jsonify({"error": f"Error general: {str(e)}"}), 500

# Otras rutas
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/material')
def material():
    return render_template('material.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)