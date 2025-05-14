from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
app = Flask(__name__,
            static_folder='frontend/static',
            template_folder='frontend/templates')

# Cargar modelo y preprocesador
MODEL_PATH = BASE_DIR / "ml_model/saved_models/xgboost.pkl"
pipeline = joblib.load(MODEL_PATH)
expected_features = pipeline.named_steps['preprocessor'].get_feature_names_out()

# Lista de distritos de Lima para one-hot encoding
districts_lima = [
    "Lima_Cercado", "Ancón", "Ate", "Barranco", "Breña", "Carabayllo",
    "Chaclacayo", "Chorrillos", "Cieneguilla", "Comas", "El_Agustino",
    "Independencia", "Jesús_María", "La_Molina", "La_Victoria", "Lince",
    "Los_Olivos", "Lurigancho", "Lurín", "Magdalena_del_Mar", "Miraflores",
    "Pachacámac", "Pucusana", "Pueblo_Libre", "Puente_Piedra", "Punta_Hermosa",
    "Punta_Negra", "Rímac", "San_Bartolo", "San_Borja", "San_Isidro",
    "San_Juan_de_Lurigancho", "San_Juan_de_Miraflores", "San_Luis",
    "San_Martín_de_Porres", "San_Miguel", "Santa_Anita", "Santa_María_del_Mar",
    "Santa_Rosa", "Santiago_de_Surco", "Surquillo", "Villa_El_Salvador",
    "Villa_María_del_Triunfo"
]

# ==============================================================
# Mapeo de campos del formulario a características del modelo
# ==============================================================
FIELD_MAPPING = {
    # Campos personales
    'genero': {
        'Masculino': 'gender_Male',
        'Femenino': 'gender_Female'
    },
    'edad': 'age',
    'estado_civil': {
        'Soltero/a': 'marital_status_Single',
        'Casado/a': 'marital_status_Married',
        'Divorciado/a': 'marital_status_Divorced',
        'Otro': 'marital_status_Other'
    },
    'discapacidad': 'disability',
    'distrito_residencia': 'residence_district',
    
    # Contexto familiar
    'nivel_educativo_padre': {
        'Primaria': 'father_education_level_Primary',
        'Secundaria': 'father_education_level_Secondary',
        'Técnico': 'father_education_level_Technical',
        'Universitario': 'father_education_level_University',
        'Postgrado': 'father_education_level_Postgraduate'
    },
    'nivel_educativo_madre': {
        'Primaria': 'mother_education_level_Primary',
        'Secundaria': 'mother_education_level_Secondary',
        'Técnico': 'mother_education_level_Technical',
        'Universitario': 'mother_education_level_University',
        'Postgrado': 'mother_education_level_Postgraduate'
    },
    'miembros_hogar': 'household_members',
    'dependencia_economica': 'economic_dependency',
    
    # Salud
    'tipo_seguro': {
        'EsSalud': 'insurance_type_EsSalud',
        'SIS': 'insurance_type_SIS',
        'Privado': 'insurance_type_Private',
        'Ninguno': 'insurance_type_None'
    },
    'enfermedad_cronica': 'chronic_disease',
    
    # Académico
    'programa_academico': {
        'Ingeniería de Software': 'academic_program_Software_Engineering',
        'Ingeniería de Sistemas': 'academic_program_Systems_Engineering'
    },
    'ciclo_actual': 'current_semester',
    'cursos_matriculados': 'enrolled_courses',
    'promedio_ponderado': 'weighted_average',
    'curso_reprobado': 'failed_course',
    'asistencia_clases': 'class_attendance',
    
    # Interés académico
    'satisfaccion_carrera': 'career_satisfaction',
    'motivacion_personal': 'personal_motivation',
    'motivacion_profesores': 'professor_motivation',
    'considero_abandono': 'considered_career_change',
    'motivo_abandono': 'dropout_reason',
    
    # Recursos universitarios
    'transporte': {
        'Público': 'transportation_Public',
        'Privado': 'transportation_Private',
        'A pie': 'transportation_Walking'
    },
    'tiempo_traslado': 'commute_time',
    'acceso_internet': 'internet_access',
    'espacio_estudio': 'study_space',
    'uso_biblioteca': 'library_usage',
    'tutoria_academica': 'academic_tutoring',
    'apoyo_universitario': 'university_support',
    
    # Situación económica
    'tipo_vivienda': {
        'Propia': 'housing_type_Owned',
        'Alquilada': 'housing_type_Rented',
        'Alojado': 'housing_type_Lodged',
        'Residencia': 'housing_type_Dormitory'
    },
    'ingreso_personal': 'personal_income',
    'ingreso_familiar': 'family_income',
    'beca': 'scholarship',
    
    # Trabajo y tiempo
    'trabajo_actual': 'currently_working',
    'horas_trabajo': 'work_hours',
    'afecta_estudios': 'work_affects_studies',
    'horas_estudio': 'study_hours',
    
    # Factores emocionales
    'acoso_universitario': 'bullying_experience',
    'desmotivacion': 'demotivation',
    'estres_academico': 'academic_stress',
    'apoyo_emocional': 'emotional_support'
}

# ==============================================================
# Reglas de conversión de valores
# ==============================================================
CONVERSION_RULES = {
    # ----------------------------------------------------------
    # Campos one-hot encoded
    # ----------------------------------------------------------
    'genero': lambda x: {v: 1 if k == x else 0 for k, v in FIELD_MAPPING['genero'].items()},
    'estado_civil': lambda x: {v: 1 if k == x else 0 for k, v in FIELD_MAPPING['estado_civil'].items()},
    'distrito_residencia': lambda x: {f'cat__residence_district_{d.replace(" ", "_")}': 1 if d == x else 0 for d in districts_lima},
    'nivel_educativo_padre': lambda x: {v: 1 if k == x else 0 for k, v in FIELD_MAPPING['nivel_educativo_padre'].items()},
    'nivel_educativo_madre': lambda x: {v: 1 if k == x else 0 for k, v in FIELD_MAPPING['nivel_educativo_madre'].items()},
    'tipo_seguro': lambda x: {v: 1 if k == x else 0 for k, v in FIELD_MAPPING['tipo_seguro'].items()},
    'programa_academico': lambda x: {v: 1 if k == x else 0 for k, v in FIELD_MAPPING['programa_academico'].items()},
    'transporte': lambda x: {v: 1 if k == x else 0 for k, v in FIELD_MAPPING['transporte'].items()},
    'tipo_vivienda': lambda x: {v: 1 if k == x else 0 for k, v in FIELD_MAPPING['tipo_vivienda'].items()},

    # Campos numéricos
    'num__age': lambda x: float(x.split('-')[0]),
    'num__household_members': lambda x: int(x.split('-')[0]),
    'num__current_semester': lambda x: int(x),
    'num__enrolled_courses': lambda x: int(x.replace(' o más', '').split('-')[0]),
    'num__class_attendance': lambda x: int(x),
    'num__career_satisfaction': lambda x: int(x),
    'num__personal_motivation': lambda x: int(x),
    'num__professor_motivation': lambda x: int(x),
    'num__commute_time': lambda x: {"<30 min": 0, "30-60 min": 1, ">1 h": 2}.get(x, 0),
    'num__library_usage': lambda x: int(x),
    'num__university_support': lambda x: int(x),
    'num__work_hours': lambda x: int(x.replace('>', '').replace('<', '').split('-')[0]),
    'num__study_hours': lambda x: int(x.replace('>', '').replace('<', '').split('-')[0]),
    'num__weighted_average': lambda x: {"<10": 9.5, "10-12": 11.0, "13-15": 14.0, "16+": 16.5}.get(x, 0.0),
    'num__personal_income': lambda x: {"0": 0, "<500": 250, "500-1000": 750, ">1000": 1500}.get(x, 0),
    'num__family_income': lambda x: {"<1000": 500, "1000-2000": 1500, "2000-4000": 3000, ">4000": 5000}.get(x, 0),

    # ----------------------------------------------------------
    # Campos binarios (Sí/No)
    # ----------------------------------------------------------
    'cat__disability_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__chronic_disease_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__failed_course_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__internet_access_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__study_space_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__academic_tutoring_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__scholarship_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__currently_working_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__work_affects_studies_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__bullying_experience_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__demotivation_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__academic_stress_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__emotional_support_Yes': lambda x: 1 if x == "Sí" else 0,
    'cat__considered_career_change_Yes': lambda x: 1 if x == "Sí" else 0
}

# ==============================================================
# Rutas de Flask
# ==============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        formatted_data = {feat: 0 for feat in expected_features}

        for field, value in data.items():
            if field in FIELD_MAPPING:
                mapping = FIELD_MAPPING[field]
                
                # Si es un diccionario (one-hot encoding)
                if isinstance(mapping, dict):
                    for option, feature_suffix in mapping.items():
                        feature_name = f"cat__{feature_suffix}"
                        if feature_name in expected_features:
                            formatted_data[feature_name] = 1 if value == option else 0
                
                # Si es un string (numérico o binario)
                elif isinstance(mapping, str):
                    # Para campos binarios (Sí/No)
                    if value in ['Sí', 'No']:
                        feature_name = f"cat__{mapping}_{'Yes' if value == 'Sí' else 'No'}"
                        if feature_name in expected_features:
                            formatted_data[feature_name] = 1 if value == 'Sí' else 0
                    # Para campos numéricos
                    else:
                        feature_name = f"num__{mapping}"
                        if feature_name in expected_features:
                            formatted_data[feature_name] = CONVERSION_RULES.get(feature_name, lambda x: x)(value)

        # Validar que tenemos datos
        if all(v == 0 for v in formatted_data.values()):
            return jsonify({"error": "No se pudo mapear ningún campo"}), 400
        
        
        # Crear DataFrame solo con las columnas esperadas
        df = pd.DataFrame([formatted_data])[expected_features]
        probability = pipeline.predict_proba(df)[0][1]

        # Generar reporte
        report_content = f"""Reporte de Predicción de Abandono
Probabilidad: {probability:.2%}
Nivel de riesgo: {'Alto' if probability > 0.7 else 'Moderado' if probability > 0.4 else 'Bajo'}

Detalles del estudiante:
{pd.Series({k:v for k,v in formatted_data.items() if v != 0}).to_string()}
"""
        return jsonify({
            "risk": float(probability),
            "report": report_content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
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
    print("🔥 Características esperadas por el modelo:", expected_features)
    app.run(host='0.0.0.0', port=5000, debug=True)