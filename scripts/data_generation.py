import pandas as pd
import numpy as np
from faker import Faker

# Configuración inicial
fake = Faker('en_US')  # Usamos localización en inglés (EE.UU.)
np.random.seed(1722)      # Para reproducibilidad
n_samples = 1500        # Número de registros
# --- Generar 'dropout_thought' PRIMERO ---
dropout_thought = np.random.choice(["Yes", "No"], n_samples, p=[0.25, 0.75])

# Lista de distritos de Lima (en inglés, pero mantenemos nombres propios)
districts_lima = [
    "Lima Cercado", "Ancón", "Ate", "Barranco", "Breña",
    "Carabayllo", "Chaclacayo", "Chorrillos", "Cieneguilla", "Comas",
    "El Agustino", "Independencia", "Jesús María", "La Molina", "La Victoria",
    "Lince", "Los Olivos", "Lurigancho-Chosica", "Lurín", "Magdalena del Mar",
    "Miraflores", "Pachacámac", "Pucusana", "Pueblo Libre", "Puente Piedra",
    "Punta Hermosa", "Punta Negra", "Rímac", "San Bartolo", "San Borja",
    "San Isidro", "San Juan de Lurigancho", "San Juan de Miraflores", "San Luis",
    "San Martín de Porres", "San Miguel", "Santa Anita", "Santa María del Mar",
    "Santa Rosa", "Santiago de Surco", "Surquillo", "Villa El Salvador",
    "Villa María del Triunfo"
]

# Generación de datos
data = {
    # --- 1. Datos Generales ---
    "gender": np.random.choice(["Male", "Female"], n_samples, p=[0.85, 0.15]),
    "age": np.random.randint(17, 35, n_samples),
    "marital_status": np.random.choice(["Single", "Married", "Divorced"], n_samples, p=[0.85, 0.1, 0.05]),
    "disability": np.random.choice(["Yes", "No"], n_samples, p=[0.05, 0.95]),
    "residence_district": np.random.choice(districts_lima, n_samples),
    
    # --- 2. Contexto Familiar ---
    "father_education_level": np.random.choice(["Primary", "Secondary", "Technical", "University", "Postgraduate"], 
                                         n_samples, p=[0.3, 0.4, 0.15, 0.1, 0.05]),
    "mother_education_level": np.random.choice(["Primary", "Secondary", "Technical", "University", "Postgraduate"], 
                                         n_samples, p=[0.25, 0.45, 0.1, 0.15, 0.05]),
    "household_members": np.random.randint(1, 8, n_samples),
    "economic_dependency": np.random.choice(["Total", "Partial", "Minimal", "Do not depend"], 
                                          n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    
    # --- 3. Salud ---
    "insurance_type": np.random.choice(["EsSalud", "SIS", "Private", "None"], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    "chronic_disease": np.random.choice(["Yes", "No"], n_samples, p=[0.15, 0.85]),
    
    # --- 4. Información Académica ---
    "academic_program": np.random.choice(["Software Engineering", "Systems Engineering"], n_samples, p=[0.6, 0.4]),
    "current_semester": np.random.randint(1, 11, n_samples),
    "enrolled_courses": np.random.choice(["3", "4–5", "6 or more"], n_samples, p=[0.2, 0.6, 0.2]),
    "weighted_average": np.random.choice(["<10", "10–12", "13–15", "16+"], n_samples, p=[0.1, 0.3, 0.5, 0.1]),
    "failed_course": np.random.choice(["Yes", "No"], n_samples, p=[0.35, 0.65]),
    "class_attendance": np.random.choice(["Always", "Almost always", "Sometimes", "Never"], 
                                      n_samples, p=[0.3, 0.4, 0.25, 0.05]),
    
    # --- 5. Interés y Satisfacción Académica ---
    "career_motivation": [fake.sentence(nb_words=8) for _ in range(n_samples)],
    "career_satisfaction": np.random.choice(["Very unsatisfied", "Unsatisfied", "Satisfied", "Very satisfied"], 
                                         n_samples, p=[0.1, 0.2, 0.5, 0.2]),
    "personal_motivation": np.random.choice(["Yes", "No"], n_samples, p=[0.6, 0.4]),
    "professor_motivation": np.random.choice(["Yes", "No"], n_samples, p=[0.5, 0.5]),
    "considered_career_change": np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7]),
    "dropout_thought": dropout_thought,  # Usamos la variable generada arriba
    
    
    # --- 6. Recursos y Apoyo Universitario ---
    "transportation": np.random.choice(["Public", "Private", "Walking"], n_samples, p=[0.7, 0.2, 0.1]),
    "commute_time": np.random.choice(["<30 min", "30–60 min", ">1 h"], n_samples, p=[0.4, 0.4, 0.2]),
    "internet_access": np.random.choice(["Yes", "No"], n_samples, p=[0.9, 0.1]),
    "study_space": np.random.choice(["Yes", "No"], n_samples, p=[0.8, 0.2]),
    "library_usage": np.random.choice(["Regularly", "Sometimes", "Never"], n_samples, p=[0.3, 0.5, 0.2]),
    "academic_tutoring": np.random.choice(["Yes", "No"], n_samples, p=[0.4, 0.6]),
    "university_support": np.random.choice(["Yes", "No"], n_samples, p=[0.5, 0.5]),
    
    # --- 7. Situación Económica ---
    "housing_type": np.random.choice(["Owned", "Rented", "Shared", "Dormitory"], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    "personal_income": np.random.choice(["0", "<500", "500–1000", ">1000"], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
    "family_income": np.random.choice(["<1000", "1000–2000", "2000–4000", ">4000"], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    "scholarship": np.random.choice(["Yes", "No"], n_samples, p=[0.2, 0.8]),
    
    # --- 8. Compromisos y Tiempo ---
    "currently_working": np.random.choice(["Yes", "No"], n_samples, p=[0.4, 0.6]),
    "work_hours": np.random.choice(["<10", "10–20", ">20"], n_samples, p=[0.5, 0.3, 0.2]),
    "work_affects_studies": np.random.choice(["Yes", "No"], n_samples, p=[0.6, 0.4]),
    "study_hours": np.random.choice(["<5", "5–10", ">10"], n_samples, p=[0.4, 0.5, 0.1]),
    
    # --- 9. Factores Emocionales ---
    "bullying_experience": np.random.choice(["Yes", "No"], n_samples, p=[0.1, 0.9]),
    "demotivation": np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7]),
    "academic_stress": np.random.choice(["Yes", "No"], n_samples, p=[0.4, 0.6]),
    "emotional_support": np.random.choice(["Yes", "No"], n_samples, p=[0.5, 0.5]),
}
# --- Generar 'dropout_reason' DESPUÉS de crear 'data' ---
data["dropout_reason"] = [fake.sentence(nb_words=10) if dropout == "Yes" else "" for dropout in data["dropout_thought"]]

# Crear DataFrame y guardar
df = pd.DataFrame(data)
def add_noise(df, noise_level=0.05):
    """
    Agrega tres tipos de ruido al dataset:
    1. Ruido categórico: Cambia valores en variables categóricas
    2. Ruido numérico: Perturbaciones en variables numéricas
    3. Valores faltantes: Introduce NaN en random
    """
    df_noised = df.copy()
    n_samples = len(df)
    
    # 1. Ruido en variables categóricas (ej: 5% de valores cambiados)
    categorical_cols = ['gender', 'marital_status', 'academic_program']
    for col in categorical_cols:
        mask = np.random.rand(n_samples) < noise_level
        if col == 'gender':
            df_noised.loc[mask, col] = np.random.choice(["Male", "Female"], sum(mask))
        elif col == 'marital_status':
            df_noised.loc[mask, col] = np.random.choice(["Single", "Married", "Divorced"], sum(mask))
        # ... agregar lógica para otras columnas

    # 2. Ruido en variables numéricas (distribución normal)
    numerical_cols = ['age', 'current_semester', 'household_members']
    for col in numerical_cols:
        noise = np.random.normal(0, 1, n_samples)  # Ajustar escala según columna
        df_noised[col] = df_noised[col] + noise.round().astype(int)
        
        # Mantener dentro de rangos válidos
        if col == 'age':
            df_noised[col] = df_noised[col].clip(17, 35)
        elif col == 'current_semester':
            df_noised[col] = df_noised[col].clip(1, 10)

    # 3. Valores faltantes (5% de datos)
    missing_mask = np.random.rand(*df.shape) < noise_level
    df_noised = df_noised.mask(missing_mask)
    
    return df_noised

# Aplicar ruido (5% en todos los tipos)
df = add_noise(df, noise_level=0.05)

# =============================================
# Guardar datos con ruido
# =============================================

df.to_csv('synthetic_dropout_data.csv', index=False)
print("¡Data sintética CON RUIDO generada con éxito!")
print("Muestra del dataset con ruido:")
print(df.head(3))

# =============================================

print("¡Data sintética en inglés generada con éxito! Muestra del dataset:")
print(df.head(3))