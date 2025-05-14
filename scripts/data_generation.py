import pandas as pd
import numpy as np
from faker import Faker

# Configuración inicial
fake = Faker('es_ES')
np.random.seed(1722)
n_samples = 1500

dropout_thought = np.random.choice(["Yes", "No"], n_samples, p=[0.25, 0.75])

# Lista de distritos de Lima
districts_lima = [
    "Lima Cercado", "Ancón", "Ate", "Barranco", "Breña",
    "Carabayllo", "Chaclacayo", "Chorrillos", "Cieneguilla", "Comas",
    "El Agustino", "Independencia", "Jesús María", "La Molina", "La Victoria",
    "Lince", "Los Olivos", "Lurigancho", "Lurín", "Magdalena del Mar",
    "Miraflores", "Pachacámac", "Pucusana", "Pueblo Libre", "Puente Piedra",
    "Punta Hermosa", "Punta Negra", "Rímac", "San Bartolo", "San Borja",
    "San Isidro", "San Juan de Lurigancho", "San Juan de Miraflores", "San Luis",
    "San Martín de Porres", "San Miguel", "Santa Anita", "Santa María del Mar",
    "Santa Rosa", "Santiago de Surco", "Surquillo", "Villa El Salvador",
    "Villa María del Triunfo"
]

# Generación de datos (nombres de columnas en inglés, valores en español)
data = {
    "gender": np.random.choice(["Masculino", "Femenino"], n_samples, p=[0.85, 0.15]),
    "age_range": np.random.choice(["16-20", "21-25", "26-30", "30+"], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
    "marital_status": np.random.choice(["Soltero(a)", "Casado(a)", "Divorciado(a)", "Otro"], n_samples, p=[0.75, 0.15, 0.05, 0.05]),
    "disability": np.random.choice(["Sí", "No"], n_samples, p=[0.05, 0.95]),
    "district": np.random.choice(districts_lima, n_samples),

    "father_education": np.random.choice(["Primaria", "Secundaria", "Técnico", "Universitario", "Postgrado"], 
                                         n_samples, p=[0.3, 0.4, 0.15, 0.1, 0.05]),
    "mother_education": np.random.choice(["Primaria", "Secundaria", "Técnico", "Universitario", "Postgrado"], 
                                         n_samples, p=[0.25, 0.45, 0.1, 0.15, 0.05]),
    "household_members": np.random.choice(["1-3", "4-5", "6+"], n_samples, p=[0.5, 0.3, 0.2]),
    "economic_dependency": np.random.choice([
        "Totalmente dependiente",
        "Mayormente dependiente",
        "Parcialmente dependiente",
        "Minimamente dependiente",
        "No dependo económicamente"
    ], n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1]),

    "health_insurance": np.random.choice(["EsSalud", "SIS", "Privado", "Ninguno"], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    "chronic_disease": np.random.choice(["Sí", "No"], n_samples, p=[0.15, 0.85]),

    "academic_program": np.random.choice(["Ingeniería de Software", "Ingeniería de Sistemas"], n_samples, p=[0.6, 0.4]),
    "current_term": np.random.randint(1, 11, n_samples),
    "enrolled_courses": np.random.choice(["3", "4-5", "6 o más"], n_samples, p=[0.2, 0.6, 0.2]),
    "gpa": np.random.choice(["<10", "10-12", "13-15", "16+"], n_samples, p=[0.1, 0.3, 0.5, 0.1]),
    "failed_course": np.random.choice(["Sí", "No"], n_samples, p=[0.35, 0.65]),
    "class_attendance": np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.25, 0.4, 0.2]),

    "career_satisfaction": np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
    "personal_motivation": np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.3, 0.4, 0.1]),
    "teacher_motivation": np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
    "thought_about_dropout": np.where(dropout_thought == "Yes", "Sí", "No"),
    "dropout_reason": [fake.text(max_nb_chars=200) if thought == "Yes" else "" for thought in dropout_thought],
    "dropout_thought": dropout_thought,
    
    "transport": np.random.choice(["Público", "Privado", "A pie"], n_samples, p=[0.7, 0.2, 0.1]),
    "commute_time": np.random.choice(["<30 min", "30-60 min", ">1 h"], n_samples, p=[0.4, 0.4, 0.2]),
    "internet_access": np.random.choice(["Sí", "No"], n_samples, p=[0.9, 0.1]),
    "study_space": np.random.choice(["Sí", "No"], n_samples, p=[0.8, 0.2]),
    "library_use": np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.3, 0.25, 0.2, 0.05]),
    "academic_tutoring": np.random.choice(["Sí", "No"], n_samples, p=[0.4, 0.6]),
    "university_support": np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1]),

    "housing_type": np.random.choice(["Propia", "Alquilada", "Alojado", "Residencia"], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    "personal_income": np.random.choice(["0", "<500", "500-1000", ">1000"], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
    "family_income": np.random.choice(["<1000", "1000-2000", "2000-4000", ">4000"], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    "scholarship": np.random.choice(["Sí", "No"], n_samples, p=[0.2, 0.8]),

    "currently_working": np.random.choice(["Sí", "No"], n_samples, p=[0.4, 0.6]),
    "work_hours": np.random.choice(["<10", "10-20", ">20"], n_samples, p=[0.5, 0.3, 0.2]),
    "affects_study": np.random.choice(["Sí", "No"], n_samples, p=[0.6, 0.4]),
    "study_hours": np.random.choice(["<5", "5-10", ">10"], n_samples, p=[0.4, 0.5, 0.1]),

    "university_harassment": np.random.choice(["Sí", "No"], n_samples, p=[0.1, 0.9]),
    "demotivation": np.random.choice(["Sí", "No"], n_samples, p=[0.4, 0.6]),
    "academic_stress": np.random.choice(["Sí", "No"], n_samples, p=[0.5, 0.5]),
    "emotional_support": np.random.choice(["Sí", "No"], n_samples, p=[0.5, 0.5]),
}

# Campo adicional
data["dropout_reason"] = [fake.sentence(nb_words=10) if thought != "" else "" for thought in data["dropout_thought"]]

# Crear DataFrame
df = pd.DataFrame(data)

# Función para agregar ruido
def add_noise(df, noise_level=0.05):
    df_noised = df.copy()
    n_samples = len(df)
    
    # Ruido en variables categóricas
    categorical_cols = ['gender', 'marital_status', 'academic_program'] 
    for col in categorical_cols:
        mask = np.random.rand(n_samples) < noise_level
        original_values = df[col].unique()
        df_noised.loc[mask, col] = np.random.choice(original_values, sum(mask))
    
    # Ruido en variables numéricas
    numerical_cols = ['current_term']
    for col in numerical_cols:
        noise = np.random.normal(0, 0.5, n_samples).round()
        df_noised[col] = np.clip(df_noised[col] + noise, 1, 10)
    
    # Valores faltantes
    missing_mask = np.random.rand(*df.shape) < noise_level
    df_noised = df_noised.mask(missing_mask)
    
    return df_noised

# Aplicar ruido
df = add_noise(df, noise_level=0.03)

# Guardar datos
df.to_csv('synthetic_dropout_data.csv', index=False, encoding='utf-8-sig')

print("¡Dataset generado con éxito!")
print("\nMuestra de datos:")
print(df.head(3))