import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

# ⬇️ Crear carpeta si no existe
os.makedirs("ml_model/saved_models", exist_ok=True)

# ⬇️ Funciones auxiliares
def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.savefig(f"ml_model/saved_models/confusion_matrix_{filename}.png")
    plt.close()

def plot_roc_curve(y_true, y_proba, model_name, filename):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"ml_model/saved_models/roc_curve_{filename}.png")
    plt.close()
    
    return fpr, tpr, roc_auc

# ⬇️ Cargar datos
df = pd.read_csv("data/synthetic/synthetic_dropout_data.csv")
df = df.dropna(subset=["dropout_thought"])

X = df.drop("dropout_thought", axis=1)
y = np.where(df["dropout_thought"] == "Yes", 1, 0)

# ⬇️ Preprocesamiento
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features),
        ("num", Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
    ],
    remainder='passthrough'
)

# ⬇️ Modelos
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=1722),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", n_estimators=100, random_state=1722),
    "XGBoost": XGBClassifier(
        scale_pos_weight=np.sum(y == 0) / np.sum(y == 1),
        eval_metric="logloss", random_state=1722, tree_method='hist', enable_categorical=True)
}

# ⬇️ División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1722
)

# ⬇️ Entrenamiento y evaluación
results = []
roc_data = []

for model_name, model in models.items():
    print(f"\n🚀 Entrenando {model_name}...")

    try:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Modelo": model_name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 2),
            "Precision": round(precision_score(y_test, y_pred), 2),
            "Recall": round(recall_score(y_test, y_pred), 2),
            "F1": round(f1_score(y_test, y_pred), 2),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 2)
        }
        results.append(metrics)

        model_filename = model_name.replace(" ", "_").lower()
        joblib.dump(pipeline, f"ml_model/saved_models/{model_filename}.pkl")

        plot_confusion_matrix(y_test, y_pred, model_name, model_filename)
        fpr, tpr, roc_auc = plot_roc_curve(y_test, y_proba, model_name, model_filename)
        roc_data.append((fpr, tpr, roc_auc, model_name))

    except Exception as e:
        print(f"❌ Error en {model_name}: {e}")
        continue

# ⬇️ Curva ROC comparativa
plt.figure(figsize=(10, 8))
for fpr, tpr, roc_auc, name in roc_data:
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparativa de Curvas ROC')
plt.legend(loc="lower right")
plt.savefig("ml_model/saved_models/comparative_roc_curves.png")
plt.close()

# ⬇️ Mostrar tabla resumen
results_df = pd.DataFrame(results)
print("\n📋 Comparación de Modelos:")
print(results_df.to_markdown(index=False))

# ⬇️ Características procesadas
try:
    print("\n🔍 Características procesadas:", 
          len(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out()))
except Exception as e:
    print("\n⚠️ Error al obtener características:", str(e))
