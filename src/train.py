import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import basic_cleaning, feature_engineering

def run_training():
    # 1. Rutas y configuración
    csv_path = os.path.join('data', 'raw', 'Diabetes_and_LifeStyle_Dataset .csv')
    model_output_path = os.path.join('models', 'diabetes_model.joblib')
    scaler_output_path = os.path.join('models', 'scaler.joblib')

    print("--- Iniciando Pipeline de Entrenamiento ---")
    
    # 2. Carga y Limpieza por etapas
    if not os.path.exists(csv_path):
        print(f"❌ Error: No se encontró el dataset en {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = basic_cleaning(df)
    df = feature_engineering(df)
    
    # 3. Selección de variables (basado en el EDA)
    variables_modelo = [
        'Age', 'bmi', 'glucose_fasting', 'hba1c', 'systolic_bp', 
        'family_history_diabetes', 'physical_activity_minutes_per_week',
        'gender_Male', 'gender_Other'
    ]
    target = 'diagnosed_diabetes'

    X = df[variables_modelo].copy()
    y = df[target].copy()

    # 4. Train/Test Split con Stratify (Mantiene proporciones de la clase objetivo)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Distribución de datos:")
    print(f"   Entrenamiento: {len(X_train)} (Si: {sum(y_train==1)}, No: {sum(y_train==0)})")
    print(f"   Prueba: {len(X_test)} (Si: {sum(y_test==1)}, No: {sum(y_test==0)})")

    # 5. Escalado de datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- MODELO 1: Regresión Logística ---
    print("\n--- Evaluando Regresión Logística ---")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Accuracy Regresión Logística: {acc_lr*100:.2f}%")

    # --- MODELO 2: Random Forest ---
    print("\n--- Evaluando Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Accuracy Random Forest: {acc_rf*100:.2f}%")

    # 6. Comparación y Selección
    # Se elige el modelo con mejor desempeño
    if acc_rf >= acc_lr:
        best_model = rf_model
        best_name = "Random Forest"
        y_pred_best = y_pred_rf
    else:
        best_model = lr_model
        best_name = "Regresión Logística"
        y_pred_best = y_pred_lr

    print(f"\nModelo seleccionado: {best_name}")

    # 7. Métricas Detalladas del Modelo Seleccionado (Matriz de Confusión)
    cm = confusion_matrix(y_test, y_pred_best)
    print("\nMatriz de Confusión del Ganador:")
    print(f"    Verdaderos Negativos: {cm[0,0]}")
    print(f"    Verdaderos Positivos: {cm[1,1]}")
    print(f"    Falsos Negativos (Riesgo): {cm[1,0]}")
    print(f"    Falsos Positivos: {cm[0,1]}")
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred_best, target_names=['Sin Diabetes', 'Con Diabetes']))

    # 8. Análisis de Importancia de Variables (Feature Importance)
    print("\n📈 Importancia de las variables (Feature Importance):")
    if best_name == "Random Forest":
        importancias = best_model.feature_importances_
    else:
        importancias = np.abs(best_model.coef_[0])

    feat_imp = pd.DataFrame({
        'Variable': variables_modelo,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=False)

    for idx, row in feat_imp.iterrows():
        print(f"   - {row['Variable']:40s}: {row['Importancia']:.4f}")

    # 9. Guardar Artefactos Finales
    joblib.dump(best_model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"\n✅ Proceso finalizado. Modelo y Escalador guardados en carpeta /models/")

if __name__ == "__main__":
    run_training()
