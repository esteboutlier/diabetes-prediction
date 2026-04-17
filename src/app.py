import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# 1. ARTIFACTS LOAD
# 1. CARGA DE ARTEFACTOS

# The chosen model and scaler, saved in train.py, are loaded 
# Se carga el modelo elegido y el escalador que se guardó en train.py
modelo = joblib.load('models/diabetes_model.joblib')
scaler = joblib.load('models/scaler.joblib')

def predecir_diabetes(age, bmi, glucose, hba1c, s_bp, activity, history, gender):
    # 2. DATA PREPARATION (Input Engineering)
    # 2. PREPARACIÓN DE DATOS (Input Engineering)
    g_male = 1 if gender == 'Male' else 0
    g_other = 1 if gender == 'Other' else 0
    f_history = 1 if history == "Sí" else 0

    # The dictionary is created with the exact names that the model expects
    # The scaler and model need the same columns that were seen in train.py

    # Se crea el diccionario con los nombres exactos que espera el modelo
    # El escalador y el modelo necesitan las mismas columnas que se vieron en train.py
    data_dict = {
        'Age': [age],
        'bmi': [bmi],
        'glucose_fasting': [glucose],
        'hba1c': [hba1c],
        'systolic_bp': [s_bp],
        'family_history_diabetes': [f_history],
        'physical_activity_minutes_per_week': [activity],
        'gender_Male': [g_male],
        'gender_Other': [g_other]
    }
    
    data = pd.DataFrame(data_dict)

    # 3. SCALING
    # 3. ESCALADO
    data_scaled = scaler.transform(data)

    # 4. PREDICTION
    # 4. PREDICCIÓN
    pred = modelo.predict(data_scaled)[0]
    prob = modelo.predict_proba(data_scaled)[0][1] * 100

    # 5. OUTPUT FORMAT
    # 5. FORMATO DE SALIDA
    resultado = "⚠️ Possible diagnosis of diabetes / Posible diagnóstico de diabetes" if pred == 1 else "✅ No signs of diabetes / Sin indicios de diabetes"
    return f"{resultado}\nEstimated Probability / Probabilidad estimada: {prob:.2f}%"

# 6. GRADIO INTERFACE - The interface is defined with the same inputs used to train the model
# 6. INTERFAZ DE GRADIO - Se define la interfaz con los mismos inputs que se usaron para entrenar el modelo
iface = gr.Interface(
    fn=predecir_diabetes,
    inputs=[
        gr.Number(label="Age (years) / Edad (años)", minimum=0, maximum=100),
        gr.Number(label="BMI / IMC", minimum=10, maximum=60),
        gr.Number(label="Fasting Glucose / Glucosa en ayunas (mg/dL)", minimum=50, maximum=300),
        gr.Number(label="HbA1c (%)", minimum=3, maximum=15),
        gr.Number(label="Systolic Blood Pressure / Presión sistólica (mmHg)", minimum=80, maximum=200),
        gr.Number(label="Physical Activity (minutes/week) / Actividad física (minutos/semana)", minimum=0, maximum=1000),
        gr.Radio(choices=["Yes", "No"], label="Family History of Diabetes / ¿Antecedentes familiares de diabetes?"),
        gr.Radio(choices=['Female', 'Male', 'Other'], label="Gender / Género")
    ],
    outputs=gr.Textbox(label="Analysis Result / Resultado del Análisis"),
    title="⚕️ Diabetes Risk Prediction System / Sistema de Predicción de Riesgo de Diabetes",
    description="EN:Tool based on Machine Learning for early detection. Please enter the patient's biometric data." \
    "\nES:Herramienta basada en Machine Learning para la detección temprana. Por favor ingrese los datos biométricos del paciente."
)

if __name__ == "__main__":
    # 7. LAUNCH - The app is launched with share=False for professional local use
    # Launch con share=False para uso local profesional
    iface.launch()