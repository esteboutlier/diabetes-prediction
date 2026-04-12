import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# 1. CARGA DE ARTEFACTOS
# Se carga el modelo elegido y el escalador que se guardó en train.py
modelo = joblib.load('models/diabetes_model.joblib')
scaler = joblib.load('models/scaler.joblib')

def predecir_diabetes(age, bmi, glucose, hba1c, s_bp, activity, history, gender):
    # 2. PREPARACIÓN DE DATOS (Input Engineering)
    g_male = 1 if gender == 'Male' else 0
    g_other = 1 if gender == 'Other' else 0
    f_history = 1 if history == "Sí" else 0
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

    # 3. ESCALADO
    data_scaled = scaler.transform(data)

    # 4. PREDICCIÓN
    pred = modelo.predict(data_scaled)[0]
    prob = modelo.predict_proba(data_scaled)[0][1] * 100

    # 5. FORMATO DE SALIDA
    resultado = "⚠️ Posible diagnóstico de diabetes" if pred == 1 else "✅ Sin indicios de diabetes"
    return f"{resultado}\nProbabilidad estimada: {prob:.2f}%"

# 6. INTERFAZ DE GRADIO - Se define la interfaz con los mismos inputs que se usaron para entrenar el modelo
iface = gr.Interface(
    fn=predecir_diabetes,
    inputs=[
        gr.Number(label="Edad (años)", minimum=0, maximum=100),
        gr.Number(label="Índice de masa corporal (BMI)", minimum=10, maximum=60),
        gr.Number(label="Glucosa en ayunas (mg/dL)", minimum=50, maximum=300),
        gr.Number(label="HbA1c (%)", minimum=3, maximum=15),
        gr.Number(label="Presión sistólica (mmHg)", minimum=80, maximum=200),
        gr.Number(label="Actividad física (minutos/semana)", minimum=0, maximum=1000),
        gr.Radio(choices=["Sí", "No"], label="¿Antecedentes familiares de diabetes?"),
        gr.Radio(choices=['Female', 'Male', 'Other'], label="Género")
    ],
    outputs=gr.Textbox(label="Resultado del Análisis"),
    title="⚕️ Sistema de Predicción de Riesgo de Diabetes",
    description="Herramienta basada en Machine Learning para la detección temprana. Ingrese los datos biométricos del paciente."
)

if __name__ == "__main__":
    # Launch con share=False para uso local profesional
    iface.launch()