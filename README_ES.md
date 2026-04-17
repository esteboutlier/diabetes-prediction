# 🩺 Diabetes Risk Predictor: End-to-End ML Pipeline

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-orange?style=for-the-badge&logo=gradio&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## 🌎 Language / Idioma

> 📖 [Read in English](README.md) | [Leer en Español](README_ES.md)

## Proyecto: Análisis Predictivo de Diabetes

Este proyecto presenta un ciclo completo de análisis de datos de salud, con un enfoque principal en el Análisis Exploratorio de Datos (EDA) y la identificación de factores de riesgo, utilizando Machine Learning para validar las hipótesis encontradas.

### Objetivo del Proyecto

Identificar los indicadores biométricos más críticos en la detección de diabetes y construir un modelo de clasificación capaz de asistir en el triaje preventivo de pacientes.

### Enfoque de Análisis (Data Analytics)

A diferencia de un enfoque puramente de ingeniería, este proyecto se centró en:

* Calidad de Datos: Tratamiento de valores nulos y ceros biológicamente imposibles (en Glucosa, BMI, etc.) mediante imputación estadística por mediana.

* Ingeniería de Variables: Transformación de variables categóricas (Género) y análisis de su correlación con el diagnóstico.

* Interpretabilidad: Uso de modelos basados en árboles para extraer la importancia de las variables, permitiendo explicar por qué el modelo toma ciertas decisiones.

### Hallazgos y Resultados

* Variable Crítica: El hba1c resultó ser el predictor más fuerte con un peso del 79.8%.

* Desempeño: Se logró un Accuracy de 92.04%, priorizando la reducción de Falsos Negativos debido al contexto de salud.

* Balanceo: Aplicación de técnicas de muestreo estratificado para corregir el desbalance en la distribución de diagnósticos.

### Estructura del Proyecto

diabetes-prediction/
├── data/
│   └── raw/             # Dataset original y archivos CSV
├── models/              # Artefactos: model.joblib y scaler.joblib
├── notebooks/           # EDA y experimentación
├── src/
│   ├── preprocessing.py # Funciones de limpieza y feature engineering
│   ├── train.py         # Script de entrenamiento y comparación de modelos
│   └── app.py           # Interfaz interactiva para el usuario final (Gradio)
├── requirements.txt     # Dependencias del proyecto
└── README.md            # Documentación principal

### Stack Técnico

* Análisis y Visualización: Pandas, Matplotlib, Seaborn.

* Modelado Estadístico: Scikit-Learn (Logistic Regression & Random Forest).

* Entorno de Trabajo: Python 3.12 gestionado con uv.

### Cómo replicar el análisis

Clonar: git clone <https://github.com/esteboutlier/diabetes-prediction.git>

Instalar dependencias: uv sync

Explorar los hallazgos en notebooks/ o ejecutar la herramienta predictiva con python src/app.py.

### 📩 Contact / Contacto

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)]<https://www.linkedin.com/in/estebgomiba>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)]
<https://github.com/esteboutlier>
