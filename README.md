# Proyecto: Análisis Predictivo de Diabetes

Este proyecto presenta un ciclo completo de análisis de datos de salud, con un enfoque principal en el Análisis Exploratorio de Datos (EDA) y la identificación de factores de riesgo, utilizando Machine Learning para validar las hipótesis encontradas.

## Objetivo del Proyecto

Identificar los indicadores biométricos más críticos en la detección de diabetes y construir un modelo de clasificación capaz de asistir en el triaje preventivo de pacientes.

## Enfoque de Análisis (Data Analytics)

A diferencia de un enfoque puramente de ingeniería, este proyecto se centró en:

* Calidad de Datos: Tratamiento de valores nulos y ceros biológicamente imposibles (en Glucosa, BMI, etc.) mediante imputación estadística por mediana.

* Ingeniería de Variables: Transformación de variables categóricas (Género) y análisis de su correlación con el diagnóstico.

* Interpretabilidad: Uso de modelos basados en árboles para extraer la importancia de las variables, permitiendo explicar por qué el modelo toma ciertas decisiones.

## Hallazgos y Resultados

* Variable Crítica: (Ejemplo: El hba1c resultó ser el predictor más fuerte con un peso del 79.8%).

* Desempeño: Se logró un Accuracy de 92.04%, priorizando la reducción de Falsos Negativos debido al contexto de salud.

* Balanceo: Aplicación de técnicas de muestreo estratificado para corregir el desbalance en la distribución de diagnósticos.

## Estructura del Proyecto

├── data/raw/       # Fuente de datos original
├── notebooks/      # Limpieza detallada y visualizaciones (EDA)
├── src/
│   ├── preprocessing.py  # Automatización de la lógica descubierta en el análisis
│   ├── train.py          # Validación estadística y entrenamiento de modelos
│   └── app.py            # Herramienta de visualización de resultados
└── models/         # Modelos y escaladores serializados

## Stack Técnico

* Análisis y Visualización: Pandas, Matplotlib, Seaborn.

* Modelado Estadístico: Scikit-Learn (Logistic Regression & Random Forest).

* Entorno de Trabajo: Python 3.12 gestionado con uv.

## Cómo replicar el análisis

Clonar: git clone https://github.com/esteboutlier/diabetes-prediction.git

Instalar dependencias: uv sync

Explorar los hallazgos en notebooks/ o ejecutar la herramienta predictiva con python src/app.py.
