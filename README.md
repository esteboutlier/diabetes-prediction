# 🩺 Diabetes Risk Predictor: End-to-End ML Pipeline

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-orange?style=for-the-badge&logo=gradio&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## 🌎 Language / Idioma

> 📖 [Read in English](README.md) | [Leer en Español](README_ES.md)

## Project: Predictive Diabetes Analysis

This project presents a complete health data analysis cycle, with a primary focus on Exploratory Data Analysis (EDA) and risk factor identification, using Machine Learning to validate the findings.

### Project Goal

Identify the most critical biometric indicators for diabetes detection and build a classification model capable of assisting in preventive patient triage.

### Analysis Approach (Data Analytics)

Unlike a purely engineering-focused approach, this project centered on:

* Data Quality: Handling null values and biologically impossible zeros (in Glucose, BMI, etc.) through median-based statistical imputation.

* Feature Engineering: Transformation of categorical variables (Gender) and analysis of their correlation with the diagnosis.

* Interpretability: Use of tree-based models to extract feature importance, enabling clear explanations of why the model makes certain decisions.

### Findings and Results

* Critical Variable: HbA1c turned out to be the strongest predictor, with a weight of 79.8%.

* Performance: An accuracy of 92.04% was achieved, prioritizing the reduction of False Negatives given the healthcare context.

* Balancing: Application of stratified sampling techniques to correct the imbalance in the diagnosis class distribution.

### Project Structure

diabetes-prediction/
├── data/
│   └── raw/             # Original dataset and CSV files
├── models/              # Artifacts: model.joblib and scaler.joblib
├── notebooks/           # EDA and experimentation
├── src/
│   ├── preprocessing.py # Cleaning and feature engineering functions
│   ├── train.py         # Training script and model comparison
│   └── app.py           # Interactive interface for end users (Gradio)
├── requirements.txt     # Project dependencies
└── README.md            # Main documentation

### Tech Stack

* Analysis & Visualization: Pandas, Matplotlib, Seaborn.

* Statistical Modeling: Scikit-Learn (Logistic Regression & Random Forest).

* Environment: Python 3.12 managed with uv.

### How to Replicate the Analysis

Clone: git clone <https://github.com/esteboutlier/diabetes-prediction.git>

Install dependencies: uv sync

Explore the findings in notebooks/ or run the predictive tool with python src/app.py.

### 📩 Contact

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)]<https://www.linkedin.com/in/estebgomiba>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)]
<https://github.com/esteboutlier>
