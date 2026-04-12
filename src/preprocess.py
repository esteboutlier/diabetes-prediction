import pandas as pd
import numpy as np

def basic_cleaning(df):
    """
    Limpia el dataset de diabetes reemplazando ceros por la mediana
    en columnas donde el cero no tiene sentido fisiológico.
    """
    cols_to_fix = ['glucose_fasting', 'insulin_level', 'hba1c', 'systolic_bp', 'bmi']
    
    for col in cols_to_fix:
        # Reemplazamos 0 por NaN para usar el método de pandas fácilmente
        df[col] = df[col].replace(0, np.nan)
        # Llenamos con la mediana (tal como hiciste en tu EDA)
        df[col] = df[col].fillna(df[col].median())
    
    return df

def feature_engineering(df):
    """
    Agrega nuevas características al dataset, como una variable binaria para el género.
    """
    if 'gender' in df.columns:
        df = pd.get_dummies(df, columns=['gender'], drop_first=True)
    
    return df    