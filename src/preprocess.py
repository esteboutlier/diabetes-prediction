import pandas as pd
import numpy as np

def clean_data(df):
    """
    Limpia el dataset de diabetes reemplazando ceros por la mediana
    en columnas donde el cero no tiene sentido fisiológico.
    """
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in cols_to_fix:
        # Reemplazamos 0 por NaN para usar el método de pandas fácilmente
        df[col] = df[col].replace(0, np.nan)
        # Llenamos con la mediana (tal como hiciste en tu EDA)
        df[col] = df[col].fillna(df[col].median())
    
    return df
    