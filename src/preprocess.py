import pandas as pd
import numpy as np

def basic_cleaning(df):
    """
    Clean the diabetes dataset by replacing zeros with the median
    in columns where zero does not make physiological sense.
    
    Limpia el dataset de diabetes reemplazando ceros por la mediana
    en columnas donde el cero no tiene sentido fisiológico.
    """
    cols_to_fix = ['glucose_fasting', 'insulin_level', 'hba1c', 'systolic_bp', 'bmi']
    
    for col in cols_to_fix:
        # Replace zeros with NaN to use pandas' fillna method easily
        # Reemplazamos 0 por NaN para usar el método de pandas fácilmente
        df[col] = df[col].replace(0, np.nan)

        # Fill NaN values with the median of the column (as done in EDA)
        # Llenamos con la mediana (tal como se hizo en el EDA)
        df[col] = df[col].fillna(df[col].median())
    
    return df

def feature_engineering(df):
    """
    Add new features to the dataset, such as a binary variable for gender.
    Agrega nuevas características al dataset, como una variable binaria para el género.
    """
    if 'gender' in df.columns:
        df = pd.get_dummies(df, columns=['gender'], drop_first=True)
    
    return df    