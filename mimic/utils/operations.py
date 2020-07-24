import logging


def add_bmi(df):
    if {"Weight_kg", "Height_inches"}.issubset(df.columns):
        logging.debug("Adding BMI")
        df['BMI'] = df['Weight_kg'] / (df['Height_inches'] * 0.0254) ** 2
    return df


def add_pao2_fio2(df):
    if {"PaO2", "FiO2"}.issubset(df.columns):
        logging.debug("Adding PaO2/FiO2")
        df['PaO2/FiO2'] = df['PaO2'] / df['FiO2']
    return df

def add_rsc(df):
    if {"Tidal_ml", "Plateau_Pressure_cmH2O", "Total_PEEP_cmH2O"}.issubset(df.columns):
        logging.debug("Adding RSC")
        df['RSC'] = df['Tidal_ml'] / (df['Plateau_Pressure_cmH2O'] - df['Total_PEEP_cmH2O'])
    return df
    