import pandas as pd

INCENTIVE_RULES = {
    "Home Visit": 50,
    "Immunization": 75,
    "ANC Visit": 60,
    "Nutrition Check": 40
}

def add_incentives_column(df):
    df = df.copy()
    df["Incentive"] = df["VisitType"].map(INCENTIVE_RULES).fillna(0)
    return df
