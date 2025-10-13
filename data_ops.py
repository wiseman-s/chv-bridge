# utils/data_ops.py

import pandas as pd
import os
from datetime import datetime

# --------------------------
# Setup data folder & file
# --------------------------
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "visits.csv")

# ensure folder exists (fixes Streamlit Cloud error)
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------
# Counties list (Kenya)
# --------------------------
KENYA_COUNTIES = [
    "Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo Marakwet", "Embu", "Garissa", 
    "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho", "Kiambu", "Kilifi", 
    "Kirinyaga", "Kisii", "Kisumu", "Kitui", "Kwale", "Laikipia", "Lamu", 
    "Machakos", "Makueni", "Mandera", "Marsabit", "Meru", "Migori", "Mombasa", 
    "Murang'a", "Nairobi", "Nakuru", "Nandi", "Narok", "Nyamira", "Nyandarua", 
    "Nyeri", "Samburu", "Siaya", "Taita Taveta", "Tana River", "Tharaka Nithi", 
    "Trans Nzoia", "Turkana", "Uasin Gishu", "Vihiga", "Wajir", "West Pokot"
]

# --------------------------
# Load visits
# --------------------------
def load_visits():
    """Load visits CSV or create empty file if missing."""
    if not os.path.exists(DATA_FILE):
        # create empty dataframe if none exists
        df = pd.DataFrame(columns=["CHV", "Client", "VisitType", "County", "Date", "Notes"])
        df.to_csv(DATA_FILE, index=False)
        return df
    df = pd.read_csv(DATA_FILE)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    return df

# --------------------------
# Save a single visit row
# --------------------------
def save_visit(row: dict):
    """Append one new record to visits.csv."""
    df = load_visits()
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# --------------------------
# Filter visits
# --------------------------
def get_filtered_visits(df, start_date, end_date, visit_types=None, counties=None):
    """Filter dataframe by date range, visit type, and county."""
    if df.empty:
        return df
    filtered = df.copy()
    if "Date" in filtered.columns:
        filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce").dt.date
        filtered = filtered[
            (filtered["Date"] >= start_date) & (filtered["Date"] <= end_date)
        ]
    if visit_types:
        filtered = filtered[filtered["VisitType"].isin(visit_types)]
    if counties:
        filtered = filtered[filtered["County"].isin(counties)]
    return filtered
