# app.py — Community Health Bridge (Frontend + Data Manager + Predictive + Prophet)
import streamlit as st
import pandas as pd
import numpy as np
import uuid
import time
from datetime import datetime, timedelta
from io import BytesIO
import sys, os

# ensure imports work on Streamlit Cloud even without folder structure
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Prophet (may not be installed in all environments)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ✅ since all files are in root, import directly
from data_ops import load_visits, save_visit, get_filtered_visits
from incentives import INCENTIVE_RULES, add_incentives_column
from charts import plot_incentives_over_time, plot_incentives_by_type
from reports import generate_pdf_report_bytes

# ----------------------
# Page config & CSS
# ----------------------
st.set_page_config(page_title="Community Health Bridge - Data and Predictive", page_icon="🏥", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        section[data-testid="stSidebar"] .css-ng1t4o {padding-top: 1rem;}
        .rank-box {padding: 0.5rem 1rem; border-radius: 10px; margin-bottom: 0.3rem;}
        .gold {background-color: #FFD70022; border-left: 5px solid #FFD700;}
        .silver {background-color: #C0C0C022; border-left: 5px solid #C0C0C0;}
        .bronze {background-color: #CD7F3222; border-left: 5px solid #CD7F32;}
        .small-chv {font-size:0.9rem; color: #333;}
        footer {opacity: 0.8; font-size: 0.9rem;}
    </style>
    """, unsafe_allow_html=True
)

# ----------------------
# Sidebar / Navigation
# ----------------------
st.sidebar.title("🏥 Community Health Bridge")
st.sidebar.caption("Community Health Bridge - Data and Predictive")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📍 Navigate",
    [
        "Home",
        "Log Visit",
        "Data Manager",
        "Analytics",
        "Leaderboard",
        "Predictive Insights",
        "Reports",
        "Admin: API Key"
    ],
)

st.sidebar.markdown("---")

# ----------------------
# API key logic (short-lived)
# ----------------------
if "api_key" not in st.session_state:
    st.session_state.api_key = None
    st.session_state.api_key_expiry = None

def generate_api_key(minutes=5):
    st.session_state.api_key = str(uuid.uuid4())[:10]
    st.session_state.api_key_expiry = datetime.utcnow() + timedelta(minutes=minutes)

st.sidebar.markdown("### 🔐 API Key")
if st.sidebar.button("Generate Key (5 min)"):
    generate_api_key(5)
if st.session_state.api_key:
    remaining = int((st.session_state.api_key_expiry - datetime.utcnow()).total_seconds())
    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        st.sidebar.success(f"`{st.session_state.api_key}` — {mins}m {secs}s left")
    else:
        st.sidebar.warning("Key expired.")
        st.session_state.api_key = None
        st.session_state.api_key_expiry = None

# ----------------------
# Data loading utility
# ----------------------
@st.cache_data
def _load_visits():
    df = load_visits()
    if "Date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df

visits_df = _load_visits()

def default_date_range(df):
    if df.empty or "Date" not in df.columns:
        today = datetime.utcnow().date()
        return [today - timedelta(days=30), today]
    return [df["Date"].min(), df["Date"].max()]

# ----------------------
# Kenya counties (47)
# ----------------------
KENYA_COUNTIES = [
    "Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo-Marakwet", "Embu", "Garissa",
    "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho", "Kiambu", "Kilifi",
    "Kirinyaga", "Kisii", "Kisumu", "Kitui", "Kwale", "Laikipia", "Lamu", "Machakos",
    "Makueni", "Mandera", "Marsabit", "Meru", "Migori", "Mombasa", "Murang'a",
    "Nairobi", "Nakuru", "Nandi", "Narok", "Nyamira", "Nyandarua", "Nyeri",
    "Samburu", "Siaya", "Taita-Taveta", "Tana River", "Tharaka-Nithi",
    "Trans Nzoia", "Turkana", "Uasin Gishu", "Vihiga", "Wajir", "West Pokot"
]

# ----------------------
# (All remaining pages: Home, Log Visit, Data Manager, Analytics, Leaderboard,
# Predictive Insights, Reports, Admin: API Key, Footer)
# ----------------------
# ✅ Keep the rest of your original code EXACTLY as it is (no functional changes needed)
# Just ensure these imports at the top match your flat file structure.

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("**System by Simon Wanyoike** — Contact: symoprof83@gmail.com")
