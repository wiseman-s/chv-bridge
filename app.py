# app.py — Optimized CHV Bridge
# FIXED:
# ✅ White screen
# ✅ Sluggish navigation
# ✅ Endless "Connecting..."
# ✅ Heavy reruns
# ✅ Expensive dataframe mutations
# ✅ Slow ML models
# ✅ Expensive uploads
# ✅ Reduced Plotly lag

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# OPTIONAL PROPHET
# -----------------------------
ENABLE_PROPHET = False

if ENABLE_PROPHET:
    try:
        from prophet import Prophet
        PROPHET_AVAILABLE = True
    except:
        PROPHET_AVAILABLE = False
else:
    PROPHET_AVAILABLE = False

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="CHV Bridge",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# LIGHT CSS
# -----------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.rank-box {
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 8px;
    background: rgba(0,0,0,0.04);
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# INCENTIVES
# -----------------------------
INCENTIVE_RULES = {
    "Antenatal Visit": 100,
    "Postnatal Visit": 150,
    "Child Immunization": 120,
    "Household Visit": 80
}

# -----------------------------
# CACHE FUNCTIONS
# -----------------------------
@st.cache_data
def add_incentives(df):
    df = df.copy()

    df["Incentive"] = (
        df["VisitType"]
        .map(INCENTIVE_RULES)
        .fillna(0)
    )

    return df


@st.cache_data
def load_analytics(df):
    return add_incentives(df)


# -----------------------------
# SESSION DATA
# -----------------------------
if "visits" not in st.session_state:

    st.session_state.visits = pd.DataFrame(columns=[
        "CHV",
        "Client",
        "County",
        "VisitType",
        "Date",
        "Notes"
    ])

# -----------------------------
# COUNTIES
# -----------------------------
KENYA_COUNTIES = [
    "Nairobi",
    "Kiambu",
    "Kisumu",
    "Mombasa",
    "Nakuru",
    "Machakos",
    "Kakamega",
    "Nyeri"
]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🏥 CHV Bridge")

page = st.sidebar.selectbox(
    "Navigate",
    [
        "Home",
        "Log Visit",
        "Data Manager",
        "Analytics",
        "Leaderboard",
        "Predictive Insights",
        "Reports",
        "Upload Guide"
    ]
)

# =========================================================
# HOME
# =========================================================
if page == "Home":

    st.title("🏥 Community Health Volunteer Bridge")

    st.success("Fast Optimized Version")

    st.markdown("""
    ### Features
    - Log visits
    - Analyze incentives
    - Predict trends
    - Export reports
    """)

# =========================================================
# LOG VISIT
# =========================================================
elif page == "Log Visit":

    st.title("📝 Log Visit")

    with st.form("visit_form"):

        col1, col2 = st.columns(2)

        with col1:
            chv = st.text_input("CHV Name")
            client = st.text_input("Client Name")
            county = st.selectbox("County", KENYA_COUNTIES)

        with col2:
            visit_type = st.selectbox(
                "Visit Type",
                list(INCENTIVE_RULES.keys())
            )

            visit_date = st.date_input(
                "Visit Date",
                value=datetime.today()
            )

            notes = st.text_area("Notes")

        submitted = st.form_submit_button("Add Visit")

    if submitted:

        new_row = pd.DataFrame([{
            "CHV": chv,
            "Client": client,
            "County": county,
            "VisitType": visit_type,
            "Date": pd.to_datetime(visit_date),
            "Notes": notes
        }])

        st.session_state.visits = pd.concat(
            [st.session_state.visits, new_row],
            ignore_index=True
        )

        st.success("Visit added successfully.")

# =========================================================
# DATA MANAGER
# =========================================================
elif page == "Data Manager":

    st.title("📥 Data Manager")

    st.markdown("Edit or upload data.")

    # -----------------------------
    # FILE UPLOAD FORM
    # -----------------------------
    with st.form("upload_form"):

        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"]
        )

        upload_btn = st.form_submit_button("Append CSV")

    if upload_btn and uploaded is not None:

        try:
            df_upload = pd.read_csv(uploaded)

            st.session_state.visits = pd.concat(
                [st.session_state.visits, df_upload],
                ignore_index=True
            )

            st.success("CSV uploaded successfully.")

        except Exception as e:
            st.error(str(e))

    # -----------------------------
    # EDITOR
    # -----------------------------
    edited_df = st.data_editor(
        st.session_state.visits,
        use_container_width=True,
        num_rows="dynamic",
        key="main_editor"
    )

    st.session_state.visits = edited_df

    # -----------------------------
    # CLEAR BUTTON
    # -----------------------------
    if st.button("🗑️ Clear Dataset"):

        st.session_state.visits = pd.DataFrame(columns=[
            "CHV",
            "Client",
            "County",
            "VisitType",
            "Date",
            "Notes"
        ])

        st.success("Dataset cleared.")

# =========================================================
# ANALYTICS
# =========================================================
elif page == "Analytics":

    st.title("📊 Analytics")

    if st.session_state.visits.empty:

        st.info("No data available.")

    else:

        with st.spinner("Loading analytics..."):

            df = load_analytics(st.session_state.visits)

            df["Date"] = pd.to_datetime(df["Date"])

            # -----------------------------
            # METRICS
            # -----------------------------
            c1, c2, c3 = st.columns(3)

            c1.metric("Total Visits", len(df))

            c2.metric(
                "Total Incentives",
                f"KES {int(df['Incentive'].sum()):,}"
            )

            c3.metric(
                "Average Incentive",
                f"KES {int(df['Incentive'].mean())}"
            )

            # -----------------------------
            # CHARTS
            # -----------------------------
            st.subheader("Incentives Over Time")

            daily = df.groupby("Date")["Incentive"].sum()

            st.line_chart(daily)

            st.subheader("Visit Types")

            visit_counts = df["VisitType"].value_counts()

            st.bar_chart(visit_counts)

# =========================================================
# LEADERBOARD
# =========================================================
elif page == "Leaderboard":

    st.title("🏆 Leaderboard")

    if st.session_state.visits.empty:

        st.info("No data available.")

    else:

        df = load_analytics(st.session_state.visits)

        leaderboard = (
            df.groupby("CHV")["Incentive"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        leaderboard.index += 1

        for idx, row in leaderboard.iterrows():

            st.markdown(f"""
            <div class="rank-box">
            <b>{idx}. {row['CHV']}</b>
            — KES {int(row['Incentive'])}
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# PREDICTIVE INSIGHTS
# =========================================================
elif page == "Predictive Insights":

    st.title("🤖 Predictive Insights")

    if len(st.session_state.visits) < 5:

        st.warning("Need at least 5 records.")

    else:

        df = load_analytics(st.session_state.visits)

        df["Date"] = pd.to_datetime(df["Date"])

        series = (
            df.groupby("Date")["Incentive"]
            .sum()
            .reset_index()
        )

        series["ordinal"] = series["Date"].map(datetime.toordinal)

        X = series[["ordinal"]]
        y = series["Incentive"]

        model_name = st.selectbox(
            "Select Model",
            [
                "Linear Regression",
                "Random Forest",
                "Gradient Boosting"
            ]
        )

        # -----------------------------
        # FAST MODELS
        # -----------------------------
        if model_name == "Linear Regression":

            model = LinearRegression()

        elif model_name == "Random Forest":

            model = RandomForestRegressor(
                n_estimators=50,
                random_state=42
            )

        else:

            model = GradientBoostingRegressor(
                n_estimators=50,
                random_state=42
            )

        with st.spinner("Training model..."):

            model.fit(X, y)

            preds = model.predict(X)

            r2 = r2_score(y, preds)

            mae = mean_absolute_error(y, preds)

        # -----------------------------
        # METRICS
        # -----------------------------
        c1, c2 = st.columns(2)

        c1.metric("R² Score", f"{r2:.3f}")

        c2.metric("MAE", f"{mae:.2f}")

        # -----------------------------
        # FORECAST
        # -----------------------------
        future_days = 30

        last_date = series["Date"].max()

        future_dates = [
            last_date + timedelta(days=i)
            for i in range(1, future_days + 1)
        ]

        future_ordinals = np.array([
            d.toordinal() for d in future_dates
        ]).reshape(-1, 1)

        future_preds = model.predict(future_ordinals)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": future_preds
        })

        # -----------------------------
        # CHART
        # -----------------------------
        fig = px.line(
            forecast_df,
            x="Date",
            y="Forecast",
            title="30-Day Forecast"
        )

        fig.update_layout(
            uirevision=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(forecast_df)

# =========================================================
# REPORTS
# =========================================================
elif page == "Reports":

    st.title("📄 Reports")

    if st.session_state.visits.empty:

        st.info("No data available.")

    else:

        csv = st.session_state.visits.to_csv(index=False)

        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="chv_visits.csv",
            mime="text/csv"
        )

# =========================================================
# UPLOAD GUIDE
# =========================================================
elif page == "Upload Guide":

    st.title("📤 Upload Guide")

    st.markdown("""
    ### Expected CSV Columns

    - CHV
    - Client
    - County
    - VisitType
    - Date
    - Notes

    ### Recommended Date Format

    YYYY-MM-DD
    """)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption("CHV Bridge — Optimized Performance Version")
