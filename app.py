# app.py — CHV Bridge (Frontend + Data Manager + Predictive)
import streamlit as st
import pandas as pd
import numpy as np
import uuid
import time
from datetime import datetime, timedelta
from io import BytesIO

import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from data_ops import load_visits, save_visit, get_filtered_visits
from incentives import INCENTIVE_RULES, add_incentives_column
from charts import plot_incentives_over_time, plot_incentives_by_type
from reports import generate_pdf_report_bytes


# ----------------------
# Page config & CSS
# ----------------------
st.set_page_config(page_title="CHV Bridge MVP", page_icon="🏥", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        section[data-testid="stSidebar"] .css-ng1t4o {padding-top: 1rem;}
        .rank-box {padding: 0.5rem 1rem; border-radius: 10px; margin-bottom: 0.3rem;}
        .gold {background-color: #FFD70022; border-left: 5px solid #FFD700;}
        .silver {background-color: #C0C0C022; border-left: 5px solid #C0C0C0;}
        .bronze {background-color: #CD7F3222; border-left: 5px solid #CD7F32;}
    </style>
    """, unsafe_allow_html=True
)

# ----------------------
# Sidebar / Navigation
# ----------------------
st.sidebar.title("🏥 CHV Bridge")
st.sidebar.caption("Frontend-first MVP — Data & Predictive")
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

# Demo API key logic (frontend only)
if "api_key" not in st.session_state:
    st.session_state.api_key = None
    st.session_state.api_key_expiry = None

def generate_api_key_demo(minutes=5):
    st.session_state.api_key = str(uuid.uuid4())[:10]
    st.session_state.api_key_expiry = datetime.utcnow() + timedelta(minutes=minutes)

st.sidebar.markdown("### 🔐 API Key (demo)")
if st.sidebar.button("Generate Key (5 min)"):
    generate_api_key_demo(5)
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
    # ensure Date is date type
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
# HOME
# ----------------------
if page == "Home":
    st.title("🏥 CHV Bridge — Demo")
    st.markdown(
        """
        CHV Bridge is a demo frontend-first app for Community Health Volunteers.
        Use Data Manager to add or upload data. Use Predictive Insights to run simple forecasts.
        """
    )
    st.info("Try: Data Manager → upload CSV or add a manual row → then Predictive Insights to forecast incentives.")

# ----------------------
# LOG VISIT (quick form)
# ----------------------
elif page == "Log Visit":
    st.title("📝 Log a Health Visit")
    with st.form("visit_form", clear_on_submit=True):
        chv = st.text_input("CHV Name", value="Jane Doe")
        client = st.text_input("Client Name", value="Baby John")
        visit_type = st.selectbox("Visit Type", options=list(INCENTIVE_RULES.keys()))
        date = st.date_input("Visit Date", value=datetime.utcnow().date())
        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Submit Visit")

    if submitted:
        new_row = {
            "CHV": chv,
            "Client": client,
            "VisitType": visit_type,
            "Date": pd.to_datetime(date).strftime("%Y-%m-%d"),
            "Notes": notes
        }
        with st.spinner("Saving visit..."):
            save_visit(new_row)
            _load_visits.clear()
            visits_df = _load_visits()
            st.success("✅ Visit recorded in sample dataset.")
            st.rerun()

# ----------------------
# DATA MANAGER (manual + upload)
# ----------------------
elif page == "Data Manager":
    st.title("📥 Data Manager — Manual entry & Upload")
    st.markdown("You can add a single visit manually, or upload a CSV/Excel with visit rows.")
    st.markdown("Expected columns (CSV/Excel): `CHV,Client,VisitType,Date,Notes` (Date like YYYY-MM-DD)")

    st.subheader("1) Manual entry")
    with st.form("manual_entry", clear_on_submit=True):
        m_chv = st.text_input("CHV Name", value="Jane Doe")
        m_client = st.text_input("Client Name", value="Baby John")
        m_visit_type = st.selectbox("Visit Type", options=list(INCENTIVE_RULES.keys()))
        m_date = st.date_input("Visit Date", value=datetime.utcnow().date())
        m_notes = st.text_area("Notes (optional)")
        add_btn = st.form_submit_button("Add row to dataset")
    if add_btn:
        row = {
            "CHV": m_chv,
            "Client": m_client,
            "VisitType": m_visit_type,
            "Date": pd.to_datetime(m_date).strftime("%Y-%m-%d"),
            "Notes": m_notes
        }
        save_visit(row)
        _load_visits.clear()
        visits_df = _load_visits()
        st.success("Manual row added and saved to sample dataset.")
        st.rerun()

    st.markdown("---")
    st.subheader("2) Upload CSV / Excel (bulk)")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded, parse_dates=["Date"])
            else:
                df_up = pd.read_excel(uploaded, parse_dates=["Date"])
            # normalize Date column to date
            if "Date" in df_up.columns:
                df_up["Date"] = pd.to_datetime(df_up["Date"]).dt.date
            st.success(f"Preview of uploaded file ({len(df_up)} rows):")
            st.dataframe(df_up.head(), use_container_width=True)

            if st.button("Append uploaded rows to dataset"):
                # append rows to sample csv via save_visit per row
                for _, r in df_up.iterrows():
                    row = {
                        "CHV": r.get("CHV", ""),
                        "Client": r.get("Client", ""),
                        "VisitType": r.get("VisitType", ""),
                        "Date": pd.to_datetime(r.get("Date")).strftime("%Y-%m-%d") if not pd.isna(r.get("Date")) else datetime.utcnow().strftime("%Y-%m-%d"),
                        "Notes": r.get("Notes", "")
                    }
                    save_visit(row)
                _load_visits.clear()
                visits_df = _load_visits()
                st.success("Uploaded rows appended to dataset.")
                st.rerun()

        except Exception as e:
            st.error(f"Failed to read file: {e}")

    st.markdown("---")
    st.subheader("Current dataset (sample)")
    st.dataframe(visits_df.head(200), use_container_width=True)

# ----------------------
# ANALYTICS
# ----------------------
elif page == "Analytics":
    st.title("📈 Incentive Analytics")
    default_start, default_end = default_date_range(visits_df)
    col1, col2 = st.columns([3,1])
    with col1:
        start_end = st.date_input("Date range", value=[default_start, default_end])
        visit_types = st.multiselect(
            "Visit types",
            options=visits_df["VisitType"].unique().tolist() if not visits_df.empty else list(INCENTIVE_RULES.keys()),
            default=visits_df["VisitType"].unique().tolist() if not visits_df.empty else list(INCENTIVE_RULES.keys())
        )
    with col2:
        st.metric("Total Visits (dataset)", int(len(visits_df)))
        df_filtered = get_filtered_visits(visits_df, start_end[0], start_end[1], visit_types) if not visits_df.empty else pd.DataFrame()
        df_with_inc = add_incentives_column(df_filtered) if not df_filtered.empty else df_filtered
        st.metric("Filtered Visits", int(len(df_with_inc)))
        st.metric("Total Incentives (KES)", int(df_with_inc["Incentive"].sum()) if not df_with_inc.empty else 0)

    st.markdown("---")
    if df_with_inc.empty:
        st.info("No data for selected filters.")
    else:
        st.subheader("Incentives Over Time")
        st.plotly_chart(plot_incentives_over_time(df_with_inc), use_container_width=True)
        st.subheader("Incentives by Type")
        st.plotly_chart(plot_incentives_by_type(df_with_inc), use_container_width=True)

# ----------------------
# LEADERBOARD
# ----------------------
elif page == "Leaderboard":
    st.title("🏆 CHV Leaderboard")
    default_start, default_end = default_date_range(visits_df)
    start_end = st.date_input("Date range", value=[default_start, default_end], key="leader_range")
    df_filtered = get_filtered_visits(visits_df, start_end[0], start_end[1], visits_df["VisitType"].unique().tolist()) if not visits_df.empty else pd.DataFrame()
    df_with_inc = add_incentives_column(df_filtered) if not df_filtered.empty else df_filtered

    if df_with_inc.empty:
        st.info("No visits found.")
    else:
        leaderboard = df_with_inc.groupby("CHV", as_index=False)["Incentive"].sum().sort_values("Incentive", ascending=False)
        leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
        medals = ["🥇", "🥈", "🥉"]
        st.markdown("### Top Performers")
        for i, row in leaderboard.head(10).iterrows():
            rank = row["Rank"]
            medal = medals[i] if i < 3 else ""
            color_class = "gold" if i == 0 else ("silver" if i == 1 else ("bronze" if i == 2 else ""))
            st.markdown(f'<div class="rank-box {color_class}"><b>{medal} {rank}. {row["CHV"]}</b> — KES {int(row["Incentive"]):,}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Full Leaderboard")
        st.dataframe(leaderboard[["Rank", "CHV", "Incentive"]], use_container_width=True)

# ----------------------
# PREDICTIVE INSIGHTS
# ----------------------
elif page == "Predictive Insights":
    st.title("🤖 Predictive Insights — Forecast Incentives")
    st.markdown("""
    Simple forecasting demo. Models available:
    - **Linear Regression** (date ordinal → incentive)
    - **Random Forest Regressor** (nonlinear)
    
    We create a time series per CHV (daily aggregated) and model incentives over time.
    """)

    if visits_df.empty:
        st.info("No data available for modeling. Add data under Data Manager first.")
    else:
        # Options
        model_name = st.selectbox("Model", ["Linear Regression", "Random Forest"])
        target_choice = st.selectbox("Predict for", ["Overall (all CHVs aggregated)"] + sorted(visits_df["CHV"].unique().tolist()))
        periods = st.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30, step=7)
        train_ratio = st.slider("Train ratio (history used for training)", 0.5, 0.95, 0.8)

        # Prepare series: aggregate daily incentives
        df_local = add_incentives_column(visits_df.copy())  # ensures Incentive column present

        # convert Date to datetime
        df_local["Date"] = pd.to_datetime(df_local["Date"])

        if target_choice == "Overall (all CHVs aggregated)":
            series = df_local.groupby(df_local["Date"].dt.date)["Incentive"].sum().reset_index()
            series.columns = ["Date", "Incentive"]
        else:
            series = df_local[df_local["CHV"] == target_choice].groupby(df_local["Date"].dt.date)["Incentive"].sum().reset_index()
            series.columns = ["Date", "Incentive"]

        if series.empty or len(series) < 5:
            st.warning("Not enough data to build a reliable model (need at least 5 days of aggregated data).")
        else:
            series = series.sort_values("Date").reset_index(drop=True)
            # create features: date ordinal and optionally lag features
            series["Date_dt"] = pd.to_datetime(series["Date"])
            series["ordinal"] = series["Date_dt"].map(lambda x: x.toordinal())
            # simple lag features (previous day incentive)
            series["lag1"] = series["Incentive"].shift(1).fillna(method="bfill")
            # features and target
            features = series[["ordinal", "lag1"]].values
            target = series["Incentive"].values

            # split train/test
            split_idx = int(len(series) * train_ratio)
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = target[:split_idx], target[split_idx:]

            # choose model
            if model_name == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            with st.spinner("Training model..."):
                model.fit(X_train, y_train)

            # evaluate on test set
            y_pred = model.predict(X_test) if len(X_test) > 0 else np.array([])
            r2 = r2_score(y_test, y_pred) if len(y_test) > 0 else float("nan")
            mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else float("nan")

            st.subheader("Model performance")
            st.write(f"- Training points: {len(X_train)}  •  Test points: {len(X_test)}")
            if not np.isnan(r2):
                st.write(f"- R² on test set: **{r2:.3f}**")
                st.write(f"- MAE on test set: **{mae:.2f}**")
            else:
                st.info("No held-out test set (all data used for training).")

            # Forecast next N days
            last_date = series["Date_dt"].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
            future_ordinals = np.array([d.toordinal() for d in future_dates])
            # last known lag value: use last observed incentive
            last_lag = series["Incentive"].iloc[-1]
            # build feature matrix for future: use lag1 = previous predicted value iteratively
            preds = []
            prev = last_lag
            for ordv in future_ordinals:
                Xf = np.array([[ordv, prev]])
                p = model.predict(Xf)[0]
                preds.append(max(0, p))  # incentives can't be negative
                prev = p  # autoregressive style

            # Build DataFrame for plotting
            hist_df = series[["Date_dt", "Incentive"]].rename(columns={"Date_dt": "Date"})
            pred_df = pd.DataFrame({"Date": future_dates, "PredictedIncentive": preds})

            # Plot history + forecast
            fig = px.line()
            fig.add_scatter(x=hist_df["Date"], y=hist_df["Incentive"], mode="lines+markers", name="Historical")
            fig.add_scatter(x=pred_df["Date"], y=pred_df["PredictedIncentive"], mode="lines+markers", name="Forecast")
            fig.update_layout(title=f"Forecast ({model_name}) for {target_choice}", xaxis_title="Date", yaxis_title="Incentives (KES)")
            st.plotly_chart(fig, use_container_width=True)

            # Show prediction table & download
            st.subheader("Forecast table (next days)")
            out_df = pred_df.copy()
            out_df["PredictedIncentive"] = out_df["PredictedIncentive"].round(2)
            st.dataframe(out_df, use_container_width=True)

            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Forecast CSV", data=csv, file_name="chv_forecast.csv", mime="text/csv")

# ----------------------
# REPORTS
# ----------------------
elif page == "Reports":
    st.title("📄 Reports & Export")
    default_start, default_end = default_date_range(visits_df)
    start_end = st.date_input("Date range", value=[default_start, default_end])
    types = st.multiselect("Visit types", options=visits_df["VisitType"].unique().tolist() if not visits_df.empty else list(INCENTIVE_RULES.keys()), default=visits_df["VisitType"].unique().tolist() if not visits_df.empty else list(INCENTIVE_RULES.keys()))

    df_filtered = get_filtered_visits(visits_df, start_end[0], start_end[1], types) if not visits_df.empty else pd.DataFrame()
    df_with_inc = add_incentives_column(df_filtered) if not df_filtered.empty else df_filtered

    if df_with_inc.empty:
        st.info("No data to export for selected filters.")
    else:
        csv_bytes = df_with_inc.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Visits CSV", data=csv_bytes, file_name=f"chv_visits_{start_end[0]}_{start_end[1]}.csv", mime="text/csv")
        if st.button("📑 Generate PDF Summary"):
            with st.spinner("Creating PDF..."):
                try:
                    pdf_bytes = generate_pdf_report_bytes(df_with_inc)
                    st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=f"chv_summary_{start_end[0]}_{start_end[1]}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

# ----------------------
# ADMIN: API KEY
# ----------------------
elif page == "Admin: API Key":
    st.title("🔐 Demo API Key")
    st.markdown("Frontend-only short-lived API key generator (demo). In production move this to backend.")
    if not st.session_state.get("api_key"):
        if st.button("Generate API Key (5 min)"):
            generate_api_key_demo(5)
            st.success("API key generated — see sidebar.")
    else:
        remaining = int((st.session_state.api_key_expiry - datetime.utcnow()).total_seconds())
        if remaining > 0:
            mins, secs = divmod(remaining, 60)
            st.info(f"Key: `{st.session_state.api_key}` — expires in {mins}m {secs}s")
            if st.button("Invalidate Key"):
                st.session_state.api_key = None
                st.session_state.api_key_expiry = None
                st.success("API key invalidated.")
        else:
            st.warning("Key expired.")
            st.session_state.api_key = None
            st.session_state.api_key_expiry = None
