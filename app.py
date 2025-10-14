
# app_silent_submit.py — CHV Bridge (Frontend + Data Manager + Predictive) - Silent submits + UI polish + dark mode
import streamlit as st
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
from io import BytesIO

import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# Local modules (ensure these exist in the same folder or package)
from data_ops import load_visits, save_visit, get_filtered_visits
from incentives import INCENTIVE_RULES, add_incentives_column
from charts import plot_incentives_over_time, plot_incentives_by_type
from reports import generate_pdf_report_bytes

# ----------------------
# Page config & CSS (spacing + small polish + dark mode)
# ----------------------
st.set_page_config(page_title="Community Health Volunteer Bridge", page_icon="🏥", layout="wide")

st.markdown(
    """
    <style>
        :root { color-scheme: light dark; }
        [data-testid="stSidebarNav"] {display: none;}
        section[data-testid="stSidebar"] .css-ng1t4o {padding-top: 1rem;}
        .rank-box {padding: 0.6rem 1rem; border-radius: 10px; margin-bottom: 0.4rem; font-size:0.95rem;}
        .gold {background-color: #FFD70022; border-left: 5px solid #FFD700;}
        .silver {background-color: #C0C0C022; border-left: 5px solid #C0C0C0;}
        .bronze {background-color: #CD7F3222; border-left: 5px solid #CD7F32;}
        .small-chv {font-size:0.95rem; color: #333; margin-bottom: 0.5rem;}
        footer {opacity: 0.8; font-size: 0.9rem; margin-top: 1.5rem;}
        .muted {color: #666; font-size:0.9rem;}
        /* Make container adapt to dark mode nicely */
        [data-testid="stAppViewContainer"] {
            transition: background-color 0.3s ease, color 0.3s ease;
        }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------
# Sidebar / Navigation
# ----------------------
st.sidebar.title("🏥 Community Health Volunteer Bridge")
st.sidebar.caption("Connecting communities with reliable health insights.")
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
        "Upload Guide"
    ],
)

st.sidebar.markdown("---")

# ----------------------
# Data loading utility + compatibility fixes
# ----------------------
@st.cache_data
def _load_visits():
    try:
        df = load_visits()
        if df is None:
            df = pd.DataFrame(columns=["CHV","Client","County","VisitType","Date","Notes"])
    except Exception:
        df = pd.DataFrame(columns=["CHV","Client","County","VisitType","Date","Notes"])
    if "Date" in df.columns and not df.empty:
        try:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
        except Exception:
            pass
    return df

def reload_visits():
    """Clear cache and reload visits (call after save)."""
    try:
        _load_visits.clear()
    except Exception:
        pass
    return _load_visits()

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

def counties_options_from_df(df):
    present = sorted(df["County"].dropna().unique().tolist()) if "County" in df.columns and not df["County"].dropna().empty else []
    merged = [c for c in KENYA_COUNTIES if c not in present]
    return present + merged

# ----------------------
# HOME
# ----------------------
if page == "Home":
    st.title("🏥 Community Health Volunteer Bridge")
    st.markdown(
        """
        A unified platform empowering Community Health Volunteers to record visits, analyze incentives,
        and run predictive forecasts.
        """
    )
    st.info("Tip: Use Data Manager → Upload CSV to add bulk visit rows, then check Analytics / Predictive Insights.")

# ----------------------
# LOG VISIT (quick form)
# ----------------------
elif page == "Log Visit":
    st.title("📝 Log a Health Visit")
    with st.form("visit_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            chv = st.text_input("CHV Name", value="Jane Doe")
            client = st.text_input("Client Name", value="Baby John")
            county = st.selectbox("County", KENYA_COUNTIES, index=KENYA_COUNTIES.index("Migori") if "Migori" in KENYA_COUNTIES else 0)
        with col2:
            visit_type = st.selectbox("Visit Type", options=list(INCENTIVE_RULES.keys()))
            date = st.date_input("Visit Date", value=datetime.utcnow().date())
            notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Submit Visit")

    if submitted:
        new_row = {
            "CHV": chv,
            "Client": client,
            "County": county,
            "VisitType": visit_type,
            "Date": pd.to_datetime(date).strftime("%Y-%m-%d"),
            "Notes": notes
        }
        # save silently and refresh cached data without UI messages or reruns
        try:
            save_visit(new_row)
        except Exception:
            pass
        try:
            visits_df = reload_visits()
        except Exception:
            pass
        # intentionally do not show success or call rerun; remain on the page

# ----------------------
# DATA MANAGER (manual + upload)
# ----------------------
elif page == "Data Manager":
    st.title("📥 Data Manager — Manual entry & Upload")
    st.markdown("You can add a single visit manually, or upload a CSV/Excel with visit rows.")
    st.markdown("Expected columns (CSV/Excel): `CHV,Client,County,VisitType,Date,Notes` (Date like YYYY-MM-DD)")

    st.subheader("1) Manual entry")
    with st.form("manual_entry", clear_on_submit=True):
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            m_chv = st.text_input("CHV Name", value="Jane Doe", key="m_chv")
            m_client = st.text_input("Client Name", value="Baby John", key="m_client")
            m_county = st.selectbox("County", KENYA_COUNTIES, index=KENYA_COUNTIES.index("Migori") if "Migori" in KENYA_COUNTIES else 0, key="m_county")
        with m_col2:
            m_visit_type = st.selectbox("Visit Type", options=list(INCENTIVE_RULES.keys()), key="m_visit")
            m_date = st.date_input("Visit Date", value=datetime.utcnow().date(), key="m_date")
            m_notes = st.text_area("Notes (optional)", key="m_notes")
        add_btn = st.form_submit_button("Add row to dataset")
    if add_btn:
        row = {
            "CHV": m_chv,
            "Client": m_client,
            "County": m_county,
            "VisitType": m_visit_type,
            "Date": pd.to_datetime(m_date).strftime("%Y-%m-%d"),
            "Notes": m_notes
        }
        try:
            save_visit(row)
        except Exception:
            pass
        try:
            visits_df = reload_visits()
        except Exception:
            pass
        # intentionally silent: no success message, no rerun

    st.markdown("---")
    st.subheader("2) Upload CSV / Excel (bulk)")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded, parse_dates=["Date"], dayfirst=False)
            else:
                df_up = pd.read_excel(uploaded, parse_dates=["Date"])
            if "Date" in df_up.columns:
                df_up["Date"] = pd.to_datetime(df_up["Date"]).dt.date
            st.success(f"Preview of uploaded file ({len(df_up)} rows):")
            st.dataframe(df_up.head(), use_container_width=True)

            if st.button("Append uploaded rows to dataset"):
                for _, r in df_up.iterrows():
                    row = {
                        "CHV": r.get("CHV", ""),
                        "Client": r.get("Client", ""),
                        "County": r.get("County", "") if "County" in r else "",
                        "VisitType": r.get("VisitType", ""),
                        "Date": pd.to_datetime(r.get("Date")).strftime("%Y-%m-%d") if not pd.isna(r.get("Date")) else datetime.utcnow().strftime("%Y-%m-%d"),
                        "Notes": r.get("Notes", "")
                    }
                    try:
                        save_visit(row)
                    except Exception:
                        pass
                try:
                    visits_df = reload_visits()
                except Exception:
                    pass
                st.success("Uploaded rows appended to dataset.")
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
        county_options = counties_options_from_df(visits_df)
        counties = st.multiselect("Counties", options=county_options, default=county_options)
    with col2:
        st.metric("Total Visits (dataset)", int(len(visits_df)))
        df_filtered = get_filtered_visits(visits_df, start_end[0], start_end[1], visit_types) if not visits_df.empty else pd.DataFrame(columns=visits_df.columns)
        if "County" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["County"].isin(counties)]
        df_with_inc = add_incentives_column(df_filtered) if not df_filtered.empty else df_filtered
        st.metric("Filtered Visits", int(len(df_with_inc)))
        st.metric("Total Incentives (KES)", int(df_with_inc["Incentive"].sum()) if not df_with_inc.empty else 0)

    st.markdown("---")
    if df_with_inc.empty:
        st.info("No data for selected filters.")
    else:
        mean_inc = df_with_inc["Incentive"].mean() if "Incentive" in df_with_inc.columns and not df_with_inc.empty else 0
        top_chv = df_with_inc.groupby("CHV")["Incentive"].sum().idxmax() if not df_with_inc.empty else None
        top_county = df_with_inc.groupby("County")["Incentive"].sum().idxmax() if "County" in df_with_inc.columns and not df_with_inc.empty else None

        st.markdown(f"**Avg incentive (filtered):** KES {mean_inc:.2f}")
        if top_chv:
            st.markdown(f"**Top CHV (filtered):** {top_chv}")
        if top_county:
            st.markdown(f"**Top County (filtered):** {top_county}")

        st.subheader("Incentives Over Time")
        # staticPlot disables interactive touch drawing/zooming on mobile
        st.plotly_chart(plot_incentives_over_time(df_with_inc), use_container_width=True, config={"staticPlot": True})

        st.subheader("Incentives by Type")
        st.plotly_chart(plot_incentives_by_type(df_with_inc), use_container_width=True, config={"staticPlot": True})

        st.subheader("Incentives by County (Top 15)")
        county_agg = df_with_inc.groupby("County")["Incentive"].sum().reset_index().sort_values("Incentive", ascending=False).head(15)
        if county_agg.empty:
            st.info("No county-level data available for selected filters.")
        else:
            fig_county = px.bar(county_agg, x="County", y="Incentive", title="Top 15 Counties by Incentive")
            fig_county.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=40, b=100))
            st.plotly_chart(fig_county, use_container_width=True, config={"staticPlot": True})

# ----------------------
# LEADERBOARD
# ----------------------
elif page == "Leaderboard":
    st.title("🏆 CHV Leaderboard")
    default_start, default_end = default_date_range(visits_df)
    start_end = st.date_input("Date range", value=[default_start, default_end], key="leader_range")
    types_all = visits_df["VisitType"].unique().tolist() if not visits_df.empty else []
    df_filtered = get_filtered_visits(visits_df, start_end[0], start_end[1], types_all) if not visits_df.empty else pd.DataFrame(columns=visits_df.columns)
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
# PREDICTIVE INSIGHTS (Advanced Models + Comparison)
# ----------------------
elif page == "Predictive Insights":
    st.title("🤖 Predictive Insights — Advanced Models")
    st.markdown(
        """
        Models available:
        - Linear Regression (date ordinal → incentive)  
        - Random Forest Regressor (nonlinear)  
        - Gradient Boosting Regressor (ensemble)  
        - Prophet (time-series with seasonality/trend) — optional  
        """)
    if visits_df.empty:
        st.info("No data available for modeling. Add data under Data Manager first.")
    else:
        model_choices = ["Linear Regression", "Random Forest", "Gradient Boosting"]
        if PROPHET_AVAILABLE:
            model_choices.append("Prophet")
        model_name = st.selectbox("Model (select one or choose 'Compare Models' below)", model_choices + ["Compare Models"])
        target_choice = st.selectbox("Predict for", ["Overall (all CHVs aggregated)"] + sorted(visits_df["CHV"].unique().tolist()))
        periods = st.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30, step=7)
        train_ratio = st.slider("Train ratio (history used for training)", 0.5, 0.95, 0.8)
        show_actual_vs_pred = st.checkbox("Show Actual vs Predicted (when available)", value=True)

        # Prepare series: aggregate daily incentives and ensure Date is datetime
        df_local = add_incentives_column(visits_df.copy())
        df_local["Date"] = pd.to_datetime(df_local["Date"])

        if target_choice == "Overall (all CHVs aggregated)":
            series = df_local.groupby(df_local["Date"].dt.date)["Incentive"].sum().reset_index()
        else:
            series = df_local[df_local["CHV"] == target_choice].groupby(df_local["Date"].dt.date)["Incentive"].sum().reset_index()

        # Graceful message when not enough data (no hard requirement displayed)
        if series.empty:
            st.info("Insufficient data for reliable modeling yet — add more recent visits to improve accuracy.")
        else:
            series.columns = ["Date", "Incentive"]
            series["Date"] = pd.to_datetime(series["Date"])
            series = series.sort_values("Date").reset_index(drop=True)

            # features for ML
            series["ordinal"] = series["Date"].map(lambda x: x.toordinal())
            series["lag1"] = series["Incentive"].shift(1).fillna(method="bfill")
            features = series[["ordinal", "lag1"]].values
            target = series["Incentive"].values

            split_idx = int(len(series) * train_ratio)
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = target[:split_idx], target[split_idx:]

            results = []

            def train_and_forecast_ml(name):
                if name == "Linear Regression":
                    m = LinearRegression()
                elif name == "Random Forest":
                    m = RandomForestRegressor(n_estimators=200, random_state=42)
                elif name == "Gradient Boosting":
                    m = GradientBoostingRegressor(n_estimators=200, random_state=42)
                else:
                    raise ValueError("Unknown model: " + str(name))
                m.fit(X_train, y_train)
                y_pred_test = m.predict(X_test) if len(X_test) > 0 else np.array([])
                r2 = r2_score(y_test, y_pred_test) if len(y_test) > 0 else float("nan")
                mae = mean_absolute_error(y_test, y_pred_test) if len(y_test) > 0 else float("nan")
                last_lag = series["Incentive"].iloc[-1]
                last_ordinal = series["ordinal"].iloc[-1]
                preds = []
                prev = last_lag
                for i in range(1, periods + 1):
                    ordv = last_ordinal + i
                    Xf = np.array([[ordv, prev]])
                    p = m.predict(Xf)[0]
                    preds.append(max(0, p))
                    prev = p
                future_dates = [series["Date"].iloc[-1] + timedelta(days=i) for i in range(1, periods + 1)]
                pred_df = pd.DataFrame({"Date": future_dates, "PredictedIncentive": preds})
                return {"name": name, "model": m, "r2": r2, "mae": mae, "pred_df": pred_df, "y_test": y_test, "y_pred_test": y_pred_test}

            def train_and_forecast_prophet():
                prophet_df = series[["Date", "Incentive"]].rename(columns={"Date": "ds", "Incentive": "y"})
                m = Prophet()
                with st.spinner("Training Prophet..."):
                    m.fit(prophet_df)
                future = m.make_future_dataframe(periods=periods)
                forecast = m.predict(future)
                forecast_future = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                pred_df = forecast_future.rename(columns={"ds": "Date", "yhat": "PredictedIncentive", "yhat_lower": "Lower", "yhat_upper": "Upper"})
                return {"name": "Prophet", "model": m, "r2": float("nan"), "mae": float("nan"), "pred_df": pred_df}

            # Single model selection
            if model_name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
                st.info(f"Training {model_name}...")
                res = train_and_forecast_ml(model_name)
                st.subheader("Model performance")
                if not np.isnan(res["r2"]):
                    st.write(f"- R²: {res['r2']:.3f}")
                    st.write(f"- MAE: {res['mae']:.2f}")
                else:
                    st.write("- No held-out test set available.")

                hist_df = series[["Date", "Incentive"]].copy()
                pred_df = res["pred_df"]

                fig = px.line()
                fig.add_scatter(x=hist_df["Date"], y=hist_df["Incentive"], mode="lines+markers", name="Historical")
                fig.add_scatter(x=pred_df["Date"], y=pred_df["PredictedIncentive"], mode="lines+markers", name="Forecast")
                if show_actual_vs_pred and len(res["y_test"])>0:
                    test_dates = series["Date"].iloc[split_idx:].reset_index(drop=True)
                    fig.add_scatter(x=test_dates, y=res["y_test"], mode="markers", name="Actual (test)")
                    fig.add_scatter(x=test_dates, y=res["y_pred_test"], mode="markers", name="Predicted (test)")
                fig.update_layout(title=f"{model_name} forecast for {target_choice}", xaxis_title="Date", yaxis_title="Incentives (KES)")
                st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

                st.subheader("Forecast table (next days)")
                st.dataframe(pred_df.reset_index(drop=True), use_container_width=True)
                st.download_button("⬇️ Download Forecast CSV", data=pred_df.to_csv(index=False).encode("utf-8"), file_name="chv_forecast.csv", mime="text/csv")

            elif model_name == "Prophet":
                if not PROPHET_AVAILABLE:
                    st.error("Prophet is not installed. Run pip install prophet to enable.")
                else:
                    res = train_and_forecast_prophet()
                    pred_df = res["pred_df"]
                    fig = px.line()
                    hist_df = series[["Date", "Incentive"]].copy()
                    fig.add_scatter(x=hist_df["Date"], y=hist_df["Incentive"], mode="lines+markers", name="Historical")
                    fig.add_scatter(x=pred_df["Date"], y=pred_df["PredictedIncentive"], mode="lines+markers", name="Prophet Forecast")
                    fig.update_layout(title=f"Prophet forecast for {target_choice}", xaxis_title="Date", yaxis_title="Incentives (KES)")
                    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})
                    st.subheader("Forecast table (next days)")
                    st.dataframe(pred_df.reset_index(drop=True), use_container_width=True)
                    st.download_button("⬇️ Download Forecast CSV", data=pred_df.to_csv(index=False).encode("utf-8"), file_name="chv_forecast_prophet.csv", mime="text/csv")

            else:
                st.info("Training and comparing Linear Regression, Random Forest, and Gradient Boosting (Prophet included if available).")
                model_list = ["Linear Regression", "Random Forest", "Gradient Boosting"]
                results = []
                for mn in model_list:
                    results.append(train_and_forecast_ml(mn))
                if PROPHET_AVAILABLE:
                    results.append(train_and_forecast_prophet())

                comp_rows = []
                for r in results:
                    comp_rows.append({"Model": r["name"], "R2": r["r2"], "MAE": r["mae"]})
                comp_df = pd.DataFrame(comp_rows)
                st.subheader("Model Comparison")
                st.dataframe(comp_df, use_container_width=True)

                fig = px.line()
                hist_df = series[["Date", "Incentive"]].copy()
                fig.add_scatter(x=hist_df["Date"], y=hist_df["Incentive"], mode="lines+markers", name="Historical")
                for r in results:
                    dfp = r["pred_df"]
                    if "PredictedIncentive" in dfp.columns:
                        fig.add_scatter(x=dfp["Date"], y=dfp["PredictedIncentive"], mode="lines+markers", name=r["name"])
                fig.update_layout(title=f"Model comparison forecast for {target_choice}", xaxis_title="Date", yaxis_title="Incentives (KES)")
                st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

                for r in results:
                    fname = f"forecast_{r['name'].replace(' ', '_').lower()}.csv"
                    st.download_button(f"⬇️ Download {r['name']} forecast CSV", data=r["pred_df"].to_csv(index=False).encode("utf-8"), file_name=fname, mime="text/csv")

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
# Upload Guide
# ----------------------
elif page == "Upload Guide":
    st.title("📤 Upload Guide")
    st.markdown("This simple guide replaces the old API key demo. Use the Data Manager page to upload CSV/Excel files with the following columns:")
    st.markdown("`CHV, Client, County, VisitType, Date, Notes` — Date should be `YYYY-MM-DD`.")
    st.markdown("If your file doesn't include `County`, you can still upload; later map or edit rows manually using the Data Manager manual entry.")
    st.markdown("Need help? Contact: **symoprof83@gmail.com**")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("<center><small>System by Simon Wanyoike — Contact: symoprof83@gmail.com</small></center>", unsafe_allow_html=True)
