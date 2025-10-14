# app_final_v3.py — CHV Bridge (Frontend + Data Manager + Predictive) - Temporary session-only, editable Data Manager, mobile-safe charts
import streamlit as st
import pandas as pd
import numpy as np
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

# Local modules (kept for compatibility; not used for persistence)
# If these modules are missing, the app will still work because persistence is session-only.
try:
    from incentives import INCENTIVE_RULES, add_incentives_column
except Exception:
    # fallback incentive rules if incentives module not available
    INCENTIVE_RULES = {
        "Antenatal Visit": 100,
        "Postnatal Visit": 150,
        "Child Immunization": 120,
        "Household Visit": 80
    }

    def add_incentives_column(df):
        if df is None or df.empty:
            return df
        df = df.copy()
        # if VisitType missing, create zero incentives
        df["Incentive"] = df.get("VisitType", pd.Series([""] * len(df))).map(INCENTIVE_RULES).fillna(0)
        return df

# charts & reports modules may provide nicer plots; if absent use inline versions
try:
    from charts import plot_incentives_over_time, plot_incentives_by_type
except Exception:
    def plot_incentives_over_time(df):
        if df is None or df.empty:
            return px.line()
        df2 = df.copy()
        df2["Date"] = pd.to_datetime(df2["Date"])
        daily = df2.groupby(df2["Date"].dt.date)["Incentive"].sum().reset_index()
        daily["Date"] = pd.to_datetime(daily["Date"])
        fig = px.line(daily, x="Date", y="Incentive", markers=True, title="Incentives Over Time")
        return fig

    def plot_incentives_by_type(df):
        if df is None or df.empty:
            return px.bar()
        df2 = df.copy()
        by_type = df2.groupby("VisitType")["Incentive"].sum().reset_index()
        fig = px.bar(by_type, x="VisitType", y="Incentive", title="Incentives by Visit Type")
        return fig

try:
    from reports import generate_pdf_report_bytes
except Exception:
    generate_pdf_report_bytes = None  # PDF generation optional

# ----------------------
# Page config & CSS
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
# Session-only dataset initialization (empty by default)
# ----------------------
if "session_visits" not in st.session_state:
    # start with an empty temporary dataset for this session
    st.session_state.session_visits = pd.DataFrame(columns=["CHV", "Client", "County", "VisitType", "Date", "Notes"])

# handy alias used across the app (always reflect session state)
def get_visits_df():
    df = st.session_state.session_visits.copy()
    # ensure Date column exists and typed consistently
    if "Date" in df.columns and not df.empty:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            pass
    return df

def set_visits_df(df):
    # normalize Date to datetime or keep as-is
    df = df.copy()
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            pass
    st.session_state.session_visits = df.reset_index(drop=True)

visits_df = get_visits_df()

def default_date_range(df):
    if df.empty or "Date" not in df.columns:
        today = datetime.utcnow().date()
        return [today - timedelta(days=30), today]
    # ensure date-only for defaults
    dmin = pd.to_datetime(df["Date"]).dt.date.min()
    dmax = pd.to_datetime(df["Date"]).dt.date.max()
    return [dmin, dmax]

# ----------------------
# HOME
# ----------------------
if page == "Home":
    st.title("🏥 Community Health Volunteer Bridge")
    st.markdown(
        """
        A unified platform empowering Community Health Volunteers to record visits, analyze incentives,
        and run predictive forecasts. All changes are temporary and stored only for this session.
        """
    )
    st.info("Tip: Use Data Manager → upload or edit your dataset, then explore Analytics and Predictive Insights.")

# ----------------------
# LOG VISIT (adds to session-only dataset)
# ----------------------
elif page == "Log Visit":
    st.title("📝 Log a Health Visit")
    with st.form("visit_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            chv = st.text_input("CHV Name", value="")
            client = st.text_input("Client Name", value="")
            county = st.selectbox("County", KENYA_COUNTIES, index=KENYA_COUNTIES.index("Migori") if "Migori" in KENYA_COUNTIES else 0)
        with col2:
            visit_type = st.selectbox("Visit Type", options=list(INCENTIVE_RULES.keys()) if isinstance(INCENTIVE_RULES, dict) else [])
            date = st.date_input("Visit Date", value=datetime.utcnow().date())
            notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Add Visit (session only)")

    if submitted:
        # require at least CHV and VisitType and Date to add
        new_row = {
            "CHV": chv.strip(),
            "Client": client.strip(),
            "County": county,
            "VisitType": visit_type,
            "Date": pd.to_datetime(date),
            "Notes": notes
        }
        # append to session dataset (temporary)
        df = get_visits_df()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        set_visits_df(df)
        # silent: no rerun, no persistent save

# ----------------------
# DATA MANAGER (editable grid, upload/replace/append/clear)
# ----------------------
elif page == "Data Manager":
    st.title("📥 Data Manager — Manual entry & Upload")
    st.markdown("Upload, edit, append or clear your session dataset. All changes are temporary and stored only for this browser session.")

    # Top control panel: Replace upload, Append upload, Save session (no-op confirmation), Clear with confirmation
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        uploaded_replace = st.file_uploader("Upload dataset to replace session dataset (CSV or XLSX)", type=["csv", "xlsx"], key="replace_uploader")
        if uploaded_replace is not None:
            if st.button("📂 Replace session dataset with uploaded file"):
                try:
                    if uploaded_replace.name.lower().endswith(".csv"):
                        df_new = pd.read_csv(uploaded_replace, parse_dates=["Date"], dayfirst=False)
                    else:
                        df_new = pd.read_excel(uploaded_replace, parse_dates=["Date"])
                    # ensure columns exist
                    for col in ["CHV","Client","County","VisitType","Date","Notes"]:
                        if col not in df_new.columns:
                            df_new[col] = ""
                    # normalize date
                    if "Date" in df_new.columns:
                        df_new["Date"] = pd.to_datetime(df_new["Date"], errors="coerce")
                    set_visits_df(df_new)
                    visits_df = get_visits_df()
                    st.success(f"Session dataset replaced with uploaded file ({len(df_new)} rows).")
                except Exception as e:
                    st.error(f"Failed to load uploaded file: {e}")

    with c2:
        uploaded_append = st.file_uploader("Upload dataset to append to session (CSV or XLSX)", type=["csv", "xlsx"], key="append_uploader")
        if uploaded_append is not None:
            if st.button("➕ Append uploaded rows to session dataset"):
                try:
                    if uploaded_append.name.lower().endswith(".csv"):
                        df_up = pd.read_csv(uploaded_append, parse_dates=["Date"], dayfirst=False)
                    else:
                        df_up = pd.read_excel(uploaded_append, parse_dates=["Date"])
                    for col in ["CHV","Client","County","VisitType","Date","Notes"]:
                        if col not in df_up.columns:
                            df_up[col] = ""
                    if "Date" in df_up.columns:
                        df_up["Date"] = pd.to_datetime(df_up["Date"], errors="coerce")
                    df_current = get_visits_df()
                    df_concat = pd.concat([df_current, df_up], ignore_index=True) if not df_current.empty else df_up.copy()
                    set_visits_df(df_concat)
                    st.success(f"Appended {len(df_up)} rows to session dataset.")
                except Exception as e:
                    st.error(f"Failed to append file: {e}")

    with c3:
        if st.button("💾 Save dataset for this session"):
            st.success("Dataset saved in session memory (temporary).")

        if st.button("🗑️ Clear session dataset"):
            # confirmation step
            if st.checkbox("Yes — I understand this will clear the session dataset", key="confirm_clear"):
                set_visits_df(pd.DataFrame(columns=["CHV","Client","County","VisitType","Date","Notes"]))
                st.success("Session dataset cleared.")

    st.markdown("---")
    st.subheader("Current dataset (editable)")
    # show editable grid where user can edit, add, delete rows inline
    df_show = get_visits_df()
    # ensure columns present
    for col in ["CHV","Client","County","VisitType","Date","Notes"]:
        if col not in df_show.columns:
            df_show[col] = ""

    # Use Streamlit data editor for inline editing (dynamic rows)
    try:
        edited = st.data_editor(df_show, num_rows="dynamic", use_container_width=True, key="data_editor_visits")
        # keep Date as datetime where possible
        if "Date" in edited.columns:
            try:
                edited["Date"] = pd.to_datetime(edited["Date"], errors="coerce")
            except Exception:
                pass
        set_visits_df(edited)
    except Exception:
        # fallback if st.data_editor not available
        st.dataframe(df_show, use_container_width=True)

# ----------------------
# ANALYTICS
# ----------------------
elif page == "Analytics":
    st.title("📈 Incentive Analytics")
    visits_df = get_visits_df()
    if visits_df.empty:
        st.info("No data to analyze. Please add or upload data in Data Manager.")
    else:
        # ensure Incentive column
        df_with_inc = add_incentives_column(visits_df.copy())
        # normalize Date
        if "Date" in df_with_inc.columns:
            df_with_inc["Date"] = pd.to_datetime(df_with_inc["Date"], errors="coerce")
        else:
            df_with_inc["Date"] = pd.NaT

        default_start, default_end = default_date_range(df_with_inc)
        col1, col2 = st.columns([3,1])
        with col1:
            start_end = st.date_input("Date range", value=[default_start, default_end])
            visit_types = st.multiselect(
                "Visit types",
                options=df_with_inc["VisitType"].dropna().unique().tolist() if not df_with_inc.empty else [],
                default=df_with_inc["VisitType"].dropna().unique().tolist() if not df_with_inc.empty else []
            )
            county_options = counties_options_from_df(df_with_inc)
            counties = st.multiselect("Counties", options=county_options, default=county_options)
        with col2:
            st.metric("Total Visits (dataset)", int(len(df_with_inc)))
            # filter
            df_filtered = df_with_inc.copy()
            # date filter
            try:
                start_dt = pd.to_datetime(start_end[0])
                end_dt = pd.to_datetime(start_end[1])
                df_filtered = df_filtered[(df_filtered["Date"].dt.date >= start_dt.date()) & (df_filtered["Date"].dt.date <= end_dt.date())]
            except Exception:
                pass
            if visit_types:
                df_filtered = df_filtered[df_filtered["VisitType"].isin(visit_types)]
            if "County" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["County"].isin(counties)]
            st.metric("Filtered Visits", int(len(df_filtered)))
            st.metric("Total Incentives (KES)", int(df_filtered["Incentive"].sum()) if "Incentive" in df_filtered.columns and not df_filtered.empty else 0)

        st.markdown("---")
        if df_filtered.empty:
            st.info("No data after applying filters.")
        else:
            # summary metrics
            mean_inc = df_filtered["Incentive"].mean() if "Incentive" in df_filtered.columns else 0
            top_chv = df_filtered.groupby("CHV")["Incentive"].sum().idxmax() if not df_filtered.empty and "CHV" in df_filtered.columns else None
            top_county = df_filtered.groupby("County")["Incentive"].sum().idxmax() if not df_filtered.empty and "County" in df_filtered.columns else None

            st.markdown(f"**Avg incentive (filtered):** KES {mean_inc:.2f}")
            if top_chv:
                st.markdown(f"**Top CHV (filtered):** {top_chv}")
            if top_county:
                st.markdown(f"**Top County (filtered):** {top_county}")

            st.subheader("Incentives Over Time")
            fig_time = plot_incentives_over_time(df_filtered)
            st.plotly_chart(fig_time, use_container_width=True, config={"staticPlot": True})

            st.subheader("Incentives by Type")
            fig_type = plot_incentives_by_type(df_filtered)
            st.plotly_chart(fig_type, use_container_width=True, config={"staticPlot": True})

            st.subheader("Incentives by County (Top 15)")
            county_agg = df_filtered.groupby("County")["Incentive"].sum().reset_index().sort_values("Incentive", ascending=False).head(15)
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
    visits_df = get_visits_df()
    if visits_df.empty:
        st.info("No visits found.")
    else:
        df_with_inc = add_incentives_column(visits_df.copy())
        leaderboard = df_with_inc.groupby("CHV", as_index=False)["Incentive"].sum().sort_values("Incentive", ascending=False)
        if not leaderboard.empty:
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
        else:
            st.info("No leaderboard data available.")

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
        """
    )
    visits_df = get_visits_df()
    if visits_df.empty:
        st.info("No data available for modeling. Add data under Data Manager first.")
    else:
        model_choices = ["Linear Regression", "Random Forest", "Gradient Boosting"]
        if PROPHET_AVAILABLE:
            model_choices.append("Prophet")
        model_name = st.selectbox("Model (select one or choose 'Compare Models' below)", model_choices + ["Compare Models"])
        target_choice = st.selectbox("Predict for", ["Overall (all CHVs aggregated)"] + sorted(visits_df["CHV"].dropna().unique().tolist()))
        periods = st.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30, step=7)
        train_ratio = st.slider("Train ratio (history used for training)", 0.5, 0.95, 0.8)
        show_actual_vs_pred = st.checkbox("Show Actual vs Predicted (when available)", value=True)

        df_local = add_incentives_column(visits_df.copy())
        if "Date" in df_local.columns:
            df_local["Date"] = pd.to_datetime(df_local["Date"], errors="coerce")
        else:
            st.error("No Date column present in dataset.")
            df_local["Date"] = pd.NaT

        if target_choice == "Overall (all CHVs aggregated)":
            series = df_local.groupby(df_local["Date"].dt.date)["Incentive"].sum().reset_index()
        else:
            series = df_local[df_local["CHV"] == target_choice].groupby(df_local["Date"].dt.date)["Incentive"].sum().reset_index()

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
                # iterative forecast
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
    visits_df = get_visits_df()
    if visits_df.empty:
        st.info("No data to export for selected filters.")
    else:
        # attempt to add incentives column for exports
        df_with_inc = add_incentives_column(visits_df.copy())
        csv_bytes = df_with_inc.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Visits CSV", data=csv_bytes, file_name=f"chv_visits_session.csv", mime="text/csv")
        if generate_pdf_report_bytes is not None:
            if st.button("📑 Generate PDF Summary"):
                with st.spinner("Creating PDF..."):
                    try:
                        pdf_bytes = generate_pdf_report_bytes(df_with_inc)
                        st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=f"chv_summary_session.pdf", mime="application/pdf")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
        else:
            st.info("PDF generation not available (reports module missing).")

# ----------------------
# Upload Guide
# ----------------------
elif page == "Upload Guide":
    st.title("📤 Upload Guide")
    st.markdown("This guide explains how to prepare and upload your CSV/Excel files. Use the Data Manager page to upload or edit data for the current session.")
    st.markdown("Expected columns: `CHV, Client, County, VisitType, Date, Notes` — Date format `YYYY-MM-DD` recommended.")
    st.markdown("Need help? Contact: **symoprof83@gmail.com**")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("<center><small>System by Simon Wanyoike — Contact: symoprof83@gmail.com</small></center>", unsafe_allow_html=True)
