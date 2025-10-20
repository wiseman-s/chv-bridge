# app_final_v4.py — CHV Bridge (Frontend + Data Manager + Predictive)
# Temporary session-only, editable Data Manager, mobile-safe charts
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
        # Map VisitType to incentive, ensure numeric Incentive column exists
        df["Incentive"] = df.get("VisitType", pd.Series([""] * len(df))).map(INCENTIVE_RULES).fillna(0)
        try:
            df["Incentive"] = pd.to_numeric(df["Incentive"], errors="coerce").fillna(0)
        except Exception:
            df["Incentive"] = df["Incentive"].astype(float)
        return df

# charts & reports modules may provide nicer plots; if absent use inline versions
try:
    from charts import plot_incentives_over_time, plot_incentives_by_type
except Exception:
    def plot_incentives_over_time(df):
        if df is None or df.empty:
            return px.line()
        df2 = df.copy()
        df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
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
st.sidebar.caption("Connecting communities with reliable health insights and nurturing informed communities.")
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
    st.session_state.session_visits = pd.DataFrame(columns=["CHV", "Client", "County", "VisitType", "Date", "Notes"])

# helpers to manage session dataset and trailing blank row
def get_visits_df():
    df = st.session_state.session_visits.copy()
    # Ensure Date typed consistently (datetime)
    if "Date" in df.columns and not df.empty:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        except Exception:
            pass
    # ensure required columns exist
    for col in ["CHV","Client","County","VisitType","Date","Notes"]:
        if col not in df.columns:
            df[col] = "" if col != "Date" else pd.NaT
    # ensure Incentive column if present later
    return df.reset_index(drop=True)

def set_visits_df(df):
    df = df.copy()
    # normalize columns and types
    for col in ["CHV","Client","County","VisitType","Notes"]:
        if col not in df.columns:
            df[col] = ""
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        except Exception:
            pass
    else:
        df["Date"] = pd.NaT
    # drop fully-empty rows except keep one blank trailing row
    def is_row_empty(r):
        empty_text = all([(pd.isna(r.get(c)) or str(r.get(c)).strip()=="") for c in ["CHV","Client","County","VisitType","Notes"]])
        date_empty = pd.isna(r.get("Date"))
        return empty_text and date_empty

    # keep rows that are not empty
    if df.empty:
        df_clean = pd.DataFrame(columns=["CHV","Client","County","VisitType","Date","Notes"])
    else:
        df_clean = df[~df.apply(is_row_empty, axis=1)].copy()
    # reindex
    df_clean = df_clean.reset_index(drop=True)
    # ensure trailing blank row exists
    blank_row = {"CHV":"", "Client":"", "County":"", "VisitType":"", "Date":pd.NaT, "Notes":""}
    if df_clean.empty:
        df_final = pd.DataFrame([blank_row])
    else:
        last = df_clean.iloc[-1]
        last_empty = ( (pd.isna(last.get("Date")) or str(last.get("Date")).strip()=="") and all([ (str(last.get(c)).strip()=="") for c in ["CHV","Client","County","VisitType","Notes"] ]) )
        if last_empty:
            df_final = df_clean
        else:
            df_final = pd.concat([df_clean, pd.DataFrame([blank_row])], ignore_index=True)
    st.session_state.session_visits = df_final.reset_index(drop=True)

def default_date_range(df):
    if df is None or df.empty or "Date" not in df.columns or df["Date"].dropna().empty:
        today = datetime.utcnow().date()
        return [today - timedelta(days=30), today]
    try:
        dmin = pd.to_datetime(df["Date"]).dt.date.min()
        dmax = pd.to_datetime(df["Date"]).dt.date.max()
        if pd.isna(dmin) or pd.isna(dmax):
            today = datetime.utcnow().date()
            return [today - timedelta(days=30), today]
        return [dmin, dmax]
    except Exception:
        today = datetime.utcnow().date()
        return [today - timedelta(days=30), today]

# ensure session dataset has trailing blank row at start
set_visits_df(get_visits_df())

# ----------------------
# HOME
# ----------------------
if page == "Home":
    st.title("🏥 Community Health Volunteer Bridge")
    st.markdown(
        """
        A unified platform empowering Community Health Volunteers to record visits, analyze incentives,
        and run predictive forecasts enabling data-driven decision-making for improved community health outcomes.
        Through CHV Bridge, you can:
    - 🏠 Log household visits and capture essential health data  
    - 📊 Visualize trends and analyze incentive performance  
    - 🤖 Generate predictive insights to forecast community needs  
    - 🧾 Export summaries and reports for easy sharing 
        """
    )
    st.info("Tip: Use Data Manager → add/edit rows (quick), then explore Analytics and Predictive Insights.")

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
            county = st.selectbox("County", KENYA_COUNTIES, index=KENYA_COUNTIES.index("Kiambu") if "Kiambu" in KENYA_COUNTIES else 0)
        with col2:
            visit_type = st.selectbox("Visit Type", options=list(INCENTIVE_RULES.keys()) if isinstance(INCENTIVE_RULES, dict) else [])
            date = st.date_input("Visit Date", value=datetime.utcnow().date())
            notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Add Visit (session only)")

    if submitted:
        new_row = {
            "CHV": chv.strip(),
            "Client": client.strip(),
            "County": county,
            "VisitType": visit_type,
            "Date": pd.to_datetime(date),
            "Notes": notes
        }
        df = get_visits_df()
        # append and persist in session (no permanent save)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        set_visits_df(df)
        st.success("Visit added to session dataset.")

    # Show current session dataset below the form (editable)
    st.markdown("**Below is data entered manually in this session (editable).**")
    df_show = get_visits_df()
    # ensure columns exist
    for col in ["CHV","Client","County","VisitType","Date","Notes"]:
        if col not in df_show.columns:
            df_show[col] = "" if col != "Date" else pd.NaT
    # enforce trailing blank row
    set_visits_df(df_show)
    try:
        edited = st.data_editor(get_visits_df(), num_rows="dynamic", use_container_width=True, key="data_editor_log_visit")
        if "Date" in edited.columns:
            try:
                edited["Date"] = pd.to_datetime(edited["Date"], errors="coerce")
            except Exception:
                pass
        set_visits_df(edited)
    except Exception:
        st.dataframe(get_visits_df(), use_container_width=True)

# ----------------------
# DATA MANAGER (editable grid, upload CSV to append, add-row & clear)
# ----------------------
elif page == "Data Manager":
    st.title("📥 Data Manager — Manual entry & Upload")
    st.markdown("Showing data entered manually on Log Visit (editable). You can also upload a CSV to append. All changes are temporary (session-only).")

    # top controls: add row, upload CSV (append), clear
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("➕ Add empty row"):
            df_tmp = get_visits_df()
            df_tmp = pd.concat([df_tmp, pd.DataFrame([{"CHV":"","Client":"","County":"","VisitType":"","Date":pd.NaT,"Notes":""}])], ignore_index=True)
            set_visits_df(df_tmp)
    with c2:
        uploaded = st.file_uploader("Upload CSV to append (only CSV)", type=["csv"], key="upload_csv_append")
        if uploaded is not None:
            if st.button("Append CSV"):
                try:
                    df_up = pd.read_csv(uploaded, parse_dates=["Date"], dayfirst=False)
                    for col in ["CHV","Client","County","VisitType","Date","Notes"]:
                        if col not in df_up.columns:
                            df_up[col] = "" if col != "Date" else pd.NaT
                    if "Date" in df_up.columns:
                        df_up["Date"] = pd.to_datetime(df_up["Date"], errors="coerce")
                    df_current = get_visits_df()
                    df_concat = pd.concat([df_current, df_up], ignore_index=True) if not df_current.empty else df_up.copy()
                    set_visits_df(df_concat)
                    st.success(f"Appended {len(df_up)} rows to session dataset.")
                except Exception as e:
                    st.error(f"Failed to append CSV: {e}")
    with c3:
        if st.button("🗑️ Clear all session data"):
            confirm = st.checkbox("Yes — clear all session data", key="confirm_clear_all")
            if confirm:
                set_visits_df(pd.DataFrame(columns=["CHV","Client","County","VisitType","Date","Notes"]))
                st.success("Session dataset cleared.")

    st.markdown("---")
    st.subheader("Current dataset (editable)")
    df_show = get_visits_df()
    # ensure columns exist
    for col in ["CHV","Client","County","VisitType","Date","Notes"]:
        if col not in df_show.columns:
            df_show[col] = "" if col != "Date" else pd.NaT

    # ensure trailing blank row exists before editing
    df_show = df_show.copy()
    set_visits_df(df_show)  # this will enforce trailing blank row

    # data_editor for inline editing (dynamic)
    try:
        edited = st.data_editor(get_visits_df(), num_rows="dynamic", use_container_width=True, key="data_editor_visits")
        # normalize Date
        if "Date" in edited.columns:
            try:
                edited["Date"] = pd.to_datetime(edited["Date"], errors="coerce")
            except Exception:
                pass
        # save edits to session, keeping one trailing blank row
        set_visits_df(edited)
    except Exception:
        # fallback to non-editable view if data_editor unavailable
        st.dataframe(get_visits_df(), use_container_width=True)

# ----------------------
# ANALYTICS
# ----------------------
elif page == "Analytics":
    st.title("📈 Incentive Analytics")
    visits_df = get_visits_df()
    if visits_df.empty or visits_df.dropna(how="all").empty:
        st.info("No data to analyze. Please add or upload data in Data Manager.")
    else:
        df_with_inc = add_incentives_column(visits_df.copy())
        if "Date" in df_with_inc.columns:
            df_with_inc["Date"] = pd.to_datetime(df_with_inc["Date"], errors="coerce")
        else:
            df_with_inc["Date"] = pd.NaT

        # ensure Incentive numeric
        if "Incentive" not in df_with_inc.columns:
            df_with_inc["Incentive"] = 0
        else:
            try:
                df_with_inc["Incentive"] = pd.to_numeric(df_with_inc["Incentive"], errors="coerce").fillna(0)
            except Exception:
                pass

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
            # apply filters
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
    if visits_df.empty or visits_df.dropna(how="all").empty:
        st.info("No visits found.")
    else:
        df_with_inc = add_incentives_column(visits_df.copy())
        if "Incentive" in df_with_inc.columns:
            try:
                df_with_inc["Incentive"] = pd.to_numeric(df_with_inc["Incentive"], errors="coerce").fillna(0)
            except Exception:
                pass
        else:
            df_with_inc["Incentive"] = 0
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
    if visits_df.empty or visits_df.dropna(how="all").empty:
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

        # ensure Incentive column exists and numeric
        if "Incentive" not in df_local.columns:
            df_local["Incentive"] = 0
        else:
            try:
                df_local["Incentive"] = pd.to_numeric(df_local["Incentive"], errors="coerce").fillna(0)
            except Exception:
                pass

        if target_choice == "Overall (all CHVs aggregated)":
            series = df_local.groupby(df_local["Date"].dt.date)["Incentive"].sum().reset_index()
        else:
            series = df_local[df_local["CHV"] == target_choice].groupby(df_local["Date"].dt.date)["Incentive"].sum().reset_index()

        # Basic sanity checks
        if series.empty or len(series) < 5:
            st.info("Insufficient data for reliable modeling . Add more visits to improve accuracy.")
        else:
            series.columns = ["Date", "Incentive"]
            series["Date"] = pd.to_datetime(series["Date"], errors="coerce")
            series = series.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            # After dropna, check length again
            if series.empty or len(series) < 5:
                st.info("Insufficient valid dated records for modeling (need at least 5).")
            else:
                series["ordinal"] = series["Date"].map(lambda x: x.toordinal())
                series["lag1"] = series["Incentive"].shift(1).fillna(method="bfill")
                # ensure numeric arrays safe
                try:
                    features = series[["ordinal", "lag1"]].astype(float).values
                except Exception:
                    features = np.array(series[["ordinal", "lag1"]].values, dtype=float)
                target = pd.to_numeric(series["Incentive"], errors="coerce").fillna(0).values

                split_idx = int(len(series) * train_ratio)
                X_train, X_test = features[:split_idx], features[split_idx:]
                y_train, y_test = target[:split_idx], target[split_idx:]

                results = []

                def train_and_forecast_ml(name):
                    # Guard: need non-empty training data
                    if len(X_train) == 0 or len(y_train) == 0:
                        return {"name": name, "model": None, "r2": float("nan"), "mae": float("nan"), "pred_df": pd.DataFrame(), "y_test": np.array([]), "y_pred_test": np.array([])}
                    if name == "Linear Regression":
                        m = LinearRegression()
                    elif name == "Random Forest":
                        m = RandomForestRegressor(n_estimators=200, random_state=42)
                    elif name == "Gradient Boosting":
                        m = GradientBoostingRegressor(n_estimators=200, random_state=42)
                    else:
                        raise ValueError("Unknown model: " + str(name))
                    try:
                        m.fit(X_train, y_train)
                    except Exception as e:
                        st.warning(f"{name} failed to train: {e}")
                        return {"name": name, "model": None, "r2": float("nan"), "mae": float("nan"), "pred_df": pd.DataFrame(), "y_test": y_test, "y_pred_test": np.array([])}
                    y_pred_test = m.predict(X_test) if len(X_test) > 0 else np.array([])
                    r2 = r2_score(y_test, y_pred_test) if len(y_test) > 0 and len(y_pred_test) == len(y_test) else float("nan")
                    mae = mean_absolute_error(y_test, y_pred_test) if len(y_test) > 0 and len(y_pred_test) == len(y_test) else float("nan")
                    last_lag = series["Incentive"].iloc[-1]
                    last_ordinal = series["ordinal"].iloc[-1]
                    preds = []
                    prev = last_lag
                    for i in range(1, periods + 1):
                        ordv = last_ordinal + i
                        Xf = np.array([[ordv, prev]])
                        try:
                            p = m.predict(Xf)[0]
                        except Exception:
                            p = prev  # conservative fallback
                        preds.append(max(0, float(p)))
                        prev = p
                    future_dates = [series["Date"].iloc[-1] + timedelta(days=i) for i in range(1, periods + 1)]
                    pred_df = pd.DataFrame({"Date": future_dates, "PredictedIncentive": preds})
                    return {"name": name, "model": m, "r2": r2, "mae": mae, "pred_df": pred_df, "y_test": y_test, "y_pred_test": y_pred_test}

                def train_and_forecast_prophet():
                    prophet_df = series[["Date", "Incentive"]].rename(columns={"Date": "ds", "Incentive": "y"})
                    m = Prophet()
                    try:
                        with st.spinner("Training Prophet..."):
                            m.fit(prophet_df)
                    except Exception as e:
                        st.warning(f"Prophet training failed: {e}")
                        return {"name": "Prophet", "model": None, "r2": float("nan"), "mae": float("nan"), "pred_df": pd.DataFrame()}
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
                    if res.get("model") is None:
                        st.write("- Model training failed or insufficient training data.")
                    else:
                        if not np.isnan(res["r2"]):
                            st.write(f"- R²: {res['r2']:.3f}")
                            st.write(f"- MAE: {res['mae']:.2f}")
                        else:
                            st.write("- No held-out test set available or insufficient test data.")

                    hist_df = series[["Date", "Incentive"]].copy()
                    pred_df = res["pred_df"] if "pred_df" in res else pd.DataFrame()

                    fig = px.line()
                    fig.add_scatter(x=hist_df["Date"], y=hist_df["Incentive"], mode="lines+markers", name="Historical")
                    if not pred_df.empty:
                        fig.add_scatter(x=pred_df["Date"], y=pred_df["PredictedIncentive"], mode="lines+markers", name="Forecast")
                    if show_actual_vs_pred and res.get("y_test") is not None and len(res.get("y_test"))>0:
                        try:
                            test_dates = series["Date"].iloc[split_idx:].reset_index(drop=True)
                            fig.add_scatter(x=test_dates, y=res["y_test"], mode="markers", name="Actual (test)")
                            fig.add_scatter(x=test_dates, y=res["y_pred_test"], mode="markers", name="Predicted (test)")
                        except Exception:
                            pass
                    fig.update_layout(title=f"{model_name} forecast for {target_choice}", xaxis_title="Date", yaxis_title="Incentives (KES)")
                    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

                    st.subheader("Forecast table (next days)")
                    if not pred_df.empty:
                        st.dataframe(pred_df.reset_index(drop=True), use_container_width=True)
                        st.download_button("⬇️ Download Forecast CSV", data=pred_df.to_csv(index=False).encode("utf-8"), file_name="chv_forecast.csv", mime="text/csv")
                    else:
                        st.info("No forecast available from this model.")

                elif model_name == "Prophet":
                    if not PROPHET_AVAILABLE:
                        st.error("Prophet is not installed. Run pip install prophet to enable.")
                    else:
                        res = train_and_forecast_prophet()
                        pred_df = res["pred_df"]
                        fig = px.line()
                        hist_df = series[["Date", "Incentive"]].copy()
                        fig.add_scatter(x=hist_df["Date"], y=hist_df["Incentive"], mode="lines+markers", name="Historical")
                        if not pred_df.empty:
                            fig.add_scatter(x=pred_df["Date"], y=pred_df["PredictedIncentive"], mode="lines+markers", name="Prophet Forecast")
                        fig.update_layout(title=f"Prophet forecast for {target_choice}", xaxis_title="Date", yaxis_title="Incentives (KES)")
                        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})
                        st.subheader("Forecast table (next days)")
                        if not pred_df.empty:
                            st.dataframe(pred_df.reset_index(drop=True), use_container_width=True)
                            st.download_button("⬇️ Download Forecast CSV", data=pred_df.to_csv(index=False).encode("utf-8"), file_name="chv_forecast_prophet.csv", mime="text/csv")
                        else:
                            st.info("No forecast available from Prophet.")

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
                        comp_rows.append({"Model": r.get("name"), "R2": r.get("r2"), "MAE": r.get("mae")})
                    comp_df = pd.DataFrame(comp_rows)
                    st.subheader("Model Comparison")
                    st.dataframe(comp_df, use_container_width=True)

                    fig = px.line()
                    hist_df = series[["Date", "Incentive"]].copy()
                    fig.add_scatter(x=hist_df["Date"], y=hist_df["Incentive"], mode="lines+markers", name="Historical")
                    for r in results:
                        dfp = r.get("pred_df", pd.DataFrame())
                        if dfp is not None and not dfp.empty and "PredictedIncentive" in dfp.columns:
                            fig.add_scatter(x=dfp["Date"], y=dfp["PredictedIncentive"], mode="lines+markers", name=r.get("name"))
                    fig.update_layout(title=f"Model comparison forecast for {target_choice}", xaxis_title="Date", yaxis_title="Incentives (KES)")
                    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

                    for r in results:
                        dfp = r.get("pred_df", pd.DataFrame())
                        if dfp is not None and not dfp.empty:
                            fname = f"forecast_{r['name'].replace(' ', '_').lower()}.csv"
                            st.download_button(f"⬇️ Download {r['name']} forecast CSV", data=dfp.to_csv(index=False).encode("utf-8"), file_name=fname, mime="text/csv")

# ----------------------
# REPORTS
# ----------------------
elif page == "Reports":
    st.title("📄 Reports & Export")
    visits_df = get_visits_df()
    if visits_df.empty or visits_df.dropna(how="all").empty:
        st.info("No data to export for selected filters.")
    else:
        df_with_inc = add_incentives_column(visits_df.copy())
        try:
            csv_bytes = df_with_inc.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Visits CSV", data=csv_bytes, file_name=f"chv_visits_session.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to prepare CSV: {e}")
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
    st.markdown("This guide explains how to prepare and upload your CSV files. Use the Data Manager page to add or edit data for the current session.")
    st.markdown("Expected columns: `CHV, Client, County, VisitType, Date, Notes` — Date format `YYYY-MM-DD` recommended.")
    st.markdown("Need help? Contact: **symoprof83@gmail.com**")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("<center><small>System by Simon Wanyoike — Contact: symoprof83@gmail.com</small></center>", unsafe_allow_html=True)
