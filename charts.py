import plotly.express as px
import pandas as pd

def plot_incentives_over_time(df):
    if df.empty:
        return px.line(pd.DataFrame({"Date":[],"Incentive":[]}), x="Date", y="Incentive", title="No data")
    agg = df.groupby("Date")["Incentive"].sum().reset_index()
    return px.line(agg, x="Date", y="Incentive", title="Incentives Over Time")

def plot_incentives_by_type(df):
    if df.empty:
        return px.bar(pd.DataFrame({"VisitType":[],"Incentive":[]}), x="VisitType", y="Incentive", title="No data")
    agg = df.groupby("VisitType")["Incentive"].sum().reset_index()
    return px.bar(agg, x="VisitType", y="Incentive", title="Incentives by Visit Type")
