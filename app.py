"""
End-to-End Sales Forecasting & Demand Intelligence System
Interactive Streamlit Dashboard

Author: Ritvik Gulati
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -----------------------------------------------------------------------
# Page config and light color theme
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide",
)

LIGHT_CSS = """
<style>
.stApp {
    background-color: #fbfcfe;
}
section[data-testid="stSidebar"] {
    background-color: #f1f6fb;
}
h1, h2, h3 {
    color: #234e70;
}
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e3ecf5;
    border-radius: 10px;
    padding: 12px;
}
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)

PALETTE = ["#7fb3d5", "#a9dfbf", "#f5cba7", "#c39bd3", "#f9e79f", "#aed6f1"]
LIGHT_TEMPLATE = "plotly_white"


# -----------------------------------------------------------------------
# Data loading and shared feature engineering (cached)
# -----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    def season(m):
        if m in (12, 1, 2):
            return "Winter"
        elif m in (3, 4, 5):
            return "Spring"
        elif m in (6, 7, 8):
            return "Summer"
        else:
            return "Fall"

    df["Season"] = df["Month"].apply(season)
    return df


@st.cache_data
def get_daily_series(_df):
    return _df.groupby("Order Date")["Sales"].sum().asfreq("D").fillna(0)


@st.cache_data
def get_monthly_series(_df, category=None, region=None):
    d = _df
    if category:
        d = d[d["Category"] == category]
    if region:
        d = d[d["Region"] == region]
    daily = d.groupby("Order Date")["Sales"].sum().asfreq("D").fillna(0)
    return daily.resample("MS").sum()


@st.cache_data
def get_weekly_series(_df):
    daily = get_daily_series(_df)
    return daily.resample("W").sum()


def make_season_code(m):
    if m in (12, 1, 2):
        return 0
    elif m in (3, 4, 5):
        return 1
    elif m in (6, 7, 8):
        return 2
    else:
        return 3


@st.cache_data(show_spinner=False)
def sarima_forecast(values_tuple, index_str_tuple, steps):
    series = pd.Series(list(values_tuple), index=pd.to_datetime(list(index_str_tuple)))

    test_horizon = min(3, max(1, len(series) // 6))
    train_series = series.iloc[:-test_horizon]
    test_series = series.iloc[-test_horizon:]

    model = SARIMAX(
        train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    test_pred = model.get_forecast(steps=test_horizon).predicted_mean

    mae = mean_absolute_error(test_series, test_pred)
    rmse = mean_squared_error(test_series, test_pred) ** 0.5

    full_model = SARIMAX(
        series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    forecast_obj = full_model.get_forecast(steps=steps)
    mean_forecast = forecast_obj.predicted_mean
    ci = forecast_obj.conf_int()

    return mean_forecast, ci, mae, rmse


@st.cache_data
def compute_anomalies(_df):
    weekly = get_weekly_series(_df)
    a_df = weekly.to_frame(name="Sales").copy()

    iso = IsolationForest(contamination=0.08, random_state=42)
    a_df["iso_anomaly"] = iso.fit_predict(a_df[["Sales"]]) == -1

    roll_window = 8
    a_df["rolling_mean"] = a_df["Sales"].rolling(roll_window, min_periods=1).mean()
    a_df["rolling_std"] = a_df["Sales"].rolling(roll_window, min_periods=1).std().fillna(0)
    a_df["z_score"] = (a_df["Sales"] - a_df["rolling_mean"]) / a_df["rolling_std"].replace(0, np.nan)
    a_df["z_anomaly"] = a_df["z_score"].abs() > 2

    return a_df


@st.cache_data
def compute_clusters(_df):
    features = []
    for sub in _df["Sub-Category"].unique():
        sub_df = _df[_df["Sub-Category"] == sub]
        monthly = sub_df.groupby(sub_df["Order Date"].dt.to_period("M"))["Sales"].sum()
        total_sales = sub_df["Sales"].sum()
        order_value = sub_df.groupby("Order ID")["Sales"].sum().mean()
        volatility = monthly.std()

        yearly = sub_df.groupby("Year")["Sales"].sum().sort_index()
        if len(yearly) >= 2 and yearly.iloc[0] > 0:
            growth = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100
        else:
            growth = 0

        features.append({
            "Sub-Category": sub, "TotalSales": total_sales,
            "GrowthRate": growth, "Volatility": volatility,
            "AvgOrderValue": order_value,
        })

    feat_df = pd.DataFrame(features).set_index("Sub-Category")
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df)

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    feat_df["Cluster"] = kmeans.fit_predict(X)

    profile = feat_df.groupby("Cluster")[["TotalSales", "GrowthRate", "Volatility", "AvgOrderValue"]].mean()

    def label_cluster(row):
        high_volume = row["TotalSales"] >= profile["TotalSales"].median()
        high_volatility = row["Volatility"] >= profile["Volatility"].median()
        growing = row["GrowthRate"] >= profile["GrowthRate"].median()
        if high_volume and not high_volatility:
            return "High Volume, Stable Demand"
        elif not high_volume and high_volatility:
            return "Low Volume, High Volatility"
        elif growing:
            return "Growing Demand"
        else:
            return "Declining Demand"

    profile["Label"] = profile.apply(label_cluster, axis=1)
    label_map = profile["Label"].to_dict()
    feat_df["ClusterLabel"] = feat_df["Cluster"].map(label_map)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    feat_df["PC1"] = coords[:, 0]
    feat_df["PC2"] = coords[:, 1]

    return feat_df, profile


# -----------------------------------------------------------------------
# Load everything once
# -----------------------------------------------------------------------
df = load_data()

st.sidebar.title("Sales Forecasting")
st.sidebar.caption("End-to-End Demand Intelligence System")
page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Author: Ritvik Gulati")


# -----------------------------------------------------------------------
# Page 1: Sales Overview Dashboard
# -----------------------------------------------------------------------
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    col3.metric("Date Range", f"{df['Order Date'].min().year} to {df['Order Date'].max().year}")

    st.markdown("### Total Sales by Year")
    yearly = df.groupby("Year")["Sales"].sum().reset_index()
    fig = px.bar(yearly, x="Year", y="Sales", template=LIGHT_TEMPLATE,
                 color_discrete_sequence=["#7fb3d5"])
    fig.update_layout(yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("### Monthly Sales Trend")
    monthly = get_monthly_series(df)
    fig2 = px.line(x=monthly.index, y=monthly.values, template=LIGHT_TEMPLATE,
                    color_discrete_sequence=["#5499c7"])
    fig2.update_layout(xaxis_title="Month", yaxis_title="Sales ($)")
    st.plotly_chart(fig2, use_container_width=True, theme=None)

    st.markdown("### Sales by Region and Category")
    fcol1, fcol2 = st.columns(2)
    region_filter = fcol1.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    category_filter = fcol2.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)]
    grouped = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig3 = px.bar(grouped, x="Region", y="Sales", color="Category", barmode="group",
                  template=LIGHT_TEMPLATE, color_discrete_sequence=PALETTE)
    fig3.update_layout(yaxis_title="Sales ($)")
    st.plotly_chart(fig3, use_container_width=True, theme=None)


# -----------------------------------------------------------------------
# Page 2: Forecast Explorer
# -----------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Forecasts are generated using SARIMA, the model selected as best in Task 3 based on held-out RMSE.")

    dim_type = st.selectbox("Select dimension type", ["Category", "Region"])
    if dim_type == "Category":
        options = sorted(df["Category"].unique())
    else:
        options = sorted(df["Region"].unique())
    dim_value = st.selectbox(f"Select {dim_type}", options)

    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    if dim_type == "Category":
        series = get_monthly_series(df, category=dim_value)
    else:
        series = get_monthly_series(df, region=dim_value)

    with st.spinner("Fitting SARIMA model..."):
        values_tuple = tuple(float(v) for v in series.values)
        index_str_tuple = tuple(d.strftime("%Y-%m-%d") for d in series.index)
        mean_forecast, ci, mae, rmse = sarima_forecast(values_tuple, index_str_tuple, horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Actual",
                              line=dict(color="#5499c7")))
    fig.add_trace(go.Scatter(x=mean_forecast.index, y=mean_forecast.values, name="Forecast",
                              line=dict(color="#e59866"), mode="lines+markers"))
    fig.add_trace(go.Scatter(
        x=list(ci.index) + list(ci.index[::-1]),
        y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
        fill="toself", fillcolor="rgba(229,152,102,0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="Confidence Interval", showlegend=True,
    ))
    fig.update_layout(template=LIGHT_TEMPLATE, title=f"{dim_value} Sales Forecast",
                       xaxis_title="Month", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("### Model Accuracy")
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("MAE", f"${mae:,.0f}")
    mcol2.metric("RMSE", f"${rmse:,.0f}")

    st.markdown("### Forecast Values")
    forecast_table = pd.DataFrame({
        "Month": [d.strftime("%b %Y") for d in mean_forecast.index],
        "Forecasted Sales ($)": mean_forecast.values.round(2),
        "Lower CI": ci.iloc[:, 0].values.round(2),
        "Upper CI": ci.iloc[:, 1].values.round(2),
    })
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------
# Page 3: Anomaly Report
# -----------------------------------------------------------------------
elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Two independent methods flag unusual weeks: Isolation Forest and rolling Z-score.")

    a_df = compute_anomalies(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a_df.index, y=a_df["Sales"], name="Weekly Sales",
                              line=dict(color="#5499c7")))
    iso_points = a_df[a_df["iso_anomaly"]]
    fig.add_trace(go.Scatter(x=iso_points.index, y=iso_points["Sales"], mode="markers",
                              name="Isolation Forest Anomaly",
                              marker=dict(color="#e74c3c", size=10)))
    z_points = a_df[a_df["z_anomaly"]]
    fig.add_trace(go.Scatter(x=z_points.index, y=z_points["Sales"], mode="markers",
                              name="Z-Score Anomaly",
                              marker=dict(color="#f39c12", size=10, symbol="diamond")))
    fig.update_layout(template=LIGHT_TEMPLATE, xaxis_title="Week", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True, theme=None)

    col1, col2, col3 = st.columns(3)
    col1.metric("Isolation Forest Flags", int(a_df["iso_anomaly"].sum()))
    col2.metric("Z-Score Flags", int(a_df["z_anomaly"].sum()))
    col3.metric("Flagged by Both", int((a_df["iso_anomaly"] & a_df["z_anomaly"]).sum()))

    st.markdown("### Detected Anomaly Weeks")
    anomaly_rows = a_df[a_df["iso_anomaly"] | a_df["z_anomaly"]].copy()
    anomaly_rows["Week"] = anomaly_rows.index.strftime("%Y-%m-%d")
    anomaly_rows["Flagged By"] = anomaly_rows.apply(
        lambda r: "Both" if r["iso_anomaly"] and r["z_anomaly"]
        else ("Isolation Forest" if r["iso_anomaly"] else "Z-Score"), axis=1,
    )
    st.dataframe(
        anomaly_rows[["Week", "Sales", "Flagged By"]].sort_values("Sales", ascending=False),
        use_container_width=True, hide_index=True,
    )


# -----------------------------------------------------------------------
# Page 4: Product Demand Segments
# -----------------------------------------------------------------------
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")
    st.caption("Sub-categories grouped by demand behavior using K-Means clustering on total sales, growth rate, volatility, and average order value.")

    feat_df, profile = compute_clusters(df)

    label_colors = {
        "High Volume, Stable Demand": "#7fb3d5",
        "Low Volume, High Volatility": "#f5cba7",
        "Growing Demand": "#a9dfbf",
        "Declining Demand": "#c39bd3",
    }
    fig = px.scatter(
        feat_df.reset_index(), x="PC1", y="PC2", color="ClusterLabel",
        text="Sub-Category", template=LIGHT_TEMPLATE,
        color_discrete_map=label_colors,
    )
    fig.update_traces(textposition="top center", marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("### Sub-Categories by Demand Cluster")
    display_df = feat_df.reset_index()[
        ["Sub-Category", "ClusterLabel", "TotalSales", "GrowthRate", "Volatility", "AvgOrderValue"]
    ].sort_values("ClusterLabel")
    display_df.columns = ["Sub-Category", "Demand Segment", "Total Sales ($)", "Growth Rate (%)",
                           "Volatility", "Avg Order Value ($)"]
    st.dataframe(display_df.round(1), use_container_width=True, hide_index=True)

    st.markdown("### Recommended Stocking Strategy")
    strategy = {
        "High Volume, Stable Demand": "Keep steady safety stock with a simple reorder-point system, stockouts here are the most costly.",
        "Low Volume, High Volatility": "Stock conservatively, reorder more frequently in smaller batches rather than holding large inventory.",
        "Growing Demand": "Increase stock ahead of the trend, review monthly since the growth rate may keep shifting.",
        "Declining Demand": "Wind stock down deliberately, avoid large reorders, consider clearance rather than carrying dead inventory.",
    }
    for label, advice in strategy.items():
        st.markdown(f"**{label}:** {advice}")
