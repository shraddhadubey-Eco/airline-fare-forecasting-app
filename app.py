import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Airline Fare Forecasting Application",
    layout="wide"
)

st.title("✈ Airline Fare Forecasting Application")
st.caption("Time-Series Forecasting of Airline Fare Levels and Volatility")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/airline_fares.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

data = load_data()

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Analysis",
    [
        "OLS Regression",
        "ARIMA Forecast",
        "GARCH Volatility",
        "Model Comparison"
    ]
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Months)",
    min_value=6,
    max_value=24,
    value=12
)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(data.head())

# ==================================================
# OLS REGRESSION (FULL OUTPUT)
# ==================================================
if model_choice == "OLS Regression":

    st.subheader("📊 Ordinary Least Squares (OLS) Regression")
    st.markdown("**Dependent variable:** Airline Fare  \n**Independent variable:** Fuel Price")

    # Model
    X = data[["Fuel"]]
    X = sm.add_constant(X)
    y = data["Fare"]

    ols_model = sm.OLS(y, X).fit()

    # --------------------------------------------------
    # FULL OLS SUMMARY
    # --------------------------------------------------
    st.markdown("### 🧾 Complete OLS Regression Output")

    summary_text = ols_model.summary().as_text()
    st.code(summary_text, language="text")

    # Download button (bonus)
    st.download_button(
        label="⬇ Download OLS Summary",
        data=summary_text,
        file_name="OLS_Regression_Summary.txt",
        mime="text/plain"
    )

    # --------------------------------------------------
    # ACTUAL VS PREDICTED PLOT
    # --------------------------------------------------
    st.markdown("### 📈 Actual vs Predicted Fare")

    data["OLS_Predicted"] = ols_model.predict(X)

    fig, ax = plt.subplots()
    ax.plot(data["Date"], data["Fare"], label="Actual Fare")
    ax.plot(
        data["Date"],
        data["OLS_Predicted"],
        linestyle="--",
        label="Predicted Fare"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Fare")
    ax.legend()
    st.pyplot(fig)

# ==================================================
# ARIMA FORECAST
# ==================================================
elif model_choice == "ARIMA Forecast":

    st.subheader("📈 ARIMA Fare Forecasting")

    fare_series = data["Fare"]

    arima_model = ARIMA(fare_series, order=(1, 1, 1))
    arima_fit = arima_model.fit()

    forecast = arima_fit.forecast(steps=forecast_horizon)

    st.markdown("### 🔮 Forecasted Fare Values")
    st.dataframe(forecast.rename("Forecasted Fare"))

    fig, ax = plt.subplots()
    ax.plot(fare_series.values, label="Historical Fare")
    ax.plot(
        range(len(fare_series), len(fare_series) + forecast_horizon),
        forecast.values,
        linestyle="--",
        label="ARIMA Forecast"
    )
    ax.set_ylabel("Fare")
    ax.legend()
    st.pyplot(fig)

# ==================================================
# GARCH VOLATILITY
# ==================================================
elif model_choice == "GARCH Volatility":

    st.subheader("📉 Volatility Modeling using GARCH(1,1)")

    returns = data["Fare"].pct_change().dropna() * 100

    garch_model = arch_model(returns, vol="Garch", p=1, q=1)
    garch_fit = garch_model.fit(disp="off")

    volatility = garch_fit.conditional_volatility

    fig, ax = plt.subplots()
    ax.plot(volatility, color="red")
    ax.set_title("Conditional Volatility")
    ax.set_ylabel("Volatility")
    st.pyplot(fig)

# ==================================================
# MODEL COMPARISON
# ==================================================
elif model_choice == "Model Comparison":

    st.subheader("📊 Model Performance Comparison")

    # OLS
    X = sm.add_constant(data[["Fuel"]])
    y = data["Fare"]
    ols_model = sm.OLS(y, X).fit()
    ols_pred = ols_model.predict(X)

    # ARIMA
    arima_model = ARIMA(y, order=(1, 1, 1)).fit()
    arima_pred = arima_model.predict(start=1, end=len(y) - 1)

    y_true = y[1:]

    comparison_df = pd.DataFrame({
        "Model": ["OLS", "ARIMA"],
        "MAE": [
            mean_absolute_error(y_true, ols_pred[1:]),
            mean_absolute_error(y_true, arima_pred)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_true, ols_pred[1:])),
            np.sqrt(mean_squared_error(y_true, arima_pred))
        ]
    })

    st.dataframe(
        comparison_df.style
        .format({"MAE": "{:.3f}", "RMSE": "{:.3f}"})
        .background_gradient(cmap="Greens")
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Academic Project: Time-Series Forecasting of Airline Fare Levels and Volatility")
