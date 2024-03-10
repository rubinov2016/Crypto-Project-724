import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import pickle

def ARIMA_forecasting(name, future_steps):
    # Forecast future values
    # Here 'future_steps' is the number of future periods you want to forecast
    with open(name+'_ARIMA.pkl', 'rb') as pkl:
        loaded_model = pickle.load(pkl)

    forecast = loaded_model.get_forecast(steps=future_steps)
    predicted_means = forecast.predicted_mean
    print(predicted_means)


if __name__ == "__main__":
    name = 'BTC-USD'
    future_steps = 30
    ARIMA_forecasting(name,30)