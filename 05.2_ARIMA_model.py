import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from datetime import timedelta
import pickle
import json
from scipy import stats
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import pickle

def ARIMA_training(df, name):
    df.to_json('ARIMA_historical.json', orient='index')
    # Choose the ARIMA Model parameters (p, d, q)
    # These should be chosen based on model diagnostics like ACF, PACF plots or grid search
    p = 2  # AR term
    d = 1  # Differencing order
    q = 1  # MA term

    model = ARIMA(df, order=(p, d, q))
    model_fit = model.fit()
    # Now, you can save the model to disk
    with open(name+'_ARIMA.pkl', 'wb') as pkl:
        pickle.dump(model_fit, pkl)

    split_point = int(len(df) * 0.90)  # For example, 80% for training and 20% for testing
    y_train, y_test = df['Price'][:split_point], df['Price'][split_point:]

    # Fit the ARIMA model
    model = ARIMA(y_train, order=(p, d, q))
    model_fit = model.fit()

    start_date = y_test.index[0]
    end_date = y_test.index[-1]

    y_pred = model_fit.predict(start=start_date, end=end_date)
    y_pred.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    print(1, y_pred)
    print(2, y_test)
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

    r_squared = r2_score(y_test, y_pred)
    print(77, r_squared)
    # Assuming `df` is your DataFrame with 'x' as the independent variable and 'y' as the dependent variable
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,y_pred)

    # The square of the correlation coefficient (r_value) is the R^2 score
    r2 = r_value ** 2
    r_squared = r2

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r_squared": r_squared
    }
    with open('ARIMA_metrics.json', 'w') as file:
        json.dump(metrics, file)
        # Fit the ARIMA model

    return model_fit.summary()

def ARIMA_forecasting(name, future_steps):
    # Forecast future values
    # Here 'future_steps' is the number of future periods you want to forecast
    with open(name+'_ARIMA.pkl', 'rb') as pkl:
        loaded_model = pickle.load(pkl)

    forecast = loaded_model.get_forecast(steps=future_steps)
    predicted_means = forecast.predicted_mean
    predicted_means.index = predicted_means.index.strftime('%Y-%m-%d')
    df = predicted_means.to_frame(name='Price')
    df.to_json('ARIMA_forecast.json', date_format='iso', orient='index')


if __name__ == "__main__":
    name = 'BTC-USD'
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']
    future_steps = 30
    summary = ARIMA_training(df, name)
    ARIMA_forecasting(name, 30)



