from prophet import Prophet
import numpy as np
import pandas as pd
import pickle
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
import json
from scipy import stats

def Prophet_training(df, name, test_n):
    df = df.reset_index()
    # Rename the new column to 'ds'
    df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'}, inplace=True)
    model = Prophet(changepoint_prior_scale=0.5)
    model.fit(df)
    with open(name+'_Prophet.pkl', 'wb') as pkl:
        pickle.dump(model, pkl)

    # model = Prophet(daily_seasonality=True) # daily_seasonality might depend on the data frequency
    df_train = df[:test_n]
    df_test = df[test_n:]
    y_test = df_test['y']
    y_test = y_test.reset_index(drop=True)

    model = Prophet(changepoint_prior_scale=0.5)
    model.fit(df_train)

    predict = model.predict(df_test)
    y_pred = predict['yhat']
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    # Calculate MAPE - Avoid division by zero by adding a small number to the denominator
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
    r2 = r_value ** 2
    r_squared = r2

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r_squared": r_squared
    }
    with open('Prophet_metrics.json', 'w') as file:
        json.dump(metrics, file)

def Prophet_forecasting(df, name, future_steps):
    df.to_json('Prophet_historical.json', orient='index')
    with open(name + '_Prophet.pkl', 'rb') as pkl:
         model = pickle.load(pkl)
    # Make a future dataframe for future predictions
    future = model.make_future_dataframe(periods=future_steps)
    # Make predictions
    forecast = model.predict(future)
    future_forecast = forecast[forecast['ds'] > df.index.max()]

    # Save forecast to a CSV file
    prediction = pd.DataFrame(future_forecast)
    prediction['ds'] = prediction['ds'].dt.strftime('%Y-%m-%d')
    prediction.set_index('ds', inplace=True)
    prediction.rename(columns={'yhat': 'Price'}, inplace=True)
    prediction = prediction[['Price']]
    prediction.to_json('Prophet_forecast.json', date_format='iso', orient='index')

    # Plot the forecast
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)

if __name__ == "__main__":
    name = 'BTC-USD'
    future_steps = 30
    df = pd.read_csv('crypto_data_clean2.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']
    test_n = -30
    Prophet_training(df, name, test_n)
    Prophet_forecasting(df, name, future_steps)