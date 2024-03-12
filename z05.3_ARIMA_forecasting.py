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
    predicted_means.index = predicted_means.index.strftime('%Y-%m-%d')
    df = predicted_means.to_frame(name='Price')
    return df


if __name__ == "__main__":
    name = 'BTC-USD'
    future_steps = 30
    predicted_means = ARIMA_forecasting(name,30)
    print(type(predicted_means))
    print(predicted_means)
    predicted_means.to_json('ARIMA_forecast.json', date_format='iso', orient='index')

    # name = 'BTC-USD'
    # df = pd.read_csv('crypto_data_clean.csv')
    # df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    # df.index.name = 'Date'
    # df.columns = ['Price']
    # # df.set_index('Date', inplace=True)  # Ensure Date is the index
    # scaler_name = name + '_scaler.joblib'
    # keras_name = name + '_LTSM.keras'
    # dataset_name = name + '_crypto_data.csv'
    # future_steps = 30
    # predictions = LSTM_forecasting(df, 30, scaler_name, keras_name, dataset_name, future_steps)
    # df.to_json('LSTM_historical.json', orient='index')
    #