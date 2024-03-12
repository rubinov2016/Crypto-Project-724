import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from joblib import load as joblib_load
import numpy as np


sequence_length = 30  # Example: 30 days
# Function to predict a single future step


if __name__ == "__main__":
    name = 'BTC-USD'
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']
    # df.set_index('Date', inplace=True)  # Ensure Date is the index
    scaler_name = name+'_scaler.joblib'

    dataset_name = name + '_crypto_data.csv'
    future_steps = 30
    LSTM_forecasting(df,30, keras_name, future_steps)

