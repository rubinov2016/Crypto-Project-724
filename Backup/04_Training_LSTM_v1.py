import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

def LSTM_training(name):
    df = pd.read_csv('crypto_data_clean.csv')
    df_filtered = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df_filtered.index.name = 'Date'
    df_filtered.columns = ['Price']

    df_pivoted = df_filtered
    df_pivoted.to_csv(name+'_crypto_data.csv', header=True)
    # Normalize  data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_filtered)
    dump(scaler, name+'_scaler.joblib')
    # Function to create sequences for LSTM


    def create_sequences(data, sequence_length=30):
        xs, ys = [], []
        for i in range(len(data) - sequence_length):
            xs.append(data[i:i+sequence_length])
            ys.append(data[i+sequence_length])
        return np.array(xs), np.array(ys)

    # Define our sequence length (you can tune this)
    sequence_length = 30  # for example, 30 days

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)

    # Split data into training and testing sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

    model.save(name+'_LTSM.keras')

if __name__ == "__main__":
    LSTM_training('BTC-USD')

