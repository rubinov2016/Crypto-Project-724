import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from math import sqrt
import json

def LSTM_training(df, name):
    # df.set_index('Date', inplace=True)  # Ensure Date is the index
    df.to_csv(name+'_crypto_data.csv', header=True)

    # print(df)
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Split the data into training and test sets
    train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

    # Define the sequence length (the time window)
    sequence_length = 30  # Example: 30 days

    # Use TimeseriesGenerator to create training and test sequences
    train_generator = TimeseriesGenerator(train_data, train_data,
                                          length=sequence_length, batch_size=1)
    test_generator = TimeseriesGenerator(test_data, test_data,
                                         length=sequence_length, batch_size=1)

    # Build the LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(train_generator, epochs=10)
    model.save(name + '_LTSM.keras')

    # Predict on the test data
    test_predictions = model.predict(test_generator)
    # Since the TimeseriesGenerator omits the first `sequence_length` samples, adjust accordingly
    adjusted_test_data = test_data[sequence_length:]

    # Calculate MSE and RMSE
    mse = mean_squared_error(adjusted_test_data, test_predictions)
    rmse = sqrt(mse)
    # Calculate MAPE - Avoid division by zero by adding a small number to the denominator
    mape = np.mean(np.abs((adjusted_test_data - test_predictions) / (adjusted_test_data + 1e-10))) * 100

    # Calculate R-squared
    r_squared = r2_score(adjusted_test_data, test_predictions)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r_squared": r_squared
    }
    with open('LSTM_metrics.json', 'w') as file:
        json.dump(metrics, file)

    print(f'MSE: {mse}, RMSE: {rmse}', mape, r_squared )
    return mse, rmse, mape, r_squared


    # If you want to forecast more steps into the future, you would repeat the process,
    # each time adding the last prediction to the end of our data and forming a new sequence.

if __name__ == "__main__":
    # Load the dataset (assuming df is our DataFrame as shown in the picture)
    # df = pd.read_csv('our_data.csv')
    name = 'BTC-USD'
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']
    LSTM_training(df, name)
    # LSTM_metrics = pd.read_json('LSTM_metrics.json', typ='series')
    # print(LSTM_metrics)