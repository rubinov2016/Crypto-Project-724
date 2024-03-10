import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def LSTM_training(name):
    # Load the dataset (assuming df is our DataFrame as shown in the picture)
    # df = pd.read_csv('our_data.csv')
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']
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


    # If you want to forecast more steps into the future, you would repeat the process,
    # each time adding the last prediction to the end of our data and forming a new sequence.

if __name__ == "__main__":
    LSTM_training('BTC-USD')