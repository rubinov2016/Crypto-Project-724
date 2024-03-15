import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from math import sqrt
import json
from joblib import load as joblib_load
from datetime import timedelta

def LSTM_training(df, name, sequence_length=30, epochs=10):
    df.to_csv(name+'_crypto_data.csv', header=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    # Split the data
    df_train, df_test = train_test_split(scaled_data, test_size=0.2, shuffle=False)
    # Create training and test sequences
    train_generator = TimeseriesGenerator(df_train, df_train,
                                          length=sequence_length, batch_size=1)
    test_generator = TimeseriesGenerator(df_test, df_test,
                                         length=sequence_length, batch_size=1)
    # Build the LSTM test model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_generator, epochs=epochs)

    # Predict on the test data
    y_pred = model.predict(test_generator)
    # Since the TimeseriesGenerator omits the first `sequence_length` samples, adjust accordingly
    y_test = df_test[sequence_length:]
    # Calculate MSE and RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    # Calculate MAPE - Avoid division by zero by adding a small number to the denominator
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred)
    print(987,r_squared)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r_squared": r_squared
    }
    with open('LSTM_metrics.json', 'w') as file:
        json.dump(metrics, file)

    production_generator = TimeseriesGenerator(df_train, df_train,
                                          length=sequence_length, batch_size=1)
    # Train the PRODUCTION
    model.fit(production_generator, epochs=epochs)
    model.save(name + '_LTSM.keras')
    
    return mse, rmse, mape, r_squared

def predict_next_time_step(last_sequence, model):
    # Reshape the last sequence to match the input shape of the model
    last_sequence = last_sequence.reshape((1, sequence_length, 1))
    # Predict the next step in the sequence
    predicted_value = model.predict(last_sequence)
    return predicted_value


# Function to forecast future values
def forecast_future(steps_to_predict, model, last_known_sequence, scaler):
    # Initialize the prediction sequence with the last known data
    prediction_sequence = last_known_sequence[-sequence_length:]

    # List to hold the predictions
    predictions = []

    # Predict 'steps_to_predict' steps into the future
    for _ in range(steps_to_predict):
        # Predict the next time step and retrieve the single value from the prediction
        next_step = predict_next_time_step(prediction_sequence, model)[0, 0]
        # Append the prediction to the predictions list
        predictions.append(next_step)
        # Add the predicted value to the sequence and remove the first value
        prediction_sequence = np.append(prediction_sequence[1:], [next_step])

    # Inverse transform the predictions to the original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions


def LSTM_forecasting(df, keras_name,future_steps):
    # Normalize the data
    df.to_json('LSTM_historical.json', orient='index')
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    model = load_model(keras_name)
    future_predictions = forecast_future(future_steps, model, scaled_data, scaler)
    last_date = pd.to_datetime(df.index[-1])
    # Create a date range for future predictions
    future_dates = [last_date + timedelta(days=x) for x in range(1, future_steps + 1)]

    future_predictions_with_dates =pd.DataFrame(future_predictions, index=future_dates, columns=['Price'])
    future_predictions_with_dates.to_json('LSTM_forecast.json', orient='index')

if __name__ == "__main__":
    # Load the dataset (assuming df is our DataFrame as shown in the picture)
    # df = pd.read_csv('our_data.csv')
    name = 'BTC-USD'
    # df = pd.read_csv('crypto_data_clean.csv')
    df = pd.read_csv('crypto_data_clean2.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']
    # !!
    df = df.dropna()
    # Define the sequence length (the time window)
    sequence_length = 30
    epochs = 10
    LSTM_training(df, name, sequence_length, epochs)
    future_steps = 30
    keras_name = name+'_LTSM.keras'
    LSTM_forecasting(df, keras_name, future_steps)