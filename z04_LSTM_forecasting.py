import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from joblib import load as joblib_load
import numpy as np
from datetime import timedelta

sequence_length = 30  # Example: 30 days
# Function to predict a single future step
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
    # Load the dataset (assuming df is our DataFrame as shown in the picture)
    # df = pd.read_csv('our_data.csv')
     # print(df)
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
    name = 'BTC-USD'
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']
    # df.set_index('Date', inplace=True)  # Ensure Date is the index
    scaler_name = name+'_scaler.joblib'
    keras_name = name+'_LTSM.keras'
    dataset_name = name + '_crypto_data.csv'
    future_steps = 30
    LSTM_forecasting(df,30, keras_name, future_steps)

