import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from joblib import load as joblib_load
import numpy as np

def LSTM_forecasting(name, days, scaler_name, keras_name,dataset_name):
    input_df = pd.read_csv(dataset_name, parse_dates=['Date'], index_col='Date')
    # print(input_df)
    # Normalize  data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(input_df)

    model = load_model(keras_name)
    scaler = joblib_load(scaler_name)
    normalized_input = scaler.transform(input_df.values)
    print(1)
    # Step 4: Run predictions with the normalized data
    predictions_normalized = model.predict(normalized_input)
    print(2)
    # Step 5: Inverse the normalization of the predictions
    predictions = scaler.inverse_transform(predictions_normalized)
 # `last_sequence` is our most recent data sequence, appropriately shaped and scaled
    print(3)
    last_sequence = normalized_input
    # Prepare the initial input sequence
    sequence_length = 367
    # input_seq = last_sequence[-1].reshape((1, last_sequence.shape[1], 1))  # Adjust the shape as per our model's input
    input_seq = last_sequence.reshape((1, sequence_length, 1))
    print(4)
    # Container for predictions
    predictions = []
    print(5)
    for _ in range(days):  # Predict for the next 300 days
        # Make a prediction
        pred = model.predict(input_seq)
        print(6)

        # Append the prediction (note: pred is scaled)
        predictions.append(pred.flatten()[0])

        # Update the input sequence with the prediction (rolling)
        input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)

    # Inverse transform the predictions
    # Ensure predictions is a numpy array and properly shaped for the inverse transformation
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_original_scale = scaler.inverse_transform(predictions).flatten()

    # Print or use the predictions
    print(predictions_original_scale)

if __name__ == "__main__":
    name = 'BTC-USD'
    scaler_name = name+'_scaler.joblib'
    keras_name = name+'_LTSM.keras'
    dataset_name = name + '_crypto_data.csv'
    LSTM_forecasting(name,30, scaler_name, keras_name,dataset_name)