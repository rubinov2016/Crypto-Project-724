import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load as joblib_load

# Step 1: Load the saved Keras model
model = load_model('my_model.keras')

# Step 2: Load input data from a CSV file
# Replace 'input_data.csv' with the path to your CSV file
# input_df= pd.read_csv('crypto_data_AAVE-USD.csv')
input_df = pd.read_csv('crypto_data_AAVE-USD.csv', parse_dates=['Date'], index_col='Date')
# print(input_df)
# Step 3: Load the MinMaxScaler and normalize the input data
scaler = joblib_load('scaler.joblib')

# Assuming your data needs to be reshaped or processed to fit the model input shape
# Reshape or process as necessary
# For example, if your model expects data shaped (samples, features), ensure input_df matches this
# Here, we assume input_df.values is already in the correct shape

normalized_input = scaler.transform(input_df.values)

# Step 4: Run predictions with the normalized data
predictions_normalized = model.predict(normalized_input)

# Step 5: Inverse the normalization of the predictions
predictions = scaler.inverse_transform(predictions_normalized)

import numpy as np

# Assuming `model` is your trained LSTM model
# `scaler` is a MinMaxScaler instance used for normalization
# `last_sequence` is your most recent data sequence, appropriately shaped and scaled

last_sequence = normalized_input
# Prepare the initial input sequence
input_seq = last_sequence[-1].reshape((1, last_sequence.shape[1], 1))  # Adjust the shape as per your model's input

# Container for predictions
predictions = []

for _ in range(300):  # Predict for the next 300 days
    # Make a prediction
    pred = model.predict(input_seq)

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
