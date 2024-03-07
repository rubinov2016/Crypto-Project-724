# Predict the next day (replace X_test[-1:] with the most recent sequence_length data points when forecasting future values)
predicted = model.predict(X_test[-1:].reshape((1, sequence_length, 1)))
predicted = scaler.inverse_transform(predicted)  # Rescale back to the original range

print(predicted)