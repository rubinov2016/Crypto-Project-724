import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from joblib import load
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

def SVR_forecasting(model, recent_data, n_steps, lag =5):
    scaler = StandardScaler()
    future_forecast = []
    print(type(recent_data))
    print(recent_data)
    input_features = recent_data[-lag:].reshape(1, -1)  # Reshape for a single sample
    # Fit the scaler to your data
    print()
    scaler.fit(input_features)
    print(input_features)
    for i in range(n_steps):
        # Scale input features
        input_features_scaled = scaler.transform(input_features)
        print(i, input_features_scaled)
        # Forecast the next step
        print(2)
        next_step = model.predict(input_features_scaled)
        print(input_features_scaled)
        # Append the forecasted value
        future_forecast.append(next_step.item())
        # Update the input features to include the new prediction
        input_features = np.roll(input_features, -1)
        input_features[0, -1] = next_step
    # forecasted_unscaled = scaler.inverse_transform(future_forecast)
    return future_forecast

if __name__ == "__main__":
    svr_model = load('svr_model.joblib')
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == ('BTC-USD')]
    df = df.transpose()
    df = df.iloc[1:]
    df.rename(columns={df.columns[0]: 'Price'}, inplace=True)
    # data = df['Price'].values.reshape(-1, 1)
    data = df['Price'].values
    print(data)
    future_steps = 100  # Number of days to forecast
    # # Ensure `recent_data` contains at least `lag` recent observations (not scaled)
    # data = data[-lag:]
    lag = 5
    forecasted_values = SVR_forecasting(svr_model, data, future_steps, lag)

    last_date = pd.to_datetime(df.index[-1])

    # Create a date range for future predictions
    future_dates = [last_date + timedelta(days=x) for x in range(1, future_steps + 1)]
    print(future_dates)
    # Combine future dates with predictions
    future_predictions_with_dates = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': forecasted_values  # Flatten the predictions array if necessary
    })
    print(future_predictions_with_dates)
    # Convert forecasted values to Series for easier handling
    # forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_steps)
    # forecast_series = pd.Series(forecasted_values, index=forecast_dates)

    # print(forecast_series)
