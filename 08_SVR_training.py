import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
from joblib import dump
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta

def lag_features(data, lag=1):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def SVR_training(df, name, lag=5):
    df.to_json('SVR_historical.json', orient='index')
    data = df['Price'].values
    # Create lagged features, using 10 past values to predict the next value
    X, y = lag_features(data, lag)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(X_train)
    X_test_scaled = scaler.transform(X_test)
    svr_model = SVR(C=1.0, epsilon=0.1)
    svr_model.fit(X_train_scaled, y_train.ravel())

    with open(name+'_SVR.pkl', 'wb') as pkl:
        pickle.dump(svr_model, pkl)
    y_pred = svr_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Test MSE:", mse)
    print("Test RMSE:", rmse)
    print("Test MAE:", mae)
    print("Test r2:", r2)
    y_pred_unscaled = y_pred.reshape(-1, 1)
    # y_pred_unscaled= scaler.inverse_transform(y_pred_unscaled)
    # print(y_pred_unscaled)
    # print(len(y))
    X_plot = data[lag:]
    # print(len(X_plot))

    # plot the predicted values against the true values
    plt.scatter(X_plot, y, color='darkorange',
                label='data')
    # plt.plot(X_plot, y_pred, color='cornflowerblue',
    #           label='prediction')
    plt.legend()
    plt.show()


def SVR_forecasting(df, name,  n_steps, lag =5):
    data = df['Price'].values
    # print(data)
    with open(name + '_SVR.pkl', 'rb') as pkl:
         model = pickle.load(pkl)
    scaler = StandardScaler()
    future_forecast = []

    input_features = data[-lag:].reshape(1, -1)  # Reshape for a single sample
    # input_features = data[-lag:]  # Reshape for a single sample
    # Fit the scaler to your data
    scaler.fit(input_features)
    for i in range(n_steps):
        # Scale input features
        input_features_scaled = scaler.transform(input_features)
        # Forecast the next step
        next_step = model.predict(input_features_scaled)
        # Append the forecasted value
        future_forecast.append(next_step.item())
        # Update the input features to include the new prediction
        input_features = np.roll(input_features, -1)
        input_features[0, -1] = next_step
    # forecasted_unscaled = scaler.inverse_transform(future_forecast)
    # predictions = pd.DataFrame(future_forecast, columns=['Price'])
    # print('p', predictions)
    last_date = pd.to_datetime(df.index[-1])
    # Create a date range for future predictions
    future_dates = [last_date + timedelta(days=x) for x in range(1, future_steps + 1)]
    dates_index = pd.to_datetime(future_dates)
    # future_predictions_with_dates  = pd.DataFrame(predictions, index=dates_index, columns=['Price'])
    # print(type(future_dates))
    future_predictions_with_dates = pd.DataFrame(future_forecast, index=future_dates, columns=['Price'])
    print(future_predictions_with_dates)

    future_predictions_with_dates.index = future_predictions_with_dates.index.strftime('%Y-%m-%d')
    print(future_predictions_with_dates)
    future_predictions_with_dates.to_json('SVR_forecasting.json', date_format='iso', orient='index')

if __name__ == "__main__":
    lag = 5
    future_steps = 30
    name = 'BTC-USD'
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']
    # print(df)
    # SVR_training(df, 'BTC-USD',lag)
    SVR_forecasting(df, 'BTC-USD', n_steps=future_steps, lag=lag)