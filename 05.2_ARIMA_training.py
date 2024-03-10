import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from datetime import timedelta
import pickle
import json
from scipy import stats

def ARIMA_training(df, name):
    df.to_json('ARIMA_historical.json', orient='index')
    # Choose the ARIMA Model parameters (p, d, q)
    # These should be chosen based on model diagnostics like ACF, PACF plots or grid search
    p = 2  # AR term
    d = 1  # Differencing order
    q = 1  # MA term

    split_point = int(len(df) * 0.90)  # For example, 80% for training and 20% for testing
    train, test = df['Price'][:split_point], df['Price'][split_point:]

    # Fit the ARIMA model
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()

    # Now, you can save the model to disk
    with open(name+'_ARIMA.pkl', 'wb') as pkl:
        pickle.dump(model_fit, pkl)

    # Get predictions
    # Forecast
    # predictions = model_fit.forecast(steps=len(test))  # steps should match the size of your test set
    start_date = test.index[0]
    end_date = test.index[-1]

    predictions = model_fit.predict(start=start_date, end=end_date)
    test.name = 'Price'
    print(len(df))
    # actuals = df['Price']  # Adjust as necessary to align with your prediction range
    # Calculate metrics
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    errors = test

    # Printing each error, symbolizing a "circular" iteration
    print("Starting error circle:")
    for i, value in enumerate(test):
        errors[i] = np.abs((test[i] - predictions[i])/test[i])
        print(i, errors[i])
        # print(f"{test[i]}, {predictions[i]}, {np.abs(test[i]- predictions[i])/test[i]} Error {i}: {error:.2f}%")
    mape = errors.mean()
    print(99, mape)

    r_squared = r2_score(test, predictions)
    # Assuming `df` is your DataFrame with 'x' as the independent variable and 'y' as the dependent variable
    slope, intercept, r_value, p_value, std_err = stats.linregress(test,predictions)

    # The square of the correlation coefficient (r_value) is the R^2 score
    r2 = r_value ** 2
    print(r2)
    r_squared = r2

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r_squared": r_squared
    }
    with open('ARIMA_metrics.json', 'w') as file:
        json.dump(metrics, file)
    # Summary of the model
    # print(f'MSE: {mse}, RMSE: {rmse}', mape, r_squared)
    return model_fit.summary()

if __name__ == "__main__":
    name = 'BTC-USD'
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']

    future_steps = 30
    # predictions = LSTM_forecasting(df, 30, scaler_name, keras_name, dataset_name, future_steps)
    summary = ARIMA_training(df, name)
    # print(type(summary))
    # print(summary)


