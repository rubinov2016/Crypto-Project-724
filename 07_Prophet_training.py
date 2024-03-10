from prophet import Prophet
import pandas as pd

def Prophet_training(name):
    future_steps = 100
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name]
    df = df.transpose()
    # Reset the index and add it as a column
    df = df.reset_index()

    # Rename the new column to 'ds'
    df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'}, inplace=True)
    df = df.iloc[1:]


    # Initialize and fit the model
    # model = Prophet(daily_seasonality=True) # daily_seasonality might depend on the data frequency
    model = Prophet(changepoint_prior_scale=0.5)
    model.fit(df)

    # Make a future dataframe for future predictions
    future = model.make_future_dataframe(periods=future_steps)

    # Make predictions
    forecast = model.predict(future)

    future_forecast = forecast[forecast['ds'] > df['ds'].max()]
    print(df)
    print(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] contains the time series forecast
    # 'yhat' is the forecasted value, 'yhat_lower' and 'yhat_upper' are the uncertainty intervals

    # Save forecast to a CSV file
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('prophet_forecast.csv', index=False)

    # Plot the forecast
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)

if __name__ == "__main__":
    Prophet_training('BTC-USD')