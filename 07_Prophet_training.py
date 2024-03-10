from prophet import Prophet
import pandas as pd
import pickle
import pickle

def Prophet_training(df, name):
    df = df.reset_index()
    print(df)
    # Rename the new column to 'ds'
    df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'}, inplace=True)
    # Initialize and fit the model
    # model = Prophet(daily_seasonality=True) # daily_seasonality might depend on the data frequency
    model = Prophet(changepoint_prior_scale=0.5)
    model.fit(df)

    with open(name+'_Prophet.pkl', 'wb') as pkl:
        pickle.dump(model, pkl)


def Prophet_forecasting(df, name, future_steps):
    df.to_json('Prophet_historical.json', orient='index')
    with open(name + '_Prophet.pkl', 'rb') as pkl:
         model = pickle.load(pkl)
    print(1)
    # Make a future dataframe for future predictions
    future = model.make_future_dataframe(periods=future_steps)
    # Convert 'ds' in 'future' DataFrame to date (without time)
    print(2)
    # Make predictions
    forecast = model.predict(future)
    print(3, df)
    print(type(forecast))
    future_forecast = forecast[forecast['ds'] > df.index.max()]
    print(4)
    # print(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] contains the time series forecast
    # 'yhat' is the forecasted value, 'yhat_lower' and 'yhat_upper' are the uncertainty intervals

    # Save forecast to a CSV file
    prediction = pd.DataFrame(future_forecast)
    prediction['ds'] = prediction['ds'].dt.strftime('%Y-%m-%d')
    prediction.set_index('ds', inplace=True)
    prediction.rename(columns={'yhat': 'Price'}, inplace=True)
    print(prediction)
    prediction = prediction[['Price']]

    prediction.to_json('Prophet_forecast.json', date_format='iso', orient='index')

    # Plot the forecast
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)

if __name__ == "__main__":
    name = 'BTC-USD'
    future_steps = 30
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']


    # Prophet_training(df, name)
    Prophet_forecasting(df, name, future_steps)