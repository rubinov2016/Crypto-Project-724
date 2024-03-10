import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import pickle

def ARIMA_training(name):
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']
    # Choose the ARIMA Model parameters (p, d, q)
    # These should be chosen based on model diagnostics like ACF, PACF plots or grid search
    p = 2  # AR term
    d = 1  # Differencing order
    q = 1  # MA term

    # Fit the ARIMA model
    model = ARIMA(df['Price'], order=(p, d, q))
    model_fit = model.fit()

    # Now, you can save the model to disk
    with open(name+'_ARIMA.pkl', 'wb') as pkl:
        pickle.dump(model_fit, pkl)

    # Summary of the model
    print(model_fit.summary())

if __name__ == "__main__":
    ARIMA_training('BTC-USD')