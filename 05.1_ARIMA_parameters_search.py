import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings

def ARIMA_parameter_search(name):
    # Load the dataset
    df = pd.read_csv('crypto_data_clean2.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    # df['Date'] = pd.to_datetime(df['Date'])
    # df.set_index('Date', inplace=True)
    df.index.name = 'Date'
    df.columns = ['Price']
    print(type(df))
    # our data column should be a univariate series, example: df['Price']
    series = df['Price']

    # Define the p, d, and q parameters to take any value between 0 and 2
    p = d = q = range(0, 3)

    # Generate all different combinations of p, d, and q triplets
    pdq = list(itertools.product(p, d, q))

    # Run a grid search to find the best ARIMA model
    best_aic = np.inf
    best_pdq = None
    best_model = None

    warnings.filterwarnings("ignore")  # Ignore warnings from ARIMA fits

    for param in pdq:
        try:
            model = ARIMA(series, order=param)
            results = model.fit()
            # print(f"ARIMA{param} - AIC:{results.aic}")
            print(f"ARIMA{param} - AIC:{results.aic:.2f}")
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_model = results
        except Exception as e:
            continue

    print(f"Best ARIMA model:")
    print(f"ARIMA{best_pdq} with AIC: {best_aic:.2f}")

    # Save the best ARIMA model to disk
    best_model.save('best_arima_model.pkl')

if __name__ == "__main__":
    ARIMA_parameter_search('BTC-USD')