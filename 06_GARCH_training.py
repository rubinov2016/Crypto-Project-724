import numpy as np
import pandas as pd
from arch import arch_model
from itertools import product
import pickle
from datetime import timedelta

def GARCH_training(name):

    future_steps = 30
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name].drop(['Symbol'], axis=1).T
    df.index.name = 'Date'
    df.columns = ['Price']

    df['Returns'] = df['Price'].pct_change().dropna()  # Using percentage change
    # df['Returns'] = np.log(df['Price']).diff().dropna()
    df = df.iloc[1:]
    print(df['Returns'])

    # Specify the range of p and q values to try
    p_values = range(1, 4)
    q_values = range(1, 4)

    # Initialize the best model with high AIC
    best_aic = np.inf
    best_p = None
    best_q = None
    best_model = None

    # Grid search for the best GARCH(p,q) model
    for p, q in product(p_values, q_values):
        try:
            model = arch_model(df['Returns'], vol='Garch', p=p, q=q)
            results = model.fit(disp='off')  # 'off' turns off the convergence messages
            print(results)
            if results.aic < best_aic:
                best_aic = results.aic
                best_p = p
                best_q = q
                best_model = results
        except Exception as e:
            print(f'Failed to fit GARCH({p},{q}): {e}')
    # Output the best model's results
    if best_model:
        print(f'Best GARCH Model: GARCH({best_p},{best_q}) with AIC: {best_aic}')
        print(best_model.summary())
    else:
        print('No suitable GARCH model found')

    # Save the best model
    if best_model:
        with open('best_garch_model.pkl', 'wb') as pkl:
            pickle.dump(best_model, pkl)

    # garch = arch_model(df['Returns'], vol='Garch', p=1, q=1)

    # Fit the model
    # res = garch.fit(update_freq=10)  # update_freq=10 for less verbose output during fitting

    print(best_model.summary())

    forecasts = best_model.forecast(horizon=future_steps)

    # Print the forecast variance
    print(forecasts.variance[-1:])
    results_as_series = pd.Series(forecasts)


    last_date = pd.to_datetime(df.index[-1])
    # Create a date range for future predictions
    future_dates = [last_date + timedelta(days=x) for x in range(1, future_steps + 1)]

    # Combine future dates with predictions
    future_predictions_with_dates = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': results_as_series  # Flatten the predictions array if necessary
    })
    print(future_predictions_with_dates)

    print(results_as_series)



    # # For out-of-sample forecasting
    # forecasts = best_model.forecast(horizon=5, start=None, align='origin')
    # print(forecasts.mean.iloc[-1])

if __name__ == "__main__":
    GARCH_training('BTC-USD')