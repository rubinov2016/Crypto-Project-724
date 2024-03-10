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
def lag_features(data, lag=1):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def SVR_training(name, lag=5):
    future_steps = 100
    df = pd.read_csv('crypto_data_clean.csv')
    df = df[df['Symbol'] == name]
    df = df.transpose()
    df = df.iloc[1:]
    df.rename(columns={df.columns[0]: 'Price'}, inplace=True)
    # data = df['Price'].values.reshape(-1, 1)
    data = df['Price'].values

    # Create lagged features, using 10 past values to predict the next value
    X, y = lag_features(data, lag)
    # print("Row Labels (Index):", X.index.name or "Unnamed")  # Index might not have a name
    # print("Column Names:", list(X.columns))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(X_train)
    X_test_scaled = scaler.transform(X_test)
    svr_model = SVR(C=1.0, epsilon=0.1)
    svr_model.fit(X_train_scaled, y_train.ravel())

    dump(svr_model, 'svr_model.joblib')

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
    print(len(y))
    X_plot = data[lag:]
    print(len(X_plot))

    # plot the predicted values against the true values
    plt.scatter(X_plot, y, color='darkorange',
                label='data')
    # plt.plot(X_plot, y_pred, color='cornflowerblue',
    #           label='prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    lag = 5
    SVR_training('BTC-USD',lag)