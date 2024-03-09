import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('crypto_data_clean.csv')
tickers = ['MKR-USD','ETH-USD','BTC-USD','USDT-USD']
# Filter the DataFrame
df_filtered = df[df['Symbol'].isin(tickers)]
# Pivot the DataFrame
df_pivoted = df_filtered.melt(id_vars=['Symbol'], var_name='Date', value_name='Price')
df_pivoted['Date'] = pd.to_datetime(df_pivoted['Date'])
df_pivoted = df_pivoted.pivot(index='Date', columns='Symbol', values='Price')

# Check the pivoted DataFrame
print(df_pivoted.head())

# Fill missing values
df_pivoted.fillna(method='ffill', inplace=True)
df_pivoted.fillna(method='bfill', inplace=True)

# Normalize the data (example: Min-Max Scaling)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# df_pivoted_scaled = pd.DataFrame(scaler.fit_transform(df_pivoted), columns=df_pivoted.columns, index=df_pivoted.index)
df_pivoted_scaled = df_pivoted
# Check the scaled DataFrame
print(df_pivoted_scaled.head())

# Define a split point
split_point = int(len(df_pivoted_scaled) * 0.8)

# Split the data
train = df_pivoted_scaled.iloc[:split_point]
test = df_pivoted_scaled.iloc[split_point:]

print(f"Training shape: {train.shape}, Testing shape: {test.shape}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Predicting 'ETH-USD' based on others
X = train.drop('ETH-USD', axis=1)
y = train['ETH-USD']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Validate the model
score = model.score(X_val, y_val)
print(f"Model accuracy: {score:.2%}")
