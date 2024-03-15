import yfinance as yf
import pandas as pd
import warnings

# List of cryptocurrency ticker symbols
ticker_symbols = [
    'BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'USDC-USD', 'STETH-USD', 'ADA-USD', 'AVAX-USD',
    'LINK-USD', 'DOGE-USD', 'WTRX-USD', 'TRX-USD', 'DOT-USD', 'MATIC-USD', 'WBTC-USD', 'TON11419-USD', 'ICP-USD',
    'SHIB-USD', 'BCH-USD', 'DAI-USD', 'LTC-USD', 'UNI7083-USD', 'LEO-USD', 'ATOM-USD', 'ETC-USD', 'IMX10603-USD', 'OP-USD',
    'TAO22974-USD','NEAR-USD', 'TIA22861-USD', 'KAS-USD', 'XLM-USD', 'INJ-USD', 'APT21794-USD', 'OKB-USD', 'FDUSD-USD', 'FIL-USD',
    'STX4847-USD','WHBAR-USD', 'HBAR-USD', 'BTCB-USD', 'LDO-USD', 'ARB11841-USD', 'WEOS-USD', 'XMR-USD', 'VET-USD', 'CRO-USD',
    'MNT27075-USD','WBETH-USD', 'SUI20947-USD', 'MKR-USD', 'RUNE-USD', 'RNDR-USD', 'HEX-USD', 'SEI-USD', 'BSV-USD', 'GRT6719-USD',
    'RETH-USD','EGLD-USD', 'TUSD-USD', 'MINA-USD', 'ALGO-USD', 'ORDI-USD', 'HNT-USD', 'AAVE-USD', 'BEAM28298-USD', 'QNT-USD',
    'FLOW-USD','FTM-USD', 'FLR-USD', 'DYM-USD', 'SNX-USD', 'SAND-USD', 'ASTR-USD', 'AXS-USD', 'THETA-USD', 'KCS-USD', 'XTZ-USD',
    'BTT-USD','BGB-USD', '1000SATS-USD', 'CHEEL-USD', 'MANA-USD', 'CHZ-USD', 'ETHDYDX-USD', 'PYTH-USD', 'NEO-USD', 'CFX-USD',
    'EOS-USD','BONK-USD', 'OSMO-USD', 'WEMIX-USD', 'BLUR-USD', 'ROSE-USD', 'RON14101-USD', 'IOTA-USD', 'KAVA-USD', 'KLAY-USD'
]

# Dictionary to store ticker symbols and their names
ticker_names = {}

# Iterate over each ticker symbol
for symbol in ticker_symbols:
    # Retrieve ticker information
    ticker_info = yf.Ticker(symbol).info
    # Get the name if available
    name = ticker_info.get('longName', 'N/A')
    # Store the ticker symbol and its name in the dictionary
    ticker_names[symbol] = name

# Create an empty list to store dataframes
crypto_dataframes = []

# Create an empty list to store unavailable tickers
unavailable_tickers = []

# Iterate over each ticker symbol
for symbol in ticker_names:
    ticker = yf.Ticker(symbol)
    ticker_data = ticker.history(period="max")
    if ticker_data.empty:
        unavailable_tickers.append(symbol)
        warnings.warn(f"Data not available for ticker symbol: {symbol}")
    else:
        # Add the ticker symbol to the dataframe
        ticker_data['Symbol'] = symbol
        # Append the dataframe to the list
        crypto_dataframes.append(ticker_data)

# Concatenate all dataframes into one
crypto_data = pd.concat(crypto_dataframes)
crypto_data.index = pd.to_datetime(crypto_data.index)
crypto_data.index = crypto_data.index.date
crypto_data.index.name = 'Date'
# crypto_data['Date'] = pd.to_datetime(crypto_data['Date']).dt.date

column_to_drop = 'Dividends'
if column_to_drop in crypto_data.columns:
    # Drop the column
    crypto_data.drop(column_to_drop, axis=1, inplace=True)
column_to_drop = 'Stock Splits'
if column_to_drop in crypto_data.columns:
    # Drop the column
    crypto_data.drop(column_to_drop, axis=1, inplace=True)

# Save the data to a CSV file
crypto_data.to_csv('crypto_data.csv')
print("Data saved to CSV successfully!")

if unavailable_tickers:
    print("Unavailable tickers:", unavailable_tickers)
