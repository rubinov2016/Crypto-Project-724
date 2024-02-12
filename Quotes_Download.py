import yfinance as yf
import pandas as pd

# List of cryptocurrency ticker symbols
ticker_symbols = [
    'BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'USDC-USD', 'STETH-USD', 'ADA-USD', 'AVAX-USD',
    'LINK-USD', 'DOGE-USD', 'WTRX-USD', 'TRX-USD', 'DOT-USD', 'MATIC-USD', 'WBTC-USD', 'TON11419-USD', 'ICP-USD',
    'SHIB-USD',    'BCH-USD', 'DAI-USD', 'LTC-USD', 'UNI7083-USD', 'LEO-USD', 'ATOM-USD', 'ETC-USD', 'IMX10603-USD', 'OP-USD',
    'TAO22974-USD',    'NEAR-USD', 'TIA22861-USD', 'KAS-USD', 'XLM-USD', 'INJ-USD', 'APT21794-USD', 'OKB-USD', 'FDUSD-USD', 'FIL-USD',
    'STX4847-USD',    'WHBAR-USD', 'HBAR-USD', 'BTCB-USD', 'LDO-USD', 'ARB11841-USD', 'WEOS-USD', 'XMR-USD', 'VET-USD', 'CRO-USD',
    'MNT27075-USD',    'WBETH-USD', 'SUI20947-USD', 'MKR-USD', 'RUNE-USD', 'RNDR-USD', 'HEX-USD', 'SEI-USD', 'BSV-USD', 'GRT6719-USD',
    'RETH-USD',    'EGLD-USD', 'TUSD-USD', 'MINA-USD', 'ALGO-USD', 'ORDI-USD', 'HNT-USD', 'AAVE-USD', 'BEAM28298-USD', 'QNT-USD',
    'FLOW-USD',    'FTM-USD', 'FLR-USD', 'DYM-USD', 'SNX-USD', 'SAND-USD', 'ASTR-USD', 'AXS-USD', 'THETA-USD', 'KCS-USD', 'XTZ-USD',
    'BTT-USD',    'BGB-USD', '1000SATS-USD', 'CHEEL-USD', 'MANA-USD', 'CHZ-USD', 'ETHDYDX-USD', 'PYTH-USD', 'NEO-USD', 'CFX-USD',
    'EOS-USD',    'BONK-USD', 'OSMO-USD', 'WEMIX-USD', 'BLUR-USD', 'ROSE-USD', 'RON14101-USD', 'IOTA-USD', 'KAVA-USD', 'KLAY-USD'
]

# Create an empty dataframe to store the data
crypto_data = pd.DataFrame()

# Iterate over each ticker symbol
for symbol in ticker_symbols:
    # Retrieve data for the ticker symbol
    ticker = yf.Ticker(symbol)
    ticker_data = ticker.history(period="1y")  # You can adjust the period as needed

    # Add the ticker symbol and name to the dataframe
    ticker_data['Symbol'] = symbol
    ticker_data['Name'] = ticker.info['longName']

    # Append the data to the combined dataframe
    crypto_data = pd.concat([crypto_data, ticker_data])
    # Get unique names for each symbol
    symbol_to_name = {symbol: yf.Ticker(symbol).info['longName'] for symbol in ticker_symbols}
    crypto_data['Name'] = crypto_data['Symbol'].map(symbol_to_name)

# Save the data to a CSV file
crypto_data.to_csv('crypto_data_with_names.csv', index=False)

print("Data saved to CSV successfully!")
