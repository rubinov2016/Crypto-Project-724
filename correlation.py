# correlation_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_top_correlated_stocks(stock_symbol, df, n=5):
    if df.index.name is None or df.index.name != df.columns[0]:
        df = df.set_index(df.columns[0])
    print(df)
    df = df.transpose()
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    # print(correlation_matrix)
    # Get the correlations for the specified stock
    # Ensure that 'stock_symbol' is actually in the correlation matrix
    if stock_symbol in correlation_matrix.columns:
        # correlations = correlation_matrix[stock_symbol]
        correlations = correlation_matrix.loc[stock_symbol]
        # Drop the self-correlation
        correlations = correlations.drop(stock_symbol, errors='ignore')  # 'errors=ignore' for safety

    else:
        raise ValueError(f"Stock symbol '{stock_symbol}' not found in DataFrame.")
    # Find the top-n positively correlated stocks
    top_positive = correlations.nlargest(n)
    # Find the top-n negatively correlated stocks
    top_negative = correlations.nsmallest(n)
    return top_positive, top_negative


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('crypto_data_clean.csv')
    stock_symbol = 'BTC-USD'
    top_positive, top_negative = get_top_correlated_stocks(stock_symbol, df)

    top_positive_df = top_positive.to_frame(name='Correlation')
    top_negative_df = top_negative.to_frame(name='Correlation')

    # Setting up the matplotlib figure for two subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap for top positive correlations
    sns.heatmap(top_positive_df, annot=True, cmap='Greens', cbar=True, fmt=".2f", ax=ax[0])
    ax[0].set_title('Top Positive Correlations')
    ax[0].set_xlabel('Correlation')
    ax[0].set_ylabel('Stock Symbols')

    # Heatmap for top negative correlations
    sns.heatmap(top_negative_df, annot=True, cmap='Reds', cbar=True, fmt=".2f", ax=ax[1])
    ax[1].set_title('Top Negative Correlations')
    ax[1].set_xlabel('Correlation')
    ax[1].set_ylabel('Stock Symbols')

    plt.tight_layout()
    plt.show()

    print("Top-10 Positively Correlated Stocks:")
    print(top_positive)
    print("\nTop-10 Negatively Correlated Stocks:")
    print(top_negative)
