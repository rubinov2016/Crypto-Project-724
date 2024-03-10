import pandas as pd
import numpy as np


# Assuming you have two pandas Series: `actual` for actual values and `predictions` for your model's predictions
# For demonstration, let's assume they are already defined. Replace these with your actual data.
# actual = pd.Series([...])
# predictions = pd.Series([...])

def calculate_mape(actual, predictions):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
    - actual: pd.Series, actual values
    - predictions: pd.Series, predicted values

    Returns:
    - mape: float, the Mean Absolute Percentage Error
    """
    # Ensure no division by zero; add a small number to the denominator
    errors = np.abs((actual - predictions) / (actual + 1e-10)) * 100
    mape = errors.mean()
    return mape


# Example usage
# Replace `actual` and `predictions` with your actual Series
# actual = pd.Series([...])
# predictions = pd.Series([...])
mape = calculate_mape(actual, predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
