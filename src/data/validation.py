import numpy as np
import pandas as pd
def validate_market_data(data):
    """
    Validate historical market data before financial analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        Historical market data with financial fields on the first column level and tickers on the second.

    Returns
    -------
    pandas.DataFrame
        Validated market data sorted by date
    """

    # Check that the data contains observations
    if data.empty:
        raise ValueError("The market dataset is empty.")

    # Verify that the index contains dates
    if not isinstance(data.index,pd.DatetimeIndex):
        raise TypeError("The index must contain dates.")

    # Each date should appear only once
    if data.index.has_duplicates:
        raise ValueError("Duplicate trading dates detected.")

    # Verify our field-field, ticker-second structure
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns.")
    if data.columns.nlevels != 2:
        raise ValueError("Expected two column levels.")

    # Check that adjusted closing prices are available
    if "Adj Close" not in data.columns.get_level_values(0):
        raise ValueError("Adjusted closing prices are missing")
    prices=data["Adj Close"]

    # Missing values cannot be used reliably in calculations
    if prices.isna().any().any():
        raise ValueError("Missing adjusted closing prices.")

    # Require finite and strictly positive prices
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Non-finite adjusted closing prices.")
    if not (prices > 0).all().all():
        raise ValueError("Non-positive adjusted closing prices.")

    # Return observations in chronological order
    return data.sort_index()