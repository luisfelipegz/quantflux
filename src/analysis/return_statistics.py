import pandas as pd

def summarize_returns(data,return_field="Log Return"):
    """
    Calculate descriptive statistics of historical financial returns

    Parameters
    ----------
    data : pandas.DataFrame
        Market data with financial variables on the first column level and tickers on the second.
    
    return_field : str
        Return variable to analyze. Default is "Log Return" 

    Returns
    -------
    pandas.DataFrame
        Descriptive statistics for each ticker.
    """

    # Select the financial return variable
    returns = data[return_field]

    # Calculate descriptive statistics for each ticker
    summary = pd.DataFrame({
        "Count": returns.count(),           # Number of valid daily return observations
        "Mean": returns.mean(),             # Average daily return
        "Std Dev": returns.std(ddof=1),     # Sample standard deviation of daily returns
        "Skewness": returns.skew(),         # Asymmetry of the return distribution
        "Excess Kurtosis": returns.kurt()   # Tail behavior and peakedness relative to a normal distribution
    })

    return summary