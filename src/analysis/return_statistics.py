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

def compute_return_covariance(data, return_field="Log Return"):
    """
    Calculate the sample covariance matrix of historical returns.

    Parameters
    ----------
    data : pandas.DataFrame
        Market data with financial variables on the first column level and tickers on the second.

    return_field : str
        Return variable to analyze. Default is "Log Return".

    Returns
    -------
    pandas.DataFrame
        Sample covariance matrix with tickers as rows and columns.
    """

    # Select the financial return variable
    returns = data[return_field]

    # Calculate sample covariance matrix
    covariance = returns.cov(ddof=1)

    return covariance

def compute_return_correlation(data, return_field="Log Return"):
    """
    Calculate the Pearson correlation matrix of historical returns.

    Parameters
    ----------
    data : pandas.DataFrame
        Market data with financial variables on the first column level and tickers on the second.

    return_field : str
        Return variable to analyze. Default is "Log Return".

    Returns
    -------
    pandas.DataFrame
        Pearson correlation matrix with tickers as rows and columns.
    """

    # Select the financial return variable
    returns = data[return_field]

    # Calculate Pearson correlation matrix
    correlation = returns.corr(method="pearson")

    return correlation