import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from statistics import NormalDist


def plot_prices(data,tickers=None,show=True):
    """
    Plot adjusted closing prices for one or more tickers.

    Parameters
    ----------
    data : pandas.DataFrame
        Market data with market fields on the first column level and tickers on the second.

    tickers : str, list[str], or None
        Ticker or tickers to plot. If None, plot all available tickers.

    show : bool, optional
        Whether to display the plot immediately. Default is True.
  
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
  
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """

    prices = data["Adj Close"]
  
    # Plot all available tickers by default
    if tickers is None:
        tickers = prices.columns

    # Convert a single ticker into a list
    elif isinstance(tickers, str):
        tickers = [tickers]
  
    # Start figure
    fig, ax = plt.subplots(figsize=(12,6))

    # Plot prices
    for ticker in tickers:
        ax.plot(prices.index, prices[ticker], label=ticker)
  
    # Plot cosmetics
    ax.set_title("Adjusted Closing Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (US$)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
  
    # Display plot
    if show: 
      plt.show()

    return fig, ax

def plot_cumulative_returns(data,tickers=None,show=True):
    """
    Plot cumulative returns for one or more tickers.

    Parameters
    ----------
    data : pandas.DataFrame
        Market data containing calculated cumulative returns.
  
    tickers : str, list[str], or None
        Ticker or tickers to plot. If None, plot all available tickers.

    show : bool, optional
        Whether to display the plot immediately. Default is True.
  
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
  
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """

    cumulative_returns = data["Cumulative Return"]

    # Plot all available tickers by default
    if tickers is None:
        tickers = cumulative_returns.columns
  
    # Convert a single ticker into a list
    elif isinstance(tickers, str):
        tickers = [tickers]
  
    # Start figure
    fig, ax = plt.subplots(figsize=(12,6))

    # Plot cumulative returns
    for ticker in tickers:
        ax.plot(cumulative_returns.index, cumulative_returns[ticker], label=ticker)
   
    # Plot cosmetics
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
  
    # Display plot
    if show: 
        plt.show()

    return fig, ax

def plot_rolling_volatility(data, tickers=None,show=True):
    """
    Plot annualized rolling volatility for one or more tickers.

    Parameters
    ----------
    data : pandas.DataFrame
        Market data containing calculated rolling volatility.

    tickers : str, list[str], or None
        Ticker or tickers to plot. If None, plot all available tickers.

    show : bool, optional
        Whether to display the plot immediately. Default is True.
  
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
  
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """

    rolling_volatility = data["Rolling Volatility"]

    # Plot all available tickers by default
    if tickers is None:
        tickers = rolling_volatility.columns

    # Convert a single ticker into a list
    elif isinstance(tickers, str):
        tickers = [tickers]

    # Start figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot rolling volatility
    for ticker in tickers:
        ax.plot(rolling_volatility.index, rolling_volatility[ticker], label=ticker)

    # Plot cosmetics
    ax.set_title("Annualized Rolling Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
  
    # Display plot
    if show: 
        plt.show()

    return fig, ax

def plot_drawdowns(data, tickers=None,show=True):
    """
    Plot historical drawdowns for one or more tickers.

    Parameters
    ----------
    data : pandas.DataFrame
        Market data containing calculated drawdowns.

    tickers : str, list[str], or None
        Ticker or tickers to plot. If None, plot all available tickers.

    show : bool, optional
        Whether to display the plot immediately. Default is True.
  
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
  
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """

    drawdowns = data["Drawdown"]

    # Plot all available tickers by default
    if tickers is None:
        tickers = drawdowns.columns

    # Convert a single ticker into a list
    elif isinstance(tickers, str):
        tickers = [tickers]

    # Start figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot drawdowns
    for ticker in tickers:
        ax.plot(drawdowns.index, drawdowns[ticker], label=ticker)

    # Plot cosmetics
    ax.set_title("Historical Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
  
    # Display plot
    if show: 
        plt.show()

    return fig, ax

def plot_return_distributions(data, tickers=None, return_field="Log Return", bins=30, show=True):
    """
    Plot historical return distributions with fitted Gaussian benchmarks.

    Parameters
    ----------
    data : pandas.DataFrame
        Market data containing calculated returns.

    tickers : str, list[str], or None
        Tickers to plot. If None, plot all available tickers

    return_field : str
        Return variable to analyze. Default is "Log Return".
    
    bins : int
        Number of histogram bins. Default is 30

    show : bool
        Whether to display the figure. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    axes : numpy.ndarray
        The axes containing the histograms.
    """

    returns = data[return_field]

    # Select tickers
    if tickers is None:
        tickers = list(returns.columns)
    elif isinstance(tickers, str):
        tickers = [tickers]

    # Create one subplot per ticker
    fig, axes = plt.subplots(len(tickers), 1, figsize=(12,4*len(tickers)), squeeze=False)

    axes = axes[:,0]

    for ax, ticker in zip(axes, tickers):
        values = returns[ticker].dropna()

        # Calculate Gaussian parameters
        mu = values.mean()
        sigma = values.std(ddof=1)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Invalid volatility for {ticker}.")

        # Historical return distribution
        ax.hist(values, bins=bins, density=True, alpha=0.6,label="Historical Returns")

        # Gaussian benchmark with matching mean and variance
        x = np.linspace(min(values.min(), mu-4*sigma), max(values.max(), mu+4*sigma), 400)
        gaussian_pdf = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
        ax.plot(x,gaussian_pdf,label="Gaussian Benchmark")

        # Plot cosmetics
        ax.set_title(f"{ticker} Return Distribution")
        ax.set_xlabel(f"Daily {return_field}")
        ax.set_ylabel("Probability Density")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes

def plot_return_boxplots(data, tickers=None, return_field="Log Return", show=True):
    """
    Compare historical return distributions using box plots.

    Parameters
    ----------
    data : pandas.DataFrame
        Market data containing calculated returns.

    tickers : str, list[str], or None
        Tickers to plot. If None, plot all available tickers.

    return_field : str
        Return variable to analyze. Default is "Log Return".

    show : bool
        Whether to display the figure. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    ax : matplotlib.axes.Axes
        The axes containing the box plots.
    """

    returns = data[return_field]

    # Select tickers
    if tickers is None:
        tickers = list(returns.columns)
    elif isinstance(tickers, str):
        tickers = [tickers]

    # Collect valid returns for each ticker
    samples = [returns[ticker].dropna().to_numpy() for ticker in tickers]

    # Start figure
    fig, ax = plt.subplots(figsize=(12,6))

    # Plot return distributions
    ax.boxplot(samples, showmeans=True)

    # Plot cosmetics
    ax.set_xticks(range(1,len(tickers)+1))
    ax.set_xticklabels(tickers)
    ax.set_title("Historical Return Distributions")
    ax.set_xlabel("Ticker")
    ax.set_ylabel(f"Daily {return_field}")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    # Display plot
    if show:
        plt.show()

    return fig, ax

def plot_return_qq(data, tickers=None, return_field="Log Return", show=True):
    """
    Compare historical return quantiles with fitted Gaussian quantiles.

    Parameters
    ----------
    data : pandas.DataFrame
        Market data containing calculated returns.

    tickers : str, list[str], or None
        Tickers to plot. If None, plot all available tickers.
    
    return_field : str
        Return variable to analyze. Default is "Log Return".
    
    show : bool
        Whether to display the figure. Default is True.

    Returns
    -------    
    fig : matplotlib.figure.Figure
        The generated figure.

    ax : matplotlib.axes.Axes
        The axes containing the box plots.
    """

    returns = data[return_field]

    # Select tickers
    if tickers is None:
        tickers = list(returns.columns)
    elif isinstance(tickers, str):
        tickers = [tickers]

    # Create one subplot per ticker
    fig, axes = plt.subplots(len(tickers), 1, figsize=(8,5*len(tickers)), squeeze=False)

    axes = axes[:,0]

    for ax, ticker in zip(axes, tickers):

        # Sort historical returns
        values = np.sort(returns[ticker].dropna().to_numpy())

        n=len(values)

        # Estimate Gaussian parameters
        mu = values.mean()
        sigma = values.std(ddof=1)
        if n < 2 or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Invalid return distribution for {ticker}.")

        # Calculate quantile probabilities
        probabilities = (np.arange(1,n+1)-0.5)/n

        # Calculate theoretical Gaussian quantile
        gaussian = NormalDist(mu, sigma)
        theoretical = np.array([gaussian.inv_cdf(float(p)) for p in probabilities])

        # Plot observed versus theoretical quantiles
        ax.scatter(theoretical, values, s=15, alpha=0.7, label="Historical Returns")

        # Perfect agreement reference line
        limits = [min(theoretical.min(), values.min()), max(theoretical.max(),values.max())]
        ax.plot(limits, limits, "k--", label="Gaussian Reference")

        # Plot cosmetics
        ax.set_title(f"{ticker} Normal Q-Q Plot")
        ax.set_xlabel("Theoretical Gaussian Quantiles")
        ax.set_ylabel("Observed Return Quantiles")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()

    # Display plot
    if show:
        plt.show()

    return fig, axes