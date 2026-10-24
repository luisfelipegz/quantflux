import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


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