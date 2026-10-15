import numpy as np
import pandas as pd
import yfinance as yf

from .validation import validate_market_data

TRADING_DAYS=252


def load_stock_data(tickers,start,end):
  """
  Download daily historical stock data from Yahoo Finance.

  Parameters
  ----------
  tickers : str or list[str]
    One stock ticker, for example "AAPL", or multiple tickers such as ["AAPL", "MSFT", "GOOG"].
  start : str
    Start date in YYYY-MM-DD format.
  end : str
    End date in YYYY-MM-DD format.

  Returns
  -------
  pandas.DataFrame
    Historical stock data.
  """

  data = yf.download(
    tickers,
    start=start,
    end=end,
    interval="1d",
    auto_adjust=False,
    group_by="column",
    progress=False,
  )

  if data.empty:
    raise ValueError(f"No data returned")
  
  # Make sure data is ordered by date
  data = validate_market_data(data)
  
  return data


def add_market_features(data, volatility_window=21):
  """
  Add useful return and volatility quantities to the stock data
  
  Parameters
  ----------
  data : pandas.DataFrame
    Multi-ticker market data with market fields on the first column level and tickes on the second.
  
  volatility_window : int
    Number of trading days used for rolling volatility.

  Returns
  -------
  pandas.DataFrame
    Original data with additional features
  """

  data = data.copy()

  # Adjusted closing prices for all tickers
  adjusted_close = data["Adj Close"]
  
  # Simple returns
  # R_t = P_t / (P_{t-1}) - 1
  returns = adjusted_close.pct_change(fill_method=None)

  # Gross returns
  # G_t = P_t / P_{t-1}
  gross_returns = returns + 1

  # Log returns
  log_returns = np.log(gross_returns)
  
  # Annualized rolling volatility
  rolling_volatility = (log_returns.rolling(volatility_window).std()*np.sqrt(TRADING_DAYS))

  # Growth of $1 invested
  cumulative_growth = adjusted_close/adjusted_close.iloc[0]

  # Cumulative return
  cumulative_return = cumulative_growth - 1
  
  # Highest adjusted closing price observed so far
  running_peak = adjusted_close.cummax()
  
  # Percentage decline from the previous peak
  drawdown = adjusted_close/running_peak - 1

  # Add the new features while keeping
  # first level = feature
  # second level = ticker
  features = pd.concat(
    {
      "Return": returns,
      "Gross Return": gross_returns,
      "Log Return": log_returns,
      "Rolling Volatility": rolling_volatility,
      "Cumulative Growth": cumulative_growth,
      "Cumulative Return": cumulative_return,
      "Running Peak": running_peak,
      "Drawdown": drawdown,
    },
    axis=1,
  )

  # Combine original market data and calculated features
  data = pd.concat([data, features], axis=1)

  return data.sort_index(axis=1)

if __name__=="__main__":
  start="2020-01-01"
  end="2026-10-15"
  tickers = "AAPL"
  df = load_stock_data(tickers,start,end)
  extDF = add_market_features(df)
