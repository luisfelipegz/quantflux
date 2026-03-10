import yfinance as yf
import numpy as np

"""

  On a regular year there are 252 trading days.

  volatility:
    Compute the volatility of a ticker starting in 2020
    Volatility of a stock is defined as the yearly standard deviation of the log-returns of the stock.

"""

def volatility(ticker: str)->float:
  '''
  Compute volatility of ticker starting in 2020 (post-COVID)
  Arguments:
    ticker    : Stock ticker
  Returns:
    volatility: Yearly standard deviation of log-returns 
  '''
  # Download stock data after COVID
  stockData = yf.download(ticker, start='2020-01-01')
  # Compute returns
  stockData['Return',ticker] = stockData['Close',ticker].pct_change()+1
  stockData['LogReturn',ticker] = np.log(stockData['Return',ticker])
  # Clean data
  stockData = stockData.dropna()
  # Volatility
  #  There are 252 trading days per year
  volatility = np.sqrt(252)*np.std(stockData['LogReturn',ticker])
  # End
  print(f'{ticker} volatility is {volatility:.3f}')
  return volatility
