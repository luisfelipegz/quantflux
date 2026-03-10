import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

"""
  constructPortfolio:
    Manual construction of portfolio for general purposes.
"""

def constructPortfolio(tickers: list[str], weights: np.ndarray, plot: bool)->np.ndarray:
  '''
  Manual construction of portfolio
  Arguments:
    tickers : Stock tickers in portfolio (i.e. KLAC, APPL)
    weights : Percentage of portfolio associated to ticker
    plot    : Plotting option
  Returns:
    cumsum  : Cumulative sum of returns of portfolio
  
  Example:
    tickers = ['KLAC','MU','ASML','NEM','TSM','LRCX','ADI','AMAT']
    equity = [1078.61,172.50,161.43,111.87,119.60,123.74,109.11,149.77]
    increase = [0,300,300,500,300,300,300,300]
    weights = (np.array(equity)+np.array(increase))/(sum(equity)+sum(increase))
    constructPortfolio(tickers,weights,True)
  '''
  # Download stock data after COVID
  stockData = yf.download(tickers, start='2020-01-01')
  stockData = stockData.dropna()
  # Add stocks returns
  for ticker in tickers:
    # Daily returns
    stockData['Return',ticker] = stockData['Close',ticker].pct_change()+1
    stockData['LogReturn',ticker] = np.log(stockData['Return',ticker])
    stockData = stockData.dropna()
    # Cumulative returns
    stockData['CumReturn',ticker] = stockData['Close',ticker]/stockData['Close',ticker].iloc[0]
    # Volatility
    volatility = np.sqrt(252)*np.std(stockData['LogReturn',ticker])
    print(f'{ticker} volatility is {volatility:.3f}')
  # Portfolio return
  portfolioCumReturn = sum(weights[i]*stockData['CumReturn',tickers[i]] for i in range(len(tickers)))
  portfolioLogReturn = sum(weights[i]*stockData['LogReturn',tickers[i]] for i in range(len(tickers)))
  # Portfolio volatility
  portfolioVolatility = np.sqrt(252)*np.std(portfolioLogReturn)
  print(f'Portfolio volatility is {portfolioVolatility:.3f}')
  # Plot
  if plot:
    plt.figure(figsize=(8,6))
    plt.title("Portfolio Returns")
    for i in range(len(tickers)):
      plt.plot(stockData['CumReturn',tickers[i]], alpha=0.7, linestyle='dashed', linewidth=0.8, label=f'{tickers[i]} ({weights[i]:.3f})')
    plt.plot(portfolioCumReturn, color='red', label='Portfolio')
    plt.xlabel("Date")
    plt.ylabel('Cumulative Returns')
    plt.legend(frameon=False)
    plt.savefig('portfolioConstruction.pdf')
  # End
  return portfolioCumReturn
