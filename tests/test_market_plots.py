import unittest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

from src.visualization.market_plots import plot_prices, plot_cumulative_returns, plot_rolling_volatility, plot_drawdowns

class TestMarketPlots(unittest.TestCase):

    def setUp(self):

        # Create synthetic historical stock prices
        dates = pd.date_range("2025-01-01", periods=3)

        columns = pd.MultiIndex.from_product([["Adj Close"],["AAPL","MSFT"]])

        self.data=pd.DataFrame([[100,200],[105,202],[103,205]],index=dates,columns=columns)

    def test_plot_prices(self):

        # Generate plot without displaying it
        fig, ax = plot_prices(self.data, show=False)

        try:
            # Verify that two ticker series were plotted
            self.assertEqual(len(ax.lines),2)
            
            # Verify ticker labels
            self.assertEqual([line.get_label() for line in ax.lines], ["AAPL","MSFT"])

            # Verify plotted prices
            self.assertEqual(list(ax.lines[0].get_ydata()),[100,105,103]) 
            self.assertEqual(list(ax.lines[1].get_ydata()),[200,202,205])

        finally:
            # Close the figure after testing
            plt.close(fig)

    def test_plot_cumulative_returns(self):

        # Add known cumulative returns to our synthetic data
        data = self.data.copy()
        data[("Cumulative Return", "AAPL")] = [0.00,0.05,0.03]
        data[("Cumulative Return", "MSFT")] = [0.00,0.01,0.025]

        # Generate plot without displaying it
        fig, ax = plot_cumulative_returns(data, show=False)

        try:
            # Verify that both tickers were plotted
            self.assertEqual(len(ax.lines),2)

            # Verify ticker labels
            self.assertEqual([line.get_label() for line in ax.lines],["AAPL","MSFT"])

            # Verify plotted cumulative returns
            self.assertEqual(list(ax.lines[0].get_ydata()),[0.00,0.05,0.03])
            self.assertEqual(list(ax.lines[1].get_ydata()),[0.00,0.01,0.025])

            # Verify percentage formatting
            self.assertIsInstance(ax.yaxis.get_major_formatter(),PercentFormatter)

        finally:
            plt.close(fig)

    def test_plot_rolling_volatility(self):
    
        # Add known annualized volatility values
        data = self.data.copy()
        data[("Rolling Volatility","AAPL")] = [float('nan'),0.20,0.25]
        data[("Rolling Volatility","MSFT")] = [float('nan'),0.15,0.18]

        # Generate plot without displaying it
        fig, ax = plot_rolling_volatility(data, show=False)

        try:
            # Verify that both tickers were plotted
            self.assertEqual(len(ax.lines), 2)

            # Verify ticker labels
            self.assertEqual([line.get_label() for line in ax.lines],["AAPL","MSFT"])

            # Verify plotted volatility values
            self.assertEqual(list(ax.lines[0].get_ydata()[1:]),[0.20,0.25])
            self.assertEqual(list(ax.lines[1].get_ydata()[1:]),[0.15,0.18])

            # Verify percentage formatting
            self.assertIsInstance(ax.yaxis.get_major_formatter(),PercentFormatter)

        finally:
            plt.close(fig)
    
    def test_plot_drawdowns(self):

        # Add known drawdown values to our synthetic data
        data = self.data.copy()
        data[("Drawdown", "AAPL")] = [0.00,-0.10,-0.05]
        data[("Drawdown", "MSFT")] = [0.00,-0.02,0.00]

        # Generate plot without displaying it
        fig, ax = plot_drawdowns(data, show=False)

        try:
            # Verify that both tickers were plotted
            self.assertEqual(len(ax.lines),3)

            # Verify ticker labels
            self.assertEqual([line.get_label() for line in ax.lines[:2]],["AAPL","MSFT"])

            # Verify plotted drawdown values
            self.assertEqual(list(ax.lines[0].get_ydata()),[0.00,-0.10,-0.05])
            self.assertEqual(list(ax.lines[1].get_ydata()),[0.00,-0.02,0.00])

            # Verify the horizontal reference line at zero
            self.assertEqual(list(ax.lines[2].get_ydata()),[0,0])

            # Verify percentage formatting
            self.assertIsInstance(ax.yaxis.get_major_formatter(),PercentFormatter)

        finally:
            plt.close(fig)