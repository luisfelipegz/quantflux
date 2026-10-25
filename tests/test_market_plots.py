import unittest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

from src.visualization.market_plots import plot_prices, plot_cumulative_returns, plot_rolling_volatility, plot_drawdowns, plot_return_distributions, plot_return_boxplots, plot_return_qq, plot_correlation_heatmap

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

    def test_plot_return_distributions(self):
        
        # Create synthetic daily logarithmic returns
        data = self.data.copy()
        data[("Log Return", "AAPL")] = [-0.02,0.00,0.03]
        data[("Log Return", "MSFT")] = [-0.01,0.01,0.02]

        # Generate histograms without displaying them
        fig, axes = plot_return_distributions(data, bins=5, show=False)

        try:
            # Verify one subplot per ticker
            self.assertEqual(len(axes),2)

            for ax in axes:
                # Verify histogram bars exist
                self.assertGreater(len(ax.patches),0)

                # Verify a Gaussian benchmark was plotted
                self.assertEqual(len(ax.lines),1)
                self.assertEqual(ax.lines[0].get_label(),"Gaussian Benchmark")

                # Verify histogram density normalization
                area = sum(bar.get_height()*bar.get_width() for bar in ax.patches)
                self.assertAlmostEqual(area, 1.0)

                # Verify percentage formatting of returns
                self.assertIsInstance(ax.xaxis.get_major_formatter(),PercentFormatter)

        finally:
            plt.close(fig)

    def test_plot_return_boxplots(self):
        # Create synthetic daily log returns
        data = self.data.copy()
        aapl = [-0.03,0.01,0.04]
        msft = [-0.02,0.00,0.03]
        data[("Log Return", "AAPL")] = aapl
        data[("Log Return", "MSFT")] = msft

        # Generate box plots without displaying them
        fig, ax = plot_return_boxplots(data,show=False)

        try:
            # Verify ticker labels
            labels = [tick.get_text() for tick in ax.get_xticklabels()]
            self.assertEqual(labels, ["AAPL", "MSFT"])

            # Verify mean markers and their numerical values
            mean_markers = [line for line in ax.lines if line.get_marker()=="^"]
            self.assertEqual(len(mean_markers),2)
            self.assertAlmostEqual(mean_markers[0].get_ydata()[0], sum(aapl)/len(aapl))
            self.assertAlmostEqual(mean_markers[1].get_ydata()[0], sum(msft)/len(msft))

            # Verify percentage formatting
            self.assertIsInstance(ax.yaxis.get_major_formatter(),PercentFormatter)

        finally:
            plt.close(fig)

    def test_plot_return_qq(self):

        # Create synthetic daily logarithmic returns
        data = self.data.copy()
        raw_returns = {
            "AAPL": [-0.02, 0.00, 0.03],
            "MSFT": [-0.01, 0.01, 0.02]
        }
        for ticker, values in raw_returns.items():
            data[("Log Return", ticker)] = values

        # Generate Q-Q plots without displaying them
        fig, axes = plot_return_qq(data, show=False)

        try:
            # Verify one subplot per ticker
            self.assertEqual(len(axes), 2)

            for ax, (ticker, values) in zip(axes, raw_returns.items()):
                # Retrieve the plotted Q-Q coordinates
                points = ax.collections[0].get_offsets()

                # One point per historical observation
                self.assertEqual(points.shape, (3, 2))

                # Verify observed quantiles are sorted returns
                for observed, expected in zip(points[:, 1], sorted(values)):
                    self.assertAlmostEqual(observed, expected)

                # Gaussian median equals the sample mean
                self.assertAlmostEqual(points[1, 0], sum(values) / len(values))

                # Verify the reference line represents y = x
                reference = ax.lines[0]
                self.assertEqual(list(reference.get_xdata()), list(reference.get_ydata()))

                # Verify percentage formatting
                self.assertIsInstance(ax.xaxis.get_major_formatter(), PercentFormatter)
                self.assertIsInstance(ax.yaxis.get_major_formatter(), PercentFormatter)

        finally:
            plt.close(fig)

    def test_plot_correlation_heatmap(self):
        # Synthetic correlation matrix
        correlation = pd.DataFrame([[1.0,-0.4], [-0.4,1.0]], index=["AAPL","MSFT"], columns=["AAPL","MSFT"])

        # Generate heatmap without displaying it
        fig, ax = plot_correlation_heatmap(correlation, show=False)

        try:
            # Retrieve the heatmap image
            image = ax.images[0]

            # Verify the numerical correlation matrix
            self.assertEqual(image.get_array().tolist(), correlation.values.tolist())

            # Verify ticker labels
            self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],["AAPL", "MSFT"])
            self.assertEqual([tick.get_text() for tick in ax.get_yticklabels()],["AAPL", "MSFT"])

            # Verify numerical annotations
            annotations = [text.get_text() for text in ax.texts]
            self.assertEqual(annotations,["1.00", "-0.40", "-0.40", "1.00"])

            # Verify diverging colormap
            self.assertEqual(image.get_cmap().name, "RdBu_r")

            # Verify fixed correlation scale
            self.assertEqual(image.get_clim(), (-1, 1))

        finally:
            plt.close(fig)