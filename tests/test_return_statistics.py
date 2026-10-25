import math
import unittest
import pandas as pd
from src.analysis.return_statistics import summarize_returns, compute_return_covariance, compute_return_correlation

class TestReturnStatistics(unittest.TestCase):

    def test_summarize_returns(self):

        # Create synthetic daily log returns
        dates = pd.date_range("2025-01-01", periods=5)
        raw_returns = {
            "AAPL": [-0.02, -0.01, 0.00, 0.01, 0.05],
            "MSFT": [-0.01, 0.00, 0.015, 0.02, 0.04]
        }
        columns = pd.MultiIndex.from_product([["Log Return"], raw_returns.keys()])
        data = pd.DataFrame(list(zip(*raw_returns.values())), index=dates, columns=columns)

        # Results from our Quantflux function
        summary = summarize_returns(data)

        # Check tickers
        self.assertEqual(list(summary.index),list(raw_returns.keys()))

        # Independently calculate statistics for each ticker
        for ticker, values in raw_returns.items():            
            n = len(values)

            # Sample mean
            mean = sum(values) / n

            # Deviations from the mean
            deviations = [x - mean for x in values]

            # Sample standard deviation
            variance = sum(d**2 for d in deviations)/(n-1)
            std = math.sqrt(variance)

            # Bias-corrected sample skewness
            skewness = n / ((n-1)*(n-2)) * sum((d/std)**3 for d in deviations)

            # Bias-corrected excess kurtosis
            kurtosis = n*(n+1) / ((n-1)*(n-2)*(n-3)) * sum((d/std)**4 for d in deviations) - 3*(n-1)**2 / ((n-2)*(n-3))

            # Compare independent results with Quantflux function
            self.assertEqual(summary.loc[ticker, "Count"], n)
            self.assertAlmostEqual(summary.loc[ticker, "Mean"], mean)
            self.assertAlmostEqual(summary.loc[ticker, "Std Dev"], std)
            self.assertAlmostEqual(summary.loc[ticker, "Skewness"], skewness)
            self.assertAlmostEqual(summary.loc[ticker, "Excess Kurtosis"], kurtosis) 

    def test_summarize_returns_missing_values(self):
        # Synthetic returns containing missing observations
        dates = pd.date_range("2025-01-01", periods=6)
        raw_returns = {
            "AAPL": [0.01, -0.02, float("nan"), 0.03, 0.00, -0.01],
            "MSFT": [float("nan"), 0.02, -0.01, 0.04, 0.01, 0.03]
        }
        data = pd.DataFrame(raw_returns, index=dates)
        data.columns = pd.MultiIndex.from_product([["Log Return"], data.columns])

        # Calculate statistics using Quantflux function
        summary = summarize_returns(data)

        # Independently verify statistics after excluding NaNs
        for ticker, values in raw_returns.items():
            valid = [x for x in values if not math.isnan(x)]

            # Counts
            n = len(valid)

            # Mean
            mean = sum(valid) / n

            # Standard deviation 
            variance = sum((x - mean)**2 for x in valid) / (n - 1)
            std = math.sqrt(variance)

            # Compare results
            self.assertEqual(summary.loc[ticker, "Count"], n)
            self.assertAlmostEqual(summary.loc[ticker, "Mean"], mean)
            self.assertAlmostEqual(summary.loc[ticker, "Std Dev"], std)

            # Higher moments should remain computable
            self.assertTrue(math.isfinite(summary.loc[ticker, "Skewness"]))
            self.assertTrue(math.isfinite(summary.loc[ticker, "Excess Kurtosis"]))

    def test_compute_return_covariance(self):
        # Synthetic daily log returns
        dates = pd.date_range("2025-01-01", periods=5)
        raw_returns = {
            "AAPL": [-0.02, -0.01, 0.00, 0.01, 0.05],
            "MSFT": [0.03, 0.01, -0.02, 0.04, 0.00]
        }
        data = pd.concat({"Log Return": pd.DataFrame(raw_returns, index=dates)}, axis=1)

        # Calculate the covariance matrix using Quantflux
        covariance = compute_return_covariance(data)

        # Verify matrix dimensions and ticker labels
        self.assertEqual(covariance.shape, (2,2))
        self.assertEqual(list(covariance.index), list(raw_returns))
        self.assertEqual(list(covariance.columns), list(raw_returns))

        # Independently calculate each covariance element
        for ticker_x, x in raw_returns.items():
            for ticker_y, y in raw_returns.items():
                n = len(x)
                mean_x = sum(x) / n
                mean_y = sum(y) / n
                expected_covariance = sum((xi - mean_x)*(yi - mean_y) for xi, yi in zip(x,y)) / (n-1)
                self.assertAlmostEqual(covariance.loc[ticker_x, ticker_y], expected_covariance)

    def test_compute_return_correlation(self):
        # Synthetic daily logarithmic returns
        dates = pd.date_range("2025-01-01", periods=5)
        raw_returns = {
            "AAPL": [-0.02, -0.01, 0.00, 0.01, 0.05],
            "MSFT": [0.03, 0.01, -0.02, 0.04, 0.00]
        }
        data = pd.concat({"Log Return": pd.DataFrame(raw_returns, index=dates)}, axis=1)

        # Calculate correlation matrix using Quantflux
        correlation = compute_return_correlation(data)

        # Verify matrix dimensions and ticker labels
        self.assertEqual(correlation.shape, (2, 2))
        self.assertEqual(list(correlation.index), list(raw_returns))
        self.assertEqual(list(correlation.columns), list(raw_returns))

        # Independently calculate each correlation coefficient
        for ticker_x, x in raw_returns.items():
            for ticker_y, y in raw_returns.items():

                mean_x = sum(x) / len(x)
                mean_y = sum(y) / len(y)

                # Deviations from the respective means
                dx = [xi - mean_x for xi in x]
                dy = [yi - mean_y for yi in y]

                # Pearson correlation from its mathematical definition
                numerator = sum(a * b for a, b in zip(dx, dy))
                denominator = (sum(a**2 for a in dx)* sum(b**2 for b in dy)) ** 0.5
                expected_correlation = numerator / denominator

                self.assertAlmostEqual(correlation.loc[ticker_x, ticker_y], expected_correlation)