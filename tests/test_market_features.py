import unittest
import math
import pandas as pd
from src.data.yfinance_provider import add_market_features

class TestMarketFeatures(unittest.TestCase):

    def test_known_prices(self):
        """
        Test financial calculations against known results.
        """
        
        # Create a hypothetical three-day price series
        dates = pd.date_range("2024-01-01", periods=3)
        columns = pd.MultiIndex.from_tuples([("Adj Close","TEST")])
        data = pd.DataFrame([[100.],[110.],[99.]],index=dates,columns=columns)
        result = add_market_features(data,volatility_window=2)

        # Verify daily simple returns
        self.assertAlmostEqual(result["Return"]["TEST"].iloc[1],0.10)
        self.assertAlmostEqual(result["Return"]["TEST"].iloc[2],-0.10)

        # Verify the gross return
        self.assertAlmostEqual(result["Gross Return"]["TEST"].iloc[2],0.90)

        # Verify the logarithmic return
        self.assertAlmostEqual(result["Log Return"]["TEST"].iloc[2],math.log(0.90))

        # Verify cumulative growth
        self.assertAlmostEqual(result["Cumulative Growth"]["TEST"].iloc[2],0.99)

        # Verify cumulative return
        self.assertAlmostEqual(result["Cumulative Return"]["TEST"].iloc[2],-0.01)

        # Verify the running peak
        self.assertAlmostEqual(result["Running Peak"]["TEST"].iloc[2], 110.0)

        # Verify drawdown
        self.assertAlmostEqual(result["Drawdown"]["TEST"].iloc[2],-0.10)

    def test_rolling_volatility(self):
        """
        Verify rolling annualized volatility using a known example.
        """

        # Create three hypotherical daily prices
        dates = pd.date_range("2024-01-01", periods=3)
        columns = pd.MultiIndex.from_tuples([("Adj Close","TEST")])
        data = pd.DataFrame([[100.0],[101.0],[99.99]],index=dates,columns=columns)

        # Calculate financial variables using our software
        result=add_market_features(data,volatility_window=2)

        # Calculate the two daily logarithmic returns
        r1 = math.log(1.01)
        r2 = math.log(0.99)

        # Analytical sample standard deviation for two observations
        expected_volatility = (abs(r1-r2)/math.sqrt(2))*math.sqrt(252)

        # Retrieve the volatility calculated by our software
        calculated_volatility = (result["Rolling Volatility"]["TEST"].iloc[2])

        # Verify numerical agreement
        self.assertAlmostEqual(calculated_volatility, expected_volatility)

        # The first two observations lack enough valid log returns
        self.assertTrue(pd.isna(result["Rolling Volatility"]["TEST"].iloc[0]))
        self.assertTrue(pd.isna(result["Rolling Volatility"]["TEST"].iloc[1]))