import unittest
import numpy as np
import pandas as pd

from src.data.validation import validate_market_data

class TestMarketDataValidation(unittest.TestCase):

    def test_missing_prices(self):
        """
        Verify that missing adjusted closing prices are rejected by the validation function
        """

        # Create three hypothetical trading dates
        dates = pd.date_range("2024-01-01",periods=3)

        # Reproduce our field-first MultiIndex structure
        columns = pd.MultiIndex.from_tuples([("Adj Close","TEST")])

        # Deliberately introduce a missing price
        data = pd.DataFrame([[100.0],[np.nan],[105.0]], index=dates, columns=columns)

        # Verify that the validator raises an error
        with self.assertRaisesRegex(ValueError, "Missing adjusted closing prices"):
            validate_market_data(data)

    def test_duplicate_dates(self):
        """
        Verify that duplicate trading dates are rejected.
        """

        # Create a date index containing a duplicate
        dates=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"])

        # Use our field-first MultiIndex structure
        columns = pd.MultiIndex.from_tuples([("Adj Close","TEST")])

        # Create otherwise valid prices
        data = pd.DataFrame([[100.0],[102.0],[103.0]],index=dates,columns=columns)

        # The validator should reject duplicate dates
        with self.assertRaisesRegex(ValueError, "Duplicate trading dates detected"):
            validate_market_data(data)

    def test_nonpositive_prices(self):
        """
        Verify that zero and negative prices are rejected
        """

        # Create three hypothetical trading dates
        dates = pd.date_range("2024-01-01", periods=3)

        # Use our field-first MultiIndex structure
        columns = pd.MultiIndex.from_tuples([("Adj Close", "TEST")])

        # Test both zero and negative prices
        for invalid_price in [0.0, -5.0]:

            with self.subTest(price=invalid_price):

                # Create a dataset containing an invalid price
                data = pd.DataFrame([[100.0], [invalid_price], [105.0]],index=dates, columns=columns)

                # The validator must reject the dataset
                with self.assertRaisesRegex(ValueError, "Non-positive adjusted closing prices"):
                    validate_market_data(data)

    def test_nonfinite_prices(self):
        """
        Verify that infinite adjusted closing prices are rejected.
        """

        # Create three hypothetical trading dates
        dates = pd.date_range("2024-01-01",periods=3)

        # Use our field-first MultiIndex structure
        columns = pd.MultiIndex.from_tuples([("Adj Close","TEST")])

        # Test both zero and negative prices
        for invalid_price in [np.inf, -np.inf]:

            with self.subTest(price=invalid_price):

                # Create a dataset containing an invalid price
                data = pd.DataFrame([[100.0], [invalid_price], [105.0]], index=dates, columns=columns)

                # Verify that validation rejects the dataset
                with self.assertRaisesRegex(ValueError, "Non-finite adjusted closing prices"):
                    validate_market_data(data)

    def test_valid_market_data(self):
        """
        Verify that valid market data is accepted and returned in chronological order.
        """

        # Create three valid but unsorted trading dates
        dates = pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03"])

        # Use our field-first MultiIndex structure
        columns = pd.MultiIndex.from_tuples([("Adj Close", "TEST")])

        # Create valid adjusted closing prices
        data = pd.DataFrame([[105.0], [100.0], [102.0]], index=dates, columns=columns)

        # Validate the dataset
        validated_data = validate_market_data(data)

        # Verify that the dates are sorted chronologically
        self.assertTrue(validated_data.index.is_monotonic_increasing)

        # Verify that the prices remain associated with their original dates
        self.assertEqual(validated_data["Adj Close"]["TEST"].tolist(), [100.0,102.0,105.0])

        # Confirm that validation did not modify the original DataFrame
        self.assertEqual(data["Adj Close"]["TEST"].tolist(), [105.0,100.0,102.0])
