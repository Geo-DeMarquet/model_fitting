import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('linear_data.csv') 

class TestLinearRegression(unittest.TestCase):
    
    def test_two_columns(self):
        """Check if the data file has exactly two columns."""
        self.assertEqual(df.shape[1], 2, "The data file must have exactly two columns.")
    
    def test_no_header_lines(self):
        """Check if the data file has headers."""
        self.assertIsInstance(df.columns[0], str, "The data file must have headers.")

    def test_no_missing_or_non_numeric_data(self):
        """Check for missing or non-numeric data."""
        self.assertFalse(df.isnull().values.any(), "The data file contains missing data.")
        self.assertTrue(np.issubdtype(df.dtypes[0], np.number) and np.issubdtype(df.dtypes[1], np.number),
                        "The data file contains non-numeric data.")

    def test_enough_points(self):
        """Check if the data file has enough points to fit a line."""
        self.assertGreaterEqual(len(df), 2, "Not enough data points to fit a line.")
    
    def test_sufficient_correlation(self):
        """Check if data points are sufficiently correlated."""
        correlation = np.corrcoef(df['X'], df['Y'])[0, 1]
        self.assertGreater(abs(correlation), 0.5, "Data points aren't sufficiently correlated.")
    
    def test_correct_slope_and_intercept(self):
        """Check if the code returns correct slope and intercept."""
        x = df[['X']].values
        y = df['Y'].values
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        expected_slope = 3.21 
        expected_intercept = 1.48  #including expected data
        
        self.assertAlmostEqual(slope, expected_slope, places=2, msg="Incorrect slope.")
        self.assertAlmostEqual(intercept, expected_intercept, places=2, msg="Incorrect intercept.")

if __name__ == '__main__':
    unittest.main() #taken from Stack Overflow note - main associated with unittest is actually an instance of TestProgram which, when instantiated, runs all your tests.
