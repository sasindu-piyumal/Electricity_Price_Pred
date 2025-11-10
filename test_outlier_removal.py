#!/usr/bin/env python
# coding: utf-8
"""
Unit test for the SMPEP2 outlier removal bug fix.

This test demonstrates that the original code (line 128) incorrectly uses OR (|) 
instead of AND (&) for filtering outliers, which causes it to keep values that 
should be removed (negatives and values > 550).
"""

import pandas as pd
import numpy as np
import unittest


class TestOutlierRemoval(unittest.TestCase):
    """Test case for SMPEP2 outlier removal logic."""
    
    def setUp(self):
        """Create a test dataframe with known outliers."""
        # Create sample data with various SMPEP2 values
        self.df_test = pd.DataFrame({
            'SMPEP2': [-100, -10, 0, 10, 50, 100, 300, 550, 551, 600, 1000],
            'other_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        })
    
    def test_buggy_outlier_removal(self):
        """
        Test the BUGGY version with OR (|) operator.
        This should fail because it doesn't properly remove outliers.
        """
        # This is the BUGGY logic from line 128
        SMPEP2_out = (self.df_test['SMPEP2'] > 0) | (self.df_test['SMPEP2'] <= 550)
        df_result = self.df_test[SMPEP2_out]
        
        # The buggy version keeps almost all values (10 out of 11)
        # It only removes values where BOTH conditions are False (none in our case)
        self.assertEqual(len(df_result), 11, 
                        "Buggy version keeps almost all values - demonstrating the bug")
        
        # Check that it incorrectly keeps negative values
        self.assertTrue((df_result['SMPEP2'] < 0).any(), 
                       "Buggy version incorrectly keeps negative values")
        
        # Check that it incorrectly keeps values > 550
        self.assertTrue((df_result['SMPEP2'] > 550).any(), 
                       "Buggy version incorrectly keeps values > 550")
    
    def test_fixed_outlier_removal(self):
        """
        Test the FIXED version with AND (&) operator.
        This should pass because it properly removes outliers.
        """
        # This is the FIXED logic (using AND instead of OR)
        SMPEP2_out = (self.df_test['SMPEP2'] > 0) & (self.df_test['SMPEP2'] <= 550)
        df_result = self.df_test[SMPEP2_out]
        
        # The fixed version should keep only values in range (0, 550]
        # Expected values: 10, 50, 100, 300, 550 = 5 values
        self.assertEqual(len(df_result), 5, 
                        "Fixed version keeps only values in range (0, 550]")
        
        # Check that all values are > 0
        self.assertTrue((df_result['SMPEP2'] > 0).all(), 
                       "Fixed version keeps only positive values")
        
        # Check that all values are <= 550
        self.assertTrue((df_result['SMPEP2'] <= 550).all(), 
                       "Fixed version keeps only values <= 550")
        
        # Verify the exact values that should remain
        expected_values = [10, 50, 100, 300, 550]
        actual_values = sorted(df_result['SMPEP2'].tolist())
        self.assertEqual(actual_values, expected_values, 
                        "Fixed version keeps exactly the expected values")
    
    def test_edge_cases(self):
        """Test edge cases: exactly 0 and exactly 550."""
        # Create dataframe with edge cases
        df_edge = pd.DataFrame({
            'SMPEP2': [0, 550],
            'other_col': [1, 2]
        })
        
        # Fixed logic with AND
        SMPEP2_out = (df_edge['SMPEP2'] > 0) & (df_edge['SMPEP2'] <= 550)
        df_result = df_edge[SMPEP2_out]
        
        # Should keep 550 but not 0
        self.assertEqual(len(df_result), 1, "Should keep only 550")
        self.assertEqual(df_result['SMPEP2'].iloc[0], 550, "Should keep 550")


if __name__ == '__main__':
    print("Running unit tests for SMPEP2 outlier removal bug fix...")
    print("=" * 70)
    print("BUG DESCRIPTION:")
    print("Line 128 uses OR (|) instead of AND (&) for outlier filtering.")
    print("Original: (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)")
    print("Fixed:    (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)")
    print("=" * 70)
    print()
    
    # Run tests
    unittest.main(verbosity=2)
