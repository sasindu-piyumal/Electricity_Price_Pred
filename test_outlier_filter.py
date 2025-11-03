#!/usr/bin/env python
# coding: utf-8
"""
Unit test for the SMPEP2 outlier filtering bug in Electricity Data.py

This test demonstrates a bug on line 128 where OR (|) is used instead of AND (&)
for filtering outliers in the SMPEP2 column.

The bug: SMPEP2_out = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
The fix: SMPEP2_out = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
"""

import pandas as pd
import numpy as np
import unittest


class TestOutlierFiltering(unittest.TestCase):
    """Test cases for SMPEP2 outlier filtering logic"""
    
    def setUp(self):
        """Create sample data with known outliers"""
        # Create a dataframe with various SMPEP2 values including outliers
        self.test_data = pd.DataFrame({
            'SMPEP2': [-100, -10, 0, 50, 100, 250, 550, 600, 1000, np.nan]
        })
        
    def test_buggy_filter_with_or(self):
        """
        Test the BUGGY version with OR logic
        This should FAIL to filter outliers properly
        """
        df_cleaned = self.test_data.copy()
        
        # Buggy logic from line 128 (using OR)
        SMPEP2_out = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
        df_result = df_cleaned[SMPEP2_out]
        
        # With OR logic, almost all values pass through
        # Negative values pass because they are <= 550
        # Large values (>550) pass because they are > 0
        # Only value that might fail is NaN
        
        # Check that negative values are incorrectly kept (BUG!)
        self.assertTrue(-100 in df_result['SMPEP2'].values, 
                       "Buggy OR logic incorrectly keeps -100")
        self.assertTrue(-10 in df_result['SMPEP2'].values, 
                       "Buggy OR logic incorrectly keeps -10")
        
        # Check that values > 550 are incorrectly kept (BUG!)
        self.assertTrue(600 in df_result['SMPEP2'].values, 
                       "Buggy OR logic incorrectly keeps 600")
        self.assertTrue(1000 in df_result['SMPEP2'].values, 
                       "Buggy OR logic incorrectly keeps 1000")
        
        # The buggy version keeps 9 out of 10 values (all except NaN)
        # This demonstrates the bug doesn't filter outliers
        self.assertEqual(len(df_result), 9, 
                        "Buggy OR logic should keep 9 values (fails to filter outliers)")
        
    def test_fixed_filter_with_and(self):
        """
        Test the FIXED version with AND logic
        This should PASS and properly filter outliers
        """
        df_cleaned = self.test_data.copy()
        
        # Fixed logic (using AND)
        SMPEP2_out = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
        df_result = df_cleaned[SMPEP2_out]
        
        # With AND logic, only values in range (0, 550] pass
        
        # Check that negative values are correctly removed
        self.assertFalse(-100 in df_result['SMPEP2'].values, 
                        "Fixed AND logic correctly removes -100")
        self.assertFalse(-10 in df_result['SMPEP2'].values, 
                        "Fixed AND logic correctly removes -10")
        
        # Check that zero is correctly removed (> 0, not >= 0)
        self.assertFalse(0 in df_result['SMPEP2'].values, 
                        "Fixed AND logic correctly removes 0")
        
        # Check that values > 550 are correctly removed
        self.assertFalse(600 in df_result['SMPEP2'].values, 
                        "Fixed AND logic correctly removes 600")
        self.assertFalse(1000 in df_result['SMPEP2'].values, 
                        "Fixed AND logic correctly removes 1000")
        
        # Check that valid values are kept
        self.assertTrue(50 in df_result['SMPEP2'].values, 
                       "Fixed AND logic keeps valid value 50")
        self.assertTrue(100 in df_result['SMPEP2'].values, 
                       "Fixed AND logic keeps valid value 100")
        self.assertTrue(250 in df_result['SMPEP2'].values, 
                       "Fixed AND logic keeps valid value 250")
        self.assertTrue(550 in df_result['SMPEP2'].values, 
                       "Fixed AND logic keeps boundary value 550")
        
        # The fixed version keeps only 4 valid values (50, 100, 250, 550)
        self.assertEqual(len(df_result), 4, 
                        "Fixed AND logic should keep only 4 valid values")
        
    def test_comparison_buggy_vs_fixed(self):
        """
        Direct comparison showing the difference between buggy and fixed logic
        """
        df_cleaned = self.test_data.copy()
        
        # Buggy version (OR)
        buggy_mask = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
        buggy_result = df_cleaned[buggy_mask]
        
        # Fixed version (AND)
        fixed_mask = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
        fixed_result = df_cleaned[fixed_mask]
        
        # The buggy version keeps many more rows (incorrectly)
        self.assertGreater(len(buggy_result), len(fixed_result),
                          "Buggy version should keep more rows than fixed version")
        
        # Specifically, buggy keeps 9 rows, fixed keeps 4 rows
        self.assertEqual(len(buggy_result), 9, "Buggy version keeps 9 rows")
        self.assertEqual(len(fixed_result), 4, "Fixed version keeps 4 rows")
        

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
