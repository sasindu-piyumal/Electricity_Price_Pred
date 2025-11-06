#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit test for the outlier removal bug fix in Electricity Data.py

Bug: Line 128 originally used OR (|) operator instead of AND (&) operator
for filtering SMPEP2 outliers, which resulted in keeping almost all values
instead of properly filtering them.

This test demonstrates:
- The buggy behavior (using OR): keeps all values except exactly 0
- The fixed behavior (using AND): keeps only values in range (0, 550]
"""

import unittest
import pandas as pd
import numpy as np


class TestOutlierRemoval(unittest.TestCase):
    """Test suite for SMPEP2 outlier removal logic"""
    
    def setUp(self):
        """Create test data with various SMPEP2 values"""
        # Create a test dataframe with various SMPEP2 values including:
        # - Negative values (should be removed)
        # - Zero (should be removed)  
        # - Positive values in range (should be kept)
        # - Values above 550 (should be removed)
        self.test_data = pd.DataFrame({
            'SMPEP2': [-100, -10, 0, 10, 100, 300, 500, 550, 600, 1000, np.nan]
        })
    
    def test_buggy_or_logic(self):
        """
        Test the BUGGY behavior using OR (|) operator.
        This should FAIL to properly filter outliers.
        
        With OR logic: (SMPEP2 > 0) | (SMPEP2 <= 550)
        - Any positive value passes (>0)
        - Any value <=550 passes (including negatives)
        - Only value that fails BOTH conditions is exactly 0
        """
        df_cleaned = self.test_data.copy()
        
        # Buggy logic using OR
        SMPEP2_out_buggy = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
        df_filtered_buggy = df_cleaned[SMPEP2_out_buggy]
        
        # With buggy OR logic, almost everything is kept
        # Only 0 is excluded (fails both >0 and <=550 is true, but we want outliers removed)
        expected_kept_values = [-100, -10, 10, 100, 300, 500, 550, 600, 1000]
        
        # Remove NaN for comparison
        actual_values = df_filtered_buggy['SMPEP2'].dropna().tolist()
        
        # This demonstrates the bug: negative and very high values are kept
        self.assertIn(-100, actual_values, "Bug: negative value -100 should be removed but is kept")
        self.assertIn(-10, actual_values, "Bug: negative value -10 should be removed but is kept")  
        self.assertIn(600, actual_values, "Bug: outlier 600 should be removed but is kept")
        self.assertIn(1000, actual_values, "Bug: outlier 1000 should be removed but is kept")
        
        # Zero should be excluded
        self.assertNotIn(0, actual_values, "Zero should be excluded even with buggy logic")
    
    def test_fixed_and_logic(self):
        """
        Test the FIXED behavior using AND (&) operator.
        This should CORRECTLY filter outliers.
        
        With AND logic: (SMPEP2 > 0) & (SMPEP2 <= 550)
        - Value must be > 0 AND <= 550
        - Removes: negatives, zero, and values > 550
        - Keeps: only values in range (0, 550]
        """
        df_cleaned = self.test_data.copy()
        
        # Fixed logic using AND
        SMPEP2_out_fixed = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
        df_filtered_fixed = df_cleaned[SMPEP2_out_fixed]
        
        # With correct AND logic, only values in (0, 550] are kept
        expected_kept_values = [10, 100, 300, 500, 550]
        
        # Remove NaN for comparison
        actual_values = df_filtered_fixed['SMPEP2'].dropna().tolist()
        
        # Verify correct values are kept
        for val in expected_kept_values:
            self.assertIn(val, actual_values, f"Value {val} should be kept")
        
        # Verify outliers are removed
        self.assertNotIn(-100, actual_values, "Negative value -100 should be removed")
        self.assertNotIn(-10, actual_values, "Negative value -10 should be removed")
        self.assertNotIn(0, actual_values, "Zero should be removed")
        self.assertNotIn(600, actual_values, "Outlier 600 should be removed")
        self.assertNotIn(1000, actual_values, "Outlier 1000 should be removed")
        
        # Verify the exact count
        self.assertEqual(len(actual_values), 5, "Should keep exactly 5 values in range (0, 550]")
    
    def test_comparison_or_vs_and(self):
        """
        Direct comparison showing the difference between buggy OR and fixed AND logic.
        """
        df_cleaned = self.test_data.copy()
        
        # Buggy OR logic
        mask_or = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
        count_or = mask_or.sum()
        
        # Fixed AND logic  
        mask_and = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
        count_and = mask_and.sum()
        
        # The bug causes many more values to be kept
        self.assertGreater(count_or, count_and, 
                          "Buggy OR logic keeps more values than correct AND logic")
        
        # Specifically, OR keeps 9 values (all except 0 and NaN)
        # while AND keeps only 5 values (10, 100, 300, 500, 550)
        self.assertEqual(count_or, 9, "Buggy OR logic keeps 9 values")
        self.assertEqual(count_and, 5, "Fixed AND logic keeps 5 values")


if __name__ == '__main__':
    unittest.main()
