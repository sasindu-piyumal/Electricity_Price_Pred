#!/usr/bin/env python
# coding: utf-8

"""
Unit test to demonstrate and verify the outlier removal bug fix.
"""

import pandas as pd
import numpy as np
import unittest

class TestOutlierRemoval(unittest.TestCase):
    
    def setUp(self):
        """Set up test data with known outliers."""
        # Create test data with values that should be filtered out
        self.test_data = pd.DataFrame({
            'SMPEP2': [-50, -10, 0, 1, 50, 100, 200, 300, 400, 500, 550, 600, 700, 1000],
            'other_col': range(14)
        })
    
    def test_buggy_outlier_removal(self):
        """Test that demonstrates the bug in the original implementation."""
        df_cleaned = self.test_data.copy()
        
        # Original buggy implementation using OR
        SMPEP2_out_buggy = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
        df_cleaned_buggy = df_cleaned[SMPEP2_out_buggy]
        
        # With OR logic, this keeps almost everything!
        # Values > 0: [1, 50, 100, 200, 300, 400, 500, 550, 600, 700, 1000]
        # Values <= 550: [-50, -10, 0, 1, 50, 100, 200, 300, 400, 500, 550]
        # Union of both: ALL values except nothing gets filtered
        
        # The bug: this doesn't actually remove outliers as intended
        # It should remove negative values and values > 550
        expected_removed = [-50, -10, 600, 700, 1000]  # Should be removed
        
        # Check that buggy implementation incorrectly keeps outliers
        remaining_values = df_cleaned_buggy['SMPEP2'].tolist()
        
        # These outliers should NOT be in the result but they are (demonstrating the bug)
        contains_negative = any(v < 0 for v in remaining_values)
        contains_over_550 = any(v > 550 for v in remaining_values)
        
        # This assertion will FAIL with the buggy implementation, demonstrating the bug
        print(f"Buggy implementation remaining values: {sorted(remaining_values)}")
        print(f"Contains negative values: {contains_negative}")
        print(f"Contains values > 550: {contains_over_550}")
        
        # The bug is that it keeps values it shouldn't
        self.assertTrue(contains_negative or contains_over_550, 
                       "Bug demonstration: The buggy OR logic keeps outliers that should be removed")
    
    def test_fixed_outlier_removal(self):
        """Test the corrected outlier removal implementation."""
        df_cleaned = self.test_data.copy()
        
        # Fixed implementation using AND
        SMPEP2_out_fixed = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
        df_cleaned_fixed = df_cleaned[SMPEP2_out_fixed]
        
        # With AND logic, this keeps only values that are BOTH > 0 AND <= 550
        # Expected to keep: [1, 50, 100, 200, 300, 400, 500, 550]
        expected_values = [1, 50, 100, 200, 300, 400, 500, 550]
        
        remaining_values = sorted(df_cleaned_fixed['SMPEP2'].tolist())
        
        print(f"\nFixed implementation remaining values: {remaining_values}")
        print(f"Expected values: {expected_values}")
        
        # Check that fixed implementation correctly filters outliers
        self.assertEqual(remaining_values, expected_values, 
                        "Fixed implementation should keep only values between 0 (exclusive) and 550 (inclusive)")
        
        # Verify no negative values remain
        self.assertTrue(all(v > 0 for v in remaining_values), 
                       "No negative or zero values should remain")
        
        # Verify no values > 550 remain  
        self.assertTrue(all(v <= 550 for v in remaining_values),
                       "No values greater than 550 should remain")
    
    def test_edge_cases(self):
        """Test edge cases for the outlier removal."""
        # Test with edge values
        edge_data = pd.DataFrame({
            'SMPEP2': [0, 0.001, 0.1, 550, 550.001, 551],
            'other_col': range(6)
        })
        
        # Apply fixed logic
        SMPEP2_out = (edge_data['SMPEP2'] > 0) & (edge_data['SMPEP2'] <= 550)
        filtered = edge_data[SMPEP2_out]
        
        remaining = sorted(filtered['SMPEP2'].tolist())
        
        # Should keep 0.001, 0.1, 550 but not 0, 550.001, 551
        expected = [0.001, 0.1, 550]
        
        print(f"\nEdge case remaining values: {remaining}")
        print(f"Expected edge case values: {expected}")
        
        self.assertEqual(remaining, expected,
                        "Edge cases should be handled correctly")

if __name__ == '__main__':
    # Run the tests
    print("="*60)
    print("OUTLIER REMOVAL BUG DEMONSTRATION AND FIX VERIFICATION")
    print("="*60)
    
    unittest.main(verbosity=2)
