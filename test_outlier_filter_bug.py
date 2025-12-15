#!/usr/bin/env python
# coding: utf-8

"""
Unit test to demonstrate the logical bug in outlier filtering.

Bug Description:
The outlier filter for SMPEP2 uses OR (|) instead of AND (&):
  SMPEP2_out = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
  
This should be:
  SMPEP2_out = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)

With OR logic:
- Negative values are kept (because they satisfy <= 550)
- Values > 550 are kept (because they satisfy > 0)
- Only values exactly 0 would be filtered out

With AND logic (correct):
- Only values where 0 < SMPEP2 <= 550 are kept
- Negative values are removed
- Values > 550 are removed
"""

import pandas as pd
import numpy as np
import unittest


class TestOutlierFilterBug(unittest.TestCase):
    """Test suite for the SMPEP2 outlier filter bug."""
    
    def setUp(self):
        """Create test data with various outlier scenarios."""
        self.test_data = pd.DataFrame({
            'SMPEP2': [-100, -10, -1, 0, 1, 50, 100, 500, 550, 551, 600, 1000]
        })
        
    def test_buggy_or_filter(self):
        """Test the buggy OR filter - this demonstrates the bug."""
        df_cleaned = self.test_data.copy()
        
        # This is the buggy version with OR
        SMPEP2_out_buggy = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
        df_filtered_buggy = df_cleaned[SMPEP2_out_buggy]
        
        # With OR logic, almost all values pass:
        # -100: False | True = True (kept)
        # -10: False | True = True (kept)
        # -1: False | True = True (kept)
        # 0: False | True = True (kept)
        # 1: True | True = True (kept)
        # 50: True | True = True (kept)
        # 550: True | True = True (kept)
        # 551: True | False = True (kept)
        # 600: True | False = True (kept)
        # 1000: True | False = True (kept)
        
        # Only value 0 might have ambiguous behavior, but let's check
        filtered_values_buggy = df_filtered_buggy['SMPEP2'].tolist()
        
        # The buggy filter keeps negative values
        self.assertIn(-100, filtered_values_buggy, 
                     "Bug: Negative values should be filtered but are kept with OR logic")
        self.assertIn(-10, filtered_values_buggy,
                     "Bug: Negative values should be filtered but are kept with OR logic")
        
        # The buggy filter keeps values > 550
        self.assertIn(551, filtered_values_buggy,
                     "Bug: Values > 550 should be filtered but are kept with OR logic")
        self.assertIn(600, filtered_values_buggy,
                     "Bug: Values > 550 should be filtered but are kept with OR logic")
        self.assertIn(1000, filtered_values_buggy,
                     "Bug: Values > 550 should be filtered but are kept with OR logic")
        
        # With the buggy OR filter, we keep 11 or 12 out of 12 values
        # (depending on how 0 is handled)
        self.assertGreaterEqual(len(df_filtered_buggy), 11,
                               "Bug: OR filter keeps almost all values")
        
    def test_correct_and_filter(self):
        """Test the correct AND filter - this is the expected behavior."""
        df_cleaned = self.test_data.copy()
        
        # This is the correct version with AND
        SMPEP2_out_correct = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
        df_filtered_correct = df_cleaned[SMPEP2_out_correct]
        
        # With AND logic, only values in range (0, 550] pass:
        # -100: False & True = False (filtered)
        # -10: False & True = False (filtered)
        # -1: False & True = False (filtered)
        # 0: False & True = False (filtered)
        # 1: True & True = True (kept)
        # 50: True & True = True (kept)
        # 550: True & True = True (kept)
        # 551: True & False = False (filtered)
        # 600: True & False = False (filtered)
        # 1000: True & False = False (filtered)
        
        filtered_values_correct = df_filtered_correct['SMPEP2'].tolist()
        
        # The correct filter removes negative values
        self.assertNotIn(-100, filtered_values_correct,
                        "Negative values should be filtered with AND logic")
        self.assertNotIn(-10, filtered_values_correct,
                        "Negative values should be filtered with AND logic")
        self.assertNotIn(-1, filtered_values_correct,
                        "Negative values should be filtered with AND logic")
        self.assertNotIn(0, filtered_values_correct,
                        "Zero should be filtered with AND logic")
        
        # The correct filter removes values > 550
        self.assertNotIn(551, filtered_values_correct,
                        "Values > 550 should be filtered with AND logic")
        self.assertNotIn(600, filtered_values_correct,
                        "Values > 550 should be filtered with AND logic")
        self.assertNotIn(1000, filtered_values_correct,
                        "Values > 550 should be filtered with AND logic")
        
        # The correct filter keeps only valid values
        self.assertIn(1, filtered_values_correct,
                     "Values in range (0, 550] should be kept")
        self.assertIn(50, filtered_values_correct,
                     "Values in range (0, 550] should be kept")
        self.assertIn(550, filtered_values_correct,
                     "Values in range (0, 550] should be kept")
        
        # With the correct AND filter, we should keep exactly 3 values: 1, 50, 550
        self.assertEqual(len(df_filtered_correct), 3,
                        "AND filter should keep only values where 0 < SMPEP2 <= 550")
        
    def test_filter_comparison(self):
        """Compare buggy vs correct filter to show the difference."""
        df_cleaned = self.test_data.copy()
        
        # Buggy OR filter
        SMPEP2_out_buggy = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
        df_buggy = df_cleaned[SMPEP2_out_buggy]
        
        # Correct AND filter
        SMPEP2_out_correct = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
        df_correct = df_cleaned[SMPEP2_out_correct]
        
        # The buggy filter should keep many more rows
        self.assertGreater(len(df_buggy), len(df_correct),
                          "Buggy OR filter keeps more rows than correct AND filter")
        
        # Calculate how many outliers are incorrectly kept
        incorrectly_kept = len(df_buggy) - len(df_correct)
        self.assertGreater(incorrectly_kept, 0,
                          "Buggy filter incorrectly keeps outliers")
        
        print(f"\nFilter Comparison:")
        print(f"  Original data: {len(self.test_data)} rows")
        print(f"  Buggy OR filter keeps: {len(df_buggy)} rows (should filter outliers)")
        print(f"  Correct AND filter keeps: {len(df_correct)} rows (correctly filtered)")
        print(f"  Outliers incorrectly kept by buggy filter: {incorrectly_kept}")


def run_demonstration():
    """Run a visual demonstration of the bug."""
    print("=" * 80)
    print("OUTLIER FILTER BUG DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    df = pd.DataFrame({
        'SMPEP2': [-100, -10, -1, 0, 1, 50, 100, 500, 550, 551, 600, 1000]
    })
    
    print("\nOriginal data (SMPEP2 values):")
    print(df['SMPEP2'].tolist())
    
    # Buggy OR filter
    print("\n" + "-" * 80)
    print("BUGGY VERSION (using OR |):")
    print("  SMPEP2_out = (df['SMPEP2'] > 0) | (df['SMPEP2'] <= 550)")
    SMPEP2_out_buggy = (df['SMPEP2'] > 0) | (df['SMPEP2'] <= 550)
    df_buggy = df[SMPEP2_out_buggy]
    print(f"\nFiltered data: {df_buggy['SMPEP2'].tolist()}")
    print(f"Kept {len(df_buggy)} out of {len(df)} rows")
    print("❌ BUG: Negative values and values > 550 are kept!")
    
    # Correct AND filter
    print("\n" + "-" * 80)
    print("CORRECT VERSION (using AND &):")
    print("  SMPEP2_out = (df['SMPEP2'] > 0) & (df['SMPEP2'] <= 550)")
    SMPEP2_out_correct = (df['SMPEP2'] > 0) & (df['SMPEP2'] <= 550)
    df_correct = df[SMPEP2_out_correct]
    print(f"\nFiltered data: {df_correct['SMPEP2'].tolist()}")
    print(f"Kept {len(df_correct)} out of {len(df)} rows")
    print("✅ CORRECT: Only values where 0 < SMPEP2 <= 550 are kept")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run visual demonstration
    run_demonstration()
    
    # Run unit tests
    print("\n")
    unittest.main(verbosity=2)
