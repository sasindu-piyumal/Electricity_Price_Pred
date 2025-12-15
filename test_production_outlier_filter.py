#!/usr/bin/env python
# coding: utf-8

"""
Unit test for the production outlier filter in hyperparameter_tuning.py

This test will FAIL with the buggy code (using |) and PASS with the fixed code (using &).
"""

import pandas as pd
import numpy as np
import unittest
import sys
import warnings
warnings.filterwarnings('ignore')

# Import the function from the production code
from hyperparameter_tuning import load_and_preprocess_data


class TestProductionOutlierFilter(unittest.TestCase):
    """Test the actual production outlier filtering function."""
    
    def test_outlier_filter_removes_negatives_and_large_values(self):
        """
        Test that the production code correctly filters out outliers.
        
        This test will:
        - FAIL if the code uses OR (|) - because outliers won't be filtered
        - PASS if the code uses AND (&) - because outliers will be filtered
        """
        # Create a temporary test CSV with known outliers
        import os
        
        # Save original electricity.csv if it exists
        original_exists = os.path.exists('electricity.csv')
        if original_exists:
            os.rename('electricity.csv', 'electricity.csv.backup')
        
        try:
            # Create test data with clear outliers
            test_data = pd.DataFrame({
                'DateTime': pd.date_range('2020-01-01', periods=100, freq='H'),
                'HolidayFlag': [0] * 100,
                'DayOfWeek': [1] * 100,
                'WeekOfYear': [1] * 100,
                'Day': [1] * 100,
                'Month': [1] * 100,
                'Year': [2020] * 100,
                'PeriodOfDay': [1] * 100,
                'ForecastWindProduction': [100] * 100,
                'SystemLoadEA': [1000] * 100,
                'SMPEA': [50] * 100,
                'ORKTemperature': [10] * 100,
                'ORKWindspeed': [5] * 100,
                'CO2Intensity': [400] * 100,
                'ActualWindProduction': [100] * 100,
                'SystemLoadEP2': [1000] * 100,
                'SMPEP2': [50] * 100,  # Start with all valid values
                'Holiday': ['None'] * 100
            })
            
            # Add outliers that should be filtered
            # Negative values (should be removed)
            test_data.loc[10:15, 'SMPEP2'] = -50
            # Large values > 550 (should be removed)
            test_data.loc[20:25, 'SMPEP2'] = 600
            # Zero (should be removed)
            test_data.loc[30:32, 'SMPEP2'] = 0
            
            # Valid boundary values (should be kept)
            test_data.loc[40, 'SMPEP2'] = 1    # Just above 0
            test_data.loc[41, 'SMPEP2'] = 550  # At upper boundary
            
            test_data.set_index('DateTime', inplace=True)
            
            # Save test data
            test_data.to_csv('electricity.csv')
            
            # Run the production preprocessing function
            df_result = load_and_preprocess_data()
            
            # Check that outliers were removed
            result_values = df_result.index.to_series().reset_index(drop=True)
            
            # Count outliers in result
            # Note: We need to check the index of the cleaned dataframe
            # After preprocessing, SMPEP2 should not contain outliers
            
            # Since the function returns df_new which has SMPEP2 column,
            # we can check if outliers were properly filtered
            
            # The original data had:
            # - 100 rows total
            # - 6 rows with negative values (10:15 = 6 rows)
            # - 6 rows with values > 550 (20:25 = 6 rows)
            # - 3 rows with zero (30:32 = 3 rows)
            # - Rest (85 rows) with valid values
            
            # After proper filtering with AND, we should have ~85 rows
            # With buggy OR filter, we would have ~97-100 rows (almost all kept)
            
            # Let's be more specific: check that we filtered out at least 10 rows
            self.assertLess(len(df_result), 90,
                           "Outlier filter should remove negative values, zeros, and values > 550. "
                           "If this fails, the filter is using OR (|) instead of AND (&).")
            
            # Also verify we kept a reasonable amount (not too aggressive)
            self.assertGreater(len(df_result), 70,
                              "Outlier filter should keep valid values in range (0, 550].")
            
        finally:
            # Clean up: restore original electricity.csv
            if os.path.exists('electricity.csv'):
                os.remove('electricity.csv')
            if original_exists:
                os.rename('electricity.csv.backup', 'electricity.csv')
    
    def test_outlier_filter_logic_directly(self):
        """
        Test the outlier filtering logic directly with a simple dataset.
        
        This test creates a minimal test case to verify the exact filtering behavior.
        """
        # Create test data
        df_test = pd.DataFrame({
            'SMPEP2': [-100, -10, 0, 1, 50, 100, 550, 551, 600, 1000]
        })
        
        # Apply the CORRECT filter logic (what the code should do)
        SMPEP2_out_correct = (df_test['SMPEP2'] > 0) & (df_test['SMPEP2'] <= 550)
        df_correct = df_test[SMPEP2_out_correct]
        
        # The correct filter should keep exactly: [1, 50, 100, 550]
        expected_kept = [1, 50, 100, 550]
        self.assertEqual(len(df_correct), 4,
                        "Correct AND filter should keep 4 values")
        self.assertListEqual(sorted(df_correct['SMPEP2'].tolist()), expected_kept,
                            "Correct AND filter should keep only values in range (0, 550]")
        
        # Apply the BUGGY filter logic (what the code had)
        SMPEP2_out_buggy = (df_test['SMPEP2'] > 0) | (df_test['SMPEP2'] <= 550)
        df_buggy = df_test[SMPEP2_out_buggy]
        
        # The buggy filter would keep: [-100, -10, 0, 1, 50, 100, 550, 551, 600, 1000]
        # (everything except maybe nothing, since OR is almost always true)
        self.assertGreaterEqual(len(df_buggy), 9,
                               "Buggy OR filter keeps almost all values (this demonstrates the bug)")
        
        # The key test: they should be different
        self.assertNotEqual(len(df_correct), len(df_buggy),
                           "Correct AND filter and buggy OR filter produce different results")
        
        print(f"\nDirect Logic Test:")
        print(f"  Buggy OR filter keeps: {len(df_buggy)} values")
        print(f"  Correct AND filter keeps: {len(df_correct)} values")
        print(f"  Difference: {len(df_buggy) - len(df_correct)} outliers incorrectly kept by buggy filter")


def run_simple_demonstration():
    """Run a simple demonstration without full data loading."""
    print("=" * 80)
    print("PRODUCTION CODE OUTLIER FILTER TEST")
    print("=" * 80)
    print("\nThis test verifies that the production code in hyperparameter_tuning.py")
    print("correctly filters outliers using AND (&) instead of OR (|).")
    print("\nTest Data: [-100, -10, 0, 1, 50, 100, 550, 551, 600, 1000]")
    
    df = pd.DataFrame({'SMPEP2': [-100, -10, 0, 1, 50, 100, 550, 551, 600, 1000]})
    
    # Buggy version
    buggy = df[(df['SMPEP2'] > 0) | (df['SMPEP2'] <= 550)]
    print(f"\n❌ BUGGY (OR): Keeps {len(buggy)} values: {buggy['SMPEP2'].tolist()}")
    
    # Correct version
    correct = df[(df['SMPEP2'] > 0) & (df['SMPEP2'] <= 550)]
    print(f"✅ CORRECT (AND): Keeps {len(correct)} values: {correct['SMPEP2'].tolist()}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstration
    run_simple_demonstration()
    
    # Run unit tests
    print("\n")
    unittest.main(verbosity=2)
