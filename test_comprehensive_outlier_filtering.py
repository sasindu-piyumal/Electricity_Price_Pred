import unittest
import pandas as pd
import numpy as np

def filter_outliers_buggy(df_cleaned):
    """Original buggy implementation using OR"""
    SMPEP2_out = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
    return df_cleaned[SMPEP2_out]

def filter_outliers_fixed(df_cleaned):
    """Fixed implementation using AND"""
    SMPEP2_out = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
    return df_cleaned[SMPEP2_out]

class TestComprehensiveOutlierFiltering(unittest.TestCase):
    
    def setUp(self):
        """Create test dataset with various values including outliers"""
        self.test_data = pd.DataFrame({
            'SMPEP2': [-200, -100, -50, -10, 0, 10, 50, 100, 200, 300, 400, 500, 550, 551, 600, 700, 800, 1000, 2000],
            'OtherColumn': range(19)  # Additional column to verify entire row filtering
        })
        
    def test_buggy_implementation_includes_all_values(self):
        """Test that the buggy OR implementation incorrectly includes all values"""
        result = filter_outliers_buggy(self.test_data)
        
        # Bug: Using OR means every value passes the filter
        # Because every number is either > 0 OR <= 550
        self.assertEqual(len(result), len(self.test_data),
                        "Buggy implementation should include ALL values due to OR logic")
        
        # Verify that even extreme outliers are included
        self.assertIn(-200, result['SMPEP2'].values)
        self.assertIn(2000, result['SMPEP2'].values)
        
    def test_fixed_implementation_filters_correctly(self):
        """Test that the fixed AND implementation correctly filters outliers"""
        result = filter_outliers_fixed(self.test_data)
        
        # Expected valid values: those that are > 0 AND <= 550
        expected_values = [10, 50, 100, 200, 300, 400, 500, 550]
        
        self.assertEqual(len(result), len(expected_values),
                        f"Fixed implementation should only include {len(expected_values)} values within range")
        
        # Verify the exact values
        self.assertListEqual(sorted(result['SMPEP2'].tolist()), expected_values,
                           "Fixed implementation should only include values > 0 and <= 550")
        
        # Verify outliers are excluded
        outliers = [-200, -100, -50, -10, 0, 551, 600, 700, 800, 1000, 2000]
        for outlier in outliers:
            self.assertNotIn(outlier, result['SMPEP2'].values,
                           f"Outlier {outlier} should be excluded")
    
    def test_edge_cases(self):
        """Test edge cases for the filtering logic"""
        edge_cases_df = pd.DataFrame({
            'SMPEP2': [0, 0.001, 0.1, 1, 549, 549.9, 550, 550.001, 550.1, 551]
        })
        
        result = filter_outliers_fixed(edge_cases_df)
        
        # Values that should be included: 0.001, 0.1, 1, 549, 549.9, 550
        expected_count = 6
        self.assertEqual(len(result), expected_count,
                        f"Should include exactly {expected_count} values within bounds")
        
        # Check specific edge cases
        self.assertNotIn(0, result['SMPEP2'].values, "0 should be excluded (not > 0)")
        self.assertIn(0.001, result['SMPEP2'].values, "0.001 should be included")
        self.assertIn(550, result['SMPEP2'].values, "550 should be included (= 550)")
        self.assertNotIn(550.001, result['SMPEP2'].values, "550.001 should be excluded (> 550)")
        
    def test_preserves_other_columns(self):
        """Test that filtering preserves other columns in the DataFrame"""
        result = filter_outliers_fixed(self.test_data)
        
        # Check that other columns are preserved
        self.assertIn('OtherColumn', result.columns,
                     "Other columns should be preserved during filtering")
        
        # Verify that rows are filtered as complete units
        # For SMPEP2 value of 100, OtherColumn should be 7 (its index)
        row_with_100 = result[result['SMPEP2'] == 100]
        self.assertEqual(row_with_100['OtherColumn'].iloc[0], 7,
                        "Other column values should match the filtered rows")

if __name__ == '__main__':
    unittest.main()
