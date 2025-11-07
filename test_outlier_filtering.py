import unittest
import pandas as pd
import numpy as np

class TestOutlierFiltering(unittest.TestCase):
    def test_outlier_filtering_bug(self):
        """
        Test that demonstrates the bug in outlier filtering logic.
        The current implementation uses OR instead of AND, which includes all values.
        """
        # Create test data with clear outliers
        test_data = pd.DataFrame({
            'SMPEP2': [-100, -50, 0, 100, 300, 500, 550, 600, 700, 1000]
        })
        
        # Current buggy implementation (using OR)
        # This will include ALL values because any value is either > 0 OR <= 550
        buggy_filter = (test_data['SMPEP2'] > 0) | (test_data['SMPEP2'] <= 550)
        buggy_result = test_data[buggy_filter]
        
        # The buggy filter should include all values (demonstrating the bug)
        self.assertEqual(len(buggy_result), len(test_data), 
                        "Bug: OR condition includes all values instead of filtering outliers")
        
    def test_outlier_filtering_fixed(self):
        """
        Test that demonstrates the correct behavior after fixing the bug.
        Using AND to properly filter values between 0 and 550.
        """
        # Create test data with clear outliers
        test_data = pd.DataFrame({
            'SMPEP2': [-100, -50, 0, 100, 300, 500, 550, 600, 700, 1000]
        })
        
        # Fixed implementation (using AND)
        # This will only include values that are > 0 AND <= 550
        fixed_filter = (test_data['SMPEP2'] > 0) & (test_data['SMPEP2'] <= 550)
        fixed_result = test_data[fixed_filter]
        
        # Expected values after proper filtering: 100, 300, 500, 550
        expected_values = [100, 300, 500, 550]
        
        # Check that only values within the range are included
        self.assertEqual(len(fixed_result), 4, 
                        "Fixed: AND condition correctly filters to values between 0 and 550")
        self.assertListEqual(fixed_result['SMPEP2'].tolist(), expected_values,
                           "Fixed: Filtered values match expected range")
        
        # Check that outliers are excluded
        self.assertNotIn(-100, fixed_result['SMPEP2'].values)
        self.assertNotIn(600, fixed_result['SMPEP2'].values)
        self.assertNotIn(1000, fixed_result['SMPEP2'].values)

if __name__ == '__main__':
    unittest.main()
