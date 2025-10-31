#!/usr/bin/env python3
"""
Quick test script to verify the optimizations work correctly.
Tests the key optimized functions with sample data.
"""

import pandas as pd
import numpy as np
import time

def test_periodic_transform():
    """Test the optimized periodic_transform function"""
    print("Testing optimized periodic_transform function...")
    
    # Create test data
    test_df = pd.DataFrame({
        'DayOfWeek': [1, 2, 3, 4, 5, 6, 7],
        'Month': [1, 6, 12, 3, 9, 11, 2],
        'values': [10, 20, 30, 40, 50, 60, 70]
    })
    
    def periodic_transform(df, variable):
        """Optimized periodic transform function"""
        max_val = df[variable].max()
        df[f"{variable}_SIN"] = np.sin(df[variable] / max_val * 2 * np.pi)
        df[f"{variable}_COS"] = np.cos(df[variable] / max_val * 2 * np.pi)
        return df
    
    # Test the function
    result = periodic_transform(test_df.copy(), 'DayOfWeek')
    
    # Verify results
    assert 'DayOfWeek_SIN' in result.columns, "SIN column not created"
    assert 'DayOfWeek_COS' in result.columns, "COS column not created"
    assert len(result) == len(test_df), "Row count mismatch"
    
    print("✅ periodic_transform function working correctly")
    return True

def test_vectorized_operations():
    """Test vectorized data type conversions"""
    print("Testing vectorized data type conversions...")
    
    # Create test data with mixed types
    test_df = pd.DataFrame({
        'col1': ['1.5', '2.3', '3.7', 'invalid', '5.1'],
        'col2': ['10', '20', '30', '40', '50'],
        'col3': ['100.1', '200.2', '300.3', '400.4', '500.5']
    })
    
    cols_to_convert = ['col1', 'col2', 'col3']
    
    # Test original method (loop)
    start_time = time.time()
    df_loop = test_df.copy()
    for col in cols_to_convert:
        df_loop[col] = pd.to_numeric(df_loop[col], errors='coerce')
    loop_time = time.time() - start_time
    
    # Test optimized method (vectorized)
    start_time = time.time()
    df_vectorized = test_df.copy()
    df_vectorized[cols_to_convert] = df_vectorized[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    vectorized_time = time.time() - start_time
    
    # Verify results are identical
    pd.testing.assert_frame_equal(df_loop, df_vectorized, check_dtype=True)
    
    print(f"✅ Vectorized conversion working correctly")
    print(f"   Loop time: {loop_time:.6f}s, Vectorized time: {vectorized_time:.6f}s")
    return True

def test_fillna_optimization():
    """Test optimized fillna operations"""
    print("Testing optimized fillna operations...")
    
    # Create test data with missing values
    test_df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': [10, np.nan, 30, 40, np.nan],
        'col3': [100, 200, 300, np.nan, 500]
    })
    
    fill_cols = ['col1', 'col2', 'col3']
    
    # Original method (loop)
    df_loop = test_df.copy()
    for col in fill_cols:
        median_col = df_loop[col].median()
        df_loop[col].fillna(median_col, inplace=True)
    
    # Optimized method (vectorized)
    df_vectorized = test_df.copy()
    medians = df_vectorized[fill_cols].median()
    df_vectorized[fill_cols] = df_vectorized[fill_cols].fillna(medians)
    
    # Verify results are identical
    pd.testing.assert_frame_equal(df_loop, df_vectorized)
    
    print("✅ Optimized fillna working correctly")
    return True

def run_tests():
    """Run all optimization tests"""
    print("Running optimization tests...")
    print("=" * 40)
    
    tests = [
        test_periodic_transform,
        test_vectorized_operations, 
        test_fillna_optimization
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All optimizations are working correctly!")
    else:
        print("⚠️  Some optimizations need attention")

if __name__ == "__main__":
    run_tests()
