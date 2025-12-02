#!/usr/bin/env python
# coding: utf-8

"""
Utility script to analyze saved hyperparameter tuning results.
"""

import sys
import pickle
import pandas as pd
import numpy as np

def load_and_analyze_results(filename='tuning_results.pkl'):
    """
    Load and display analysis of hyperparameter tuning results.
    """
    print("="*80)
    print("HYPERPARAMETER TUNING RESULTS ANALYSIS")
    print("="*80)
    
    try:
        # Load results
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        
        # Extract key information
        baseline_r2 = results['baseline_model']['r2']
        final_r2 = results['best_model']['r2']
        improvement = ((final_r2 - baseline_r2) / baseline_r2) * 100
        
        print(f"\nPerformance Summary:")
        print(f"  Baseline R² Score: {baseline_r2:.4f}")
        print(f"  Optimized R² Score: {final_r2:.4f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        print(f"\nDetailed Metrics Comparison:")
        print(f"  {'Metric':<10} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
        print(f"  {'-'*48}")
        
        baseline_metrics = results['baseline_model']['metrics']
        optimized_metrics = results['best_model']['metrics']
        
        try:
            # Define required metrics
            required_metrics = ['r2', 'mae', 'mse', 'rmse']
            
            # Validate baseline_metrics
            missing_baseline = [m for m in required_metrics if m not in baseline_metrics]
            if missing_baseline:
                print(f"  Warning: Missing baseline metrics: {missing_baseline}")
                print(f"  Available baseline metrics: {list(baseline_metrics.keys())}")
                # Use only available metrics
                required_metrics = [m for m in required_metrics if m in baseline_metrics]
            
            # Validate optimized_metrics
            missing_optimized = [m for m in required_metrics if m not in optimized_metrics]
            if missing_optimized:
                print(f"  Warning: Missing optimized metrics: {missing_optimized}")
                print(f"  Available optimized metrics: {list(optimized_metrics.keys())}")
                required_metrics = [m for m in required_metrics if m in optimized_metrics and m in baseline_metrics]
            
            # Check if we have any common metrics
            if not required_metrics:
                print("  Error: No common metrics found between baseline and optimized results")
                print(f"  Baseline metrics: {list(baseline_metrics.keys())}")
                print(f"  Optimized metrics: {list(optimized_metrics.keys())}")
                print("  Skipping metrics comparison...")
            else:
                # Use required_metrics list and .get() for safe access
                for metric in required_metrics:
                    baseline_val = baseline_metrics.get(metric, 0)
                    optimized_val = optimized_metrics.get(metric, 0)
                    
                    # Validate numeric types
                    if baseline_val is None or pd.isna(baseline_val):
                        print(f"  Warning: Baseline {metric} is None or NaN, using 0")
                        baseline_val = 0
                    if optimized_val is None or pd.isna(optimized_val):
                        print(f"  Warning: Optimized {metric} is None or NaN, using 0")
                        optimized_val = 0
                    
                    # Calculate and display comparison
                    if metric == 'r2':
                        if baseline_val != 0:
                            change = ((optimized_val - baseline_val) / baseline_val) * 100
                            print(f"  {metric.upper():<10} {baseline_val:>12.4f} {optimized_val:>12.4f} {change:>11.1f}%")
                        else:
                            print(f"  {metric.upper():<10} {baseline_val:>12.4f} {optimized_val:>12.4f} {'N/A':>12}")
                    else:
                        if baseline_val != 0:
                            change = ((baseline_val - optimized_val) / baseline_val) * 100
                            print(f"  {metric.upper():<10} {baseline_val:>12.4f} {optimized_val:>12.4f} {-change:>11.1f}%")
                        else:
                            print(f"  {metric.upper():<10} {baseline_val:>12.4f} {optimized_val:>12.4f} {'N/A':>12}")
        
        except KeyError as e:
            print(f"  Error: Missing metric key: {e}")
            print(f"  Baseline metrics available: {list(baseline_metrics.keys())}")
            print(f"  Optimized metrics available: {list(optimized_metrics.keys())}")
            print("  Skipping metrics comparison...")
        except Exception as e:
            print(f"  Error during metrics comparison: {str(e)}")
            print("  Skipping metrics comparison...")
        
        print(f"\nBest Hyperparameters:")
        best_params = results['grid_search']['best_params']
        for param, value in best_params.items():
            if param not in ['random_state', 'bootstrap']:
                print(f"  {param}: {value}")
        
        print(f"\nTiming Information:")
        print(f"  Total time: {results['metadata']['total_time_minutes']:.2f} minutes")
        print(f"  RandomizedSearchCV: {results['random_search']['time_minutes']:.2f} minutes")
        print(f"  GridSearchCV: {results['grid_search']['time_minutes']:.2f} minutes")
        
        print(f"\nTop 5 Important Features:")
        feature_importance = results['best_model']['feature_importance']
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nModel Configuration:")
        print(f"  Training samples: {results['metadata']['train_shape'][0]}")
        print(f"  Test samples: {results['metadata']['test_shape'][0]}")
        print(f"  Features: {results['metadata']['train_shape'][1]}")
        print(f"  CV Splits: {results['metadata']['cv_splits']}")
        print(f"  Random State: {results['metadata']['random_state']}")
        
        print(f"\nAnalysis completed at: {results['metadata']['timestamp']}")
        
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run hyperparameter_tuning.py first.")
    except Exception as e:
        print(f"Error loading results: {str(e)}")
    
    print("="*80)

if __name__ == "__main__":
    load_and_analyze_results()
