#!/usr/bin/env python
# coding: utf-8

"""
Utility script to analyze saved hyperparameter tuning results.
"""

import joblib
import pandas as pd
import numpy as np

def load_and_analyze_results(filename='tuning_results.joblib'):
    """
    Load and display analysis of hyperparameter tuning results.
    """
    print("="*80)
    print("HYPERPARAMETER TUNING RESULTS ANALYSIS")
    print("="*80)
    
    try:
        # Load results using joblib (secure alternative to pickle)
        results = joblib.load(filename)
        
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
        
        for metric in ['r2', 'mae', 'mse', 'rmse']:
            baseline_val = baseline_metrics[metric]
            optimized_val = optimized_metrics[metric]
            if metric == 'r2':
                change = ((optimized_val - baseline_val) / baseline_val) * 100
                print(f"  {metric.upper():<10} {baseline_val:>12.4f} {optimized_val:>12.4f} {change:>11.1f}%")
            else:
                change = ((baseline_val - optimized_val) / baseline_val) * 100
                print(f"  {metric.upper():<10} {baseline_val:>12.4f} {optimized_val:>12.4f} {change:>11.1f}%")
        
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
