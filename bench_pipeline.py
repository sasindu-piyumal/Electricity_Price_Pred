#!/usr/bin/env python
# coding: utf-8

"""
Benchmark script for electricity price prediction pipeline.
Times key preprocessing and training steps, and reports metrics to artemis_results.json.
"""

import json
import time
import numpy as np
from statistics import median

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from hyperparameter_tuning import load_and_preprocess_data, prepare_training_data


def benchmark_pipeline():
    """
    Benchmark the data preprocessing and model training pipeline.
    Reports timing and performance metrics in artemis_results.json.
    """
    
    # Number of repeats for timing measurements
    n_repeats = 3
    
    preprocess_times = []
    feature_prep_times = []
    rf_fit_times = []
    r2_scores = []
    
    print("Starting pipeline benchmark...")
    print(f"Number of repeats: {n_repeats}\n")
    
    for repeat_idx in range(n_repeats):
        print(f"Repeat {repeat_idx + 1}/{n_repeats}")
        
        # Benchmark: Load and preprocess data
        print("  - Loading and preprocessing data...", end=" ", flush=True)
        start = time.time()
        df = load_and_preprocess_data()
        preprocess_time = time.time() - start
        preprocess_times.append(preprocess_time)
        print(f"{preprocess_time:.3f}s")
        
        # Benchmark: Prepare training data (cyclic features, split, scale)
        print("  - Preparing training data...", end=" ", flush=True)
        start = time.time()
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_training_data(df)
        feature_prep_time = time.time() - start
        feature_prep_times.append(feature_prep_time)
        print(f"{feature_prep_time:.3f}s")
        
        # Benchmark: Fit Random Forest (n_estimators=50, lightweight config)
        print("  - Training RandomForest...", end=" ", flush=True)
        start = time.time()
        rf = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=1  # No parallelization to isolate this component
        )
        rf.fit(X_train_scaled, y_train)
        rf_fit_time = time.time() - start
        rf_fit_times.append(rf_fit_time)
        print(f"{rf_fit_time:.3f}s")
        
        # Evaluate: Compute R² on test set as a regression guard
        y_pred = rf.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        r2_scores.append(test_r2)
        print(f"  - Test R² Score: {test_r2:.4f}\n")
    
    # Use median of repeats for final metrics
    preprocess_time_s = median(preprocess_times)
    feature_prep_time_s = median(feature_prep_times)
    rf_fit_time_s = median(rf_fit_times)
    test_r2 = median(r2_scores)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS (median of repeats)")
    print("="*60)
    print(f"Preprocessing time:    {preprocess_time_s:.3f}s")
    print(f"Feature prep time:     {feature_prep_time_s:.3f}s")
    print(f"RandomForest fit time: {rf_fit_time_s:.3f}s")
    print(f"Test R² Score:         {test_r2:.4f}")
    print("="*60 + "\n")
    
    # Write results in artemis_results.json format (custom metrics)
    results = {
        "preprocess_time_s": preprocess_time_s,
        "feature_prep_time_s": feature_prep_time_s,
        "rf_fit_time_s": rf_fit_time_s,
        "test_r2": test_r2
    }
    
    with open("artemis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results written to artemis_results.json")


if __name__ == "__main__":
    benchmark_pipeline()
