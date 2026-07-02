#!/usr/bin/env python
# coding: utf-8

"""
Hyperparameter Tuning for Random Forest Model
Electricity Price Prediction

This script implements comprehensive hyperparameter tuning to improve
the Random Forest model performance beyond the baseline R² of 0.6502.
"""

import pandas as pd
import numpy as np
import warnings
import time
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, 
    RandomizedSearchCV, 
    GridSearchCV,
    TimeSeriesSplit
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
# Keep the second-stage GridSearchCV bounded so it cannot dominate runtime.
MAX_REFINED_GRID_COMBINATIONS = 108
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("HYPERPARAMETER TUNING FOR RANDOM FOREST - ELECTRICITY PRICE PREDICTION")
print("=" * 80)
print(f"Start Time: {datetime.now()}")
print("=" * 80)

# Data Loading and Preprocessing Functions
def load_and_preprocess_data():
    """
    Load and preprocess the electricity dataset following the same steps
    as in the original implementation.
    """
    print("\n1. Loading Data...")
    
    # Load data
    data = pd.read_csv("electricity.csv", index_col=0, parse_dates=[0])
    df = pd.DataFrame(data)
    
    print(f"   - Dataset shape: {df.shape}")
    
    # Convert columns to numeric
    print("\n2. Converting columns to numeric...")
    cols_to_numeric = ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 
                      'ORKTemperature', 'ORKWindspeed', 'CO2Intensity', 
                      'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    print("\n3. Handling missing values...")
    df.replace(['', 'NA', 'N/A', None], np.nan, inplace=True)
    
    # Remove rows with missing critical values
    df_cleaned = df.dropna(subset=['ORKTemperature','ORKWindspeed'])
    
    # Remove outliers from SMPEP2
    print("\n4. Removing outliers...")
    # Fixed: Changed OR (|) to AND (&) to properly filter values between 0 and 550
    SMPEP2_out = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
    df_cleaned = df_cleaned[SMPEP2_out]
    
    # Fill missing values with median for skewed distributions
    fill_with_median = ['ForecastWindProduction','SystemLoadEA','SMPEA',
                       'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
    for col in fill_with_median:
        median_col = df_cleaned[col].median()
        df_cleaned[col].fillna(median_col, inplace=True)
    
    # Fill CO2Intensity with mean (normal distribution)
    mean_CO2Intensity = df_cleaned['CO2Intensity'].mean()
    df_cleaned['CO2Intensity'].fillna(mean_CO2Intensity, inplace=True)
    
    # Drop highly correlated features
    print("\n5. Dropping highly correlated features...")
    df_new = df_cleaned.drop(columns=['Holiday','WeekOfYear','ForecastWindProduction','SystemLoadEA'])
    
    print(f"   - Cleaned dataset shape: {df_new.shape}")
    
    return df_new

def add_cyclic_features(df_scaled):
    """
    Add sine and cosine transformations for cyclic features.
    """
    print("\n6. Adding cyclic features...")
    
    # Define periodic transform function
    def periodic_transform(df, variable):
        df[f"{variable}_SIN"] = np.sin(df[variable] / df[variable].max() * 2 * np.pi)
        df[f"{variable}_COS"] = np.cos(df[variable] / df[variable].max() * 2 * np.pi)
        return df
    
    # Apply transformations
    df_scaled = periodic_transform(df_scaled, 'DayOfWeek')
    df_scaled = periodic_transform(df_scaled, 'Day')
    df_scaled = periodic_transform(df_scaled, 'Month')
    df_scaled = periodic_transform(df_scaled, 'PeriodOfDay')
    
    # Drop original cyclic columns
    df_scaled = df_scaled.drop(columns=['DayOfWeek', 'Day', 'Month', 'PeriodOfDay'])
    
    return df_scaled

def prepare_training_data(df_new):
    """
    Prepare data for model training with proper train-test split and scaling.
    """
    print("\n7. Preparing training data...")
    
    # Reset index to access DateTime
    df_scaled = df_new.reset_index()
    
    # Add cyclic features
    df_scaled = add_cyclic_features(df_scaled)
    
    # Drop DateTime column
    df_scaled = df_scaled.drop(columns=['DateTime'])
    
    # Split features and target
    X = df_scaled.drop(columns='SMPEP2', axis=1)
    y = df_scaled['SMPEP2']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=False
    )
    
    # Scale features
    print("\n8. Scaling features...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   - Training set shape: {X_train_scaled.shape}")
    print(f"   - Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Cross-validation and Parameter Search Space
def setup_cross_validation(n_splits=5):
    """
    Setup TimeSeriesSplit for proper time series cross-validation.
    """
    print(f"\n9. Setting up TimeSeriesSplit cross-validation with {n_splits} splits...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv

def define_parameter_space():
    """
    Define the comprehensive parameter search space as specified.
    """
    print("\n10. Defining parameter search space...")
    
    param_space = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.8],
        'bootstrap': [True],  # Default for Random Forest
        'random_state': [RANDOM_STATE]
    }
    
    # Calculate total combinations
    total_combinations = 1
    for param, values in param_space.items():
        if param not in ['bootstrap', 'random_state']:
            total_combinations *= len(values)
    
    print(f"   - Total parameter combinations: {total_combinations}")
    print(f"   - Parameters to tune: {list(param_space.keys())}")
    
    return param_space

# Hyperparameter Tuning Functions
def perform_randomized_search(X_train, y_train, param_space, cv, n_iter=100):
    """
    Perform initial broad search using RandomizedSearchCV.
    """
    print(f"\n11. Performing RandomizedSearchCV with {n_iter} iterations...")
    print("    This may take some time...")
    
    # Create base model. Let RandomizedSearchCV parallelize candidates/folds;
    # keeping each forest single-threaded avoids CPU oversubscription.
    rf_base = RandomForestRegressor(n_jobs=1)
    
    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_space,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    # Track time
    start_time = time.time()
    
    # Fit the random search
    random_search.fit(X_train, y_train)
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    
    print(f"\n    RandomizedSearchCV completed in {elapsed_time:.2f} minutes")
    print(f"    Best R² score: {random_search.best_score_:.4f}")
    print(f"    Best parameters: {random_search.best_params_}")
    
    return random_search, elapsed_time

def create_refined_param_grid(best_params):
    """
    Create a compact refined parameter grid around RandomizedSearchCV results.

    The original second-stage grid could expand to hundreds of candidates and
    dominate runtime. This keeps local refinement for every tuned parameter but
    caps the Cartesian product to a predictable size (typically <= 108 configs).
    """
    print("\n12. Creating compact refined parameter grid for GridSearchCV...")
    
    def unique_sorted(values):
        """Return sorted unique values while preserving None at the end."""
        non_none = sorted({value for value in values if value is not None})
        return non_none + ([None] if any(value is None for value in values) else [])
    
    refined_grid = {}
    
    # n_estimators is the most expensive dimension; compare the winner with one
    # nearby cheaper option rather than expanding to multiple larger forests.
    best_n_estimators = best_params['n_estimators']
    refined_grid['n_estimators'] = unique_sorted([
        max(50, best_n_estimators - 100),
        best_n_estimators
    ])
    
    # Refine depth locally. If the random search selected unlimited depth, test
    # it only against one strong bounded alternative to avoid very deep trees
    # across a large grid.
    if best_params['max_depth'] is not None:
        best_max_depth = best_params['max_depth']
        refined_grid['max_depth'] = unique_sorted([
            max(5, best_max_depth - 5),
            best_max_depth,
            best_max_depth + 5
        ])
    else:
        refined_grid['max_depth'] = [50, None]
    
    # Keep a small local neighbourhood for split/leaf regularisation.
    best_min_samples_split = best_params['min_samples_split']
    refined_grid['min_samples_split'] = unique_sorted([
        max(2, best_min_samples_split - 5),
        best_min_samples_split,
        best_min_samples_split + 5
    ])
    
    best_min_samples_leaf = best_params['min_samples_leaf']
    refined_grid['min_samples_leaf'] = unique_sorted([
        max(1, best_min_samples_leaf - 2),
        best_min_samples_leaf,
        best_min_samples_leaf + 2
    ])
    
    # max_features affects both quality and split cost. Keep the random-search
    # winner and one adjacent alternative instead of four broad options.
    best_max_features = best_params['max_features']
    if isinstance(best_max_features, str):
        refined_grid['max_features'] = unique_sorted([
            best_max_features,
            'sqrt' if best_max_features == 'log2' else 'log2'
        ])
    else:
        refined_grid['max_features'] = unique_sorted([
            best_max_features,
            max(0.1, best_max_features - 0.1)
        ])
    
    # Keep fixed parameters
    refined_grid['bootstrap'] = [True]
    refined_grid['random_state'] = [RANDOM_STATE]
    
    # Calculate refined combinations
    refined_combinations = 1
    for param, values in refined_grid.items():
        if param not in ['bootstrap', 'random_state']:
            refined_combinations *= len(values)
    
    if refined_combinations > MAX_REFINED_GRID_COMBINATIONS:
        raise ValueError(
            "Refined grid has "
            f"{refined_combinations} combinations, exceeding the configured "
            f"budget of {MAX_REFINED_GRID_COMBINATIONS}. Tighten the local "
            "parameter neighbourhood before running GridSearchCV."
        )
    
    print(
        f"    Refined grid combinations: {refined_combinations} "
        f"(budget: {MAX_REFINED_GRID_COMBINATIONS})"
    )
    print(f"    Refined parameters: {refined_grid}")
    
    return refined_grid

def perform_grid_search(X_train, y_train, refined_grid, cv):
    """
    Perform fine-tuning using GridSearchCV with refined parameters.
    """
    print(f"\n13. Performing GridSearchCV for fine-tuning...")
    print("    This may take some time...")
    
    # Create base model. Let GridSearchCV parallelize candidates/folds;
    # keeping each forest single-threaded avoids CPU oversubscription.
    rf_base = RandomForestRegressor(n_jobs=1)
    
    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=refined_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    # Track time
    start_time = time.time()
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    
    print(f"\n    GridSearchCV completed in {elapsed_time:.2f} minutes")
    print(f"    Best R² score: {grid_search.best_score_:.4f}")
    print(f"    Best parameters: {grid_search.best_params_}")
    
    return grid_search, elapsed_time

# Performance Evaluation and Analysis
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance with multiple metrics.
    """
    print(f"\n14. Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"    R² Score: {r2:.4f}")
    print(f"    MAE: {mae:.4f}")
    print(f"    MSE: {mse:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    
    return {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'predictions': y_pred
    }

def analyze_feature_importance(best_model, feature_names):
    """
    Analyze and visualize feature importance from the best model.
    """
    print("\n15. Analyzing feature importance...")
    
    # Get feature importances
    importances = best_model.feature_importances_
    
    # Create DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'][:15], feature_importance_df['importance'][:15])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("    Top 5 most important features:")
    for idx, row in feature_importance_df.head(5).iterrows():
        print(f"    - {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance_df

def analyze_hyperparameter_impact(search_results):
    """
    Analyze the impact of different hyperparameters on model performance.
    """
    print("\n16. Analyzing hyperparameter impact...")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(search_results.cv_results_)
    
    # Find parameters with highest variance in scores
    param_impacts = {}
    for param in ['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 
                  'param_min_samples_leaf', 'param_max_features']:
        if param in results_df.columns:
            # Group by parameter value and calculate mean score
            grouped = results_df.groupby(param)['mean_test_score'].agg(['mean', 'std'])
            param_impacts[param] = {
                'range': grouped['mean'].max() - grouped['mean'].min(),
                'best_value': grouped['mean'].idxmax(),
                'values': grouped
            }
    
    # Sort by impact (range)
    sorted_impacts = sorted(param_impacts.items(), 
                           key=lambda x: x[1]['range'], 
                           reverse=True)
    
    print("    Hyperparameter impact (by score range):")
    for param, impact in sorted_impacts[:5]:
        param_name = param.replace('param_', '')
        print(f"    - {param_name}: range={impact['range']:.4f}, best={impact['best_value']}")
    
    return param_impacts, results_df

def save_results(results_dict, filename='tuning_results.joblib'):
    """
    Save all tuning results using joblib (secure alternative to pickle).
    """
    print(f"\n17. Saving results to {filename}...")
    
    joblib.dump(results_dict, filename)
    
    print(f"    Results saved successfully!")

def create_comparison_plot(baseline_r2, final_r2, y_test, y_pred):
    """
    Create visualization comparing baseline vs optimized model.
    """
    print("\n18. Creating comparison visualizations...")
    
    # Performance comparison bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # R² comparison
    models = ['Baseline RF', 'Optimized RF']
    scores = [baseline_r2, final_r2]
    colors = ['#ff7f0e', '#2ca02c'] if final_r2 > baseline_r2 else ['#2ca02c', '#ff7f0e']
    
    bars = ax1.bar(models, scores, color=colors, alpha=0.8)
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom')
    
    # Add improvement percentage
    improvement = ((final_r2 - baseline_r2) / baseline_r2) * 100
    ax1.text(0.5, 0.5, f'Improvement: {improvement:.1f}%', 
            transform=ax1.transAxes, ha='center', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Predictions vs Actual scatter plot
    ax2.scatter(y_test[:1000], y_pred[:1000], alpha=0.5, s=10)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect prediction')
    ax2.set_xlabel('Actual SMPEP2')
    ax2.set_ylabel('Predicted SMPEP2')
    ax2.set_title('Predictions vs Actual (First 1000 samples)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Improvement over baseline: {improvement:.1f}%")

# Main Execution
def main():
    """
    Main execution function for hyperparameter tuning.
    """
    try:
        # Track overall time
        overall_start = time.time()
        
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Prepare training data
        X_train, X_test, y_train, y_test, scaler = prepare_training_data(df)
        
        # Get feature names
        feature_names = df.drop(columns=['SMPEP2']).columns.tolist()
        if 'DateTime' in feature_names:
            feature_names.remove('DateTime')
        # Add cyclic feature names
        for col in ['DayOfWeek', 'Day', 'Month', 'PeriodOfDay']:
            feature_names.extend([f"{col}_SIN", f"{col}_COS"])
        # Remove original cyclic columns
        for col in ['DayOfWeek', 'Day', 'Month', 'PeriodOfDay']:
            if col in feature_names:
                feature_names.remove(col)
        
        # Setup cross-validation
        cv = setup_cross_validation(n_splits=5)
        
        # Define parameter space
        param_space = define_parameter_space()
        
        # Baseline model evaluation
        print("\n" + "="*80)
        print("BASELINE MODEL EVALUATION")
        print("="*80)
        baseline_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        baseline_rf.fit(X_train, y_train)
        baseline_results = evaluate_model(baseline_rf, X_test, y_test, "Baseline Random Forest")
        baseline_r2 = baseline_results['r2']
        
        # Perform RandomizedSearchCV
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING - PHASE 1: RANDOMIZED SEARCH")
        print("="*80)
        random_search, random_time = perform_randomized_search(
            X_train, y_train, param_space, cv, n_iter=100
        )
        
        # Create refined parameter grid
        refined_grid = create_refined_param_grid(random_search.best_params_)
        
        # Perform GridSearchCV
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING - PHASE 2: GRID SEARCH")
        print("="*80)
        grid_search, grid_time = perform_grid_search(
            X_train, y_train, refined_grid, cv
        )
        
        # Final evaluation with best model
        print("\n" + "="*80)
        print("FINAL MODEL EVALUATION")
        print("="*80)
        best_model = grid_search.best_estimator_
        final_results = evaluate_model(best_model, X_test, y_test, "Optimized Random Forest")
        final_r2 = final_results['r2']
        
        # Feature importance analysis
        feature_importance_df = analyze_feature_importance(best_model, feature_names)
        
        # Hyperparameter impact analysis
        param_impacts_random, _ = analyze_hyperparameter_impact(random_search)
        param_impacts_grid, _ = analyze_hyperparameter_impact(grid_search)
        
        # Create comparison visualizations
        create_comparison_plot(baseline_r2, final_r2, y_test, final_results['predictions'])
        
        # Calculate total time
        overall_end = time.time()
        total_time = (overall_end - overall_start) / 60
        
        # Prepare results dictionary
        results = {
            'baseline_model': {
                'model': baseline_rf,
                'metrics': baseline_results,
                'r2': baseline_r2
            },
            'random_search': {
                'search_object': random_search,
                'best_params': random_search.best_params_,
                'best_score_cv': random_search.best_score_,
                'time_minutes': random_time,
                'param_impacts': param_impacts_random
            },
            'grid_search': {
                'search_object': grid_search,
                'best_params': grid_search.best_params_,
                'best_score_cv': grid_search.best_score_,
                'time_minutes': grid_time,
                'param_impacts': param_impacts_grid,
                'refined_grid': refined_grid
            },
            'best_model': {
                'model': best_model,
                'metrics': final_results,
                'r2': final_r2,
                'feature_importance': feature_importance_df
            },
            'metadata': {
                'total_time_minutes': total_time,
                'random_state': RANDOM_STATE,
                'cv_splits': 5,
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Save results
        save_results(results)
        
        # Print summary
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*80)
        print(f"Baseline R² Score: {baseline_r2:.4f}")
        print(f"Optimized R² Score: {final_r2:.4f}")
        print(f"Improvement: {((final_r2 - baseline_r2) / baseline_r2) * 100:.1f}%")
        print(f"\nTotal time: {total_time:.2f} minutes")
        print(f"RandomizedSearchCV time: {random_time:.2f} minutes")
        print(f"GridSearchCV time: {grid_time:.2f} minutes")
        
        if final_r2 > baseline_r2:
            print(f"\n✅ SUCCESS: Achieved R² score > {baseline_r2:.4f}")
        else:
            print(f"\n⚠️  WARNING: Did not improve over baseline R² of {baseline_r2:.4f}")
        
        print("\nBest Hyperparameters:")
        for param, value in grid_search.best_params_.items():
            if param not in ['random_state', 'bootstrap']:
                print(f"  - {param}: {value}")
        
        print("\nFiles saved:")
        print("  - tuning_results.joblib (complete results)")
        print("  - feature_importance.png")
        print("  - model_comparison.png")
        
        print("\n" + "="*80)
        print(f"End Time: {datetime.now()}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()
