# Electricity Price Prediction - Refactored Version
# This script addresses the following improvements:
# 1. Modularized code structure with reusable functions
# 2. Hyperparameter tuning for all ML models using GridSearchCV
# 3. Fixed random_state for reproducibility
# 4. Eliminated code repetition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


#############################################################################
# MODULAR FUNCTIONS
#############################################################################

def load_data(file_path):
    """Load electricity data from CSV file
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Loaded data with datetime index
    """
    data = pd.read_csv(file_path, index_col=0, parse_dates=[0])
    return pd.DataFrame(data)


def clean_data(df):
    """Clean the electricity data by converting types, handling missing values, and removing outliers
    
    Parameters:
    df (pd.DataFrame): Raw dataframe
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    # Convert columns to numeric
    cols_to_numeric = ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ORKTemperature', 
                       'ORKWindspeed', 'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace various null representations
    df.replace(['', 'NA', 'N/A', None], np.nan, inplace=True)
    
    # Drop rows with missing critical values
    df_cleaned = df.dropna(subset=['ORKTemperature', 'ORKWindspeed'])
    
    # Remove outliers
    SMPEP2_out = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)
    df_cleaned = df_cleaned[SMPEP2_out]
    
    # Fill missing values with median for skewed distributions
    fill_with_median = ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 
                        'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
    for col in fill_with_median:
        median_col = df_cleaned[col].median()
        df_cleaned[col].fillna(median_col, inplace=True)
    
    # Fill CO2Intensity with mean (normal distribution)
    mean_CO2Intensity = df_cleaned['CO2Intensity'].mean()
    df_cleaned['CO2Intensity'].fillna(mean_CO2Intensity, inplace=True)
    
    # Drop highly correlated columns
    df_cleaned = df_cleaned.drop(columns=['Holiday', 'WeekOfYear', 'ForecastWindProduction', 'SystemLoadEA'])
    
    return df_cleaned


def apply_feature_engineering(df):
    """Apply feature engineering including cyclic transformations
    
    Parameters:
    df (pd.DataFrame): Cleaned dataframe
    
    Returns:
    pd.DataFrame: Dataframe with engineered features
    """
    df_eng = df.reset_index()
    
    # Apply periodic transformation for cyclic features
    def periodic_transform(df, variable):
        df[f"{variable}_SIN"] = np.sin(df[variable] / df[variable].max() * 2 * np.pi)
        df[f"{variable}_COS"] = np.cos(df[variable] / df[variable].max() * 2 * np.pi)
        return df
    
    df_eng = periodic_transform(df_eng, 'DayOfWeek')
    df_eng = periodic_transform(df_eng, 'Day')
    df_eng = periodic_transform(df_eng, 'Month')
    df_eng = periodic_transform(df_eng, 'PeriodOfDay')
    
    # Drop original cyclic columns and DateTime
    df_eng = df_eng.drop(columns=['DateTime', 'DayOfWeek', 'Day', 'Month', 'PeriodOfDay'])
    
    return df_eng


def get_hyperparameter_grids():
    """Define hyperparameter grids for each model
    
    Returns:
    dict: Dictionary of model names and their hyperparameter grids
    """
    param_grids = {
        'LinearRegression': {},  # LinearRegression has no hyperparameters to tune
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'random_state': [42]
        },
        'DecisionTreeRegressor': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [42]
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [42]
        },
        'SVR': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }
    return param_grids


def evaluate_models_with_tuning(X_train, X_test, y_train, y_test, cv=3):
    """Evaluate multiple models with hyperparameter tuning
    
    Parameters:
    X_train: Training features
    X_test: Test features
    y_train: Training target
    y_test: Test target
    cv (int): Number of cross-validation folds
    
    Returns:
    dict: Dictionary containing best models, scores, and predictions
    """
    models = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'SVR': SVR()
    }
    
    param_grids = get_hyperparameter_grids()
    
    results = {}
    best_score = -np.inf
    best_model_name = None
    
    for model_name, model in models.items():
        print(f"\nTuning {model_name}...")
        
        if model_name == 'LinearRegression':
            # LinearRegression has no hyperparameters to tune
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            best_estimator = model
        else:
            # Perform GridSearchCV for other models
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=cv, 
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            score = grid_search.score(X_test, y_test)
            best_estimator = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        
        print(f"{model_name} --> R² Score: {score:.4f}")
        
        results[model_name] = {
            'model': best_estimator,
            'score': score,
            'predictions': best_estimator.predict(X_test)
        }
        
        if score > best_score:
            best_score = score
            best_model_name = model_name
    
    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name} with R² Score: {best_score:.4f}")
    print(f"{'='*50}")
    
    results['best_model_name'] = best_model_name
    results['best_score'] = best_score
    
    return results


def visualize_results(y_test, y_pred, title_prefix=""):
    """Visualize predictions vs actual values and print metrics
    
    Parameters:
    y_test: Actual test values
    y_pred: Predicted values
    title_prefix (str): Prefix for plot title
    
    Returns:
    tuple: (final_df, mae, mse, rmse)
    """
    y_pred_flat = y_pred.flatten()
    
    # Handle both Series and numpy arrays
    if isinstance(y_test, pd.Series):
        y_test_values = y_test.values
    else:
        y_test_values = y_test
    
    # Create results dataframe
    final_df = pd.DataFrame(
        np.hstack((y_pred_flat[:, np.newaxis], y_test_values[:, np.newaxis])), 
        columns=['Prediction', 'Real']
    )
    
    # Print metrics
    mae = mean_absolute_error(final_df["Real"], final_df["Prediction"])
    mse = mean_squared_error(final_df["Real"], final_df["Prediction"])
    rmse = np.sqrt(mse)
    
    print(f'\n{title_prefix} Metrics:')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    
    # Plot predictions vs actual
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.lineplot(x=range(len(final_df['Real'])), y=final_df['Real'], color='black', label='Real')
    sns.lineplot(x=range(len(final_df['Prediction'])), y=final_df['Prediction'], color='red', label='Prediction')
    ax.set_xlim([100, 200])
    plt.title(f'{title_prefix} Real vs. Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('SMPEP2')
    plt.legend()
    plt.show()
    
    return final_df, mae, mse, rmse


#############################################################################
# MAIN EXECUTION
#############################################################################

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = load_data("Z:\\Sasindu\\Data set\\electricity.csv")
    print(f"Data shape: {data.shape}")
    
    # Clean data
    print("\nCleaning data...")
    df_cleaned = clean_data(data)
    print(f"Cleaned data shape: {df_cleaned.shape}")
    
    # Apply feature engineering
    print("\nApplying feature engineering...")
    df_engineered = apply_feature_engineering(df_cleaned)
    print(f"Engineered data shape: {df_engineered.shape}")
    
    # Split features and target
    X = df_engineered.drop(columns='SMPEP2', axis=1)
    y = df_engineered['SMPEP2']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Scenario 1: Scaled Data with Hyperparameter Tuning
    print("\n" + "="*80)
    print("SCENARIO 1: SCALED DATA WITH HYPERPARAMETER TUNING")
    print("="*80)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results_scaled = evaluate_models_with_tuning(X_train_scaled, X_test_scaled, y_train, y_test, cv=3)
    
    best_model_scaled = results_scaled[results_scaled['best_model_name']]['model']
    y_pred_scaled = results_scaled[results_scaled['best_model_name']]['predictions']
    
    final_df_scaled, mae_scaled, mse_scaled, rmse_scaled = visualize_results(
        y_test, y_pred_scaled, "Scaled Data"
    )
    
    # Scenario 2: PCA Transformed Data with Hyperparameter Tuning
    print("\n" + "="*80)
    print("SCENARIO 2: PCA TRANSFORMED DATA WITH HYPERPARAMETER TUNING")
    print("="*80)
    
    # PCA analysis
    pca = PCA()
    pca.fit(X_train_scaled)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.show()
    
    # Apply PCA with 11 components
    pca = PCA(n_components=11)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    results_pca = evaluate_models_with_tuning(X_train_pca, X_test_pca, y_train, y_test, cv=3)
    
    best_model_pca = results_pca[results_pca['best_model_name']]['model']
    y_pred_pca = results_pca[results_pca['best_model_name']]['predictions']
    
    final_df_pca, mae_pca, mse_pca, rmse_pca = visualize_results(
        y_test, y_pred_pca, "PCA Data"
    )
    
    # Scenario 3: Unscaled Data with Hyperparameter Tuning
    print("\n" + "="*80)
    print("SCENARIO 3: UNSCALED DATA WITH HYPERPARAMETER TUNING")
    print("="*80)
    
    # Use original cleaned data without feature engineering for comparison
    df_new = df_cleaned.reset_index()
    X_unscaled = df_new.drop(columns=['DateTime', 'SMPEP2'], axis=1)
    Y_unscaled = df_new['SMPEP2']
    
    X_train_unscaled, X_test_unscaled, Y_train_unscaled, Y_test_unscaled = train_test_split(
        X_unscaled, Y_unscaled, test_size=0.2, random_state=42
    )
    
    results_unscaled = evaluate_models_with_tuning(
        X_train_unscaled, X_test_unscaled, Y_train_unscaled, Y_test_unscaled, cv=3
    )
    
    best_model_unscaled = results_unscaled[results_unscaled['best_model_name']]['model']
    y_pred_unscaled = results_unscaled[results_unscaled['best_model_name']]['predictions']
    
    final_dfr, mae_unscaled, mse_unscaled, rmse_unscaled = visualize_results(
        Y_test_unscaled, y_pred_unscaled, "Unscaled Data"
    )
    
    # Final Comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON OF ALL SCENARIOS")
    print("="*80)
    
    print(f"\nScaled Data:")
    print(f"  Best Model: {results_scaled['best_model_name']}")
    print(f"  R² Score: {results_scaled['best_score']:.4f}")
    print(f"  MAE: {mae_scaled:.4f}")
    print(f"  MSE: {mse_scaled:.4f}")
    print(f"  RMSE: {rmse_scaled:.4f}")
    
    print(f"\nPCA Data:")
    print(f"  Best Model: {results_pca['best_model_name']}")
    print(f"  R² Score: {results_pca['best_score']:.4f}")
    print(f"  MAE: {mae_pca:.4f}")
    print(f"  MSE: {mse_pca:.4f}")
    print(f"  RMSE: {rmse_pca:.4f}")
    
    print(f"\nUnscaled Data:")
    print(f"  Best Model: {results_unscaled['best_model_name']}")
    print(f"  R² Score: {results_unscaled['best_score']:.4f}")
    print(f"  MAE: {mae_unscaled:.4f}")
    print(f"  MSE: {mse_unscaled:.4f}")
    print(f"  RMSE: {rmse_unscaled:.4f}")
    
    print("\n" + "="*80)
    print("SUMMARY OF IMPROVEMENTS:")
    print("="*80)
    print("1. Code has been modularized into reusable functions for:")
    print("   - Data loading")
    print("   - Data cleaning")
    print("   - Feature engineering")
    print("   - Model evaluation with hyperparameter tuning")
    print("   - Results visualization")
    print("\n2. All models now use hyperparameter tuning via GridSearchCV")
    print("\n3. Random state is fixed (random_state=42) for:")
    print("   - DecisionTreeRegressor")
    print("   - RandomForestRegressor")
    print("   - Lasso")
    print("   - Train-test split")
    print("\n4. Code repetition has been eliminated")
    print("   - Original: 3 nearly identical blocks of model training code")
    print("   - Refactored: Single reusable function called 3 times")
    print("="*80)
