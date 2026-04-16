# Electricity Price Prediction — Project Description

## Overview

This project develops a machine learning pipeline to **predict electricity prices** (`SMPEP2` — System Marginal Price, Euro per MWh) using historical electricity grid data from Ireland. The goal is to build an accurate regression model and systematically optimise it through hyperparameter tuning to exceed a baseline R² score of **0.6502**.

---

## Dataset

| Property | Detail |
|---|---|
| File | `electricity.csv` |
| Index | `DateTime` (parsed as dates) |
| Target variable | `SMPEP2` — actual electricity spot price (€/MWh) |

### Key Features

| Feature | Description |
|---|---|
| `HolidayFlag` | Binary flag indicating a public holiday |
| `DayOfWeek` | Day of the week (0–6) |
| `Day` / `Month` / `Year` | Calendar components |
| `PeriodOfDay` | Half-hour period within a day |
| `ForecastWindProduction` | Forecasted wind energy (MW) |
| `ActualWindProduction` | Actual wind energy generated (MW) |
| `SystemLoadEA` | Forecasted national system load (MW) |
| `SystemLoadEP2` | Actual national system load (MW) |
| `SMPEA` | Forecasted system marginal price (€/MWh) |
| `ORKTemperature` | Temperature at Cork Airport (°C) |
| `ORKWindspeed` | Wind speed at Cork Airport (km/h) |
| `CO2Intensity` | CO₂ intensity of electricity generation (gCO₂/kWh) |

---

## Project Workflow

### 1. Data Cleaning
- Converts mixed-type columns to numeric with `pd.to_numeric(..., errors='coerce')`.
- Drops rows where `ORKTemperature` or `ORKWindspeed` are missing (too many NaN values for safe imputation).
- Removes price outliers: keeps only records where `0 < SMPEP2 ≤ 550`.
- Fills remaining missing values with **median** (skewed columns) or **mean** (`CO2Intensity`, which is normally distributed).

### 2. Exploratory Data Analysis (EDA)
- Distribution plots (histograms + KDE) and box plots for all numeric features.
- Correlation heatmap revealing:
  - High correlation between `Month` ↔ `WeekOfYear`
  - High correlation between `ForecastWindProduction` ↔ `ActualWindProduction`
  - High correlation between `SystemLoadEA` ↔ `SystemLoadEP2`
- Redundant features removed to reduce multicollinearity: `Holiday`, `WeekOfYear`, `ForecastWindProduction`, `SystemLoadEA`.

### 3. Feature Engineering
- **Cyclic transformations** applied to `DayOfWeek`, `Day`, `Month`, and `PeriodOfDay` using sine and cosine mappings to preserve their circular nature:

  ```
  feature_SIN = sin(feature / max * 2π)
  feature_COS = cos(feature / max * 2π)
  ```

- Original cyclic columns are then dropped.

### 4. Data Splitting & Scaling
- An 80/20 **train-test split** with `random_state=42`.
- Features scaled with **MinMaxScaler** (fitted on training data only to prevent data leakage).

### 5. Model Selection
Multiple regression models are evaluated:

| Model | Notes |
|---|---|
| Linear Regression | Baseline linear model |
| Lasso Regression | Regularised linear model |
| Decision Tree Regressor | Non-linear, prone to overfitting |
| **Random Forest Regressor** | **Best performer — selected as final model** |
| Support Vector Regressor (SVR) | Kernel-based approach |

### 6. Dimensionality Reduction (PCA)
- PCA is explored to find the number of components explaining the most variance.
- 11 principal components are selected and models are re-evaluated on the PCA-transformed data.

### 7. Hyperparameter Tuning
A dedicated two-phase tuning pipeline (`hyperparameter_tuning.py`) optimises the Random Forest model:

| Phase | Method | Detail |
|---|---|---|
| Phase 1 | `RandomizedSearchCV` | 100 random iterations across the full parameter space |
| Phase 2 | `GridSearchCV` | Fine-grained search around the best parameters from Phase 1 |

**Cross-validation**: `TimeSeriesSplit` (5 splits) to respect temporal ordering and avoid look-ahead bias.

**Search space**:
```python
{
    'n_estimators':      [100, 200, 500, 1000],
    'max_depth':         [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf':  [1, 2, 5, 10],
    'max_features':      ['sqrt', 'log2', 0.3, 0.5, 0.8]
}
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **R² Score** | Primary metric — proportion of variance explained |
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |

**Baseline R² (Random Forest, default params):** `0.6502`  
**Target R² (after tuning):** `> 0.6502`

---

## Project Files

| File | Purpose |
|---|---|
| `electricity.csv` | Raw dataset |
| `Electricity Data.py` | Main analysis and modelling notebook/script |
| `Electricity Data.ipynb` | Jupyter Notebook version of the analysis |
| `hyperparameter_tuning.py` | Hyperparameter tuning pipeline |
| `analyze_tuning_results.py` | Utility to load and inspect saved tuning results |
| `HYPERPARAMETER_TUNING_README.md` | Detailed documentation for the tuning script |
| `IMPLEMENTATION_SUMMARY.md` | Summary of the tuning implementation |
| `BUG_FIX_SUMMARY.md` | Log of bugs identified and resolved |
| `Electricity Price Prediction Model.pdf` | Project report / reference document |

---

## How to Run

```bash
# Run the full analysis pipeline
python "Electricity Data.py"

# Run hyperparameter tuning (may take 1–2 hours)
python hyperparameter_tuning.py

# Analyse tuning results after completion
python analyze_tuning_results.py
```

---

## Key Technical Decisions

- **Median imputation** over mean for skewed features avoids distorting the distribution.
- **Dropping rows** with missing `ORKTemperature`/`ORKWindspeed` is preferred over imputation due to the high proportion of missing values.
- **Cyclic encoding** (sin/cos) correctly represents periodic time features, unlike ordinal encoding.
- **TimeSeriesSplit** instead of random cross-validation prevents future data leaking into training folds.
- **`random_state=42`** is fixed throughout for full reproducibility.
- **Joblib** is used for model serialisation (more secure than `pickle` for scikit-learn objects).
