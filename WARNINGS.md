# Warning Suppressions Documentation

This document explains all warning suppressions in the Electricity Data project and their justifications.

## Overview

Previously, this project used a global `warnings.filterwarnings('ignore')` statement that suppressed **all** warnings indiscriminately. This has been removed and replaced with targeted warning filters that only suppress specific, known-safe warnings.

## Current Warning Suppressions

### 1. Sklearn FutureWarnings

**Filter:**
```python
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
```

**Justification:**
- Sklearn regularly issues FutureWarnings about upcoming API changes
- These are informational only and do not affect current functionality
- The warnings relate to deprecations that will occur in future sklearn versions
- Fixing these requires sklearn version-specific code or upgrading dependencies
- Suppressing these allows the code to run cleanly while remaining compatible with current sklearn versions

**Action Required:** Monitor sklearn release notes and update code when upgrading to newer versions.

---

### 2. Sklearn ConvergenceWarnings

**Filter:**
```python
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
```

**Justification:**
- This project performs model comparison across multiple algorithms (LinearRegression, Lasso, DecisionTreeRegressor, RandomForestRegressor, SVR)
- Models like Lasso and SVR may not converge with default hyperparameters on this dataset
- The purpose is **exploratory analysis** to identify the best model type, not production deployment
- ConvergenceWarnings are expected and acceptable in this context
- The models still produce valid predictions for comparison purposes
- Final model selection (RandomForestRegressor) does not have convergence issues

**When This Matters:**
- If deploying Lasso or SVR to production, convergence warnings should be addressed by:
  - Increasing `max_iter` parameter
  - Adjusting solver settings
  - Scaling features differently
  - Using cross-validation to tune hyperparameters

**Current Impact:** Minimal - these models are not selected as the final model. RandomForestRegressor (the chosen model) does not produce convergence warnings.

---

## Fixed Warnings (No Longer Suppressed)

### Pandas FutureWarnings for Inplace Operations

**Issue:** Pandas deprecated the `inplace=True` parameter for many operations, issuing FutureWarnings.

**Resolution:** Fixed by replacing inplace operations with assignment:
- Line ~108: `df = df.replace(['', 'NA', 'N/A', None], np.nan)`
- Line ~150: `df_cleaned[col] = df_cleaned[col].fillna(median_col)`
- Line ~160: `df_cleaned['CO2Intensity'] = df_cleaned['CO2Intensity'].fillna(mean_CO2Intensity)`

**Benefit:** Code is now future-proof for upcoming pandas versions and follows best practices.

---

## Warnings That Should Be Visible

The following warnings are **not suppressed** and will appear if issues occur:

1. **RuntimeWarning** - Indicates potential numerical issues (division by zero, invalid operations)
2. **UserWarning** (non-sklearn) - General warnings from libraries
3. **DeprecationWarning** (from non-sklearn modules) - Important for identifying deprecated code
4. **Data-related warnings** - Issues with data loading, parsing, or processing

---

## Testing Process

To verify warning handling:

1. **Enable all warnings temporarily:**
   ```python
   warnings.filterwarnings('default')  # Show all warnings
   ```

2. **Run the code and observe output**

3. **Expected warnings:**
   - None from pandas (all fixed)
   - Possible sklearn ConvergenceWarnings from Lasso/SVR (intentionally suppressed)
   - Possible sklearn FutureWarnings (intentionally suppressed)

4. **Unexpected warnings should be investigated and fixed**

---

## Maintenance Guidelines

### When to Add a Warning Filter

✅ **Add a filter when:**
- The warning is from a third-party library and cannot be fixed in our code
- The warning is informational and doesn't affect functionality
- The warning is expected behavior for exploratory analysis
- Fixing the warning would require significant refactoring without benefit

❌ **Do NOT add a filter when:**
- The warning indicates a bug in our code
- The warning suggests incorrect usage of an API
- The warning relates to data quality issues
- The warning indicates potential runtime errors

### Review Schedule

- Review this document when upgrading major dependencies (pandas, sklearn, numpy)
- Quarterly review to check if suppressed warnings can be resolved
- Before production deployment, ensure all warnings are addressed or justified

---

## Summary

This project now follows warning best practices:
- ✅ No global warning suppression
- ✅ Fixed underlying pandas issues
- ✅ Targeted suppressions for unavoidable warnings
- ✅ Clear documentation of all suppressions
- ✅ Warnings that indicate real issues remain visible

For questions or concerns about specific warnings, please refer to this document or update it as needed.
