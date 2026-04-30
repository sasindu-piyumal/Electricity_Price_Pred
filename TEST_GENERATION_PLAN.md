# Test Generation Plan

## P1: Time-Series Cross-Validation and Tuning Configuration

### Coverage Map

- **Priority:** P1
- **Rationale:** Tuning configuration is a public pipeline surface in `hyperparameter_tuning.py`, but full hyperparameter searches are too slow and can be non-deterministic for unit tests. Unit coverage should validate configuration shape, fold chronology, and refined-grid bounds without fitting models or running searches.
- **Target file/functions:**
  - `hyperparameter_tuning.py`
  - `setup_cross_validation`
  - `define_parameter_space`
  - `create_refined_param_grid`
- **Recommended test file name:** `test_tuning_configuration.py`
- **Test framework and execution convention:**
  - Use plain Python `unittest` with `unittest.TestCase`.
  - Keep the file runnable with both:
    - `python -m unittest test_tuning_configuration.py`
    - `python test_tuning_configuration.py`
  - Include the direct-execution block:
    ```python
    if __name__ == "__main__":
        unittest.main(verbosity=2)
    ```
- **Fixture strategy:**
  - Use tiny synthetic arrays, especially `numpy.arange`, only for fold inspection.
  - Use fake `best_params` dictionaries for refined grid tests.
  - Avoid real model fitting and do not load the real electricity dataset.
- **Verification style:**
  - Use `sklearn.model_selection.TimeSeriesSplit` type checks.
  - Use assertions for dictionary keys, representative values, valid bounds, and fold ordering.
  - Use `unittest.mock.patch` only if search wrapper construction is covered in a broader test plan.
  - Do not require `pytest` or any pytest-specific features.
- **Search avoidance rule:**
  - Tests in this plan must not run full `RandomizedSearchCV` or `GridSearchCV`.
  - Do not invoke real `RandomizedSearchCV.fit`.
  - Do not invoke real `GridSearchCV.fit`.
  - If search-wrapper behavior is included later, test it only with patching/mocks to verify object construction and expected method calls, not actual model search.

### Proposed Test Cases

#### Cross-validation tests

1. **`test_setup_cross_validation_returns_time_series_split`**
   - Call `setup_cross_validation` with an explicit requested `n_splits`, such as `5`.
   - Assert the returned object is an instance of `TimeSeriesSplit`.
   - Assert the splitter uses the requested split count.
   - Recommended assertion detail:
     - Prefer `self.assertEqual(cv.n_splits, n_splits)` if available on the installed scikit-learn version.
     - If compatibility requires it, inspect the fold count generated from a tiny sequence as a secondary confirmation.

2. **`test_time_series_split_folds_are_monotonic_and_leakage_safe`**
   - Create a tiny ordered sequence such as `np.arange(20)`.
   - Call `setup_cross_validation(n_splits=5)` and materialize `list(cv.split(data))`.
   - Assert the number of folds equals `n_splits`.
   - For each `(train_index, test_index)` pair, assert:
     - both train and test index arrays are non-empty;
     - train indices are strictly before test indices;
     - `max(train_index) < min(test_index)`;
     - `train_index` is monotonically increasing;
     - `test_index` is monotonically increasing;
     - train windows expand or otherwise follow the documented `TimeSeriesSplit` behavior for the installed scikit-learn version.
   - Suggested expansion checks:
     - Track the previous train length and assert the current train length is greater than the previous train length.
     - Track the previous test start and assert each new test window starts after the previous test window.

3. **`test_setup_cross_validation_with_custom_split_count`**
   - Use a non-default split count such as `3`.
   - Call `setup_cross_validation(n_splits=3)`.
   - Assert the returned object configuration reflects `3` splits.
   - Generate folds from a tiny ordered sequence such as `np.arange(12)` or `np.arange(20)`.
   - Assert exactly `3` folds are produced and each fold remains chronological using the same basic leakage-safe checks as above.

#### Parameter-space tests

4. **`test_define_parameter_space_contains_expected_keys`**
   - Call `define_parameter_space()`.
   - Assert the returned value is a dictionary.
   - Assert it includes the Random Forest hyperparameter keys currently returned by the function, aligning with actual public behavior rather than inventing unsupported keys.
   - Expected keys for the current implementation are:
     - `n_estimators`
     - `max_depth`
     - `min_samples_split`
     - `min_samples_leaf`
     - `max_features`
     - `bootstrap`
     - `random_state`
   - The test should be updated if the function output intentionally changes.

5. **`test_define_parameter_space_includes_representative_values`**
   - Call `define_parameter_space()`.
   - Assert each parameter has a non-empty candidate list or distribution.
   - Assert representative values exist, including:
     - `None` in `max_depth` where applicable;
     - numeric max-depth options such as values currently returned by the function;
     - estimator counts such as `100`, `200`, `500`, or `1000` as applicable;
     - split options such as `2`, `5`, `10`, or `20` as applicable;
     - leaf options such as `1`, `2`, `5`, or `10` as applicable;
     - representative `max_features` options currently returned by the function, such as string options (`sqrt`, `log2`) and fractional numeric options if present.
   - Keep assertions representative rather than duplicating every implementation detail unless exact output stability is intended.

6. **`test_parameter_space_values_are_valid_for_random_forest`**
   - Call `define_parameter_space()`.
   - Assert numeric candidates are valid for `RandomForestRegressor`:
     - every `n_estimators` candidate is a positive integer;
     - every non-`None` `max_depth` candidate is a positive integer;
     - every `min_samples_split` candidate is at least `2`;
     - every `min_samples_leaf` candidate is at least `1`;
     - every numeric `max_features` candidate is greater than `0` and less than or equal to `1`; string `max_features` candidates should be accepted only if they are among values supported by the current scikit-learn API and returned by the function;
     - `bootstrap` candidates are booleans;
     - `random_state` candidates are integers or otherwise match the project constant behavior.

#### Refined-grid tests

7. **`test_create_refined_param_grid_contains_expected_keys_from_best_params`**
   - Use a small fake `best_params` dictionary containing all keys required by the current function, for example:
     - `n_estimators: 200`
     - `max_depth: 20`
     - `min_samples_split: 5`
     - `min_samples_leaf: 2`
     - `max_features: "sqrt"`
   - Call `create_refined_param_grid(best_params)`.
   - Assert the returned value is a dictionary.
   - Assert it contains expected refined-grid keys aligned with current behavior:
     - `n_estimators`
     - `max_depth`
     - `min_samples_split`
     - `min_samples_leaf`
     - `max_features`
     - `bootstrap`
     - `random_state`
   - Assert every refined-grid value is represented as a candidate list suitable for `GridSearchCV`.
   - Assert each candidate list is non-empty.

8. **`test_create_refined_param_grid_generates_bounded_numeric_candidates`**
   - Use fake best values near lower boundaries, such as:
     - `n_estimators: 50` or `100`
     - `max_depth: 1` or `5`
     - `min_samples_split: 2`
     - `min_samples_leaf: 1`
     - `max_features: 0.1` or another supported low numeric value
   - Call `create_refined_param_grid(best_params)`.
   - Assert generated numeric candidates remain valid:
     - no non-positive `n_estimators` values;
     - no non-`None` `max_depth` values below `1`;
     - no `min_samples_split` values below `2`;
     - no `min_samples_leaf` values below `1`;
     - numeric `max_features` values remain within `(0, 1]`.
   - Also assert the original best value remains included where that is current documented behavior.
   - This test is expected to reveal boundary bugs if the implementation subtracts from `max_depth` without clamping all generated candidates. Assert the actual intended public behavior once confirmed.

9. **`test_create_refined_param_grid_preserves_none_and_categorical_values`**
   - Use fake `best_params` with `max_depth: None`.
   - Include categorical/string parameters if supported by the function, such as `max_features: "sqrt"` or `"log2"`.
   - Call `create_refined_param_grid(best_params)`.
   - Assert `None` remains represented correctly in `refined_grid["max_depth"]` and is not incorrectly used in arithmetic.
   - Assert categorical values are preserved or expanded only according to documented/current behavior.
   - For current behavior, if string `max_features` expands to a list such as `['sqrt', 'log2', 0.3, 0.5]`, assert the best string value is present and all candidates are valid supported values.

10. **`test_create_refined_param_grid_handles_missing_or_partial_best_params_predictably`**
    - Use a fake `best_params` dictionary missing one optional or required key.
    - Assert the current public behavior and document it in the test name or assertion message:
      - either a clear exception, such as `KeyError`, for missing required keys; or
      - a refined grid generated for available keys if the implementation is changed to support partial dictionaries.
    - For the current implementation, missing keys are accessed directly, so the likely expectation is `KeyError` for missing required keys such as `max_features`, `n_estimators`, `max_depth`, `min_samples_split`, or `min_samples_leaf`.
    - Keep this test focused on predictable behavior, not on changing the implementation.

#### Search avoidance / mock-only guidance

11. **`test_search_wrappers_are_not_executed_in_unit_tests`**
    - Treat this as a planning rule, not necessarily a required standalone test.
    - Do not run real `RandomizedSearchCV` or `GridSearchCV` in this P1 test file.
    - If later tests cover functions that wrap searches, such as `perform_randomized_search` or `perform_grid_search`:
      - patch the sklearn search class used by `hyperparameter_tuning.py` at the import location in that module;
      - assert constructor arguments include the configured `TimeSeriesSplit` object and the expected parameter space or refined grid;
      - assert `.fit()` is called only on tiny fake data and only on the mocked search object;
      - never let a real search object perform model training;
      - never use the real electricity dataset.

### Non-Goals and Constraints

- Do not invoke real `RandomizedSearchCV.fit`.
- Do not invoke real `GridSearchCV.fit`.
- Do not perform model training.
- Do not use the real electricity dataset.
- Do not require pytest.
- Do not depend on random search behavior.
- Keep tests fast, deterministic, and independent.

### Implementation Notes for Teammates

- Imports in `test_tuning_configuration.py` should be minimal: `unittest`, `numpy as np`, `TimeSeriesSplit`, and the three target functions from `hyperparameter_tuning.py`.
- If print output from the target functions is noisy, tests can ignore it unless a future convention adds output suppression.
- Prefer assertions against behavior and validity over brittle exact-list checks, except for stable fixed values such as required keys or documented lower bounds.
- Ensure the test module has no side effects at import time beyond importing the functions under test.
