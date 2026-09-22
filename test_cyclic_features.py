#!/usr/bin/env python
# coding: utf-8

import unittest

import numpy as np
import pandas as pd

from hyperparameter_tuning import add_cyclic_features


class TestCyclicFeatures(unittest.TestCase):

    def test_zero_valued_cyclic_feature_does_not_create_nan(self):
        df = pd.DataFrame({
            'DayOfWeek': [0, 0, 0],
            'Day': [1, 2, 3],
            'Month': [1, 1, 1],
            'PeriodOfDay': [1, 2, 3],
        })

        transformed = add_cyclic_features(df)

        self.assertFalse(transformed['DayOfWeek_SIN'].isna().any())
        self.assertFalse(transformed['DayOfWeek_COS'].isna().any())
        self.assertTrue((transformed['DayOfWeek_SIN'] == 0).all())
        self.assertTrue((transformed['DayOfWeek_COS'] == 1).all())

    def test_encodings_are_independent_of_observed_domain(self):
        domains = {
            'DayOfWeek': range(7),
            'Day': range(1, 32),
            'Month': range(1, 13),
            'PeriodOfDay': range(1, 49),
        }
        representative_values = {
            'DayOfWeek': [0, 3, 6],
            'Day': [1, 15, 31],
            'Month': [1, 6, 12],
            'PeriodOfDay': [1, 24, 48],
        }

        for feature, full_domain in domains.items():
            with self.subTest(feature=feature):
                full = self._transform_feature(feature, list(full_domain))
                subset = self._transform_feature(
                    feature, representative_values[feature]
                )

                for value in representative_values[feature]:
                    full_row = full.loc[value]
                    subset_row = subset.loc[value]
                    np.testing.assert_allclose(
                        subset_row,
                        full_row,
                        atol=1e-12,
                        rtol=0,
                    )

    def test_first_valid_categories_have_zero_phase(self):
        first_categories = {
            'DayOfWeek': 0,
            'Day': 1,
            'Month': 1,
            'PeriodOfDay': 1,
        }

        for feature, first_value in first_categories.items():
            with self.subTest(feature=feature):
                transformed = self._transform_feature(feature, [first_value])
                self.assertAlmostEqual(
                    transformed.loc[first_value, f'{feature}_SIN'], 0.0
                )
                self.assertAlmostEqual(
                    transformed.loc[first_value, f'{feature}_COS'], 1.0
                )

    @staticmethod
    def _transform_feature(feature, values):
        defaults = {
            'DayOfWeek': 0,
            'Day': 1,
            'Month': 1,
            'PeriodOfDay': 1,
        }
        frame = pd.DataFrame({
            name: values if name == feature else [default] * len(values)
            for name, default in defaults.items()
        })
        frame.index = values
        transformed = add_cyclic_features(frame)
        return transformed[[f'{feature}_SIN', f'{feature}_COS']]


if __name__ == '__main__':
    unittest.main()
