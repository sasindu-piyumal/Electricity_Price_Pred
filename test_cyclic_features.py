#!/usr/bin/env python
# coding: utf-8

import unittest

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


if __name__ == '__main__':
    unittest.main()
