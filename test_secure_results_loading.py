#!/usr/bin/env python
# coding: utf-8

"""
Regression tests for safe tuning-result serialization.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import analyze_tuning_results
import hyperparameter_tuning


class TestSecureResultsLoading(unittest.TestCase):
    def build_minimal_results(self):
        return {
            'baseline_model': {
                'r2': 0.5,
                'metrics': {'r2': 0.5, 'mae': 2.0, 'mse': 4.0, 'rmse': 2.0},
            },
            'best_model': {
                'r2': 0.75,
                'metrics': {'r2': 0.75, 'mae': 1.0, 'mse': 1.0, 'rmse': 1.0},
                'feature_importance': [{'feature': 'SystemLoadEP2', 'importance': 0.9}],
            },
            'grid_search': {'best_params': {'n_estimators': 100}, 'time_minutes': 1.0},
            'random_search': {'time_minutes': 0.5},
            'metadata': {
                'total_time_minutes': 1.5,
                'train_shape': [10, 3],
                'test_shape': [2, 3],
                'cv_splits': 5,
                'random_state': 42,
                'timestamp': '2024-01-01 00:00:00',
            },
        }

    def test_analyzer_rejects_joblib_pickle_payload_without_executing_it(self):
        """A malicious pickle/joblib payload must not be deserialized."""
        marker_path = '/tmp/unsafe_joblib_load'
        if os.path.exists(marker_path):
            os.remove(marker_path)

        malicious_pickle_payload = (
            b"\x80\x04cos\nsystem\n\x94"
            b"\x8c\x1dtouch /tmp/unsafe_joblib_load\x94\x85\x94R\x94."
        )

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as payload_file:
            payload_file.write(malicious_pickle_payload)
            payload_path = payload_file.name

        try:
            with patch('os.system') as mocked_system:
                analyze_tuning_results.load_and_analyze_results(payload_path)

            mocked_system.assert_not_called()
            self.assertFalse(os.path.exists(marker_path))
        finally:
            os.remove(payload_path)
            if os.path.exists(marker_path):
                os.remove(marker_path)

    def test_safe_json_results_can_still_be_analyzed(self):
        """Valid JSON result summaries remain supported after removing joblib loading."""
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as results_file:
            json.dump(self.build_minimal_results(), results_file)
            results_path = results_file.name

        try:
            analyze_tuning_results.load_and_analyze_results(results_path)
        finally:
            os.remove(results_path)

    def test_save_results_writes_json_not_pickle(self):
        """Saving results should produce JSON that does not start with pickle magic bytes."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as results_file:
            results_path = results_file.name

        try:
            hyperparameter_tuning.save_results(self.build_minimal_results(), results_path)
            with open(results_path, 'rb') as saved_file:
                first_two_bytes = saved_file.read(2)

            self.assertNotEqual(first_two_bytes, b'\x80\x04')
            with open(results_path, 'r', encoding='utf-8') as saved_file:
                loaded = json.load(saved_file)
            self.assertEqual(loaded['best_model']['r2'], 0.75)
        finally:
            os.remove(results_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
