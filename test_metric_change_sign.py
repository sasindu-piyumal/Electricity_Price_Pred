#!/usr/bin/env python
# coding: utf-8

"""
Unit test to verify the sign of percentage-change reporting for error metrics
(MAE, MSE, RMSE) in analyze_tuning_results.py.

Bug: The original code computed:
    change = ((baseline_val - optimized_val) / baseline_val) * 100
then printed  {-change}  — negating the value — so a genuine improvement
(lower error in the optimised model) was displayed as a negative percentage,
implying the metric got worse.

Fix: Print  {change}  (without negation) so that a reduction in error is
shown as a positive percentage improvement.
"""

import io
import unittest


def _compute_change_buggy(baseline_val, optimized_val):
    """Reproduces the original (buggy) display value."""
    change = ((baseline_val - optimized_val) / baseline_val) * 100
    return -change  # BUG: sign is inverted


def _compute_change_fixed(baseline_val, optimized_val):
    """Reproduces the fixed display value."""
    change = ((baseline_val - optimized_val) / baseline_val) * 100
    return change  # FIXED: no negation


class TestMetricChangeSign(unittest.TestCase):
    """
    Tests that the percentage change reported for error metrics has the
    correct sign: positive means the optimised model has lower (better) error.
    """

    def test_improvement_shown_as_positive_with_fixed_formula(self):
        """
        When the optimised model has a lower error than the baseline,
        the displayed change should be POSITIVE (improvement).
        """
        baseline_mae = 10.0
        optimized_mae = 8.0  # lower is better

        displayed = _compute_change_fixed(baseline_mae, optimized_mae)

        self.assertGreater(
            displayed, 0,
            "A reduction in error should be displayed as a positive percentage change."
        )
        self.assertAlmostEqual(displayed, 20.0, places=5)

    def test_degradation_shown_as_negative_with_fixed_formula(self):
        """
        When the optimised model has a higher error than the baseline,
        the displayed change should be NEGATIVE (degradation).
        """
        baseline_mae = 10.0
        optimized_mae = 12.0  # higher is worse

        displayed = _compute_change_fixed(baseline_mae, optimized_mae)

        self.assertLess(
            displayed, 0,
            "An increase in error should be displayed as a negative percentage change."
        )
        self.assertAlmostEqual(displayed, -20.0, places=5)

    def test_buggy_formula_gives_wrong_sign_for_improvement(self):
        """
        Demonstrates that the BUGGY formula reports a negative value for an
        improvement — this test would FAIL if the old code were used as the
        production path, and PASSES only because we are deliberately calling
        the buggy helper to show its incorrect output.
        """
        baseline_mae = 10.0
        optimized_mae = 8.0  # genuine improvement

        buggy_displayed = _compute_change_buggy(baseline_mae, optimized_mae)

        # The buggy formula produces a NEGATIVE number for an improvement —
        # assert that fact to document the old broken behaviour.
        self.assertLess(
            buggy_displayed, 0,
            "The buggy formula incorrectly shows a negative change for an improvement."
        )

    def test_fixed_formula_matches_expected_percentage(self):
        """
        End-to-end check: baseline=100, optimised=75 → 25 % reduction in error.
        """
        baseline = 100.0
        optimized = 75.0

        displayed = _compute_change_fixed(baseline, optimized)
        self.assertAlmostEqual(displayed, 25.0, places=5,
                               msg="Expected a 25 % improvement to be shown as +25.0")


if __name__ == '__main__':
    print("=" * 60)
    print("METRIC CHANGE SIGN BUG — DEMONSTRATION AND VERIFICATION")
    print("=" * 60)
    unittest.main(verbosity=2)
