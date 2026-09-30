# test_gate.py
"""Tests of the gate's LOGIC with synthetic results. These say nothing about
whether any real model is good; they prove the gate can pass, and that each
condition can fail on its own."""
import unittest
from run_eval import decide

GATE = {"min_heldout_gain": 0.02, "max_case_regressions": 0, "min_case_accuracy": 0.70}
INC_HELD = [True] * 204 + [False] * 96            # incumbent: 204/300 held-out
INC_CASES = [True] * 14 + [False] * 6              # incumbent: 14/20 authored


class GateLogic(unittest.TestCase):
    def verdict(self, new_held, new_cases):
        checks, _ = decide(GATE, INC_HELD, new_held, INC_CASES, new_cases)
        return list(checks.values())

    def test_can_pass(self):
        self.assertEqual(self.verdict([True] * 214 + [False] * 86, [True] * 16 + [False] * 4), [True] * 3)

    def test_exact_two_point_gain_passes(self):
        # 210/300 - 204/300 is 0.0199999... in floating point; exact counting must still pass it.
        self.assertTrue(self.verdict([True] * 210 + [False] * 90, [True] * 16 + [False] * 4)[0])

    def test_small_gain_fails_alone(self):
        self.assertEqual(self.verdict([True] * 209 + [False] * 91, [True] * 16 + [False] * 4), [False, True, True])

    def test_one_regression_fails_alone(self):
        new_cases = [True] * 13 + [False] + [True] * 6    # breaks case 13, fixes the other six
        self.assertEqual(self.verdict([True] * 214 + [False] * 86, new_cases), [True, False, True])

    def test_floor_fails_alone(self):
        weak_inc = [True] * 10 + [False] * 10              # a weak incumbent at 10/20
        checks, _ = decide(GATE, INC_HELD, [True] * 214 + [False] * 86, weak_inc, [True] * 13 + [False] * 7)
        self.assertEqual(list(checks.values()), [True, True, False])


if __name__ == "__main__":
    unittest.main()
