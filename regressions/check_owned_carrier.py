"""Opt-in native tests: execute only the 16 exact owner-authored Python texts."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixtures import owned_finite_carrier as owned
from scripts.measure_owned_carrier import run


class OwnedCarrierTests(unittest.TestCase):
    def test_complete_predeclared_carrier_population_and_domain(self):
        decisions = []
        for bits in range(8):
            for edited in (False, True):
                text = owned.source_for(bits, edited)
                passed, calls = owned.measured_validation(text)
                self.assertTrue(passed)
                self.assertEqual(len(calls), 5)
                self.assertEqual([call["input"] for call in calls], list(owned.DOMAIN))
                score, decision = owned.measured_detection(text)
                self.assertEqual(score, format(bits, "03b").count("1"))
                decisions.append(decision)
        self.assertEqual(sum(decisions), 8)

    def test_unknown_even_benign_body_is_not_executed(self):
        with self.assertRaisesRegex(ValueError, "unknown source"):
            owned.measured_validation("def step(x):\n    return x + 2\n")
        for value in (-1, 8, True, "000"):
            with self.assertRaises(ValueError):
                owned.source_for(value)

    def test_raw_workflow_same_information_zero_change_and_no_unsupported_gain(self):
        with tempfile.TemporaryDirectory() as td:
            value = run(Path(td) / "new")
            self.assertEqual((value["pair_count"], value["finite_contract_calls"]), (8, 80))
            self.assertEqual(value["same_information_direct"]["transitions"],
                             dict(n00=4, n01=0, n10=0, n11=4))
            self.assertEqual(value["same_information_direct"]["change"], [0, 1])
            self.assertEqual(value["same_information_direct"]["survival"], [1, 1])
            self.assertEqual(value["same_information_direct"]["marginal_survival_bounds"],
                             [[0, 1], [1, 1]])
            self.assertEqual(value["selection_or_detection_gain"], 0)
            for claim, outcome in value["claim_outcomes"].items():
                self.assertTrue(outcome["raw_pair_membership_rederived"])
                self.assertEqual(outcome["exit_code"], 0 if claim in
                                 ("positive_decision_change", "detection_survival") else 3)

    def test_occupied_output_refused_before_native_execution(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(owned, "measured_validation", side_effect=AssertionError("execution")):
                with self.assertRaisesRegex(ValueError, "must be new"):
                    run(Path(td))


if __name__ == "__main__":
    unittest.main()
