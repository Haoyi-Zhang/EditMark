"""Small mathematical fixtures for the declared missing-bound information set."""
from dataclasses import asdict, replace
import itertools
import unittest

from editmark_audit.analysis import report, summarize
from editmark_audit.bounds import decision_difference_range
from editmark_audit.constants import PAIR_SCHEMA
from editmark_audit.records import Pair


def pair(after_decision, after_score):
    def program(source, decision, score):
        return {
            'source': source,
            'validation': {
                'available': True, 'passed': True, 'contract_id': 'toy-check',
            },
            'detection': {
                'available': True, 'detected': decision, 'score': score,
                'threshold': 1, 'comparator': 'ge', 'rule_id': 'toy-rule',
            },
        }
    return Pair.parse({
        'schema_version': PAIR_SCHEMA, 'population': 'watermarked_positive',
        'evidence_kind': 'synthetic_test', 'experiment_id': 'toy',
        'method': 'toy', 'model': 'none', 'source_group': 'toy',
        'language': 'text', 'sample_id': 'one', 'replicate_id': 'one',
        'edit': 'toy-edit', 'variant_id': 'one', 'supported': True,
        'original': program('before toy', True, 2),
        'edited': program('after toy', after_decision, after_score),
    })


class MissingBoundInformationTests(unittest.TestCase):
    def test_ranges_are_sharp_for_boolean_null_projection(self):
        for before, after in itertools.product((False, True, None), repeat=2):
            completions = [int(b) - int(a)
                           for b in ((False, True) if before is None else (before,))
                           for a in ((False, True) if after is None else (after,))]
            self.assertEqual(decision_difference_range(before, after),
                             (min(completions), max(completions)))

    def test_score_determined_missing_decision_stays_recorded_missing(self):
        row = pair(None, 2)
        result = summarize([row])
        self.assertFalse(row.edited.detection.observed)
        self.assertEqual(result['admitted_pairs'], 0)
        self.assertEqual(result['eligible_target_pairs'], 1)
        change = result['missing_decision_change_bounds']
        loss = result['missing_decision_bounds']
        self.assertEqual((change['lower'], change['upper']), (-1, 0))
        self.assertEqual((loss['lower'], loss['upper']), (0, 1))
        # Both scores imply true, so the richer information fixes change at 0.
        self.assertLessEqual(change['lower'], 0)
        self.assertGreaterEqual(change['upper'], 0)
        for interval in (change, loss):
            self.assertIn('coarsened observed-Boolean/null projection',
                          interval['interpretation'])
            self.assertIn('conservative', interval['interpretation'])

    def test_complete_cohort_arithmetic_is_unchanged(self):
        result = summarize([pair(False, 0)])
        self.assertEqual(result['admitted_pairs'], 1)
        self.assertEqual(result['net_detection_change_exact'],
                         {'numerator': -1, 'denominator': 1, 'decimal': -1.0})
        interval = result['missing_decision_change_bounds']
        self.assertEqual((interval['lower'], interval['upper']), (-1, -1))

    def assert_selected_mode_target_description(self, row, mode):
        row = Pair.parse(asdict(row) | {
            'schema_version': PAIR_SCHEMA, 'population': 'watermarked_positive',
        })
        estimates = report([row])['strata'][0]['estimates']
        self.assertEqual(estimates['changed_only']['eligible_target_pairs'], 0)
        result = estimates[mode]
        self.assertEqual(result['eligible_target_pairs'], 1)
        self.assertEqual(result['admitted_pairs'], 1)
        self.assertEqual(result['net_detection_change_exact'],
                         {'numerator': 0, 'denominator': 1, 'decimal': 0.0})
        change = result['missing_decision_change_bounds']
        loss = result['missing_decision_bounds']
        for interval in (change, loss):
            self.assertEqual((interval['lower'], interval['upper']), (0, 0))
            self.assertIn('coarsened observed-Boolean/null projection',
                          interval['interpretation'])
            self.assertIn('conservative', interval['interpretation'])
        self.assertIn('eligible under the selected analysis mode',
                      change['interpretation'])
        self.assertNotIn('Only jointly valid, changed, same-rule pairs',
                         change['interpretation'])

    def test_include_known_noops_target_description_matches_mode(self):
        seed = pair(True, 2)
        noop = replace(seed, edited=seed.original, changed=False)
        self.assert_selected_mode_target_description(noop, 'include_known_noops')

    def test_all_supported_target_description_matches_mode(self):
        seed = pair(True, 2)
        rows = (
            replace(seed, edited=seed.original, changed=False),
            replace(seed, changed=None,
                    original=replace(seed.original, source_sha256=''),
                    edited=replace(seed.edited, source_sha256='')),
        )
        for row in rows:
            with self.subTest(changed=row.changed):
                self.assert_selected_mode_target_description(row, 'all_supported')


if __name__ == '__main__':
    unittest.main()
