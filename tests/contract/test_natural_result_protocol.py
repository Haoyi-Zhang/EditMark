"""Data-only checks of the retained RecordedResult protocol; no upstream run."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_natural_modules as check
import check_released_style as style

SOURCE = 'def example():\n    return 1\n'
IDS = ['owned.Test.test_one', 'owned.Test.test_two']


def task(events, passed=None):
    counts = Counter(e['status'] for e in events)
    digest = hashlib.sha256(SOURCE.encode()).hexdigest()
    if passed is None:
        passed = not any(counts[s] for s in ('failed', 'error', 'subtest_failure', 'unexpected_success')) and len(IDS) > counts['skipped']
    return dict(status='completed', module='owned', source_sha256=digest,
                bound_modules=[dict(binding='owned', source_sha256=digest)], test_ids=IDS.copy(),
                started_tests=IDS.copy(), tests_scheduled=2, tests_run=2, outcomes=events,
                failures=counts['failed'] + counts['subtest_failure'], errors=counts['error'],
                skipped=counts['skipped'], expected_failures=counts['expected_failure'],
                unexpected_successes=counts['unexpected_success'], passed=passed, executed_lines=[2])


class NaturalResultProtocolTests(unittest.TestCase):
    def test_missing_events_and_unknown_states_reject(self):
        good = task([dict(id=i, status='passed') for i in IDS])
        for events in ([], good['outcomes'][:1], [dict(id=IDS[0], status='not_completed'), good['outcomes'][1]],
                       [dict(id=IDS[0], status=[]), good['outcomes'][1]]):
            value = copy.deepcopy(good); value['outcomes'] = events
            with self.subTest(events=events), self.assertRaises(ValueError): check.verify_task(value, SOURCE)
            with self.assertRaises(ValueError): check.task_validation(value, good)

    def test_duplicate_success_and_conflicting_terminals_reject(self):
        for extra in ('passed', 'failed', 'error', 'expected_failure', 'unexpected_success'):
            events = [dict(id=i, status='passed') for i in IDS] + [dict(id=IDS[0], status=extra)]
            with self.subTest(extra=extra), self.assertRaises(ValueError): check.verify_task(task(events), SOURCE)

    def test_subtest_failures_are_complete_nonpassing_outcomes(self):
        events = [dict(id=IDS[0], status='subtest_failure', detail='subtest (x=1)'),
                  dict(id=IDS[0], status='subtest_failure', detail='subtest (x=2)'),
                  dict(id=IDS[1], status='passed')]
        value = task(events); check.verify_task(value, SOURCE)
        self.assertFalse(value['passed'])
        # A teardown/cleanup error after a subtest failure is also legal.
        events.append(dict(id=IDS[0], status='error'))
        check.verify_task(task(events), SOURCE)

    def test_skips_and_expected_failures_follow_collector_protocol(self):
        for event in (dict(id=IDS[0], status='skipped', reason='registered skip'),
                      dict(id=IDS[0] + ' (x=1)', status='skipped', reason='subtest skip'),
                      dict(id=IDS[0], status='expected_failure')):
            value = task([event, dict(id=IDS[1], status='passed')])
            check.verify_task(value, SOURCE); self.assertTrue(value['passed'])
        events = [dict(id=IDS[0], status='skipped', reason='skip'), dict(id=IDS[0], status='error'),
                  dict(id=IDS[1], status='passed')]
        check.verify_task(task(events), SOURCE)

    def test_cleanup_errors_and_repeated_subtest_descriptions_are_not_successes(self):
        for event in (dict(id=IDS[0], status='error'),
                      dict(id=IDS[0], status='subtest_failure', detail='subtest (x=1)')):
            value = task([event, copy.deepcopy(event), dict(id=IDS[1], status='passed')])
            check.verify_task(value, SOURCE); self.assertFalse(value['passed'])

    def test_typed_counters_event_shapes_and_expected_failure_summary(self):
        good = task([dict(id=i, status='passed') for i in IDS])
        for field, replacement in (('tests_run', 2.0), ('tests_scheduled', True),
                                   ('passed', 1), ('expected_failures', 1), ('skipped', False)):
            value = copy.deepcopy(good); value[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError): check.verify_task(value, SOURCE)
        bad = copy.deepcopy(good); bad['outcomes'][0]['unexpected'] = True
        with self.assertRaises(ValueError): check.verify_task(bad, SOURCE)

    def test_all_72_saved_tasks_and_admission_values_are_unchanged(self):
        root = ROOT / 'results/natural-modules'
        result = check.derive(root)
        self.assertEqual(result['suite_processes'], 72)
        self.assertEqual(result['same_information_direct_admission'], {'0': 5, '1': 5, '2': 6, '3': 7})
        saved = check.read(root / 'summary.json')
        self.assertEqual(result['key_strata'], saved['key_strata'])
        self.assertEqual((result['marked_task_failures'], result['edited_task_failures']), (4, 4))

    def test_32_saved_style_pairs_and_denominators_are_unchanged(self):
        result = style.derive(ROOT / 'results/released-style')
        saved = json.loads((ROOT / 'results/released-style/summary.json').read_text(encoding='utf-8'))
        self.assertEqual(result['pairs'], 32)
        self.assertEqual(result['same_information_direct_admission'], 22)
        self.assertEqual(result['key_strata'], saved['key_strata'])

    def test_saved_rollback_and_followup_still_validate_without_execution(self):
        root = ROOT / 'results/natural-modules'
        self.assertEqual(check.verify_attribution(root)['restored_passes'], 4)
        result = check.verify_statistics_followup(root)
        self.assertEqual((result['passing'], result['tests_per_source'], result['separately_validated_pairs']), (9, 346, 4))


if __name__ == '__main__':
    unittest.main()
