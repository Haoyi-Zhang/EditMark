"""Owned synthetic/schema cases and independent arithmetic; no detector execution."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from editmark_audit.aggregates import analyze
from editmark_audit.certificate import certify_analysis
from editmark_audit.constants import CERTIFICATE_BUNDLE_SCHEMA
from editmark_audit.io import EvidenceError, load_rows
from editmark_audit.verify import verify_bundle
from scripts.diagnose_information import direct_counts, fixture_checks, inventory, rounded_components


def bundle():
    counts = load_rows(ROOT / 'examples/synthetic_cohort_counts.json')[0]
    return {'schema_version': CERTIFICATE_BUNDLE_SCHEMA,
            'input_file': {'name': 'owned-counts.json', 'sha256': 'a' * 64},
            'requested_claim': 'positive_decision_change',
            'certificates': [certify_analysis(analyze(counts), 'positive_decision_change')]}


class BundleSchemaRegressionTests(unittest.TestCase):
    def test_empty_short_nonhex_and_uppercase_digest_rejected(self):
        for value in ('', 'a' * 63, 'g' * 64, 'A' * 64, None, 1):
            item = bundle()
            item['input_file']['sha256'] = value
            with self.subTest(value=value), self.assertRaises(EvidenceError):
                verify_bundle(item)

    def test_unknown_bundle_field_rejected(self):
        item = bundle()
        item['basis_verifed'] = True
        with self.assertRaisesRegex(EvidenceError, 'unknown fields'):
            verify_bundle(item)

    def test_unknown_digest_record_field_rejected(self):
        item = bundle()
        item['input_file']['sha265'] = 'a' * 64
        with self.assertRaisesRegex(EvidenceError, 'unknown fields'):
            verify_bundle(item)

    def test_missing_empty_and_whitespace_input_name_rejected(self):
        for value in (None, '', ' ', ' counts.json', 1):
            item = bundle()
            item['input_file']['name'] = value
            with self.subTest(value=value), self.assertRaises(EvidenceError):
                verify_bundle(item)

    def test_syntactically_valid_digest_does_not_authenticate_basis(self):
        result = verify_bundle(bundle())
        self.assertFalse(result['basis_verified'])
        self.assertTrue(result['certificates'][0]['embedded_facts_verified'])


class InformationDiagnosticTests(unittest.TestCase):
    def test_direct_marginal_arithmetic_and_undefined_denominators(self):
        document = load_rows(ROOT / 'examples/synthetic_cohort_counts.json')[0]
        self.assertEqual(direct_counts(document),
                         {'change': [-2, 5], 'survival_bounds': [[1, 4], [1, 2]]})
        document['counts'] = {'n': 0, 'before_positive': 0, 'after_positive': 0}
        self.assertEqual(direct_counts(document),
                         {'change': None, 'survival_bounds': [None, None]})

    def test_contradictory_or_noninteger_direct_counts_rejected(self):
        for counts in ({'n': 1, 'before_positive': 2, 'after_positive': 1},
                       {'n': True, 'before_positive': 1, 'after_positive': 1}):
            with self.subTest(counts=counts), self.assertRaises(EvidenceError):
                direct_counts({'counts': counts})

    def test_retained_fixtures_and_missing_only_targets_agree(self):
        result = fixture_checks(ROOT)
        self.assertEqual(len(result['pair_cases']), 9)
        self.assertTrue(all(c['agrees'] for c in result['pair_cases']))
        absent = next(c for c in result['pair_cases'] if c['variant'] == 'all_decisions_missing')
        self.assertEqual((absent['complete_pairs'], absent['target_pairs']), (0, 5))
        self.assertIsNone(absent['change'])
        self.assertEqual(absent['target_change_bounds'], [[-1, 1], [1, 1]])
        self.assertEqual(absent['completion_worlds'], 1024)

    def test_rounded_components_do_not_become_paired_change(self):
        cells = rounded_components(ROOT)
        self.assertEqual(len(cells), 28)
        self.assertTrue(all(c['paired_change'] is None for c in cells))

    def test_inventory_errors_stay_unresolved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'bad.json').write_text('{invalid', encoding='utf-8')
            result = inventory(root)
            self.assertEqual(len(result['read_errors']), 1)
            self.assertEqual(result['paired_effect_status'], 'requires_record_review')
            self.assertIsNone(result['paired_change'])

    def test_schema_properties_are_not_observations(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'schema.json').write_text(
                '{"$schema":"https://json-schema.org/draft/2020-12/schema",'
                '"properties":{"original":{},"edited":{}}}', encoding='utf-8')
            result = inventory(root)
            self.assertEqual(result['unresolved_candidate_files'], [])

    def test_archived_nan_is_tagged_not_replaced_by_measurement(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'aggregate.json'
            path.write_text('[{"mean_attacked_score":NaN}]', encoding='utf-8')
            saved = path.read_bytes()
            result = inventory(root)
            self.assertEqual(result['archived_format_issues'][0]['recorded_literals'], ['NaN'])
            self.assertEqual(result['read_errors'], [])
            self.assertIsNone(result['paired_change'])
            self.assertEqual(path.read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
