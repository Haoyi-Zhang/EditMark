"""Owned synthetic coverage regressions; no observations or metrics are invented."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from editmark_audit import certificate, verify
from editmark_audit.analysis import report
from editmark_audit.constants import (
    CERTIFICATE_BUNDLE_SCHEMA, COVERAGE_BUNDLE_SCHEMA, UNIDENTIFIED_STRATUM_SCHEMA,
)
from editmark_audit.io import EvidenceError, digest
from editmark_audit.records import Pair
from test_contract import fixture


def missing_pair(name='missing', sides=('edited',)):
    row = fixture(name)
    for side in sides:
        row[side]['detection'].update(available=False, detected=None, score=None)
    return row


def rehash(value):
    core = dict(value)
    core.pop('certificate_sha256')
    value['certificate_sha256'] = certificate.canonical_hash(core)


def pair_bundle(basis, values, claim='positive_decision_change'):
    return {
        'schema_version': certificate.bundle_schema_for(values),
        'input_file': {'name': basis.name, 'sha256': digest(basis)},
        'requested_claim': claim, 'certificates': values,
    }


class UnidentifiedExportTests(unittest.TestCase):
    def test_missing_only_report_preserves_unidentified_record(self):
        basis = report([Pair.parse(missing_pair())])
        values = certificate.certificates_from_pair_report(basis, 'positive_decision_change')
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]['decision'], 'not_admissible')
        self.assertEqual(values[0]['identification'], 'unidentified_missing_decisions')
        self.assertIsNone(values[0]['cohort_sha256'])
        self.assertIsNone(values[0]['estimate'])
        self.assertNotIn('counts', values[0])

    def test_mixed_report_keeps_complete_and_missing_strata(self):
        known, unknown = fixture('known'), missing_pair()
        unknown['method'] = 'missing-method'
        basis = report([Pair.parse(known), Pair.parse(unknown)])
        values = certificate.certificates_from_pair_report(basis, 'positive_decision_change')
        self.assertEqual(len(values), basis['stratum_count'])
        by_method = {value['stratum']['method']: value for value in values}
        self.assertEqual(by_method['toy']['decision'], 'admissible')
        self.assertEqual(by_method['missing-method']['decision'], 'not_admissible')

    def test_missing_only_cli_writes_denial(self):
        with tempfile.TemporaryDirectory() as tmp:
            basis, output = Path(tmp)/'pairs.json', Path(tmp)/'bundle.json'
            basis.write_text(json.dumps([missing_pair()]), encoding='utf-8')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = certificate.main([str(basis), '--claim', 'positive_decision_change',
                                           '--out', str(output)])
            self.assertEqual(status, 3)
            self.assertEqual(len(json.loads(output.read_text())['certificates']), 1)

    def test_each_missing_side_retains_exact_coverage_without_denominator(self):
        for sides in (('original',), ('edited',), ('original', 'edited')):
            with self.subTest(sides=sides):
                basis = report([Pair.parse(missing_pair(sides=sides))])
                estimate = basis['strata'][0]['estimates']['changed_only']
                value = certificate.certificates_from_pair_report(
                    basis, 'positive_decision_change')[0]
                self.assertEqual(value['schema_version'], UNIDENTIFIED_STRATUM_SCHEMA)
                self.assertEqual(value['attempted_pairs'], 1)
                self.assertEqual(value['eligible_target_pairs'], 1)
                self.assertEqual(value['admitted_pairs'], 0)
                self.assertEqual(value['first_exclusion_counts'], estimate['first_exclusion_counts'])
                self.assertEqual(value['known_target_sha256'], estimate['known_target_sha256'])
                self.assertIsNone(value['cohort_sha256'])
                for absent in ('counts', 'transitions', 'bounds', 'denominator', 'estimand'):
                    self.assertNotIn(absent, value)
                self.assertIsNone(value['estimate'])
                self.assertFalse(value['membership_verified'])
                self.assertEqual(value['synthetic_warning'], 'NOT AN EMPIRICAL RESULT')
                self.assertTrue(verify.verify_certificate(value)['embedded_facts_verified'])

    def test_scores_do_not_impute_missing_decisions(self):
        row = fixture()
        row['edited']['detection']['detected'] = None
        value = certificate.certificates_from_raw_pairs([row], 'positive_decision_change')[0]
        self.assertEqual(value['identification'], 'unidentified_missing_decisions')
        self.assertIsNone(value['estimate'])

    def test_unavailable_validation_stays_outside_known_target(self):
        row = missing_pair()
        row['edited']['validation'] = {}
        value = certificate.certificates_from_raw_pairs([row], 'positive_decision_change')[0]
        self.assertEqual(value['identification'], 'undefined_no_eligible_target')
        self.assertEqual(value['eligible_target_pairs'], 0)
        self.assertIsNone(value['known_target_sha256'])
        self.assertEqual(value['first_exclusion_counts'], {'edited_validation_unavailable': 1})
        verify.verify_certificate(value)

    def test_unknown_rule_is_retained_not_filled(self):
        row = missing_pair()
        for side in ('original', 'edited'):
            row[side]['detection'].update(rule_id=None, threshold=None, comparator=None)
        value = certificate.certificates_from_raw_pairs([row], 'detection_survival')[0]
        self.assertEqual(value['decision_rule'], dict(rule_id='', threshold=None, comparator=''))
        self.assertEqual(value['first_exclusion_counts'], {'decision_rule_unknown': 1})
        self.assertEqual(value['identification'], 'undefined_no_eligible_target')
        verify.verify_certificate(value)

    def test_every_requested_claim_remains_denied(self):
        for claim in certificate.CLAIMS:
            with self.subTest(claim=claim):
                value = certificate.certificates_from_raw_pairs([missing_pair()], claim)[0]
                self.assertEqual(value['decision'], 'not_admissible')
                self.assertIsNone(value['estimate'])
                verify.verify_certificate(value)

    def test_partial_stratum_keeps_original_complete_cohort_contract(self):
        values = certificate.certificates_from_raw_pairs(
            [fixture('complete'), missing_pair()], 'positive_decision_change')
        self.assertEqual(len(values), 1)
        self.assertEqual(certificate.bundle_schema_for(values), CERTIFICATE_BUNDLE_SCHEMA)
        self.assertEqual(values[0]['counts']['n'], 1)
        self.assertEqual(values[0]['attempted_pairs'], 2)
        self.assertEqual(values[0]['first_exclusion_counts'], {'edited_decision_unavailable': 1})
        verify.verify_certificate(values[0])

    def test_raw_json_jsonl_single_pair_and_report_basis_round_trip(self):
        row = missing_pair()
        cases = {
            'pairs.json': [row],
            'pairs.jsonl': row,
            'pair.json': row,
            'report.json': report([Pair.parse(row)]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, value in cases.items():
                with self.subTest(name=name):
                    basis, output = Path(tmp)/name, Path(tmp)/(name+'.bundle.json')
                    basis.write_text(json.dumps(value)+'\n', encoding='utf-8')
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self.assertEqual(certificate.main([
                            str(basis), '--claim', 'positive_decision_change', '--out', str(output)]), 3)
                    bundle = json.loads(output.read_text())
                    self.assertEqual(bundle['schema_version'], COVERAGE_BUNDLE_SCHEMA)
                    result = verify.verify_bundle_against_basis(bundle, basis)
                    self.assertTrue(result['basis_verified'])
                    self.assertEqual(result['certificates_verified'], 1)
                    self.assertEqual(result['raw_pair_membership_rederived'], name != 'report.json')

    def test_verification_cli_accepts_valid_denial_not_the_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            basis, output = Path(tmp)/'pairs.json', Path(tmp)/'bundle.json'
            verified = Path(tmp)/'verified.json'
            basis.write_text(json.dumps([missing_pair()]), encoding='utf-8')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(certificate.main([
                    str(basis), '--claim', 'positive_decision_change', '--out', str(output)]), 3)
                self.assertEqual(verify.main([
                    str(output), '--basis', str(basis), '--out', str(verified)]), 0)
            self.assertEqual(json.loads(verified.read_text())['certificates'][0]['decision'],
                             'not_admissible')

    def test_require_observations_still_rejects_synthetic_missing_stratum(self):
        with tempfile.TemporaryDirectory() as tmp:
            basis, output = Path(tmp)/'pairs.json', Path(tmp)/'bundle.json'
            basis.write_text(json.dumps([missing_pair()]), encoding='utf-8')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(certificate.main([
                    str(basis), '--claim', 'positive_decision_change', '--out', str(output),
                    '--require-observations']), 2)
            self.assertFalse(output.exists())

    def test_rehashed_false_metric_or_admission_is_rejected(self):
        original = certificate.certificates_from_raw_pairs(
            [missing_pair()], 'positive_decision_change')[0]
        mutations = (
            ('decision', 'admissible'), ('estimate', {'numerator': 0, 'denominator': 1, 'decimal': 0.0}),
            ('counts', {'n': 1, 'before_positive': 0, 'after_positive': 0}),
            ('cohort_sha256', '0'*64), ('membership_verified', True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                edited = copy.deepcopy(original)
                edited[field] = value
                rehash(edited)
                with self.assertRaises(EvidenceError):
                    verify.verify_certificate(edited)

    def test_typed_coverage_and_partitions_are_rederived(self):
        original = certificate.certificates_from_raw_pairs(
            [missing_pair()], 'positive_decision_change')[0]
        for field, value in (
            ('attempted_pairs', True), ('admitted_pairs', False),
            ('eligible_target_pairs', 1.0), ('attempted_pairs', 2),
            ('eligible_target_pairs', 0), ('known_target_sha256', None),
        ):
            with self.subTest(field=field, value=value):
                edited = copy.deepcopy(original)
                edited[field] = value
                rehash(edited)
                with self.assertRaises(EvidenceError):
                    verify.verify_certificate(edited)

    def test_empty_report_transition_contradiction_is_rejected(self):
        value = report([Pair.parse(missing_pair())])
        value['strata'][0]['estimates']['changed_only']['n11'] = 1
        with self.assertRaises(EvidenceError):
            certificate.certificates_from_pair_report(value, 'positive_decision_change')

    def test_v1_bundle_cannot_hide_new_record_type(self):
        values = certificate.certificates_from_raw_pairs([missing_pair()], 'positive_decision_change')
        bundle = {'schema_version': CERTIFICATE_BUNDLE_SCHEMA,
                  'input_file': {'name': 'pairs.json', 'sha256': '0'*64},
                  'requested_claim': 'positive_decision_change', 'certificates': values}
        with self.assertRaisesRegex(EvidenceError, 'v2'):
            verify.verify_bundle(bundle)

    def test_v2_basis_rejects_omitted_stratum(self):
        known, unknown = fixture('known'), missing_pair()
        unknown['method'] = 'missing-method'
        with tempfile.TemporaryDirectory() as tmp:
            basis = Path(tmp)/'pairs.json'
            basis.write_text(json.dumps([known, unknown]), encoding='utf-8')
            values = certificate.certificates_from_raw_pairs([known, unknown], 'positive_decision_change')
            bundle = pair_bundle(basis, values)
            verify.verify_bundle_against_basis(bundle, basis)
            bundle['certificates'] = [v for v in values if v['schema_version'] != UNIDENTIFIED_STRATUM_SCHEMA]
            verify.verify_bundle(bundle)
            with self.assertRaisesRegex(EvidenceError, 'regeneration'):
                verify.verify_bundle_against_basis(bundle, basis)

    def test_legacy_basis_preserves_omitted_stratum_as_explicit_coverage(self):
        known, unknown = fixture('known'), missing_pair()
        unknown['method'] = 'missing-method'
        with tempfile.TemporaryDirectory() as tmp:
            basis = Path(tmp)/'pairs.json'
            basis.write_text(json.dumps([known, unknown]), encoding='utf-8')
            values = certificate.certificates_from_raw_pairs([known, unknown], 'positive_decision_change')
            bundle = pair_bundle(basis, values)
            omitted = [v for v in values if v['schema_version'] == UNIDENTIFIED_STRATUM_SCHEMA]
            bundle['certificates'] = [v for v in values if v['schema_version'] != UNIDENTIFIED_STRATUM_SCHEMA]
            bundle['schema_version'] = CERTIFICATE_BUNDLE_SCHEMA
            result = verify.verify_bundle_against_basis(bundle, basis)
            self.assertTrue(result['basis_verified'])
            self.assertFalse(result['basis_coverage_complete'])
            self.assertEqual(result['unidentified_strata'], omitted)

    def test_complete_legacy_basis_remains_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            basis = Path(tmp)/'pairs.json'
            basis.write_text(json.dumps([fixture()]), encoding='utf-8')
            values = certificate.certificates_from_raw_pairs([fixture()], 'positive_decision_change')
            result = verify.verify_bundle_against_basis(pair_bundle(basis, values), basis)
            self.assertTrue(result['basis_coverage_complete'])
            self.assertEqual(result['unidentified_strata'], [])

    def test_basis_detects_substituted_target_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            basis = Path(tmp)/'pairs.json'
            basis.write_text(json.dumps([missing_pair()]), encoding='utf-8')
            values = certificate.certificates_from_raw_pairs([missing_pair()], 'positive_decision_change')
            bundle = pair_bundle(basis, values)
            values[0]['known_target_sha256'] = '0'*64
            rehash(values[0])
            verify.verify_bundle(bundle)  # Embedded identities are assertions, not authentication.
            with self.assertRaisesRegex(EvidenceError, 'regeneration'):
                verify.verify_bundle_against_basis(bundle, basis)


if __name__ == '__main__':
    unittest.main()
