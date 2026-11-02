"""Synthetic acceptance-boundary regressions; never execute source programs."""
import copy
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editmark_audit import aggregates, certificate, verify
from editmark_audit.analysis import report
from editmark_audit.constants import CERTIFICATE_BUNDLE_SCHEMA, COVERAGE_BUNDLE_SCHEMA
from editmark_audit.io import EvidenceError, digest, load_rows, loads
from editmark_audit.records import Pair
from tests.contract.test_contract import fixture

ROOT = Path(__file__).resolve().parents[2]
CLAIM = 'positive_decision_change'


def basis():
    rows = []
    for i in range(3):
        raw = fixture(str(i)); raw['edit'] = 'edit' + str(i)
        if i:
            raw['edited']['detection'].update(available=False, detected=None, score=None)
        rows.append(Pair.parse(raw))
    return report(rows)


class PairReportConsistencyTests(unittest.TestCase):
    def reject_both_routes(self, value):
        with self.assertRaises(EvidenceError):
            certificate.certificates_from_pair_report(value, CLAIM)
        with patch.object(verify, '_load_input', return_value=('pair_report', value)):
            with self.assertRaises(EvidenceError):
                verify._regenerate_from_basis(Path('unused-memory-report.json'), CLAIM)

    def test_deleted_stratum_with_unchanged_totals_rejects(self):
        value = basis(); value['strata'].pop()
        self.reject_both_routes(value)

    def test_declared_attempted_coverage_must_match_strata(self):
        value = basis(); value.update(input_pair_count=4, unique_pair_count=4)
        self.reject_both_routes(value)

    def test_duplicate_stratum_rejects_even_with_adjusted_totals(self):
        value = basis(); value['strata'].append(copy.deepcopy(value['strata'][0]))
        value.update(stratum_count=4, input_pair_count=4, unique_pair_count=4)
        self.reject_both_routes(value)

    def test_report_schema_types_and_required_fields(self):
        for field, replacement in (
            ('schema_version', 'legacy'), ('strata', {}), ('stratum_count', True),
            ('unique_pair_count', 3.0), ('input_pair_count', '3'),
            ('identical_duplicates_removed', False), ('evidence_kind', 'empty'),
            ('synthetic_warning', None), ('policy', []), ('unexpected', 1),
        ):
            value = basis(); value[field] = replacement
            with self.subTest(field=field): self.reject_both_routes(value)
        value = basis(); del value['stratum_count']; self.reject_both_routes(value)

    def test_stratum_header_and_estimate_schema_types(self):
        for field, replacement in (('method', None), ('original_threshold', True),
                                   ('original_comparator', 'eq'), ('estimates', [])):
            value = basis(); value['strata'][0][field] = replacement
            with self.subTest(field=field): self.reject_both_routes(value)
        for field in ('status', 'net_detection_change_exact', 'marginal_coverage'):
            value = basis(); del value['strata'][0]['estimates']['changed_only'][field]
            with self.subTest(missing=field): self.reject_both_routes(value)

    def test_support_marginals_cannot_conflict_with_first_exclusion(self):
        a, b = fixture('a'), fixture('b'); b['supported'] = False
        value = report([Pair.parse(a), Pair.parse(b)])
        value['strata'][0]['estimates']['changed_only']['marginal_coverage']['supported'] = 2
        self.reject_both_routes(value)
        value = report([Pair.parse(a), Pair.parse(b)])
        e = value['strata'][0]['estimates']['changed_only']
        e['first_exclusion_counts'] = {'edited_checks_failed': 1}
        self.reject_both_routes(value)

    def test_status_and_authoritative_derivatives_are_rederived(self):
        changes = (
            ('status', 'undefined_empty_cohort'), ('net_detection_change', 1.0),
            ('net_detection_change_exact', dict(numerator=1, denominator=1, decimal=1.0)),
            ('tpr_before', False), ('tpr_after', 0), ('net_detection_loss', True),
            ('loss', 0.0), ('survival', False), ('survival_denominator', True),
            ('n10', True), ('admitted_pairs', 1.0), ('tpr_after', float('nan')),
        )
        for field, replacement in changes:
            value = basis(); value['strata'][0]['estimates']['changed_only'][field] = replacement
            with self.subTest(field=field): self.reject_both_routes(value)

    def test_nested_rates_bounds_and_fingerprints_are_checked(self):
        changes = (
            ('functional_preservation', 'rate', True),
            ('distinct_originals', 'pass_rate', 1),
            ('distinct_originals', 'passing', 0),
            ('missing_decision_bounds', 'denominator', True),
            ('missing_decision_change_bounds', 'estimand', 'net_detection_loss'),
            ('missing_decision_change_bounds', 'upper', 1.0),
        )
        for group, field, replacement in changes:
            value = basis(); value['strata'][0]['estimates']['changed_only'][group][field] = replacement
            with self.subTest(group=group, field=field): self.reject_both_routes(value)
        value = basis(); value['strata'][0]['estimates']['changed_only']['known_target_sha256'] = 'a' * 64
        self.reject_both_routes(value)
        a, b = fixture('a'), fixture('b')
        b['edited']['detection']['rule_id'] = 'changed-rule'
        value = report([Pair.parse(a), Pair.parse(b)])
        value['strata'][0]['estimates']['changed_only']['functional_preservation'].update(numerator=1, rate=0.5)
        self.reject_both_routes(value)

    def test_both_bundle_versions_reject_inconsistent_report_basis(self):
        original = basis()
        values = certificate.certificates_from_pair_report(original, CLAIM)
        bad = copy.deepcopy(original); bad['strata'].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'; path.write_text(json.dumps(bad), encoding='utf-8')
            for schema in (CERTIFICATE_BUNDLE_SCHEMA, COVERAGE_BUNDLE_SCHEMA):
                bundle = dict(schema_version=schema, input_file=dict(name=path.name, sha256=digest(path)),
                              requested_claim=CLAIM, certificates=[values[0]])
                with self.subTest(schema=schema), self.assertRaises(EvidenceError):
                    verify.verify_bundle_against_basis(bundle, path)
            out = Path(directory) / 'must-not-exist.json'
            with contextlib.redirect_stderr(io.StringIO()):
                status = certificate.main([str(path), '--claim', CLAIM, '--out', str(out)])
            self.assertEqual(status, 2); self.assertFalse(out.exists())

    def test_valid_duplicates_in_raw_rows_and_missing_denials_are_preserved(self):
        raw = fixture('one')
        result = report([Pair.parse(raw), Pair.parse(raw)])
        aggregates.validate_pair_report(result)
        self.assertEqual((result['input_pair_count'], result['unique_pair_count']), (2, 1))
        values = certificate.certificates_from_pair_report(basis(), CLAIM)
        self.assertEqual([v['decision'] for v in values], ['admissible', 'not_admissible', 'not_admissible'])
        for value in values: verify.verify_certificate(value)

    def test_retained_certificates_keep_exact_content(self):
        claims = certificate.CLAIMS
        for directory in ('semcodebook_cpu', 'owned_finite_carrier'):
            root = ROOT / 'results' / directory
            raw = load_rows(root / 'pairs.jsonl')
            for claim in claims:
                saved = loads((root / (claim + '.json')).read_text(encoding='utf-8'))
                current = certificate.certificates_from_raw_pairs(raw, claim)
                with self.subTest(directory=directory, claim=claim):
                    self.assertEqual(current, saved['certificates'])
                    self.assertTrue(verify.verify_bundle_against_basis(saved, root / 'pairs.jsonl')['basis_verified'])

    def test_retained_owned_source_inventory_requires_exact_live_or_preimage_bytes(self):
        saved_root = ROOT / 'results/owned_finite_carrier'
        saved = loads((saved_root / (CLAIM + '.json')).read_text(encoding='utf-8'))
        bindings = loads((saved_root / 'measurement.json').read_text(encoding='utf-8'))['source_bindings']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / 'results/owned_finite_carrier'
            cohort.mkdir(parents=True)
            for name in ('pairs.jsonl', 'measurement.json'):
                (cohort / name).write_bytes((saved_root / name).read_bytes())
            originals = {}
            for binding in bindings:
                path, expected = binding['path'], binding['sha256']
                source = ROOT / path
                if not source.is_file() or digest(source) != expected:
                    source = saved_root / 'source-preimage' / path
                self.assertEqual(digest(source), expected)
                originals[path] = source.read_bytes()
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(originals[path])
            basis_path = cohort / 'pairs.jsonl'
            bundle_path = root / 'bundle.json'
            bundle_path.write_text(json.dumps(saved), encoding='utf-8')
            rejected_output = root / 'must-not-publish.json'
            # These copied .py files are bytes only, never imported or executed.
            self.assertTrue(verify.verify_bundle_against_basis(saved, basis_path)['basis_verified'])
            for binding in bindings:
                path = binding['path']
                live, preimage = root / path, cohort / 'source-preimage' / path
                with self.subTest(path=path):
                    live.unlink()
                    with self.assertRaisesRegex(EvidenceError, 'source missing or SHA-256 mismatch'):
                        verify.verify_bundle_against_basis(saved, basis_path)
                    preimage.parent.mkdir(parents=True, exist_ok=True)
                    preimage.write_bytes(originals[path])
                    self.assertTrue(verify.verify_bundle_against_basis(saved, basis_path)['basis_verified'])
                    lf = originals[path].replace(b'\r\n', b'\n')
                    changed_endings = lf if lf != originals[path] else lf.replace(b'\n', b'\r\n')
                    self.assertNotEqual(changed_endings, originals[path])
                    live.write_bytes(changed_endings)
                    self.assertTrue(verify.verify_bundle_against_basis(saved, basis_path)['basis_verified'])
                    preimage.write_bytes(changed_endings)
                    with self.assertRaisesRegex(EvidenceError, 'source missing or SHA-256 mismatch'):
                        verify.verify_bundle_against_basis(saved, basis_path)
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(verify.main([str(bundle_path), '--basis', str(basis_path),
                                                      '--out', str(rejected_output)]), 2)
                    self.assertFalse(rejected_output.exists())
                    live.write_bytes(originals[path])
                    self.assertTrue(verify.verify_bundle_against_basis(saved, basis_path)['basis_verified'])
                    preimage.unlink()
            measurement = cohort / 'measurement.json'
            measurement.write_bytes(measurement.read_bytes() + b'\n')
            with self.assertRaisesRegex(EvidenceError, 'measurement SHA-256'):
                verify.verify_bundle_against_basis(saved, basis_path)
            measurement.unlink()
            with self.assertRaises(FileNotFoundError):
                verify.verify_bundle_against_basis(saved, basis_path)

    def test_retained_source_binding_paths_and_hashes_are_strict(self):
        good = dict(path='editmark_audit/verify.py', sha256='a' * 64)
        self.assertEqual(verify._source_binding(good), (good['path'], good['sha256']))
        for path in ('', '.', './x.py', '../x.py', 'a/../x.py', '/x.py',
                     'a//x.py', 'a/./x.py', 'a/', 'C:/x.py', 'C:x.py',
                     '\\\\host\\share\\x.py', 'a\\x.py', 'a:x.py', ' x.py'):
            with self.subTest(path=path), self.assertRaises(EvidenceError):
                verify._source_binding(dict(good, path=path))
        for value in (None, {}, [], dict(good, extra=True), dict(good, path=False),
                      dict(good, sha256=None), dict(good, sha256='A' * 64),
                      dict(good, sha256='a' * 63), dict(good, sha256='z' * 64)):
            with self.subTest(binding=value), self.assertRaises(EvidenceError):
                verify._source_binding(value)

    def test_finite_valid_gate_reports_remain_accepted(self):
        # 2,187 Boolean/null disclosures, not executions or watermark samples.
        for support, change, before_pass, after_pass, before, after, rule in itertools.product(
                (None, False, True), repeat=7):
            raw = fixture(before=before, after=after); raw['supported'] = support
            if change is False:
                raw['edited']['source'] = raw['original']['source']; raw['changed'] = False
            elif change is None:
                for side in ('original', 'edited'): del raw[side]['source']
                raw['changed'] = None
            for side, passed, decision in (('original', before_pass, before), ('edited', after_pass, after)):
                raw[side]['validation'].update(available=passed is not None, passed=passed)
                if decision is None:
                    raw[side]['detection'].update(available=False, detected=None, score=None)
                if rule is None:
                    raw[side]['detection']['rule_id'] = None
            if rule is False: raw['edited']['detection']['rule_id'] = 'other-rule'
            aggregates.validate_pair_report(report([Pair.parse(raw)]))


if __name__ == '__main__':
    unittest.main()
