"""Synthetic acceptance-boundary regressions; never execute source programs."""
import copy
import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
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


def original_check_basis(reason):
    """Two synthetic identities: one admitted, one stopped at the named gate."""
    raw = fixture('excluded')
    side, field, changes = {
        'original_validation_unavailable': (
            'original', 'validation', dict(available=False, passed=None)),
        'original_checks_unknown': ('original', 'validation', dict(passed=None)),
        'original_checks_failed': ('original', 'validation', dict(passed=False)),
        'edited_validation_unavailable': (
            'edited', 'validation', dict(available=False, passed=None)),
        'edited_checks_failed': ('edited', 'validation', dict(passed=False)),
        'edited_checks_unknown': ('edited', 'validation', dict(passed=None)),
        'check_contract_unknown': ('edited', 'validation', dict(contract_id=None)),
        'check_contract_changed': ('edited', 'validation', dict(contract_id='other')),
        'decision_rule_unknown': ('edited', 'detection', dict(rule_id=None)),
        'decision_rule_changed': ('edited', 'detection', dict(rule_id='other')),
    }[reason]
    raw[side][field].update(changes)
    return report([Pair.parse(fixture('admitted')), Pair.parse(raw)])


class PairReportConsistencyTests(unittest.TestCase):
    def run_cli(self, module, *args):
        return subprocess.run(
            [sys.executable, '-B', '-m', module, *map(str, args)],
            cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
            capture_output=True, text=True, timeout=30)

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

        # Mirror the same impossible interval into every mode: cross-mode
        # differences alone cannot see a shared within-mode contradiction.
        for nonempty in (False, True):
            missing = fixture('missing', before=True, after=True)
            missing['edited']['detection'].update(available=False, detected=None, score=None)
            rows = [Pair.parse(missing)]
            if nonempty:
                rows.append(Pair.parse(fixture('complete', before=True, after=False)))
            valid = report(rows)
            aggregates.validate_pair_report(valid)
            certificates = certificate.certificates_from_pair_report(valid, CLAIM)
            bad = copy.deepcopy(valid)
            for stratum in bad['strata']:
                for estimate in stratum['estimates'].values():
                    loss = estimate['missing_decision_bounds']
                    complete_loss = estimate['n10'] - estimate['n01']
                    low, high = complete_loss - 1, complete_loss + 1
                    denominator = loss['denominator']
                    loss.update(lower_numerator=low, upper_numerator=high,
                                lower=low / denominator, upper=high / denominator)
                    estimate['missing_decision_change_bounds'].update(
                        lower_numerator=-high, upper_numerator=-low,
                        lower=-high / denominator, upper=-low / denominator)
            with self.subTest(nonempty=nonempty):
                self.reject_both_routes(bad)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory);basis_path = root / 'report.json'
                    basis_path.write_text(json.dumps(bad), encoding='utf-8')
                    for schema in (CERTIFICATE_BUNDLE_SCHEMA, COVERAGE_BUNDLE_SCHEMA):
                        bundle = dict(schema_version=schema,
                                      input_file=dict(name=basis_path.name, sha256=digest(basis_path)),
                                      requested_claim=CLAIM, certificates=certificates)
                        source = root / ('bundle-' + schema + '.json')
                        source.write_text(json.dumps(bundle), encoding='utf-8')
                        output = root / 'must-not-publish.json'
                        with contextlib.redirect_stderr(io.StringIO()):
                            self.assertEqual(verify.main([str(source), '--basis', str(basis_path),
                                                          '--out', str(output)]), 2)
                        self.assertFalse(output.exists())
                    output = root / 'must-not-produce.json'
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(certificate.main([str(basis_path), '--claim', CLAIM,
                                                            '--out', str(output)]), 2)
                    self.assertFalse(output.exists())
        both = fixture('both', before=True, after=True)
        for side in ('original', 'edited'):
            both[side]['detection'].update(available=False, detected=None, score=None)
        valid = report([Pair.parse(both)])
        aggregates.validate_pair_report(valid)
        bounds = valid['strata'][0]['estimates']['changed_only']['missing_decision_bounds']
        self.assertEqual(bounds['upper_numerator'] - bounds['lower_numerator'], 2)

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

    def test_later_exclusions_require_passing_original_checks_in_every_mode(self):
        reasons = (
            'edited_validation_unavailable', 'edited_checks_failed', 'edited_checks_unknown',
            'check_contract_unknown', 'check_contract_changed',
            'decision_rule_unknown', 'decision_rule_changed',
        )
        for reason in reasons:
            valid = original_check_basis(reason)
            aggregates.validate_pair_report(valid)
            for estimate in valid['strata'][0]['estimates'].values():
                self.assertEqual(estimate['first_exclusion_counts'], {reason: 1})
                self.assertEqual(estimate['distinct_originals'], dict(
                    count=2, known_check_results=2, passing=2, pass_rate=1.0))
            # Mirror the contradiction in all modes so nesting is not the gate
            # that rejects it. Exercise missing and falsely failed disclosures.
            for known in (1, 2):
                bad = copy.deepcopy(valid)
                for mode, estimate in bad['strata'][0]['estimates'].items():
                    estimate['distinct_originals'].update(
                        known_check_results=known, passing=1, pass_rate=1 / known)
                    with self.subTest(reason=reason, known=known, mode=mode):
                        with self.assertRaisesRegex(EvidenceError, 'original-check'):
                            aggregates._pair_estimate(estimate, mode)
                with self.subTest(reason=reason, known=known):
                    with self.assertRaises(EvidenceError):
                        aggregates.validate_pair_report(bad)
                    self.reject_both_routes(bad)

    def test_original_checks_missing_unknown_or_failed_remain_excluded(self):
        for reason in ('original_validation_unavailable', 'original_checks_unknown',
                       'original_checks_failed'):
            value = original_check_basis(reason)
            aggregates.validate_pair_report(value)
            known = 2 if reason == 'original_checks_failed' else 1
            for estimate in value['strata'][0]['estimates'].values():
                self.assertEqual(estimate['first_exclusion_counts'], {reason: 1})
                self.assertEqual(estimate['distinct_originals'], dict(
                    count=2, known_check_results=known, passing=1, pass_rate=1 / known))
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'synthetic-report.json'
                path.write_text(json.dumps(value), encoding='utf-8')
                bundle, result = Path(directory) / 'bundle.json', Path(directory) / 'verify.json'
                created = self.run_cli('editmark_audit.certificate', path, '--claim', CLAIM,
                                       '--out', bundle)
                self.assertEqual(created.returncode, 0, created.stderr)
                checked = self.run_cli('editmark_audit.verify', bundle, '--basis', path,
                                       '--out', result)
                self.assertEqual(checked.returncode, 0, checked.stderr)
                self.assertTrue(loads(result.read_text(encoding='utf-8'))['basis_verified'])

    def test_original_check_contradictions_reject_real_cli_and_both_basis_bundles(self):
        valid = original_check_basis('edited_validation_unavailable')
        certificates = certificate.certificates_from_pair_report(valid, CLAIM)
        for known in (1, 2):
            bad = copy.deepcopy(valid)
            for estimate in bad['strata'][0]['estimates'].values():
                estimate['distinct_originals'].update(
                    known_check_results=known, passing=1, pass_rate=1 / known)
            with self.subTest(known=known), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / 'synthetic-inconsistent-report.json'
                path.write_text(json.dumps(bad), encoding='utf-8')
                output = root / 'must-not-certify.json'
                created = self.run_cli('editmark_audit.certificate', path, '--claim', CLAIM,
                                       '--out', output)
                self.assertEqual(created.returncode, 2, created.stderr)
                self.assertIn('original-check', created.stderr)
                self.assertFalse(output.exists())
                for schema in (CERTIFICATE_BUNDLE_SCHEMA, COVERAGE_BUNDLE_SCHEMA):
                    bundle = dict(schema_version=schema,
                                  input_file=dict(name=path.name, sha256=digest(path)),
                                  requested_claim=CLAIM, certificates=certificates)
                    # A matching input digest must not hide invalid report facts.
                    self.assertEqual(verify.verify_bundle(bundle)['certificates_verified'], 1)
                    with self.assertRaises(EvidenceError):
                        verify.verify_bundle_against_basis(bundle, path)
                    source, result = root / (schema + '.json'), root / 'must-not-verify.json'
                    source.write_text(json.dumps(bundle), encoding='utf-8')
                    checked = self.run_cli('editmark_audit.verify', source, '--basis', path,
                                           '--out', result)
                    self.assertEqual(checked.returncode, 2, checked.stderr)
                    self.assertIn('original-check', checked.stderr)
                    self.assertFalse(result.exists())

    def test_reused_originals_are_counted_per_stratum_not_per_report_or_hash(self):
        raw = fixture('shared')
        rows = [Pair.parse(raw), Pair.parse(raw)]  # An identical duplicate is removed.
        for field, replacement in (('edit', 'another'), ('variant_id', 'v1'),
                                   ('replicate_id', 'r1')):
            reused = copy.deepcopy(raw); reused[field] = replacement
            rows.append(Pair.parse(reused))
        self.assertEqual(len({pair.original_key for pair in rows}), 2)
        self.assertEqual(len({pair.original.source_sha256 for pair in rows}), 1)
        value = report(rows)
        aggregates.validate_pair_report(value)
        self.assertEqual((value['input_pair_count'], value['unique_pair_count'],
                          value['stratum_count']), (5, 4, 3))
        for stratum in value['strata']:
            for estimate in stratum['estimates'].values():
                self.assertEqual(estimate['distinct_originals']['count'],
                                 estimate['attempted_pairs'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'synthetic-reuse-report.json'
            path.write_text(json.dumps(value), encoding='utf-8')
            bundle, result = root / 'bundle.json', root / 'verify.json'
            created = self.run_cli('editmark_audit.certificate', path, '--claim', CLAIM,
                                   '--out', bundle)
            self.assertEqual(created.returncode, 0, created.stderr)
            checked = self.run_cli('editmark_audit.verify', bundle, '--basis', path,
                                   '--out', result)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            verified = loads(result.read_text(encoding='utf-8'))
            self.assertTrue(verified['basis_verified'])
            self.assertEqual(verified['certificates_verified'], 3)

    def test_retained_certificates_keep_exact_content(self):
        claims = certificate.CLAIMS
        for directory in ('semcodebook_cpu', 'owned_finite_carrier'):
            root = ROOT / 'results' / directory
            raw = load_rows(root / 'pairs.jsonl')
            retained_report = loads((root / 'pair-report.json').read_text(encoding='utf-8'))
            aggregates.validate_pair_report(retained_report)
            for claim in claims:
                saved = loads((root / (claim + '.json')).read_text(encoding='utf-8'))
                current = certificate.certificates_from_raw_pairs(raw, claim)
                with self.subTest(directory=directory, claim=claim):
                    self.assertEqual(current, saved['certificates'])
                    self.assertEqual(certificate.certificates_from_pair_report(retained_report, claim),
                                     saved['certificates'])
                    self.assertTrue(verify.verify_bundle_against_basis(saved, root / 'pairs.jsonl')['basis_verified'])

    def test_retained_owned_source_inventory_requires_exact_live_or_preimage_bytes(self):
        saved_root = ROOT / 'results/owned_finite_carrier'
        saved = loads((saved_root / (CLAIM + '.json')).read_text(encoding='utf-8'))
        bindings = loads((saved_root / 'measurement.json').read_text(encoding='utf-8'))['source_bindings']
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            private = Path(directory).resolve()
            root = private / 'artifact'
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

            @contextlib.contextmanager
            def directory_alias(link, target):
                # Real links, confined to this test's private temporary tree.
                self.assertTrue(link.absolute().is_relative_to(private))
                self.assertTrue(target.resolve().is_relative_to(private))
                if os.name == 'nt':
                    import _winapi
                    _winapi.CreateJunction(str(target), str(link))
                else:
                    link.symlink_to(target, target_is_directory=True)
                try:
                    yield link
                finally:
                    if os.name == 'nt':
                        link.rmdir()  # Remove the junction, never its target.
                    else:
                        link.unlink()

            aliases = {
                'canonical': basis_path,
                'parent-traversal': cohort / '../owned_finite_carrier/pairs.jsonl',
                'cohort-link': stack.enter_context(directory_alias(private / 'alias', cohort)) / 'pairs.jsonl',
                'root-link': stack.enter_context(directory_alias(private / 'root-alias', root))
                             / 'results/owned_finite_carrier/pairs.jsonl',
            }
            for label, alias in (
                ('root-hardlink', root / 'linked-basis.jsonl'),
                ('cohort-hardlink-renamed', cohort / 'linked-basis.data'),
            ):
                os.link(basis_path, alias)
                stack.callback(alias.unlink)
                aliases[label] = alias
            if os.name == 'nt':
                aliases.update({
                    'file-case': cohort / 'PAIRS.JSONL',
                    'directory-case': root / 'RESULTS/OWNED_FINITE_CARRIER/pairs.jsonl',
                    'trailing-dot': cohort / 'pairs.jsonl.',
                    'trailing-space': cohort / 'pairs.jsonl ',
                })
                import ctypes
                short_name = ctypes.windll.kernel32.GetShortPathNameW
                short_name.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
                short_name.restype = ctypes.c_uint32
                buffer = ctypes.create_unicode_buffer(32768)
                length = short_name(str(basis_path), buffer, len(buffer))
                self.assertGreater(length, 0)
                self.assertLess(length, len(buffer))
                if buffer.value != str(basis_path):
                    aliases['windows-short-name'] = Path(buffer.value)
            # File symlinks may need a Windows privilege; junctions are mandatory.
            # A skipped subcase is visible and does not skip the remaining checks.
            with self.subTest(link='file-symlink-alias-and-escapes'):
                file_alias = private / 'linked-pairs.data'
                try:
                    file_alias.symlink_to(basis_path)
                except OSError as exc:
                    if os.name != 'nt' or exc.winerror != 1314:
                        raise
                    self.skipTest('file symlinks unavailable (Windows privilege 1314); junction cases run')
                stack.callback(file_alias.unlink)
                aliases['file-link'] = file_alias
            for label, alias in aliases.items():
                with self.subTest(alias=label):
                    self.assertTrue(alias.samefile(basis_path))

            def check(path, error=None):
                if error is None:
                    result = (verify.verify_bundle(saved) if path is None else
                              verify.verify_bundle_against_basis(saved, path))
                    self.assertEqual(result['basis_verified'], path is not None)
                    self.assertEqual([row['decision'] for row in result['certificates']],
                                     [row['decision'] for row in saved['certificates']])
                    if path is not None:
                        self.assertTrue(result['raw_pair_membership_rederived'])
                        self.assertTrue(result['basis_coverage_complete'])
                else:
                    with self.assertRaisesRegex((EvidenceError, OSError), error):
                        verify.verify_bundle_against_basis(saved, path)
                args = [str(bundle_path), '--out', str(rejected_output)]
                if path is not None:
                    args += ['--basis', str(path)]
                # Exercise the public CLI with only publication mocked, not a gate.
                with patch.object(verify, 'write_new_json') as publish, \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(verify.main(args), 0 if error is None else 2)
                if error is None:
                    publish.assert_called_once_with(rejected_output, result)
                else:
                    publish.assert_not_called()
                self.assertFalse(rejected_output.exists())

            def check_aliases(state, error=None):
                for label, alias in aliases.items():
                    with self.subTest(state=state, alias=label):
                        check(alias, error)

            # These copied .py files are bytes only, never imported or executed.
            check_aliases('exact-live')
            for binding in bindings:
                path = binding['path']
                live, preimage = root / path, cohort / 'source-preimage' / path
                with self.subTest(path=path):
                    live.unlink()
                    check_aliases('missing-source', 'source missing or SHA-256 mismatch')
                    preimage.parent.mkdir(parents=True, exist_ok=True)
                    preimage.write_bytes(originals[path])
                    check_aliases('exact-preimage')
                    lf = originals[path].replace(b'\r\n', b'\n')
                    changed_endings = lf if lf != originals[path] else lf.replace(b'\n', b'\r\n')
                    self.assertNotEqual(changed_endings, originals[path])
                    live.write_bytes(changed_endings)
                    check_aliases('changed-live-exact-preimage')
                    preimage.write_bytes(changed_endings)
                    check_aliases('both-changed', 'source missing or SHA-256 mismatch')
                    live.write_bytes(originals[path])
                    check_aliases('exact-live-recovery')
                    preimage.unlink()
            measurement = cohort / 'measurement.json'
            measurement.write_bytes(measurement.read_bytes() + b'\n')
            check_aliases('changed-measurement', 'measurement SHA-256')
            measurement.unlink()
            check_aliases('missing-measurement', 'measurement.json')

            # An unclassifiable hard link must not silently become generic.
            # This external sibling has no retained anchor among its ancestors.
            unknown_link = private / 'unknown-hardlink.jsonl'
            os.link(basis_path, unknown_link)
            stack.callback(unknown_link.unlink)
            self.assertTrue(unknown_link.samefile(basis_path))
            check(unknown_link, 'multiply-linked basis')

            # Copies/renames are generic inputs, not aliases of the owned path.
            generic = private / 'generic/pairs.jsonl'
            generic.parent.mkdir()
            for path in (generic, cohort / 'renamed.jsonl'):
                path.write_bytes(basis_path.read_bytes())
                with self.subTest(generic=path.name):
                    self.assertFalse(path.samefile(basis_path))
                    check(path)
            if os.name != 'nt':
                path = root / 'RESULTS/OWNED_FINITE_CARRIER/PAIRS.JSONL'
                path.parent.mkdir(parents=True)
                path.write_bytes(basis_path.read_bytes())
                with self.subTest(generic='case-distinct-on-posix'):
                    check(path)
            check(None)  # Standalone still makes no basis-verification claim.
            measurement.write_bytes((saved_root / 'measurement.json').read_bytes())

            # A lexically owned path must not become generic on resolution.
            outside = private / 'outside'
            outside.mkdir()
            (outside / 'pairs.jsonl').write_bytes(basis_path.read_bytes())
            outside_results = private / 'outside-results'
            (outside_results / 'owned_finite_carrier').mkdir(parents=True)
            (outside_results / 'owned_finite_carrier/pairs.jsonl').write_bytes(basis_path.read_bytes())
            for label, link, target, relative in (
                ('cohort-escape', private / 'escape/results/owned_finite_carrier', outside, 'pairs.jsonl'),
                ('results-escape', private / 'escape-results/results', outside_results,
                 'owned_finite_carrier/pairs.jsonl'),
                ('other-owned-cohort', private / 'escape-owned/results/owned_finite_carrier',
                 cohort, 'pairs.jsonl'),
            ):
                link.parent.mkdir(parents=True, exist_ok=True)
                with directory_alias(link, target), self.subTest(escape=label):
                    canonical = link / relative
                    for path in (canonical, canonical.parent / '../owned_finite_carrier/pairs.jsonl'):
                        check(path, 'retained basis path leaves its cohort')

            # Exact bytes outside either source inventory root are insufficient.
            source_path = bindings[0]['path']
            live = root / source_path
            preimage = cohort / 'source-preimage' / source_path
            for label, candidate in (('live', live), ('preimage', preimage)):
                live.unlink()
                target = outside / label
                target.mkdir()
                (target / candidate.name).write_bytes(originals[source_path])
                link = candidate.parent
                holding = None
                if link.exists():
                    holding = link.with_name(link.name + '-held')
                    link.rename(holding)
                else:
                    link.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with directory_alias(link, target), self.subTest(inventory_escape=label):
                        check_aliases(label + '-directory-escape', 'source path leaves its inventory root')
                finally:
                    if holding is not None:
                        holding.rename(link)
                live.parent.mkdir(parents=True, exist_ok=True)
                live.write_bytes(originals[source_path])

            if 'file-link' in aliases:
                for label, path, target in (
                    ('basis', basis_path, outside / 'pairs.jsonl'),
                    ('measurement', measurement, outside / 'measurement.json'),
                    ('live', live, outside / 'source.py'),
                    ('preimage', preimage, outside / 'preimage.py'),
                ):
                    original = path.read_bytes() if path.exists() else originals[source_path]
                    target.write_bytes(original)
                    path.unlink(missing_ok=True)
                    if label == 'preimage':
                        live.unlink()
                        path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        path.symlink_to(target)
                        with self.subTest(file_escape=label):
                            check(basis_path, 'path leaves its')
                    finally:
                        path.unlink(missing_ok=True)
                        if label != 'preimage':
                            path.write_bytes(original)
                        else:
                            live.write_bytes(originals[source_path])

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
