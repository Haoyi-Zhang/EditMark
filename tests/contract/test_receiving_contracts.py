"""Public receiving-contract regressions on authored data, never program execution."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from editmark_audit import certificate, verify
from editmark_audit.analysis import report
from editmark_audit.io import EvidenceError, digest
from editmark_audit.records import Pair
from scripts.audit_frozen_evidence import audit
from scripts.build_contract_artifacts import pair_fixture

ROOT = Path(__file__).resolve().parents[2]
CLAIM = 'positive_decision_change'


def pair_report(rows=None):
    return report([Pair.parse(r) for r in (rows if rows is not None else [pair_fixture()])])


def rehash(value):
    value['certificate_sha256'] = certificate.canonical_hash(
        {k: v for k, v in value.items() if k != 'certificate_sha256'})
    return value


def bundle(records, basis=None):
    return dict(schema_version=certificate.bundle_schema_for(records),
                input_file=dict(name=basis.name if basis else 'authored.json',
                                sha256=digest(basis) if basis else 'a' * 64),
                requested_claim=CLAIM, certificates=records)


class ReceivingContractTests(unittest.TestCase):
    def reject_report(self, value):
        with self.assertRaises(EvidenceError):
            certificate.certificates_from_pair_report(value, CLAIM)
        # Use actual serialization and both public entrypoints, not a mocked gate.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'report.json'
            path.write_text(json.dumps(value), encoding='utf-8')
            cert_out, verified_out = Path(td) / 'cert.json', Path(td) / 'verified.json'
            certs = certificate.certificates_from_pair_report(pair_report(), CLAIM)
            cert_path = Path(td) / 'bundle.json'
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(certificate.main([
                    str(path), '--claim', CLAIM, '--out', str(cert_out)]), 2)
                for schema in ('editmark-claim-certificate-bundle',
                               'editmark-claim-certificate-bundle-v2'):
                    supplied = dict(bundle(certs, path), schema_version=schema)
                    cert_path.write_text(json.dumps(supplied), encoding='utf-8')
                    self.assertEqual(verify.main([
                        str(cert_path), '--basis', str(path), '--out', str(verified_out)]), 2)
            self.assertFalse(cert_out.exists())
            self.assertFalse(verified_out.exists())

    def test_public_pair_report_chain_and_strict_optional_input_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            raw, result, cert = (Path(td) / n for n in ('pairs.json', 'report.json', 'cert.json'))
            raw.write_text(json.dumps([pair_fixture()]), encoding='utf-8')
            for args in (
                ['-m', 'editmark_audit', str(raw), '--out', str(result)],
                ['-m', 'editmark_audit.certificate', str(result), '--claim', CLAIM, '--out', str(cert)],
                ['-m', 'editmark_audit.verify', str(cert), '--basis', str(result)],
            ):
                done = subprocess.run([sys.executable, *([] if __debug__ else ['-O']), '-B', *args], cwd=ROOT,
                                      capture_output=True, text=True, timeout=30)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            produced = json.loads(result.read_text(encoding='utf-8'))
            self.assertEqual(produced['input_files'], [{'name': raw.name, 'sha256': digest(raw)}])
            self.assertEqual(certificate.certificates_from_pair_report(produced, CLAIM),
                             certificate.certificates_from_pair_report(pair_report(), CLAIM))
        good = {'name': 'authored.json', 'sha256': 'a' * 64}
        for value in (None, {}, [], [None], [dict(good, extra=0)],
                      [{'name': '', 'sha256': 'a' * 64}],
                      [{'name': False, 'sha256': 'a' * 64}],
                      [{'name': '\ud800', 'sha256': 'a' * 64}],
                      [{'name': 'authored.json'}],
                      [dict(good, sha256='A' * 64)],
                      [dict(good, sha256='a' * 63)], [dict(good, sha256=1)]):
            bad = pair_report(); bad['input_files'] = value
            with self.subTest(input_files=value):
                self.reject_report(bad)
        bad = pair_report(); bad['input_filez'] = [good]
        self.reject_report(bad)

    def test_modes_cannot_rewrite_observations_when_relaxing_eligibility(self):
        changes = []
        missing = pair_fixture()
        missing['edited']['detection'].update(available=False, detected=None, score=None)
        flipped = pair_fixture(before=False, after=True)
        for replacement in (pair_report([missing]), pair_report([flipped])):
            value = pair_report()
            value['strata'][0]['estimates']['include_known_noops'] = (
                replacement['strata'][0]['estimates']['include_known_noops'])
            changes.append(value)
        value = pair_report()
        for field in ('cohort_sha256', 'known_target_sha256'):
            value['strata'][0]['estimates']['all_supported'][field] = 'b' * 64
        changes.append(value)
        failed = pair_fixture()
        failed['edited']['validation']['passed'] = False
        unavailable = copy.deepcopy(failed)
        unavailable['edited']['validation'].update(available=False, passed=None)
        value = pair_report([failed])
        value['strata'][0]['estimates']['include_known_noops'] = (
            pair_report([unavailable])['strata'][0]['estimates']['include_known_noops'])
        changes.append(value)
        # Increasing admitted size must not hide loss of an existing transition.
        noop = pair_fixture('noop', before=False, after=False)
        noop['edited']['source'] = noop['original']['source']; noop['changed'] = False
        value = pair_report([pair_fixture(), noop])
        value['strata'][0]['estimates']['include_known_noops'] = (
            pair_report([flipped, noop])['strata'][0]['estimates']['include_known_noops'])
        changes.append(value)
        # A relaxed mode cannot enlarge the functional denominator of an
        # already selected failing row merely by changing its contract identity.
        failed = pair_fixture('failed'); failed['edited']['validation']['passed'] = False
        drifted = copy.deepcopy(failed); drifted['edited']['validation']['contract_id'] = 'other'
        value = pair_report([pair_fixture(), drifted])
        value['strata'][0]['estimates']['include_known_noops'] = (
            pair_report([pair_fixture(), failed])['strata'][0]['estimates']['include_known_noops'])
        changes.append(value)
        value = pair_report([missing])
        bounds = value['strata'][0]['estimates']['include_known_noops']
        bounds['missing_decision_bounds'].update(lower_numerator=-1, upper_numerator=0,
                                                lower=-1.0, upper=0.0)
        bounds['missing_decision_change_bounds'].update(lower_numerator=0, upper_numerator=1,
                                                       lower=0.0, upper=1.0)
        changes.append(value)
        for index, value in enumerate(changes):
            with self.subTest(contradiction=index):
                self.reject_report(value)
        # All three populations really can differ; keep that valid sensitivity path.
        unknown = pair_fixture('unknown'); unknown['changed'] = None
        for side in ('original', 'edited'):
            unknown[side].pop('source')
        unhashed = pair_fixture('unhashed'); unhashed['original'].pop('source')
        unsupported = pair_fixture('unsupported'); unsupported['supported'] = False
        missing['sample_id'] = 'missing'
        good = pair_report([pair_fixture(), noop, unknown, unhashed, unsupported, missing])
        records = certificate.certificates_from_pair_report(good, CLAIM)
        self.assertEqual(records[0]['counts']['n'], 1)
        self.assertEqual([good['strata'][0]['estimates'][m]['admitted_pairs'] for m in
                          ('changed_only', 'include_known_noops', 'all_supported')], [1, 2, 4])

    def test_unidentified_stratum_identity_is_not_its_certificate_hash(self):
        row = pair_fixture()
        row['edited']['detection'].update(available=False, detected=None, score=None)
        original = certificate.certificates_from_raw_pairs([row], CLAIM)[0]
        duplicates = []
        for variation in ('metadata', 'counts', 'target', 'rule', 'boolean'):
            other = copy.deepcopy(original); other['claim_id'] = 'another-label'
            if variation == 'counts':
                other['attempted_pairs'] = other['eligible_target_pairs'] = 2
                other['first_exclusion_counts']['edited_decision_unavailable'] = 2
            elif variation == 'target':
                other['known_target_sha256'] = 'b' * 64
            elif variation == 'rule':
                other['decision_rule']['rule_id'] = 'another-rule'
            elif variation == 'boolean':
                other['attempted_pairs'] = True
            duplicates.append(rehash(other))
        for other in duplicates:
            with self.subTest(other=other['certificate_sha256']):
                with self.assertRaises(EvidenceError):
                    verify.verify_bundle(bundle([original, other]))
                with tempfile.TemporaryDirectory() as td:
                    path, out = Path(td) / 'bundle.json', Path(td) / 'verification.json'
                    path.write_text(json.dumps(bundle([original, other])), encoding='utf-8')
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(verify.main([str(path), '--out', str(out)]), 2)
                    self.assertFalse(out.exists())
        # Distinct rules really are separate strata, even with the same method/edit.
        second = copy.deepcopy(row); second['sample_id'] = 'second'
        for side in ('original', 'edited'):
            second[side]['detection']['rule_id'] = 'second-rule'
        records = certificate.certificates_from_raw_pairs([row, second], CLAIM)
        self.assertEqual(verify.verify_bundle(bundle(records))['certificates_verified'], 2)
        self.assertEqual(verify.verify_bundle(bundle([original]))['certificates_verified'], 1)
        self.assertEqual(original['decision'], 'not_admissible')

    def test_fresh_passive_audit_does_not_call_input_selection_execution(self):
        with tempfile.TemporaryDirectory() as td:
            result = audit(ROOT, Path(td))
            wording = (Path(td) / 'README.md').read_text(encoding='utf-8')
            self.assertIn('configured canonical design: **1605**', wording)
            self.assertNotIn('canonical executed design', wording)
            self.assertEqual((result['release_records'], result['canonical_source_records'],
                              result['excluded_source_records']), (1662, 1605, 57))
            self.assertIn('no experiment rerun', result['basis'])


if __name__ == '__main__':
    unittest.main()
