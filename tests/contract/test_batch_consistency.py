"""Finite synthetic batch consistency across public producers and basis routes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from editmark_audit import aggregates, certificate, verify
from editmark_audit.analysis import report
from editmark_audit.constants import COUNT_REPORT_SCHEMA
from editmark_audit.io import EvidenceError, digest, load_rows
from editmark_audit.records import Pair

ROOT = Path(__file__).resolve().parents[2]
CLAIM = 'positive_decision_change'


def counts():
    return load_rows(ROOT / 'examples/synthetic_cohort_counts.json')[0]


def conflicting():
    a = counts(); b = copy.deepcopy(a)
    b['counts']['after_positive'] = 3
    return [a, b]


def count_report(documents):
    return {'schema_version': COUNT_REPORT_SCHEMA,
            'analyses': [aggregates.analyze(d) for d in documents]}


def pair_report(conflict=False, duplicate=False):
    rows = [Pair.parse(d) for d in load_rows(ROOT / 'examples/synthetic_count_basis.jsonl')]
    value = report(rows)
    if conflict or duplicate:
        other = copy.deepcopy(value['strata'][0])
        if conflict:
            e = other['estimates']['changed_only']
            e['n10'] -= 1; e['n11'] += 1
        value['strata'].append(other)
    return value


def manual_bundle(path, documents):
    return {'schema_version': certificate.CERTIFICATE_BUNDLE_SCHEMA,
            'input_file': {'name': path.name, 'sha256': digest(path)},
            'requested_claim': CLAIM,
            'certificates': [certificate.certify_analysis(aggregates.analyze(d), CLAIM)
                             for d in documents]}


class BatchConsistencyTests(unittest.TestCase):
    def certify(self, document):
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td)/'basis.json', Path(td)/'certificate.json'
            src.write_text(json.dumps(document), encoding='utf-8')
            status = certificate.main([str(src), '--claim', CLAIM, '--out', str(out)])
            return status, out.exists(), json.loads(out.read_text()) if out.exists() else None

    def test_conflicting_counts_cli_fails_before_publication(self):
        self.assertEqual(self.certify(conflicting())[:2], (2, False))

    def test_conflicting_count_report_cli_fails_before_publication(self):
        self.assertEqual(self.certify(count_report(conflicting()))[:2], (2, False))

    def test_conflicting_pair_report_cli_fails_before_publication(self):
        self.assertEqual(self.certify(pair_report(conflict=True))[:2], (2, False))

    def test_direct_count_and_pair_functions_reject_conflicts(self):
        with self.assertRaisesRegex(EvidenceError, 'cohort/stratum'):
            certificate.certificates_from_count_documents(conflicting(), CLAIM)
        with self.assertRaisesRegex(EvidenceError, 'cohort/stratum'):
            certificate.certificates_from_pair_report(pair_report(conflict=True), CLAIM)

    def test_conflicting_basis_rejected_by_regeneration_for_every_report_kind(self):
        cases = [conflicting(), count_report(conflicting()), pair_report(conflict=True)]
        with tempfile.TemporaryDirectory() as td:
            for i, value in enumerate(cases):
                path = Path(td)/f'basis-{i}.json'
                path.write_text(json.dumps(value), encoding='utf-8')
                with self.subTest(kind=i), self.assertRaisesRegex(EvidenceError, 'cohort/stratum'):
                    verify._regenerate_from_basis(path, CLAIM)

    def test_basis_verification_runs_batch_gate_even_with_single_valid_certificate(self):
        with tempfile.TemporaryDirectory() as td:
            for i, value in enumerate([conflicting(), count_report(conflicting())]):
                path = Path(td)/f'basis-{i}.json'
                path.write_text(json.dumps(value), encoding='utf-8')
                bundle = manual_bundle(path, [counts()])
                with self.subTest(kind=i), self.assertRaisesRegex(EvidenceError, 'cohort/stratum'):
                    verify.verify_bundle_against_basis(bundle, path)

    def test_standalone_and_basis_verifiers_reject_conflicting_certificates_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); basis=td/'basis.json'; bundle_path=td/'bundle.json'
            basis.write_text(json.dumps(conflicting()), encoding='utf-8')
            bundle = manual_bundle(basis, conflicting())
            bundle_path.write_text(json.dumps(bundle), encoding='utf-8')
            with self.assertRaisesRegex(EvidenceError, 'cohort/stratum'):
                verify.verify_bundle(bundle)
            for mode, extra in [('standalone', []), ('basis', ['--basis', str(basis)])]:
                out = td/('verified-'+mode+'.json')
                self.assertEqual(verify.main([str(bundle_path), *extra, '--out', str(out)]), 2)
                self.assertFalse(out.exists())

    def test_rule_transition_and_partition_conflicts_are_not_deduplicated(self):
        mutations = [lambda d: d['decision_rule'].update(threshold=0.75),
                     lambda d: d.update(attempted_pairs=6, first_exclusion_counts={'unchanged':1}),
                     lambda d: d.update(transitions={'n00':0,'n01':1,'n10':3,'n11':1})]
        for change in mutations:
            a=counts(); b=copy.deepcopy(a); change(b)
            with self.subTest(change=change), self.assertRaisesRegex(EvidenceError, 'cohort/stratum'):
                certificate.certificates_from_count_documents([a,b], CLAIM)

    def test_identical_counts_and_reports_are_idempotent(self):
        a=counts()
        for value in ([a,a], count_report([a,a]), pair_report(duplicate=True)):
            status, exists, bundle = self.certify(value)
            self.assertEqual((status, exists), (0, True))
            self.assertEqual(len(bundle['certificates']), 1)
            self.assertEqual(verify.verify_bundle(bundle)['certificates_verified'], 1)

    def test_identical_bundle_and_basis_are_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'counts.json'; path.write_text(json.dumps([counts(),counts()]))
            bundle=manual_bundle(path,[counts(),counts()])
            result=verify.verify_bundle_against_basis(bundle,path)
            self.assertTrue(result['basis_verified'])
            self.assertEqual(result['certificates_verified'],1)

    def test_distinct_cohorts_remain_legal_and_do_not_pool(self):
        a=counts(); b=copy.deepcopy(a); b['cohort_sha256']='b'*64; b['counts']['after_positive']=3
        status, exists, bundle=self.certify([a,b])
        self.assertEqual((status,exists),(0,True))
        self.assertEqual(len(bundle['certificates']),2)
        self.assertEqual([c['estimate']['numerator'] for c in bundle['certificates']],[-2,-1])

    def test_mixed_evidence_still_fails_before_publication(self):
        a=counts(); b=copy.deepcopy(a); b['cohort_sha256']='b'*64; b['evidence_kind']='saved_observation'
        self.assertEqual(self.certify([a,b])[:2],(2,False))

    def test_default_count_report_certificate_basis_chain(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); src=td/'counts.json'; counted=td/'report.json'; cert=td/'cert.json'; checked=td/'checked.json'
            src.write_text(json.dumps([counts(),counts()]),encoding='utf-8')
            self.assertEqual(aggregates.main([str(src),'--out',str(counted)]),0)
            self.assertEqual(len(json.loads(counted.read_text())['analyses']),1)
            self.assertEqual(certificate.main([str(counted),'--claim',CLAIM,'--out',str(cert)]),0)
            self.assertEqual(verify.main([str(cert),'--basis',str(counted),'--out',str(checked)]),0)
            self.assertTrue(json.loads(checked.read_text())['basis_verified'])

    def test_raw_pair_basis_stays_source_bound_and_idempotent(self):
        rows=load_rows(ROOT/'examples/synthetic_count_basis.jsonl')
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td)/'raw.jsonl'; out=Path(td)/'cert.json'
            raw.write_text('\n'.join(json.dumps(r) for r in rows+[rows[0]])+'\n')
            self.assertEqual(certificate.main([str(raw),'--claim',CLAIM,'--out',str(out)]),0)
            result=verify.verify_bundle_against_basis(json.loads(out.read_text()),raw)
            self.assertTrue(result['raw_pair_membership_rederived'])
            self.assertEqual(result['certificates_verified'],1)
