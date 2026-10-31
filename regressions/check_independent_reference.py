"""Opt-in independent table arithmetic/dependence recount; no vendor execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editmark_audit import reference
from editmark_audit.aggregates import analyze
from editmark_audit.certificate import certify_analysis

ROOT = Path(__file__).resolve().parents[1]


def rehash(cert):
    cert['certificate_sha256'] = hashlib.sha256(json.dumps(
        {k:v for k,v in cert.items() if k != 'certificate_sha256'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return cert


class IndependentReferenceTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / 'results/semcodebook_cpu'

    def test_all_retained_cpu_claims_and_dependency_units(self):
        for claim in ('positive_decision_change', 'detection_survival',
                      'edited_false_positive_rate', 'auroc_change', 'post_edit_provenance_probability'):
            bundle = reference.load((self.path / (claim + '.json')).read_text())
            with patch('editmark_audit.aggregates.analyze', side_effect=RuntimeError('no shared analyzer')):
                self.assertEqual(len(reference.check_bundle(bundle)), 2)
                for cert in bundle['certificates']:
                    reference.check_certificate(cert)
                groups = reference.check_complete_basis(bundle, self.path / 'pairs.jsonl')
                repeated = copy.deepcopy(bundle)
                repeated['certificates'].append(copy.deepcopy(repeated['certificates'][0]))
                self.assertEqual(len(reference.check_bundle(repeated)), 2)
            self.assertEqual(len(groups), 2)
            self.assertTrue(all(g['configuration_pairs'] == 16 and
                g['distinct_original_source_texts'] == 4 and g['declared_clusters'] == 1 and
                g['independent_program_count_established'] is False for g in groups))

    def test_marginal_tables_and_wrong_rehashed_bounds(self):
        saved = reference.load((self.path / 'detection_survival.json').read_text())['certificates'][0]
        facts = {k:saved[k] for k in ('population', 'evidence_kind', 'analysis_contract', 'cohort_sha256',
            'stratum', 'decision_rule', 'counts', 'attempted_pairs', 'first_exclusion_counts')}
        facts['schema_version'] = 'editmark-cohort-counts'
        cert = certify_analysis(analyze(facts), 'detection_survival')
        self.assertEqual(reference.check_certificate(cert)['feasible_tables'], 2)
        wrong = copy.deepcopy(cert)
        wrong['bounds']['lower'] = dict(numerator=0, denominator=1, decimal=0.0)
        with self.assertRaisesRegex(ValueError, 'bound arithmetic'):
            reference.check_certificate(rehash(wrong))

    def test_rehashed_estimate_status_and_type_mutations_reject(self):
        bundle = reference.load((self.path / 'positive_decision_change.json').read_text())
        for mutate in (lambda c:c['estimate'].update(numerator=1, denominator=1, decimal=1.0),
                       lambda c:c.update(decision='not_admissible'),
                       lambda c:c['counts'].update(n=True)):
            wrong = copy.deepcopy(bundle['certificates'][0])
            mutate(wrong)
            with self.assertRaises(ValueError):
                reference.check_certificate(rehash(wrong))
        wrong_bundle = copy.deepcopy(bundle)
        wrong_bundle['requested_claim'] = 'detection_survival'
        with self.assertRaisesRegex(ValueError, 'claim mismatch'):
            reference.check_bundle(wrong_bundle)
        wrong_bundle = copy.deepcopy(bundle)
        wrong_bundle['certificates'] = []
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            reference.check_bundle(wrong_bundle)

    def test_positive_only_denial_cannot_be_promoted(self):
        cert = reference.load((self.path / 'edited_false_positive_rate.json').read_text())['certificates'][0]
        cert.update(decision='admissible', identification='identified',
                    estimate=dict(numerator=0, denominator=1, decimal=0.0))
        with self.assertRaisesRegex(ValueError, 'claim status'):
            reference.check_certificate(rehash(cert))
        bundle = reference.load((self.path / 'detection_survival.json').read_text())
        first = bundle['certificates'][0]
        facts = {k:first[k] for k in ('population', 'evidence_kind', 'analysis_contract', 'cohort_sha256',
            'stratum', 'decision_rule', 'counts', 'transitions', 'attempted_pairs', 'first_exclusion_counts')}
        facts = copy.deepcopy(facts)
        facts['schema_version'] = 'editmark-cohort-counts'
        facts['counts']['after_positive'] = 6
        facts['transitions'].update(n10=9, n11=6)
        bundle['certificates'].append(certify_analysis(analyze(facts), 'detection_survival'))
        with self.assertRaisesRegex(ValueError, 'conflicting cohort'):
            reference.check_bundle(bundle)

    def test_reference_is_explicitly_bounded_and_strict_json(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            reference.load('{"n":1,"n":2}')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            reference.load('{"n":NaN}')
        cert = reference.load((self.path / 'detection_survival.json').read_text())['certificates'][0]
        cert['counts']['n'] = 10001
        with self.assertRaisesRegex(ValueError, 'bounded arithmetic'):
            reference.check_certificate(rehash(cert))

    def test_raw_source_or_cohort_substitution_rejects_even_with_new_basis_hash(self):
        bundle = reference.load((self.path / 'detection_survival.json').read_text())
        rows = [reference.load(s) for s in (self.path / 'pairs.jsonl').read_text().splitlines()]
        rows[0]['original']['source'] += '# changed but not rehashed\n'
        with tempfile.TemporaryDirectory() as temp:
            basis = Path(temp) / 'pairs.jsonl'
            basis.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            bundle['input_file']['sha256'] = hashlib.sha256(basis.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'source digest'):
                reference.check_complete_basis(bundle, basis)
        bundle = reference.load((self.path / 'detection_survival.json').read_text())
        bundle['certificates'][0]['cohort_sha256'] = '0' * 64
        rehash(bundle['certificates'][0])
        reference.check_bundle(bundle)  # Embedded counts alone cannot recover membership.
        with self.assertRaisesRegex(ValueError, 'cohort fingerprint'):
            reference.check_complete_basis(bundle, self.path / 'pairs.jsonl')


if __name__ == '__main__':
    unittest.main()
