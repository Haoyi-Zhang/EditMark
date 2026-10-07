"""Owned synthetic fixtures; independent truth tables and finite completions."""
import copy
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from editmark_audit import analysis, aggregates, certificate, verify
from editmark_audit.io import EvidenceError
from editmark_audit.records import Pair

MODES = ('changed_only', 'include_known_noops', 'all_supported')


def fixture(name, before=True, after=False):
    def program(source, decision):
        return dict(source=source,
                    validation=dict(available=True, passed=True, contract_id='owned-checks'),
                    detection=dict(available=decision is not None, detected=decision,
                                   rule_id='owned-rule', threshold=0.5, comparator='ge'))
    return dict(schema_version='editmark-pair', population='watermarked_positive',
                evidence_kind='synthetic_test', experiment_id='OWNED', method='toy', model='toy',
                source_group='owned', language='text', sample_id=name, replicate_id='r0',
                edit='space', variant_id='v0', supported=True, changed=True,
                original=program('before ' + name, before), edited=program('after ' + name, after))


def eligibility_cases(before=True, after=False):
    """Explicit expected first gates from the paper; no analyzer gate calls."""
    cases = []
    def add(label, reasons, edit):
        row = fixture(label, before, after)
        edit(row)
        cases.append((row, reasons))
    clear = (None, None, None)
    add('valid', clear, lambda d: None)
    for value, reason in ((False, 'unsupported'), (None, 'support_unknown')):
        add(reason, (reason,) * 3, lambda d, value=value: d.update(supported=value))
    def noop(d):
        d['edited']['source'] = d['original']['source']
        d['changed'] = False
    add('noop', ('unchanged', None, None), noop)
    def unknown_change(d):
        del d['original']['source']
        del d['edited']['source']
        d['changed'] = None
    add('unknown-change', ('change_unknown', 'change_unknown', None), unknown_change)
    def missing_hash(d):
        del d['original']['source']
    add('missing-hash', ('source_identity_missing', 'source_identity_missing', None), missing_hash)
    for side in ('original', 'edited'):
        for available, passed, suffix in ((False, None, 'validation_unavailable'),
                                          (None, None, 'validation_unavailable'),
                                          (True, False, 'checks_failed'), (True, None, 'checks_unknown')):
            reason = side + '_' + suffix
            def change(d, side=side, available=available, passed=passed):
                d[side]['validation'].update(available=available, passed=passed)
            add(reason + str(available), (reason,) * 3, change)
    for label, reason, side, field, value in (
        ('empty-contract', 'check_contract_unknown', 'edited', 'contract_id', ''),
        ('contract-drift', 'check_contract_changed', 'edited', 'contract_id', 'other-checks'),
        ('empty-rule', 'decision_rule_unknown', 'edited', 'rule_id', ''),
        ('rule-drift', 'decision_rule_changed', 'edited', 'rule_id', 'other-rule'),
        ('threshold-drift', 'decision_rule_changed', 'edited', 'threshold', 0.75),
        ('comparator-drift', 'decision_rule_changed', 'edited', 'comparator', 'gt'),
    ):
        def change(d, side=side, field=field, value=value):
            group = 'validation' if field == 'contract_id' else 'detection'
            d[side][group][field] = value
        add(label, (reason,) * 3, change)
    def precedence(d):
        d['supported'] = False
        d['edited']['validation'].update(passed=False)
        d['edited']['detection']['rule_id'] = 'other-rule'
    add('first-gate', ('unsupported',) * 3, precedence)
    return cases


def expected_fingerprint(row, before, after):
    key = tuple(row[k] for k in ('evidence_kind', 'experiment_id', 'method', 'model',
                                'source_group', 'language', 'sample_id', 'replicate_id',
                                'edit', 'variant_id'))
    hashes = tuple(hashlib.sha256(row[side]['source'].encode()).hexdigest()
                   if 'source' in row[side] else '' for side in ('original', 'edited'))
    rule = row['original']['detection']
    payload = [(key, *hashes, row['original']['validation']['contract_id'],
                (rule['rule_id'], float(rule['threshold']), rule['comparator']), before, after)]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


class AdmittedReuseTests(unittest.TestCase):
    def test_gate_truth_table_and_completion_bounds(self):
        for before, after in product((None, False, True), repeat=2):
            for row, reasons in eligibility_cases(before, after):
                for mode, gate in zip(MODES, reasons):
                    pair = Pair.parse(row)
                    actual = analysis.summarize(iter((pair,)), mode)
                    exclusion = gate or ('original_decision_unavailable' if before is None else
                                         'edited_decision_unavailable' if after is None else None)
                    self.assertEqual(analysis.exclusion(pair, mode), exclusion)
                    self.assertEqual(actual['first_exclusion_counts'], {exclusion: 1} if exclusion else {})
                    self.assertEqual(actual['admitted_pairs'], int(exclusion is None))
                    self.assertEqual(actual['eligible_target_pairs'], int(gate is None))
                    fingerprint = expected_fingerprint(row, before, after) if gate is None else None
                    self.assertEqual(actual['known_target_sha256'], fingerprint)
                    self.assertEqual(actual['cohort_sha256'], fingerprint if exclusion is None else None)
                    self.assertEqual(sum(actual[k] for k in ('n00', 'n01', 'n10', 'n11')),
                                     int(exclusion is None))
                    if exclusion is None:
                        self.assertEqual(actual[f'n{int(before)}{int(after)}'], 1)
                    if gate is None:
                        completions = [int(a) - int(b) for b, a in product(
                            (False, True) if before is None else (before,),
                            (False, True) if after is None else (after,))]
                        bounds = actual['missing_decision_change_bounds']
                        self.assertEqual((bounds['lower_numerator'], bounds['upper_numerator']),
                                         (min(completions), max(completions)))
                    else:
                        self.assertIsNone(actual['missing_decision_change_bounds']['lower'])

    def test_mixed_modes_permutations_and_local_predicate_reuse(self):
        pairs = [Pair.parse(row) for row, _ in eligibility_cases()]
        for mode in MODES:
            first = analysis.summarize(pairs, mode)
            self.assertEqual(first, analysis.summarize(list(reversed(pairs)), mode))
            self.assertEqual(first['attempted_pairs'], first['admitted_pairs'] +
                             sum(first['first_exclusion_counts'].values()))
            # Operation counts only, not timings: one public exclusion call per
            # immutable row. The existing target precondition pass is retained.
            with patch.object(analysis, 'exclusion', wraps=analysis.exclusion) as call:
                self.assertEqual(first, analysis.summarize(pairs, mode))
                self.assertEqual(call.call_count, len(pairs))
        with self.assertRaises(EvidenceError):
            analysis.summarize([], 'not-a-mode')
        empty = analysis.summarize([])
        self.assertIsNone(empty['cohort_sha256'])
        self.assertIsNone(empty['net_detection_change'])

    def test_roundtrip_all_claims_and_non_authentication(self):
        rows = [fixture(str(i), b, a) for i, (b, a) in enumerate(product((False, True), repeat=2))]
        basis = analysis.report([Pair.parse(row) for row in rows])
        disclosure = aggregates.from_pair_stratum(basis['strata'][0], 'synthetic_test')
        counts = aggregates.analyze(disclosure)
        self.assertEqual(disclosure['transitions'], dict(n00=1, n01=1, n10=1, n11=1))
        self.assertEqual(counts['net_detection_change'], dict(numerator=0, denominator=1, decimal=0.0))
        for claim in certificate.CLAIMS:
            cert = certificate.certificates_from_pair_report(basis, claim)[0]
            verify.verify_certificate(cert)
            self.assertIs(cert['membership_verified'], False)
            self.assertIs(cert['requirements']['cohort_membership_authenticated'], False)
            self.assertEqual(cert['decision'], 'admissible' if claim in
                             ('positive_decision_change', 'detection_survival') else 'not_admissible')

    def test_duplicate_conflicts_and_unknowns_are_not_false(self):
        row = fixture('identity')
        pair = Pair.parse(row)
        report = analysis.report([pair, pair])
        self.assertEqual(report['identical_duplicates_removed'], 1)
        changed = copy.deepcopy(row)
        changed['edited']['detection']['detected'] = True
        with self.assertRaises(EvidenceError):
            analysis.report([pair, Pair.parse(changed)])
        row = fixture('missing', None, None)
        row['original']['detection']['available'] = None
        row['edited']['detection']['available'] = True
        result = analysis.summarize([Pair.parse(row)])
        self.assertEqual(result['first_exclusion_counts'], {'original_decision_unavailable': 1})
        self.assertEqual((result['missing_decision_change_bounds']['lower'],
                          result['missing_decision_change_bounds']['upper']), (-1.0, 1.0))


if __name__ == '__main__':
    unittest.main()
