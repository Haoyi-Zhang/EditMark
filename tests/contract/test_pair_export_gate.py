"""Synthetic diagnostic-mode reports must not acquire strict cohort claims."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from editmark_audit import aggregates, certificate
from editmark_audit.analysis import report
from editmark_audit.constants import STRICT_CONTRACT
from editmark_audit.io import EvidenceError
from editmark_audit.records import Pair
from tests.contract.test_contract import fixture


def basis(noop=False):
    raw = fixture()
    if noop:
        raw['edited']['source'] = raw['original']['source']
        raw['changed'] = False
    return report([Pair.parse(raw)])


class PairExportGateTests(unittest.TestCase):
    def test_non_strict_mode_cannot_be_relabelled_by_slot_name(self):
        for mode in ('all_supported', 'include_known_noops'):
            value = basis(noop=True)
            value['strata'][0]['estimates']['changed_only'] = copy.deepcopy(value['strata'][0]['estimates'][mode])
            with self.assertRaises(EvidenceError):
                aggregates.from_pair_stratum(value['strata'][0], value['evidence_kind'])
            with self.assertRaises(EvidenceError):
                certificate.certificates_from_pair_report(value, 'positive_decision_change')

    def test_contradictory_noop_marginal_still_rejects_after_label_spoof(self):
        value = basis(noop=True)
        estimate = copy.deepcopy(value['strata'][0]['estimates']['all_supported'])
        estimate.update(mode='changed_only', analysis_contract=STRICT_CONTRACT)
        value['strata'][0]['estimates']['changed_only'] = estimate
        with self.assertRaisesRegex(EvidenceError, 'marginal coverage'):
            certificate.certificates_from_pair_report(value, 'positive_decision_change')

    def test_missing_or_contradictory_contract_fields_reject(self):
        for key, replacement in (('mode', None), ('analysis_contract', 'all_supported'),
                                  ('eligible_target_pairs', 0), ('marginal_coverage', None),
                                  ('first_exclusion_counts', {'unchanged': 1})):
            value = basis()
            value['strata'][0]['estimates']['changed_only'][key] = replacement
            with self.subTest(key=key), self.assertRaises(EvidenceError):
                certificate.certificates_from_pair_report(value, 'positive_decision_change')

    def test_empty_strict_noop_report_remains_a_denial(self):
        result = certificate.certificates_from_pair_report(basis(noop=True), 'positive_decision_change')
        self.assertEqual(result[0]['decision'], 'not_admissible')

    def test_genuine_strict_and_raw_pair_routes_agree(self):
        raw = fixture()
        self.assertEqual(certificate.certificates_from_pair_report(basis(), 'positive_decision_change'),
                         certificate.certificates_from_raw_pairs([raw], 'positive_decision_change'))

    def test_cli_refuses_before_publishing_a_certificate(self):
        value = basis(noop=True)
        value['strata'][0]['estimates']['changed_only'] = value['strata'][0]['estimates']['all_supported']
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'basis.json', Path(directory) / 'certificate.json'
            source.write_text(json.dumps(value), encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()):
                status = certificate.main([str(source), '--claim', 'positive_decision_change', '--out', str(output)])
            self.assertEqual(status, 2)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
