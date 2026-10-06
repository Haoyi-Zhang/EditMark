"""Owned synthetic regressions for typed re-derivation and publication."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editmark_audit.aggregates import analyze
from editmark_audit.analysis import report
from editmark_audit.certificate import (
    canonical_hash, certificates_from_pair_report, certify_analysis,
)
from editmark_audit.io import EvidenceError, write_new_json
from editmark_audit.records import Pair
from editmark_audit.verify import main, verify_certificate
from test_certificate_verify import bundle, counts
from test_contract import fixture


def rehash(certificate):
    core = dict(certificate)
    core.pop('certificate_sha256')
    certificate['certificate_sha256'] = canonical_hash(core)


class TypedCertificateTests(unittest.TestCase):
    def test_boolean_fraction_component_is_not_an_integer(self):
        original = certify_analysis(analyze(counts()), 'detection_survival')
        for endpoint in ('lower', 'upper'):
            certificate = copy.deepcopy(original)
            certificate['bounds'][endpoint]['numerator'] = True
            rehash(certificate)
            with self.subTest(endpoint=endpoint), self.assertRaises(EvidenceError):
                verify_certificate(certificate)

    def test_integer_requirement_is_not_a_boolean(self):
        certificate = certify_analysis(analyze(counts()), 'positive_decision_change')
        certificate['requirements']['positive_cohort_named'] = 1
        rehash(certificate)
        with self.assertRaises(EvidenceError):
            verify_certificate(certificate)

    def test_float_fraction_denominator_is_not_an_exact_count(self):
        certificate = certify_analysis(analyze(counts()), 'positive_decision_change')
        certificate['estimate']['denominator'] = 5.0
        rehash(certificate)
        with self.assertRaises(EvidenceError):
            verify_certificate(certificate)

    def test_analysis_boolean_fraction_rejected_before_issuance(self):
        analysis = analyze(counts())
        analysis['survival_bounds']['lower']['numerator'] = True
        with self.assertRaises(EvidenceError):
            certify_analysis(analysis, 'detection_survival')

    def test_pair_report_is_not_membership_authentication(self):
        basis = report([Pair.parse(fixture())])
        certificate = certificates_from_pair_report(basis, 'positive_decision_change')[0]
        self.assertIs(certificate['membership_verified'], False)
        self.assertIs(certificate['requirements']['cohort_membership_authenticated'], False)

    def test_membership_assertion_cannot_authenticate_count_analysis(self):
        analysis = analyze(counts())
        analysis['membership_verified'] = True
        with self.assertRaises(EvidenceError):
            certify_analysis(analysis, 'positive_decision_change')


class VerificationPublicationTests(unittest.TestCase):
    def test_verification_uses_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'certificate.json'
            output = Path(td) / 'verification.json'
            value = bundle(counts())
            value['input_file']['sha256'] = '0' * 64
            source.write_text(json.dumps(value), encoding='utf-8')
            with patch('editmark_audit.verify.write_new_json', wraps=write_new_json) as writer:
                self.assertEqual(main([str(source), '--out', str(output)]), 0)
                writer.assert_called_once()
            saved = output.read_bytes()
            self.assertEqual(main([str(source), '--out', str(output)]), 2)
            self.assertEqual(output.read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
