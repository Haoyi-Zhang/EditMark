"""Independent verification of generated claim certificates."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from editmark_audit.aggregates import analyze
from editmark_audit.certificate import (
    CERTIFICATE_BUNDLE_SCHEMA,
    canonical_hash,
    certify_analysis,
)
from editmark_audit.constants import COUNT_SCHEMA, STRICT_CONTRACT
from editmark_audit.io import EvidenceError
from editmark_audit.verify import (
    main,
    verify_bundle,
    verify_bundle_against_basis,
    verify_certificate,
)


def counts(transitions=None):
    return {
        'schema_version': COUNT_SCHEMA,
        'population': 'watermarked_positive',
        'evidence_kind': 'synthetic_test',
        'analysis_contract': STRICT_CONTRACT,
        'cohort_sha256': 'a' * 64,
        'stratum': {
            'experiment_id': 'x', 'method': 'm', 'model': 'g',
            'source_group': 's', 'language': 'py', 'edit': 'rename',
            'variant_id': 'v',
        },
        'decision_rule': {'rule_id': 'r', 'threshold': 0.5, 'comparator': 'ge'},
        'counts': {'n': 5, 'before_positive': 4, 'after_positive': 2},
        'transitions': transitions,
        'attempted_pairs': 5,
        'first_exclusion_counts': {},
    }


def bundle(document, claim='positive_decision_change'):
    certificate = certify_analysis(analyze(document), claim)
    return {
        'schema_version': CERTIFICATE_BUNDLE_SCHEMA,
        'input_file': {'name': 'counts.json', 'sha256': ''},
        'requested_claim': claim,
        'certificates': [certificate],
    }


class CertificateVerificationTests(unittest.TestCase):
    def test_valid_certificate_rederives(self):
        certificate = certify_analysis(analyze(counts()), 'positive_decision_change')
        result = verify_certificate(certificate)
        self.assertTrue(result['embedded_facts_verified'])
        self.assertFalse(result['basis_verified'])

    def test_hash_only_tampering_is_rejected(self):
        certificate = certify_analysis(analyze(counts()), 'positive_decision_change')
        certificate['supported_statement'] = 'The effect is perfect.'
        with self.assertRaisesRegex(EvidenceError, 'hash'):
            verify_certificate(certificate)

    def test_rehashed_semantic_tampering_is_rejected(self):
        certificate = certify_analysis(analyze(counts()), 'positive_decision_change')
        certificate['estimate'] = {'numerator': 1, 'denominator': 1, 'decimal': 1.0}
        core = dict(certificate)
        core.pop('certificate_sha256')
        certificate['certificate_sha256'] = canonical_hash(core)
        with self.assertRaisesRegex(EvidenceError, 'deterministic result'):
            verify_certificate(certificate)

    def test_transition_tampering_is_rejected(self):
        document = counts({'n00': 0, 'n01': 1, 'n10': 3, 'n11': 1})
        certificate = certify_analysis(analyze(document), 'detection_survival')
        certificate['transitions']['n11'] = 2
        core = dict(certificate)
        core.pop('certificate_sha256')
        certificate['certificate_sha256'] = canonical_hash(core)
        with self.assertRaises(EvidenceError):
            verify_certificate(certificate)

    def test_bundle_claim_mismatch_is_rejected(self):
        document = counts()
        item = bundle(document)
        item['requested_claim'] = 'detection_survival'
        item['input_file']['sha256'] = '0' * 64
        with self.assertRaisesRegex(EvidenceError, 'disagree'):
            verify_bundle(item)

    def test_basis_regeneration_verifies(self):
        document = counts()
        with tempfile.TemporaryDirectory() as td:
            basis = Path(td) / 'counts.json'
            basis.write_text(json.dumps(document), encoding='utf-8')
            from editmark_audit.io import digest
            item = bundle(document)
            item['input_file']['sha256'] = digest(basis)
            result = verify_bundle_against_basis(item, basis)
            self.assertTrue(result['basis_verified'])

    def test_wrong_basis_is_rejected(self):
        document = counts()
        with tempfile.TemporaryDirectory() as td:
            basis = Path(td) / 'counts.json'
            basis.write_text(json.dumps(document), encoding='utf-8')
            item = bundle(document)
            item['input_file']['sha256'] = '0' * 64
            with self.assertRaisesRegex(EvidenceError, 'SHA-256'):
                verify_bundle_against_basis(item, basis)

    def test_cli_verifies_bundle_and_basis(self):
        document = counts()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            basis = td / 'counts.json'
            certs = td / 'certificates.json'
            output = td / 'verification.json'
            basis.write_text(json.dumps(document), encoding='utf-8')
            from editmark_audit.io import digest
            item = bundle(document)
            item['input_file']['sha256'] = digest(basis)
            certs.write_text(json.dumps(item), encoding='utf-8')
            self.assertEqual(
                main([str(certs), '--basis', str(basis), '--out', str(output)]),
                0,
            )
            self.assertTrue(json.loads(output.read_text())['basis_verified'])

    def test_free_form_claim_text_is_not_authoritative(self):
        certificate = certify_analysis(
            analyze(counts()),
            'positive_decision_change',
            claim_text='This free-form text is not automatically verified.',
        )
        self.assertFalse(certificate['claim_text_verified'])
        self.assertIn('authoritative', certificate['prohibited_interpretations'][-1])
        verify_certificate(certificate)


if __name__ == '__main__':
    unittest.main()
