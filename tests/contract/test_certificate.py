"""Claim-certificate behavior and prohibited inferences."""
import copy
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from editmark_audit.aggregates import analyze
from editmark_audit.certificate import certify_analysis, certificates_from_pair_report, main
from editmark_audit.constants import COUNT_SCHEMA, STRICT_CONTRACT
from editmark_audit.io import EvidenceError
from editmark_audit.analysis import report
from editmark_audit.records import Pair
from tests.contract.test_contract import fixture


def counts(n=5,b=4,a=2,transitions=None):
    return {
        'schema_version':COUNT_SCHEMA,'population':'watermarked_positive',
        'evidence_kind':'synthetic_test','analysis_contract':STRICT_CONTRACT,
        'cohort_sha256':'a'*64,
        'stratum':{'experiment_id':'x','method':'m','model':'g','source_group':'s','language':'py','edit':'rename','variant_id':'v'},
        'decision_rule':{'rule_id':'r','threshold':0.5,'comparator':'ge'},
        'counts':{'n':n,'before_positive':b,'after_positive':a},
        'transitions':transitions,'attempted_pairs':n,'first_exclusion_counts':{},
    }

class CertificateTests(unittest.TestCase):
    def test_change_is_identified(self):
        certificate=certify_analysis(analyze(counts()),'positive_decision_change')
        self.assertEqual(certificate['decision'],'admissible')
        self.assertEqual(certificate['estimate']['numerator'],-2)
        self.assertEqual(certificate['estimate']['denominator'],5)
    def test_survival_is_bounded_without_joint_table(self):
        certificate=certify_analysis(analyze(counts()),'detection_survival')
        self.assertEqual(certificate['decision'],'partially_admissible')
        self.assertEqual(certificate['bounds']['lower']['numerator'],1)
        self.assertEqual(certificate['bounds']['lower']['denominator'],4)
        self.assertEqual(certificate['bounds']['upper']['numerator'],1)
        self.assertEqual(certificate['bounds']['upper']['denominator'],2)
    def test_survival_is_identified_with_transitions(self):
        document=counts(transitions={'n00':0,'n01':1,'n10':3,'n11':1})
        certificate=certify_analysis(analyze(document),'detection_survival')
        self.assertEqual(certificate['decision'],'admissible')
        self.assertEqual(certificate['estimate']['decimal'],.25)
    def test_false_positive_claim_rejected(self):
        certificate=certify_analysis(analyze(counts()),'edited_false_positive_rate')
        self.assertEqual(certificate['decision'],'not_admissible')
        self.assertEqual(certificate['identification'],'missing_negative_cohort')
    def test_auroc_claim_rejected(self):
        certificate=certify_analysis(analyze(counts()),'auroc_change')
        self.assertEqual(certificate['decision'],'not_admissible')
    def test_posterior_provenance_claim_rejected(self):
        certificate=certify_analysis(analyze(counts()),'post_edit_provenance_probability')
        self.assertEqual(certificate['decision'],'not_admissible')
    def test_certificate_hash_changes_with_claim(self):
        a=certify_analysis(analyze(counts()),'positive_decision_change')
        b=certify_analysis(analyze(counts()),'detection_survival')
        self.assertNotEqual(a['certificate_sha256'],b['certificate_sha256'])
    def test_pair_report_round_trip(self):
        pair_report=report([Pair.parse(fixture())])
        certificates=certificates_from_pair_report(pair_report,'positive_decision_change')
        self.assertEqual(len(certificates),1)
        self.assertEqual(certificates[0]['decision'],'admissible')
    def test_cli_returns_three_for_bounded_or_rejected_claim(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'counts.json';out=Path(td)/'cert.json'
            source.write_text(json.dumps(counts()))
            self.assertEqual(main([str(source),'--claim','detection_survival','--out',str(out)]),3)
            self.assertTrue(out.exists())
    def test_cli_require_observations_rejects_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'counts.json';out=Path(td)/'cert.json'
            source.write_text(json.dumps(counts()))
            self.assertEqual(main([str(source),'--claim','positive_decision_change','--require-observations','--out',str(out)]),2)
            self.assertFalse(out.exists())
