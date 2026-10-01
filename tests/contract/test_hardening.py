"""Regression inputs are synthetic; they do not reproduce watermark experiments."""
import unittest
from editmark_audit.analysis import summarize
from editmark_audit.io import EvidenceError, loads
from editmark_audit.records import Pair, number
from test_contract import fixture

class NumericBoundaryTests(unittest.TestCase):
    def test_unrepresentable_integer_threshold_rejected(self):
        d=fixture();d['original']['detection'].update(score=2**53,threshold=2**53+1,detected=True)
        with self.assertRaisesRegex(EvidenceError, 'exactly'): Pair.parse(d)
    def test_unrepresentable_integer_score_rejected(self):
        with self.assertRaises(EvidenceError): number(2**53+1,'score')
    def test_negative_unrepresentable_integer_rejected(self):
        with self.assertRaises(EvidenceError): number(-2**53-1,'score')
    def test_exactly_representable_integer_accepted(self):
        self.assertEqual(number(2**53,'score'),2**53)
    def test_positive_underflow_rejected(self):
        with self.assertRaisesRegex(EvidenceError, 'underflows'): loads('{"x":1e-400}')
    def test_negative_underflow_rejected(self):
        with self.assertRaises(EvidenceError): loads('{"x":-1e-400}')
    def test_true_exponent_zero_preserved(self):
        self.assertEqual(loads('{"x":0e-400}')['x'],0.)
    def test_representable_subnormal_accepted(self):
        self.assertGreater(loads('{"x":5e-324}')['x'],0.)
    def test_score_model_documents_binary64_rounding(self):
        self.assertEqual(loads('{"x":0.1}')['x'],0.1)

class FieldBoundaryTests(unittest.TestCase):
    def test_unknown_top_level(self):
        d=fixture();d['suppported']=True
        with self.assertRaisesRegex(EvidenceError,'unknown fields'): Pair.parse(d)
    def test_unknown_program_field(self):
        d=fixture();d['edited']['sorce']='xyz'
        with self.assertRaises(EvidenceError): Pair.parse(d)
    def test_unknown_validation_field(self):
        d=fixture();d['edited']['validation']['passsed']=True
        with self.assertRaises(EvidenceError): Pair.parse(d)
    def test_unknown_detection_field(self):
        d=fixture();d['edited']['detection']['detectd']=d['edited']['detection'].pop('detected')
        with self.assertRaises(EvidenceError): Pair.parse(d)
    def test_whitespace_identity_not_silently_collapsed(self):
        d=fixture();d['sample_id']=' s1'
        with self.assertRaises(EvidenceError): Pair.parse(d)
    def test_invalid_unicode_identity_rejected(self):
        d=fixture();d['sample_id']='\ud800'
        with self.assertRaises(EvidenceError): Pair.parse(d)
    def test_invalid_unicode_source_rejected(self):
        d=fixture();d['edited']['source']='\ud800'
        with self.assertRaises(EvidenceError): Pair.parse(d)
    def test_legitimate_unicode_source_preserved(self):
        d=fixture();d['edited']['source']='变量 = 1'
        self.assertIsNotNone(Pair.parse(d).edited.source_sha256)
    def test_empty_object_means_unknown_not_false(self):
        d=fixture();d['edited']['validation']={}
        self.assertIsNone(Pair.parse(d).edited.validation.passed)

class TargetFingerprintTests(unittest.TestCase):
    def test_missing_decision_target_has_hash(self):
        d=fixture();d['edited']['detection'].update(available=False,detected=None,score=None)
        r=summarize([Pair.parse(d)])
        self.assertIsNone(r['cohort_sha256']);self.assertIsNotNone(r['known_target_sha256'])
    def test_target_hash_changes_when_partial_observation_changes(self):
        a,b=fixture(),fixture(before=False)
        for d in (a,b):d['edited']['detection'].update(available=False,detected=None,score=None)
        self.assertNotEqual(summarize([Pair.parse(a)])['known_target_sha256'],summarize([Pair.parse(b)])['known_target_sha256'])
    def test_exact_missing_range_counts_retained(self):
        d=fixture();d['edited']['detection'].update(available=False,detected=None,score=None)
        b=summarize([Pair.parse(d)])['missing_decision_bounds']
        self.assertEqual((b['lower_numerator'],b['upper_numerator'],b['denominator']),(0,1,1))
    def test_target_hash_order_invariant(self):
        a,b=Pair.parse(fixture('a')),Pair.parse(fixture('b'))
        self.assertEqual(summarize([a,b])['known_target_sha256'],summarize([b,a])['known_target_sha256'])

class ExtremeExponentTests(unittest.TestCase):
    def test_extreme_underflow_is_evidence_error_not_decimal_traceback(self):
        with self.assertRaises(EvidenceError): loads('1e-99999999999999999999999')
    def test_exact_zero_with_extreme_exponent_remains_zero(self):
        self.assertEqual(loads('0e-99999999999999999999999'),0.0)
