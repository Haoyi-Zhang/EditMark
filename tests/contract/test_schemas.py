"""Check shipped schema identity; semantic validation is performed by parser tests."""
import json
from pathlib import Path
import unittest
from editmark_audit.schemas import pair_schema,count_schema,certificate_schema,certificate_bundle_schema
from editmark_audit.aggregates import analyze
from editmark_audit.records import Pair
from editmark_audit.io import EvidenceError, load_rows
ROOT=Path(__file__).resolve().parents[2]
class SchemaTests(unittest.TestCase):
    def test_shipped_pair_schema_matches_export(self):
        self.assertEqual(json.loads((ROOT/'schemas/pair.schema.json').read_text()),pair_schema())
    def test_shipped_count_schema_matches_export(self):
        self.assertEqual(json.loads((ROOT/'schemas/cohort_counts.schema.json').read_text()),count_schema())
        self.assertEqual(json.loads((ROOT/'schemas/claim_certificate.schema.json').read_text()),certificate_schema())
        self.assertEqual(json.loads((ROOT/'schemas/claim_certificate_bundle.schema.json').read_text()),certificate_bundle_schema())
    def test_pair_example_is_explicitly_synthetic(self):
        self.assertEqual({Pair.parse(d).evidence_kind for d in load_rows(ROOT/'examples/synthetic_pairs.jsonl')},{'synthetic_test'})
    def test_count_example_is_explicitly_synthetic(self):
        self.assertEqual(analyze(load_rows(ROOT/'examples/synthetic_cohort_counts.json')[0])['evidence_kind'],'synthetic_test')
    def test_explicit_exclusions_required(self):
        d=load_rows(ROOT/'examples/synthetic_cohort_counts.json')[0];d.pop('first_exclusion_counts')
        with self.assertRaises(EvidenceError):analyze(d)
