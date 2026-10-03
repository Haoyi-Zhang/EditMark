"""Cross-check final artifact facts against the included source inventory."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]

class ArtifactFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=json.loads((ROOT/'analysis/frozen_audit.json').read_text(encoding='utf-8'))
    def test_release_and_configured_populations_are_distinct(self):
        self.assertEqual(self.audit['release_records'],1662)
        self.assertEqual(self.audit['canonical_source_records'],1605)
        self.assertEqual(self.audit['excluded_source_records'],57)
    def test_source_files_reproduce_inventory_counts(self):
        total=0
        for row in self.audit['corpus']:
            path=ROOT/row['release_file']
            count=sum(bool(line.strip()) for line in path.read_text(encoding='utf-8').splitlines())
            self.assertEqual(count,row['release_records'],row['source_group'])
            total+=count
        self.assertEqual(total,self.audit['release_records'])
    def test_mbxp_exclusion_is_visible(self):
        mbxp=next(row for row in self.audit['corpus'] if row['source_group']=='public_mbxp_5lang')
        self.assertEqual((mbxp['release_records'],mbxp['canonical_records'],mbxp['excluded_records']),(200,143,57))
        self.assertEqual(self.audit['mbxp_canonical_go_records'],0)
    def test_unavailable_evidence_is_not_claimed(self):
        availability=self.audit['availability']
        for key in ('strict_paired_detection','transformed_negative_controls','no_op_sensitivity','exact_joint_admission_counts'):
            self.assertIn('not',availability[key].lower())
    def test_basis_states_no_experiment_rerun(self):
        self.assertIn('no experiment rerun',self.audit['basis'].lower())
