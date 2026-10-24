"""Opt-in tests; execute only the reviewed finite SemCodebook case."""
import json
import copy
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.measure_semcodebook_cpu import AFTER,BASE,DOMAIN,check_upstream,direct,run,validate
class SemCodebookCPUTests(unittest.TestCase):
    def test_source_git_blobs_match_pinned_owner_repository(self):
        self.assertEqual(len(check_upstream()['files']),21)
    def test_unreviewed_source_is_not_executed(self):
        with self.assertRaisesRegex(ValueError,'whitelist'):
            validate("def solve(items):\n    return 123\n")
    def test_same_finite_function_contract(self):
        self.assertEqual(len(DOMAIN),156)
        for source in (BASE,AFTER):
            passed,calls=validate(source)
            self.assertTrue(passed)
            self.assertEqual(len(calls),156)
    def test_occupied_output_refused(self):
        with self.assertRaisesRegex(ValueError,'new'):
            run(ROOT/'results')
    def test_actual_whole_payload_cohort_and_same_information_replay(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='semcodebook-test-') as td:
            with contextlib.redirect_stdout(io.StringIO()):
                value=run(Path(td)/'run')
            self.assertEqual((value['pairs'],value['functional_calls'],value['functional_failures']),(32,9984,0))
            self.assertEqual((value['distinct_before_source_texts'],value['distinct_source_texts']),(4,9))
            records={r['edit']:r['ordinary'] for r in value['direct_comparisons']}
            self.assertEqual(records['builtin-sum-refactor']['transitions'],dict(n00=1,n01=0,n10=8,n11=7))
            self.assertEqual(records['builtin-sum-refactor']['change'],[-1,2])
            self.assertEqual(records['builtin-sum-refactor']['survival'],[7,15])
            self.assertEqual(records['review-comment']['transitions'],dict(n00=1,n01=0,n10=0,n11=15))
            for name,outcome in value['claim_outcomes'].items():
                self.assertTrue(outcome['raw_pair_membership_rederived'])
                self.assertEqual(outcome['exit_code'],0 if name in ('positive_decision_change','detection_survival') else 3)
    def test_retained_raw_basis_rule_and_source_links(self):
        p=ROOT/'results/semcodebook_cpu/pairs.jsonl'
        rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows),32)
        self.assertEqual({r['cluster_id'] for r in rows},{'one-arithmetic-task'})
        for edit in ('review-comment','builtin-sum-refactor'):
            self.assertEqual(direct([r for r in rows if r['edit']==edit])['n'],16)
    def test_reference_rejects_unknown_contract_and_rule(self):
        p=ROOT/'results/semcodebook_cpu/pairs.jsonl'
        original=json.loads(p.read_text(encoding='utf-8').splitlines()[0])
        for group,field,value in (('validation','contract_id',''),('validation','available',False),
                                  ('detection','rule_id',''),('detection','detected',None)):
            row=copy.deepcopy(original)
            for side in ('original','edited'):
                row[side][group][field]=value
            with self.assertRaises(ValueError):
                direct([row])
if __name__=='__main__':
    unittest.main()
