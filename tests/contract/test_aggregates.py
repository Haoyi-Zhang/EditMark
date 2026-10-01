"""Exact-count tests: every input is synthetic, never an empirical observation."""
import copy
from fractions import Fraction
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from editmark_audit.aggregates import analyze, from_pair_stratum
from editmark_audit.analysis import report
from editmark_audit.io import EvidenceError
from editmark_audit.records import Pair
from test_contract import fixture, ROOT


def counts(n=4,b=2,a=2):
    r=report([Pair.parse(fixture(str(i),x,y)) for i,(x,y) in enumerate(itertools.product([False,True],repeat=2))])
    d=from_pair_stratum(r['strata'][0],'synthetic_test')
    d.pop('transitions');d['counts']={'n':n,'before_positive':b,'after_positive':a};d['attempted_pairs']=n
    return d

class AggregateTests(unittest.TestCase):
    def test_exact_net_loss_from_marginals(self):
        r=analyze(counts(5,4,2));self.assertEqual(r['loss']['numerator'],2);self.assertEqual(r['loss']['denominator'],5)
    def test_survival_not_point_identified_generally(self):
        r=analyze(counts());self.assertIsNone(r['survival']);self.assertEqual(r['survival_status'],'partially_identified')
        self.assertEqual((r['survival_bounds']['lower']['decimal'],r['survival_bounds']['upper']['decimal']),(0.,1.))
    def test_survival_zero_identified_in_degenerate_marginal(self):
        self.assertEqual(analyze(counts(4,2,0))['survival']['decimal'],0.)
    def test_survival_one_identified_all_after_positive(self):
        self.assertEqual(analyze(counts(4,2,4))['survival']['decimal'],1.)
    def test_no_initial_detection_survival_undefined(self):
        r=analyze(counts(4,0,2));self.assertIsNone(r['survival']);self.assertIsNone(r['survival_bounds']['lower'])
    def test_empty_cohort_undefined(self):
        r=analyze(counts(0,0,0));self.assertIsNone(r['loss']);self.assertEqual(r['status'],'undefined_empty_cohort')
    def test_exact_counts_above_binary64_limit_remain_exact(self):
        n=2**60;r=analyze(counts(n,n-1,n-2));self.assertEqual(r['loss']['numerator'],1);self.assertEqual(r['loss']['denominator'],n)
    def test_rounded_percentage_rejected(self):
        d=counts();d['counts']['before_positive']=.5
        with self.assertRaises(EvidenceError):analyze(d)
    def test_boolean_count_rejected(self):
        d=counts();d['counts']['n']=True
        with self.assertRaises(EvidenceError):analyze(d)
    def test_negative_count_rejected(self):
        with self.assertRaises(EvidenceError):analyze(counts(4,-1,2))
    def test_excess_count_rejected(self):
        with self.assertRaises(EvidenceError):analyze(counts(4,5,2))
    def test_transition_marginal_conflict(self):
        d=counts();d['transitions']={'n00':4,'n01':0,'n10':0,'n11':0}
        with self.assertRaises(EvidenceError):analyze(d)
    def test_exclusion_partition_required(self):
        d=counts();d['attempted_pairs']=7
        with self.assertRaises(EvidenceError):analyze(d)
    def test_known_exclusion_partition_accepted(self):
        d=counts();d['attempted_pairs']=7;d['first_exclusion_counts']={'unchanged':3}
        self.assertEqual(analyze(d)['attempted_pairs'],7)
    def test_exclusion_typo_rejected(self):
        d=counts();d['first_exclusion_counts']={'unchangd':1}
        with self.assertRaises(EvidenceError):analyze(d)
    def test_missing_rule_rejected(self):
        d=counts();d['decision_rule']['rule_id']=''
        with self.assertRaises(EvidenceError):analyze(d)
    def test_missing_hash_rejected(self):
        d=counts();d.pop('cohort_sha256')
        with self.assertRaises(EvidenceError):analyze(d)
    def test_different_analysis_contract_rejected(self):
        d=counts();d['analysis_contract']='all_supported'
        with self.assertRaises(EvidenceError):analyze(d)
    def test_controls_rejected(self):
        d=counts();d['population']='unmarked_control'
        with self.assertRaises(EvidenceError):analyze(d)
    def test_synthetic_label_preserved(self):
        self.assertEqual(analyze(counts())['synthetic_warning'],'NOT AN EMPIRICAL RESULT')
    def test_report_roundtrip_matches_pairs(self):
        p=report([Pair.parse(fixture('a')),Pair.parse(fixture('b',before=False,after=True)),Pair.parse(fixture('c',after=True))])
        s=p['strata'][0];c=from_pair_stratum(s,p['evidence_kind']);r=analyze(c)
        self.assertEqual(r['loss']['decimal'],s['estimates']['changed_only']['loss'])
        self.assertEqual(r['survival']['decimal'],s['estimates']['changed_only']['survival'])
        self.assertEqual(r['cohort_sha256'],s['estimates']['changed_only']['cohort_sha256'])
    def test_empty_pair_export_refused(self):
        r=report([Pair.parse(fixture(edited_pass=False))])
        with self.assertRaises(EvidenceError):from_pair_stratum(r['strata'][0],r['evidence_kind'])
    def test_sharp_survival_bounds_exhaustive_through_five(self):
        for n in range(1,6):
            possible={}
            for bs in itertools.product([0,1],repeat=n):
                for es in itertools.product([0,1],repeat=n):
                    b,a=sum(bs),sum(es)
                    if b:possible.setdefault((b,a),set()).add(Fraction(sum(x*y for x,y in zip(bs,es)),b))
            for (b,a),vals in possible.items():
                r=analyze(counts(n,b,a));lo=r['survival_bounds']['lower'];hi=r['survival_bounds']['upper']
                self.assertEqual((Fraction(lo['numerator'],lo['denominator']),Fraction(hi['numerator'],hi['denominator'])),(min(vals),max(vals)))

class AggregateCLITests(unittest.TestCase):
    def call(self,doc,*args):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'counts.json';out=Path(td)/'report.json'
            src.write_text(json.dumps(doc));p=subprocess.run([sys.executable,'-m','editmark_audit.aggregates',str(src),'--out',str(out),*args],cwd=ROOT,capture_output=True,text=True)
            return p.returncode, out.exists(), json.loads(out.read_text()) if out.exists() else None
    def test_cli_valid(self):self.assertEqual(self.call(counts())[0],0)
    def test_cli_requires_observations(self):self.assertEqual(self.call(counts(),'--require-observations')[:2],(2,False))
    def test_cli_empty_required_estimate_keeps_report(self):self.assertEqual(self.call(counts(0,0,0),'--require-estimate')[:2],(3,True))
    def test_cli_legacy_composite_rejected(self):self.assertEqual(self.call({'D':.7,'R':.4})[:2],(2,False))
    def test_cli_mixed_evidence_rejected(self):
        a,b=counts(),counts();b['evidence_kind']='saved_observation'
        self.assertEqual(self.call([a,b])[:2],(2,False))
    def test_cli_duplicate_cohort_rejected(self):self.assertEqual(self.call([counts(),counts()])[:2],(2,False))
