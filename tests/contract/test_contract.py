"""All observations here are SYNTHETIC SOFTWARE FIXTURES, not experiment results."""
from __future__ import annotations
import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from editmark_audit.records import Pair, Detection, deduplicate, compare_score
from editmark_audit.analysis import summarize, report, exclusion
from editmark_audit.io import EvidenceError, loads, load_rows, write_new_json

ROOT=Path(__file__).resolve().parents[2]


def fixture(name='s1', before=True, after=False, original_pass=True, edited_pass=True):
    def program(code,passed,detected):
        return dict(source=code,validation=dict(available=True,passed=passed,contract_id='synthetic-tests'),
             detection=dict(available=True,detected=detected,score=1.0 if detected else 0.0,
                            threshold=.5,comparator='ge',rule_id='synthetic-rule'))
    return dict(schema_version='editmark-pair',population='watermarked_positive',evidence_kind='synthetic_test',experiment_id='SYNTHETIC',
       method='toy',model='toy',source_group='toy',language='python',sample_id=name,replicate_id='r0',
       edit='format',variant_id='v0',cluster_id=name,supported=True,changed=True,
       original=program(f'x = {name!r}',original_pass,before),edited=program(f'x={name!r}',edited_pass,after))


class PairContractTests(unittest.TestCase):
    def test_extreme_json_exponent_rejected(self):
        with self.assertRaises(EvidenceError): loads('{"anything":1e999}')
    def test_giant_int_score_rejected(self):
        d=fixture();d['original']['detection']['score']=10**1000
        with self.assertRaises(EvidenceError): Pair.parse(d)
    def test_thresholds_are_separate_strata(self):
        a,b=fixture('a'),fixture('b')
        for side in ('original','edited'): b[side]['detection']['threshold']=.6
        self.assertEqual(report([Pair.parse(a),Pair.parse(b)])['stratum_count'],2)
    def test_comparators_are_separate_strata(self):
        a,b=fixture('a'),fixture('b')
        for side in ('original','edited'): b[side]['detection']['comparator']='gt'
        self.assertEqual(report([Pair.parse(a),Pair.parse(b)])['stratum_count'],2)
    def test_missing_and_present_threshold_strata_sort(self):
        a,b=fixture('a'),fixture('b')
        for side in ('original','edited'): b[side]['detection']['threshold']=None
        self.assertEqual(report([Pair.parse(a),Pair.parse(b)])['stratum_count'],2)
    def test_population_required(self):
        d=fixture();del d['population']
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_unmarked_controls_not_labelled_tpr(self):
        d=fixture();d['population']='unmarked_control'
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_cohort_digest_order_invariant(self):
        a,b=Pair.parse(fixture('a')),Pair.parse(fixture('b',after=True))
        self.assertEqual(summarize([a,b])['cohort_sha256'],summarize([b,a])['cohort_sha256'])
    def test_cohort_digest_changes_with_decision(self):
        a,b=Pair.parse(fixture()),Pair.parse(fixture(after=True))
        self.assertNotEqual(summarize([a])['cohort_sha256'],summarize([b])['cohort_sha256'])
    def test_empty_cohort_digest_null(self):
        self.assertIsNone(summarize([])['cohort_sha256'])
    def test_positive_pair(self):
        r=summarize([Pair.parse(fixture())]);self.assertEqual((r['n10'],r['loss'],r['survival']),(1,1.,0.))
    def test_all_four_transitions(self):
        ps=[Pair.parse(fixture(str(i),a,b)) for i,(a,b) in enumerate(itertools.product([False,True],repeat=2))]
        r=summarize(ps);self.assertEqual([r[k] for k in ('n00','n01','n10','n11')],[1]*4)
        self.assertEqual((r['loss'],r['tpr_before'],r['tpr_after'],r['survival']),(0.,.5,.5,.5))
    def test_count_partition(self):
        ps=[Pair.parse(fixture()),Pair.parse(fixture('bad',edited_pass=False))]
        r=summarize(ps);self.assertEqual(r['attempted_pairs'],r['admitted_pairs']+sum(r['first_exclusion_counts'].values()))
    def test_no_denominator_is_null(self):
        r=summarize([]);self.assertIsNone(r['loss']);self.assertIsNone(r['survival'])
    def test_no_initial_detection_survival_null(self):
        r=summarize([Pair.parse(fixture(before=False,after=True))]);self.assertIsNone(r['survival']);self.assertEqual(r['loss'],-1.)
    def test_original_failure_excluded(self):
        p=Pair.parse(fixture(original_pass=False));self.assertEqual(exclusion(p),'original_checks_failed')
    def test_edited_failure_excluded(self):
        p=Pair.parse(fixture(edited_pass=False));self.assertEqual(exclusion(p),'edited_checks_failed')
    def test_absent_validation_excluded_not_failed(self):
        d=fixture();d['edited']['validation']={};self.assertEqual(exclusion(Pair.parse(d)),'edited_validation_unavailable')
    def test_unknown_pass_excluded(self):
        d=fixture();d['edited']['validation']['passed']=None;self.assertEqual(exclusion(Pair.parse(d)),'edited_checks_unknown')
    def test_missing_check_contract(self):
        d=fixture();del d['edited']['validation']['contract_id'];self.assertEqual(exclusion(Pair.parse(d)),'check_contract_unknown')
    def test_changed_check_contract(self):
        d=fixture();d['edited']['validation']['contract_id']='different';self.assertEqual(exclusion(Pair.parse(d)),'check_contract_changed')
    def test_rule_identity_is_per_side(self):
        d=fixture();d['edited']['detection']['rule_id']='other-detector';self.assertEqual(exclusion(Pair.parse(d)),'decision_rule_changed')
    def test_missing_rule_not_filled_by_hashing_empty_config(self):
        d=fixture();del d['edited']['detection']['rule_id'];self.assertEqual(exclusion(Pair.parse(d)),'decision_rule_unknown')
    def test_threshold_change(self):
        d=fixture();d['edited']['detection']['threshold']=.6;self.assertEqual(exclusion(Pair.parse(d)),'decision_rule_changed')
    def test_comparator_change(self):
        d=fixture();d['edited']['detection']['comparator']='gt';self.assertEqual(exclusion(Pair.parse(d)),'decision_rule_changed')
    def test_score_decision_contradiction(self):
        d=fixture();d['edited']['detection']['detected']=True
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_comparison_boundaries(self):
        self.assertEqual([compare_score(.5,.5,c) for c in ('ge','gt','le','lt')],[True,False,True,False])
    def test_unknown_comparator(self):
        with self.assertRaises(EvidenceError):compare_score(1.,0.,'equal')
    def test_unavailable_detection_not_negative(self):
        d=fixture();d['edited']['detection'].update(available=False,detected=None,score=None)
        r=summarize([Pair.parse(d)]);self.assertEqual(r['admitted_pairs'],0)
        self.assertEqual((r['missing_decision_bounds']['lower'],r['missing_decision_bounds']['upper']),(0.,1.))
    def test_partially_observed_negative_narrows_bound(self):
        d=fixture(before=False);d['edited']['detection'].update(available=False,detected=None,score=None)
        r=summarize([Pair.parse(d)]);self.assertEqual((r['missing_decision_bounds']['lower'],r['missing_decision_bounds']['upper']),(-1.,0.))
    def test_missing_validation_not_in_known_target_bounds(self):
        d=fixture();d['edited']['validation']={};r=summarize([Pair.parse(d)])
        self.assertEqual(r['missing_decision_bounds']['target_pairs'],0);self.assertIsNone(r['missing_decision_bounds']['lower'])
    def test_unavailable_detection_cannot_carry_false_decision(self):
        d=fixture();d['edited']['detection']['available']=False
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_unavailable_validation_cannot_carry_fail(self):
        d=fixture();d['edited']['validation'].update(available=False,passed=False)
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_support_missing_is_unknown(self):
        d=fixture();del d['supported'];self.assertEqual(exclusion(Pair.parse(d)),'support_unknown')
    def test_unsupported_not_noop(self):
        d=fixture();d['supported']=False;self.assertEqual(exclusion(Pair.parse(d)),'unsupported')
    def test_known_noop_sensitivity(self):
        d=fixture();d['edited']['source']=d['original']['source'];d['changed']=False
        p=Pair.parse(d);self.assertEqual(exclusion(p),'unchanged');self.assertIsNone(exclusion(p,'include_known_noops'))
    def test_change_evidence_derived(self):
        d=fixture();del d['changed'];self.assertTrue(Pair.parse(d).changed)
    def test_saved_flag_without_identity_not_strict(self):
        d=fixture();del d['original']['source'];self.assertEqual(exclusion(Pair.parse(d)),'source_identity_missing')
    def test_unknown_change_only_diagnostic(self):
        d=fixture();d['changed']=None;del d['original']['source'];del d['edited']['source'];p=Pair.parse(d)
        self.assertEqual(exclusion(p),'change_unknown');self.assertIsNone(exclusion(p,'all_supported'))
    def test_flag_hash_contradiction(self):
        d=fixture();d['changed']=False
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_single_source_hash_verified_even_without_other_source(self):
        d=fixture();d['original']['source_sha256']='a'*64;del d['edited']['source']
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_hash_format(self):
        d=fixture();d['original']['source_sha256']='wrong'
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_string_boolean(self):
        d=fixture();d['supported']='false'
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_numeric_boolean(self):
        d=fixture();d['supported']=1
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_nonfinite_score(self):
        d=fixture();d['original']['detection']['score']=float('nan')
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_boolean_threshold(self):
        d=fixture();d['original']['detection']['threshold']=True
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_exact_duplicates(self):
        p=Pair.parse(fixture());r=report([p,p]);self.assertEqual((r['unique_pair_count'],r['identical_duplicates_removed']),(1,1))
    def test_hash_distinguishes_conflicting_duplicates(self):
        p=Pair.parse(fixture());d=fixture();d['edited']['source']='different source'
        with self.assertRaises(EvidenceError):report([p,Pair.parse(d)])
    def test_inconsistent_original_across_edits(self):
        p=Pair.parse(fixture());d=fixture();d['edit']='rename';d['original']['source']='different original'
        with self.assertRaises(EvidenceError):report([p,Pair.parse(d)])
    def test_distinct_originals_not_edit_weighted(self):
        p=Pair.parse(fixture());q=replace(p,edit='another');r=summarize([p,q]);self.assertEqual(r['distinct_originals']['count'],1)
    def test_replicates_do_not_collapse(self):
        p=Pair.parse(fixture());q=replace(p,replicate_id='r1');self.assertEqual(report([p,q])['unique_pair_count'],2)
    def test_no_default_cross_method_pool(self):
        p=Pair.parse(fixture());q=replace(p,method='other');r=report([p,q]);self.assertEqual(r['stratum_count'],2);self.assertNotIn('pooled',r)
    def test_synthetic_never_mixed_with_observation(self):
        p=Pair.parse(fixture());q=replace(p,evidence_kind='saved_observation')
        with self.assertRaises(EvidenceError):report([p,q])
    def test_explicit_synthetic_banner(self):
        self.assertEqual(report([Pair.parse(fixture())])['synthetic_warning'],'NOT AN EMPIRICAL RESULT')
    def test_old_schema_not_silently_upgraded(self):
        d=fixture();d['schema_version']='editmark-pair-v1'
        with self.assertRaises(EvidenceError):Pair.parse(d)
    def test_unknown_mode_even_empty(self):
        with self.assertRaises(EvidenceError):summarize([],'unknown')
    def test_no_network_path(self):
        with patch('socket.socket',side_effect=AssertionError('network forbidden')):
            self.assertEqual(report([Pair.parse(fixture())])['unique_pair_count'],1)
    def test_functional_preservation_denominator(self):
        ps=[Pair.parse(fixture()),Pair.parse(fixture('bad',edited_pass=False)),Pair.parse(fixture('origbad',original_pass=False))]
        r=summarize(ps);self.assertEqual(r['functional_preservation'],dict(numerator=1,denominator=2,rate=.5))
    def test_missing_contract_not_preservation(self):
        d=fixture();d['edited']['validation']['contract_id']='wrong';r=summarize([Pair.parse(d)])
        self.assertEqual(r['functional_preservation']['denominator'],0)
    def test_shifted_scores_preserve_decisions_and_loss(self):
        p=Pair.parse(fixture());d=fixture()
        for side in ('original','edited'):
            d[side]['detection']['score']+=10;d[side]['detection']['threshold']+=10
        self.assertEqual(summarize([p])['loss'],summarize([Pair.parse(d)])['loss'])


class StrictIOTests(unittest.TestCase):
    def test_duplicate_keys_rejected(self):
        with self.assertRaises(EvidenceError):loads('{"x":1,"x":2}')
    def test_nonfinite_constants_rejected(self):
        for text in ('NaN','Infinity','-Infinity'):
            with self.assertRaises(EvidenceError):loads('{"x":'+text+'}')
    def test_utf8_bom_accepted(self):self.assertEqual(loads('\ufeff{"x":1}'),{'x':1})
    def test_empty_jsonl_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.jsonl';p.write_text(' \n')
            with self.assertRaises(EvidenceError):load_rows(p)
    def test_nonobject_rows_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json';p.write_text('[1]')
            with self.assertRaises(EvidenceError):load_rows(p)
    def test_output_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json';p.write_text('old')
            with self.assertRaises(FileExistsError):write_new_json(p,{'new':1})
            self.assertEqual(p.read_text(),'old');self.assertEqual(len(list(Path(t).iterdir())),1)
    def test_symlink_not_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'original';p.write_text('old');q=Path(t)/'out';q.symlink_to(p)
            with self.assertRaises(FileExistsError):write_new_json(q,{'new':1})
            self.assertEqual(p.read_text(),'old')
    def test_nonfinite_output_not_created(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'out.json'
            with self.assertRaises(ValueError):write_new_json(p,{'x':float('nan')})
            self.assertFalse(p.exists())
    def test_cli_roundtrip_synthetic(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'in.json',Path(t)/'out.json';a.write_text(json.dumps([fixture()]))
            r=subprocess.run([sys.executable,'-m','editmark_audit',str(a),'--out',str(b)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(loads(b.read_text())['evidence_kind'],'synthetic_test')
    def test_cli_synthetic_refused_as_empirical(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'in.json',Path(t)/'out.json';a.write_text(json.dumps([fixture()]))
            r=subprocess.run([sys.executable,'-m','editmark_audit',str(a),'--out',str(b),'--require-observations'],cwd=ROOT,capture_output=True)
            self.assertEqual(r.returncode,2);self.assertFalse(b.exists())
    def test_cli_no_estimate_exit_status(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'in.json',Path(t)/'out.json';a.write_text(json.dumps([fixture(edited_pass=False)]))
            r=subprocess.run([sys.executable,'-m','editmark_audit',str(a),'--out',str(b),'--require-estimate'],cwd=ROOT,capture_output=True)
            self.assertEqual(r.returncode,3);self.assertTrue(b.exists())
    def test_cli_aggregate_refused(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'in.json',Path(t)/'out.json';a.write_text('{"robustness":0.5}')
            r=subprocess.run([sys.executable,'-m','editmark_audit',str(a),'--out',str(b)],cwd=ROOT,capture_output=True)
            self.assertEqual(r.returncode,2);self.assertFalse(b.exists())

if __name__=='__main__':unittest.main()
