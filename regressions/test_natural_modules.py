"""Natural-module observations, unknowns and expression attribution stay bound."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('natural_module_check',ROOT/'scripts/check_natural_modules.py')
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)

class NaturalModuleEvidenceTests(unittest.TestCase):
    def test_complete_fixed_cohort_and_nonpooled_rules(self):
        r=check.derive(ROOT/'results/natural-modules')
        self.assertEqual((r['module_cohort'],r['upstream_projects'],r['pairs'],r['suite_processes']),(8,1,32,72))
        self.assertEqual((r['baseline_modules_passed'],r['accepted_embeddings'],r['marked_task_failures']),(7,31,4))
        self.assertEqual(r['same_information_direct_admission'],{'0':5,'1':5,'2':6,'3':7})
        self.assertFalse(r['population_estimate'])

    def test_false_success_source_rule_and_duplicate_reject(self):
        root=ROOT/'results/natural-modules';original=check.read(root/'rows.json')
        for field in ('task-status','source','unavailable','rule','duplicate','raw-count'):
            rows=copy.deepcopy(original)
            bad=next(r for r in rows if r['pair']['original']['validation']['passed'] is False)
            if field=='task-status':bad['marked_task']['passed']=True
            elif field=='source':rows[0]['unmarked_source']='pass\n'
            elif field=='unavailable':
                row=next(r for r in rows if not r['pair']['original']['validation']['available'])
                row['pair']['original']['validation']=dict(row['pair']['original']['validation'],available=True,passed=True)
            elif field=='rule':rows[0]['pair']['edited']['detection']['rule_id']='different'
            elif field=='duplicate':rows[-1]=copy.deepcopy(rows[0])
            elif field=='raw-count':rows[0]['marked_task']['tests_run']-=1
            read=check.read
            def altered(path):return rows if path.name=='rows.json' else read(path)
            with self.subTest(field=field),mock.patch.object(check,'read',side_effect=altered):
                with self.assertRaises(ValueError):check.derive(root)

    def test_rollback_preserves_original_failures(self):
        r=check.verify_attribution(ROOT/'results/natural-modules')
        self.assertEqual(r['restored_passes'],4)
        self.assertTrue(r['original_failures_preserved'])
        self.assertTrue(r['post_hoc_not_population_evidence'])

    def test_statistics_followup_is_separate_and_bound(self):
        r=check.verify_statistics_followup(ROOT/'results/natural-modules')
        self.assertEqual((r['source_programs'],r['passing'],r['tests_per_source'],r['separately_validated_pairs']),(9,9,346,4))
        self.assertFalse(r['statement_coverage_claimed'])
        self.assertTrue(r['original_activity_unknowns_unchanged'])

if __name__=='__main__':unittest.main()
