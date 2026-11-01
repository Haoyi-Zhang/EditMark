"""Saved method execution is bound to source, task outcome and key stratum."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('released_style_check',ROOT/'scripts/check_released_style.py')
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)

class ReleasedMethodEvidence(unittest.TestCase):
    def test_full_saved_design_and_separate_key_cohorts(self):
        r=check.derive(ROOT/'results/released-style')
        self.assertEqual(r['pairs'],32)
        self.assertEqual(r['marked_task_failures'],4)
        self.assertEqual(len(r['key_strata']),4)
        self.assertFalse(r['cross_key_TPR_FPR_or_survival_estimate'])
        self.assertEqual(r['same_information_direct_admission'],22)

    def test_disagreement_cannot_be_relabelled_as_success(self):
        rows=json.loads((ROOT/'results/released-style/rows.json').read_text(encoding='utf-8'))
        index=next(i for i,r in enumerate(rows) if not r['marked_observation']['passed'])
        for field in ('observation','source','changed','decision-rule','duplicate'):
            changed=copy.deepcopy(rows)
            if field=='observation':changed[index]['marked_observation']['passed']=True
            if field=='source':changed[0]['pair']['original']['source']='def probe(): return 0\n'
            if field=='changed':changed[0]['pair']['changed']=not changed[0]['pair']['changed']
            if field=='decision-rule':changed[0]['pair']['original']['detection']['rule_id']='unbound'
            if field=='duplicate':changed[-1]=changed[0]
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                path=Path(d);(path/'rows.json').write_text(json.dumps(changed),encoding='utf-8')
                with self.assertRaises(ValueError):check.derive(path)

if __name__=='__main__':unittest.main()
