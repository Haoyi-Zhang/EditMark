"""Exact-arithmetic validation of hypothetical examples, not model experiments."""
import unittest
from fractions import Fraction as F
from editmark_audit.witness import construct
from editmark_audit.bounds import selected_loss_bounds

class WitnessTests(unittest.TestCase):
    def test_same_legacy_components(self):
        a,b=construct().values()
        for k in ('before_marginal','after_marginal','score_retention','edited_pass_detection','pass_ratio','legacy_R','clean_AUROC','n'):
            self.assertEqual(a[k],b[k],k)
    def test_opposite_conditional_changes(self):
        a,b=construct().values();self.assertEqual(a['paired_loss'],1);self.assertEqual(b['paired_loss'],-1)
    def test_legacy_mean_exact(self):
        for w in construct().values(): self.assertEqual(w['legacy_R'],F(7,12))
    def test_selected_bound_attained(self):
        a,b=construct().values();lo,hi=selected_loss_bounds(4,1,2,2)
        self.assertEqual((lo,hi),(b['paired_loss'],a['paired_loss']))
    def test_conditional_marginals_suffice_for_loss(self):
        for w in construct().values():
            before=sum(w['before'][i] for i in w['selected'])
            after=sum(w['after'][i] for i in w['selected'])
            self.assertEqual(F(before-after,w['n']),w['paired_loss'])
    def test_shift_changes_ratio_not_decision(self):
        self.assertNotEqual(F(1,2),F(11,12))
        self.assertEqual((2>=F(3,2),1>=F(3,2)),(12>=F(23,2),11>=F(23,2)))
