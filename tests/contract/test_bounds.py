"""Exhaustive finite examples check both validity and attainability of bounds."""
from __future__ import annotations
from fractions import Fraction
import itertools
import unittest
from editmark_audit.bounds import intersection_bounds, selected_loss_bounds, incomplete_loss_bounds, decision_difference_range
from editmark_audit.io import EvidenceError

class BoundTests(unittest.TestCase):
    def test_intersection_exhaustive(self):
        for n in range(0,7):
            observed={}
            bits=list(itertools.product((0,1),repeat=n))
            for a in bits:
                for b in bits:
                    key=(sum(a),sum(b));v=sum(x*y for x,y in zip(a,b))
                    lo,hi=observed.get(key,(v,v));observed[key]=(min(lo,v),max(hi,v))
            for (a,b),bounds in observed.items():self.assertEqual(intersection_bounds(n,a,b),bounds)
    def test_selected_loss_exhaustive_sharp(self):
        for n in range(1,6):
            observed={}
            bits=list(itertools.product((0,1),repeat=n))
            for s in bits:
                q=sum(s)
                if not q:continue
                for a in bits:
                    for b in bits:
                        key=(q,sum(a),sum(b));v=Fraction(sum(z*(x-y) for z,x,y in zip(s,a,b)),q)
                        lo,hi=observed.get(key,(v,v));observed[key]=(min(lo,v),max(hi,v))
            for (q,a,b),bounds in observed.items():self.assertEqual(selected_loss_bounds(n,q,a,b),bounds)
    def test_missing_bound_exhaustive(self):
        for complete in range(0,5):
            for missing in range(0,5):
                n=complete+missing
                for s in range(-complete,complete+1):
                    out=incomplete_loss_bounds(n,complete,s)
                    if n==0:self.assertIsNone(out);continue
                    values=[Fraction(s+sum(z),n) for z in itertools.product((-1,0,1),repeat=missing)]
                    self.assertEqual(out,(min(values),max(values)))
    def test_partial_decision_bounds_exhaustive(self):
        for a,b in itertools.product((None,False,True),repeat=2):
            aa=(False,True) if a is None else (a,);bb=(False,True) if b is None else (b,)
            values=[int(x)-int(y) for x,y in itertools.product(aa,bb)]
            self.assertEqual(decision_difference_range(a,b),(min(values),max(values)))
    def test_selected_zero_is_undefined(self):self.assertIsNone(selected_loss_bounds(4,0,2,2))
    def test_no_selection_is_known_marginal_difference(self):self.assertEqual(selected_loss_bounds(10,10,6,4),(Fraction(1,5),Fraction(1,5)))
    def test_all_decisions_missing_full_range(self):self.assertEqual(incomplete_loss_bounds(10,0,0),(Fraction(-1),Fraction(1)))
    def test_rounding_rates_cannot_be_counts(self):
        with self.assertRaises(EvidenceError):intersection_bounds(100,1.25,3.4)
    def test_negative_count_rejected(self):
        with self.assertRaises(EvidenceError):intersection_bounds(4,-1,2)
    def test_boolean_count_rejected(self):
        with self.assertRaises(EvidenceError):intersection_bounds(4,True,2)
    def test_oversize_count_rejected(self):
        with self.assertRaises(EvidenceError):selected_loss_bounds(4,5,2,2)
    def test_invalid_sum_rejected(self):
        with self.assertRaises(EvidenceError):incomplete_loss_bounds(4,1,2)
    def test_noninteger_difference_rejected(self):
        with self.assertRaises(EvidenceError):incomplete_loss_bounds(4,2,0.5)
    def test_unknown_decision_boolean_type_rejected(self):
        with self.assertRaises(EvidenceError):decision_difference_range(1,None)

if __name__=='__main__':unittest.main()
