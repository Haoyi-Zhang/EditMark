"""Finite-population identification bounds, NOT sampling confidence intervals.

Input counts must refer to the same exact finite population. Rounded, weighted,
or differently filtered summary rates are not valid inputs. Classical set-count
bounds are applied here; their mathematics is not claimed as a new theorem.
"""
from __future__ import annotations
from fractions import Fraction
from .io import EvidenceError


def _count(x: int, name: str) -> int:
    if type(x) is not int or x < 0: raise EvidenceError(f'{name}: expected nonnegative integer count')
    return x


def intersection_bounds(total: int, first: int, second: int) -> tuple[int,int]:
    """Sharp bounds for |A intersect B| from |A|, |B| in one population."""
    for n,x in (('total',total),('first',first),('second',second)): _count(x,n)
    if max(first,second)>total: raise EvidenceError('A marginal count exceeds the population')
    return max(0,first+second-total), min(first,second)


def selected_loss_bounds(total: int, selected: int, before: int, after: int) -> tuple[Fraction,Fraction] | None:
    """Sharp L=mean(D_before-D_after | selected) from exact detection marginals.

    Requires only the selected-set size, not its relation to detection. If that
    relation or the full before/after joint table is known, tighter bounds may
    be possible. 'selected=0' has no conditional estimand and returns None.
    """
    for n,x in (('total',total),('selected',selected),('before',before),('after',after)): _count(x,n)
    if max(selected,before,after)>total: raise EvidenceError('A count exceeds the population')
    if selected==0: return None
    b0,b1 = intersection_bounds(total,selected,before)
    a0,a1 = intersection_bounds(total,selected,after)
    return Fraction(b0-a1,selected),Fraction(b1-a0,selected)


def incomplete_loss_bounds(total: int, complete: int, observed_difference_sum: int) -> tuple[Fraction,Fraction] | None:
    """Worst-case bounds when both decisions are missing on remaining target pairs."""
    _count(total,'total');_count(complete,'complete')
    if complete>total or type(observed_difference_sum) is not int or abs(observed_difference_sum)>complete:
        raise EvidenceError('Inconsistent count or difference sum')
    if total==0: return None
    missing=total-complete
    return (Fraction(observed_difference_sum-missing,total),Fraction(observed_difference_sum+missing,total))


def decision_difference_range(before: bool | None, after: bool | None) -> tuple[int,int]:
    """Loss range sharp over unrestricted completion of Boolean/null inputs.

    A known decision is retained. Scores and rules are not inputs here, so this
    range can be conservative relative to richer supplied score/rule facts.
    """
    for x in (before,after):
        if x is not None and type(x) is not bool: raise EvidenceError('Expected boolean or null decision')
    blo,bhi=(0,1) if before is None else (int(before),int(before))
    alo,ahi=(0,1) if after is None else (int(after),int(after))
    return blo-ahi,bhi-alo
