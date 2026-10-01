"""Analytical counterexample on hypothetical observations, NOT experimental data.

It demonstrates non-identification of conditional net detection change from
specified aggregate components. It does not assert realizability by any particular
watermark algorithm or establish an empirical attack success rate.
"""
from fractions import Fraction as F


def construct() -> dict:
    original_valid = [1, 1, 0, 0]
    edited_valid = [1, 0, 1, 0]
    worlds = {'A': ([1, 1, 0, 0], [0, 1, 1, 0]),
              'B': ([0, 1, 1, 0], [1, 1, 0, 0])}
    result = {}
    for name,(before,after) in worlds.items():
        selected=[i for i,(w,e) in enumerate(zip(original_valid,edited_valid)) if w and e]
        retention=sum(F(min(1,b/a)) if a>0 else F(0) for a,b in zip(before,after))/4
        edited_detection=F(sum(d*v for d,v in zip(after,edited_valid)),sum(edited_valid))
        # Separately stipulated unmarked-pass rate is 1/2 in both worlds.
        preservation=min(F(1),F(sum(edited_valid),4)/F(1,2))
        auc=sum(F(1) if s>F(1,4) else F(1,2) if s==F(1,4) else F(0) for s in before)/4
        result[name]=dict(before=before,after=after,original_valid=original_valid,edited_valid=edited_valid,
             selected=selected,n=len(selected),before_marginal=F(sum(before),4),after_marginal=F(sum(after),4),
             score_retention=retention,edited_pass_detection=edited_detection,pass_ratio=preservation,
             legacy_R=(retention+edited_detection+preservation)/3,clean_AUROC=auc,
             paired_loss=sum(F(before[i]-after[i]) for i in selected)/len(selected))
    return result
