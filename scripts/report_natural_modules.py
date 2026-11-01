"""Generate manuscript numbers and exact per-key rows from verified records."""
import argparse
from fractions import Fraction
from pathlib import Path
from check_natural_modules import derive,verify_attribution,verify_statistics_followup,ROOT
def render(out):
    out.mkdir(parents=True,exist_ok=True);data=derive(ROOT/'results/natural-modules');attribution=verify_attribution(ROOT/'results/natural-modules')
    values=dict(NaturalModules=data['module_cohort'],NaturalAttempts=data['pairs'],NaturalProcesses=data['suite_processes'],
        NaturalBaselines=data['baseline_modules_passed'],NaturalEmbeddings=data['accepted_embeddings'],
        NaturalFailures=data['marked_task_failures'],NaturalAdmitted=sum(data['same_information_direct_admission'].values()),NaturalRestored=attribution['restored_passes'])
    follow=verify_statistics_followup(ROOT/'results/natural-modules')
    values.update(NaturalFollowupPairs=follow['separately_validated_pairs'],NaturalFollowupTests=follow['tests_per_source'])
    (out/'natural_module_values.tex').write_text(''.join('\\newcommand{\\'+key+'}{'+str(value)+'}\n' for key,value in values.items()),encoding='utf-8')
    rows=[]
    for key,r in data['key_strata'].items():
        e=r['first_exclusion_counts'];delta=r['net_detection_change_exact'];n,d=delta['numerator'],delta['denominator']
        text=str(n) if d==1 else ('-' if n<0 else '')+r'\frac{'+str(abs(n))+'}{'+str(d)+'}'
        rows.append('K'+str(int(key)+1)+' & 8 & '+str(e.get('unsupported',0))+' & '+str(e.get('original_checks_failed',0)+e.get('edited_checks_failed',0))+' & '+str(e.get('original_validation_unavailable',0)+e.get('edited_validation_unavailable',0))+' & '+str(r['admitted_pairs'])+' & $'+text+r'$ \\'+'\n')
    (out/'natural_module_rows.tex').write_text(''.join(rows)+r'\bottomrule'+'\n',encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);render(p.parse_args().out)
