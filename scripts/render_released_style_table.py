"""One manuscript table from saved per-key admissions; no experiment runs."""
import argparse
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('released_style_check',ROOT/'scripts/check_released_style.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def render(destination):
    result=module.derive(ROOT/'results/released-style');rows=[]
    for key,record in sorted(result['key_strata'].items()):
        exclusions=record['first_exclusion_counts']
        values=[exclusions.get(name,0) for name in ('original_checks_failed','unchanged','original_decision_unavailable')]
        if record['attempted_pairs']!=sum(values)+record['admitted_pairs']:raise ValueError('Exclusion partition differs')
        rows.append('K'+str(int(key)+1)+' & '+str(record['attempted_pairs'])+' & '+' & '.join(map(str,values))+' & '+str(record['admitted_pairs'])+r' \\')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n',encoding='utf-8')
    return destination
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(render(parser.parse_args().output))
