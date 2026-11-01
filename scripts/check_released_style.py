"""Re-derive saved released-tool observations without importing that tool."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fixtures.released_style import CASES
from editmark_audit.records import Pair
from editmark_audit.analysis import summarize

def derive(path):
    rows=json.loads((path/'rows.json').read_text(encoding='utf-8'))
    cases={c['name']:c for c in CASES};seen=set();parsed=[]
    if len(rows)!=32:raise ValueError('Incomplete fixed design')
    for row in rows:
        key=(row['case'],row['key'])
        if key in seen or key[0] not in cases or type(key[1]) is not int or key[1] not in range(4):
            raise ValueError('Invalid or duplicated case/key')
        seen.add(key);case=cases[row['case']]
        if row['unmarked_source']!=case['source'] or row['expected']!=case['expected']:
            raise ValueError('Fixture identity differs')
        for label in ('unmarked','marked','edited'):
            source=row[label+'_source'];observed=row[label+'_observation']
            actual=(observed.get('status')=='returned' and
                json.dumps(observed.get('value'),sort_keys=True)==json.dumps(case['expected'],sort_keys=True))
            if type(observed['passed']) is not bool or observed['passed']!=actual:
                raise ValueError('Task outcome is not supported by saved return value')
        sources=[row[label+'_source'] for label in ('unmarked','marked','edited')]
        if row['source_identities']!=[hashlib.sha256(s.encode()).hexdigest() for s in sources]:
            raise ValueError('Source bytes differ')
        pair=Pair.parse(row['pair'])
        expected_rule=('SEW:3c873186a56033e73347391b7aedebce1f7f71d0:calibrated:alpha=0.01:key='
                       +bytes([row['key']]).hex()*16)
        if (row['pair']['original']['source']!=sources[1] or row['pair']['edited']['source']!=sources[2]
                or pair.original.source_sha256!=row['source_identities'][1]
                or pair.edited.source_sha256!=row['source_identities'][2]
                or pair.changed!=(sources[1]!=sources[2])
                or pair.supported!=(row['tool_status']['status']=='ok')
                or pair.original.detection.rule_id!=expected_rule
                or pair.edited.detection.rule_id!=expected_rule):raise ValueError('Pair source, support or rule identity differs')
        if (pair.replicate_id!='key'+str(row['key']) or pair.sample_id!=row['case'] or
                pair.original.validation.passed!=row['marked_observation']['passed'] or
                pair.edited.validation.passed!=row['edited_observation']['passed']):raise ValueError('Pair observation binding differs')
        for label,program in [('marked',pair.original),('edited',pair.edited)]:
            result=row[label+'_detection'];available=result['verdict']!='Insufficient evidence'
            if program.detection.available!=available:raise ValueError('Insufficient evidence misencoded')
            if available and (program.detection.score!=result['p'] or
                    program.detection.detected!=(result['verdict']=='Watermark detected')):raise ValueError('Detector result differs')
        parsed.append(pair)
    strata={str(k):summarize([p for p in parsed if p.replicate_id=='key'+str(k)]) for k in range(4)}
    for k,report in strata.items():
        group=[p for p in parsed if p.replicate_id=='key'+k]
        if len({p.original.detection.rule for p in group})!=1:raise ValueError('Mixed detector rules within key stratum')
        direct=sum(p.supported is True and p.changed is True and
            p.original.validation.passed is True and p.edited.validation.passed is True and
            p.original.detection.observed and p.edited.detection.observed for p in group)
        if direct!=report['admitted_pairs']:raise ValueError('Same-information direct admission differs')
    return dict(status='PASS',pairs=len(rows),distinct_authored_programs=len(CASES),key_strata=strata,
        accepted_embeddings=sum(r['tool_status']['status']=='ok' for r in rows),
        marked_task_failures=sum(not r['marked_observation']['passed'] for r in rows),
        edited_task_failures=sum(not r['edited_observation']['passed'] for r in rows),
        distinct_failure_cases=sorted({r['case'] for r in rows if not r['marked_observation']['passed']}),
        insufficient_marked_detections=sum(r['marked_detection']['verdict']=='Insufficient evidence' for r in rows),
        same_information_direct_admission=sum(r['admitted_pairs'] for r in strata.values()),
        cross_key_TPR_FPR_or_survival_estimate=False,independent_natural_program_sample=False,
        verification_scope='Data-only rederivation from saved source-bound observations; not new execution or authenticated independent measurement')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'results/released-style');p.add_argument('--output',type=Path)
    a=p.parse_args();result=derive(a.data)
    if a.output:
        if a.output.exists():raise ValueError('Fresh analysis output required')
        a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','pairs','marked_task_failures','same_information_direct_admission')}))
