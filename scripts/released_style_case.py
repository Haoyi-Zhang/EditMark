"""Execute one unmodified public post-hoc method with separate task checks.

No model, model API, generation, training, public-repository mutation, or network
is used. The cases deliberately exercise language assumptions and cannot supply
population TPR/FPR, a method ranking, or independent natural-program coverage.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fixtures.released_style import CASES
from editmark_audit.records import Pair
from editmark_audit.analysis import summarize

PIN='3c873186a56033e73347391b7aedebce1f7f71d0'
DEPENDENCIES={'tree-sitter':'0.23.2','tree-sitter-python':'0.23.6',
              'tree-sitter-java':'0.23.5','tree-sitter-cpp':'0.23.4'}
KEYS=[bytes([value])*16 for value in range(4)]
sha=lambda data:hashlib.sha256(data).hexdigest()
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

BOOT=r'''import json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
namespace={}
try:
    exec(compile(sys.stdin.read(),'<owned-functional-case>','exec'),namespace)
    value=namespace['probe']()
    print(json.dumps({'status':'returned','value':value},allow_nan=False,sort_keys=True))
except Exception as error:
    print(json.dumps({'status':'raised','exception':type(error).__name__}))
'''

def observe(source,expected):
    p=subprocess.run([sys.executable,'-I','-S','-B','-c',BOOT],input=source,
        text=True,capture_output=True,timeout=4)
    if p.returncode or p.stderr.strip():
        return dict(status='process_failed',returncode=p.returncode,stderr=p.stderr[:2000],passed=False)
    result=json.loads(p.stdout)
    # JSON type equality is stricter than Python's True == 1.
    result['passed']=(result.get('status')=='returned' and
        json.dumps(result.get('value'),sort_keys=True)==json.dumps(expected,sort_keys=True))
    return result

def detection_record(result,rule):
    available=result['verdict']!='Insufficient evidence'
    return dict(available=available,detected=(result['verdict']=='Watermark detected') if available else None,
        score=result['p'] if available else None,threshold=0.01,comparator='lt',rule_id=rule)

def run(tool,out):
    tool=tool.resolve();out=out.resolve()
    if out.exists() or out.is_relative_to(ROOT.resolve()):raise ValueError('Fresh output outside artifact required')
    manifest_path=ROOT/'data/released-style/tool.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['revision']!=PIN:raise ValueError('Wrong tool identity')
    for rel,info in manifest['files'].items():
        p=tool/rel
        if not p.is_file() or sha(p.read_bytes())!=info['sha256']:raise ValueError('Unbound public tool source '+rel)
    versions={name:importlib.metadata.version(name) for name in DEPENDENCIES}
    if versions!=DEPENDENCIES:raise ValueError('Pinned minimal parser environment required')
    out.mkdir(parents=True)
    registration=dict(tool_revision=PIN,fixture_source_sha256=sha((ROOT/'fixtures/released_style.py').read_bytes()),
        driver_source_sha256=sha(Path(__file__).read_bytes()),
        fixtures=[c['name'] for c in CASES],keys=[key.hex() for key in KEYS],
        selection='All eight authored cases and all four fixed keys; no outcome filtering or adaptive key search',
        edit='Python AST unparse, with the same fixed probe contract before and after',
        maximum_pairs=32,maximum_program_processes=72,per_process_wall_seconds=4,wall_cap_seconds=120,
        independent_natural_program_sample=False,model_calls=0,security_or_exploit_testing=False)
    save(out/'registration.json',registration)
    started=time.monotonic();sys.path.insert(0,str(tool))
    import sew
    if not Path(sew.__file__).resolve().is_relative_to(tool):raise ValueError('Wrong imported tool')
    rows=[];pairs=[];baseline={}
    save(out/'environment.json',dict(python=sys.version,platform=platform.platform(),dependencies=versions))
    report=dict(status='FAILED_OR_PARTIAL',registration=registration)
    try:
        for case in CASES:
            observation=observe(case['source'],case['expected'])
            baseline[case['name']]=observation
            if not observation['passed']:raise ValueError('Invalid original task fixture '+case['name'])
        for case in CASES:
            for index,key in enumerate(KEYS):
                if time.monotonic()-started>120:raise TimeoutError('Registered wall cap reached')
                embedded=sew.embed(case['source'],'python',key)
                marked=embedded['code']
                normalized=ast.unparse(ast.parse(marked))+'\n'
                outputs=[observe(source,case['expected']) for source in (marked,normalized)]
                detections=[sew.detect(source,'python',key) for source in (case['source'],marked,normalized)]
                rule='SEW:'+PIN+':calibrated:alpha=0.01:key='+key.hex()
                def program(source,observation,detection):
                    return dict(source=source,validation=dict(available=True,passed=observation['passed'],
                        contract_id='fixed-probe:'+case['name']+':'+sha(json.dumps(case['expected'],sort_keys=True).encode())),
                        detection=detection_record(detection,rule))
                pair=dict(schema_version='editmark-pair',population='watermarked_positive',
                    evidence_kind='saved_observation',experiment_id='released-style-boundaries',
                    method='SEW/key'+str(index),model='none',source_group='authored-language-boundaries',
                    language='python',sample_id=case['name'],replicate_id='key'+str(index),edit='ast-unparse',
                    variant_id='normalized',cluster_id=case['name'],supported=embedded['status']=='ok',
                    changed=marked!=normalized,original=program(marked,outputs[0],detections[1]),
                    edited=program(normalized,outputs[1],detections[2]))
                parsed=Pair.parse(pair)
                row=dict(case=case['name'],key=index,tool_status={k:v for k,v in embedded.items() if k!='code'},
                    expected=case['expected'],unmarked_source=case['source'],marked_source=marked,edited_source=normalized,
                    source_identities=[sha(value.encode()) for value in (case['source'],marked,normalized)],
                    unmarked_observation=baseline[case['name']],marked_observation=outputs[0],edited_observation=outputs[1],
                    unmarked_detection=detections[0],marked_detection=detections[1],edited_detection=detections[2],
                    pair=pair)
                rows.append(row);pairs.append(parsed)
        contracts={str(index):summarize([p for p in pairs if p.replicate_id=='key'+str(index)])
                   for index in range(len(KEYS))}
        # A direct, same-information gate must agree with the production contract.
        expected_admitted=sum(p.supported is True and p.changed is True and
            p.original.validation.passed is True and p.edited.validation.passed is True and
            p.original.detection.observed and p.edited.detection.observed for p in pairs)
        if sum(c['admitted_pairs'] for c in contracts.values())!=expected_admitted:raise ValueError('Same-information admission mismatch')
        report.update(status='PASS',pairs=len(rows),distinct_authored_programs=len(CASES),
            accepted_embeddings=sum(r['tool_status']['status']=='ok' for r in rows),
            changed_by_embedding=sum(r['source_identities'][0]!=r['source_identities'][1] for r in rows),
            marked_task_failures=sum(not r['marked_observation']['passed'] for r in rows),
            edited_task_failures=sum(not r['edited_observation']['passed'] for r in rows),
            distinct_failure_cases=sorted({r['case'] for r in rows if not r['marked_observation']['passed']}),
            tool_status_counts=dict(Counter(r['tool_status']['status'] for r in rows)),
            insufficient_marked_detections=sum(r['marked_detection']['verdict']=='Insufficient evidence' for r in rows),
            key_strata=contracts,same_information_direct_admission=expected_admitted,
            cross_key_TPR_FPR_or_survival_estimate=False,
            independent_natural_program_sample=False,source_program_processes=72,
            interpretation='Executed released-method boundary study with authored probes, not paper-level performance replication or prevalence estimation')
        save(out/'rows.json',rows)
        (out/'pairs.jsonl').write_text(''.join(json.dumps(r['pair'],sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    finally:
        report['elapsed_wall_seconds']=time.monotonic()-started
        save(out/'summary.json',report)
    print(json.dumps({k:report[k] for k in ('status','pairs','marked_task_failures','distinct_failure_cases','insufficient_marked_detections')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tool-root',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();run(args.tool_root,args.out)
