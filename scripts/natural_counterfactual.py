"""One predeclared source-level rollback for each observed failing attempt.

These post-hoc diagnostics explain failures, not new prevalence observations.
The released watermark implementation and original cohort are never changed.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sha=lambda raw:hashlib.sha256(raw).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def restore(source,module):
    tree=ast.parse(source);matches=[]
    if module=='heapq':
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_siftdown')
        matches=[n for n in ast.walk(function) if isinstance(n,ast.Compare) and ast.dump(n,include_attributes=False)==ast.dump(ast.parse('parent > newitem',mode='eval').body,include_attributes=False)]
        replacement='newitem < parent'
    elif module=='shlex':
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='shlex')
        function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        matches=[n for n in ast.walk(function) if isinstance(n,ast.Compare) and ast.dump(n,include_attributes=False)==ast.dump(ast.parse('len(punctuation_chars) == 0',mode='eval').body,include_attributes=False)]
        replacement='not punctuation_chars'
    else:raise ValueError('Unregistered diagnostic module')
    if len(matches)!=1:raise ValueError('Diagnostic expression not uniquely identified')
    node=matches[0];lines=source.encode().splitlines(keepends=True)
    start=sum(map(len,lines[:node.lineno-1]))+node.col_offset
    end=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
    original=source.encode()[start:end].decode();raw=source.encode()
    result=(raw[:start]+replacement.encode()+raw[end:]).decode();ast.parse(result)
    return result,dict(start_byte=start,end_byte=end,old=original,new=replacement)

def run(data,out):
    if out.exists() or out.resolve().is_relative_to(ROOT):raise ValueError('Fresh external diagnostic output required')
    all_rows=read(data/'rows.json')
    cases=[r for r in all_rows if r['module'] in ('heapq','shlex') and r['key'] in (0,1)]
    if len(cases)!=4 or any(r['pair']['original']['validation']['passed'] is not False for r in cases):raise ValueError('Changed diagnostic cohort')
    prepared=[]
    for row in cases:
        fixed,edit=restore(row['marked_source'],row['module'])
        prepared.append(dict(module=row['module'],key=row['key'],original_source=row['marked_source'],restored_source=fixed,edit=edit))
    out.mkdir(parents=True);start=time.monotonic()
    save(out/'registration.json',dict(source_rows_sha256=sha((data/'rows.json').read_bytes()),
        diagnostic_source_sha256=sha(Path(__file__).read_bytes()),suite_source_sha256=sha((ROOT/'scripts/upstream_suite.py').read_bytes()),
        cases=[dict(module=r['module'],key=r['key'],original_sha256=sha(r['original_source'].encode()),restored_sha256=sha(r['restored_source'].encode()),edit=r['edit']) for r in prepared],
        maximum_suite_processes=4,per_suite_wall_seconds=30,whole_wall_seconds=150,
        purpose='Post-hoc expression rollback attribution; not an extra natural-module cohort or repaired released method'))
    results=[];save(out/'sources.json',prepared)
    with tempfile.TemporaryDirectory(prefix='natural-attribution-') as d:
        work=Path(d);inputs=work/'inputs';inputs.mkdir()
        with zipfile.ZipFile(ROOT/'data/natural-modules/inputs.zip') as z:z.extractall(inputs)
        for row in prepared:
            if time.monotonic()-start>150:raise TimeoutError('Attribution budget exhausted')
            candidate=work/(row['module']+'-'+str(row['key']));candidate.mkdir()
            (candidate/(row['module']+'.py')).write_text(row['restored_source'],encoding='utf-8')
            p=subprocess.run([sys.executable,'-I','-B',str(ROOT/'scripts/upstream_suite.py'),'--module',row['module'],'--candidate',str(candidate),'--test-root',str(inputs/'Lib')],
                text=True,capture_output=True,timeout=30,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',TZ='UTC'))
            clean=lambda s:s.replace(str(work),'<WORK_ROOT>').replace(str(ROOT),'<ARTIFACT_ROOT>')
            observed=json.loads(clean(p.stdout)) if p.returncode==0 else dict(status='process_failed',passed=False)
            results.append(dict(module=row['module'],key=row['key'],observation=observed,returncode=p.returncode,stdout=clean(p.stdout),stderr=clean(p.stderr)))
            save(out/'observations.json',results)
    summary=dict(status='COMPLETE',cases=len(results),restored_passes=sum(r['observation'].get('passed') is True for r in results),elapsed_wall_seconds=time.monotonic()-start,
        scope='Full same-contract upstream tests after one expression rollback per already observed failing output; original observations are unchanged')
    save(out/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'results/natural-modules');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.data,a.out)
