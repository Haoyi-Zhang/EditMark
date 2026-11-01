"""A fixed follow-up of the unresolved statistics module, with no budget reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
from check_natural_modules import ROOT,read,raw_read,derive
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def save(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def run(out):
    data=ROOT/'results/natural-modules';derive(data)
    if out.exists() or out.resolve().is_relative_to(ROOT):raise ValueError('Fresh follow-up output required')
    rows=[r for r in read(data/'rows.json') if r['module']=='statistics']
    if len(rows)!=4:raise ValueError('Fixed follow-up cohort differs')
    sources=[dict(label='original',source=rows[0]['unmarked_source'])]
    sources += [dict(label=str(r['key'])+'-'+name,source=r[name+'_source']) for r in rows for name in ('marked','edited')]
    out.mkdir(parents=True);start=time.monotonic()
    save(out/'registration.json',dict(sources=[dict(label=r['label'],sha256=sha(r['source'].encode())) for r in sources],
        source_rows_sha256=sha(raw_read(data/'rows.json')),
        driver_sha256=sha(Path(__file__).read_bytes()),validator_sha256=sha((ROOT/'scripts/call_profile_suite.py').read_bytes()),
        shared_test_policy_sha256=sha((ROOT/'scripts/upstream_suite.py').read_bytes()),
        process_limit=9,per_process_CPU_seconds=20,per_process_wall_seconds=30,whole_wall_seconds=240,
        design='All nine previously fixed original/marked/edited sources; same upstream test selection and single-process limits. Function-entry profiling replaces line tracing. No watermark call, source retuning or repetition of the original activity.',
        no_original_budget_reset=True,model_calls=0))
    records=[]
    with tempfile.TemporaryDirectory(prefix='statistics-call-validation-') as directory:
        work=Path(directory);inputs=work/'inputs';inputs.mkdir()
        with zipfile.ZipFile(ROOT/'data/natural-modules/inputs.zip') as z:
            # derive above checks the exact licensed input archive and members.
            z.extractall(inputs)
        for row in sources:
            if time.monotonic()-start>240:raise TimeoutError('Follow-up budget exhausted')
            candidate=work/row['label'];candidate.mkdir();(candidate/'statistics.py').write_text(row['source'],encoding='utf-8')
            command=[sys.executable,'-I','-B',str(ROOT/'scripts/call_profile_suite.py'),'--module','statistics','--candidate',str(candidate),'--test-root',str(inputs/'Lib')]
            try:
                p=subprocess.run(command,text=True,capture_output=True,timeout=30,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',TZ='UTC'))
                clean=lambda s:s.replace(str(work),'<WORK_ROOT>').replace(str(ROOT),'<ARTIFACT_ROOT>')
                result=json.loads(clean(p.stdout)) if p.returncode==0 else dict(status='process_failed',passed=False)
                record=dict(label=row['label'],source_sha256=sha(row['source'].encode()),observation=result,returncode=p.returncode,stdout=clean(p.stdout),stderr=clean(p.stderr))
            except subprocess.TimeoutExpired:record=dict(label=row['label'],source_sha256=sha(row['source'].encode()),observation=dict(status='process_timeout',passed=False),returncode=None,stdout='',stderr='registered wall limit reached')
            records.append(record);save(out/'observations.json',records)
    summary=dict(status='COMPLETE',source_programs=len(records),passed=sum(r['observation'].get('passed') is True for r in records),elapsed_wall_seconds=time.monotonic()-start,
        original_activity_unknowns_unchanged=True,statement_coverage_claimed=False)
    save(out/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
