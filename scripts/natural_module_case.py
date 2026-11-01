"""Fixed released-module study with unmodified upstream task assertions.

Requires the separately acquired pinned SEW source and minimal parser venv.
No generation model, model API, live service or security exploit is used.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from released_style_case import detection_record,DEPENDENCIES
from editmark_audit.records import Pair
from editmark_audit.analysis import summarize
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def save(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def normalized(text,out,work,tool):
    return text.replace(str(out),'<OUTPUT>').replace(str(work),'<WORK_ROOT>').replace(str(ROOT),'<ARTIFACT_ROOT>').replace(str(tool),'<TOOL_ROOT>').replace(sys.executable,'<PYTHON>')

def run(tool,out):
    tool=tool.resolve();out=out.resolve()
    if out.exists() or out.is_relative_to(ROOT):raise ValueError('Fresh output outside artifact required')
    protocol=json.loads((ROOT/'data/natural-modules/protocol.json').read_text())
    manifest=json.loads((ROOT/'data/released-style/tool.json').read_text())
    for rel,info in manifest['files'].items():
        if sha((tool/rel).read_bytes())!=info['sha256']:raise ValueError('Unbound tool source')
    versions={n:importlib.metadata.version(n) for n in DEPENDENCIES}
    if versions!=DEPENDENCIES:raise ValueError('Wrong parser environment')
    archive=ROOT/'data/natural-modules/inputs.zip'
    if sha(archive.read_bytes())!=protocol['input_archive_sha256']:raise ValueError('Input archive changed')
    out.mkdir(parents=True);started=time.monotonic()
    registration=dict(protocol=protocol,tool_manifest_sha256=sha((ROOT/'data/released-style/tool.json').read_bytes()),
        driver_source_sha256=sha(Path(__file__).read_bytes()),suite_source_sha256=sha((ROOT/'scripts/upstream_suite.py').read_bytes()),
        note='Registration written before any outcome; keys and module cohort are fixed')
    save(out/'registration.json',registration)
    save(out/'environment.json',dict(python=sys.version,platform=platform.platform(),dependencies=versions))
    sys.path.insert(0,str(tool));import sew
    if not Path(sew.__file__).resolve().is_relative_to(tool):raise ValueError('Wrong public tool import')
    rows=[];raw_processes=[];baselines={};budget=protocol['budget'];report=dict(status='FAILED_OR_PARTIAL')
    try:
      with tempfile.TemporaryDirectory(prefix='natural-module-study-') as directory:
        work=Path(directory);inputs=work/'inputs';inputs.mkdir()
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist())!=set(protocol['source_files']):raise ValueError('Input member set')
            for rel,identity in protocol['source_files'].items():
                data=z.read(rel)
                if len(data)!=identity['bytes'] or sha(data)!=identity['sha256']:raise ValueError('Input member differs')
                p=inputs/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        test_root=work/'tests';test_root.mkdir();(test_root/'test').symlink_to(inputs/'Lib/test',target_is_directory=True)
        def suite(module,source,label):
            if len(raw_processes)>=budget['maximum_suite_processes'] or time.monotonic()-started>budget['whole_wall_seconds']:raise TimeoutError('Registered study budget exhausted')
            candidate=work/label;candidate.mkdir();(candidate/(module+'.py')).write_text(source,encoding='utf-8')
            command=[sys.executable,'-I','-B',str(ROOT/'scripts/upstream_suite.py'),'--module',module,'--candidate',str(candidate),'--test-root',str(test_root)]
            try:
                p=subprocess.run(command,text=True,capture_output=True,timeout=budget['per_suite_wall_seconds'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',TZ='UTC'))
                result=json.loads(p.stdout) if p.returncode==0 else dict(status='process_failed',passed=False)
                raw=dict(label=label,returncode=p.returncode,stdout=normalized(p.stdout,out,work,tool),stderr=normalized(p.stderr,out,work,tool))
            except subprocess.TimeoutExpired as error:
                result=dict(status='process_timeout',passed=False)
                raw=dict(label=label,returncode=None,stdout=normalized((error.stdout or b'').decode() if isinstance(error.stdout,bytes) else error.stdout or '',out,work,tool),stderr='registered suite wall limit reached')
            raw_processes.append(raw);save(out/'raw-processes.json',raw_processes)
            return json.loads(normalized(json.dumps(result),out,work,tool))
        for module in protocol['selected_modules']:
            source=(inputs/'Lib'/str(module+'.py')).read_text(encoding='utf-8')
            baselines[module]=suite(module,source,'baseline-'+module)
            save(out/'baselines.json',baselines)
        for module in protocol['selected_modules']:
          source=(inputs/'Lib'/str(module+'.py')).read_text(encoding='utf-8')
          base=baselines[module]
          for key_index,key_text in enumerate(protocol['keys']):
            if time.monotonic()-started>budget['whole_wall_seconds']:raise TimeoutError('Registered study budget exhausted')
            key=bytes.fromhex(key_text)
            embedded=sew.embed(source,'python',key);marked=embedded['code'];edit_error=None
            try:edited=ast.unparse(ast.parse(marked))+'\n'
            except Exception as error:edited=marked;edit_error=type(error).__name__
            marked_result=suite(module,marked,str(module)+'-'+str(key_index)+'-marked')
            edited_result=suite(module,edited,str(module)+'-'+str(key_index)+'-edited')
            detections=[sew.detect(value,'python',key) for value in (source,marked,edited)]
            rule='SEW:'+manifest['revision']+':calibrated:alpha=0.01:key='+key_text
            # A suite cannot qualify a candidate if it silently changes test
            # membership or host-dependent skips relative to the original.
            def validation(value):
                available=base.get('status')=='completed' and value.get('status')=='completed' and base.get('passed') is True
                stable=available and value['test_ids']==base['test_ids'] and sorted(r['id'] for r in value['outcomes'] if r['status']=='skipped')==sorted(r['id'] for r in base['outcomes'] if r['status']=='skipped')
                return dict(available=available,passed=bool(value.get('passed') and stable) if available else None,
                    contract_id='CPython-upstream-python-tests:'+module+':'+sha(json.dumps(base.get('test_ids',[])).encode()))
            def program(code,observation,detector):return dict(source=code,validation=validation(observation),detection=detection_record(detector,rule))
            pair=dict(schema_version='editmark-pair',population='watermarked_positive',evidence_kind='saved_observation',
                experiment_id='released-natural-modules',method='SEW/key'+str(key_index),model='none',source_group='CPython-standard-library',
                language='python',sample_id=module,replicate_id='key'+str(key_index),edit='ast-unparse',variant_id='normalized',
                cluster_id=module,supported=embedded['status']=='ok' and edit_error is None,changed=marked!=edited,
                original=program(marked,marked_result,detections[1]),edited=program(edited,edited_result,detections[2]))
            Pair.parse(pair)
            rows.append(dict(module=module,key=key_index,unmarked_source=source,marked_source=marked,edited_source=edited,
                tool_status={k:v for k,v in embedded.items() if k!='code'},edit_error=edit_error,
                source_identities=[sha(s.encode()) for s in (source,marked,edited)],marked_task=marked_result,edited_task=edited_result,
                unmarked_detection=detections[0],marked_detection=detections[1],edited_detection=detections[2],pair=pair))
            save(out/'rows.json',rows)
        strata={str(k):summarize([Pair.parse(r['pair']) for r in rows if r['key']==k]) for k in range(4)}
        report=dict(status='PASS_COMPLETE_ACTIVITY',module_cohort=len(protocol['selected_modules']),upstream_projects=1,
            pairs=len(rows),suite_processes=len(raw_processes),baseline_modules_passed=sum(r.get('passed') is True for r in baselines.values()),
            accepted_embeddings=sum(r['tool_status']['status']=='ok' for r in rows),
            marked_task_failures=sum(r['pair']['original']['validation']['passed'] is False for r in rows),
            edited_task_failures=sum(r['pair']['edited']['validation']['passed'] is False for r in rows),
            key_strata=strata,pooled_population_estimate=False,API_calls=0)
        (out/'pairs.jsonl').write_text(''.join(json.dumps(r['pair'],sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    finally:
        report.update(elapsed_wall_seconds=time.monotonic()-started,raw_suite_processes_retained=len(raw_processes))
        save(out/'summary.json',report)
    print(json.dumps({k:report[k] for k in ('status','pairs','baseline_modules_passed','accepted_embeddings','marked_task_failures','edited_task_failures')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tool-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.tool_root,a.out)
