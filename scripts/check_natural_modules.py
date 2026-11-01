"""Reconstruct the fixed natural-module contract from retained raw observations."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path,PurePosixPath
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from editmark_audit.records import Pair
from editmark_audit.analysis import exclusion,summarize
from released_style_case import detection_record
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def raw_read(path):
    if path.is_file():return path.read_bytes()
    for parent in list(path.parents)[:4]:
        archive=parent/'evidence.zip'
        if archive.is_file():
            rel=path.relative_to(parent).as_posix()
            with zipfile.ZipFile(archive) as z:
                if rel in z.namelist():return z.read(rel)
    raise FileNotFoundError(path)
read=lambda p:json.loads(raw_read(p).decode('utf-8'))

def verify_archive(path):
    archive=path/'evidence.zip'
    if not archive.is_file():return
    manifest=read(path/'evidence-members.json')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(manifest) or set(z.namelist())!=set(manifest):raise ValueError('Retained evidence member set differs')
        if sum(v.file_size for v in z.infolist())>64*1024*1024:raise ValueError('Retained evidence expanded-byte cap')
        for rel,identity in manifest.items():
            if PurePosixPath(rel).is_absolute() or '..' in PurePosixPath(rel).parts or '\\' in rel:raise ValueError('Unsafe retained evidence member')
            raw=z.read(rel)
            if len(raw)!=identity['bytes'] or sha(raw)!=identity['sha256']:raise ValueError('Retained raw observation bytes differ')
            adjacent=path/rel
            if adjacent.is_file() and adjacent.read_bytes()!=raw:raise ValueError('Readable metadata differs from raw activity')

def task_validation(value,baseline):
    available=baseline.get('status')=='completed' and value.get('status')=='completed' and baseline.get('passed') is True
    stable=available and value['test_ids']==baseline['test_ids'] and sorted(r['id'] for r in value['outcomes'] if r['status']=='skipped')==sorted(r['id'] for r in baseline['outcomes'] if r['status']=='skipped')
    return dict(available=available,passed=bool(value.get('passed') and stable) if available else None,
        contract_id='CPython-upstream-python-tests:'+value['module']+':'+sha(json.dumps(baseline.get('test_ids',[])).encode()))

def verify_task(value,source,probe='line'):
    if value.get('status')!='completed':
        if value.get('passed') is not False:raise ValueError('Unavailable execution cannot pass')
        return
    digest=sha(source.encode())
    if value['source_sha256']!=digest:raise ValueError('Executed source identity differs')
    if not value['bound_modules'] or any(r['source_sha256']!=digest for r in value['bound_modules']):raise ValueError('Unbound module copy')
    ids=value['test_ids'];started=value['started_tests'];events=value['outcomes']
    if len(ids)!=len(set(ids)) or ids!=started or value['tests_scheduled']!=len(ids) or value['tests_run']!=len(started):raise ValueError('Upstream test identity/accounting differs')
    for event in events:
        if event['id'] not in ids:raise ValueError('Outcome outside registered test set')
    failures=sum(e['status']=='failed' for e in events)
    errors=sum(e['status']=='error' for e in events)
    subtest_failures=sum(e['status']=='subtest_failure' for e in events)
    skipped=sum(e['status']=='skipped' for e in events)
    unexpected=sum(e['status']=='unexpected_success' for e in events)
    # A subtest event records non-success but does not separately encode an
    # assertion failure versus another exception; both must block admission.
    if value['failures']+value['errors']!=failures+errors+subtest_failures or (value['skipped'],value['unexpected_successes'])!=(skipped,unexpected):raise ValueError('Raw test outcomes disagree with summary')
    observed=value.get('executed_lines',[]) if probe=='line' else value.get('entered_python_functions',[])
    expected=not (failures or errors or subtest_failures or unexpected) and len(started)>skipped and bool(observed)
    if value['passed']!=expected:raise ValueError('Task pass status not supported')
    if probe=='line' and any(type(n) is not int or n<1 or n>len(source.splitlines()) for n in observed):raise ValueError('Invalid source coverage')
    if probe=='call' and (not all(isinstance(n,str) and n for n in observed) or value['provenance_kind']!='function-entry calls; not statement coverage'):raise ValueError('Invalid function-entry provenance')

def derive(path):
    verify_archive(path)
    protocol=read(ROOT/'data/natural-modules/protocol.json');registration=read(path/'registration.json')
    if registration['protocol']!=protocol:raise ValueError('Study protocol differs')
    for file,key in [('scripts/natural_module_case.py','driver_source_sha256'),('scripts/upstream_suite.py','suite_source_sha256'),('data/released-style/tool.json','tool_manifest_sha256')]:
        if sha((ROOT/file).read_bytes())!=registration[key]:raise ValueError('Execution input changed '+file)
    archive=ROOT/'data/natural-modules/inputs.zip'
    if sha(archive.read_bytes())!=protocol['input_archive_sha256']:raise ValueError('Natural input archive differs')
    sources={}
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(protocol['source_files']) or set(z.namelist())!=set(protocol['source_files']):raise ValueError('Natural input member set differs')
        if sum(v.file_size for v in z.infolist())>8*1024*1024:raise ValueError('Natural input expanded-byte cap')
        for rel,identity in protocol['source_files'].items():
            if PurePosixPath(rel).is_absolute() or '..' in PurePosixPath(rel).parts:raise ValueError('Unsafe input member')
            raw=z.read(rel)
            if len(raw)!=identity['bytes'] or sha(raw)!=identity['sha256']:raise ValueError('Natural source/test bytes differ')
            if rel in ['Lib/'+m+'.py' for m in protocol['selected_modules']]:sources[PurePosixPath(rel).stem]=raw.decode('utf-8')
    rows=read(path/'rows.json');baselines=read(path/'baselines.json');raw=read(path/'raw-processes.json')
    if set(baselines)!=set(protocol['selected_modules']):raise ValueError('Incomplete original-module cohort')
    expected={(m,k) for m in protocol['selected_modules'] for k in range(4)};seen=set();pairs=[]
    if len(rows)!=len(expected) or len(raw)!=protocol['budget']['maximum_suite_processes']:raise ValueError('Incomplete fixed experiment')
    raw_by={r['label']:r for r in raw}
    if len(raw_by)!=len(raw):raise ValueError('Repeated raw process identity')
    def bind(label,task):
        observed=raw_by[label]
        if observed['returncode']==0 and json.loads(observed['stdout'])!=task:raise ValueError('Task result differs from raw process')
        if observed['returncode']!=0 and task['status'] not in ('process_failed','process_timeout'):raise ValueError('Failed process misrepresented')
    for module,value in baselines.items():
        verify_task(value,sources[module]);bind('baseline-'+module,value)
    for row in rows:
        module=row['module'];key=row['key'];identity=(module,key)
        if identity in seen or identity not in expected:raise ValueError('Duplicate/unknown module-key pair')
        seen.add(identity);base=baselines[module]
        if row['unmarked_source']!=sources[module]:raise ValueError('Natural source was rewritten before study')
        values=[row[v+'_source'] for v in ('unmarked','marked','edited')]
        if row['source_identities']!=[sha(s.encode()) for s in values]:raise ValueError('Retained candidate bytes differ')
        for label in ('marked','edited'):
            task=row[label+'_task'];verify_task(task,row[label+'_source']);bind(module+'-'+str(key)+'-'+label,task)
            # Missing tasks still carry their module label in the stored row.
            value=dict(task,module=module)
            expected_validation=task_validation(value,base)
            if row['pair']['original' if label=='marked' else 'edited']['validation']!=expected_validation:raise ValueError('Pair validation not supported by task observations')
        rule='SEW:'+read(ROOT/'data/released-style/tool.json')['revision']+':calibrated:alpha=0.01:key='+protocol['keys'][key]
        for label,program in [('marked','original'),('edited','edited')]:
            if row['pair'][program]['source']!=row[label+'_source'] or row['pair'][program]['detection']!=detection_record(row[label+'_detection'],rule):raise ValueError('Detector/source binding differs')
        if row['pair']['supported']!=(row['tool_status']['status']=='ok' and row['edit_error'] is None) or row['pair']['changed']!=(values[1]!=values[2]):raise ValueError('Transformation admission differs')
        pairs.append(Pair.parse(row['pair']))
    strata={str(k):summarize([p for p,r in zip(pairs,rows) if r['key']==k]) for k in range(4)}
    def direct_admission(row):
        base=baselines[row['module']]
        if row['tool_status']['status']!='ok' or row['edit_error'] is not None or row['marked_source']==row['edited_source']:return False
        if base.get('passed') is not True:return False
        base_skips=sorted(r['id'] for r in base['outcomes'] if r['status']=='skipped')
        for label in ('marked','edited'):
            value=row[label+'_task']
            if value.get('status')!='completed' or value.get('passed') is not True:return False
            if value['test_ids']!=base['test_ids'] or sorted(r['id'] for r in value['outcomes'] if r['status']=='skipped')!=base_skips:return False
            if row[label+'_detection']['verdict']=='Insufficient evidence':return False
        return True
    simple={str(k):sum(direct_admission(r) for r in rows if r['key']==k) for k in range(4)}
    if any(strata[k]['admitted_pairs']!=simple[k] for k in strata):raise ValueError('Same-information admission differs')
    return dict(status='PASS',module_cohort=len(sources),upstream_projects=1,pairs=len(rows),suite_processes=len(raw),
        baseline_modules_passed=sum(r.get('passed') is True for r in baselines.values()),
        accepted_embeddings=sum(r['tool_status']['status']=='ok' for r in rows),
        marked_task_failures=sum(r['pair']['original']['validation']['passed'] is False for r in rows),
        edited_task_failures=sum(r['pair']['edited']['validation']['passed'] is False for r in rows),
        key_strata=strata,same_information_direct_admission=simple,
        attempt_categories=[dict(module=r['module'],key=r['key'],reason=exclusion(p) or 'admitted') for p,r in zip(pairs,rows)],
        natural_modules_not_independent_projects=True,population_estimate=False,
        scope='Retained raw process and source-bound upstream-test rederivation; not new execution or independent personnel replication')

def verify_attribution(path):
    from natural_counterfactual import restore
    registration=read(path/'attribution/registration.json');sources=read(path/'attribution/sources.json')
    observations=read(path/'attribution/observations.json');rows=read(path/'rows.json')
    if registration['source_rows_sha256']!=sha(raw_read(path/'rows.json')):raise ValueError('Attribution source cohort differs')
    if registration['diagnostic_source_sha256']!=sha((ROOT/'scripts/natural_counterfactual.py').read_bytes()) or registration['suite_source_sha256']!=sha((ROOT/'scripts/upstream_suite.py').read_bytes()):raise ValueError('Attribution executed source differs')
    wanted={(m,k) for m in ('heapq','shlex') for k in (0,1)}
    if len(sources)!=4 or len(observations)!=4 or {(r['module'],r['key']) for r in sources}!=wanted or {(r['module'],r['key']) for r in observations}!=wanted:raise ValueError('Attribution fixed design differs')
    for record in sources:
        row=next(r for r in rows if (r['module'],r['key'])==(record['module'],record['key']))
        if row['marked_source']!=record['original_source'] or row['pair']['original']['validation']['passed'] is not False:raise ValueError('Original failure changed')
        expected,edit=restore(row['marked_source'],row['module'])
        if expected!=record['restored_source'] or edit!=record['edit']:raise ValueError('Rollback changed more than registered expression')
        observed=next(r for r in observations if (r['module'],r['key'])==(record['module'],record['key']))
        if observed['returncode']!=0 or json.loads(observed['stdout'])!=observed['observation']:raise ValueError('Attribution raw process differs')
        verify_task(observed['observation'],expected)
        if not observed['observation']['passed'] or observed['observation']['test_ids']!=row['marked_task']['test_ids']:raise ValueError('Same-contract rollback not successful')
    return dict(cases=4,restored_passes=4,original_failures_preserved=True,post_hoc_not_population_evidence=True)

def verify_statistics_followup(path):
    follow=path/'statistics-validation';verify_archive(follow)
    registration=read(follow/'registration.json');observations=read(follow/'observations.json')
    rows=[r for r in read(path/'rows.json') if r['module']=='statistics']
    if registration['source_rows_sha256']!=sha(raw_read(path/'rows.json')):raise ValueError('Follow-up cohort changed')
    for rel,key in [('scripts/statistics_validation.py','driver_sha256'),('scripts/call_profile_suite.py','validator_sha256'),('scripts/upstream_suite.py','shared_test_policy_sha256')]:
        if registration[key]!=sha((ROOT/rel).read_bytes()):raise ValueError('Follow-up source identity changed')
    sources={'original':rows[0]['unmarked_source']}
    sources.update({str(r['key'])+'-'+label:r[label+'_source'] for r in rows for label in ('marked','edited')})
    if len(observations)!=9 or len({r['label'] for r in observations})!=9 or set(sources)!={r['label'] for r in observations}:raise ValueError('Incomplete follow-up source cohort')
    base=next(r['observation'] for r in observations if r['label']=='original')
    for r in observations:
        source=sources[r['label']]
        if r['source_sha256']!=sha(source.encode()) or r['returncode']!=0 or json.loads(r['stdout'])!=r['observation']:raise ValueError('Follow-up raw execution differs')
        verify_task(r['observation'],source,probe='call')
        if not r['observation']['passed'] or r['observation']['test_ids']!=base['test_ids']:raise ValueError('Follow-up tests or result differs')
        if sorted(e['id'] for e in r['observation']['outcomes'] if e['status']=='skipped')!=sorted(e['id'] for e in base['outcomes'] if e['status']=='skipped'):raise ValueError('Follow-up skip set differs')
    if registration['sources']!=[dict(label=label,sha256=sha(source.encode())) for label,source in sources.items()]:raise ValueError('Follow-up sources not preregistered')
    strata={}
    for r in rows:
        pair=dict(r['pair']);contract='upstream-call-profile:statistics:'+sha(json.dumps(base['test_ids']).encode())
        for label in ('original','edited'):
            pair[label]=dict(pair[label],validation=dict(available=True,passed=True,contract_id=contract))
        strata[str(r['key'])]=summarize([Pair.parse(pair)])
    if any(r['admitted_pairs']!=1 for r in strata.values()):raise ValueError('Follow-up pair not admissible')
    return dict(source_programs=9,passing=9,tests_per_source=base['tests_run'],separately_validated_pairs=4,key_strata=strata,
        original_activity_unknowns_unchanged=True,statement_coverage_claimed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'results/natural-modules');p.add_argument('--output',type=Path)
    a=p.parse_args();result=derive(a.data);result['attribution']=verify_attribution(a.data);result['statistics_followup']=verify_statistics_followup(a.data)
    if a.output:
        if a.output.exists():raise ValueError('Fresh analysis output required')
        a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','module_cohort','pairs','baseline_modules_passed','marked_task_failures','edited_task_failures','same_information_direct_admission')}))
