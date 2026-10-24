"""Opt-in CPU execution of the author's existing SemCodebook bootstrap encoder.

Executes only five reviewed arithmetic texts and their fixed comment variants.
Uses unchanged upstream detector Boolean output, not confidence thresholding.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'vendor'))
from semcodebook.inference import rule_based_generate
from semcodebook.detector import SemCodebookDetector
from semcodebook.protocol import CarrierScheduleEntry, GenerationRequest, WatermarkSpec
from editmark_audit.analysis import report
from editmark_audit.records import Pair
from editmark_audit.aggregates import from_pair_stratum, analyze
from editmark_audit import certificate, verify
from editmark_audit.io import write_new_json, digest

BASE="def solve(items):\n    total = 0\n    for item in items:\n        if item < 0:\n            return 0\n        total += item\n    return total\n"
AFTER="def solve(items):\n    if any(item < 0 for item in items):\n        return 0\n    return sum(items)\n"
FOOTER="# Reviewed arithmetic task; executable statements unchanged.\n"
APPROVED={
"889f664436946ce0ad5f05d833aa488b77da5365f5aea9428e7007a28ae13cc7",
"cb86497ef4e3779602a3d36b78d6d3b48b57b4ebc47d48f568308f0fb42e3e70",
"36ebd54100b7b5653dc3d1a181465bbaa15952f1e53220b8f7c0c5879086cab7",
"a13b226b0e6f5f0404328fbc6bc21ebf76c5b1d6a52bba6a891b3e935a90f5f1",
"63a62c32f12bdf4f56d7950e658f53101419bde542265cab6538ac4385c2b3b4"}
DOMAIN=[list(t) for length in range(4) for t in product((-2,-1,0,1,2),repeat=length)]
CONTRACT={"task":"return zero if any integer is negative, otherwise the sum","domain":DOMAIN,
          "output":"exact Python int; identical original/edited input and expected output"}
def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
def require(x,message):
    if not x:
        raise ValueError(message)
def validate(source):
    base=source[:-len(FOOTER)] if source.endswith(FOOTER) else source
    require(sha(base) in APPROVED,"source not in reviewed arithmetic whitelist")
    namespace={}
    exec(compile(source,'reviewed_semcodebook_arithmetic.py','exec'),namespace,namespace)
    calls=[]
    for values in DOMAIN:
        expected=0 if any(v<0 for v in values) else sum(values)
        actual=namespace['solve'](values.copy())
        passed=type(actual) is int and actual==expected
        calls.append({"input":values,"expected":expected,"actual":actual,"passed":passed})
    return all(c["passed"] for c in calls),calls
def check_upstream():
    provenance=json.loads((ROOT/'vendor/SemCodebook-provenance.json').read_text(encoding='utf-8'))
    for entry in provenance['files']:
        path=ROOT/'vendor/semcodebook'/Path(entry['path']).name
        data=path.read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        require(actual==entry['git_blob'],"upstream byte mismatch: "+path.name)
    return provenance
def direct(rows):
    joint=Counter()
    for row in rows:
        b,a=row['original'],row['edited']
        require(row['supported'] is True and sha(b['source'])!=sha(a['source']),"unsupported or unchanged")
        require(all(p['source_sha256']==sha(p['source']) for p in (b,a)),"source identity")
        require(b['validation']==a['validation'] and b['validation']['passed'] is True
                and b['validation']['available'] is True and bool(b['validation']['contract_id']),"functional gate")
        require(all(p['detection']['available'] is True for p in (b,a)),"missing decision")
        require(all(b['detection'][k]==a['detection'][k] for k in ('rule_id','threshold','comparator')),"rule drift")
        require(bool(b['detection']['rule_id']) and b['detection']['threshold']==0.5
                and b['detection']['comparator']=='ge',"rule unknown")
        require(all(type(p['detection']['detected']) is bool for p in (b,a)),"decision unknown")
        joint[f"n{int(b['detection']['detected'])}{int(a['detection']['detected'])}"]+=1
    counts={k:joint[k] for k in ('n00','n01','n10','n11')}
    n=sum(counts.values());bp=counts['n10']+counts['n11'];ap=counts['n01']+counts['n11']
    def f(v): return [v.numerator,v.denominator]
    return {'transitions':counts,'n':n,'before_positive':bp,'after_positive':ap,
            'change':f(Fraction(ap-bp,n)),
            'survival':f(Fraction(counts['n11'],bp)) if bp else None,
            'marginal_survival_bounds':[f(Fraction(max(0,ap+bp-n),bp)),f(Fraction(min(ap,bp),bp))] if bp else None}
def run(output):
    output=Path(output)
    require(not output.exists(),"output must be new")
    provenance=check_upstream()
    data=ROOT/'data/semcodebook_case'
    frozen=json.loads((data/'generations.json').read_text(encoding='utf-8'))
    require([v['payload'] for v in frozen]==list(range(16)),"all 16 payload attempts required")
    contract_id="integer-sum-finite:"+sha(json.dumps(CONTRACT,sort_keys=True))
    config={"upstream_commit":provenance['commit'],"generations_sha256":digest(data/'generations.json'),
            "adapter_sha256":digest(Path(__file__)),
            "native_threshold":0.5,"ecc_scheme":"soft_secded84_adaptive_v1",
            "decision":"native is_watermarked; abstain and reject are observed false for this Boolean estimand",
            "score":None,"side_information":"original generation schedule registered by sample ID, unchanged after edit",
            "key":"semcodebook-demo-key","payload_bits":4}
    rule_id="semcodebook-registered-schedule:"+sha(json.dumps(config,sort_keys=True))
    rows=[];executions=[];detector=SemCodebookDetector()
    for item in frozen:
        payload=item['payload']
        request=GenerationRequest(**item['request'])
        regenerated=rule_based_generate(request,BASE)
        require(json.loads(json.dumps(asdict(regenerated)))==item['generation'],"encoder replay differs")
        source=regenerated.watermarked_code
        schedule=regenerated.carrier_schedule
        spec=WatermarkSpec(wm_id=payload,carrier_schedule=schedule,implementation_stage=regenerated.implementation_stage)
        for edit,after in (('review-comment',source+FOOTER),('builtin-sum-refactor',AFTER)):
            programs=[];observations=[]
            for side,text in (('original',source),('edited',after)):
                passed,calls=validate(text)
                detection=detector.detect(text,spec,language='python')
                programs.append({"source":text,"source_sha256":sha(text),
                    "validation":{"available":True,"passed":passed,"contract_id":contract_id},
                    "detection":{"available":True,"detected":detection.is_watermarked,"score":None,
                                 "threshold":0.5,"comparator":"ge","rule_id":rule_id}})
                observations.append({"side":side,"source_sha256":sha(text),"calls":calls,
                                     "native_detection":asdict(detection)})
            rows.append({"schema_version":"editmark-pair","population":"watermarked_positive",
                "evidence_kind":"saved_observation","experiment_id":"SEMCODEBOOK-CPU-CONTROLLED",
                "method":"SemCodebook-bootstrap-rule-based","model":"none",
                "source_group":"author-owned-integer-sum-controlled","language":"python",
                "sample_id":f"payload-{payload:02d}","replicate_id":"one",
                "cluster_id":"one-arithmetic-task","edit":edit,"variant_id":"fixed-source-edit",
                "supported":True,"changed":True,"original":programs[0],"edited":programs[1]})
            executions.append({"sample_id":f"payload-{payload:02d}","edit":edit,"observations":observations})
    paired=report([Pair.parse(r) for r in rows])
    comparisons=[]
    for s in paired['strata']:
        subset=[r for r in rows if r['edit']==s['edit']]
        ordinary=direct(subset)
        counts=from_pair_stratum(s,'saved_observation');exact=analyze(counts)
        require(ordinary['transitions']==counts['transitions'],"direct transition mismatch")
        require(ordinary['change']==[exact['net_detection_change'][k] for k in ('numerator','denominator')],"direct change mismatch")
        require(ordinary['survival']==[exact['survival'][k] for k in ('numerator','denominator')],"direct survival mismatch")
        marginal=dict(counts);marginal.pop('transitions')
        coarse=analyze(marginal)
        require(ordinary['marginal_survival_bounds']==[[coarse['survival_bounds'][side][k] for k in ('numerator','denominator')] for side in ('lower','upper')],"marginal bound mismatch")
        comparisons.append({"edit":s['edit'],"ordinary":ordinary,"joint_counts":counts,"marginal_counts":marginal})
    output.mkdir(parents=True,exist_ok=False)
    raw=output/'pairs.jsonl'
    with raw.open('x',encoding='utf-8',newline='\n') as stream:
        for row in rows: stream.write(json.dumps(row,sort_keys=True)+'\n')
    write_new_json(output/'pair-report.json',paired)
    outcomes={}
    for claim in certificate.CLAIMS:
        cp=output/(claim+'.json')
        status=certificate.main([str(raw),'--claim',claim,'--require-observations','--out',str(cp)])
        bundle=json.loads(cp.read_text(encoding='utf-8'))
        checked=verify.verify_bundle_against_basis(bundle,raw)
        require(checked['raw_pair_membership_rederived'] and checked['basis_coverage_complete'],"basis not rederived")
        write_new_json(output/(claim+'-verification.json'),checked)
        outcomes[claim]={"exit_code":status,"decisions":[c['decision'] for c in bundle['certificates']],
                         "raw_pair_membership_rederived":True}
    unique_before={r['original']['source_sha256'] for r in rows}
    all_sources={r[side]['source_sha256'] for r in rows for side in ('original','edited')}
    value={"origin":"Actual controlled execution of author's pre-existing implementation; not independent or published-method evidence",
        "evidence_kind":"saved_observation","program_task_count":1,"payload_attempts":16,"edit_variants":2,
        "pairs":len(rows),"distinct_before_source_texts":len(unique_before),"distinct_source_texts":len(all_sources),
        "functional_calls":sum(len(o['calls']) for x in executions for o in x['observations']),
        "functional_failures":sum(not c['passed'] for x in executions for o in x['observations'] for c in o['calls']),
        "contract":CONTRACT,"configuration":config,"rule_id":rule_id,
        "direct_comparisons":comparisons,"executions":executions,"claim_outcomes":outcomes,
        "environment":{"python":sys.version,"platform":platform.platform(),"utc":datetime.now(timezone.utc).isoformat()},
        "model_inference":False,"attack_workflow":False,"selection_or_detection_gain":0,
        "limits":"One authored task, all 16 payload attempts including misses; shared code and schedules are dependent. Native Boolean hit is not exact payload recovery; confidence is not a scalar decision score. Registered schedule side information is supplied, not blind reconstruction. No controls, AUROC, FPR, posterior or independent robustness study."}
    write_new_json(output/'measurement.json',value)
    print(json.dumps({k:value[k] for k in ('pairs','distinct_before_source_texts','distinct_source_texts','functional_calls','functional_failures','claim_outcomes')},indent=2))
    return value
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
