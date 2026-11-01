"""Stdlib ZIP extraction + exact bytes + separate scoring + rank-sum AUC.

No encoder, detector, task/source execution, external packages or network.
"""
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile


def need(v,m):
    if not v: raise ValueError(m)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text('utf-8-sig'))


def rank_sum_auc(positive,negative):
    """Independent arithmetic: tie-block mean ranks, not pairwise comparisons."""
    need(positive and negative,'Both classes required')
    values=sorted([(x,1) for x in positive]+[(x,0) for x in negative])
    i=0; rank_total=Fraction(0)
    while i<len(values):
        j=i+1
        while j<len(values) and values[j][0]==values[i][0]: j+=1
        mean_rank=Fraction(i+1+j,2)
        rank_total += mean_rank*sum(y for _,y in values[i:j]); i=j
    p=len(positive); n=len(negative)
    return (rank_total-Fraction(p*(p+1),2))/(p*n)


def independent_recount(root):
    result=read(root/'run_001/result.json'); checks=[]
    for split in ('development','holdout'):
        rows=[json.loads(s) for s in (root/'run_001'/split/'pairs.jsonl').read_text('utf-8').splitlines()]
        for m in result['metrics'][split]:
            groups={}
            for r in rows:
                if r['edit']==m['edit'] and r['first_exclusion'] is None:
                    groups.setdefault(r['context_id'],{})[r['label']]=r
            matches=[pair for pair in groups.values() if set(pair)=={0,1}]
            for side in ('original','edited'):
                if 'pooled_score_auc_matched_population' not in m[side]: continue
                values={label:[Fraction(str(pair[label][side]['detections'][m['rule']]['output']['positive_support_score'])) for pair in matches] for label in (0,1)}
                actual=rank_sum_auc(values[1],values[0])
                target=m[side]['pooled_score_auc_matched_population']['value']
                need(actual==Fraction(target['numerator'],target['denominator']),'Independent rank-sum disagreement')
                neg=[r for r in rows if r['edit']==m['edit'] and r['label']==0 and r['first_exclusion'] is None]
                hits=sum(r[side]['detections'][m['rule']]['output']['is_watermarked'] for r in neg)
                rate=m[side]['native_boolean_fpr_all_eligible_controls']
                need(Fraction(hits,len(neg))==Fraction(rate['numerator'],rate['denominator']),'Independent FPR disagreement')
                checks.append({'split':split,'rule':m['rule'],'edit':m['edit'],'side':side,
                               'auc':[actual.numerator,actual.denominator],'control_hits':hits,'control_contexts':len(neg)})
    return checks


def verify(archive,out):
    archive=archive.resolve(); out=out.resolve()
    need(not out.exists(),'Extraction directory must be new')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist(); need(len(names)==len(set(names)),'Duplicate ZIP entries')
        for info in z.infolist():
            path=PurePosixPath(info.filename)
            need(not path.is_absolute() and '..' not in path.parts and '\\' not in info.filename and ':' not in info.filename,'Unsafe ZIP name')
            need(stat.S_IFMT(info.external_attr>>16)==stat.S_IFREG,'Only ordinary files')
            need((out/Path(*path.parts)).resolve().is_relative_to(out),'Escaping member')
        manifest=json.loads(z.read('MANIFEST.json'))
        files=manifest['files']
        need(set(names)==set(files)|{'MANIFEST.json'},'Extra or missing ZIP member')
        need(sum(i.file_size for i in z.infolist())<=67108864,'Unexpected extraction size')
        need(z.testzip() is None,'ZIP CRC failed')
        z.extractall(out)  # ordinary stdlib extraction after path/type/size checks
    for rel,expected in files.items():
        p=out/rel
        need(p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'],'Original bytes differ: '+rel)
    original_hashes={rel:sha(out/rel) for rel in files}
    recalculated=out/'verification-recomputed-result.json'
    command=[sys.executable,'-B',str(out/'candidate/score_consumer.py'),'report',
        '--development',str(out/'run_001/development'),'--holdout',str(out/'run_001/holdout'),
        '--freeze',str(out/'run_001/freeze.json'),'--selection',str(out/'run_001/selection.json'),
        '--out',str(recalculated)]
    checked=subprocess.run(command,capture_output=True,text=True,timeout=60)
    need(checked.returncode==0,'Extracted scoring consumer failed: '+checked.stderr)
    need(recalculated.read_bytes()==(out/'run_001/result.json').read_bytes(),'Score result not byte-identical')
    ranks=independent_recount(out)
    need(all(sha(out/rel)==value for rel,value in original_hashes.items()),'Original member mutated during scoring')
    value={'status':'LOSSLESS_EXTRACTION_AND_SCORE_VALIDATION_PASSED','archive_sha256':sha(archive),
        'manifest_sha256':sha(out/'MANIFEST.json'),'original_files_checked':len(files),
        'original_bytes_checked':sum(x['bytes'] for x in files.values()),
        'all_original_bytes_identical':True,'scoring_result_byte_identical':True,
        'scoring_result_sha256':sha(recalculated),'scoring_command':command,
        'independent_rank_sum_and_fpr_checks':ranks,'native_execution_repeated':False,
        'scope':'Integrity/membership/arithmetic rederivation from supplied observations, not external execution attestation'}
    with (out/'verification-receipt.json').open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2); f.write('\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('archive',type=Path); p.add_argument('out',type=Path)
    a=p.parse_args(); result=verify(a.archive,a.out)
    print(json.dumps({k:result[k] for k in ('status','original_files_checked','original_bytes_checked','archive_sha256')}))
