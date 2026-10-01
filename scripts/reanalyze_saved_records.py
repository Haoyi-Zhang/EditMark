#!/usr/bin/env python3
"""Reanalyze EXPLICIT saved observation files. This never reruns experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_v2.paired import EvidenceError, cluster_interval, grouped_report, read_records


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+',type=Path,help='Raw BenchmarkReport JSON, raw JSONL, or normalized editmark-pair-v1 records; NOT summary tables')
    p.add_argument('--experiment-id',help='Required for legacy BenchmarkRow inputs; assign a stable archive/study identity')
    p.add_argument('--out',type=Path,required=True,help='A new output JSON; existing paths are protected')
    p.add_argument('--bootstrap',type=int,default=0,help='Optional cluster resamples; requires audited cluster_id sidecars')
    args=p.parse_args()
    try:
        if args.out.exists():
            raise EvidenceError(f'Refusing to overwrite {args.out}; choose a new output')
        rows=[]; hashes=[]
        for path in args.inputs:
            rows.extend(read_records(path,experiment_id=args.experiment_id))
            hashes.append({'name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        report=grouped_report(rows)
        report['input_files']=hashes
        if args.bootstrap:
            report['cluster_resampling']=cluster_interval(rows,repetitions=args.bootstrap)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8') as handle:
            handle.write(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
        n=report['pooled']['changed_only']['admitted_pairs']
        print(f'SAVED_RECORD_ANALYSIS_OK rows={report["unique_input_rows"]} strict_pairs={n} output={args.out}')
        if not n:
            print('No strict paired estimate: inspect exclusion_counts. null is not zero.')
        return 0
    except (EvidenceError, OSError, json.JSONDecodeError) as exc:
        print(f'EVIDENCE_ERROR: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
