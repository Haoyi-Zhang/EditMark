"""Offline normalized-pair analysis: python -m editmark_audit INPUT --out NEW.json."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .analysis import report
from .records import Pair
from .io import EvidenceError, digest, load_rows, write_new_json


def main(argv: list[str] | None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path,nargs='+')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-estimate',action='store_true',help='Exit 3 unless every stratum has at least one strict pair')
    p.add_argument('--require-observations',action='store_true',help='Reject explicitly synthetic fixtures')
    args=p.parse_args(argv)
    try:
        pairs=[];inputs=[]
        for path in args.inputs:
            h=digest(path)
            for i,row in enumerate(load_rows(path),1):
                try:pairs.append(Pair.parse(row))
                except EvidenceError as exc:raise EvidenceError(f'{path.name}, row {i}: {exc}') from exc
            if digest(path)!=h: raise EvidenceError('Input changed during analysis')
            inputs.append(dict(name=path.name,sha256=h))
        out=report(pairs);out['input_files']=inputs
        if args.require_observations and out['evidence_kind']!='saved_observation':
            raise EvidenceError('This file is explicitly synthetic, not archived observations')
        write_new_json(args.out,out)
        complete=all(s['estimates']['changed_only']['admitted_pairs']>0 for s in out['strata'])
        print(f'REPORT_CREATED unique_pairs={out["unique_pair_count"]} strata={out["stratum_count"]} kind={out["evidence_kind"]}')
        if args.require_estimate and not complete:return 3
        return 0
    except (EvidenceError,OSError,UnicodeError,ValueError) as exc:
        print(f'EVIDENCE_ERROR: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
