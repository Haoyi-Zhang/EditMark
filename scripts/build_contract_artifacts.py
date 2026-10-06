#!/usr/bin/env python3
"""Build deterministic paper facts, mutation checks, and the contract figure."""
from __future__ import annotations
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from editmark_audit.aggregates import analyze as analyze_counts
from editmark_audit.analysis import report, summarize
from editmark_audit.certificate import canonical_hash, certify_analysis
from editmark_audit.constants import COUNT_SCHEMA, STRICT_CONTRACT
from editmark_audit.io import EvidenceError, write_new_json
from editmark_audit.records import Pair, deduplicate
from editmark_audit.verify import verify_certificate


def pair_fixture(name='sample', before=True, after=False):
    def program(source, detected):
        return {
            'source': source,
            'validation': {'available': True, 'passed': True, 'contract_id': 'synthetic-checks'},
            'detection': {
                'available': True, 'detected': detected,
                'score': 1.0 if detected else 0.0,
                'threshold': 0.5, 'comparator': 'ge', 'rule_id': 'synthetic-rule',
            },
        }
    return {
        'schema_version': 'editmark-pair', 'population': 'watermarked_positive',
        'evidence_kind': 'synthetic_test', 'experiment_id': 'SYNTHETIC',
        'method': 'toy', 'model': 'toy', 'source_group': 'toy', 'language': 'python',
        'sample_id': name, 'replicate_id': 'r0', 'edit': 'rename', 'variant_id': 'v0',
        'cluster_id': name, 'supported': True, 'changed': True,
        'original': program(f'x = {name!r}', before),
        'edited': program(f'x={name!r}', after),
    }


def count_fixture():
    return {
        'schema_version': COUNT_SCHEMA, 'population': 'watermarked_positive',
        'evidence_kind': 'synthetic_test', 'analysis_contract': STRICT_CONTRACT,
        'cohort_sha256': 'a' * 64,
        'stratum': {
            'experiment_id': 'SYNTHETIC', 'method': 'toy', 'model': 'toy',
            'source_group': 'toy', 'language': 'python', 'edit': 'rename', 'variant_id': 'v0',
        },
        'decision_rule': {'rule_id': 'synthetic-rule', 'threshold': 0.5, 'comparator': 'ge'},
        'counts': {'n': 5, 'before_positive': 4, 'after_positive': 2},
        'transitions': {'n00': 0, 'n01': 1, 'n10': 3, 'n11': 1},
        'attempted_pairs': 5, 'first_exclusion_counts': {},
    }


def rejected(call: Callable[[], Any]) -> str:
    try:
        call()
    except (EvidenceError, FileExistsError, ValueError) as exc:
        return type(exc).__name__
    raise AssertionError('mutation was unexpectedly accepted')


def mutation_rows():
    rows=[]
    def add(identifier, stage, pressure, disposition, check):
        outcome=check()
        rows.append({
            'id': identifier, 'stage': stage, 'attack': pressure,
            'expected_disposition': disposition, 'observed': outcome,
        })

    d=pair_fixture();d['original']['unexpected']=1
    add('unknown-field','parse','Misspelled or undeclared record field','reject',lambda: rejected(lambda: Pair.parse(d)))

    d=pair_fixture();d['original']['source_sha256']='0'*64
    add('hash-mismatch','parse','Source text disagrees with supplied digest','reject',lambda: rejected(lambda: Pair.parse(d)))

    d=pair_fixture();d['changed']=False
    add('change-contradiction','parse','Change flag contradicts source hashes','reject',lambda: rejected(lambda: Pair.parse(d)))

    d=pair_fixture();d['edited']['validation']['contract_id']='other-checks'
    add('check-contract-drift','eligibility','Before and after use different validators','exclude',lambda: summarize([Pair.parse(d)])['first_exclusion_counts'].get('check_contract_changed'))

    d=pair_fixture();d['edited']['detection']['threshold']=0.8
    add('decision-rule-drift','eligibility','Before and after use different detector rules','exclude',lambda: summarize([Pair.parse(d)])['first_exclusion_counts'].get('decision_rule_changed'))

    d=pair_fixture();d['edited']['detection'].update(available=False,detected=None,score=None)
    add('missing-decision','identification','Eligible pair has an unobserved edited decision','bound',lambda: summarize([Pair.parse(d)])['missing_decision_change_bounds']['upper'])

    a=Pair.parse(pair_fixture('same'));b=copy.deepcopy(pair_fixture('same'));b['edited']['detection']['detected']=True;b['edited']['detection']['score']=1.0;b=Pair.parse(b)
    add('conflicting-duplicate','deduplication','Same identity carries incompatible outcomes','reject',lambda: rejected(lambda: deduplicate([a,b])))

    d=count_fixture();d['counts']['before_positive']=0.8
    add('rounded-count','count analysis','Rounded rate supplied where exact integer count is required','reject',lambda: rejected(lambda: analyze_counts(d)))

    d=count_fixture();d['attempted_pairs']=6
    add('broken-partition','count analysis','Attempted total does not equal admitted plus exclusions','reject',lambda: rejected(lambda: analyze_counts(d)))

    d=count_fixture();d['transitions']['n11']=2
    add('transition-conflict','count analysis','Joint table contradicts its marginals','reject',lambda: rejected(lambda: analyze_counts(d)))

    analysis=analyze_counts(count_fixture())
    add('positive-only-fpr','certificate','Edited FPR requested from positive observations','not admissible',lambda: certify_analysis(analysis,'edited_false_positive_rate')['decision'])
    add('positive-only-auroc','certificate','AUROC change requested without negative score distributions','not admissible',lambda: certify_analysis(analysis,'auroc_change')['decision'])
    add('posterior-provenance','certificate','Posterior origin probability requested without prevalence/calibration','not admissible',lambda: certify_analysis(analysis,'post_edit_provenance_probability')['decision'])

    certificate=certify_analysis(analysis,'positive_decision_change')
    certificate['supported_statement']='tampered statement'
    add('certificate-hash-tamper','verification','Certificate content changed without updating its hash','reject',lambda: rejected(lambda: verify_certificate(certificate)))

    certificate=certify_analysis(analysis,'positive_decision_change')
    certificate['estimate']={'numerator':1,'denominator':1,'decimal':1.0}
    core=dict(certificate);core.pop('certificate_sha256')
    certificate['certificate_sha256']=canonical_hash(core)
    add('certificate-semantic-tamper','verification','Estimate changed and certificate hash recomputed','reject',lambda: rejected(lambda: verify_certificate(certificate)))

    with tempfile.TemporaryDirectory() as td:
        output=Path(td)/'out.json';output.write_text('{}')
        add('existing-output','publication','A report path already exists','reject',lambda: rejected(lambda: write_new_json(output,{'x':1})))
    return rows


def tex_escape(value: str) -> str:
    return value.replace('\\','\\textbackslash{}').replace('_','\\_').replace('%','\\%').replace('&','\\&')


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def count_test_methods() -> int:
    total=0
    for path in sorted((ROOT/'tests/contract').glob('test_*.py')):
        tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        total += sum(
            isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_')
            for node in ast.walk(tree)
        )
    return total


def build_facts(mutation_count: int):
    audit=json.loads((ROOT/'analysis/frozen_audit.json').read_text())
    profiles=audit['method_profiles']
    edited=[100*float(row['descriptive_edited_test_pass_mean']) for row in profiles]
    attack_csv=ROOT/'results/tables/suite_all_models_methods/method_attack_summary.csv'
    with attack_csv.open(newline='',encoding='utf-8-sig') as handle:
        attack_rows=list(csv.DictReader(handle))
    edit_pass={
        (row['method'],row['attack']):100*float(row['attacked_test_pass_rate'])
        for row in attack_rows
    }
    attack_names={row['attack'] for row in attack_rows}
    adaptive_matches=[name for name in attack_names if 'adaptive' in name]
    if len(adaptive_matches) != 1:
        raise EvidenceError(f'expected one adaptive edit, found {sorted(adaptive_matches)!r}')
    high_coverage_edits={
        'whitespace_normalize','comment_strip','noise_insert','block_shuffle',
        adaptive_matches[0],
    }
    facts={
        'release_records':audit['release_records'],
        'configured_records':audit['canonical_source_records'],
        'excluded_records':audit['excluded_source_records'],
        'source_groups':audit['source_group_count'],
        'models':audit['model_count'],
        'methods':audit['method_count'],
        'run_inventory_rows':audit['run_inventory_rows'],
        'edited_pass_min':min(edited),
        'edited_pass_max':max(edited),
        'edit_pass_cell_min':min(edit_pass.values()),
        'edit_pass_cell_max':max(edit_pass.values()),
        'high_coverage_edit_pass_max':max(
            value for (_method,edit),value in edit_pass.items()
            if edit in high_coverage_edits
        ),
        'mbxp_go_configured':audit['mbxp_canonical_go_records'],
        'contract_test_count':count_test_methods(),
        'mutation_case_count':mutation_count,
        'claim_type_count':5,
    }
    lines=[
        f"\\newcommand{{\\ReleaseRecords}}{{{facts['release_records']:,}}}",
        f"\\newcommand{{\\ConfiguredRecords}}{{{facts['configured_records']:,}}}",
        f"\\newcommand{{\\ExcludedRecords}}{{{facts['excluded_records']:,}}}",
        f"\\newcommand{{\\SourceGroups}}{{{facts['source_groups']}}}",
        f"\\newcommand{{\\ModelSettings}}{{{facts['models']}}}",
        f"\\newcommand{{\\MethodIntegrations}}{{{facts['methods']}}}",
        f"\\newcommand{{\\RunInventoryRows}}{{{facts['run_inventory_rows']}}}",
        f"\\newcommand{{\\EditedPassMin}}{{{facts['edited_pass_min']:.2f}\\%}}",
        f"\\newcommand{{\\EditedPassMax}}{{{facts['edited_pass_max']:.2f}\\%}}",
        f"\\newcommand{{\\EditPassCellMin}}{{{facts['edit_pass_cell_min']:.2f}\\%}}",
        f"\\newcommand{{\\EditPassCellMax}}{{{facts['edit_pass_cell_max']:.2f}\\%}}",
        f"\\newcommand{{\\HighCoverageEditPassMax}}{{{facts['high_coverage_edit_pass_max']:.2f}\\%}}",
        f"\\newcommand{{\\ConfiguredGoRecords}}{{{facts['mbxp_go_configured']}}}",
        f"\\newcommand{{\\ContractTestCount}}{{{facts['contract_test_count']}}}",
        f"\\newcommand{{\\MutationCaseCount}}{{{facts['mutation_case_count']}}}",
        f"\\newcommand{{\\ClaimTypeCount}}{{{facts['claim_type_count']}}}",
    ]
    (ROOT.parent / 'paper/tosem/generated/release_facts.tex').write_text('\n'.join(lines)+'\n')
    write_json(ROOT/'analysis/release_facts.json',facts)
    return facts


def build_mutations():
    rows=mutation_rows()
    write_json(ROOT/'analysis/contract_mutations.json',{'synthetic':True,'rows':rows})
    with (ROOT/'analysis/contract_mutations.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tex=[]
    label={'reject':'Rejected','exclude':'Excluded','bound':'Bounded','not admissible':'Denied'}
    for row in rows:
        observed=row['observed']
        if isinstance(observed,float):observed=f'{observed:.2f}'
        tex.append(tex_escape(row['stage']) + ' & ' + tex_escape(row['attack']) + ' & ' +
                   tex_escape(label.get(row['expected_disposition'], row['expected_disposition'])) + r' \\')
    tex.append(r'\bottomrule')
    (ROOT.parent / 'paper/tosem/generated/mutation_rows.tex').write_text('\n'.join(tex)+'\n')
    return rows


def build_certificate_example():
    analysis=analyze_counts(count_fixture())
    certificates={claim:certify_analysis(analysis,claim) for claim in (
        'positive_decision_change','detection_survival','edited_false_positive_rate','auroc_change','post_edit_provenance_probability')}
    write_json(ROOT/'examples/synthetic_claim_certificates.json',{
        'synthetic':True,'certificates':certificates,
    })
    return certificates


def build_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    figure,axis=plt.subplots(figsize=(9.4,3.2))
    axis.set_xlim(0,10);axis.set_ylim(0,4);axis.axis('off')
    boxes=[
        (0.2,1.05,2.0,2.0,'1. Typed observation','source hash\ntest contract\ndetector rule'),
        (2.8,1.05,2.0,2.0,'2. Eligibility gate','supported + changed\nboth checks pass\nsame rule'),
        (5.4,1.05,2.0,2.0,'3. Sufficient facts','n00, n01, n10, n11\nexclusion partition\ncohort fingerprint'),
        (8.0,1.05,1.8,2.0,'4. Claim certificate','identified\nbounded\nnot admissible'),
    ]
    fills=['#eef3f7','#edf6f2','#f7f3e8','#f5eef6']
    for (x,y,w,h,title,body),fill in zip(boxes,fills):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=fill,edgecolor='#4a4a4a',linewidth=1.1)
        axis.add_patch(patch)
        axis.text(x+w/2,y+h-.34,title,ha='center',va='center',fontsize=10,fontweight='bold')
        axis.text(x+w/2,y+.75,body,ha='center',va='center',fontsize=9,linespacing=1.35)
    for x1,x2 in ((2.2,2.8),(4.8,5.4),(7.4,8.0)):
        axis.add_patch(FancyArrowPatch((x1,2.05),(x2,2.05),arrowstyle='-|>',mutation_scale=14,linewidth=1.2,color='#555555'))
    axis.text(5,0.55,'The certificate limits the sentence as well as the number.',ha='center',fontsize=10,fontweight='bold')
    axis.text(5,0.18,'Positive pairs can certify same-rule decision change; they cannot certify edited FPR, AUROC change, or posterior provenance.',ha='center',fontsize=8.5)
    figure.tight_layout(pad=.2)
    out=ROOT.parent / 'paper/tosem/figures';out.mkdir(parents=True,exist_ok=True)
    figure.savefig(out/'evidence_contract.pdf',bbox_inches='tight')
    figure.savefig(out/'evidence_contract.png',dpi=220,bbox_inches='tight')
    plt.close(figure)


def manifest(paths):
    output = ROOT.parent / 'paper/tosem/generated/asset_manifest.json'
    result = json.loads(output.read_text(encoding='utf-8')) if output.exists() else {'assets': []}
    if not isinstance(result, dict):
        raise EvidenceError('Expected an asset manifest object')
    entries = {entry['path']: entry for entry in result.get('assets', [])}
    for path in paths:
        data = path.read_bytes()
        relative = path.relative_to(ROOT.parent).as_posix()
        entries[relative] = {'path': relative, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    result['assets'] = list(entries.values())
    write_json(output, result)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-figure',action='store_true')
    args=parser.parse_args(argv)
    rows=build_mutations();build_facts(len(rows));build_certificate_example()
    if not args.skip_figure:build_figure()
    paths=[ROOT/'analysis/release_facts.json',ROOT/'analysis/contract_mutations.json',ROOT/'analysis/contract_mutations.csv',
           ROOT/'examples/synthetic_claim_certificates.json',ROOT.parent / 'paper/tosem/generated/release_facts.tex',ROOT.parent / 'paper/tosem/generated/mutation_rows.tex']
    if not args.skip_figure:paths += [ROOT.parent / 'paper/tosem/figures/evidence_contract.pdf',ROOT.parent / 'paper/tosem/figures/evidence_contract.png']
    manifest(paths)
    print('CONTRACT_ASSETS_BUILT')
    return 0

if __name__=='__main__':raise SystemExit(main())
