#!/usr/bin/env python3
"""Audit frozen inputs and exported aggregates; never execute a model or a task.

This is an inventory/consistency audit, NOT a recomputation of experimental
metrics from observations. Rounded aggregate rates are never converted to counts.
Only Python's standard library is used. Outputs are deterministic.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TABLES = Path('results/tables/suite_all_models_methods')
METHODS = ('KGW', 'SWEET', 'STONE', 'EWD')
EDITS = ('whitespace_normalize', 'comment_strip', 'identifier_rename',
         'noise_insert', 'block_shuffle', 'control_flow_flatten', 'budgeted_adaptive')


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f'No rows for {path.name}; do not silently emit an empty table')
    with path.open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(root: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    used: set[Path] = set()
    def read(rel: Path | str) -> Any:
        p = root / rel
        used.add(p)
        return load(p)
    corpus, language_counts, exclusions, curated = [], [], [], []
    for p in sorted((root/'data/release/sources').glob('*.normalized.jsonl')):
        used.add(p)
        rows = [json.loads(line) for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        groups = {r['source_group'] for r in rows}
        if len(groups) != 1:
            raise ValueError(f'Mixed groups in {p.name}: {groups}')
        group = groups.pop()
        included = [r for r in rows if r.get('reference_kind') == 'canonical']
        corpus.append(dict(source_group=group, release_records=len(rows),
                           canonical_records=len(included), excluded_records=len(rows)-len(included),
                           inventory_family_ids=len({r.get('family_id') for r in rows}),
                           release_file=str(p.relative_to(root))))
        for lang in sorted({r['language'] for r in rows}):
            rr = [r for r in rows if r['language'] == lang]
            language_counts.append(dict(source_group=group, language=lang,
                release_records=len(rr), canonical_records=sum(r.get('reference_kind')=='canonical' for r in rr),
                excluded_records=sum(r.get('reference_kind')!='canonical' for r in rr)))
        for row in rows:
            if row.get('reference_kind') != 'canonical':
                exclusions.append(dict(source_group=group, task_id=row.get('task_id'),
                    language=row['language'], reference_kind=row.get('reference_kind'),
                    canonical_available=row.get('canonical_available'),
                    source_file=str(p.relative_to(root))))
        if group.startswith('crafted_'):
            curated.append(dict(source_group=group, records=len(rows),
                named_family_ids=len({r.get('family_id') for r in rows}),
                template_families=dict(sorted(Counter(str(r.get('template_family')) for r in rows).items())),
                categories=dict(sorted(Counter(str(r.get('category')) for r in rows).items())),
                difficulty=dict(sorted(Counter(str(r.get('difficulty')) for r in rows).items())),
                note='Static template inventory, not independent expert review or observed test success'))
    desc = {r['method']:r for r in read(TABLES/'method_summary.json')}
    masters = {r['method']:r for r in read(TABLES/'suite_all_models_methods_method_master_leaderboard.json')}
    timing = {r['method']:r for r in read(TABLES/'timing_summary.json')}
    inv = read(TABLES/'suite_all_models_methods_run_inventory.json')
    if isinstance(inv, dict):
        inv = inv.get('runs', inv.get('run_inventory', []))
    if not isinstance(inv, list) or not inv:
        raise ValueError('Unrecognized run inventory schema')
    per_edit, methods, negatives, source_view = [], [], [], []
    for method in METHODS:
        d, m = desc[method], masters[method]
        cov = m['score_coverage']
        methods.append(dict(method=method,
            descriptive_tasks_all_models=d['task_count'], descriptive_edit_rows=d['row_count'],
            deduplicated_edit_rows=m['row_count'], duplicate_edit_rows_removed=m['duplicate_rows_removed'],
            master_positive_examples=cov['positive_examples'],
            descriptive_clean_auroc=d['detection_separability'],
            descriptive_unmarked_test_pass_mean=d['clean_test_pass_rate'],
            descriptive_marked_test_pass_mean=d['watermarked_test_pass_rate'],
            descriptive_edited_test_pass_mean=d['attacked_test_pass_rate'],
            descriptive_clean_control_fpr_mean=d['negative_control_fpr'],
            legacy_descriptive_robustness_index=d['robustness'],
            legacy_descriptive_utility_index=d['utility'],
            legacy_preservation_ratio_not_admission=d['attacked_pass_preservation'],
            master_clean_auroc=m['detection_separability'],
            legacy_master_robustness_index=m['robustness'],
            legacy_master_utility_index=m['utility'],
            master_edited_validation_pass_mean=m['semantic_preservation_rate'],
            master_edited_detected_among_edited_pass_mean=m['attacked_detected_semantic_rate'],
            paired_valid_changed_n=None, paired_valid_changed_tpr_before=None,
            paired_valid_changed_tpr_after=None, paired_tpr_loss=None,
            paired_status='NOT_IDENTIFIABLE_FROM_RELEASED_SUMMARIES'))
        negatives.append(dict(method=method, master_positive_examples=cov['positive_examples'],
            human_reference_control_count=cov['human_reference_negatives'],
            clean_generation_control_count=cov['clean_generation_negatives'],
            clean_control_coverage=cov['negative_control_support_rate'],
            transformed_control_count=None, transformed_fpr=None,
            transformed_status='NO_TRANSFORMED_CONTROL_PIPELINE_IDENTIFIED'))
        for name in EDITS:
            a = cov['attack_breakdown'][name]
            factors = [a[k] for k in ('attack_retention','attack_attacked_detected_semantic_rate','attack_attacked_pass_preservation') if a[k] is not None]
            recomposed = sum(factors)/len(factors) if factors else None
            residual = abs(recomposed-a['attack_robustness']) if recomposed is not None and a['attack_robustness'] is not None else None
            if residual is not None and residual > 0.00011:
                raise ValueError(f'Unexpected component arithmetic: {method}, {name}, {residual}')
            per_edit.append(dict(method=method, edit=name, tier=a['attack_tier'],
                total_rows=a['attack_total_rows'], supported_rows=a['attack_supported_rows'],
                application_coverage=a['attack_supported_rows']/a['attack_total_rows'] if a['attack_total_rows'] else None,
                legacy_factor_availability_not_application_coverage=a['attack_support_rate'],
                legacy_score_ratio_component=a['attack_retention'],
                edited_detection_given_edited_pass=a['attack_attacked_detected_semantic_rate'],
                legacy_unpaired_pass_ratio_component=a['attack_attacked_pass_preservation'],
                legacy_pooled_composite=a['attack_robustness'],
                arithmetic_check_from_rounded_components=recomposed,
                recomposition_absolute_rounding_error=residual,
                changed_rows=None, originally_and_edited_pass_rows=None,
                source='master.score_coverage.attack_breakdown (pooled, NOT source-balanced headline)'))
        for s, v in cov['source_balanced_sources'].items():
            if v['row_count'] % 35:
                raise ValueError(f'Cannot reconcile 5 models x 7 edits: {method}/{s}')
            source_view.append(dict(method=method, source_group=s,
                master_edit_rows=v['row_count'],
                implied_source_records_after_legacy_dedup=v['row_count']//35,
                basis='Exported edit-row count divided by documented 5-model x 7-edit design; not re-observed raw records'))
    config_matrix = read('configs/matrices/suite_all_models_methods.json')
    configurations = []
    def merge(a: dict, b: dict) -> dict:
        ans = dict(a)
        for k,v in b.items():
            ans[k] = merge(ans[k],v) if isinstance(v,dict) and isinstance(ans.get(k),dict) else v
        return ans
    for r in config_matrix['runs']:
        cfg = merge(read(r['config']), r.get('config_overrides',{}))
        wm = cfg['watermark']
        configurations.append(dict(run_id=r['run_id'], method=r['method'], model=r['model'],
            source_slug=r['source_slug'], seed=cfg['project']['seed'],
            model_revision=r.get('model_revision'), gamma=wm.get('gamma'), delta=wm.get('delta'),
            z_threshold=wm.get('z_threshold'), max_new_tokens=wm.get('max_new_tokens'),
            temperature=wm.get('temperature'), top_p=wm.get('top_p'),
            entropy_threshold=wm.get('entropy_threshold'), prefix_length=wm.get('prefix_length'),
            seeding_scheme=wm.get('seeding_scheme'), no_repeat_ngram_size=wm.get('no_repeat_ngram_size'),
            config_path=r['config'], reference_kinds=';'.join(cfg['benchmark'].get('include_reference_kinds',[]))))
    observed_status = Counter(str(r.get('status')) for r in inv)
    result = dict(schema_version='editmark-frozen-audit-v1',
        basis='Read-only static inventory and arithmetic checks of frozen aggregates; no experiment rerun',
        release_records=sum(r['release_records'] for r in corpus),
        canonical_source_records=sum(r['canonical_records'] for r in corpus),
        excluded_source_records=sum(r['excluded_records'] for r in corpus),
        mbxp_canonical_go_records=sum(r['canonical_records'] for r in language_counts if r['source_group']=='public_mbxp_5lang' and r['language']=='go'),
        run_inventory_rows=len(inv), run_status_counts=dict(observed_status),
        model_count=len(config_matrix['model_roster']), method_count=len(METHODS),
        source_group_count=len(corpus),
        raw_results_directories_present={str(p): (root/p).exists() for p in (Path('results/runs'), Path('results/matrix'))},
        availability=dict(strict_paired_detection='not available in supplied archive',
            no_op_sensitivity='not available; legacy BenchmarkRow serialization does not explicitly persist outcome.changed',
            transformed_negative_controls='not identified in saved pipeline',
            exact_joint_admission_counts='not available in aggregate exports',
            root_causes_of_failed_execution='not inferable from these summaries'),
        caveat='Rates are published rounded summaries, not recovered raw-event counts. Master and descriptive aggregation views must not be mixed.',
        corpus=corpus, method_profiles=methods, negative_control_coverage=negatives,
        curated_static_inventory=curated)
    if (result['release_records'],result['canonical_source_records'],result['excluded_source_records']) != (1662,1605,57):
        raise ValueError('This frozen-profile audit expects 1662/1605/57; investigate changed inputs')
    for name, rows in [('source_inventory',corpus),('language_inventory',language_counts),
        ('excluded_records',exclusions),('method_profiles',methods),('edit_component_audit',per_edit),
        ('negative_controls',negatives),('source_view_reconciliation',source_view),
        ('runtime_configuration',configurations),('timing_stages',[timing[m] for m in METHODS])]:
        write_json(out/(name+'.json'),rows)
        write_csv(out/(name+'.csv'),rows)
    write_json(out/'frozen_audit.json', result)
    used.update(root/p for p in ['posteditbench/scorecard.py','posteditbench/models.py',
        'posteditbench/pipeline/orchestrator.py','posteditbench/leaderboards.py',
        'posteditbench/attacks/implementations.py','posteditbench/crafted_benchmarks.py','posteditbench/crafted_templates.py'])
    write_json(out/'input_hashes.json', {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(used)})
    lines = ['# Frozen-evidence audit','',result['basis'],'',
        f"Inventory: **{result['release_records']}** source records; canonical executed design: **{result['canonical_source_records']}**; excluded overlay: **{result['excluded_source_records']}**.",
        f"MBXP canonical Go records: **{result['mbxp_canonical_go_records']}**. Run inventory: **{len(inv)}**.", '',
        '## What the exported scores mean','',
        'D is clean AUROC. R is a composite of a clipped detector-score ratio, an edited-pass-conditioned detection rate, and an unpaired test-pass-rate ratio. **D - R is not a detector-evidence loss.**', '',
        'EditSup is a clipped pass-rate ratio; U includes similarity and validation availability. Neither is an absolute joint pass probability.', '',
        '## Data that cannot be recovered from rounded summaries','',
        'The joint original/edited test outcomes, actual-change flags, paired before/after decisions, and transformed controls are missing. Their counts and estimates stay null, not zero.', '',
        '## Reading the exports','',
        '`method_profiles` keeps descriptive and master summaries in separate columns. `edit_component_audit` uses pooled per-edit components and does not reconstruct the source-balanced headline. `source_view_reconciliation` accounts for the additional 62 source records removed by the legacy cross-source deduplication design (38 HumanEval-X and 24 MBXP), inferred from exported row counts; it does not certify equivalence of generated outputs.', '',
        'Every `input_hashes.json` entry is an actual input to this audit. Historical experiment files were not edited. All displayed rates retain the precision and limitations of their source exports.', '']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=ROOT/'analysis')
    args=p.parse_args()
    report=audit(args.root.resolve(), args.out.resolve())
    print(f"AUDIT_OK inventory={report['release_records']} canonical={report['canonical_source_records']} excluded={report['excluded_source_records']} runs={report['run_inventory_rows']}")
    print('PAIRED_EFFECT=NOT_IDENTIFIABLE_FROM_RELEASED_SUMMARIES')

if __name__=='__main__':
    main()
