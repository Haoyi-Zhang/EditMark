#!/usr/bin/env python3
"""Build deterministic manuscript tables and figures from static audit exports.

This script reads checked-in summaries only. It does not execute a model,
watermark detector, transformation, validator, or generated program.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'analysis'
P = ROOT.parent / 'paper' / 'tosem'
G = P / 'generated'
F = P / 'figures'
G.mkdir(parents=True, exist_ok=True)
F.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
except ImportError as exc:
    raise SystemExit('Plotting requires matplotlib.') from exc


def load(name: str):
    return json.loads((A / name).read_text(encoding='utf-8'))


def file_entry(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        'path': str(path.relative_to(ROOT)),
        'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
    }


methods = ['KGW', 'SWEET', 'STONE', 'EWD']
source_labels = {
    'public_humaneval_plus': 'HumanEval+',
    'public_mbpp_plus': 'MBPP+',
    'public_humaneval_x': 'HumanEval-X slice',
    'public_mbxp_5lang': 'MBXP slice',
    'crafted_original': 'Constructed original',
    'crafted_translation': 'Constructed translation',
    'crafted_stress': 'Constructed stress',
}
source_order = list(source_labels)

# Source inventory table.
sources = {row['source_group']: row for row in load('source_inventory.json')}
rows = []
for key in source_order:
    row = sources[key]
    rows.append(
        f"{source_labels[key]} & {row['release_records']:,} & "
        f"{row['canonical_records']:,} & {row['excluded_records']:,} & "
        f"{row['inventory_family_ids']:,} \\\\"
    )
rows.append(r'\midrule Total & 1,662 & 1,605 & 57 & 766 \\')
(G / 'source_rows.tex').write_text('\n'.join(rows) + '\n' + r'\bottomrule' + '\n', encoding='utf-8')

# Method profile table.
profiles = {row['method']: row for row in load('method_profiles.json')}
rows = []
for method in methods:
    row = profiles[method]
    rows.append(
        f"{method} & {row['descriptive_clean_auroc']:.4f} & "
        f"{100 * row['descriptive_clean_control_fpr_mean']:.2f} & "
        f"{100 * row['descriptive_unmarked_test_pass_mean']:.2f} & "
        f"{100 * row['descriptive_marked_test_pass_mean']:.2f} & "
        f"{100 * row['descriptive_edited_test_pass_mean']:.2f} \\\\"
    )
(G / 'profile_rows.tex').write_text('\n'.join(rows) + '\n' + r'\bottomrule' + '\n', encoding='utf-8')

# Edit-specific absolute test-pass figure from the checked-in descriptive export.
attack_csv = ROOT / 'results' / 'tables' / 'suite_all_models_methods' / 'method_attack_summary.csv'
with attack_csv.open(newline='', encoding='utf-8-sig') as handle:
    attack_rows = list(csv.DictReader(handle))
edit_order = [
    'whitespace_normalize', 'comment_strip', 'noise_insert', 'identifier_rename',
    'block_shuffle', 'control_flow_flatten', 'budgeted_adaptive',
]
edit_labels = [
    'Whitespace', 'Comment removal', 'Comment insertion', 'Identifier renaming',
    'Block shuffle', 'Control-flow edit', 'Budgeted adaptive',
]
attack_present = {row['attack'] for row in attack_rows}
if 'budgeted_adaptive' not in attack_present:
    matches = [name for name in attack_present if 'adaptive' in name]
    if len(matches) != 1:
        raise SystemExit(f'Unexpected attack names: {sorted(attack_present)!r}')
    edit_order[-1] = matches[0]
edit_pass = {
    (row['method'], row['attack']): 100.0 * float(row['attacked_test_pass_rate'])
    for row in attack_rows
}
markers = ['o', 's', '^', 'D']
fig, ax = plt.subplots(figsize=(8.0, 4.7))
for j, method in enumerate(methods):
    ax.plot(
        [edit_pass[(method, name)] for name in edit_order],
        [i + (j - 1.5) * 0.10 for i in range(len(edit_order))],
        markers[j], markersize=5, label=method, linestyle='none'
    )
ax.set_yticks(range(len(edit_order)), edit_labels)
ax.invert_yaxis()
ax.set_xlabel('Edited-program test-pass fraction in method-edit export (%)')
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.01), frameon=False)
ax.grid(axis='x', linestyle=':', alpha=0.45)
fig.tight_layout()
fig.savefig(F / 'edit_functionality.pdf', metadata={'CreationDate': None, 'ModDate': None})
fig.savefig(F / 'edit_functionality.png', dpi=180)
plt.close(fig)

# Transformation applicability figure.
components = load('edit_component_audit.json')
present = {row['edit'] for row in components}
if not set(edit_order).issubset(present):
    raise SystemExit(f'Unexpected edit names: {sorted(present)!r}')
fig, ax = plt.subplots(figsize=(8.0, 4.7))
for j, method in enumerate(methods):
    by_edit = {row['edit']: row for row in components if row['method'] == method}
    ax.plot(
        [100 * by_edit[name]['application_coverage'] for name in edit_order],
        [i + (j - 1.5) * 0.10 for i in range(len(edit_order))],
        markers[j], markersize=5, label=method, linestyle='none'
    )
ax.set_yticks(range(len(edit_order)), edit_labels)
ax.invert_yaxis()
ax.set_xlim(-2, 103)
ax.set_xlabel('Rows reported applicable / attempted edit rows (%)')
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.01), frameon=False)
ax.grid(axis='x', linestyle=':', alpha=0.45)
fig.tight_layout()
fig.savefig(F / 'application_coverage.pdf', metadata={'CreationDate': None, 'ModDate': None})
fig.savefig(F / 'application_coverage.png', dpi=180)
plt.close(fig)

# Remove manuscript assets that are no longer referenced.
for name in ('utility_vs_pass.pdf', 'utility_vs_pass.png',
             'identification_bounds.pdf', 'identification_bounds.png',
             'source_functionality.pdf', 'source_functionality.png'):
    path = F / name
    if path.exists():
        path.unlink()

tracked = [
    G / 'release_facts.tex', G / 'mutation_rows.tex', G / 'source_rows.tex',
    G / 'profile_rows.tex', G / 'witness_rows.tex',
    F / 'evidence_contract.pdf', F / 'evidence_contract.png',
    F / 'edit_functionality.pdf', F / 'edit_functionality.png',
    F / 'application_coverage.pdf', F / 'application_coverage.png',
]
manifest = {
    'basis': [
        'analysis/frozen_audit.json', 'analysis/source_inventory.json',
        'analysis/method_profiles.json', 'analysis/edit_component_audit.json',
        'results/tables/suite_all_models_methods/method_attack_summary.csv',
    ],
    'no_experiment_execution': True,
    'assets': [file_entry(path) for path in tracked],
}
(G / 'asset_manifest.json').write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
print('PAPER_ASSETS_OK')
