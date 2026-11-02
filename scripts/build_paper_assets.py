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
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'analysis'
P = ROOT.parent / 'paper' / 'tosem'
G = P / 'generated'
F = P / 'figures'
sys.path.insert(0, str(ROOT))

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
        'path': path.relative_to(ROOT.parent).as_posix(),
        'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
    }


methods = ['KGW', 'SWEET', 'STONE', 'EWD']
from plot_figures import build_figures, figure_data
from check_natural_modules import derive, verify_attribution, verify_statistics_followup
from report_natural_modules import render as render_natural_modules
from plot_natural_modules import draw as plot_natural_modules
from scripts.build_contract_artifacts import build_facts

natural_basis = [
    ROOT / name for name in (
        'data/natural-modules/protocol.json', 'data/natural-modules/inputs.zip',
        'data/released-style/tool.json',
        'results/natural-modules/evidence.zip',
        'results/natural-modules/evidence-members.json',
        'results/natural-modules/registration.json',
        'results/natural-modules/statistics-validation/evidence.zip',
        'results/natural-modules/statistics-validation/evidence-members.json',
        'results/natural-modules/statistics-validation/registration.json',
    )
]

# Reject invalid consumed cells before publishing tables or refreshing figures.
figure_data(ROOT)
# Require the retained activities even if loose files or cached displays exist.
for path in natural_basis:
    if not path.is_file():
        raise ValueError('Missing retained natural evidence: ' + str(path.relative_to(ROOT)))
natural_data = ROOT / 'results/natural-modules'
derive(natural_data)
verify_attribution(natural_data)
verify_statistics_followup(natural_data)
G.mkdir(parents=True, exist_ok=True)
F.mkdir(parents=True, exist_ok=True)
# The standalone paper entry also refreshes both current source selections.
build_facts(len(load('contract_mutations.json')['rows']))
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

# Reuse the standalone vector producer; it reads the same retained audit exports.
generated_figures = build_figures(ROOT)
render_natural_modules(G)
plot_natural_modules(ROOT / 'figures')
for suffix in ('pdf', 'png'):
    path = ROOT / 'figures' / ('natural_modules.' + suffix)
    shutil.copyfile(path, F / path.name)
    generated_figures.append(path)
expected_figures = {f'{name}.{suffix}' for name in
    ('evidence_contract', 'edit_functionality', 'application_coverage', 'natural_modules')
    for suffix in ('pdf', 'png')}
if {path.name for path in generated_figures} != expected_figures:
    raise ValueError('The figure producer did not regenerate every PDF/PNG consumer')


# Remove manuscript assets that are no longer referenced.
for name in ('utility_vs_pass.pdf', 'utility_vs_pass.png',
             'identification_bounds.pdf', 'identification_bounds.png',
             'source_functionality.pdf', 'source_functionality.png'):
    path = F / name
    if path.exists():
        path.unlink()

from render_released_style_table import render as render_released_style
render_released_style(G / 'released_style_rows.tex')

tracked = [
    G / 'release_facts.tex', G / 'mutation_rows.tex', G / 'source_rows.tex',
    G / 'profile_rows.tex', G / 'witness_rows.tex', G / 'released_style_rows.tex',
    G / 'natural_module_values.tex', G / 'natural_module_rows.tex',
    F / 'evidence_contract.pdf', F / 'evidence_contract.png',
    F / 'edit_functionality.pdf', F / 'edit_functionality.png',
    F / 'application_coverage.pdf', F / 'application_coverage.png',
    F / 'natural_modules.pdf', F / 'natural_modules.png',
    ROOT / 'figures/natural_modules.pdf', ROOT / 'figures/natural_modules.png',
    ROOT / 'figures/natural_modules_binding.json', ROOT / 'analysis/release_facts.json',
]
manifest = {
    'basis': [
        'analysis/frozen_audit.json', 'analysis/source_inventory.json',
        'analysis/method_profiles.json', 'analysis/edit_component_audit.json',
        'results/tables/suite_all_models_methods/method_attack_summary.csv',
        'results/released-style/rows.json',
    ] + [path.relative_to(ROOT).as_posix() for path in natural_basis],
    'basis_files': [file_entry(path) for path in natural_basis],
    'no_experiment_execution': True,
    'assets': [file_entry(path) for path in tracked],
}
(A / 'manuscript_assets.json').write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
print('PAPER_ASSETS_OK')
