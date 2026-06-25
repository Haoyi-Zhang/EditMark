# Full-Run Summary Figures

This directory contains the repository-tracked reviewer-facing full-run figure surface for the canonical
`suite_all_models_methods` result of record. This release surface was materialized from the single-host
canonical matrix with `140/140` successful runs and `failed_count = 0`.

The review figure roster is intentionally narrow:

- `suite_all_models_methods_score_decomposition.png`
- `suite_all_models_methods_score_decomposition.csv`
- `suite_all_models_methods_score_decomposition.json`
- `suite_all_models_methods_detection_vs_utility.png`
- `suite_all_models_methods_detection_vs_utility.csv`
- `suite_all_models_methods_detection_vs_utility.json`

The rendered figures intentionally avoid explanatory footnotes and panel titles so they remain readable after insertion
into a paper column. Semantics that would otherwise require small in-figure text are recorded here and in the JSON/CSV
sidecars: `Headline Gen` uses `//` for unsupported raw generalization rendered as the neutral `0.50` headline value, and
`xx` for supported-zero generalization rendered with the release `0.05` headline floor. These are figure-only headline
rendering conventions: raw unsupported values remain null/N.A., supported zero remains visible in exact-value tables and
strict diagnostics, and the figure should not replace the table-first evidence surface. The detection-utility figure is
drawn over the observed value range; exact values are available in the sidecar files.

Exact-value leaderboards, utility-versus-robustness comparisons, sparse generalization views, source/model/language breakdowns, attack breakdowns, and timing-heavy comparisons are table-first artifacts under `results/tables/suite_all_models_methods/`.

Regenerate the canonical full figure/table surface from a restored raw matrix artifact with:

```bash
python scripts/reviewer_workflow.py regenerate --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --figure-dir results/figures/suite_all_models_methods --table-dir results/tables/suite_all_models_methods
```

For a figure-only rerender after the canonical summary tables and the matching export-identity sidecar already exist, run `python scripts/render_materialized_summary_figures.py --table-dir results/tables/suite_all_models_methods --output-dir results/figures/suite_all_models_methods --export-identity results/tables/suite_all_models_methods/suite_all_models_methods_export_identity.json --require-times-new-roman`.
