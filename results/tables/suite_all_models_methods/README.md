# Full-Run Summary Tables

This directory contains the repository-tracked full-run summary tables for the canonical
`suite_all_models_methods` result of record. This release surface was materialized from the single-host
canonical matrix with `140/140` successful runs, `failed_count = 0`, and
`execution_mode = single_host_canonical`.

For review, these tracked tables are the evidence surface used by the paper and supplement. A fresh full rerun is optional and requires the pinned model snapshots, compatible upstream baselines, and a Linux GPU environment; it is not part of the default anonymous review path.

Regenerate tables from a restored raw matrix artifact with:

```bash
python scripts/export_full_run_tables.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --output-dir results/tables/suite_all_models_methods
```

The raw matrix index is intentionally omitted from the static anonymous package. The command above applies only after a separate raw rerun artifact has been restored; a reviewer browse message saying the matrix index is not materialized is expected for this package.

The result-backed canonical table surface includes:

- method-level and method-by-model leaderboards, including `suite_all_models_methods_method_master_leaderboard.*`, `suite_all_models_methods_method_model_leaderboard.*`, and `suite_all_models_methods_upstream_only_leaderboard.*`
- language-level, source-level, and attack-level rollups
- functional-quality and timing exports
- table-first tradeoff surfaces such as `suite_all_models_methods_utility_robustness_summary.*`
- run inventory and export-identity sidecars
- diagnostic exact-value tables such as `per_attack_robustness_breakdown.*`, `core_vs_stress_robustness_summary.*`, `robustness_factor_decomposition.*`, `utility_factor_decomposition.*`, `generalization_axis_breakdown.*`, and `gate_decomposition.*`

These tables are the primary release-facing evidence surface. The tracked figures remain a narrower structure-oriented public surface.

For reviewer interpretation, see [`../../../docs/result_interpretation.md`](../../../docs/result_interpretation.md). In particular, low robustness is a benchmark finding rather than a run failure; strict zero diagnostics such as `raw_generalization_strict` and `raw_composite_strict` are intentionally fail-closed; and constant support fields such as `utility_support_rate` document coverage rather than perfect performance.
