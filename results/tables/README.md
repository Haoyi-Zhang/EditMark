# Tables

This directory contains the repository-tracked release summary tables that are safe to keep in the anonymous review artifact.

- `dataset_statistics/`: tracked active-slice benchmark-definition and release-suite statistics
- `suite_all_models_methods/`: materialized full-suite summary tables for the reviewer-facing canonical matrix; this release surface records the single-host `140/140` run with `failed_count = 0`

These table directories are the canonical home for exact-value leaderboard artifacts, benchmark-definition tables, language-level exact-value tables, run inventories, functional-quality tables, and timing summaries.

For the review artifact, these exact-value tables are the primary evidence surface. `PostEditScore` remains a tracked secondary summary rather than the only result readers should rely on.

The materialized canonical `suite_all_models_methods` table export family includes:

- exact-value leaderboards such as `suite_all_models_methods_method_master_leaderboard.*`, `suite_all_models_methods_method_model_leaderboard.*`, and `suite_all_models_methods_upstream_only_leaderboard.*`
- reviewer-facing functional-quality tables such as `suite_all_models_methods_model_method_functional_quality.*`
- descriptive timing tables such as `timing_summary.*` and `suite_all_models_methods_model_method_timing.*`
- supporting rollups such as `method_summary.*`, `model_summary.*`, `model_method_summary.*`, `method_source_summary.*`, `method_language_summary.*`, `method_attack_summary.*`, and `suite_all_models_methods_run_inventory.*`
- diagnostic exact-value tables such as `per_attack_robustness_breakdown.*`, `core_vs_stress_robustness_summary.*`, `robustness_factor_decomposition.*`, `utility_factor_decomposition.*`, `generalization_axis_breakdown.*`, and `gate_decomposition.*`

Timing semantics stay split on purpose:

- `clean_generation_seconds_per_1k_token` and `watermarked_generation_seconds_per_1k_token` are generation-stage timing surfaces used by the public efficiency metric
- full-pipeline totals such as `attack_hours_total`, `validation_hours_total`, `detection_hours_total`, and `total_example_hours_total` are descriptive systems outputs, not headline-score inputs

The static anonymous artifact keeps table semantics in this directory-level guide and in each export sidecar. A separate machine-readable raw-export schema is intentionally not shipped because the review package exposes materialized summaries rather than the transient aggregation workspace.

For the canonical `suite_all_models_methods` release surface, `suite_all_models_methods_export_identity.json` is the result-backed sidecar that records the matrix identity, scoring contract, required table hashes, execution mode, and the narrowed reviewer-facing figure roster used by the redraw-only figure path.

Rows that carry score fields also expose `score_semantics` so reviewers can distinguish genuine grouped scorecard recomputations from descriptive rollups. In particular, `model_summary.*` is a descriptive mean over model-method rows, while `method_summary.*`, `model_method_summary.*`, `method_source_summary.*`, `method_language_summary.*`, and `method_attack_summary.*` remain grouped scorecard exports. When a descriptive `model_summary.*` row mixes method-level status values, it can export `generalization_status = descriptive_mixed`, `robustness_status = descriptive_mixed`, or `utility_status = descriptive_mixed` to signal that the row is a descriptive rollup rather than a fresh grouped verdict.

The repository does not keep raw per-run reports or transient aggregation outputs in git.
