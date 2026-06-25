# Dataset Statistics Tables

This directory contains repository-tracked active-slice dataset tables.

They are generated with:

```bash
python scripts/export_dataset_statistics.py
```

The JSON and CSV files summarize:

- the active canonical release-slice execution
- the benchmark-definition table
- the language-level release breakdown and manifest index, including reference-kind counts where a release slice uses smoke-overlay support
- crafted-only category and family coverage views

The canonical benchmark-definition files are:

- `benchmark_definition_summary.csv`
- `benchmark_definition_summary.json`

They are the primary disambiguation table for the canonical release suite: `HumanEval-X` and `MBXP-5lang` are presented here only as the active five-language balanced execution slice.

`release_source_manifest_index.{csv,json}` is the complementary audit table for reference support. Its `executed_release_count` column is the reviewer-facing release size; the `canonical_reference_count` and `smoke_overlay_reference_count` columns disclose how that release size is supported at the reference level for multilingual execution.
