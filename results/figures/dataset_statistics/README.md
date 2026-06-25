# Dataset Statistics Figures

This directory contains the reviewer-facing dataset statistics figures for the canonical artifact suite.

Use the figures with the following contract:

- `release_slice_composition.png`: empirical release-slice composition for the seven executed source groups
- `evaluation_dimensions_overview.png`: conceptual scorecard overview, not empirical method performance. The radar panel
  uses full metric labels and a schematic structural wheel to make the scorecard hierarchy legible; it is not a radar
  score plot and does not encode additive weights, coefficients, measured values, or method scores. The PL-style
  derivation panel shows metric dependencies: `HeadlineGen` denotes the headline-transformed generalization term, and
  the `Stealth` / `Efficiency` nodes correspond to the conditioned headline-core factors. Exact values remain
  table-first.

For exact counts, the machine-readable tables under `results/tables/dataset_statistics/` remain the primary evidence surface. In particular, `release_slice_language_breakdown.*`, `dataset_task_category_breakdown.*`, and `dataset_family_breakdown.*` are table-first release artifacts rather than standalone figures.

In particular:

- `evaluation_dimensions_overview.png` is a structural explanation figure that should be read together with [`docs/result_interpretation.md`](../../../docs/result_interpretation.md)
- `release_slice_composition.png` is the only empirical dataset figure retained in the default review figure roster
