# Frozen-evidence audit

Read-only static inventory and arithmetic checks of frozen aggregates; no experiment rerun

Inventory: **1662** source records; canonical executed design: **1605**; excluded overlay: **57**.
MBXP canonical Go records: **0**. Run inventory: **140**.

## What the exported scores mean

D is clean AUROC. R is a composite of a clipped detector-score ratio, an edited-pass-conditioned detection rate, and an unpaired test-pass-rate ratio. **D - R is not a detector-evidence loss.**

EditSup is a clipped pass-rate ratio; U includes similarity and validation availability. Neither is an absolute joint pass probability.

## Data that cannot be recovered from rounded summaries

The joint original/edited test outcomes, actual-change flags, paired before/after decisions, and transformed controls are missing. Their counts and estimates stay null, not zero.

## Reading the exports

`method_profiles` keeps descriptive and master summaries in separate columns. `edit_component_audit` uses pooled per-edit components and does not reconstruct the source-balanced headline. `source_view_reconciliation` accounts for the additional 62 source records removed by the legacy cross-source deduplication design (38 HumanEval-X and 24 MBXP), inferred from exported row counts; it does not certify equivalence of generated outputs.

Every `input_hashes.json` entry is an actual input to this audit. Historical experiment files were not edited. All displayed rates retain the precision and limitations of their source exports.
