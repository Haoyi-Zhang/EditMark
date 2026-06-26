# Reproduction Levels

## Level 1: Static Review

Run the integrity and browse commands. This is the expected review path and requires no credentials.

This level is the submitted evidence surface. It checks the frozen source slices, materialized summary tables, rendered figures, claim map, and headline-gap sidecar. The anonymous package provides a rounded-table recomputation check that gives 0.3221 rather than the paper's unrounded 0.3220, plus leave-one-out stability diagnostics over the 20 method-by-generator slices. It also exposes the admission funnel, strict raw utility, support, and control summaries used to interpret the post-edit estimand.

Level 1 is sufficient for checking the paper-level numerical claims. It is not a fresh-execution claim: the submitted object is a frozen evidence release with integrity checks, traceability tables, and reviewer-browse scripts.

## Level 2: Summary Regeneration

Regenerate figures and tables only after restoring the raw matrix tree from an archival raw-run store. Do not write regenerated outputs over the shipped canonical summary paths unless the restored matrix identity matches the canonical manifest.

The default anonymous package intentionally omits the raw matrix index because the review package is the frozen summary evidence used by the paper and supplement. Regeneration is therefore a separate audit path, not the baseline review path.

## Level 3: Fresh Rerun

A fresh full rerun requires a Linux GPU host, pinned model snapshots, upstream baseline availability, and explicit reviewer intent. It is outside the default anonymous review path.
