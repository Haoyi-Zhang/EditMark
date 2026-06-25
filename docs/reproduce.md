# Reproduction Levels

## Level 1: Static Review

Run the integrity and browse commands. This is the expected review path and requires no credentials.

This level is the submitted evidence surface. It checks the frozen source slices, materialized summary tables, rendered figures, claim map, and headline-gap sidecar. The anonymous package provides a rounded-table recomputation check that gives 0.3221 rather than the paper's unrounded 0.3220, plus leave-one-out stability diagnostics over the 20 method-by-generator slices.

## Level 2: Summary Regeneration

Regenerate figures and tables only after restoring the raw matrix tree from a separately supplied review artifact. Do not write regenerated outputs over the shipped canonical summary paths unless the restored matrix identity matches the canonical manifest.

The default anonymous package intentionally omits the raw matrix index. Regeneration is therefore a separate audit path, not the baseline review path.

## Level 3: Fresh Rerun

A fresh full rerun requires a Linux GPU host, pinned model snapshots, upstream baseline availability, and explicit reviewer intent. It is outside the default anonymous review path.
