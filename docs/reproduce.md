# Reproduction Levels

## Level 1: Static Review

Run the integrity and browse commands. This is the expected review path and requires no credentials.

This level is the submitted evidence surface. It checks the frozen source slices, materialized summary tables, rendered figures, claim map, and headline-gap sidecar. The anonymous package provides a rounded-table recomputation check that gives 0.3221 rather than the paper's unrounded 0.3220, records the paper's descriptive bootstrap range of 0.2619--0.3836, and exposes leave-one-out stability diagnostics over the 20 method-by-generator slices. It also exposes the admission funnel, strict raw utility, support, and control summaries used to interpret the post-edit estimand.

Level 1 is sufficient for checking the paper-level numerical claims. It is not a fresh-execution claim: the submitted object is a frozen evidence release with integrity checks, traceability tables, and reviewer-browse scripts.

## Level 2: Summary Regeneration

Regenerate figures and tables only when working from a matrix tree whose identity matches the canonical manifest. Do not write regenerated outputs over the shipped canonical summary paths unless that identity check passes.

The submitted package is organized around the materialized evidence layer used by the paper and supplement: summary tables, rendered figures, sidecar statistics, integrity metadata, and claim-to-evidence mappings. Regeneration is a separate audit path for rebuilding those surfaces from a verified raw matrix tree.

## Level 3: Fresh Rerun

A fresh full rerun is the deepest reproduction tier. It requires a Linux GPU host, pinned model snapshots, upstream baseline availability, and explicit reviewer intent.
