# Dataset And Release Surface

This artifact exposes the release surface used by the submitted paper and supplementary material. It is intended for anonymous review of the frozen evidence, not for collecting new benchmark tasks during review.

## Source Groups

The release layer combines four public executable benchmark slices with three crafted families:

- public programming-task slices with executable task-level checks;
- multilingual public slices where validation can be run through the configured language backend;
- crafted original tasks designed to cover algorithmic, data-structure, state-machine, parsing, and simulation behavior;
- crafted translation and stress families used to test whether the same evidence protocol remains interpretable across source variation.

Together these groups form the comparison surface described in the paper: seven source groups, five model settings, four runnable watermarking baselines, and 140 completed canonical configurations.

## Admission Boundary

A record enters the release surface only when it has a normalized task statement, a target language, an executable validation route, and a task-level correctness check. The release records task-exposed behavior after software change and keeps stronger equivalence or coverage analyses outside the retained-evidence estimate rather than mixing them into the provenance score.

## Where To Inspect

- `data/release/sources/` contains the normalized release-source records.
- `results/tables/dataset_statistics/` contains reviewer-facing source counts and language coverage summaries.
- `artifact/CLAIM_TO_EVIDENCE.md` maps the paper claims to the relevant source, support, control, and result tables.

The release surface is deliberately static. Reviewers can inspect the shipped records and tables without credentials, model caches, or a GPU.
