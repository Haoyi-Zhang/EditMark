# Post-Edit Provenance Artifact

This repository is the anonymous artifact companion for a double-anonymous software-engineering submission on source-code watermarking after test-passing software edits. It provides the frozen evidence release behind the paper: benchmark code, source slices, tracked summary tables, rendered result figures, and lightweight integrity checks. Optional rerun material is documented for completeness, while the primary review path is inspection of the shipped summaries used by the paper and supplement.

The paper, supplement, and artifact are meant to be read together. The paper makes the central claim about the evaluated open-source methods, the supplement exposes the additional evidence surface, and this repository lets a reviewer check the frozen summaries and navigation path behind both. The repository binds each reported claim to a reviewer-checkable evidence release rather than to an undocumented rerun environment.

For the shortest reviewer path, start with `REVIEWER_GUIDE.md`. It separates a three-minute integrity check, a fifteen-minute claim audit, and a deeper evidence browse, all without credentials or a fresh GPU execution.

## What the Artifact Supports

The submitted paper asks when clean watermark detection can support provenance evidence after software edits that pass the same task-level correctness checks. The artifact exposes the evidence surface behind that question for the evaluated open, locally auditable comparison surface:

- four open runnable watermarking families under a shared benchmark harness, selected because each exposes generation, a declared detector and threshold, negative controls, transformed-code validation, support, and cost under one auditable protocol;
- seven source groups and five model settings in the canonical comparison surface;
- tracked summary exports for 140 completed configurations;
- transformation-conditioned evidence for detection, retained robustness, utility, control behavior, support, and efficiency, including ordinary workflow edits and stress probes such as block shuffling, control-flow flattening, and budgeted adaptive edits;
- reviewer-safe scripts for browsing and checking the shipped evidence without credentials.

The main claim is a post-edit provenance claim. Clean detection is useful, but provenance decisions are made on software after ordinary edits, validation, and review. A post-edit claim therefore states the edit scope, test-admission rule, admitted support surface, false-positive behavior, and cost needed to interpret the detector output. The reported detection and retained-evidence values are unit-scale summaries formed from each method's declared detector and threshold. The headline gap is an engineering proxy break: it measures how much evidence is lost when the object under review is the edited program rather than the fresh generation. The test-admission rule records task-exposed behavior after software change, which is the object available to downstream reviewers; it is not presented as full semantic equivalence.

The repository is also organized as a reusable comparison protocol: a new method is comparable only when it exposes generation, a declared detector and threshold, transformed-code validation, controls, utility, cost, and support under the same reviewer-facing summaries.

## Repository Layout

- `posteditbench/`: benchmark harness, transformations, validation, scoring, and baseline adapters. The package name is retained only so the shipped scripts and imports remain reproducible during anonymous review; it is a submission-local name, not a public package identity.
- `configs/`: canonical comparison manifests and lightweight configuration files.
- `data/release/`: de-identified release-source slices used by the shipped evidence surface.
- `results/tables/`: materialized summary tables used by the manuscript and supplement.
- `results/figures/`: rendered summary figures and their small sidecar data.
- `scripts/`: reviewer browse and integrity checks plus maintained utility scripts.
- `artifact/`: anonymous review notes, claim-to-evidence map, and evidence-protocol statement.
- `third_party/`: upstream baseline provenance and redistribution notes.

## Reviewer Quick Start

The default review path needs no API key, no model cache, and no GPU. Use Python 3.10 or newer; a virtual environment is recommended for a fresh clone:

```bash
python -m pip install -r requirements.txt
python scripts/verify_release_integrity.py
python scripts/reviewer_workflow.py browse --summary-only
```

For a three-minute claim check, use this path:

1. Run `python scripts/verify_release_integrity.py` and confirm `release_integrity=passed`.
2. Run `python scripts/reviewer_workflow.py browse --summary-only` and check the printed gap line: the rounded 20-slice comparison surface gives a mean detection-minus-robustness gap of 0.3221, tracing the paper's unrounded reported gap of 0.3220. Dropping any one method-by-generator slice keeps the mean gap between 0.3093 and 0.3327; slice counts are descriptive support, not separate significance tests. The reviewer-facing check file is `artifact/headline_gap_statistics.*`.
3. Open `artifact/CLAIM_TO_EVIDENCE.md` and follow the row for the detection-to-robustness gap to the named summary tables and figures.

For a fuller local browse, run:

```bash
python scripts/reviewer_workflow.py browse
```

The shipped tables and figures are the primary review surface. They are the paper's frozen evidence layer: the layer a reviewer can inspect quickly and deterministically. A separate full-execution tier is documented for readers who want to rebuild the same surface from pinned model snapshots, baseline upstream checkouts, and a matching Linux GPU host. The script `scripts/audit_anonymous_artifact.py` checks the release package for author cues, private infrastructure, credentials, paper build products, and other review-unsafe files.

The browse command starts from the materialized evidence layer used by the paper: summary tables, rendered figures, sidecar statistics, and the claim map. Raw-matrix rebuilding is a deeper reproduction tier, not the entry point for reviewing the submitted claims.

To prepare a file archive for anonymous review, export only tracked release files:

```bash
python scripts/export_anonymous_review_package.py
```

The exporter writes a zip archive outside the repository root and refuses to include `.git`, bytecode caches, TeX/PDF paper products, private credentials, or ignored local state. This is the intended archive path when the review system expects an upload rather than an anonymous repository URL.

## Reviewer Reading Path

The fastest way to audit the submission package is:

1. Confirm the package integrity with `python scripts/verify_release_integrity.py`. This checks that the shipped release manifest, source slices, and summary surfaces match the recorded evidence release.
2. Inspect `artifact/CLAIM_TO_EVIDENCE.md`. It maps the paper's main claims to concrete result tables, figures, and scripts, and records the evidence surface for each claim.
3. Browse the frozen evidence with `python scripts/reviewer_workflow.py browse --summary-only`. This shows the comparison surface without credentials or a fresh execution.
4. When reading the supplement, use the repository tables as the check surface: source admission is under `results/tables/dataset_statistics/`, method and transformation evidence is under `results/tables/suite_all_models_methods/`, and rendered summaries are under `results/figures/`.

This order mirrors the paper's logic: source admission first, then clean detection, then post-edit retained evidence, then controls, support, utility, and cost. Central claims are the ones that can be followed through this chain.

## Anonymity Boundary

This repository is prepared for double-anonymous review. It omits author names, affiliations, email addresses, acknowledgments, public archival links that identify the authors, private server information, API credentials, model caches, raw matrix archives, and paper build artifacts. Public upstream baseline URLs are retained only when needed to identify third-party methods.

For submission, the repository must be hosted through an anonymous review URL or an anonymous account, and any exported archive must exclude `.git` metadata. Git remotes, fetch logs, commit identities, bytecode caches, and hosting owners are outside the reviewer-facing archive; if they identify the authors, the package should be considered not submission-ready even when the working tree itself is anonymized. Use `scripts/export_anonymous_review_package.py` for a tracked-only archive rather than compressing the working directory by hand.

## Reading the Result

The artifact should be read as a post-edit evidence surface, not as a single leaderboard. If a method has high clean detection but weak retained evidence after test-passing edits, the artifact changes the provenance sentence from fresh-output detection to workflow-conditioned evidence. If a method retains evidence but has visible cost or control conditions, those dimensions become part of the operational provenance claim.

This is the distinction the paper asks reviewers to judge: the contribution is a workflow-conditioned post-edit robustness reporting protocol for provenance use, instantiated as a benchmark showing where fresh-output detector evidence breaks as provenance evidence under test-passing software edits. Robustness, controls, support, utility, and cost are the dimensions that make that break auditable.
