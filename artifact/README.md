# Anonymous Artifact Notes

This artifact supports anonymous review of a benchmark for post-edit source-code watermark provenance. It is designed for static inspection and lightweight consistency checks.

The artifact is aligned with the submitted paper and supplement as follows. The paper states the claim being tested: clean detector evidence is not automatically provenance evidence after test-passing software edits. The supplement gives the fuller evidence surface. This repository gives the concrete, de-identified objects used to audit that surface.

## Included

- benchmark harness and scoring code;
- canonical configuration manifests;
- de-identified release source slices;
- tracked result tables and figures;
- third-party baseline provenance notes;
- lightweight integrity and browse scripts.

## Not Included

- author names, emails, affiliations, acknowledgments, or public self-identifying archival links;
- private credentials, server addresses, API keys, SSH material, or cloud-specific configuration;
- paper source, submitted PDFs, review logs, status ledgers, or local build products;
- model weights, model caches, raw full-matrix archives, or GPU rerun state.

## Review Interpretation

The artifact checks whether the paper's claims follow from the frozen comparison surface. Its supported question is reviewable and workflow-conditioned: whether clean detector-and-threshold evidence remains admissible as provenance evidence after edits that pass the same task-level correctness checks, under visible support, control, utility, and efficiency conditions.

The artifact should therefore be read as evidence for the reporting protocol, not as a hidden extension of the paper. A reviewer can inspect whether each post-edit provenance claim names its edit scope, admission rule, controls, support, utility, and cost before accepting detector output as origin evidence.

For review, start with `CLAIM_TO_EVIDENCE.md`, then run the integrity check from the repository root, then inspect the result tables named by the claim map. The intended reading is conditional: a table supports only the claim whose source group, transformation family, control behavior, support, utility, and cost are visible in the shipped evidence.
