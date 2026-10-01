# Reviewer guide

The fastest model-free audit path from this artifact directory, with the sibling `../paper/tosem/` manuscript directory from the full package available, is:

```bash
make check
editmark-counts examples/synthetic_cohort_counts.json --out /tmp/counts.json
editmark-certificate /tmp/counts.json \
  --claim positive_decision_change --out /tmp/certificates.json
editmark-verify /tmp/certificates.json --basis /tmp/counts.json
```

The examples are explicitly synthetic. The workflow checks arithmetic, cohort/rule identities, certificate integrity, and deterministic regeneration; it does not create watermark observations.

## What to inspect

1. `../paper/tosem/main.pdf` for the argument, theory, and read-only artifact case study. A code-only checkout can run `make test`; `make check` additionally requires the manuscript sources.
2. `docs/evidence_contract.md` for field definitions and estimands.
3. `editmark_audit/records.py`, `analysis.py`, `aggregates.py`, `certificate.py`, and `verify.py` for the executable contract.
4. `analysis/frozen_audit.json` for included artifact facts.
5. `analysis/contract_mutations.json` for generated adversarial checker cases.
6. `tests/contract/` for exact, exhaustive, tamper, schema, and file-safety tests.

## What the package does not claim

- It does not rerun language models, watermark embedding, detection, transformations, or generated-program validation.
- It does not infer record-level outcomes from rounded aggregate rates.
- It does not report edited false-positive rate without edited negative controls.
- It does not subtract clean AUROC from a composite robustness index and call the result detector loss.
- It does not treat transformation applicability as proof that source bytes changed.
- It does not treat a certificate hash as trusted execution attestation or authorship proof.

The central technical object is a verifiable claim package: a structured claim, named population, eligibility gate, fixed detector rule, sufficient statistics, exclusion partition, identification result, generated supported statement, and prohibited interpretations.
