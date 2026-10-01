# Reproduction guide

## Model-free verification

```bash
make check
```

Expected components:

- evidence-contract unit and exhaustive-enumeration tests;
- included source-inventory arithmetic checks;
- synthetic mutation and certificate-tamper checks;
- JSON Schema and certificate-regeneration checks;
- manuscript citation, label, and source-graph checks.

## Figures and paper

```bash
python -m pip install '.[figures]'
make assets
make paper
```

The generated files have deterministic numerical inputs. PDF metadata can differ across TeX installations, so verification should compare extracted text and rendered layout rather than raw PDF bytes.

## Experimental runtime

The archived runtime is not required for the paper's contract, proofs, or case-study audit. Running it can download models, consume substantial compute, and execute generated programs. It is outside the default verification path.


## Certificate regeneration

```bash
editmark-counts examples/synthetic_cohort_counts.json --out /tmp/counts.json
editmark-certificate /tmp/counts.json --claim positive_decision_change --out /tmp/certs.json
editmark-verify /tmp/certs.json --basis /tmp/counts.json
```

These examples are software fixtures, not experimental evidence.
