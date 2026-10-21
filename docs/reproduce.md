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

`paper` requires `assets`, including seven optional finite figure-consumer tests.
All quantitative PDFs and PNGs are rebuilt together from the included CSV/audit
exports. The build fails on missing dependencies, missing/duplicate cells,
non-finite or out-of-range consumed rates and coverage/count contradictions.
No cached preview substitutes for a missing or invalid input.

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

For raw-pair source-basis regeneration, use fresh outputs:

```bash
python -B -m editmark_audit.certificate examples/synthetic_count_basis.jsonl --claim detection_survival --out NEW_CERTIFICATE.json
python -B -m editmark_audit.verify NEW_CERTIFICATE.json --basis examples/synthetic_count_basis.jsonl --out NEW_VERIFICATION.json
```

The separate opt-in command `scripts/measure_owned_carrier.py --out NEW_DIRECTORY`
executes only its exact owner-authored finite Python source set. Its six current tests
use `python -B -m unittest discover -s regressions -p check_owned_carrier.py -v`.
The retained raw measurement and source/environment bindings are in
`results/owned_finite_carrier/`. Neither this fixture nor direct arithmetic on
its pairs supplies the missing independent benchmark observations.
