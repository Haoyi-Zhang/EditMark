# EditMark

**EditMark is a machine-checkable evidence contract for evaluating code watermarks after software edits.** It separates four questions that are often collapsed into one score:

1. Did the transformation actually change the program?
2. Did the original and edited programs pass the same task contract?
3. Were both detector decisions produced by the same rule?
4. Which claim is identified by the available positive and negative cohorts?

This code repository contains the archived benchmark implementation and summaries used as a case study, and a model-free analysis package. The full distribution places the manuscript in the sibling `../paper/tosem/` directory. The default workflow does **not** download models, generate code, run watermark detectors, or execute generated programs.

## One-minute verification

Python 3.10 or later is sufficient for the core checks. With this directory as the working directory, the code-only test entrypoint is:

```bash
make test
```

The full package checks also require the sibling `../paper/tosem/` sources:

```bash
make check
```

This command runs the evidence-contract tests, rebuilds the machine-readable facts from included files, and checks the active manuscript source graph. It does not invoke the experimental runtime.

The current Ubuntu 24.04 / Python 3.12 run passed all 203 contract test methods. Its passive inventory reconstruction recovered 1,662 released records, 1,605 configured records, 57 exclusions, five model settings, and four methods. Raw outputs are retained in `results/current/`. This run did not generate programs or rerun models or watermark detectors; it does not supply the missing joint program observations described below.

`summarize` retains admitted immutable pairs during its existing eligibility
traversal and sorts that invocation-local list for the cohort fingerprint.
Public exclusion precedence and all three selection modes remain unchanged.
Four supplementary tests in `regressions/` independently specify eligibility
truth tables, enumerate missing-decision completions, and check exact hash
payloads and certificate round trips. `make test` and the scientific workflow
run them as a separate mandatory stage. The retained 203-test transcript and
source-derived contract test count are unchanged; the four supplementary
regressions are not retroactively attributed to that archived run. This is a
bounded implementation correctness change, not measured speedup or new
watermark observations.

To rebuild the figure and paper:

```bash
python -m pip install '.[figures]'
make assets
make paper
```

## Three analysis interfaces and one verifier

### 1. Pair observations

```bash
python -m editmark_audit examples/synthetic_pairs.jsonl \
  --out /tmp/editmark-pair-report.json
```

The parser rejects inconsistent hashes, validators, detector rules, duplicate identities, and malformed numeric values. The strict estimand is the after-minus-before positive-decision rate on programs that are supported, genuinely changed, pass the same task contract on both sides, and use the same detector rule.

### 2. Exact sufficient counts

```bash
python -m editmark_audit.aggregates examples/synthetic_cohort_counts.json \
  --out /tmp/editmark-count-report.json
```

Exact integer counts can identify net decision change without releasing source text. A joint transition table identifies survival; marginals alone produce sharp finite-cohort bounds. Rounded percentages and broad composite scores are rejected.

### 3. Claim certificates

```bash
python -m editmark_audit.certificate /tmp/editmark-count-report.json \
  --claim positive_decision_change \
  --out /tmp/editmark-certificate.json
```

The certificate records whether a requested sentence is identified, only bounded, or not admissible. Positive pairs can support same-rule decision change. They cannot establish edited false-positive rate, AUROC change, calibration, source prevalence, or posterior provenance probability without additional evidence.

### 4. Certificate verification

```bash
editmark-verify /tmp/editmark-certificate.json \
  --basis /tmp/editmark-count-report.json \
  --out /tmp/editmark-verification.json
```

Standalone verification checks the canonical certificate hash and re-derives every semantic field from embedded sufficient facts. With `--basis`, the verifier also checks the source-file digest and regenerates the complete certificate bundle. It does not treat a cohort fingerprint as proof that an experiment was honestly executed.

The verifier is a separate entrypoint but shares the analyzer and certifier code; it is not an independent arithmetic implementation. A pair report is supplied evidence, not raw-record membership authentication, so both certificate routes retain `membership_verified=false` and `cohort_membership_authenticated=false`. Count-report re-derivation distinguishes JSON Booleans from exact integer fields, and verification outputs use the same exclusive, non-overwriting publication routine as the other reports.

All files under `examples/` are explicitly synthetic software fixtures, not experimental observations.

## Repository map

| Path | Purpose |
|---|---|
| `../paper/tosem/` (full package) | ACM journal manuscript, bibliography, figures, and generated table fragments |
| `editmark_audit/` | Typed pair parser, eligibility gate, exact-count analysis, certificates, and independent verification |
| `schemas/` | Local JSON Schemas for pairs, sufficient counts, and certificates |
| `tests/contract/` | Model-free unit, exhaustive-enumeration, mutation, and file-safety tests |
| `analysis/` | Frozen case-study facts, exact figure inputs, exclusions, controls, and checker outputs |
| `data/`, `configs/`, `results/` | Included benchmark sources, configuration surface, and archived aggregate evidence |
| `posteditbench/` | Experimental pipeline retained for inspectability; not invoked by `make check` |
| `docs/` | Evidence contract, interpretation, manuscript format, safety, authorship, and reviewer guidance |

## Evidence boundary

The included aggregate evidence supports an audit of measurement and reporting. It does not contain the joint real-program observations needed to estimate a corrected post-edit detector effect. The paper therefore does not publish a replacement method ranking or fabricate missing records. It contributes a reusable contract, formal identification results, claim certificates, and an executable checker that make future post-edit claims auditable.

## Authors

- Haoyi Zhang, Xi'an Jiaotong-Liverpool University — `hyeliozhang@gmail.com`
- Huaijin Ran, Nanyang Technological University — `huaijin003@e.ntu.edu.sg`
- Xunzhu Tang, University of Luxembourg — `realdanieltang@gmail.com`


## Safety

The evidence-contract tools treat source code as text and never execute it. The experimental pipeline can execute generated programs and is not a security sandbox. Read `docs/runtime_safety.md` before using any runtime command.
