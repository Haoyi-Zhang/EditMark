# EditMark

**EditMark is a machine-checkable evidence contract for evaluating code watermarks after software edits.** It separates four questions that are often collapsed into one score:

1. Did the transformation actually change the program?
2. Did the original and edited programs pass the same task contract?
3. Were both detector decisions produced by the same rule?
4. Which claim is identified by the available positive and negative cohorts?

This code repository contains the archived benchmark implementation and summaries used as a case study, and a model-free analysis package. The full distribution places the manuscript in the sibling `../paper/tosem/` directory. The default workflow does **not** download models, generate code, run watermark detectors, or execute generated programs.

## One-minute verification

Python 3.10 or later is sufficient for the core checks; no model/runtime
dependencies or package installation are needed when running from this
directory. The portable code-only test commands are:

```bash
python -B -m unittest discover -s tests/contract -v
python -B -m unittest discover -s regressions -v
```

With GNU Make and a POSIX shell, `make test` runs the same two suites.
The full package checks also require those tools and the sibling
`../paper/tosem/` sources:

```bash
make check
```

This command runs the evidence-contract tests, rebuilds the machine-readable facts from included files, and checks the active manuscript source graph. It does not invoke the experimental runtime.

The retained Ubuntu 24.04 / Python 3.12 run passed all 203 contract test methods. Its passive inventory reconstruction recovered 1,662 released records, 1,605 configured records, 57 exclusions, five model settings, and four methods. Raw outputs are retained in `results/current/`. This run did not generate programs or rerun models or watermark detectors; it does not supply the missing joint program observations described below.

The current model-free suite runs 230 contract methods and 23 supplementary
methods separately. It checks eligibility truth tables, missing-decision
completions, typed certificate commitments, source-basis consistency, exact
fingerprints and exclusive publication. The archived 203-method Ubuntu
execution remains separate. The opt-in finite source fixture has six regression
methods, including admission and direct/analyzer agreement gates that remain
active under optimized Python. Its programs are restricted to the exact
owner-authored carrier set and are not executed by default model-free test
discovery.

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

Pair exports retain missing-only strata as explicit non-admissible coverage
records, without an admitted-cohort hash, point estimate or metric denominator.
Bundles containing them use the v2 schema; complete-only and exact-count bundles
keep the original schema. See `docs/claim_certificates.md` for the record contract.

### 4. Certificate verification

```bash
python -m editmark_audit.verify /tmp/editmark-certificate.json \
  --basis /tmp/editmark-count-report.json \
  --out /tmp/editmark-verification.json
```

For use outside this directory, install the core package in an activated
environment with `python -m pip install .`; this also provides the
`editmark-pairs`, `editmark-counts`, `editmark-certificate` and `editmark-verify`
console commands. Report output paths must be new; existing files are never
overwritten.

Standalone verification checks the canonical certificate hash and re-derives every semantic field from embedded sufficient facts. With `--basis`, it checks the source-file digest, regenerates the complete certificate bundle and rechecks that digest. A basis changed during regeneration is rejected. Normalized raw pairs can be supplied directly:

```bash
python -B -m editmark_audit.certificate examples/synthetic_count_basis.jsonl --claim detection_survival --out NEW_CERTIFICATE.json
python -B -m editmark_audit.verify NEW_CERTIFICATE.json --basis examples/synthetic_count_basis.jsonl --out NEW_VERIFICATION.json
```

The verifier is a separate entrypoint but shares the analyzer and certifier code; it is not an independent arithmetic implementation. Raw-pair basis mode reparses source identities, conflicting duplicates, eligibility and transitions and reports `raw_pair_membership_rederived=true`. A supplied pair report does not reconstruct raw membership. Neither mode authenticates external origin or execution, so certificates retain `membership_verified=false` and `cohort_membership_authenticated=false`. Typed canonical commitments distinguish Booleans from exact integer fields; verification uses the same exclusive publication routine as other reports.

V2 basis verification requires every stratum record. Legacy v1 cohort bundles
remain verifiable, but omitted strata are returned as `unidentified_strata` with
`basis_coverage_complete=false`; basis agreement alone is not full coverage.

All files under `examples/` are explicitly synthetic software fixtures, not experimental observations.

## Bounded same-information diagnostic

`scripts/diagnose_information.py` inventories retained JSON/JSONL/CSV evidence,
independently redoes ordinary exact-count and pair arithmetic, and keeps
missing-only target cohorts explicit. It neither executes source programs nor
runs watermarking, detector inference, or transformations. Run from this directory:

```bash
python -B scripts/diagnose_information.py --out results/repair_diagnostic/new-information.json
python -B -m unittest discover -s regressions -v
```

The output path must be new and within `results/repair_diagnostic/`. The retained
`information-v3.json` records nine synthetic pair cases, the exact-count
comparison, all 28 rounded component checks, 142 serialized-file inventory
entries, and local processing costs. The costs include a separately measured
fresh-process/durable-publication workflow, not just arithmetic time. Direct
arithmetic and the contract receive the same disclosed information but provide
different guarantees; this is not a speedup or new detector-performance result.
The archived non-finite EWD summary literal is flagged and preserved, not turned
into a measured zero. No real paired effect is recoverable from this inventory.

Bundle verification enforces the declared bundle fields and a nonempty
input name plus a lowercase 64-hex input digest. A syntactically valid digest
still does not authenticate the basis in standalone mode.

## Opt-in owned source-to-certificate measurement

`fixtures/owned_finite_carrier.py` declares all eight three-bit comment states,
one fixed threshold and an `x + 1` contract on five integer inputs. A neutral
review footer preserves the carrier. The runner accepts only 16 exact approved
source versions, not arbitrary programs or benchmark task code:

```bash
python -B scripts/measure_owned_carrier.py --out NEW_DIRECTORY
python -B -m unittest discover -s regressions -p check_owned_carrier.py -v
```

The actual run and raw inputs are in `results/owned_finite_carrier/`: eight
pairs, 80 CPython fixture calls, transitions (4,0,0,4), net change 0 and joint
survival 1. Same-cohort marginals alone yield survival [0,1]. Direct arithmetic
on the same paired inputs agrees exactly; selection/detection gain is zero.
All five bundles are verified against the raw JSONL, with FPR/AUROC/posterior
requests still denied. Origin stays `synthetic_test`; this is an authored
native fixture integration check, not independently sourced study evidence or
validation of KGW/SWEET/STONE/EWD. Default model-free commands do not execute it.

No independent human review or empirical novelty is inferred from these checks.

## Repository map

| Path | Purpose |
|---|---|
| `../paper/tosem/` (full package) | ACM journal manuscript, bibliography, figures, and generated table fragments |
| `editmark_audit/` | Typed pair parser, eligibility gate, exact-count analysis, certificates, and separate verification with shared arithmetic |
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

## Portable path notation

Published execution records use `<PYTHON_ROOT>`, `<NODE_ROOT>`, `<RUNTIME_ROOT>` and `<PROJECTS_ROOT>` for machine-local installation and workspace prefixes. These are path aliases, not literal directories; substitute the corresponding local roots when replaying a command. Recorded interpreter and dependency versions, inputs, measurements and outcomes are unchanged. File inventories bind the published portable serializations; any `recorded_` identity denotes the pre-normalization bytes.
