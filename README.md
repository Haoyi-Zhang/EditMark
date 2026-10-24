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

The current model-free suite runs 247 contract methods and 23 supplementary
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

`make paper` depends on `assets`: it rebuilds the real-data vector PDFs and PNG
previews and runs seven finite plotting-consumer tests. `make assets` can also be
used on its own. Missing plotting dependencies or malformed consumed data fail
nonzero; no old PNG is a successful fallback. The default `make check` remains
model-free and does not require plotting dependencies.

The full packet's `paper/tosem/Makefile` also regenerates all three actual
PDF/PNG figure consumers and table fragments before TeX, through the same
`build_paper_assets.py` entrypoint. Its standalone paper target does not refresh
only the contract diagram while accepting cached data plots.

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

Every count, certificate and basis-regeneration route uses one batch check.
Within a cohort/stratum, the rule, counts, optional transitions and exclusion
partition must agree. Identical normalized facts contribute once (idempotently);
conflicts or mixed synthetic/saved kinds fail before publication. Distinct
cohorts remain separate and are never pooled.

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
`information.json` records nine synthetic pair cases, the exact-count
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

## Opt-in existing SemCodebook CPU measurement

The source-pinned, author-owned SemCodebook case runs the existing
`rule_based_generate` encoder and unchanged native detector. This is the
bootstrap rule-based operating mode, not model inference, blind schedule
reconstruction, an independent study, or a claim of publication.

```bash
python -B scripts/measure_semcodebook_cpu.py --out NEW_DIRECTORY
python -B -m unittest discover -s regressions -p check_semcodebook_cpu.py -v
```

The retained result is `results/semcodebook_cpu/`. All 16 four-bit payload
attempts are kept, including misses and wrong recovered values. There is one
authored arithmetic task, four distinct before texts, nine distinct texts
overall, and two source-authored edits: a comment footer and an `any`/`sum`
refactor. Every one of the 32 pairs passes the same exhaustive 156-input finite
contract (9,984 function calls, zero failures). The population is payload
configurations, not independent programs.

The original generation schedule is registered by sample ID and held fixed on
both sides. The rule ID binds that entire registry, the upstream commit, key,
native threshold 0.5 and adapter bytes. The recorded Boolean is the native
`is_watermarked`; its confidence is not substituted for a scalar decision
score. Abstain/reject are observed false for that Boolean, with all native
three-state outputs retained. A hit does not imply correct payload recovery:
payload 13 remains a hit after refactoring but recovers 12.

The comment transitions are (1,0,0,15), change 0 and survival 1.
The refactor transitions are (1,0,8,7), change -1/2 and survival 7/15;
marginals alone bound the latter survival by [2/5,7/15]. Separately written
direct arithmetic on the same rows agrees exactly. All five claim bundles
are re-derived against source-linked raw JSONL. FPR, AUROC and posterior
requests remain denied; there is no selection or detection gain from EditMark.

Only five reviewed arithmetic texts and four fixed comment variants may be
executed. Upstream Git blob identities are checked before measurement; encoder
replay must reproduce the frozen generations before their code is accepted.
The vendor provenance file identifies commit
`f2434164a4bd022099d98e5fa6236cd04e01968a` and every included original file.
No LICENSE/COPYING file was present in that source tree; no new license or
independent authorship is asserted for the included owner-provided source.
Training, model loading, attack workflows and public benchmark execution are
not invoked. Default model-free discovery does not execute this opt-in case.

## Hand-worked arithmetic examples

Run `python -B scripts/check_hand_calculations.py` for four short authored
Boolean lists with predeclared exact answers. The examples cover negative and
positive net change, a missing after-decision, and zero initial positives.
Expected arithmetic is separate from the production analyzer and verifier.
The retained output is `analysis/hand_calculations.json`; Appendix A consumes
the same explicit lists and fractions. These are synthetic arithmetic examples,
not independent watermark experiments. Successful verification establishes
consistency of structured statements with the supplied facts, not source truth,
execution authenticity, or arbitrary prose.

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

The archived aggregate evidence supports an audit of measurement and reporting, but not a corrected four-method paired effect. The separate SemCodebook CPU case supplies actual source-linked observations from an existing author-owned implementation under a finite controlled task contract. It does not supply independent validation, a replacement method ranking, or missing observations for the original summary population. The contribution is a reusable contract, identification results, claim certificates, and an executable checker for auditable post-edit claims.

## Authors

- Haoyi Zhang, Xi'an Jiaotong-Liverpool University — `hyeliozhang@gmail.com`
- Huaijin Ran, Nanyang Technological University — `huaijin003@e.ntu.edu.sg`
- Xunzhu Tang, University of Luxembourg — `realdanieltang@gmail.com`
- Kisub Kim, Daegu Gyeongbuk Institute of Science and Technology — `kisub.kim@dgist.ac.kr`
- Tegawendé F. Bissyandé, University of Luxembourg — `tegawende.bissyande@uni.lu`

## Safety

The evidence-contract tools treat source code as text and never execute it. The experimental pipeline can execute generated programs and is not a security sandbox. Read `docs/runtime_safety.md` before using any runtime command.

## Portable path notation

Published execution records use `<PYTHON_ROOT>`, `<NODE_ROOT>` and `<RUNTIME_ROOT>` for local tool installations, `<DEPENDENCIES_ROOT>` for local dependencies, and `<PROJECTS_ROOT>`, `<ARTIFACT_ROOT>`, `<SOURCE_ROOT>` and `<WORK_ROOT>` for the workspace, artifact checkout, source checkout and temporary work directory of the recorded run. These are path aliases, not literal directories or claims that historical temporary files are included in this bundle. Substitute the corresponding local roots when replaying a command. Recorded tool versions, source/input identities, measurements, outcomes and failures are unchanged. Current file inventories bind the portable published bytes; any `recorded_` identity denotes pre-normalization bytes.
