# Owned finite-carrier paired measurement

This opt-in integration fixture is authored, not independently sourced
CodeMarkBench evidence. Its eight three-bit carrier states, detector threshold
`score >= 2`, neutral review footer and five-input `x + 1` contract were fixed
before execution. Both source versions are retained in `pairs.jsonl`.
The footer leaves the carrier and executable body intact; no attack generator,
upstream watermark, model, public task or external target is invoked.

The actual CPython run executes 16 exact approved source texts and 80
finite-domain calls. `measurement.json` retains every call, score, decision,
actual environment and source binding. These are native Python fixture calls,
not native validation of a real watermark benchmark, C programs or general
semantic equivalence. The core evidence kind remains `synthetic_test`.

All eight pairs are admitted; `n00=4,n01=0,n10=0,n11=4`. Net change is zero,
joint survival is one, and ordinary direct arithmetic on the same complete
inputs gives exactly the same answers. Removing transitions leaves
`(n,b,a)=(8,4,4)`: net change stays zero and survival is bounded by `[0,1]`.
The full joint result is not inferred from those marginals. Selection/detection
gain is zero. Edited FPR, AUROC change and posterior provenance remain denied.

The directory retains raw pairs, pair report, joint/marginal disclosures,
all five claim bundles and their raw-basis verification outputs. Raw-basis
rederivation checks the supplied source hashes, duplicate identities,
eligibility and transitions; it does not authenticate execution or external
origin. `membership_verified` remains false in every claim.

`measurement.json` and its `source_bindings` remain the original measured
record, not a map of the current approved source. The exact measured versions
of `scripts/measure_owned_carrier.py`, `editmark_audit/aggregates.py`,
`editmark_audit/certificate.py` and `editmark_audit/verify.py` are preserved under
`source-preimage/` at those same relative paths. They are DATA ONLY: do not
import or execute them. The other three bindings resolve to exact live bytes.
No raw measurements, environment, source map or expected digest was rewritten,
and no rerun or independent deployment is claimed.

The existing basis verifier checks the original measurement's raw SHA-256 and
every recorded source path/hash when receiving this retained cohort. Each must
resolve to exact live bytes or the corresponding exact preimage; missing or
tampered bytes fail closed, without line-ending normalization. For saved-data
verification only, from the artifact root:

```bash
python -B -m editmark_audit.verify results/owned_finite_carrier/positive_decision_change.json --basis results/owned_finite_carrier/pairs.jsonl
python -B -m unittest tests.contract.test_report_consistency -v
```

These commands never import the fixture, measurement runner or preimages, and
do not authenticate execution or external origin. The following commands use
the current runner, not the historical measured producer. From the artifact
root, use a new output directory:

```bash
python -B scripts/measure_owned_carrier.py --out NEW_DIRECTORY
python -B -m unittest discover -s regressions -p check_owned_carrier.py -v
```

Native fixture execution is opt-in and is not part of the default model-free
test discovery. Cross-platform runtime metadata and line endings can differ;
the raw JSONL is explicitly LF and certificate hashes use canonical JSON.
Historical benchmark files and the archived diagnostic/timing outputs are
unchanged. This fixture does not close the independent-study evidence gap.
