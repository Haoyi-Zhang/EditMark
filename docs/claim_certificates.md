# Claim certificates

A claim certificate is a canonical JSON document for a structured claim request. It records:

- the requested claim type;
- the named cohort and its SHA-256 fingerprint;
- the validation and detector rule;
- exact admitted, transition, and excluded counts;
- the estimand and its point value or sharp interval;
- satisfied and missing requirements;
- a generated supported statement and prohibited interpretations;
- a canonical certificate digest.

The checker returns `admissible`, `partially_admissible`, or `not_admissible`. Free-form `claim_text` is metadata and is never treated as verified prose; the generated `supported_statement` is authoritative.

`editmark-verify` has two modes. Standalone mode checks the canonical digest and deterministically re-derives every semantic field from the embedded sufficient facts, using the same analyzer/certifier implementation. Basis mode checks and rechecks the source-file digest and regenerates from normalized raw pairs, a pair report or exact counts. Raw pairs rederive membership under the recorded gate and are distinguished from a supplied report; neither authenticates the flags, execution or external origin. A coordinated change to both basis and all dependent digests still requires an external trust anchor, such as a signed release or trusted execution record.

## Strata without complete decision pairs

Pair/raw-pair exports retain every analyzed stratum. A stratum with no admitted
complete pair is an `editmark-unidentified-stratum` record, not an empty exact-count
cohort. It has `decision=not_admissible`, `estimate=null`, `cohort_sha256=null`, and
no `counts`, transition table, bounds, or metric denominator. Actual attempts,
first-exclusion counts, and the number/identity of known eligible targets remain
explicit. Missing decisions yield `unidentified_missing_decisions`; a stratum
with no established eligible target yields `undefined_no_eligible_target`.
An unknown validation/rule does not become a failed decision. A known-target
fingerprint is never substituted for an admitted-cohort fingerprint.

Bundles containing these records use `editmark-claim-certificate-bundle-v2`.
Complete-only/count exports keep the original bundle and certificate schemas
and deterministic content. The current verifier accepts both bundle versions;
the original bundle version cannot contain the new record type. Consumers must
dispatch on `schema_version` rather than assume every item has admitted counts.
Both bundle variants and the new record have shipped structural schemas.

Standalone verification checks the embedded coverage partition and regenerates
the denial. Basis verification additionally regenerates every stratum, so deleting
a missing-only record from a v2 bundle fails basis verification. A v1 bundle is
still checked against its original nonempty-cohort export contract. If its basis
also contains omitted strata, the verification output retains their full
unidentified records in `unidentified_strata` and sets `basis_coverage_complete=false`;
the CLI also prints `stratum_coverage=incomplete`. `basis_verified=true` still
means the supplied cohort certificates match their basis, not that v1 covered all
strata. New full-coverage consumers must check `basis_coverage_complete` or require
v2. Neither a schema downgrade nor an old omission silently yields complete
coverage. No historical observation is changed.
Certificate export returns exit status 3 when a claim is denied, including a
missing-only bundle; verification returns 0 for a valid denial record, not for an
admissible empirical claim. `--require-observations` still rejects synthetic input.

`from_pair_stratum` remains a nonempty count-export boundary and still refuses
an empty admitted cohort. The analyzer's existing Boolean/null completion bounds
are unchanged and remain in the pair report; this coverage record neither invents
a point estimate nor replaces those bounds with zero.
