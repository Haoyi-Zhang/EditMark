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
