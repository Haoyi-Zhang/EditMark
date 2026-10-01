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

`editmark-verify` has two modes. Standalone mode checks the canonical digest and independently re-derives every semantic field from the sufficient facts embedded in the certificate. Basis mode additionally checks the source-file digest and regenerates the bundle from the pair report or exact-count disclosure. A coordinated change to both the basis and all dependent digests still requires an external trust anchor, such as a signed release or trusted execution record.
