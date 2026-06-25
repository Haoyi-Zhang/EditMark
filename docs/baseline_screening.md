# Baseline Screening Notes

The artifact includes four runnable watermarking baselines because each one exposes the minimum contract needed for the paper's comparison: generation, a declared detector or rule, a threshold or decision rule, negative controls, transformation-time validation, support accounting, and cost accounting under the same harness.

## Inclusion Criteria

A baseline is included only when it satisfies all of the following reviewer-facing requirements:

- it can be executed locally or through a pinned upstream checkout;
- it exposes a detector, rule, or score that can be evaluated on generated and transformed programs;
- it allows negative controls and task-level validation to be recorded in the same result table;
- it does not require a private service, private model checkpoint, or undisclosed provider-side detector;
- it can be represented without shipping third-party code whose license status is unclear.

## Exclusion Boundary

Methods are outside this artifact when they require proprietary provider access, lack a runnable implementation for source-code generation, omit a detector usable on transformed programs, or cannot be redistributed safely for anonymous review. This boundary is a reproducibility choice, not a claim that excluded methods are weaker.

## Third-Party Handling

The `third_party/` manifests record upstream URL, pinned commit, source subpath, and redistribution status. The default review package does not vendor upstream runtime checkouts whose license status is unverified. Reviewers who want a fresh execution can fetch those upstreams explicitly, but the submitted evidence surface is the shipped static table and figure layer.

## Interpretation

The four included baselines support a completed-surface claim: clean detection is not automatically exchangeable with post-edit provenance evidence on this reproducible comparison surface. The artifact does not claim population coverage over every watermarking family, future method, closed provider detector, or deployment environment.
