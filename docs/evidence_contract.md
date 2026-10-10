# Evidence contract

A post-edit watermark claim is a conditional before/after statement. The default target contains a pair only when:

- the edit reports support;
- original and edited source identities are known and differ;
- the original and edited programs both pass the same named validation contract;
- both detector decisions use the same rule identifier, threshold, comparator, tokenizer/configuration, and key reference;
- both decisions are observed for a point estimate.

Unknown values remain unknown. They are neither successes nor failures.

## Estimands

For transition counts `n00`, `n01`, `n10`, and `n11` on the admitted positive cohort:

- before positive rate: `(n10 + n11) / n`;
- after positive rate: `(n01 + n11) / n`;
- net decision change: `(n01 - n10) / n`;
- net decision loss: `(n10 - n01) / n`;
- survival among initially positive programs: `n11 / (n10 + n11)`.

Net change is identified by exact cohort size and before/after positive marginals. Survival generally needs the joint table. With only `n`, `b`, and `a`, the sharp survival interval is

`max(0, b + a - n) / b <= survival <= min(b, a) / b`, for `b > 0`.

These are finite-cohort identification bounds, not confidence intervals.

For target pairs with missing decisions, the analyzer uses only each observed Boolean or null; it does not infer a missing Boolean from a retained score. The reported change and loss intervals are sharp over unrestricted completions of that coarsened projection. They may be conservative relative to all supplied facts when a score, threshold, and comparator determine a missing decision. Recorded availability and complete-cohort membership remain unchanged.

## Unsupported inferences

Positive pairs alone do not identify:

- clean or edited false-positive rate;
- AUROC or AUROC change;
- equal cross-method calibration;
- the prevalence of generated code in deployment;
- the posterior probability that a flagged program has a given origin.

Those claims require separately identified negative cohorts, score distributions, calibration data, and/or a deployment prevalence model.


## Certificate verification

The certificate generator and verifier are separate command-line paths. The verifier rejects a modified certificate when its hash is stale and also rejects a semantically inconsistent certificate after an attacker recomputes the hash. With the original evidence basis, it verifies the recorded SHA-256 and regenerates the complete bundle. This establishes integrity and logical sufficiency, not the truth of the supplied observations.

These paths share the analyzer and certifier implementation. Basis verification against a pair report checks consistency with that report; it does not reconstruct raw-record membership or authenticate execution. Both certificate disclosure paths leave `membership_verified` and `cohort_membership_authenticated` false. Re-derived count-report fields must agree as canonical JSON, including the distinction between Booleans and integers. All report writers, including the verification writer, publish exclusively without replacing an existing path.
