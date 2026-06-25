# Third-Party Baselines

This directory records the exact upstream provenance for the baseline implementations used in this artifact: repository URL, pinned commit, source subpath, and reviewer-facing checkout label.

- `STONE-watermarking.UPSTREAM.json` captures the pinned upstream URL and commit for `stone_runtime`.
- `SWEET-watermark.UPSTREAM.json` captures the pinned upstream URL and commit for `sweet_runtime`.
- `EWD.UPSTREAM.json` captures the pinned upstream URL and commit for `ewd_runtime`.
- `KGW-lm-watermarking.UPSTREAM.json` captures the pinned upstream URL and commit for `kgw_runtime`.
- Imported pinned upstream checkouts stay outside the anonymous review artifact.
- The review artifact relies on the pinned provenance manifests plus fetch scripts instead of shipping the external checkout contents directly.
- `STONE`, `SWEET`, and `EWD` are currently tracked as `license_status: unverified`, so this artifact does not vendor those upstream runtime checkouts.
- `KGW` is tracked as `license_status: redistributable`, but the default artifact path still uses provenance manifests plus fetch scripts rather than shipping a second source-code host inside the artifact bundle.
- The tracked manifests use neutral `external_checkout/...` labels, and local fetch helpers materialize runtime checkouts under that same namespace without bundling those fetched checkouts by default.
- Reviewer-facing baseline-screening notes for included and excluded methods live in [`docs/baseline_screening.md`](../docs/baseline_screening.md).
