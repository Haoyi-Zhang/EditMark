# Third-Party Notices

This repository is the anonymous review artifact for the submitted benchmark. It records
third-party provenance needed to inspect the benchmark while avoiding accidental
redistribution of upstream source trees whose license status has not been
verified for bundling.

## Runtime Watermarking Baselines

| Method | Upstream repository | Pinned commit | Redistribution status in this release |
| --- | --- | --- | --- |
| `STONE` / `stone_runtime` | `https://github.com/inistory/STONE-watermarking.git` | `bb5d809c0c494a219411e861f2313cca2b9fd7b4` | License status `unverified`; not redistributed in this artifact. |
| `SWEET` / `sweet_runtime` | `https://github.com/hongcheki/sweet-watermark.git` | `853b47eb064c180beebd383302d09491fc98a565` | License status `unverified`; not redistributed in this artifact. |
| `EWD` / `ewd_runtime` | `https://github.com/luyijian3/EWD.git` | `605756acf802528a3df89d95a4661a031eafc79b` | License status `unverified`; not redistributed in this artifact. |
| `KGW` / `kgw_runtime` | `https://github.com/jwkirchenbauer/lm-watermarking.git` | `82922516930c02f8aa322765defdb5863d07a00e` | Marked redistributable in the provenance manifest; fetched by pinned workflow for consistency with the other baselines. |

The machine-readable manifests live under [`third_party/`](third_party). The
runtime fetch helper is [`scripts/fetch_runtime_upstreams.sh`](scripts/fetch_runtime_upstreams.sh).

## Adapter Boundary

The benchmark layer adapts orchestration, model loading, decoding policy, per-run
reporting, and failure handling around the pinned baseline implementations. The
release claim is limited to this adapter boundary: the benchmark layer routes
inputs and outputs through a shared execution contract and does not intentionally
edit the upstream watermark/detector algorithm logic for the active runtime
comparisons.

## Benchmark Sources And Models

The anonymous artifact contains normalized benchmark inputs under
[`data/release/sources/`](data/release/sources). The repository does not
redistribute Hugging Face model weights. Full Level 3 reruns require external
availability of the exact model identifiers and snapshot revisions listed in
the [`README.md`](README.md), plus the pinned upstream baseline repositories
above.

The packaged repository is sufficient for Level 1 artifact inspection and for
regenerating the included summary tables and figures. It is not a redistribution
vehicle for all external model weights or every third-party baseline source
tree.
