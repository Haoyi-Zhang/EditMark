# Release Sources

This directory contains the exact normalized source files executed by the canonical review artifact suite.

The release truth for this artifact is:

- `suite_humaneval_plus_release.normalized.jsonl`
- `suite_mbpp_plus_release.normalized.jsonl`
- `suite_humanevalx_release.normalized.jsonl`
- `suite_mbxp_release.normalized.jsonl`
- `crafted_original_release.normalized.jsonl`
- `crafted_translation_release.normalized.jsonl`
- `crafted_stress_release.normalized.jsonl`

These files are the reviewer-facing execution truth for dataset statistics, subset manifests, and full-suite manifests.

## Current Checkout Source Identity

Static checkout audit `paper/data_folder_current_checkout_alignment_audit_20260611_0905.md` records the following row-count and SHA-256 identity for the seven normalized JSONL release sources. This table is a source-identity guard only; it is not experiment, model, GPU, scoring, or regeneration evidence.

| Source group | Rows | Normalized JSONL | SHA-256 |
| --- | ---: | --- | --- |
| `crafted_original` | 240 | `crafted_original_release.normalized.jsonl` | `ED1811846A4797E96DB35D3682D99E24F211ED25A5560FD7078F308952A4EB62` |
| `crafted_stress` | 240 | `crafted_stress_release.normalized.jsonl` | `76842FF72868B4274781277CDDFD2B3FEF4A41B37793E83FF09413FB9806A470` |
| `crafted_translation` | 240 | `crafted_translation_release.normalized.jsonl` | `DBD7A26F6D7DD5BCEC2C5A0C88B7168877CF9C071FDF3E091417CB5ED6FE5AC3` |
| `suite_humaneval_plus` | 164 | `suite_humaneval_plus_release.normalized.jsonl` | `977D2798566B6E7CC32DEC2AA24D65B8F028565945E63FAB039850EE8FFC3C55` |
| `suite_humanevalx` | 200 | `suite_humanevalx_release.normalized.jsonl` | `6E5FC7A1DF02B3238C5B748B9CAD9FDB7E70BFAE785AE7A904174985D8F647BA` |
| `suite_mbpp_plus` | 378 | `suite_mbpp_plus_release.normalized.jsonl` | `1724A5EB3C3A39461F12A379047F1CD12D1C41EDDFA02767B691ABCF80C1CAFB` |
| `suite_mbxp` | 200 | `suite_mbxp_release.normalized.jsonl` | `9C46DD838D902814A27C30470B8C94010AE323EF8841922BC8F92D5564E75E5D` |
| **Total** | **1,662** | seven normalized JSONL sources | - |

Provenance is intentionally split:

- `suite_humaneval_plus_release`, `suite_mbpp_plus_release`, `suite_humanevalx_release`, and `suite_mbxp_release` are public benchmark execution slices
- `crafted_original_release`, `crafted_translation_release`, and `crafted_stress_release` are curated crafted benchmark families with manually finalized release records

For the multilingual public execution layer, `HumanEval-X`, `MBXP-5lang`, and all three crafted families are interpreted through the same balanced five-language runtime set: `python`, `cpp`, `java`, `javascript`, and `go`. `MBXP-5lang` remains a deterministic five-language balanced slice with explicit smoke-overlay support in the release metadata.

For those multilingual public slices, the balanced five-language claim is defined at the source level. The normalized JSONL files are intentionally serialized one executed language per row, so reviewer-facing row metadata should be read as per-language execution records inside a fixed five-language release slice, not as a contradiction of the slice definition.

Naming is also layered on purpose:

- public labels: `HumanEval-X (5-language balanced slice)`, `MBXP-5lang (5-language balanced slice)`
- manifest keys: `humaneval_x`, `mbxp_5lang`
- release files: `suite_humanevalx_release.normalized.jsonl`, `suite_mbxp_release.normalized.jsonl`

Those aliases all refer to the same two public multilingual sources.

The normalized source files are frozen execution inputs for the completed run. Some task text may preserve generation-time wording from the canonical input snapshot; the release-facing claim is the documented curated/manual-review process for the three crafted families, not a credential claim about external certified experts.

Some public benchmark rows also retain upstream test-section wording such as `manually generated tests` or `automatically generated tests`. That wording belongs to the upstream executable benchmark assets and should not be read as a claim about how the crafted release families in this repository were constructed.

This source layer tracks release truth and finalized provenance. Execution-backed semantic-validation evidence is generated during benchmark runs and export regeneration rather than embedded as row-level runtime annotations in the pre-run public snapshot.
