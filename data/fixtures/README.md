# PostEditBench Fixtures

This directory contains self-contained sample inputs used by the reproducibility scripts.

Files:

- `benchmarks/synthetic_tasks.jsonl` is a small benchmark fixture with mixed-language tasks.
- `benchmark.normalized.jsonl` is the canonical normalized benchmark fixture consumed by the runtime and is rebuildable via `python scripts/prepare_data.py --bootstrap-fixture`.
- `attacks/attack_matrix.json` defines the deterministic attack sweep used by the local experiment runner.

These fixtures are intentionally synthetic so the repository can be shared without any external dataset access or identity-bearing metadata.
