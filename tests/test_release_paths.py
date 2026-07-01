from __future__ import annotations

from pathlib import Path

from posteditbench.language_support import source_relative_to
from posteditbench.public_benchmarks import PublicBenchmarkSpec, normalize_public_records


def test_release_path_helpers_redact_coordination_paths() -> None:
    root = Path("R:/review/artifact/PostEditBench")
    source = root / ".coordination" / "external" / "mxeval" / "data" / "mbxp" / "mbcpp_release_v1.2.jsonl"

    assert source_relative_to(root, source) == "mbcpp_release_v1.2.jsonl"
    spec = PublicBenchmarkSpec(
        name="human_eval",
        dataset_label="HumanEval",
        source_url="https://example.invalid/human_eval.jsonl.gz",
        source_revision="deadbeef",
        license_note="test",
        split="test",
        task_count=1,
        adapter_name="human-eval",
    )
    rows = normalize_public_records(
        spec,
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):",
                "canonical_solution": "\n    return a + b\n",
                "test": "assert add(1, 2) == 3",
            }
        ],
        source,
    )

    assert rows[0]["source_path"] == "mbcpp_release_v1.2.jsonl"
    assert rows[0]["official_problem_file"] == "mbcpp_release_v1.2.jsonl"


def test_release_path_helpers_redact_public_cache_paths() -> None:
    root = Path("R:/review/artifact/PostEditBench")
    source = root / "data" / "public" / "_cache" / "humaneval_plus.source.jsonl.gz"

    assert source_relative_to(root, source) == "humaneval_plus.source.jsonl.gz"


def test_release_path_helpers_redact_public_cache_repo_paths() -> None:
    root = Path("R:/review/artifact/PostEditBench")
    source = (
        root
        / "data"
        / "public"
        / "_cache"
        / "repos"
        / "zai-org-CodeGeeX-2838420b7b44"
        / "codegeex"
        / "benchmark"
        / "humaneval-x"
        / "cpp"
        / "data"
        / "humaneval_cpp.jsonl.gz"
    )

    assert source_relative_to(root, source) == "humaneval_cpp.jsonl.gz"
