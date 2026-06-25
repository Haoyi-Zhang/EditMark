from __future__ import annotations

from pathlib import Path


def test_makefile_targets_default_to_suite_article_flow() -> None:
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    removed_matrix_alias = "paper" + "-matrix"
    removed_precheck_alias = "paper" + "-suite-precheck"

    assert "SUITE_MATRIX_MANIFEST ?= configs/matrices/suite_all_models_methods.json" in makefile
    assert "SUITE_MATRIX_PROFILE ?= suite_all_models_methods" in makefile
    assert "PRECHECK_STAGE_A_MANIFEST ?= configs/matrices/suite_canary_heavy.json" in makefile
    assert "PRECHECK_STAGE_A_PROFILE ?= suite_canary_heavy" in makefile
    assert "PRECHECK_STAGE_B_MANIFEST ?= configs/matrices/model_invocation_smoke.json" in makefile
    assert "PRECHECK_STAGE_B_PROFILE ?= model_invocation_smoke" in makefile
    assert "PRECHECK_GATE_OUTPUT ?= results/certifications/suite_precheck_gate.json" in makefile
    assert "dev-suite-matrix-dry-run" in makefile
    assert "suite-precheck" in makefile
    assert "suite-clean" in makefile
    assert "scripts/certify_suite_precheck.py" in makefile
    assert "scripts/clean_suite_outputs.py" in makefile
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $(PYTHON) scripts/capture_environment.py --label formal_execution_host_pre_rerun --execution-mode single_host_canonical --output-json results/environment/runtime_environment.json --output-md results/environment/runtime_environment.md" in makefile
    assert "\nrun:" not in makefile
    assert "\ndebug:" not in makefile
    assert "remote-preflight" not in makefile
    assert "remote-run-full" not in makefile
    assert "article-preflight" not in makefile
    assert "package-anon" not in makefile
    assert removed_matrix_alias not in makefile
    assert removed_precheck_alias not in makefile
