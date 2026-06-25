from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for candidate in (ROOT, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import render_paper_figures as rpf
from posteditbench.suite import suite_model_spec

DEFAULT_SUMMARY_DIR = ROOT / "results" / "figures" / "suite_all_models_methods"
DEFAULT_TABLE_DIR = ROOT / "results" / "tables" / "suite_all_models_methods"
DEFAULT_EXPORT_IDENTITY = DEFAULT_TABLE_DIR / "suite_all_models_methods_export_identity.json"
EXPECTED_METHODS = {"STONE", "SWEET", "EWD", "KGW"}
EXPECTED_MODELS = tuple(rpf.SUITE_MODEL_ROSTER)
REQUIRED_EXPORT_IDENTITY_TABLES = (
    "method_summary.json",
    "suite_all_models_methods_method_master_leaderboard.json",
    "suite_all_models_methods_method_model_leaderboard.json",
    "suite_all_models_methods_model_method_functional_quality.json",
)
REQUIRED_EXPORT_IDENTITY_FIGURE_SUFFIXES = (".png", ".pdf", ".json", ".csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw shipped full-run figures from materialized summary artifacts.")
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--export-identity", type=Path, default=DEFAULT_EXPORT_IDENTITY)
    parser.add_argument("--prefix", type=str, default="suite_all_models_methods")
    parser.add_argument(
        "--require-times-new-roman",
        dest="require_times_new_roman",
        action="store_true",
        default=True,
    )
    parser.add_argument("--allow-font-fallback", dest="require_times_new_roman", action="store_false")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object at {path}, but found {type(payload).__name__}.")
    return payload


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_export_identity(
    *,
    export_identity_path: Path,
    table_dir: Path,
    prefix: str,
    method_summary_rows: list[dict[str, Any]],
    method_master_rows: list[dict[str, Any]],
) -> None:
    if not export_identity_path.exists():
        raise SystemExit(
            f"Figure-only redraw requires the export identity sidecar at {export_identity_path}. "
            "Regenerate the canonical summary tables first."
        )
    identity = _load_object(export_identity_path)
    if str(identity.get("artifact_role", "")).strip() != "suite_all_models_methods_release_summary_export_identity":
        raise SystemExit(f"{export_identity_path} is not a recognized suite_all_models_methods export identity sidecar.")
    if str(identity.get("manifest", "")).strip() != rpf._CANONICAL_SUITE_MANIFEST:
        raise SystemExit(f"{export_identity_path} does not point to the canonical suite manifest.")
    if str(identity.get("profile", "")).strip() != rpf._CANONICAL_SUITE_PROFILE:
        raise SystemExit(f"{export_identity_path} does not point to the canonical suite profile.")
    if str(identity.get("canonical_manifest_digest", "")).strip() != rpf._CANONICAL_SUITE_MANIFEST_DIGEST:
        raise SystemExit(f"{export_identity_path} does not carry the canonical suite manifest digest.")
    if str(identity.get("execution_mode", "")).strip() != rpf._FORMAL_CANONICAL_EXECUTION_MODE:
        raise SystemExit(f"{export_identity_path} must record execution_mode={rpf._FORMAL_CANONICAL_EXECUTION_MODE}.")
    if int(identity.get("run_count", 0) or 0) != 140 or int(identity.get("success_count", 0) or 0) != 140:
        raise SystemExit(f"{export_identity_path} does not describe the canonical 140/140 release surface.")
    if int(identity.get("failed_count", 0) or 0) != 0:
        raise SystemExit(f"{export_identity_path} must record failed_count=0 for the canonical redraw path.")

    expected_stems = {
        f"{prefix}_score_decomposition",
        f"{prefix}_detection_vs_utility",
    }
    observed_stems = {str(value).strip() for value in identity.get("figure_stems", []) if str(value).strip()}
    if observed_stems != expected_stems:
        raise SystemExit(
            f"{export_identity_path} does not match the narrowed publication-facing figure roster: "
            f"expected {sorted(expected_stems)}, observed {sorted(observed_stems)}."
        )

    recorded_hashes = identity.get("required_table_hashes", {})
    if not isinstance(recorded_hashes, dict):
        raise SystemExit(f"{export_identity_path} must record required_table_hashes for the redraw contract.")
    for filename in REQUIRED_EXPORT_IDENTITY_TABLES:
        recorded_hash = str(recorded_hashes.get(filename, "")).strip().lower()
        table_path = table_dir / filename
        if not recorded_hash:
            raise SystemExit(f"{export_identity_path} is missing the recorded hash for {filename}.")
        if not table_path.exists():
            raise SystemExit(f"Figure-only redraw requires {table_path} to exist.")
        actual_hash = _sha256(table_path)
        if actual_hash != recorded_hash:
            raise SystemExit(
                f"{table_path} does not match the recorded export identity. "
                "Regenerate the canonical summary tables before redrawing figures."
            )

    matrix_index_relpath = str(identity.get("matrix_index", "")).strip()
    matrix_index_sha256 = str(identity.get("matrix_index_sha256", "")).strip().lower()
    if matrix_index_relpath and matrix_index_sha256:
        matrix_index_path = Path(matrix_index_relpath)
        if not matrix_index_path.is_absolute():
            matrix_index_path = ROOT / matrix_index_path
        if matrix_index_path.exists():
            actual_matrix_index_sha256 = _sha256(matrix_index_path)
            if actual_matrix_index_sha256 != matrix_index_sha256:
                raise SystemExit(
                    f"{matrix_index_path} does not match the recorded matrix_index_sha256 in {export_identity_path}."
                )

    summary_score_versions = {
        str(row.get("score_version", "")).strip()
        for row in method_summary_rows
        if str(row.get("score_version", "")).strip()
    }
    if len(summary_score_versions) != 1 or str(identity.get("score_version", "")).strip() not in summary_score_versions:
        raise SystemExit(
            f"{export_identity_path} does not agree with method_summary.json on the active score_version."
        )

    master_score_versions = {
        str(row.get("score_version", "")).strip()
        for row in method_master_rows
        if str(row.get("score_version", "")).strip()
    }
    if len(master_score_versions) != 1 or str(identity.get("score_version", "")).strip() not in master_score_versions:
        raise SystemExit(
            f"{export_identity_path} does not agree with suite_all_models_methods_method_master_leaderboard.json on the active score_version."
        )

    observed_methods = {
        str(row.get("method", "")).strip()
        for row in method_master_rows
        if str(row.get("method", "")).strip()
    }
    if observed_methods != EXPECTED_METHODS:
        raise SystemExit(
            "Figure-only redraw requires the canonical four-method release leaderboard surface; "
            f"observed methods={sorted(observed_methods)}."
        )

    recorded_model_roster = identity.get("model_roster", [])
    if not isinstance(recorded_model_roster, list) or not recorded_model_roster:
        raise SystemExit(f"{export_identity_path} must record the canonical model_roster with pinned revisions.")
    normalized_model_roster: list[dict[str, str]] = []
    for index, entry in enumerate(recorded_model_roster, start=1):
        if not isinstance(entry, dict):
            raise SystemExit(f"{export_identity_path} model_roster[{index}] is not an object.")
        model_name = str(entry.get("model", "")).strip()
        model_revision = str(entry.get("model_revision", "")).strip()
        if not model_name or not model_revision:
            raise SystemExit(f"{export_identity_path} model_roster[{index}] must carry model and model_revision.")
        spec = suite_model_spec(model_name)
        if spec is None:
            raise SystemExit(f"{export_identity_path} model_roster[{index}] names a non-canonical model: {model_name}.")
        if model_revision != spec.revision:
            raise SystemExit(
                f"{export_identity_path} model_roster[{index}] revision mismatch for {model_name}: "
                f"expected {spec.revision}, observed {model_revision}."
            )
        normalized_model_roster.append({"model": model_name, "model_revision": model_revision})
    observed_model_order = [entry["model"] for entry in normalized_model_roster]
    if observed_model_order != list(EXPECTED_MODELS):
        raise SystemExit(
            f"{export_identity_path} must record the canonical five-model roster in release order; "
            f"expected {list(EXPECTED_MODELS)}, observed {observed_model_order}."
        )


def _overall_method_order(rows: list[dict[str, Any]]) -> list[str]:
    sorted_rows = sorted(rows, key=lambda row: float(row.get(rpf.HEADLINE_SCORE_FIELD, 0.0)), reverse=True)
    return [str(row.get("method", "")).strip() for row in sorted_rows if str(row.get("method", "")).strip()]


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    return 0.0 if math.isnan(numeric) or math.isinf(numeric) else numeric


def _rows_by_key(rows: list[dict[str, Any]], *, key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key, "")).strip(): row for row in rows if str(row.get(key, "")).strip()}


def _render_overall(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    ordered = sorted(rows, key=lambda row: _safe_float(row.get(rpf.HEADLINE_SCORE_FIELD, 0.0)), reverse=True)
    labels = [rpf._paper_label(str(row.get("method", ""))) for row in ordered]
    scores = [_safe_float(row.get(rpf.HEADLINE_SCORE_FIELD, 0.0)) for row in ordered]
    axis_limit = rpf._score_axis_limit(scores)
    fig, ax = plt.subplots(
        figsize=(rpf.SINGLE_COLUMN_WIDTH, rpf._adaptive_track_figure_height(len(ordered), base=3.7, per_item=0.34, floor=3.7)),
        constrained_layout=True,
    )
    positions = list(range(len(ordered)))
    bars = ax.barh(
        positions,
        scores,
        color=[rpf._method_color(str(row.get("method", "")), str(row.get("origin", ""))) for row in ordered],
        edgecolor="white",
        linewidth=0.8,
    )
    for position, score, bar in zip(positions, scores, bars):
        if score > 0.0:
            ax.text(score + axis_limit * 0.025, position, f"{score:.2f}", va="center", ha="left", fontsize=9)
        else:
            ax.scatter([axis_limit * 0.01], [position], s=34, color=bar.get_facecolor(), edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(axis_limit * 0.035, position, "0.00", va="center", ha="left", fontsize=9)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, axis_limit)
    ax.set_xlabel(rpf.HEADLINE_SCORE_FIELD)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    return rpf._save_figure(fig, output_dir, stem)


def _render_score_decomposition(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    ordered = sorted(rows, key=lambda row: _safe_float(row.get(rpf.HEADLINE_SCORE_FIELD, 0.0)), reverse=True)
    metric_specs = [
        ("detection_separability", "Det"),
        ("robustness", "Rob"),
        ("utility", "Utility"),
        ("stealth_conditioned", "Stealth*"),
        ("efficiency_conditioned", "Eff*"),
    ]
    fig, ax = plt.subplots(
        figsize=(rpf.SINGLE_COLUMN_WIDTH, rpf._adaptive_track_figure_height(len(ordered), base=3.65, per_item=0.3, floor=3.65)),
        constrained_layout=True,
    )
    positions = list(range(len(ordered)))
    labels = [rpf._paper_label(str(row.get("method", ""))) for row in ordered]
    cumulative = [0.0] * len(ordered)
    for key, label in metric_specs:
        values = [_safe_float(row.get(key, 0.0)) for row in ordered]
        ax.barh(
            positions,
            values,
            left=cumulative,
            color=rpf._METRIC_COLORS.get(key, "#9fb3c8"),
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        cumulative = [left + value for left, value in zip(cumulative, values)]
    for position, total in zip(positions, cumulative):
        ax.text(total + 0.012, position, f"{total:.2f}", va="center", ha="left", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Conditioned core factors")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0, columnspacing=0.9, handlelength=1.6)
    return rpf._save_figure(fig, output_dir, stem, data=ordered)


def _render_generalization(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    ordered = sorted(rows, key=lambda row: _safe_float(row.get("generalization", 0.0)), reverse=True)
    fig, ax = plt.subplots(
        figsize=(rpf.SINGLE_COLUMN_WIDTH, rpf._adaptive_track_figure_height(len(ordered), base=3.45, per_item=0.28, floor=3.45)),
        constrained_layout=True,
    )
    labels = [rpf._paper_label(str(row.get("method", ""))) for row in ordered]
    values = [_safe_float(row.get("generalization", 0.0)) for row in ordered]
    positions = list(range(len(ordered)))
    ax.hlines(positions, [0.0] * len(values), values, color="#d7dde5", linewidth=2.2, zorder=1)
    for position, row, value in zip(positions, ordered, values):
        ax.scatter(value, position, s=70, color=rpf._method_color(str(row.get("method", "")), str(row.get("origin", ""))), edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(min(1.02, value + 0.02), position, f"{value:.2f}", va="center", ha="left", fontsize=9)
    ax.set_xlim(0.0, 1.05)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Generalization")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    return rpf._save_figure(fig, output_dir, stem, data=ordered)


def _render_quality_vs_robustness(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(rpf.SINGLE_COLUMN_WIDTH, 3.55), constrained_layout=True)
    ax.axvline(0.5, color="#d7dde5", linewidth=1.0, linestyle="--", zorder=1)
    ax.axhline(0.5, color="#d7dde5", linewidth=1.0, linestyle="--", zorder=1)
    ax.grid(alpha=0.25, linewidth=0.8)
    ordered = sorted(rows, key=lambda row: _safe_float(row.get(rpf.HEADLINE_SCORE_FIELD, 0.0)), reverse=True)
    for index, row in enumerate(ordered):
        utility = _safe_float(row.get("utility", 0.0))
        robustness = _safe_float(row.get("robustness", 0.0))
        label = rpf._paper_label(str(row.get("method", "")))
        ax.scatter(
            utility,
            robustness,
            s=92,
            color=rpf._method_color(str(row.get("method", "")), str(row.get("origin", ""))),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offset_x, offset_y = rpf._scatter_label_offset(index)
        ax.annotate(
            label,
            (utility, robustness),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Utility")
    ax.set_ylabel("Robustness")
    return rpf._save_figure(fig, output_dir, stem, data=ordered)


def _render_functional_summary(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    return rpf._plot_functional_dotplot(plt, rows_payload=rows, title="", output_dir=output_dir, stem=stem)


def _render_source_breakdown(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    source_groups = [rpf.normalize_source_group(source_group) for source_group in rpf.SUITE_AGGREGATE_SOURCE_GROUPS]
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not methods:
        methods = sorted({str(row.get("method", "")).strip() for row in rows if str(row.get("method", "")).strip()})
    by_pair = {(str(row.get("method", "")).strip(), rpf.normalize_source_group(str(row.get("source_group", "")).strip())): row for row in rows}
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        for source_group in source_groups:
            row = by_pair.get((method, source_group), {})
            value = _safe_float(row.get("core_score", 0.0))
            values.append(value)
            payload.append(
                {
                    "method": method,
                    "paper_label": rpf._paper_label(method),
                    "source_group": source_group,
                    "source_label": rpf._suite_source_label(source_group),
                    "source_short_label": rpf._paper_source_release_label(source_group),
                    "core_score": value,
                    "row_count": int(row.get("row_count", 0)) if row else 0,
                }
            )
        matrix.append(values)
    max_score = max((max(values) for values in matrix), default=0.0)
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_label(method) for method in methods],
        col_labels=[rpf._paper_source_release_label(source_group) for source_group in source_groups],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=stem,
        cmap="Blues",
        vmax=max(0.55, max_score + 0.04),
        colorbar_label="Core Score",
        data=payload,
        max_columns_per_panel=len(source_groups),
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def _render_model_breakdown(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    model_order = [model for model in rpf.SUITE_MODEL_ROSTER if any(str(row.get("model", "")).strip() == model for row in rows)]
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not methods:
        methods = sorted({str(row.get("method", "")).strip() for row in rows if str(row.get("method", "")).strip()})
    by_pair = {(str(row.get("method", "")).strip(), str(row.get("model", "")).strip()): row for row in rows}
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        for model in model_order:
            row = by_pair.get((method, model), {})
            value = _safe_float(row.get(rpf.HEADLINE_SCORE_FIELD, 0.0))
            values.append(value)
            payload.append(
                {
                    "method": method,
                    "paper_label": rpf._paper_label(method),
                    "model": model,
                    "model_label": rpf._paper_model_label(model),
                    rpf.HEADLINE_SCORE_FIELD: value,
                    "row_count": int(row.get("row_count", 0)) if row else 0,
                }
            )
        matrix.append(values)
    max_score = max((max(values) for values in matrix), default=0.0)
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_label(method) for method in methods],
        col_labels=[rpf._paper_model_axis_label(model) for model in model_order],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=stem,
        cmap="Blues",
        vmax=max(0.6, min(1.0, max_score + 0.08)),
        data=payload,
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def _render_language_breakdown(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    languages = [language for language in rpf.OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES if any(str(row.get("language", "")).strip() == language for row in rows)]
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not languages or not methods:
        return []
    by_pair = {(str(row.get("method", "")).strip(), str(row.get("language", "")).strip()): row for row in rows}
    fig_height = max(4.0, 1.55 * len(languages) + 0.65)
    fig, axes = plt.subplots(len(languages), 1, figsize=(rpf.SINGLE_COLUMN_WIDTH, fig_height), sharex=True, constrained_layout=True)
    if len(languages) == 1:
        axes = [axes]
    method_labels = [rpf._paper_label(method) for method in methods]
    for axis, language in zip(axes, languages):
        values = [_safe_float(by_pair.get((method, language), {}).get("utility", 0.0)) for method in methods]
        positions = list(range(len(methods)))
        axis.hlines(positions, [0.0] * len(values), values, color="#d7dde5", linewidth=2.2, zorder=1)
        for position, method, value in zip(positions, methods, values):
            axis.scatter(value, position, s=46, color=rpf._method_color(method), edgecolor="white", linewidth=0.8, zorder=3)
        axis.text(0.0, 1.01, language.title(), transform=axis.transAxes, ha="left", va="bottom", fontsize=10)
        axis.set_yticks(positions, method_labels)
        axis.set_xlim(0.0, 1.05)
        axis.grid(axis="x", alpha=0.25, linewidth=0.8)
        axis.invert_yaxis()
    axes[-1].set_xlabel("Utility")
    axes[len(axes) // 2].set_ylabel("Method")
    return rpf._save_figure(fig, output_dir, stem, data=rows)


def _render_attack_breakdown(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    attack_order = [
        "block_shuffle",
        "budgeted_adaptive",
        "comment_strip",
        "control_flow_flatten",
        "identifier_rename",
        "noise_insert",
        "whitespace_normalize",
    ]
    attacks = [attack for attack in attack_order if any(str(row.get("attack", "")).strip() == attack for row in rows)]
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not attacks or not methods:
        return []
    by_pair = {(str(row.get("method", "")).strip(), str(row.get("attack", "")).strip()): row for row in rows}
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        for attack in attacks:
            row = by_pair.get((method, attack), {})
            value = _safe_float(row.get("attacked_detect_rate", 0.0))
            values.append(value)
            payload.append(
                {
                    "method": method,
                    "paper_label": rpf._paper_label(method),
                    "attack": attack,
                    "attack_label": rpf._paper_attack_label(attack),
                    "attacked_detect_rate": value,
                    "row_count": int(row.get("row_count", 0)) if row else 0,
                }
            )
        matrix.append(values)
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_label(method) for method in methods],
        col_labels=[rpf._paper_attack_label(attack) for attack in attacks],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=stem,
        cmap="Purples",
        data=payload,
        max_columns_per_panel=len(attacks),
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def _render_source_language_coverage(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    source_order = [source.source_group for source in rpf.SUITE_AGGREGATE_SOURCES]
    languages = list(rpf.OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES)
    by_pair = {(rpf.normalize_source_group(str(row.get("source_group", "")).strip()), str(row.get("language", "")).strip().lower()): row for row in rows}
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for source_group in source_order:
        values: list[float] = []
        for language in languages:
            row = by_pair.get((rpf.normalize_source_group(source_group), language), {})
            value = _safe_float(row.get("count", 0))
            values.append(value)
            payload.append(
                {
                    "source_group": rpf.normalize_source_group(source_group),
                    "source_label": rpf._suite_source_label(rpf.normalize_source_group(source_group)),
                    "language": language,
                    "count": int(value),
                }
            )
        matrix.append(values)
    max_value = max((max(values) for values in matrix), default=0.0)
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_source_release_label(source_group) for source_group in source_order],
        col_labels=[language.title() for language in languages],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=stem,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=max(1.0, max_value),
        colorbar_label="Release Records",
        data=payload,
        annotate=True,
        annotation_fmt="{:.0f}",
    )


def _render_method_stability(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not methods:
        methods = sorted({str(row.get("method", "")).strip() for row in rows if str(row.get("method", "")).strip()})
    by_method = _rows_by_key(rows, key="method")
    metric_names = [
        ("source_stability", "Source"),
        ("task_stability", "Task"),
        ("language_stability", "Language"),
        ("cross_family_transfer", "Cross-family"),
    ]
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        row = by_method.get(method, {})
        values = [_safe_float(row.get(metric_name, 0.0)) for metric_name, _ in metric_names]
        matrix.append(values)
        payload.append({"method": method, "paper_label": rpf._paper_label(method), **{metric_name: _safe_float(row.get(metric_name, 0.0)) for metric_name, _ in metric_names}})
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_label(method) for method in methods],
        col_labels=[label for _, label in metric_names],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=stem,
        cmap="Blues",
        data=payload,
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def _render_detection_vs_utility(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    ordered = sorted(rows, key=lambda row: _safe_float(row.get(rpf.HEADLINE_SCORE_FIELD, 0.0)), reverse=True)
    fig, ax = plt.subplots(figsize=(rpf.SINGLE_COLUMN_WIDTH, 3.45), constrained_layout=True)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.axvline(0.5, color="#d7dde5", linewidth=1.0, linestyle="--", zorder=1)
    ax.axhline(0.5, color="#d7dde5", linewidth=1.0, linestyle="--", zorder=1)
    ax.grid(alpha=0.25, linewidth=0.8)
    for index, row in enumerate(ordered):
        detection = _safe_float(row.get("detection_separability", 0.0))
        utility = _safe_float(row.get("utility", 0.0))
        label = rpf._paper_label(str(row.get("method", "")))
        ax.scatter(
            detection,
            utility,
            s=92,
            color=rpf._method_color(str(row.get("method", "")), str(row.get("origin", ""))),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offset_x, offset_y = rpf._scatter_label_offset(index)
        ax.annotate(
            label,
            (detection, utility),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    ax.set_xlabel("Detection Separability")
    ax.set_ylabel("Utility")
    return rpf._save_figure(fig, output_dir, stem, data=ordered)


def _render_model_generalization(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not methods:
        methods = sorted({str(row.get("method", "")).strip() for row in rows if str(row.get("method", "")).strip()})
    by_method = _rows_by_key(rows, key="method")
    metric_names = [
        ("cross_family_transfer", "Cross-family"),
        ("scale_consistency", "Within-family scale"),
    ]
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        row = by_method.get(method, {})
        values = [_safe_float(row.get(metric_name, 0.0)) for metric_name, _ in metric_names]
        matrix.append(values)
        payload.append(
            {
                "method": method,
                "paper_label": rpf._paper_label(method),
                "cross_family_transfer": _safe_float(row.get("cross_family_transfer", 0.0)),
                "scale_consistency": _safe_float(row.get("scale_consistency", 0.0)),
                "scale_supported_families": list(row.get("scale_supported_families", [])),
                "scale_supported_family_count": int(row.get("scale_supported_family_count", 0) or 0),
            }
        )
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_label(method) for method in methods],
        col_labels=[label for _, label in metric_names],
        title="",
        xlabel="Model Generalization",
        ylabel="Method",
        output_dir=output_dir,
        stem=stem,
        cmap="GnBu",
        data=payload,
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def _remove_duplicate_leaderboard_sidecars(output_dir: Path, prefix: str) -> None:
    for stem in (
        f"{prefix}_overall_leaderboard",
        f"{prefix}_method_model_leaderboard",
        f"{prefix}_method_master_leaderboard",
        f"{prefix}_reference_code_method_model_leaderboard",
        f"{prefix}_reference_code_method_master_leaderboard",
        f"{prefix}_upstream_only_leaderboard",
    ):
        for suffix in (".json", ".csv"):
            path = output_dir / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def _update_export_identity_figure_hashes(*, export_identity_path: Path, output_dir: Path, prefix: str) -> None:
    identity = _load_object(export_identity_path)
    figure_hashes: dict[str, str] = {}
    for stem in (
        f"{prefix}_score_decomposition",
        f"{prefix}_detection_vs_utility",
    ):
        for suffix in REQUIRED_EXPORT_IDENTITY_FIGURE_SUFFIXES:
            path = output_dir / f"{stem}{suffix}"
            if not path.exists():
                raise SystemExit(
                    f"Figure-only redraw requires {path} to exist before the export identity is finalized."
                )
            figure_hashes[path.name] = _sha256(path)
    identity["required_figure_hashes"] = figure_hashes
    export_identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    table_dir = args.table_dir.resolve()
    output_dir = args.output_dir.resolve()
    export_identity_path = args.export_identity.resolve()
    method_summary_rows = _load_rows(table_dir / "method_summary.json")
    method_master_rows = _load_rows(table_dir / "suite_all_models_methods_method_master_leaderboard.json")
    _require_export_identity(
        export_identity_path=export_identity_path,
        table_dir=table_dir,
        prefix=args.prefix,
        method_summary_rows=method_summary_rows,
        method_master_rows=method_master_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _, plt = rpf.configure_matplotlib(require_times_new_roman=args.require_times_new_roman)

    rpf.plot_score_decomposition(
        plt,
        method_master_rows,
        output_dir=output_dir,
        prefix=args.prefix,
        paper_track=rpf.GENERATION_TIME_TRACK,
        suite_balanced=True,
    )
    rpf.plot_detection_vs_utility(
        plt,
        method_master_rows,
        output_dir=output_dir,
        prefix=args.prefix,
        paper_track=rpf.GENERATION_TIME_TRACK,
    )
    _remove_duplicate_leaderboard_sidecars(output_dir, args.prefix)
    _update_export_identity_figure_hashes(
        export_identity_path=export_identity_path,
        output_dir=output_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
