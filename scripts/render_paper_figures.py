from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from pathlib import Path
from statistics import mean
from math import ceil
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SINGLE_COLUMN_WIDTH = 3.35
# Keep an explicit double-column constant for the few diagnostic cases that may
# need wider canvases, but default all active publication figures to single-column-
# first layouts with per-figure faceting when density grows.
DOUBLE_COLUMN_WIDTH = 6.85
DOUBLE_COLUMN_TALL = 4.8

from posteditbench.leaderboards import GENERATION_TIME_TRACK, REFERENCE_CODE_TRACK
from posteditbench.leaderboards import (
    build_reference_track_master_leaderboard,
    build_reference_track_model_leaderboard,
    build_method_master_leaderboard,
    collect_report_rows,
    build_suite_method_master_leaderboard,
    build_suite_method_model_leaderboard,
    build_track_method_model_leaderboard,
    build_upstream_only_leaderboard,
)
from posteditbench.report import summarize_rows
from posteditbench.scorecard import HEADLINE_SCORE_FIELD, scorecard_for_rows
from posteditbench.suite import (
    OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
    SUITE_AGGREGATE_SOURCE_GROUPS,
    SUITE_AGGREGATE_SOURCES,
    SUITE_MODEL_ROSTER,
    normalize_source_group,
    suite_source_by_group,
)

PUBLIC_HEADLINE_METRIC = "PostEditScore"
_CANONICAL_SUITE_MANIFEST = "configs/matrices/suite_all_models_methods.json"
_CANONICAL_SUITE_PROFILE = "suite_all_models_methods"
_CANONICAL_SUITE_MANIFEST_DIGEST = hashlib.sha256((ROOT / _CANONICAL_SUITE_MANIFEST).read_bytes()).hexdigest()
_FORMAL_CANONICAL_EXECUTION_MODE = "single_host_canonical"
_EXPECTED_SHARDED_EXECUTION_MODE = "sharded_identical_execution_class"
_KNOWN_EXECUTION_MODES = {_FORMAL_CANONICAL_EXECUTION_MODE, _EXPECTED_SHARDED_EXECUTION_MODE}
_SCORE_DECOMPOSITION_SEMANTICS = "multiplicative_headline_component_decomposition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-ready figures with publication-safe typography.")
    parser.add_argument("--manifest", action="append", type=Path, default=None, help="One or more collection manifest JSON paths.")
    parser.add_argument("--report", action="append", type=Path, default=None, help="One or more report.json paths.")
    parser.add_argument("--baseline-eval", action="append", type=Path, default=None, help="One or more baseline evaluation JSON paths.")
    parser.add_argument(
        "--matrix-index",
        type=Path,
        default=None,
        help="run_full_matrix index to aggregate from. Required for suite=all publication-facing figure exports.",
    )
    parser.add_argument("--anchor-report", type=Path, default=None, help="Explicit anchor report for single-run paper figures.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for rendered figures.")
    parser.add_argument("--prefix", type=str, default="paper", help="Filename prefix for generated figures.")
    parser.add_argument("--suite", choices=("basic", "all"), default="basic", help="Figure suite to render.")
    parser.add_argument(
        "--paper-track",
        choices=(GENERATION_TIME_TRACK, REFERENCE_CODE_TRACK),
        default=GENERATION_TIME_TRACK,
        help="Track to treat as paper-safe input. Mixed-track inputs fail unless explicitly allowed.",
    )
    parser.add_argument("--allow-mixed-tracks", action="store_true", help="Allow diagnostic mixed-track figure exports.")
    parser.add_argument(
        "--include-reference-artifacts",
        action="store_true",
        help="Export reference-code leaderboard artifacts alongside paper-track outputs.",
    )
    parser.add_argument(
        "--require-times-new-roman",
        dest="require_times_new_roman",
        action="store_true",
        default=True,
        help="Fail if Times New Roman is unavailable in the Matplotlib font registry.",
    )
    parser.add_argument(
        "--allow-font-fallback",
        dest="require_times_new_roman",
        action="store_false",
        help="Allow serif fallback fonts instead of failing closed on missing Times New Roman.",
    )
    return parser.parse_args()


def configure_matplotlib(*, require_times_new_roman: bool = False) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times New Roman PS MT",
                "Times",
                "Nimbus Roman No9 L",
                "DejaVu Serif",
            ],
            "font.size": 10.4,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.6,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "legend.fontsize": 9.2,
            "figure.titlesize": 10.8,
            "axes.titlepad": 6.0,
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if require_times_new_roman:
        available_fonts = {entry.name for entry in font_manager.fontManager.ttflist}
        if "Times New Roman" not in available_fonts:
            raise RuntimeError("Times New Roman is not available in the current Matplotlib font registry.")
    return matplotlib, plt


_SHORT_LABELS = {
    "stone_runtime": "STONE",
    "sweet_runtime": "SWEET",
    "ewd_runtime": "EWD",
    "kgw_runtime": "KGW",
}

_ATTACK_LABELS = {
    "block_shuffle": "Shuffle",
    "comment_strip": "Comm",
    "identifier_rename": "Rename",
    "whitespace_normalize": "WS",
    "control_flow_flatten": "C-Flow",
    "budgeted_adaptive": "Budget",
    "noise_insert": "Noise",
}

_METHOD_COLORS = {
    "stone_runtime": "#5B7B9A",
    "sweet_runtime": "#3D7C67",
    "ewd_runtime": "#B85C38",
    "kgw_runtime": "#7A4E9D",
    "STONE": "#5B7B9A",
    "SWEET": "#3D7C67",
    "EWD": "#B85C38",
    "KGW": "#7A4E9D",
}

_METRIC_COLORS = {
    "detection_separability": "#355C7D",
    "robustness": "#4F7C82",
    "utility": "#2A9D8F",
    "stealth": "#7A8FA6",
    "efficiency": "#8F6C3E",
    "core_score": "#B85C75",
    "gate": "#4C566A",
    "generalization": "#7B6D8D",
}

_LANGUAGE_COLORS = {
    "python": "#355C7D",
    "cpp": "#6C5B7B",
    "java": "#C06C84",
    "javascript": "#F4A261",
    "go": "#2A9D8F",
}

_SOURCE_RELEASE_LABELS = {
    "public_humaneval_plus": "HE+",
    "public_mbpp_plus": "MBPP+",
    "public_humaneval_x": "HEX",
    "public_mbxp_5lang": "MBXP",
    "crafted_original": "Orig.",
    "crafted_translation": "Trans.",
    "crafted_stress": "Stress",
}

_MODEL_COMPACT_LABELS = {
    "Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen14",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen7",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "Qwen1.5",
    "bigcode/starcoder2-7b": "Star2-7B",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "DeepSeek",
}

_MODEL_AXIS_LABELS = {
    "Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen\n14B",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen\n7B",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "Qwen\n1.5B",
    "bigcode/starcoder2-7b": "Star2\n7B",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "DeepSeek\n6.7B",
}


def _paper_label(value: str) -> str:
    normalized = str(value).strip()
    return _SHORT_LABELS.get(normalized, normalized)


def _paper_attack_label(value: str) -> str:
    normalized = str(value).strip()
    return _ATTACK_LABELS.get(normalized, normalized)


def _paper_source_release_label(source_group: str) -> str:
    normalized = normalize_source_group(source_group)
    return _SOURCE_RELEASE_LABELS.get(normalized, _suite_source_label(normalized))


def _paper_model_label(model: str) -> str:
    normalized = str(model).strip()
    return _MODEL_COMPACT_LABELS.get(normalized, normalized.replace("Qwen/Qwen2.5-Coder-", "Qwen ").replace("-Instruct", "").replace("bigcode/", "").replace("deepseek-ai/", ""))


def _paper_model_axis_label(model: str) -> str:
    normalized = str(model).strip()
    return _MODEL_AXIS_LABELS.get(normalized, _paper_model_label(normalized))


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _method_color(method: str, origin: str = "") -> str:
    normalized = str(method).strip()
    if normalized in _METHOD_COLORS:
        return _METHOD_COLORS[normalized]
    return "#274c77" if str(origin).strip() == "upstream" else "#c1121f"


def _score_axis_limit(values: Iterable[float]) -> float:
    numeric = [_safe_float(value) for value in values]
    if not numeric:
        return 0.15
    upper = max(numeric)
    if upper <= 0.12:
        return 0.15
    if upper <= 0.2:
        return 0.25
    if upper <= 0.35:
        return 0.4
    if upper <= 0.5:
        return 0.6
    if upper <= 0.7:
        return 0.8
    return 1.0


def _rate_axis_limit(values: Iterable[float]) -> float:
    numeric = [_safe_float(value) for value in values]
    if not numeric:
        return 0.1
    upper = max(numeric)
    if upper <= 0.05:
        return 0.05
    if upper <= 0.1:
        return 0.1
    if upper <= 0.2:
        return 0.2
    if upper <= 0.4:
        return 0.4
    if upper <= 0.6:
        return 0.6
    if upper <= 0.8:
        return 0.8
    return 1.0


def _track_title_prefix(track: str) -> str:
    normalized = str(track).strip().lower()
    if normalized == REFERENCE_CODE_TRACK:
        return "Reference-Code"
    return "Generation-Time"


def _report_track_set(report: dict[str, Any]) -> set[str]:
    summary = dict(report.get("summary", {}))
    tracks = summary.get("evaluation_tracks", [])
    if isinstance(tracks, list) and tracks:
        return {str(track).strip() for track in tracks if str(track).strip()}
    track = str(summary.get("paper_primary_track", "")).strip()
    return {track} if track else set()


def _paper_track_ready(report: dict[str, Any]) -> bool:
    summary = dict(report.get("summary", {}))
    ready = summary.get("paper_track_ready")
    if ready is not None:
        return bool(ready)
    return len(_report_track_set(report)) == 1


def _paper_safe_reports(
    reports: list[dict[str, Any]],
    *,
    paper_track: str,
    allow_mixed_tracks: bool,
) -> list[dict[str, Any]]:
    if not reports:
        return []
    if allow_mixed_tracks:
        return [report for report in reports if paper_track in _report_track_set(report)]

    track_sets = [_report_track_set(report) for report in reports]
    if any(len(track_set) != 1 for track_set in track_sets):
        raise ValueError("paper figure rendering requires reports with exactly one evaluation track each")
    unique_tracks = sorted({next(iter(track_set)) for track_set in track_sets if track_set})
    if unique_tracks != [paper_track]:
        raise ValueError(
            f"paper figure rendering requires a single explicit track '{paper_track}', got {unique_tracks}"
        )
    if not all(_paper_track_ready(report) for report in reports):
        raise ValueError("paper figure rendering requires reports marked paper_track_ready=true")
    return list(reports)


def _anchor_report(
    paper_reports: list[dict[str, Any]],
    *,
    resolved_report_paths: list[Path],
    anchor_report_path: Path | None,
) -> dict[str, Any] | None:
    if not paper_reports:
        return None
    if anchor_report_path is not None:
        anchor = _load_json(_resolve_report_path(anchor_report_path))
        if anchor not in paper_reports:
            raise ValueError("anchor report does not belong to the selected paper-track report set")
        return anchor
    if len(paper_reports) == 1:
        return paper_reports[0]
    report_labels = [str(path) for path in resolved_report_paths]
    raise ValueError(
        "multiple paper-track reports were provided; pass --anchor-report to choose the report used for single-run figures. "
        + f"Candidates: {report_labels}"
    )


def _optional_anchor_report(
    paper_reports: list[dict[str, Any]],
    *,
    resolved_report_paths: list[Path],
    anchor_report_path: Path | None,
) -> dict[str, Any] | None:
    if anchor_report_path is not None or len(paper_reports) <= 1:
        return _anchor_report(
            paper_reports,
            resolved_report_paths=resolved_report_paths,
            anchor_report_path=anchor_report_path,
        )
    return None


def _required_suite_source_groups() -> set[str]:
    return {
        normalize_source_group(source_group)
        for source_group in SUITE_AGGREGATE_SOURCE_GROUPS
        if normalize_source_group(source_group)
    }


def _validate_suite_atomic_rows(rows: list[Any]) -> None:
    required = _required_suite_source_groups()
    present = {
        normalize_source_group(getattr(row, "source_group", ""))
        for row in rows
        if normalize_source_group(getattr(row, "source_group", ""))
    }
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"suite aggregate rows are missing atomic source groups: {missing}")


def _validate_suite_atomic_leaderboard(leaderboard: list[dict[str, Any]]) -> None:
    required = _required_suite_source_groups()
    if not leaderboard:
        raise ValueError("suite aggregate leaderboard is empty")
    for entry in leaderboard:
        comparable = {
            normalize_source_group(value)
            for value in entry.get("comparable_source_groups", []) or []
            if normalize_source_group(value)
        }
        coverage = dict(entry.get("score_coverage", {}) or {})
        aggregated = {
            normalize_source_group(value)
            for value in coverage.get("aggregated_source_groups", []) or []
            if normalize_source_group(value)
        }
        missing = sorted(required - comparable)
        if missing:
            raise ValueError(
                f"suite aggregate lost atomic source groups for {entry.get('method', 'method')}: {missing}"
            )
        if aggregated and aggregated != required:
            raise ValueError(
                f"suite aggregate score coverage is incomplete for {entry.get('method', 'method')}: "
                f"expected {sorted(required)}, got {sorted(aggregated)}"
            )


def _expected_models_for_track(report_paths: list[Path], *, paper_track: str) -> list[str]:
    expected: set[str] = set()
    for path in report_paths:
        if not path.exists():
            continue
        report = _load_json(path)
        if paper_track not in _report_track_set(report):
            continue
        expected.update(_report_models(report))
    return sorted(model for model in expected if model)


def _source_balanced_functional_summary(rows: list[Any], *, paper_track: str, report_count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        source_group = normalize_source_group(getattr(row, "source_group", ""))
        if source_group:
            grouped.setdefault(source_group, []).append(row)
    ordered_groups = [group for group in _required_suite_source_groups() if group in grouped]
    source_summaries = {group: summarize_rows(grouped[group], benchmark_manifest={}) for group in ordered_groups}
    series_palette: dict[str, str] = {}
    series_by_label: dict[str, list[dict[str, Any]]] = {}
    for summary in source_summaries.values():
        for label, metrics, color in _functional_series(summary):
            series_palette.setdefault(label, color)
            series_by_label.setdefault(label, []).append(dict(metrics))
    if not series_by_label:
        return []
    metric_specs = _common_functional_metric_specs(
        [(label, metrics_list[0], series_palette[label]) for label, metrics_list in series_by_label.items() if metrics_list]
    )
    if not metric_specs:
        return []
    rows_payload: list[dict[str, Any]] = []
    for label, metric_payloads in series_by_label.items():
        for metric_name, metric_label in metric_specs:
            values = [float(metrics.get(metric_name, 0.0)) for metrics in metric_payloads if metric_name in metrics]
            rows_payload.append(
                {
                    "series": label,
                    "metric": metric_label,
                    "value": round(mean(values), 4) if values else 0.0,
                    "aggregation_mode": "suite_source_aggregate",
                    "paper_track": paper_track,
                    "row_count": len(rows),
                    "report_count": report_count,
                    "source_groups": ordered_groups,
                }
            )
    return rows_payload


def _expected_models_from_matrix_index(matrix_index_path: Path | None) -> list[str]:
    if matrix_index_path is None or not matrix_index_path.exists():
        return []
    payload = _load_json(matrix_index_path)
    expected: list[str] = []
    for run in payload.get("runs", []):
        if str(run.get("resource", "")).strip().lower() != "gpu":
            continue
        status = str(run.get("status", "")).strip().lower()
        if status and status not in {"running", "success", "skipped"}:
            continue
        model = str(run.get("effective_model", "")).strip()
        if model and model not in expected:
            expected.append(model)
    return expected


def _validate_suite_model_roster(leaderboard: list[dict[str, Any]], *, expected_models: Iterable[str]) -> None:
    expected = [str(model).strip() for model in expected_models if str(model).strip()]
    if not expected:
        return
    present = {str(entry.get("model", "")).strip() for entry in leaderboard if str(entry.get("model", "")).strip()}
    missing = sorted(model for model in expected if model not in present)
    if missing:
        raise ValueError(f"suite aggregate model leaderboard is missing expected models: {missing}")


def _suite_atomic_rows_for_track(reports: list[dict[str, Any]], *, paper_track: str) -> list[Any]:
    rows = collect_report_rows(
        reports,
        track=paper_track,
        allowed_source_groups=SUITE_AGGREGATE_SOURCE_GROUPS,
        dedupe=True,
        collapse_cross_source_overlaps=True,
    )
    if rows:
        _validate_suite_atomic_rows(rows)
    return rows


def _suite_source_label(source_group: str) -> str:
    spec = suite_source_by_group(source_group)
    if spec is None:
        return str(source_group).strip()
    label = str(spec.dataset_label).strip()
    if "(" in label and ")" in label:
        return label
    if spec.slug in {"humaneval_x", "mbxp_5lang"}:
        languages = "/".join(language for language in OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES)
        return f"{label} ({languages})"
    return label


def _heatmap_height(row_count: int, *, base: float = 2.9, per_row: float = 0.34, floor: float = 3.2) -> float:
    return max(floor, base + max(0, row_count - 4) * per_row)


def _plot_heatmap(
    plt,
    *,
    matrix: list[list[float]],
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    x_labels: list[str] | None = None,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    output_dir: Path,
    stem: str,
    prefix: str = "",
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar_label: str = "Score",
    data: Any = None,
    max_columns_per_panel: int = 5,
    annotate: bool = False,
    annotation_fmt: str = "{:.2f}",
) -> list[Path]:
    row_labels = list(row_labels or y_labels or [])
    col_labels = list(col_labels or x_labels or [])
    if not matrix or not row_labels or not col_labels:
        return []
    panel_size = max(1, min(int(max_columns_per_panel), len(col_labels)))
    panels = [(start, min(len(col_labels), start + panel_size)) for start in range(0, len(col_labels), panel_size)]
    fig_width = max(SINGLE_COLUMN_WIDTH, min(DOUBLE_COLUMN_WIDTH, 1.25 + (0.62 * panel_size)))
    fig_height = _heatmap_height(len(row_labels), base=2.5, per_row=0.28, floor=2.9) * len(panels)
    fig_height += max(0, len(panels) - 1) * 0.55
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=False,
    )
    images = []
    for panel_index, (start, end) in enumerate(panels):
        ax = axes[panel_index][0]
        subset_labels = col_labels[start:end]
        subset_matrix = [row[start:end] for row in matrix]
        image = ax.imshow(subset_matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        images.append(image)
        if title:
            ax.set_title(title if panel_index == 0 else f"{title} (cont.)")
        ax.set_xlabel(xlabel if panel_index == len(panels) - 1 else "")
        ax.set_ylabel(ylabel if ylabel else "")
        ax.set_xticks(list(range(len(subset_labels))), subset_labels)
        ax.set_yticks(list(range(len(row_labels))), row_labels)
        max_label_length = max((len(str(label).replace("\n", "")) for label in subset_labels), default=0)
        has_multiline_labels = any("\n" in str(label) for label in subset_labels)
        ax.tick_params(axis="x", rotation=18 if max_label_length > 10 and not has_multiline_labels else 0)
        if annotate:
            threshold = vmin + (vmax - vmin) * 0.58 if vmax > vmin else vmax
            for row_index, row in enumerate(subset_matrix):
                for column_index, value in enumerate(row):
                    color = "white" if float(value) >= threshold and vmax > vmin else "#102A43"
                    ax.text(
                        column_index,
                        row_index,
                        annotation_fmt.format(value),
                        ha="center",
                        va="center",
                        fontsize=8.4,
                        color=color,
                    )
    fig.subplots_adjust(right=0.88, hspace=0.42)
    colorbar = fig.colorbar(images[-1], ax=axes[:, 0], fraction=0.035, pad=0.02)
    if colorbar_label:
        colorbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=12)
    if prefix:
        stem = f"{prefix}_{stem}"
    return _save_figure(fig, output_dir, stem, data=data)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_report_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "report.json"
        if candidate.exists():
            return candidate
    return path


def _report_label(report: dict[str, Any]) -> str:
    config = dict(report.get("config", {}))
    metadata = dict(config.get("metadata", {}))
    project = dict(metadata.get("project", {}))
    provider_summary = dict(metadata.get("provider_summary", {}))
    provider_model = str(provider_summary.get("provider_model", "")).strip()
    if provider_model:
        return provider_model
    name = str(project.get("name", "")).strip()
    if name:
        return name.replace("posteditbench-", "")
    watermark = str(config.get("watermark_name", "")).strip()
    if watermark:
        return watermark
    return "run"


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


_PRESENTATION_DATASET_LABELS = {
    "CraftedOriginal": "Crafted Original",
    "CraftedTranslation": "Crafted Translation",
    "CraftedStress": "Crafted Stress",
    "HumanEval-X": "HumanEval-X (5-language balanced slice)",
    "MBXP 5-language subset": "MBXP-5lang (5-language balanced slice)",
    "MBXP-5lang": "MBXP-5lang (5-language balanced slice)",
}


def _presentation_text(value: str) -> str:
    normalized = str(value)
    for raw, pretty in _PRESENTATION_DATASET_LABELS.items():
        if raw == pretty:
            continue
        if pretty in normalized:
            continue
        normalized = normalized.replace(raw, pretty)
    return normalized


def _presentation_dataset_label(value: Any) -> Any:
    raw = str(value).strip()
    if not raw:
        return value
    return _PRESENTATION_DATASET_LABELS.get(raw, _presentation_text(raw))


def _presentation_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if isinstance(normalized.get("datasets"), list):
        normalized["datasets"] = [_presentation_dataset_label(value) for value in normalized["datasets"]]
    elif isinstance(normalized.get("datasets"), str):
        normalized["datasets"] = _presentation_text(normalized["datasets"])
    if isinstance(normalized.get("dataset"), str):
        normalized["dataset"] = _presentation_dataset_label(normalized["dataset"])
    return normalized


def _write_rows_artifacts(output_dir: Path, stem: str, rows: list[dict[str, Any]]) -> list[Path]:
    rows = [_presentation_row(row) for row in rows]
    outputs: list[Path] = []
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    outputs.append(json_path)
    csv_path = output_dir / f"{stem}.csv"
    _write_rows_csv(csv_path, rows)
    outputs.append(csv_path)
    return outputs


def _normalize_figure_data(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [_presentation_row(dict(item)) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if all(not isinstance(value, dict) for value in data.values()):
            return [_presentation_row({"key": key, "value": value}) for key, value in data.items()]
        rows: list[dict[str, Any]] = []
        for key, value in data.items():
            if isinstance(value, dict):
                rows.append(_presentation_row({"key": key, **value}))
            else:
                rows.append(_presentation_row({"key": key, "value": value}))
        return rows
    return [{"value": data}]


def _save_figure(fig, output_dir: Path, stem: str, *, data: Any = None) -> list[Path]:
    outputs: list[Path] = []
    constrained_layout = False
    try:
        constrained_layout = bool(fig.get_constrained_layout())
    except Exception:
        constrained_layout = False
    if not constrained_layout:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                fig.tight_layout(pad=0.45)
            except Exception:
                pass
    for suffix in (".pdf", ".png"):
        path = output_dir / f"{stem}{suffix}"
        save_kwargs = {"pad_inches": 0.08}
        if not constrained_layout:
            save_kwargs["bbox_inches"] = "tight"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="constrained_layout not applied.*collapsed to zero.*",
                category=UserWarning,
            )
            fig.savefig(path, **save_kwargs)
        outputs.append(path)
    if data is not None:
        normalized_data = _normalize_figure_data(data)
        json_path = output_dir / f"{stem}.json"
        json_path.write_text(json.dumps(normalized_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        outputs.append(json_path)
        csv_path = output_dir / f"{stem}.csv"
        _write_rows_csv(csv_path, normalized_data)
        outputs.append(csv_path)
    try:
        import matplotlib.pyplot as _plt

        _plt.close(fig)
    except Exception:
        pass
    return outputs


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


def _annotate_bars(ax, bars, *, fmt: str = "{:.0f}", max_group_size: int = 4, annotation_budget: int = 8) -> None:
    bars = list(bars)
    if len(bars) > max_group_size:
        return
    used_annotations = int(getattr(ax, "_posteditbench_annotation_count", 0))
    if used_annotations + len(bars) > int(annotation_budget):
        return
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    setattr(ax, "_posteditbench_annotation_count", used_annotations + len(bars))


def _adaptive_track_figure_height(item_count: int, *, base: float = 3.7, per_item: float = 0.28, floor: float | None = None) -> float:
    minimum = base if floor is None else floor
    if item_count <= 5:
        return max(base, minimum)
    return max(minimum, base + (item_count - 5) * per_item)


def _attack_quality_metric(attack_summary: dict[str, Any]) -> float:
    return float(attack_summary.get("mean_quality_score", attack_summary.get("avg_quality", 0.0)))


def _functional_series(summary: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    candidates = [
        ("Clean", dict(summary.get("clean_functional_metrics", {})), "#274c77"),
        ("Watermarked", dict(summary.get("watermarked_functional_metrics", {})), "#6096ba"),
        ("Attacked", dict(summary.get("attacked_functional_metrics", {})), "#c1121f"),
    ]
    return [(label, metrics, color) for label, metrics, color in candidates if metrics]


def _common_functional_metric_specs(series: list[tuple[str, dict[str, Any], str]]) -> list[tuple[str, str]]:
    if not series:
        return []
    metric_specs = [
        ("compile_success_rate", "Compile"),
        ("test_pass_rate", "Pass"),
        ("pass@1", "Pass@1"),
    ]
    common: list[tuple[str, str]] = []
    for key, label in metric_specs:
        if all(key in metrics for _, metrics, _ in series):
            common.append((key, label))
    if common:
        return common
    fallback = [("compile_success_rate", "Compile"), ("test_pass_rate", "Pass")]
    return [(key, label) for key, label in fallback if any(key in metrics for _, metrics, _ in series)]


def _scatter_label_offset(index: int) -> tuple[int, int]:
    offsets = [(-24, 9), (8, 8), (8, -13), (-28, -12), (10, 16), (-20, 16)]
    return offsets[index % len(offsets)]


def _plot_functional_dotplot(
    plt,
    *,
    rows_payload: list[dict[str, Any]],
    title: str,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    if not rows_payload:
        return []
    by_series: dict[str, dict[str, float]] = {}
    metric_labels: list[str] = []
    for row in rows_payload:
        series = str(row["series"])
        metric = str(row["metric"])
        value = float(row["value"])
        by_series.setdefault(series, {})[metric] = value
        if metric not in metric_labels:
            metric_labels.append(metric)
    series_order = [label for label in ("Clean", "Watermarked", "Attacked") if label in by_series]
    if not series_order or not metric_labels:
        return []
    max_value = max((float(row["value"]) for row in rows_payload), default=0.0)
    axis_limit = _rate_axis_limit([max_value])
    fig_height = max(2.15, 1.85 + 0.28 * len(metric_labels))
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, fig_height), constrained_layout=True)
    y_positions = list(range(len(metric_labels)))
    palette = {"Clean": "#274c77", "Watermarked": "#6096ba", "Attacked": "#c1121f"}
    for position, metric_label in enumerate(metric_labels):
        x_values = [float(by_series[series_label].get(metric_label, 0.0)) for series_label in series_order]
        ax.plot(x_values, [position] * len(x_values), color="#d7dde5", linewidth=2.2, zorder=1)
        for series_label, x_value in zip(series_order, x_values):
            ax.scatter(
                x_value,
                position,
                s=48,
                color=palette[series_label],
                edgecolor="white",
                linewidth=0.8,
                label=series_label if position == 0 else None,
                zorder=3,
            )
    if title:
        ax.set_title(title)
    ax.set_xlabel("Rate")
    ax.set_ylabel("")
    ax.set_xlim(0.0, axis_limit)
    ax.set_yticks(y_positions, metric_labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=min(3, len(series_order)), loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0)
    return _save_figure(fig, output_dir, stem, data=rows_payload)


def _require_canonical_suite_identity_for_matrix_index(payload: dict[str, Any], *, label: str) -> None:
    manifest = str(payload.get("manifest", "")).strip()
    profile = str(payload.get("profile", "")).strip()
    run_count = int(payload.get("run_count", 0) or 0)
    if manifest != _CANONICAL_SUITE_MANIFEST or profile != _CANONICAL_SUITE_PROFILE or run_count != 140:
        raise ValueError(
            f"{label} requires the canonical suite identity "
            f"(manifest={_CANONICAL_SUITE_MANIFEST}, profile={_CANONICAL_SUITE_PROFILE}, run_count=140). "
            f"Observed manifest={manifest or '<missing>'}, profile={profile or '<missing>'}, run_count={run_count}."
        )
    canonical_manifest_digest = str(payload.get("canonical_manifest_digest", "")).strip()
    execution_mode = str(payload.get("execution_mode", "")).strip()
    if not canonical_manifest_digest:
        raise ValueError(f"{label} is missing canonical_manifest_digest.")
    if canonical_manifest_digest != _CANONICAL_SUITE_MANIFEST_DIGEST:
        raise ValueError(
            f"{label} canonical_manifest_digest mismatch: expected {_CANONICAL_SUITE_MANIFEST_DIGEST}, "
            f"observed {canonical_manifest_digest}."
        )
    if not execution_mode:
        raise ValueError(f"{label} is missing execution_mode.")
    if execution_mode not in _KNOWN_EXECUTION_MODES:
        raise ValueError(
            f"{label} execution_mode must be one of {sorted(_KNOWN_EXECUTION_MODES)}; observed {execution_mode}."
        )
    shard_profiles = payload.get("shard_profiles", [])
    if execution_mode != _EXPECTED_SHARDED_EXECUTION_MODE and shard_profiles:
        raise ValueError(
            f"{label} shard_profiles require execution_mode={_EXPECTED_SHARDED_EXECUTION_MODE}; observed {execution_mode}."
        )
    if execution_mode == _EXPECTED_SHARDED_EXECUTION_MODE:
        code_snapshot_digest = str(payload.get("code_snapshot_digest", "")).strip()
        if not code_snapshot_digest:
            raise ValueError(f"{label} is missing code_snapshot_digest.")
        execution_environment_fingerprint = str(payload.get("execution_environment_fingerprint", "")).strip()
        if not execution_environment_fingerprint:
            raise ValueError(f"{label} is missing execution_environment_fingerprint.")


def _require_formal_single_host_execution_mode(payload: dict[str, Any], *, label: str) -> None:
    execution_mode = str(payload.get("execution_mode", "")).strip()
    if execution_mode != _FORMAL_CANONICAL_EXECUTION_MODE:
        raise ValueError(
            f"{label} requires execution_mode={_FORMAL_CANONICAL_EXECUTION_MODE}; observed {execution_mode or '<missing>'}."
        )
    code_snapshot_digest = str(payload.get("code_snapshot_digest", "")).strip()
    if not code_snapshot_digest:
        raise ValueError(f"{label} is missing code_snapshot_digest.")
    execution_environment_fingerprint = str(payload.get("execution_environment_fingerprint", "")).strip()
    if not execution_environment_fingerprint:
        raise ValueError(f"{label} is missing execution_environment_fingerprint.")
    source_modes = payload.get("assembly_source_execution_modes", [])
    source_indexes = payload.get("assembly_source_indexes", [])
    if bool(source_modes) != bool(source_indexes):
        raise ValueError(
            f"{label} must provide both assembly_source_execution_modes and assembly_source_indexes when publication "
            "provenance is assembled from multiple source indexes."
        )
    if not source_modes:
        raise ValueError(
            f"{label} must retain non-empty assembly_source_execution_modes and assembly_source_indexes for the "
            "publication-facing canonical single-host matrix."
        )
    if not isinstance(source_indexes, list) or len(source_indexes) != 1:
        raise ValueError(
            f"{label} must be backed by exactly one source index for the formal one-shot release path; "
            f"observed {len(source_indexes) if isinstance(source_indexes, list) else '<missing>'}."
        )
    normalized_source_modes = sorted({str(mode).strip() for mode in source_modes if str(mode).strip()})
    if normalized_source_modes != [_FORMAL_CANONICAL_EXECUTION_MODE]:
        raise ValueError(
            f"{label} must be backed only by {_FORMAL_CANONICAL_EXECUTION_MODE} source indexes; "
            f"observed {normalized_source_modes}."
        )


def _reports_from_matrix_index(matrix_index_path: Path) -> tuple[list[Path], list[Path]]:
    payload = _load_json(matrix_index_path)
    _require_canonical_suite_identity_for_matrix_index(payload, label="render_paper_figures matrix index")
    if str(payload.get("execution_mode", "")).strip() == _FORMAL_CANONICAL_EXECUTION_MODE:
        _require_formal_single_host_execution_mode(payload, label="render_paper_figures matrix index")
    raw_runs = [run for run in payload.get("runs", []) if isinstance(run, dict)]
    expected_count = int(payload.get("run_count", len(raw_runs)) or len(raw_runs))
    exportable_count = 0
    blocked: list[str] = []
    for run in raw_runs:
        run_id = str(run.get("run_id", "")).strip() or "<unknown>"
        status = str(run.get("status", "")).strip()
        reason = str(run.get("reason", "")).strip()
        report_path = str(run.get("report_path", "")).strip()
        if status == "success" and reason != "resume_existing_report" and report_path and Path(report_path).exists():
            exportable_count += 1
            continue
        if status == "success" and reason == "resume_existing_report":
            blocked.append(f"{run_id}:resume_existing_report")
            continue
        blocked.append(f"{run_id}:{status or 'missing_status'}")
    if blocked or exportable_count != expected_count:
        preview = ", ".join(blocked[:8])
        if len(blocked) > 8:
            preview += ", ..."
        raise ValueError(
            "render_paper_figures requires a complete result-backed canonical matrix with only success runs. "
            f"exportable={exportable_count} expected={expected_count} blocked=[{preview}]"
        )
    report_paths: list[Path] = []
    baseline_paths: list[Path] = []
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        status = str(run.get("status", "")).strip()
        reason = str(run.get("reason", "")).strip()
        if status and status != "success":
            continue
        if reason == "resume_existing_report":
            continue
        report_path = str(run.get("report_path", "")).strip()
        if report_path:
            candidate = Path(report_path)
            if candidate.exists():
                report_paths.append(candidate)
        baseline_eval_path = str(run.get("baseline_eval_path", "")).strip()
        if baseline_eval_path:
            candidate = Path(baseline_eval_path)
            if candidate.exists():
                baseline_paths.append(candidate)
    return report_paths, baseline_paths


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for path in paths:
        seen[str(path)] = path
    return list(seen.values())


def _report_tracks(report: dict[str, Any]) -> list[str]:
    summary = dict(report.get("summary", {}))
    tracks = summary.get("evaluation_tracks", [])
    if isinstance(tracks, list):
        values = [str(track).strip() for track in tracks if str(track).strip()]
        if values:
            return values
    track = str(summary.get("paper_primary_track", "")).strip()
    return [track] if track else []


def _report_source_groups(report: dict[str, Any]) -> set[str]:
    summary = dict(report.get("summary", {}))
    groups = summary.get("by_source_group", {})
    if isinstance(groups, dict):
        return {str(key).strip() for key in groups if str(key).strip()}
    return set()


def _report_models(report: dict[str, Any]) -> set[str]:
    summary = dict(report.get("summary", {}))
    groups = summary.get("by_model_label", {})
    if isinstance(groups, dict):
        return {str(key).strip() for key in groups if str(key).strip()}
    return set()


def _baseline_eval_context(evaluation: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    tracks = tuple(sorted(str(item).strip() for item in evaluation.get("evaluation_tracks", []) if str(item).strip()))
    models = tuple(sorted(str(item).strip() for item in evaluation.get("model_labels", []) if str(item).strip()))
    sources = tuple(sorted(str(item).strip() for item in evaluation.get("source_groups", []) if str(item).strip()))
    return tracks, models, sources


def _baseline_context_label(evaluation: dict[str, Any]) -> str:
    datasets = sorted(str(item).strip() for item in evaluation.get("datasets", []) if str(item).strip())
    models = sorted(str(item).strip() for item in evaluation.get("model_labels", []) if str(item).strip())
    pieces: list[str] = []
    if datasets:
        pieces.append("/".join(datasets))
    if models:
        pieces.append("/".join(models))
    return " | ".join(pieces)


def _paper_safe_baseline_evaluations(
    evaluations: list[dict[str, Any]],
    *,
    paper_track: str,
) -> list[dict[str, Any]]:
    if not evaluations:
        return []
    if len(evaluations) == 1:
        tracks, _, _ = _baseline_eval_context(evaluations[0])
        if tracks and paper_track not in tracks:
            return []
        return evaluations
    contexts = {_baseline_eval_context(evaluation) for evaluation in evaluations}
    if any(not any(component for component in context) for context in contexts):
        return []
    if len(contexts) != 1:
        return []
    tracks, _, _ = next(iter(contexts))
    if tracks and paper_track not in tracks:
        return []
    return evaluations


def _select_paper_anchor_report(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not reports:
        return None

    def _priority(report: dict[str, Any]) -> tuple[int, int, int, int, int]:
        config = dict(report.get("config", {}))
        tracks = set(_report_tracks(report))
        source_groups = _report_source_groups(report)
        models = _report_models(report)
        watermark_name = str(config.get("watermark_name", "")).strip()
        provider_mode = str(config.get("provider_mode", "")).strip()
        return (
            0 if "generation_time" in tracks else 1,
            0 if watermark_name == "kgw_runtime" else 1,
            0 if "public_humaneval_plus" in source_groups else (1 if "public_mbpp_plus" in source_groups else 2),
            0 if "bigcode/starcoder2-7b" in models else 1,
            0 if provider_mode == "local_hf" else 1,
        )

    return min(reports, key=_priority)


def plot_language_coverage(plt, manifest: dict[str, Any], *, output_dir: Path, prefix: str) -> list[Path]:
    language_counts = dict(manifest.get("language_counts", {}))
    if not language_counts:
        return []
    languages = list(language_counts.keys())
    counts = [int(language_counts[language]) for language in languages]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.6), constrained_layout=True)
    bars = ax.bar(languages, counts, color=["#274c77", "#6096ba", "#a3cef1", "#8b8c89", "#d9dcd6"])
    ax.set_xlabel("Language")
    ax.set_ylabel("Task Count")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    _annotate_bars(ax, bars)
    data = [{"language": language, "count": count} for language, count in zip(languages, counts)]
    return _save_figure(fig, output_dir, f"{prefix}_language_coverage", data=data)


def plot_source_language_coverage(plt, manifests: dict[str, dict[str, Any]], *, output_dir: Path, prefix: str) -> list[Path]:
    if not manifests:
        return []
    ordered_sources = [source for source in SUITE_AGGREGATE_SOURCES if source.dataset_label in manifests]
    if not ordered_sources:
        return []
    present_languages = {
        str(language).strip().lower()
        for manifest in manifests.values()
        for language in dict(manifest.get("language_counts", {})).keys()
    }
    canonical_languages = ["python", "cpp", "java", "javascript", "go"]
    languages = [language for language in canonical_languages if language in present_languages]
    if not languages:
        return []
    matrix: list[list[float]] = []
    rows: list[dict[str, Any]] = []
    for source in ordered_sources:
        manifest = dict(manifests.get(source.dataset_label, {}))
        counts = {
            str(language).strip().lower(): int(count)
            for language, count in dict(manifest.get("language_counts", {})).items()
        }
        values = [float(counts.get(language, 0)) for language in languages]
        matrix.append(values)
        rows.extend(
            {
                "source_group": source.source_group,
                "source_label": source.dataset_label,
                "language": language,
                "count": int(counts.get(language, 0)),
            }
            for language in languages
        )
    max_count = max((max(values) for values in matrix), default=0.0)
    return _plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_paper_source_release_label(source.source_group) for source in ordered_sources],
        col_labels=[language.title() for language in languages],
        title="",
        xlabel="Language",
        ylabel="",
        output_dir=output_dir,
        stem=f"{prefix}_source_language_coverage",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=max(1.0, max_count),
        colorbar_label="Release Records",
        data=rows,
        annotate=True,
        annotation_fmt="{:.0f}",
    )


def _suite_atomic_source_manifests() -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for source in SUITE_AGGREGATE_SOURCES:
        manifest_path = (ROOT / source.prepared_output).with_suffix(".manifest.json")
        if manifest_path.exists():
            manifests[source.dataset_label] = _load_json(manifest_path)
    return manifests


def plot_attack_robustness(plt, report: dict[str, Any], *, output_dir: Path, prefix: str) -> list[Path]:
    summary = dict(report.get("summary", {}))
    attack_view = dict(summary.get("by_attack", {}))
    if not attack_view:
        return []
    attacks = sorted(attack_view)
    attack_labels = [_paper_attack_label(attack) for attack in attacks]
    detect_rates = [float(attack_view[attack].get("attacked_detect_rate", 0.0)) for attack in attacks]
    quality = [_attack_quality_metric(dict(attack_view[attack])) for attack in attacks]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.8), constrained_layout=True)
    bars = ax.bar(attack_labels, detect_rates, color="#274c77", label="Attacked Detect Rate")
    ax.plot(attack_labels, quality, marker="o", linewidth=2.2, color="#c1121f", label="Mean Quality")
    ax.set_xlabel("Attack")
    ax.set_ylabel("Rate / Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.tick_params(axis="x", rotation=30)
    _annotate_bars(ax, bars, fmt="{:.2f}", max_group_size=0)
    ax.legend(frameon=False, loc="lower left")
    data = [
        {"attack": attack, "paper_label": label, "attacked_detect_rate": detect_rate, "mean_quality_score": quality_score}
        for attack, label, detect_rate, quality_score in zip(attacks, attack_labels, detect_rates, quality)
    ]
    return _save_figure(fig, output_dir, f"{prefix}_attack_robustness", data=data)


def plot_semantic_attack_robustness(plt, report: dict[str, Any], *, output_dir: Path, prefix: str) -> list[Path]:
    summary = dict(report.get("summary", {}))
    robustness = {key: value for key, value in dict(summary.get("semantic_attack_robustness", {})).items() if key != "overall"}
    if not robustness:
        return []
    attacks = sorted(robustness)
    attack_labels = [_paper_attack_label(attack) for attack in attacks]
    values = [float(robustness[attack]) for attack in attacks]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.7), constrained_layout=True)
    bars = ax.bar(attack_labels, values, color="#2a9d8f")
    ax.set_xlabel("Attack")
    ax.set_ylabel("Semantic Preservation Rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.tick_params(axis="x", rotation=30)
    _annotate_bars(ax, bars, fmt="{:.2f}", max_group_size=0)
    data = [{"attack": attack, "paper_label": label, "semantic_preservation_rate": value} for attack, label, value in zip(attacks, attack_labels, values)]
    return _save_figure(fig, output_dir, f"{prefix}_semantic_attack_robustness", data=data)


def plot_functional_summary(plt, report: dict[str, Any], *, output_dir: Path, prefix: str) -> list[Path]:
    summary = dict(report.get("summary", {}))
    series = _functional_series(summary)
    metric_specs = _common_functional_metric_specs(series)
    if not series or not metric_specs:
        return []
    rows_payload: list[dict[str, Any]] = []
    for series_label, metrics, _ in series:
        rows_payload.extend(
            {"series": series_label, "metric": label, "value": float(metrics.get(metric_name, 0.0))}
            for metric_name, label in metric_specs
        )
    return _plot_functional_dotplot(
        plt,
        rows_payload=rows_payload,
        title="",
        output_dir=output_dir,
        stem=f"{prefix}_functional_summary",
    )


def plot_suite_functional_summary(
    plt,
    reports: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str,
) -> list[Path]:
    rows = collect_report_rows(
        reports,
        track=paper_track,
        allowed_source_groups=SUITE_AGGREGATE_SOURCE_GROUPS,
        dedupe=True,
    )
    if not rows:
        return []
    _validate_suite_atomic_rows(rows)
    rows_payload = _source_balanced_functional_summary(rows, paper_track=paper_track, report_count=len(reports))
    if not rows_payload:
        return []
    return _plot_functional_dotplot(
        plt,
        rows_payload=rows_payload,
        title="",
        output_dir=output_dir,
        stem=f"{prefix}_functional_summary",
    )


def plot_reference_kind_comparison(plt, report: dict[str, Any], *, output_dir: Path, prefix: str) -> list[Path]:
    breakdown = dict(report.get("summary", {}).get("by_reference_kind", {}))
    if not breakdown:
        return []
    kinds = sorted(breakdown)
    detect = [float(breakdown[kind].get("attacked_detect_rate", 0.0)) for kind in kinds]
    quality = [float(breakdown[kind].get("mean_quality_score", 0.0)) for kind in kinds]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.6), constrained_layout=True)
    bars = ax.bar(kinds, detect, color="#274c77", label="Attacked Detect Rate")
    ax.plot(kinds, quality, marker="o", linewidth=2.2, color="#c1121f", label="Mean Quality")
    ax.set_xlabel("Reference Kind")
    ax.set_ylabel("Rate / Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    _annotate_bars(ax, bars, fmt="{:.2f}", max_group_size=0)
    ax.legend(frameon=False, loc="lower left")
    data = [
        {"reference_kind": kind, "attacked_detect_rate": detect_rate, "mean_quality_score": quality_score}
        for kind, detect_rate, quality_score in zip(kinds, detect, quality)
    ]
    return _save_figure(fig, output_dir, f"{prefix}_reference_kind_comparison", data=data)


def plot_budget_curve(plt, report: dict[str, Any], *, output_dir: Path, prefix: str) -> list[Path]:
    curves = dict(report.get("summary", {}).get("budget_curves", {}))
    budget_points = list(curves.get("budgeted_adaptive", []))
    if not budget_points:
        return []
    budgets = [int(point.get("budget", 0)) for point in budget_points]
    detector = [float(point.get("mean_detector_score", 0.0)) for point in budget_points]
    quality = [float(point.get("mean_quality_score", 0.0)) for point in budget_points]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.8), constrained_layout=True)
    ax.plot(budgets, detector, marker="o", linewidth=2.2, color="#274c77", label="Detector Score")
    ax.plot(budgets, quality, marker="s", linewidth=2.2, color="#c1121f", label="Quality Score")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")
    return _save_figure(fig, output_dir, f"{prefix}_budget_curve", data=budget_points)


def plot_by_language_metrics(plt, report: dict[str, Any], *, output_dir: Path, prefix: str) -> list[Path]:
    breakdown = dict(report.get("summary", {}).get("by_language", {}))
    if not breakdown:
        return []
    languages = sorted(breakdown)
    robustness = [float(breakdown[language].get("attacked_detect_rate", 0.0)) for language in languages]
    quality = [float(breakdown[language].get("mean_quality_score", 0.0)) for language in languages]
    semantics = [float(breakdown[language].get("semantic_preservation_rate", 0.0)) for language in languages]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.8), constrained_layout=True)
    width = 0.24
    positions = list(range(len(languages)))
    bars_1 = ax.bar([position - width for position in positions], robustness, width=width, label="Robustness", color="#274c77")
    bars_2 = ax.bar(positions, quality, width=width, label="Quality", color="#6096ba")
    bars_3 = ax.bar([position + width for position in positions], semantics, width=width, label="Semantic Preservation", color="#2a9d8f")
    ax.set_xlabel("Language")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(positions, languages)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    for bars in (bars_1, bars_2, bars_3):
        _annotate_bars(ax, bars, fmt="{:.2f}", max_group_size=0)
    data = [
        {
            "language": language,
            "attacked_detect_rate": robust,
            "mean_quality_score": quality_score,
            "semantic_preservation_rate": semantic_score,
        }
        for language, robust, quality_score, semantic_score in zip(languages, robustness, quality, semantics)
    ]
    return _save_figure(fig, output_dir, f"{prefix}_by_language_metrics", data=data)


def plot_baseline_comparison(
    plt,
    evaluations: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    if not evaluations:
        return []
    labels = []
    auroc = []
    pass_rates = []
    data: list[dict[str, Any]] = []
    for evaluation in evaluations:
        schemes = evaluation.get("watermark_schemes", [])
        label = ", ".join(schemes) if schemes else str(evaluation.get("baseline_family", "baseline"))
        labels.append(_paper_label(label))
        auroc_value = float(evaluation.get("clean_reference_vs_watermarked_auroc", 0.0))
        pass_rate = float(evaluation.get("watermarked_pass_rate", 0.0))
        auroc.append(auroc_value)
        pass_rates.append(pass_rate)
        data.append({"baseline": label, "paper_label": _paper_label(label), "clean_reference_vs_watermarked_auroc": auroc_value, "watermarked_pass_rate": pass_rate})
    context_label = _baseline_context_label(evaluations[0])
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, _adaptive_track_figure_height(len(labels), base=3.8, per_item=0.16, floor=3.8)),
        constrained_layout=True,
    )
    bars = ax.bar(labels, auroc, color="#274c77", label="AUROC")
    ax.plot(labels, pass_rates, marker="o", linewidth=2.2, color="#c1121f", label="Watermarked Pass Rate")
    ax.set_xlabel("Baseline")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    _annotate_bars(ax, bars, fmt="{:.2f}", max_group_size=0)
    ax.legend(frameon=False, loc="lower left")
    return _save_figure(fig, output_dir, f"{prefix}_baseline_comparison", data=data)


def plot_runtime_family_detailed(plt, evaluations: list[dict[str, Any]], *, output_dir: Path, prefix: str) -> list[Path]:
    rows = []
    for evaluation in evaluations:
        label = ", ".join(evaluation.get("watermark_schemes", [])) or str(evaluation.get("baseline_family", "baseline"))
        row = {
            "baseline": label,
            "paper_label": _paper_label(label),
            "auroc": float(evaluation.get("clean_reference_vs_watermarked_auroc", 0.0)),
            "pass_rate": float(evaluation.get("watermarked_pass_rate", 0.0)),
        }
        if evaluation.get("average_perplexity_watermarked") is not None:
            row["perplexity"] = float(evaluation.get("average_perplexity_watermarked", 0.0))
        if evaluation.get("stem_clean_reference") is not None:
            row["stem"] = float(evaluation.get("stem_clean_reference", 0.0))
        rows.append(row)
    if not rows or not any("perplexity" in row or "stem" in row for row in rows):
        return []
    labels = [str(row["paper_label"]) for row in rows]
    ppl = [float(row.get("perplexity", 0.0)) for row in rows]
    stem = [float(row.get("stem", 0.0)) for row in rows]
    context_label = _baseline_context_label(evaluations[0])
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, _adaptive_track_figure_height(len(labels), base=3.8, per_item=0.16, floor=3.8)),
        constrained_layout=True,
    )
    bars = ax.bar(labels, stem, color="#2a9d8f", label="STEM")
    ax.plot(labels, ppl, marker="o", linewidth=2.2, color="#c1121f", label="PPL")
    ax.set_xlabel("Baseline")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    _annotate_bars(ax, bars, fmt="{:.2f}", max_group_size=0)
    ax.legend(frameon=False, loc="upper right")
    return _save_figure(fig, output_dir, f"{prefix}_runtime_baseline_ppl_stem", data=rows)


def _suite_slice_score_rows(
    reports: list[dict[str, Any]],
    *,
    paper_track: str,
    axis: str,
) -> tuple[list[str], list[str], list[list[float]], list[dict[str, Any]]]:
    rows = collect_report_rows(
        reports,
        track=paper_track,
        allowed_source_groups=SUITE_AGGREGATE_SOURCE_GROUPS,
        dedupe=True,
    )
    if not rows:
        return [], [], [], []
    _validate_suite_atomic_rows(rows)
    methods = sorted({str(row.watermark_scheme).strip() for row in rows if str(row.watermark_scheme).strip()}, key=_paper_label)
    if axis == "source":
        slices = [source.dataset_label for source in SUITE_AGGREGATE_SOURCES]
        slice_rows = {source.dataset_label: [row for row in rows if normalize_source_group(getattr(row, "source_group", "")) == source.source_group] for source in SUITE_AGGREGATE_SOURCES}
        metric_name = HEADLINE_SCORE_FIELD
        metric_label = "slice_local_gate_core"
        include_generalization = False
    elif axis == "model":
        slices = sorted({str(row.model_label).strip() for row in rows if str(row.model_label).strip()})
        slice_rows = {label: [row for row in rows if str(row.model_label).strip() == label] for label in slices}
        metric_name = HEADLINE_SCORE_FIELD
        metric_label = "slice_local_gate_core"
        include_generalization = False
    elif axis == "language":
        slices = sorted({str(getattr(row, "language", "")).strip() for row in rows if str(getattr(row, "language", "")).strip()})
        slice_rows = {label: [row for row in rows if str(getattr(row, "language", "")).strip() == label] for label in slices}
        metric_name = "utility"
        metric_label = "utility"
        include_generalization = False
    elif axis == "attack":
        slices = [_paper_attack_label(attack) for attack in _ATTACK_LABELS]
        reverse = {v: k for k, v in _ATTACK_LABELS.items()}
        slice_rows = {label: [row for row in rows if str(getattr(row, "attack_name", "")).strip() == reverse[label]] for label in slices}
        metric_name = "robustness"
        metric_label = "robustness"
        include_generalization = False
    else:
        raise ValueError(f"unsupported suite slice axis: {axis}")

    data: list[dict[str, Any]] = []
    matrix: list[list[float]] = []
    for method in methods:
        row_values: list[float] = []
        method_rows = [row for row in rows if str(row.watermark_scheme).strip() == method]
        for slice_label in slices:
            subset = [row for row in method_rows if row in slice_rows[slice_label]]
            if not subset:
                value = 0.0
            else:
                scorecard = scorecard_for_rows(subset, include_generalization=include_generalization)
                value = float(scorecard.get(metric_name, 0.0))
            row_values.append(round(value, 4))
            data.append(
                {
                    "method": method,
                    "paper_label": _paper_label(method),
                    "slice": slice_label,
                    "metric": metric_label,
                    "value": round(value, 4),
                }
            )
        matrix.append(row_values)
    return [_paper_label(method) for method in methods], slices, matrix, data


def plot_suite_source_breakdown(plt, reports: list[dict[str, Any]], *, output_dir: Path, prefix: str, paper_track: str) -> list[Path]:
    methods, sources, matrix, data = _suite_slice_score_rows(reports, paper_track=paper_track, axis="source")
    return _plot_heatmap(
        plt,
        output_dir=output_dir,
        prefix=prefix,
        stem="per_source_breakdown",
        title="",
        x_labels=sources,
        y_labels=methods,
        matrix=matrix,
        colorbar_label="Slice-local Gate x Core",
        data=data,
        vmax=1.0,
    )


def plot_suite_model_breakdown(
    plt,
    leaderboard: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str,
) -> list[Path]:
    if not leaderboard:
        return []
    models = sorted({str(entry.get("model", "")).strip() for entry in leaderboard if str(entry.get("model", "")).strip()})
    methods = sorted({str(entry.get("method", "")).strip() for entry in leaderboard if str(entry.get("method", "")).strip()}, key=_paper_label)
    by_pair = {
        (str(entry.get("method", "")).strip(), str(entry.get("model", "")).strip()): float(entry.get(HEADLINE_SCORE_FIELD, 0.0))
        for entry in leaderboard
    }
    matrix: list[list[float]] = []
    data: list[dict[str, Any]] = []
    for method in methods:
        row_values: list[float] = []
        for model in models:
            value = round(float(by_pair.get((method, model), 0.0)), 4)
            row_values.append(value)
            data.append(
                {
                    "method": method,
                    "paper_label": _paper_label(method),
                    "model": model,
                    "model_label": _paper_model_label(model),
                    "metric": PUBLIC_HEADLINE_METRIC,
                    "value": value,
                }
            )
        matrix.append(row_values)
    return _plot_heatmap(
        plt,
        output_dir=output_dir,
        prefix=prefix,
        stem="per_model_breakdown",
        title="",
        x_labels=[_paper_model_label(model) for model in models],
        y_labels=[_paper_label(method) for method in methods],
        matrix=matrix,
        colorbar_label=PUBLIC_HEADLINE_METRIC,
        data=data,
        vmax=1.0,
    )


def plot_suite_language_breakdown(plt, reports: list[dict[str, Any]], *, output_dir: Path, prefix: str, paper_track: str) -> list[Path]:
    methods, languages, matrix, data = _suite_slice_score_rows(reports, paper_track=paper_track, axis="language")
    return _plot_heatmap(
        plt,
        output_dir=output_dir,
        prefix=prefix,
        stem="per_language_breakdown",
        title="",
        x_labels=languages,
        y_labels=methods,
        matrix=matrix,
        colorbar_label="Utility",
        data=data,
        vmax=1.0,
    )


def plot_suite_attack_breakdown(plt, reports: list[dict[str, Any]], *, output_dir: Path, prefix: str, paper_track: str) -> list[Path]:
    methods, attacks, matrix, data = _suite_slice_score_rows(reports, paper_track=paper_track, axis="attack")
    return _plot_heatmap(
        plt,
        output_dir=output_dir,
        prefix=prefix,
        stem="attack_robustness_breakdown",
        title="",
        x_labels=attacks,
        y_labels=methods,
        matrix=matrix,
        colorbar_label="Robustness",
        data=data,
        vmax=1.0,
    )


def export_leaderboards(
    reports: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str = GENERATION_TIME_TRACK,
    include_reference_artifacts: bool = False,
    all_reports: list[dict[str, Any]] | None = None,
    suite_atomic_only: bool = False,
) -> tuple[list[Path], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if suite_atomic_only:
        method_model = build_suite_method_model_leaderboard(reports, track=paper_track)
        method_master = build_suite_method_master_leaderboard(reports, track=paper_track)
        reference_master = build_reference_track_master_leaderboard(
            all_reports or reports,
            allowed_source_groups=SUITE_AGGREGATE_SOURCE_GROUPS,
            balance_by_source_group=True,
            collapse_cross_source_overlaps=True,
        )
        reference_model = build_reference_track_model_leaderboard(
            all_reports or reports,
            allowed_source_groups=SUITE_AGGREGATE_SOURCE_GROUPS,
            balance_by_source_group=True,
            collapse_cross_source_overlaps=True,
        )
        upstream_only = build_upstream_only_leaderboard(
            reports,
            allowed_source_groups=SUITE_AGGREGATE_SOURCE_GROUPS,
            balance_by_source_group=True,
            collapse_cross_source_overlaps=True,
        )
    else:
        method_model = build_track_method_model_leaderboard(reports, track=paper_track)
        method_master = build_method_master_leaderboard(reports, track=paper_track)
        reference_master = build_reference_track_master_leaderboard(all_reports or reports)
        reference_model = build_reference_track_model_leaderboard(all_reports or reports)
        upstream_only = build_upstream_only_leaderboard(reports)
    return [], method_model, method_master, reference_master


def plot_overall_leaderboard(
    plt,
    leaderboard: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str = GENERATION_TIME_TRACK,
    suite_balanced: bool = False,
) -> list[Path]:
    if not leaderboard:
        return []
    labels = [_paper_label(str(row.get("method", ""))) for row in leaderboard]
    scores = [float(row.get(HEADLINE_SCORE_FIELD, 0.0)) for row in leaderboard]
    colors = [_method_color(str(row.get("method", "")), str(row.get("origin", ""))) for row in leaderboard]
    axis_limit = _score_axis_limit(scores)
    fig, ax = plt.subplots(
        figsize=(max(SINGLE_COLUMN_WIDTH, 4.65), 2.85),
        constrained_layout=True,
    )
    positions = list(range(len(labels)))
    bars = ax.bar(positions, scores, color=colors, edgecolor="white", linewidth=0.8, width=0.62)
    ax.set_ylabel(PUBLIC_HEADLINE_METRIC)
    ax.set_xlabel("")
    ax.set_ylim(0.0, axis_limit)
    ax.set_xticks(positions, labels)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    if len(scores) <= 6:
        label_y_offset = max(0.008, axis_limit * 0.035)
        zero_marker_y = max(0.004, axis_limit * 0.03)
        for bar, score, color in zip(bars, scores, colors):
            x_center = bar.get_x() + bar.get_width() / 2.0
            if score <= 0.0:
                ax.scatter(
                    [x_center],
                [zero_marker_y],
                s=16,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                    zorder=3,
                )
            ax.text(
                x_center,
                (score + label_y_offset) if score > 0.0 else (zero_marker_y + label_y_offset),
                f"{score:.2f}",
                va="bottom",
                ha="center",
                fontsize=9,
                color="#243B53",
            )
    return _save_figure(fig, output_dir, f"{prefix}_overall_leaderboard")


def plot_score_decomposition(
    plt,
    leaderboard: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str = GENERATION_TIME_TRACK,
    suite_balanced: bool = False,
) -> list[Path]:
    if not leaderboard:
        return []
    labels = [_paper_label(str(row.get("method", ""))) for row in leaderboard]
    metric_specs = (
        ("detection_separability", "Detect.", _METRIC_COLORS["detection_separability"]),
        ("robustness", "Robust.", _METRIC_COLORS["robustness"]),
        ("utility", "Utility", _METRIC_COLORS["utility"]),
        ("stealth_conditioned", "Stealth", _METRIC_COLORS["stealth"]),
        ("efficiency_conditioned", "Efficiency", _METRIC_COLORS["efficiency"]),
    )
    component_specs = (
        ("gate", "Gate", _METRIC_COLORS["gate"], "o"),
        ("headline_core_score", "Core", _METRIC_COLORS["core_score"], "s"),
        ("headline_generalization", "Gen.", _METRIC_COLORS["generalization"], "^"),
        (HEADLINE_SCORE_FIELD, "Score", "#243B53", "D"),
    )
    component_values: dict[str, list[float]] = {
        "gate": [float(row.get("gate", 0.0)) for row in leaderboard],
        "headline_core_score": [
            float(row.get("headline_core_score", row.get("core_score", 0.0)))
            for row in leaderboard
        ],
        "headline_generalization": [
            float(row.get("headline_generalization", row.get("generalization", 0.0)))
            for row in leaderboard
        ],
        HEADLINE_SCORE_FIELD: [float(row.get(HEADLINE_SCORE_FIELD, 0.0)) for row in leaderboard],
    }
    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 4.55), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(0.88, 1.12), hspace=0.16)
    top_ax = fig.add_subplot(grid[0, 0])
    bottom_ax = fig.add_subplot(grid[1, 0])
    positions = list(range(len(labels)))
    for position in positions:
        row_values = [component_values[name][position] for name, _, _, _ in component_specs]
        top_ax.hlines(
            position,
            min(row_values),
            max(row_values),
            color="#D9E2EC",
            linewidth=1.7,
            zorder=1,
        )
    for component_name, component_label, color, marker in component_specs:
        values = component_values[component_name]
        top_ax.scatter(
            values,
            positions,
            s=58 if component_name == HEADLINE_SCORE_FIELD else 46,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=component_label,
        )
    headline_scores = component_values[HEADLINE_SCORE_FIELD]
    for position, score in zip(positions, headline_scores):
        top_ax.text(
            min(1.012, score + 0.02),
            position,
            f"{score:.2f}",
            ha="left",
            va="center",
            fontsize=8.8,
            color="#243B53",
            fontweight="semibold",
        )
    top_ax.set_xlim(0.0, 1.05)
    top_ax.set_ylim(-0.55, len(labels) - 0.45)
    top_ax.set_yticks(positions, labels)
    top_ax.invert_yaxis()
    top_ax.set_xlabel("Normalized component value")
    top_ax.grid(axis="x", alpha=0.23, linewidth=0.8)
    top_ax.spines["top"].set_visible(False)
    top_ax.spines["right"].set_visible(False)
    top_ax.spines["left"].set_visible(False)
    top_ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        borderaxespad=0.0,
        columnspacing=0.9,
        handlelength=1.0,
        handletextpad=0.35,
    )

    rows_payload: list[dict[str, Any]] = []
    metric_matrix = [
        [float(row.get(metric_name, 0.0)) for metric_name, _, _ in metric_specs]
        for row in leaderboard
    ]
    heatmap = bottom_ax.imshow(metric_matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    bottom_ax.set_xticks(list(range(len(metric_specs))), [label for _, label, _ in metric_specs])
    bottom_ax.set_yticks(positions, labels)
    for row_index, metric_values in enumerate(metric_matrix):
        for metric_index, value in enumerate(metric_values):
            bottom_ax.text(
                metric_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color="white" if value >= 0.58 else "#102A43",
            )
    for spine in bottom_ax.spines.values():
        spine.set_visible(False)
    bottom_ax.set_xticks([index - 0.5 for index in range(1, len(metric_specs))], minor=True)
    bottom_ax.set_yticks([index - 0.5 for index in range(1, len(labels))], minor=True)
    bottom_ax.grid(which="minor", color="white", linewidth=1.0)
    bottom_ax.tick_params(which="minor", bottom=False, left=False)
    bottom_ax.set_xlabel("Scorecard factor")
    colorbar = fig.colorbar(heatmap, ax=bottom_ax, fraction=0.035, pad=0.02)
    colorbar.ax.set_ylabel("Factor value", rotation=270, labelpad=12)
    for row, label in zip(leaderboard, labels):
        method = str(row.get("method", ""))
        base_payload = {
            "method": method,
            "paper_label": label,
            "origin": str(row.get("origin", "")),
            "aggregation_mode": "multiplicative_components",
            "score_semantics": _SCORE_DECOMPOSITION_SEMANTICS,
            "generalization_status": str(row.get("generalization_status", "unsupported")),
            "raw_generalization": row.get("generalization"),
            "headline_generalization_semantics": (
                "neutral_unsupported"
                if str(row.get("generalization_status", "unsupported")) == "unsupported"
                else (
                    "soft_floor_supported_zero"
                    if str(row.get("generalization_status", "unsupported")) == "supported_zero"
                    else "headline_from_supported_generalization"
                )
            ),
        }
        for component_name in ("gate", "headline_core_score", "headline_generalization", HEADLINE_SCORE_FIELD):
            rows_payload.append(
                {
                    **base_payload,
                    "panel": "headline_components",
                    "component": PUBLIC_HEADLINE_METRIC if component_name == HEADLINE_SCORE_FIELD else component_name,
                    "value": round(float(row.get(component_name, 0.0)), 4),
                }
            )
        for metric_name, _, _ in metric_specs:
            rows_payload.append(
                {
                    **base_payload,
                    "panel": "core_factors",
                    "component": metric_name,
                    "value": round(float(row.get(metric_name, 0.0)), 4),
                }
            )
    return _save_figure(fig, output_dir, f"{prefix}_score_decomposition", data=rows_payload)


def plot_generalization_breakdown(
    plt,
    leaderboard: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str = GENERATION_TIME_TRACK,
    suite_balanced: bool = False,
) -> list[Path]:
    if not leaderboard:
        return []
    labels = [_paper_label(str(row.get("method", ""))) for row in leaderboard]
    raw_scores = [row.get("generalization") for row in leaderboard]
    scores = [_safe_float(value) for value in raw_scores]
    colors = [_method_color(str(row.get("method", "")), str(row.get("origin", ""))) for row in leaderboard]
    payload = [
        {
            "method": str(row.get("method", "")),
            "paper_label": label,
            "generalization": row.get("generalization"),
            "generalization_status": str(row.get("generalization_status", "unsupported")),
            "generalization_supported": bool(row.get("generalization_supported", False)),
            "source_stability": float(row.get("source_stability") or 0.0),
            "task_stability": float(row.get("task_stability") or 0.0),
            "language_stability": float(row.get("language_stability") or 0.0),
            "cross_family_transfer": float(row.get("cross_family_transfer") or 0.0),
        }
        for row, label in zip(leaderboard, labels)
    ]
    fig, ax = plt.subplots(
        figsize=(max(SINGLE_COLUMN_WIDTH, 4.55), 2.8),
        constrained_layout=True,
    )
    positions = list(range(len(labels)))
    for position, raw_score, score, color in zip(positions, raw_scores, scores, colors):
        ax.vlines(position, 0.0, score, color="#d7dde5", linewidth=2.2, zorder=1)
        ax.scatter(position, score, s=78, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        label_text = "N/A" if raw_score is None else f"{score:.2f}"
        ax.text(position + 0.06, score, label_text, va="center", ha="left", fontsize=9, color="#243B53")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Generalization")
    ax.set_xlabel("")
    ax.set_xticks(positions, labels)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    return _save_figure(fig, output_dir, f"{prefix}_generalization_breakdown", data=payload)


def plot_quality_vs_robustness(
    plt,
    leaderboard: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str = GENERATION_TIME_TRACK,
    suite_balanced: bool = False,
) -> list[Path]:
    if not leaderboard:
        return []
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, 3.15),
        constrained_layout=True,
    )
    rows: list[dict[str, Any]] = []
    utility_values = [float(row.get("utility", 0.0)) for row in leaderboard]
    robustness_values = [float(row.get("robustness", 0.0)) for row in leaderboard]
    y_min = max(0.0, min(min(utility_values), min(robustness_values)) - 0.02)
    y_max = min(1.0, max(max(utility_values), max(robustness_values)) + 0.03)
    x_left = 0.0
    x_right = 1.0
    for row in leaderboard:
        utility = float(row.get("utility", 0.0))
        robustness = float(row.get("robustness", 0.0))
        label = _paper_label(str(row.get("method", "method")))
        color = _method_color(str(row.get("method", "")), str(row.get("origin", "")))
        ax.plot(
            [x_left, x_right],
            [utility, robustness],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=8.8,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            label,
            xy=(x_left, utility),
            xytext=(-12, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.6,
            color=color,
            fontweight="semibold",
        )
        ax.annotate(
            f"{robustness:.2f}",
            xy=(x_right, robustness),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.4,
            color="#243B53",
        )
        rows.append({"method": str(row.get("method", "method")), "paper_label": label, "origin": row.get("origin", ""), "utility": utility, "robustness": robustness})
    ax.set_xlim(-0.32, 1.28)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([x_left, x_right], ["Utility", "Robustness"])
    ax.set_ylabel("Normalized score")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_title("Utility-to-robustness drop", loc="left", fontsize=11.5, pad=4)
    return _save_figure(fig, output_dir, f"{prefix}_quality_vs_robustness", data=rows)


def plot_detection_vs_utility(
    plt,
    leaderboard: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str = GENERATION_TIME_TRACK,
) -> list[Path]:
    if not leaderboard:
        return []
    def _focused_bounds(
        values: list[float],
        *,
        pad_fraction: float = 0.18,
        min_pad: float = 0.015,
        upper_extra: float = 0.0,
    ) -> tuple[float, float]:
        low = min(values)
        high = max(values)
        span = max(high - low, min_pad)
        pad = max(min_pad, span * pad_fraction)
        return max(0.0, low - pad), min(1.0, high + pad + upper_extra)

    fig, (ax, interval_ax) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 3.05),
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.28},
        constrained_layout=True,
    )
    rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for row in leaderboard:
        detection = float(row.get("detection_separability", 0.0))
        utility = float(row.get("utility", 0.0))
        label = _paper_label(str(row.get("method", "method")))
        color = _method_color(str(row.get("method", "")), str(row.get("origin", "")))
        points.append(
            {
                "method": str(row.get("method", "method")),
                "paper_label": label,
                "origin": row.get("origin", ""),
                "detection_separability": detection,
                "utility": utility,
                "delta_detection_utility": detection - utility,
                "color": color,
            }
        )
    if not points:
        return []

    all_values = [
        value
        for point in points
        for value in (point["detection_separability"], point["utility"])
    ]
    x_min, x_max = _focused_bounds(all_values, pad_fraction=0.16, upper_extra=0.02)
    y_min, y_max = _focused_bounds([point["utility"] for point in points], pad_fraction=0.22)
    label_offsets = {
        "STONE": (9, 10, "left"),
        "EWD": (-11, 10, "right"),
        "KGW": (9, -3, "left"),
        "SWEET": (0, -14, "center"),
    }
    for point in points:
        detection = point["detection_separability"]
        utility = point["utility"]
        label = point["paper_label"]
        color = point["color"]
        ax.scatter(detection, utility, color=color, s=92, edgecolor="white", linewidth=0.9, zorder=3)
        dx, dy, ha = label_offsets.get(label, (6, 6, "left"))
        ax.annotate(
            label,
            xy=(detection, utility),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="bottom" if dy >= 0 else "top",
            fontsize=9.0,
            color=color,
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "#D9E2EC", "lw": 0.6, "alpha": 0.94},
            arrowprops={"arrowstyle": "-", "color": "#9FB3C8", "lw": 0.7, "shrinkA": 3, "shrinkB": 4},
        )
        rows.append(
            {
                "method": point["method"],
                "paper_label": label,
                "origin": point["origin"],
                "detection_separability": detection,
                "utility": utility,
                "delta_detection_utility": point["delta_detection_utility"],
            }
        )
    ax.set_title("A. Detection-utility plane", loc="left", fontsize=10.6, pad=4)
    ax.set_xlabel("Detection separability")
    ax.set_ylabel("Utility")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.tick_params(labelsize=8.8)

    ordered = sorted(points, key=lambda point: point["detection_separability"], reverse=True)
    y_positions = list(range(len(ordered)))
    for i, point in enumerate(ordered):
        detection = point["detection_separability"]
        utility = point["utility"]
        lo, hi = sorted((detection, utility))
        interval_ax.plot([lo, hi], [i, i], color="#BCCCDC", lw=3.0, solid_capstyle="round", zorder=1)
        interval_ax.scatter(
            utility,
            i,
            color="#2A9D8F",
            s=50,
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
            label="Utility" if i == 0 else None,
        )
        interval_ax.scatter(
            detection,
            i,
            color="#355C7D",
            s=54,
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
            label="Detection" if i == 0 else None,
        )
        interval_ax.annotate(
            f"{point['delta_detection_utility']:+.2f}",
            xy=(hi, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.3,
            color="#52616B",
        )
    interval_ax.set_title("B. Component separation", loc="left", fontsize=10.6, pad=4)
    interval_ax.set_yticks(y_positions)
    interval_ax.set_yticklabels([point["paper_label"] for point in ordered])
    interval_ax.invert_yaxis()
    interval_ax.set_xlim(x_min, x_max)
    interval_ax.set_xlabel("Component value")
    interval_ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    interval_ax.tick_params(labelsize=8.8)
    interval_ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#D9E2EC",
        fontsize=8.4,
        handlelength=1.1,
        borderpad=0.35,
    )
    return _save_figure(fig, output_dir, f"{prefix}_detection_vs_utility", data=rows)


def plot_method_stability_heatmap(
    plt,
    leaderboard: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str = GENERATION_TIME_TRACK,
) -> list[Path]:
    if not leaderboard:
        return []
    metric_names = (
        ("source_stability", "Source"),
        ("task_stability", "Task"),
        ("language_stability", "Language"),
        ("cross_family_transfer", "Cross-family"),
    )
    row_labels = [_paper_label(str(row.get("method", ""))) for row in leaderboard]
    col_labels = [label for _, label in metric_names]
    matrix = [
        [float(row.get(metric_name) or 0.0) for metric_name, _ in metric_names]
        for row in leaderboard
    ]
    data = [
        {
            "method": str(row.get("method", "")),
            "paper_label": _paper_label(str(row.get("method", ""))),
            **{metric_name: float(row.get(metric_name) or 0.0) for metric_name, _ in metric_names},
        }
        for row in leaderboard
    ]
    return _plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title="",
        xlabel="Stability Slice",
        ylabel="Method",
        output_dir=output_dir,
        stem=f"{prefix}_method_stability_heatmap",
        cmap="Blues",
        data=data,
    )


def plot_model_generalization_breakdown(
    plt,
    leaderboard: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    if not leaderboard:
        return []
    metric_names = (
        ("cross_family_transfer", "Cross-family"),
        ("scale_consistency", "Within-family scale"),
    )
    row_labels = [_paper_label(str(row.get("method", ""))) for row in leaderboard]
    matrix = [
        [float(row.get(metric_name) or 0.0) for metric_name, _ in metric_names]
        for row in leaderboard
    ]
    data = [
        {
            "method": str(row.get("method", "")),
            "paper_label": _paper_label(str(row.get("method", ""))),
            "cross_family_transfer": float(row.get("cross_family_transfer") or 0.0),
            "scale_consistency": float(row.get("scale_consistency") or 0.0),
            "scale_supported_families": list(row.get("scale_supported_families", [])),
            "scale_supported_family_count": int(row.get("scale_supported_family_count", 0) or 0),
        }
        for row in leaderboard
    ]
    return _plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=row_labels,
        col_labels=[label for _, label in metric_names],
        title="",
        xlabel="Model Generalization",
        ylabel="Method",
        output_dir=output_dir,
        stem=f"{prefix}_model_generalization_breakdown",
        cmap="GnBu",
        data=data,
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def plot_source_breakdown(
    plt,
    rows: list[Any],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str = GENERATION_TIME_TRACK,
) -> list[Path]:
    if not rows:
        return []
    methods = sorted({str(getattr(row, "watermark_scheme", "")).strip() for row in rows if str(getattr(row, "watermark_scheme", "")).strip()})
    source_groups = [normalize_source_group(source_group) for source_group in SUITE_AGGREGATE_SOURCE_GROUPS]
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        for source_group in source_groups:
            slice_rows = [row for row in rows if str(getattr(row, "watermark_scheme", "")).strip() == method and normalize_source_group(getattr(row, "source_group", "")) == source_group]
            score = float(scorecard_for_rows(slice_rows, restrict_source_groups={source_group}).get("core_score", 0.0)) if slice_rows else 0.0
            values.append(score)
            payload.append(
                {
                    "method": method,
                    "paper_label": _paper_label(method),
                    "source_group": source_group,
                    "source_label": _suite_source_label(source_group),
                    "source_short_label": _paper_source_release_label(source_group),
                    "core_score": score,
                    "row_count": len(slice_rows),
                }
            )
        matrix.append(values)
    max_score = max((max(values) for values in matrix), default=0.0)
    return _plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_paper_label(method) for method in methods],
        col_labels=[_paper_source_release_label(source_group) for source_group in source_groups],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=f"{prefix}_per_source_breakdown",
        cmap="Blues",
        vmax=max(0.55, max_score + 0.04),
        colorbar_label="Core Score",
        data=payload,
        max_columns_per_panel=len(source_groups),
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def plot_model_breakdown(
    plt,
    leaderboard: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    paper_track: str = GENERATION_TIME_TRACK,
) -> list[Path]:
    if not leaderboard:
        return []
    methods = sorted({str(row.get("method", "")).strip() for row in leaderboard if str(row.get("method", "")).strip()})
    model_order = [model for model in SUITE_MODEL_ROSTER if any(str(row.get("model", "")).strip() == model for row in leaderboard)]
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        for model in model_order:
            entry = next((row for row in leaderboard if str(row.get("method", "")).strip() == method and str(row.get("model", "")).strip() == model), None)
            score = float(entry.get(HEADLINE_SCORE_FIELD, 0.0)) if entry is not None else 0.0
            values.append(score)
            payload.append(
                {
                    "method": method,
                    "paper_label": _paper_label(method),
                    "model": model,
                    "model_label": _paper_model_label(model),
                    HEADLINE_SCORE_FIELD: score,
                    "row_count": int(entry.get("row_count", 0)) if entry is not None else 0,
                }
            )
        matrix.append(values)
    model_labels = [_paper_model_axis_label(model) for model in model_order]
    max_score = max((max(values) for values in matrix), default=0.0)
    return _plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_paper_label(method) for method in methods],
        col_labels=model_labels,
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=f"{prefix}_per_model_breakdown",
        cmap="Blues",
        vmax=max(0.6, min(1.0, max_score + 0.08)),
        data=payload,
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def plot_efficiency_breakdown(
    plt,
    leaderboard: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    if not leaderboard:
        return []
    methods = sorted({str(row.get("method", "")).strip() for row in leaderboard if str(row.get("method", "")).strip()})
    model_order = [model for model in SUITE_MODEL_ROSTER if any(str(row.get("model", "")).strip() == model for row in leaderboard)]
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        for model in model_order:
            entry = next((row for row in leaderboard if str(row.get("method", "")).strip() == method and str(row.get("model", "")).strip() == model), None)
            value = float(entry.get("efficiency", 0.0)) if entry is not None else 0.0
            values.append(value)
            payload.append(
                {
                    "method": method,
                    "paper_label": _paper_label(method),
                    "model": model,
                    "model_label": _paper_model_label(model),
                    "efficiency": value,
                    "row_count": int(entry.get("row_count", 0)) if entry is not None else 0,
                }
            )
        matrix.append(values)
    return _plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_paper_label(method) for method in methods],
        col_labels=[_paper_model_axis_label(model) for model in model_order],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=f"{prefix}_efficiency_breakdown",
        cmap="YlOrBr",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Efficiency",
        data=payload,
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def plot_language_breakdown(
    plt,
    rows: list[Any],
    *,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    if not rows:
        return []
    multilingual_source_groups = {
        normalize_source_group(source.source_group)
        for source in SUITE_AGGREGATE_SOURCES
        if source.validation_scope == "multilingual_exec"
    }
    multilingual_rows = [row for row in rows if normalize_source_group(getattr(row, "source_group", "")) in multilingual_source_groups]
    if not multilingual_rows:
        return []
    languages = [language for language in OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES if any(str(getattr(row, "language", "")).strip() == language for row in multilingual_rows)]
    if not languages:
        return []
    methods = sorted({str(getattr(row, "watermark_scheme", "")).strip() for row in multilingual_rows if str(getattr(row, "watermark_scheme", "")).strip()})
    payload: list[dict[str, Any]] = []
    utility_by_language: dict[str, list[float]] = {language: [] for language in languages}
    for method in methods:
        for language in languages:
            slice_rows = [
                row
                for row in multilingual_rows
                if str(getattr(row, "watermark_scheme", "")).strip() == method and str(getattr(row, "language", "")).strip() == language
            ]
            utility = float(scorecard_for_rows(slice_rows).get("utility", 0.0)) if slice_rows else 0.0
            utility_by_language[language].append(utility)
            payload.append(
                {
                    "method": method,
                    "paper_label": _paper_label(method),
                    "language": language,
                    "utility": utility,
                    "row_count": len(slice_rows),
                }
            )
    fig_height = max(4.0, 1.55 * len(languages) + 0.65)
    fig, axes = plt.subplots(len(languages), 1, figsize=(SINGLE_COLUMN_WIDTH, fig_height), sharex=True, constrained_layout=True)
    if len(languages) == 1:
        axes = [axes]
    method_labels = [_paper_label(method) for method in methods]
    for axis, language in zip(axes, languages):
        values = utility_by_language[language]
        positions = list(range(len(methods)))
        axis.hlines(positions, [0.0] * len(values), values, color="#d7dde5", linewidth=2.2, zorder=1)
        for position, method, value in zip(positions, methods, values):
            axis.scatter(value, position, s=46, color=_method_color(method), edgecolor="white", linewidth=0.8, zorder=3)
        axis.text(0.0, 1.01, language.title(), transform=axis.transAxes, ha="left", va="bottom", fontsize=10)
        axis.set_yticks(positions, method_labels)
        axis.set_xlim(0.0, 1.05)
        axis.grid(axis="x", alpha=0.25, linewidth=0.8)
        axis.invert_yaxis()
    axes[-1].set_xlabel("Utility")
    axes[len(axes) // 2].set_ylabel("Method")
    return _save_figure(fig, output_dir, f"{prefix}_per_language_breakdown", data=payload)


def plot_attack_breakdown(
    plt,
    rows: list[Any],
    *,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    if not rows:
        return []
    attacks = sorted({str(getattr(row, "attack_name", "")).strip() for row in rows if str(getattr(row, "attack_name", "")).strip()})
    methods = sorted({str(getattr(row, "watermark_scheme", "")).strip() for row in rows if str(getattr(row, "watermark_scheme", "")).strip()})
    if not attacks or not methods:
        return []
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        for attack in attacks:
            slice_rows = [
                row
                for row in rows
                if str(getattr(row, "watermark_scheme", "")).strip() == method and str(getattr(row, "attack_name", "")).strip() == attack
            ]
            attacked_detect_rate = (
                sum(1.0 for row in slice_rows if bool(getattr(row, "attacked_detected", False))) / float(len(slice_rows))
                if slice_rows
                else 0.0
            )
            values.append(attacked_detect_rate)
            payload.append(
                {
                    "method": method,
                    "paper_label": _paper_label(method),
                    "attack": attack,
                    "attack_label": _paper_attack_label(attack),
                    "attacked_detect_rate": attacked_detect_rate,
                    "row_count": len(slice_rows),
                }
            )
        matrix.append(values)
    return _plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_paper_label(method) for method in methods],
        col_labels=[_paper_attack_label(attack) for attack in attacks],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=f"{prefix}_attack_breakdown",
        cmap="Purples",
        data=payload,
        max_columns_per_panel=len(attacks),
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def render_figures(
    *,
    manifest_path: Path | list[Path] | None,
    report_paths: Iterable[Path] | None,
    baseline_eval_paths: Iterable[Path] | None,
    output_dir: Path,
    prefix: str,
    suite: str = "basic",
    matrix_index_path: Path | None = None,
    anchor_report_path: Path | None = None,
    paper_track: str = GENERATION_TIME_TRACK,
    allow_mixed_tracks: bool = False,
    include_reference_artifacts: bool = False,
    require_times_new_roman: bool = False,
) -> list[Path]:
    if suite == "all" and matrix_index_path is None:
        raise ValueError("suite=all publication-facing figure exports require --matrix-index.")
    _, plt = configure_matplotlib(require_times_new_roman=require_times_new_roman)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    manifest_paths = []
    if manifest_path is None:
        manifest_paths = []
    elif isinstance(manifest_path, list):
        manifest_paths = manifest_path
    else:
        manifest_paths = [manifest_path]
    resolved_report_paths = [_resolve_report_path(path) for path in list(report_paths or [])]
    resolved_baseline_paths = list(baseline_eval_paths or [])
    if matrix_index_path is not None:
        matrix_index_payload = _load_json(matrix_index_path)
        if suite == "all":
            _require_canonical_suite_identity_for_matrix_index(
                matrix_index_payload,
                label="render_figures suite=all matrix index",
            )
            _require_formal_single_host_execution_mode(
                matrix_index_payload,
                label="render_figures suite=all matrix index",
            )
        matrix_reports, matrix_baselines = _reports_from_matrix_index(matrix_index_path)
        resolved_report_paths.extend(matrix_reports)
        resolved_baseline_paths.extend(matrix_baselines)
    resolved_report_paths = _dedupe_paths(resolved_report_paths)
    resolved_baseline_paths = _dedupe_paths(resolved_baseline_paths)

    manifests = [_load_json(path) for path in manifest_paths if path.exists()]
    reports = [_load_json(path) for path in resolved_report_paths if path.exists()]
    evaluations = [_load_json(path) for path in resolved_baseline_paths if path.exists()]
    paper_reports = _paper_safe_reports(reports, paper_track=paper_track, allow_mixed_tracks=allow_mixed_tracks)
    suite_reports = paper_reports
    if suite != "all":
        primary_report = _anchor_report(
            paper_reports,
            resolved_report_paths=resolved_report_paths,
            anchor_report_path=anchor_report_path,
        )
    else:
        primary_report = None
    paper_evaluations = _paper_safe_baseline_evaluations(evaluations, paper_track=paper_track)

    if manifests and suite != "all":
        outputs.extend(plot_language_coverage(plt, manifests[0], output_dir=output_dir, prefix=prefix))
    if primary_report is not None:
        outputs.extend(plot_attack_robustness(plt, primary_report, output_dir=output_dir, prefix=prefix))
        if suite != "all":
            outputs.extend(plot_functional_summary(plt, primary_report, output_dir=output_dir, prefix=prefix))
    if suite != "all":
        outputs.extend(plot_baseline_comparison(plt, paper_evaluations, output_dir=output_dir, prefix=prefix))
    leaderboard_outputs: list[Path] = []
    method_model_leaderboard: list[dict[str, Any]] = []
    method_master_leaderboard: list[dict[str, Any]] = []
    leaderboard_reports = paper_reports
    if leaderboard_reports:
        leaderboard_outputs, method_model_leaderboard, method_master_leaderboard, _ = export_leaderboards(
            leaderboard_reports,
            output_dir=output_dir,
            prefix=prefix,
            paper_track=paper_track,
            include_reference_artifacts=include_reference_artifacts,
            all_reports=reports,
            suite_atomic_only=suite == "all",
        )
        outputs.extend(leaderboard_outputs)
        if suite == "all":
            _validate_suite_atomic_leaderboard(method_master_leaderboard)
            _validate_suite_model_roster(
                method_model_leaderboard,
                expected_models=_expected_models_from_matrix_index(matrix_index_path),
            )

    if suite == "all":
        outputs.extend(plot_score_decomposition(plt, method_master_leaderboard, output_dir=output_dir, prefix=prefix, paper_track=paper_track, suite_balanced=True))
        outputs.extend(plot_detection_vs_utility(plt, method_master_leaderboard, output_dir=output_dir, prefix=prefix, paper_track=paper_track))
    _remove_duplicate_leaderboard_sidecars(output_dir, prefix)
    return [path for path in outputs if path.exists()]


def main() -> int:
    args = parse_args()
    if args.suite == "all":
        if args.matrix_index is None:
            raise SystemExit("suite=all publication-facing figure exports require --matrix-index.")
        if args.report or args.baseline_eval or args.manifest or args.anchor_report:
            raise SystemExit(
                "suite=all publication-facing figure exports must derive from the canonical matrix index only; "
                "do not mix --report, --baseline-eval, --manifest, or --anchor-report inputs."
            )
        if args.paper_track != GENERATION_TIME_TRACK:
            raise SystemExit("suite=all publication-facing figure exports require --paper-track generation_time.")
        if args.allow_mixed_tracks:
            raise SystemExit("suite=all publication-facing figure exports do not allow --allow-mixed-tracks.")
        canonical_output_dir = (ROOT / "results" / "figures" / "suite_all_models_methods").resolve(strict=False)
        requested_output_dir = args.output_dir.resolve(strict=False)
        requested_prefix = str(args.prefix).strip()
        if requested_output_dir == canonical_output_dir and requested_prefix == "suite_all_models_methods":
            payload = _load_json(args.matrix_index)
            try:
                _require_canonical_suite_identity_for_matrix_index(
                    payload,
                    label="suite=all publication-facing figure export matrix index",
                )
                _require_formal_single_host_execution_mode(
                    payload,
                    label="suite=all publication-facing figure export matrix index",
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
    outputs = render_figures(
        manifest_path=args.manifest,
        report_paths=args.report,
        baseline_eval_paths=args.baseline_eval,
        output_dir=args.output_dir,
        prefix=args.prefix,
        suite=args.suite,
        matrix_index_path=args.matrix_index,
        anchor_report_path=args.anchor_report,
        paper_track=args.paper_track,
        allow_mixed_tracks=args.allow_mixed_tracks,
        include_reference_artifacts=args.include_reference_artifacts,
        require_times_new_roman=args.require_times_new_roman,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
