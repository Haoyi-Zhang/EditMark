from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

try:
    from _shared import dump_json, read_jsonl
except ModuleNotFoundError:  # pragma: no cover
    from scripts._shared import dump_json, read_jsonl

from posteditbench.suite import OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES
from scripts.render_paper_figures import DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH, configure_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dataset statistics tables and single-column figures for the canonical public release suite.")
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=ROOT / "results" / "tables" / "dataset_statistics",
        help="Output directory for dataset statistics tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "results" / "figures" / "dataset_statistics",
        help="Output directory for dataset statistics figures.",
    )
    parser.add_argument(
        "--require-times-new-roman",
        dest="require_times_new_roman",
        action="store_true",
        default=True,
        help="Fail closed if Times New Roman is unavailable.",
    )
    parser.add_argument(
        "--allow-font-fallback",
        dest="require_times_new_roman",
        action="store_false",
        help="Allow serif fallback fonts instead of failing on missing Times New Roman.",
    )
    return parser.parse_args()


RELEASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "slug": "humaneval_plus",
        "dataset_label": "HumanEval+",
        "source_group": "public_humaneval_plus",
        "scope": "release",
        "path": ROOT / "data" / "release" / "sources" / "suite_humaneval_plus_release.normalized.jsonl",
        "manifest_path": ROOT / "data" / "release" / "sources" / "suite_humaneval_plus_release.normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
        "execution_slice": "python",
    },
    {
        "slug": "mbpp_plus",
        "dataset_label": "MBPP+",
        "source_group": "public_mbpp_plus",
        "scope": "release",
        "path": ROOT / "data" / "release" / "sources" / "suite_mbpp_plus_release.normalized.jsonl",
        "manifest_path": ROOT / "data" / "release" / "sources" / "suite_mbpp_plus_release.normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
        "execution_slice": "python",
    },
    {
        "slug": "humaneval_x",
        "dataset_label": "HumanEval-X (5-language balanced slice)",
        "source_group": "public_humaneval_x",
        "scope": "release",
        "path": ROOT / "data" / "release" / "sources" / "suite_humanevalx_release.normalized.jsonl",
        "manifest_path": ROOT / "data" / "release" / "sources" / "suite_humanevalx_release.normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
        "execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
    },
    {
        "slug": "mbxp_5lang",
        "dataset_label": "MBXP-5lang (5-language balanced slice)",
        "source_group": "public_mbxp_5lang",
        "scope": "release",
        "path": ROOT / "data" / "release" / "sources" / "suite_mbxp_release.normalized.jsonl",
        "manifest_path": ROOT / "data" / "release" / "sources" / "suite_mbxp_release.normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
        "execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
    },
    {
        "slug": "crafted_original",
        "dataset_label": "Crafted Original",
        "source_group": "crafted_original",
        "scope": "release",
        "path": ROOT / "data" / "release" / "sources" / "crafted_original_release.normalized.jsonl",
        "manifest_path": ROOT / "data" / "release" / "sources" / "crafted_original_release.normalized.manifest.json",
        "source_type": "crafted",
        "aggregate_score": True,
        "execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
    },
    {
        "slug": "crafted_translation",
        "dataset_label": "Crafted Translation",
        "source_group": "crafted_translation",
        "scope": "release",
        "path": ROOT / "data" / "release" / "sources" / "crafted_translation_release.normalized.jsonl",
        "manifest_path": ROOT / "data" / "release" / "sources" / "crafted_translation_release.normalized.manifest.json",
        "source_type": "crafted",
        "aggregate_score": True,
        "execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
    },
    {
        "slug": "crafted_stress",
        "dataset_label": "Crafted Stress",
        "source_group": "crafted_stress",
        "scope": "release",
        "path": ROOT / "data" / "release" / "sources" / "crafted_stress_release.normalized.jsonl",
        "manifest_path": ROOT / "data" / "release" / "sources" / "crafted_stress_release.normalized.manifest.json",
        "source_type": "crafted",
        "aggregate_score": True,
        "execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
    },
)

_SOURCE_TYPE_COLORS = {"public": "#355C7D", "crafted": "#D17C48"}
_LANGUAGE_HEATMAP = "YlGnBu"
_CATEGORY_HEATMAP = "YlOrRd"
_FAMILY_HEATMAP = "PuBuGn"
_LANGUAGE_DISPLAY_LABELS = {"Python": "Py", "Cpp": "C++", "Java": "Java", "Javascript": "JS", "Go": "Go"}

_STALE_TABLE_FILES: tuple[str, ...] = ()

_STALE_FIGURE_FILES: tuple[str, ...] = (
    "release_slice_source_language_coverage.png",
    "release_slice_source_language_coverage.pdf",
    "task_category_distribution.png",
    "task_category_distribution.pdf",
    "crafted_family_coverage.png",
    "crafted_family_coverage.pdf",
)

_SOURCE_SHORT_LABELS = {
    "HumanEval": "HE",
    "HumanEval+": "HE+",
    "MBPP+": "MBPP+",
    "HumanEval-X": "HEX",
    "HumanEval-X (5-language balanced slice)": "HEX",
    "MBXP-5lang": "MBXP",
    "MBXP-5lang (5-language balanced slice)": "MBXP",
    "Crafted Original": "Orig.",
    "Crafted Translation": "Trans.",
    "Crafted Stress": "Stress",
}

_RELEASE_FIGURE_SOURCE_LABELS = {
    "HumanEval+": "HE+\n(active)",
    "MBPP+": "MBPP+\n(active)",
    "HumanEval-X (5-language balanced slice)": "HumanEval-X\n(5-language\nbalanced slice)",
    "MBXP-5lang (5-language balanced slice)": "MBXP-5lang\n(5-language\nbalanced slice)",
    "Crafted Original": "Orig.\n(active)",
    "Crafted Translation": "Trans.\n(active)",
    "Crafted Stress": "Stress\n(active)",
}

_RELEASE_FIGURE_COMPACT_LABELS = {
    "HumanEval+": "HE+",
    "MBPP+": "MBPP+",
    "HumanEval-X (5-language balanced slice)": "HEX-5L",
    "MBXP-5lang (5-language balanced slice)": "MBXP-5L",
    "Crafted Original": "Orig.",
    "Crafted Translation": "Trans.",
    "Crafted Stress": "Stress",
}

_CATEGORY_SHORT_LABELS = {
    "class/object interaction": "Object\ninteraction",
    "cross-language idiom preservation": "Cross-lang\nidioms",
    "data structures": "Data\nstructures",
    "exception/error handling": "Errors",
    "graph/search": "Graph /\nsearch",
    "numeric/boundary conditions": "Numeric /\nboundary",
    "recursion/dp": "Recursion /\nDP",
    "state machines/simulation": "State /\nsim.",
    "strings/parsing": "Strings /\nparsing",
    "Other categories": "Other\ncats.",
}

_FAMILY_SHORT_LABELS = {
    "API-style normalization": "API\nnormalize",
    "arrays/lists": "Arrays /\nlists",
    "dp/recursion": "DP /\nrec.",
    "graph/search": "Graph /\nsearch",
    "maps/sets": "Maps /\nsets",
    "math/bit ops": "Math /\nbit",
    "parsing": "Parse",
    "stateful update": "Stateful\nupdate",
    "strings": "Strings",
    "Other families": "Other\nfam.",
}

_ACTIVE_RELEASE_POLICY: dict[str, dict[str, Any]] = {
    "humaneval_plus": {
        "active_execution_slice": "python",
        "sampling_rule": "full retained",
        "release_scored": True,
    },
    "mbpp_plus": {
        "active_execution_slice": "python",
        "sampling_rule": "full retained",
        "release_scored": True,
    },
    "humaneval_x": {
        "active_execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        "sampling_rule": "deterministic five-language balanced slice",
        "release_scored": True,
    },
    "mbxp_5lang": {
        "active_execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        "sampling_rule": "deterministic five-language balanced slice with smoke-overlay support",
        "release_scored": True,
    },
    "crafted_original": {
        "active_execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        "sampling_rule": "project-authored curated five-language family/category-balanced crafted release source",
        "release_scored": True,
    },
    "crafted_translation": {
        "active_execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        "sampling_rule": "project-authored curated five-language family/category-balanced crafted release source",
        "release_scored": True,
    },
    "crafted_stress": {
        "active_execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        "sampling_rule": "project-authored curated five-language family/category-balanced crafted release source",
        "release_scored": True,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def _label_map(rows: Iterable[dict[str, Any]], field: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in rows:
        value = _normalize(row.get(field))
        raw = str(row.get(field, "")).strip()
        if value and raw and value not in labels:
            labels[value] = raw
    return labels


def _count_field(rows: list[dict[str, Any]], field: str) -> tuple[dict[str, int], dict[str, str]]:
    labels = _label_map(rows, field)
    counts = Counter(_normalize(row.get(field)) for row in rows if _normalize(row.get(field)))
    return dict(sorted(counts.items())), labels


def _spec_summary(spec: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(spec["path"])
    manifest = _load_json(spec["manifest_path"])
    release_policy = _ACTIVE_RELEASE_POLICY[spec["slug"]]
    language_counts, language_labels = _count_field(rows, "language")
    category_counts, category_labels = _count_field(rows, "category")
    family_ids = {str(row.get("family_id", "")).strip() for row in rows if str(row.get("family_id", "")).strip()}
    template_family_counts, template_family_labels = _count_field(rows, "template_family")
    aggregate_score = bool(release_policy["release_scored"])
    execution_slice = str(spec.get("execution_slice", release_policy["active_execution_slice"]))
    active_execution_slice = str(release_policy["active_execution_slice"])
    sampling_rule = str(release_policy["sampling_rule"])
    scoring_status = "active_release_slice_scored" if aggregate_score else "active_release_slice_not_scored"
    return {
        "slug": spec["slug"],
        "dataset_label": spec["dataset_label"],
        "source_group": spec["source_group"],
        "scope": spec["scope"],
        "source_type": spec["source_type"],
        "aggregate_score": aggregate_score,
        "execution_slice": execution_slice,
        "active_execution_slice": active_execution_slice,
        "sampling_rule": sampling_rule,
        "scoring_status": scoring_status,
        "path": str(Path(spec["path"]).relative_to(ROOT)).replace("\\", "/"),
        "manifest_path": str(Path(spec["manifest_path"]).relative_to(ROOT)).replace("\\", "/"),
        "record_count": int(len(rows)),
        "language_count": int(len(language_counts)),
        "languages": [language_labels.get(key, key) for key in language_counts],
        "language_counts": language_counts,
        "category_count": int(len(category_counts)),
        "category_counts": category_counts,
        "category_labels": category_labels,
        "family_count": int(len(family_ids) or manifest.get("family_count") or 0),
        "template_family_count": int(len(template_family_counts)),
        "template_family_counts": template_family_counts,
        "template_family_labels": template_family_labels,
        "reference_kind_counts": dict(manifest.get("reference_kind_counts", {})),
        "canonical_reference_count": int(manifest.get("canonical_reference_count", 0) or 0),
        "smoke_overlay_reference_count": int(manifest.get("smoke_overlay_reference_count", 0) or 0),
        "validation_scope": str(manifest.get("validation_scope", "")),
        "source_manifest_count": int(len(manifest.get("source_manifests", []) or [])),
        "claimed_languages": list(manifest.get("claimed_languages", manifest.get("languages", [])) or []),
    }


def _write_json_csv(path_root: Path, rows: list[dict[str, Any]]) -> None:
    path_root.parent.mkdir(parents=True, exist_ok=True)
    dump_json(path_root.with_suffix(".json"), rows)
    if not rows:
        path_root.with_suffix(".csv").write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path_root.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _source_short_label(value: str) -> str:
    return _SOURCE_SHORT_LABELS.get(str(value).strip(), str(value).strip())


def _release_figure_source_label(value: str) -> str:
    raw = str(value).strip()
    return _RELEASE_FIGURE_SOURCE_LABELS.get(raw, _source_short_label(raw))


def _release_figure_compact_label(value: str) -> str:
    raw = str(value).strip()
    return _RELEASE_FIGURE_COMPACT_LABELS.get(raw, _source_short_label(raw))
def _category_short_label(value: str) -> str:
    return _CATEGORY_SHORT_LABELS.get(str(value).strip(), str(value).strip().title())


def _family_short_label(value: str) -> str:
    return _FAMILY_SHORT_LABELS.get(str(value).strip(), str(value).strip())


def _save_plot(fig: Any, output_root: Path) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_root.with_suffix(".png"), bbox_inches="tight")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _annotate_hbar_end(ax: Any, values: list[int | float], y_positions: list[int], *, pad_fraction: float = 0.015) -> None:
    if not values:
        return
    max_value = max(float(value) for value in values)
    pad = max(1.0, max_value * pad_fraction)
    x0, x1 = ax.get_xlim()
    right_guard = x1 - (x1 - x0) * 0.035
    for y_position, value in zip(y_positions, values):
        label_x = float(value) + pad
        horizontal = "left"
        if label_x > right_guard:
            label_x = float(value) - pad
            horizontal = "right"
        ax.text(
            label_x,
            y_position,
            f"{int(value)}",
            va="center",
            ha=horizontal,
            fontsize=9.8,
            color="#243B53",
        )


def _draw_count_heatmap(
    plt: Any,
    *,
    matrix: list[list[int]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_root: Path,
    cmap: str,
    colorbar_label: str,
    subtitle: str = "",
    figure_width: float | None = None,
    color_power_gamma: float | None = None,
) -> None:
    from matplotlib.colors import PowerNorm

    fig_height = max(2.45, 1.6 + 0.27 * len(row_labels))
    fig, ax = plt.subplots(figsize=(max(SINGLE_COLUMN_WIDTH, figure_width or SINGLE_COLUMN_WIDTH), fig_height), constrained_layout=True)
    vmax = max(max(row) for row in matrix) if matrix else 1
    if color_power_gamma is not None:
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            norm=PowerNorm(gamma=color_power_gamma, vmin=0, vmax=max(1, vmax)),
        )
    else:
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(1, vmax))
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=12.8)
    if subtitle:
        fig.text(0.02, 0.94, subtitle, ha="left", va="top", fontsize=9, color="#52606D")
    ax.set_xticks(list(range(len(col_labels))), col_labels)
    ax.set_yticks(list(range(len(row_labels))), row_labels)
    ax.tick_params(axis="x", rotation=0, labelsize=8.4, pad=5)
    ax.tick_params(axis="y", labelsize=8.9)
    threshold = max(1, vmax) * 0.58
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            text_color = "white" if value >= threshold and vmax > 0 else "#102A43"
            ax.text(column_index, row_index, str(int(value)), ha="center", va="center", fontsize=8.4, color=text_color)
    ax.set_xticks([index - 0.5 for index in range(1, len(col_labels))], minor=True)
    ax.set_yticks([index - 0.5 for index in range(1, len(row_labels))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=13)
    colorbar.ax.tick_params(labelsize=8.1)
    _save_plot(fig, output_root)
    plt.close(fig)


def _release_slice_composition_figure(plt: Any, release_rows: list[dict[str, Any]], output_root: Path) -> None:
    from matplotlib.lines import Line2D

    labels = [_release_figure_compact_label(str(row["dataset_label"])) for row in release_rows]
    counts = [int(row["record_count"]) for row in release_rows]
    colors = [_SOURCE_TYPE_COLORS[str(row["source_type"])] for row in release_rows]
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(SINGLE_COLUMN_WIDTH, 4.65), 2.9), constrained_layout=True)
    ax.hlines(positions, [0.0] * len(counts), counts, color="#D7DDE5", linewidth=2.6, zorder=1)
    ax.scatter(counts, positions, s=64, color=colors, edgecolor="white", linewidth=0.9, zorder=3)
    ax.set_yticks(positions, labels)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    right_limit = max(300, (((max(counts, default=0) + 40) // 50) + 1) * 50)
    ax.set_xlim(-5, right_limit)
    ax.set_xlabel("Active release records")
    ax.axhline(3.5, color="#E4E7EB", linewidth=0.9, zorder=0)
    ax.legend(
        [
            Line2D([0], [0], marker="o", linestyle="", markersize=7.5, markerfacecolor=_SOURCE_TYPE_COLORS["public"], markeredgecolor="white"),
            Line2D([0], [0], marker="o", linestyle="", markersize=7.5, markerfacecolor=_SOURCE_TYPE_COLORS["crafted"], markeredgecolor="white"),
        ],
        ["Public slices", "Crafted slices"],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.60, 1.10),
        borderaxespad=0.0,
        columnspacing=1.2,
        handletextpad=0.4,
        fontsize=9.4,
    )
    ax.invert_yaxis()
    _annotate_hbar_end(ax, counts, positions)
    _save_plot(fig, output_root)
    plt.close(fig)


def _source_language_coverage_figure(plt: Any, release_rows: list[dict[str, Any]], output_root: Path) -> None:
    languages = list(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES)
    matrix = [
        [int(dict(row["language_counts"]).get(language, 0)) for language in languages]
        for row in release_rows
    ]
    _draw_count_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_release_figure_compact_label(str(row["dataset_label"])) for row in release_rows],
        col_labels=[_LANGUAGE_DISPLAY_LABELS.get(language.title(), language.title()) for language in languages],
        title="",
        subtitle="",
        output_root=output_root,
        cmap=_LANGUAGE_HEATMAP,
        colorbar_label="Active release records",
        figure_width=4.25,
        color_power_gamma=0.65,
    )


def _task_category_distribution_figure(plt: Any, category_rows: list[dict[str, Any]], output_root: Path) -> None:
    crafted_rows = [row for row in category_rows if row["slug"] in {"crafted_original", "crafted_translation", "crafted_stress"}]
    totals = Counter()
    for row in crafted_rows:
        totals[str(row["category_label"])] += int(row["record_count"])
    top_categories = [category for category, _ in totals.most_common(4)]
    columns = top_categories + ["Other categories"]
    dataset_order = ["Crafted Original", "Crafted Translation", "Crafted Stress"]
    matrix: list[list[int]] = []
    for dataset_label in dataset_order:
        dataset_rows = [row for row in crafted_rows if row["dataset_label"] == dataset_label]
        counts = {str(row["category_label"]): int(row["record_count"]) for row in dataset_rows}
        other_count = sum(value for category, value in counts.items() if category not in top_categories)
        matrix.append([counts.get(category, 0) for category in top_categories] + [other_count])
    _draw_count_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_source_short_label(label) for label in dataset_order],
        col_labels=[_category_short_label(label) for label in columns],
        title="",
        subtitle="",
        output_root=output_root,
        cmap=_CATEGORY_HEATMAP,
        colorbar_label="Release crafted records",
        figure_width=4.2,
        color_power_gamma=0.72,
    )


def _crafted_family_coverage_figure(plt: Any, family_rows: list[dict[str, Any]], output_root: Path) -> None:
    crafted_rows = [row for row in family_rows if row["source_type"] == "crafted"]
    totals = Counter()
    for row in crafted_rows:
        totals[str(row["template_family_label"])] += int(row["record_count"])
    top_families = [family for family, _ in totals.most_common(4)]
    columns = top_families + ["Other families"]
    dataset_order = ["Crafted Original", "Crafted Translation", "Crafted Stress"]
    matrix: list[list[int]] = []
    for dataset_label in dataset_order:
        dataset_rows = [row for row in crafted_rows if row["dataset_label"] == dataset_label]
        counts = {str(row["template_family_label"]): int(row["record_count"]) for row in dataset_rows}
        other_count = sum(value for family, value in counts.items() if family not in top_families)
        matrix.append([counts.get(family, 0) for family in top_families] + [other_count])
    _draw_count_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_source_short_label(label) for label in dataset_order],
        col_labels=[_family_short_label(label) for label in columns],
        title="",
        subtitle="",
        output_root=output_root,
        cmap=_FAMILY_HEATMAP,
        colorbar_label="Release crafted records",
        figure_width=4.2,
        color_power_gamma=0.72,
    )


def _evaluation_dimensions_overview_figure(plt: Any, output_root: Path) -> None:
    figure_width = max(DOUBLE_COLUMN_WIDTH, 9.50)
    fig = plt.figure(figsize=(figure_width, 3.76), constrained_layout=False)
    radar_ax = fig.add_axes([0.035, 0.088, 0.315, 0.835], projection="polar")
    tree_ax = fig.add_axes([0.430, 0.088, 0.555, 0.835])

    def _term(
        ax: Any,
        *,
        x: float,
        y: float,
        text: str,
        fontsize: float = 8.75,
        weight: str = "normal",
        color: str = "#111827",
    ) -> None:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            linespacing=1.05,
        )

    def _vline(ax: Any, *, x: float, y0: float, y1: float, color: str = "#6B7280", linewidth: float = 0.58) -> None:
        ax.plot([x, x], [y0, y1], color=color, linewidth=linewidth, alpha=0.86)

    def _polyline(ax: Any, points: list[tuple[float, float]], color: str = "#6B7280", linewidth: float = 0.58) -> None:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.86)

    def _inference(
        ax: Any,
        *,
        x0: float,
        x1: float,
        y: float,
        label: str,
        label_x: float | None = None,
        label_y_offset: float = 0.010,
        linewidth: float = 0.95,
    ) -> None:
        ax.plot([x0, x1], [y, y], color="#111827", linewidth=linewidth, solid_capstyle="butt")
        if label:
            ax.text(
                x1 + 0.008 if label_x is None else label_x,
                y + label_y_offset,
                label,
                ha="left",
                va="bottom",
                fontsize=7.6,
                color="#374151",
            )

    dimensions = [
        "Gate",
        "Detection",
        "Robustness",
        "Utility",
        "HeadlineGeneralization",
        "Stealth",
        "Efficiency",
    ]
    # This is a schematic structural wheel, not a measured radar chart.
    # All axes use the same radius so the panel cannot be read as weights,
    # coefficients, measured values, or method scores. Tiny marker-size changes
    # only cue the headline assembly roles explained by the derivation tree.
    emphasis = [1.00] * len(dimensions)
    marker_sizes = [34, 28, 28, 28, 26, 24, 24]
    angles = [2 * math.pi * index / len(dimensions) for index in range(len(dimensions))]
    closed_angles = angles + angles[:1]
    closed_values = emphasis + emphasis[:1]
    radar_ax.set_theta_offset(math.pi / 2)
    radar_ax.set_theta_direction(-1)
    radar_ax.set_ylim(0.0, 1.34)
    radar_ax.set_yticks([0.42, 0.72, 1.00])
    radar_ax.set_yticklabels(["", "", ""], color="#9AA5B1")
    radar_ax.set_xticks([])
    radar_ax.grid(color="#D1D5DB", linewidth=0.72, alpha=0.78)
    radar_ax.spines["polar"].set_visible(False)
    radar_ax.plot(closed_angles, closed_values, color="#2F5F79", linewidth=1.55)
    for angle in angles:
        radar_ax.plot([angle, angle], [0.0, 1.0], color="#C9D6E2", linewidth=0.72, alpha=0.72, zorder=1)
    radar_ax.scatter(angles, emphasis, s=marker_sizes, color="#2F5F79", edgecolor="white", linewidth=0.55, zorder=3)
    radar_labels = {
        "Gate": "Gate",
        "Detection": "Detection",
        "Robustness": "Robustness",
        "Utility": "Utility",
        "HeadlineGeneralization": "Headline\nGeneralization",
        "Stealth": "Stealth",
        "Efficiency": "Efficiency",
    }
    for angle, label, value in zip(angles, dimensions, emphasis):
        radius = 1.18 if label == "Gate" else 1.25
        if label == "HeadlineGeneralization":
            radius = 1.22
        angle_for_alignment = (math.pi / 2) - angle
        horizontal = "center"
        cosine = math.cos(angle_for_alignment)
        if cosine > 0.25:
            horizontal = "left"
        elif cosine < -0.25:
            horizontal = "right"
        radar_ax.text(
            angle,
            radius,
            radar_labels[label],
            ha=horizontal,
            va="center",
            fontsize=8.55,
            color="#111827",
            linespacing=0.90,
        )

    tree_ax.set_xlim(0.0, 1.0)
    tree_ax.set_ylim(0.0, 1.0)
    tree_ax.axis("off")

    # Bottom-up derivation tree. The mini-rules at the top make the signal-to-
    # metric dependencies explicit; the two wider rules encode the published
    # score assembly without presenting it as additive weighting.
    mini_rules = (
        (0.07, 0.025, 0.115, "pass\n+ neg ctrl", "Gate"),
        (0.22, 0.170, 0.270, "detector\nscores", "Detection"),
        (0.36, 0.300, 0.420, "attack\nretention", "Robustness"),
        (0.50, 0.450, 0.550, "quality\nchecks", "Utility"),
        (0.64, 0.580, 0.700, "detector +\nquality", "Stealth"),
        (0.78, 0.715, 0.845, "runtime\nconditioned", "Efficiency"),
        (0.93, 0.860, 0.995, "available\naxes", "HeadlineGen"),
    )
    premise_y = 0.910
    mini_rule_y = 0.820
    metric_y = 0.720
    for x, x0, x1, premise, conclusion in mini_rules:
        _term(tree_ax, x=x, y=premise_y, text=premise, fontsize=8.2, color="#374151")
        _inference(tree_ax, x0=x0, x1=x1, y=mini_rule_y, label="")
        _term(tree_ax, x=x, y=metric_y, text=conclusion, fontsize=8.35)
        _vline(tree_ax, x=x, y0=mini_rule_y, y1=metric_y + 0.038, color="#111827", linewidth=0.52)

    core_rule_y = 0.535
    for x in (0.22, 0.36, 0.50, 0.64, 0.78):
        _vline(tree_ax, x=x, y0=metric_y - 0.044, y1=core_rule_y)
    _inference(tree_ax, x0=0.165, x1=0.835, y=core_rule_y, label="")

    final_premise_y = 0.330
    summary_rule_y = 0.235
    final_nodes = ((0.07, "Gate"), (0.93, "HeadlineGen"))
    _vline(tree_ax, x=0.07, y0=metric_y - 0.044, y1=final_premise_y + 0.038)
    _vline(tree_ax, x=0.50, y0=core_rule_y, y1=0.452, color="#111827", linewidth=0.62)
    _vline(tree_ax, x=0.50, y0=0.358, y1=summary_rule_y, color="#111827", linewidth=0.62)
    _vline(tree_ax, x=0.93, y0=metric_y - 0.044, y1=final_premise_y + 0.038)
    _term(tree_ax, x=0.500, y=0.405, text="HeadlineCore", fontsize=9.35, weight="semibold")
    for x, text in final_nodes:
        _term(tree_ax, x=x, y=final_premise_y, text=text, fontsize=8.35)
        _vline(tree_ax, x=x, y0=final_premise_y - 0.038, y1=summary_rule_y)
    _inference(tree_ax, x0=0.035, x1=0.965, y=summary_rule_y, label="")
    _term(tree_ax, x=0.500, y=0.125, text="PostEditScore", fontsize=10.35, weight="semibold")
    _vline(tree_ax, x=0.500, y0=summary_rule_y, y1=0.158, color="#111827", linewidth=0.62)

    _save_plot(fig, output_root)
    plt.close(fig)


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scope": row["scope"],
            "slug": row["slug"],
            "dataset_label": row["dataset_label"],
            "source_group": row["source_group"],
            "source_type": row["source_type"],
            "aggregate_score": row["aggregate_score"],
            "execution_slice": row["execution_slice"],
            "active_execution_slice": row["active_execution_slice"],
            "sampling_rule": row["sampling_rule"],
            "scoring_status": row["scoring_status"],
            "record_count": row["record_count"],
            "language_count": row["language_count"],
            "languages": ",".join(row["languages"]),
            "family_count": row["family_count"],
            "category_count": row["category_count"],
            "validation_scope": row["validation_scope"],
            "path": row["path"],
        }
        for row in records
    ]


def _benchmark_definition_rows(release_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for release_record in release_records:
        rows.append(
            {
                "source": str(release_record["dataset_label"]),
                "slug": str(release_record["slug"]),
                "source_group": str(release_record["source_group"]),
                "source_type": str(release_record["source_type"]),
                "active_release_size": int(release_record["record_count"]),
                "scored_in_aggregate": bool(release_record["aggregate_score"]),
                "execution_slice": str(release_record["execution_slice"]),
                "execution_languages": ",".join(release_record["languages"]),
                "sampling_rule": str(release_record["sampling_rule"]),
            }
        )
    return rows


def _source_manifest_index_rows(release_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in release_records:
        canonical_reference_count = int(record["canonical_reference_count"])
        smoke_overlay_reference_count = int(record["smoke_overlay_reference_count"])
        reference_support_mode = "canonical_only" if smoke_overlay_reference_count == 0 else "mixed_canonical_and_smoke_overlay"
        if smoke_overlay_reference_count == 0:
            reference_support_note = "All executed release rows use canonical references."
        else:
            reference_support_note = (
                "Executed release size stays fixed; some rows rely on smoke-overlay reference support for multilingual parity."
            )
        rows.append(
            {
                "source": str(record["dataset_label"]),
                "slug": str(record["slug"]),
                "source_group": str(record["source_group"]),
                "source_type": str(record["source_type"]),
                "data_path": str(record["path"]),
                "manifest_path": str(record["manifest_path"]),
                "record_count": int(record["record_count"]),
                "executed_release_count": int(record["record_count"]),
                "family_count": int(record["family_count"]),
                "languages": ",".join(str(value) for value in record["languages"]),
                "canonical_reference_count": canonical_reference_count,
                "smoke_overlay_reference_count": smoke_overlay_reference_count,
                "reference_support_mode": reference_support_mode,
                "reference_support_note": reference_support_note,
                "sampling_rule": str(record["sampling_rule"]),
            }
        )
    return rows


def _public_release_summary_rows(release_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in release_records:
        rows.append(
            {
                "scope": str(row["scope"]),
                "slug": str(row["slug"]),
                "dataset_label": str(row["dataset_label"]),
                "source_group": str(row["source_group"]),
                "source_type": str(row["source_type"]),
                "aggregate_score": bool(row["aggregate_score"]),
                "execution_slice": str(row["execution_slice"]),
                "active_execution_slice": str(row["active_execution_slice"]),
                "sampling_rule": str(row["sampling_rule"]),
                "scoring_status": str(row["scoring_status"]),
                "record_count": int(row["record_count"]),
                "language_count": int(row["language_count"]),
                "languages": ",".join(str(value) for value in row["languages"]),
                "family_count": int(row["family_count"]),
                "category_count": int(row["category_count"]),
                "validation_scope": str(row["validation_scope"]),
            }
        )
    return rows


def _language_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for language, count in sorted(record["language_counts"].items()):
            rows.append(
                {
                    "scope": record["scope"],
                    "slug": record["slug"],
                    "dataset_label": record["dataset_label"],
                    "language": language,
                    "record_count": int(count),
                }
            )
    return rows


def _category_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["source_type"] != "crafted":
            continue
        labels = dict(record.get("category_labels", {}))
        for category, count in sorted(record["category_counts"].items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "analysis_view": "crafted_only",
                    "scope": record["scope"],
                    "slug": record["slug"],
                    "dataset_label": record["dataset_label"],
                    "category": category,
                    "category_label": labels.get(category, category),
                    "record_count": int(count),
                }
            )
    return rows


def _family_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["source_type"] != "crafted":
            continue
        labels = dict(record.get("template_family_labels", {}))
        for family, count in sorted(record["template_family_counts"].items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "analysis_view": "crafted_only",
                    "scope": record["scope"],
                    "slug": record["slug"],
                    "dataset_label": record["dataset_label"],
                    "source_type": record["source_type"],
                    "template_family": family,
                    "template_family_label": labels.get(family, family),
                    "record_count": int(count),
                    "family_count": int(record["family_count"]),
                }
            )
    return rows


def _remove_stale_release_aliases(*, table_dir: Path, figure_dir: Path) -> None:
    for filename in _STALE_TABLE_FILES:
        path = table_dir / filename
        if path.exists():
            path.unlink()
    for filename in _STALE_FIGURE_FILES:
        path = figure_dir / filename
        if path.exists():
            path.unlink()


def main() -> int:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_release_aliases(table_dir=args.table_dir, figure_dir=args.figure_dir)

    release_records = [_spec_summary(spec) for spec in RELEASE_SPECS]

    release_rows = _public_release_summary_rows(release_records)
    benchmark_definition_rows = _benchmark_definition_rows(release_records)
    language_rows = _language_rows(release_records)
    manifest_index_rows = _source_manifest_index_rows(release_records)
    category_rows = _category_rows(release_records)
    family_rows = _family_rows(release_records)

    _write_json_csv(args.table_dir / "release_slice_summary", release_rows)
    _write_json_csv(args.table_dir / "benchmark_definition_summary", benchmark_definition_rows)
    _write_json_csv(args.table_dir / "release_slice_language_breakdown", language_rows)
    _write_json_csv(args.table_dir / "release_source_manifest_index", manifest_index_rows)
    _write_json_csv(args.table_dir / "dataset_task_category_breakdown", category_rows)
    _write_json_csv(args.table_dir / "dataset_family_breakdown", family_rows)

    _, plt = configure_matplotlib(require_times_new_roman=args.require_times_new_roman)
    _release_slice_composition_figure(plt, release_rows, args.figure_dir / "release_slice_composition")
    _evaluation_dimensions_overview_figure(plt, args.figure_dir / "evaluation_dimensions_overview")

    summary_payload = {
        "active_sources": len(release_rows),
        "table_dir": _display_path(args.table_dir),
        "figure_dir": _display_path(args.figure_dir),
        "manual_review_files": [
            "benchmark_definition_summary.csv",
            "release_slice_language_breakdown.csv",
            "dataset_task_category_breakdown.csv",
            "dataset_family_breakdown.csv",
            "release_source_manifest_index.csv",
            "release_slice_composition.png",
            "evaluation_dimensions_overview.png",
        ],
    }
    dump_json(args.table_dir / "dataset_statistics_manifest.json", summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

