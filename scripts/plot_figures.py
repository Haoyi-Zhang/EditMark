#!/usr/bin/env python3
"""Draw the evidence contract and two vector plots from retained exports."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]

def configure_fonts(font_dir: Path | None = None) -> str:
    """Use the bundled, unmodified OFL Libertine TTFs; never a silent fallback."""
    from matplotlib import font_manager
    directory = font_dir or ROOT / "fonts"
    regular = directory / "LinLibertine_Rah.ttf"
    italic = directory / "LinLibertine_RIah.ttf"
    if regular.is_file() and italic.is_file():
        for path in (regular, italic):
            font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(regular)).get_name()
    else:
        family = "Linux Libertine"
        for style in ("normal", "italic"):
            font_manager.findfont(
                font_manager.FontProperties(family=family, style=style),
                fallback_to_default=False,
            )
    matplotlib.rcParams.update({
        "font.family": family, "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False,
        "text.usetex": False, "mathtext.fontset": "custom",
        "mathtext.rm": family, "mathtext.it": family + ":italic",
        "mathtext.bf": family, "mathtext.sf": family, "mathtext.tt": family,
        "mathtext.cal": family, "mathtext.fallback": None,
        "axes.unicode_minus": True, "axes.linewidth": 0.55,
        "xtick.major.width": 0.45, "ytick.major.width": 0.45,
        "xtick.major.size": 3, "ytick.major.size": 3,
    })
    resolved = font_manager.findfont(
        font_manager.FontProperties(family=family), fallback_to_default=False)
    print("FIGURE_FONT", resolved)
    return family

METHODS = ["KGW", "SWEET", "STONE", "EWD"]
EDIT_ORDER = [
    "whitespace_normalize", "comment_strip", "noise_insert", "identifier_rename",
    "block_shuffle", "control_flow_flatten", "budgeted_adaptive",
]
EDIT_LABELS = [
    "Whitespace", "Comment removal", "Comment insertion", "Identifier renaming",
    "Block shuffle", "Control-flow edit", "Budgeted adaptive",
]
MARKERS = ["o", "s", "^", "d"]
COLORS = ["#00008c", "#737373", "#000000", "#000000"]


def figure_data(root: Path = ROOT) -> tuple[list[str], dict, dict]:
    """Keep the existing descriptive denominators and retained precision."""
    path = root / "results/tables/suite_all_models_methods/method_attack_summary.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    order = EDIT_ORDER.copy()
    present = {row["attack"] for row in rows}
    if order[-1] not in present:
        matches = [name for name in present if "adaptive" in name]
        if len(matches) != 1:
            raise ValueError(f"Unexpected attack names: {sorted(present)!r}")
        order[-1] = matches[0]
    expected = {(method, edit) for method in METHODS for edit in order}
    def rate(value, field):
        if isinstance(value, bool):
            raise ValueError(f'{field}: a finite rate in [0,1] is required')
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f'{field}: a finite rate in [0,1] is required')
        return number
    passes = {}
    for row in rows:
        key = (row['method'], row['attack'])
        if key in passes:
            raise ValueError(f'Duplicate retained method/edit cell: {key}')
        passes[key] = 100 * rate(row['attacked_test_pass_rate'], 'attacked_test_pass_rate')
    components = json.loads((root / "analysis/edit_component_audit.json")
                            .read_text(encoding="utf-8"))
    coverage = {}
    for row in components:
        key = (row['method'], row['edit'])
        if key in coverage:
            raise ValueError(f'Duplicate retained method/edit cell: {key}')
        total, supported = row['total_rows'], row['supported_rows']
        if (type(total) is not int or type(supported) is not int
                or total <= 0 or not 0 <= supported <= total):
            raise ValueError(f'Invalid coverage integer counts: {key}')
        value = rate(row['application_coverage'], 'application_coverage')
        if not math.isclose(value, supported / total, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'application coverage disagrees with retained counts: {key}')
        coverage[key] = 100 * value
    if set(passes) != expected or set(coverage) != expected:
        raise ValueError('Missing or unexpected retained method/edit cells')
    return order, passes, coverage


def build_contract(root: Path = ROOT, font_dir: Path | None = None) -> list[Path]:
    """Draw the four-stage explanatory diagram, without measured quantities."""
    configure_fonts(font_dir)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paper = root.parent / "paper/tosem/figures"
    fig = plt.figure(figsize=(342 / 72, 80 / 72))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 342), ylim=(0, 80))
    ax.axis("off")
    centers = [38.5 + 88.2 * i for i in range(4)]
    labels = [
        ("1. Typed\nobservation", "raw pairs / counts\ntest contract\ndetector rule"),
        ("2. Eligibility gate", "supported\nand changed\nboth checks pass\nsame rule"),
        ("3. Sufficient facts", "n00, n01, n10, n11\nexclusion partition\ncohort fingerprint"),
        ("4. Claim\ncertificate", "identified / bounded\nnot admissible\nbasis re-derivation"),
    ]
    artists = []
    for x, (title, facts) in zip(centers, labels):
        ax.add_patch(Rectangle((x - 38, 1.5), 76, 77,
                               facecolor="white", edgecolor="black", linewidth=.6))
        artists.append(ax.text(x, 63, title, ha="center", va="center",
                               fontsize=8.5, linespacing=1.1))
        artists.append(ax.text(x, 30, facts, ha="center", va="center",
                               fontsize=8.5, linespacing=1.15))
    for a, b in zip(centers, centers[1:]):
        ax.add_patch(FancyArrowPatch((a + 38, 40), (b - 38, 40),
                                    arrowstyle="-|>", mutation_scale=7,
                                    linewidth=.6, color="black", shrinkA=0, shrinkB=0))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in artists:
        box = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        x = artist.get_position()[0]
        if box.x0 < x - 36 or box.x1 > x + 36 or box.y0 < 3 or box.y1 > 77:
            raise ValueError("Contract label does not fit: " + artist.get_text())
    assets = []
    for suffix in ("pdf", "png"):
        path = output / ("evidence_contract." + suffix)
        metadata = {"Creator": None, "Producer": None, "CreationDate": None,
                    "ModDate": None} if suffix == "pdf" else None
        fig.savefig(path, dpi=300, metadata=metadata)
        if paper.parent.is_dir():
            paper.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, paper / path.name)
        assets.append(path)
    plt.close(fig)
    return assets


def build_figures(root: Path = ROOT, font_dir: Path | None = None) -> list[Path]:
    configure_fonts(font_dir)
    order, passes, coverage = figure_data(root)
    output = root / "figures"
    paper = root.parent / "paper/tosem/figures"
    output.mkdir(parents=True, exist_ok=True)
    if paper.parent.is_dir():
        paper.mkdir(parents=True, exist_ok=True)
    assets = build_contract(root, font_dir)
    specs = [
        ("edit_functionality", passes, (0, 13.5), list(range(0, 14, 2)),
         "Edited-program test-pass fraction in method-edit export (%)"),
        ("application_coverage", coverage, (-2, 103), list(range(0, 101, 20)),
         "Rows reported applicable / attempted edit rows (%)"),
    ]
    for name, values, xlim, xticks, xlabel in specs:
        # Prior outer box: 411.15 x 190.40 bp; add 3.85 bp of side padding.
        # The 340.73 x 147.92 bp plotting rectangle is not enlarged.
        fig = plt.figure(figsize=(415 / 72, 190.4 / 72))
        ax = fig.add_axes([72.4 / 415, 27.4 / 190.4, 340.73 / 415, 147.92 / 190.4])
        for j, method in enumerate(METHODS):
            ax.plot(
                [values[(method, edit)] for edit in order],
                [i + (j - 1.5) * 0.10 for i in range(7)],
                linestyle="none", marker=MARKERS[j], markersize=3.4,
                color=COLORS[j], markerfacecolor="white" if j == 3 else COLORS[j],
                markeredgewidth=0.45, label=method,
            )
        upper = max(xlim[1], max(values.values()) * 1.03)
        ax.set(xlim=(xlim[0], upper), ylim=(6.45, -0.45), xlabel=xlabel)
        if upper == xlim[1]:
            ax.set_xticks(xticks)
        ax.set_yticks(range(7), EDIT_LABELS)
        ax.tick_params(direction="in", pad=3, labelsize=8.47)
        ax.xaxis.label.set_size(8.47)
        ax.xaxis.labelpad = 4
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#d9d9d9", linestyle=":", linewidth=0.35)
        ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.014),
                  frameon=False, borderaxespad=0, handletextpad=0.35,
                  columnspacing=0.8, handlelength=1.1, fontsize=8.47)
        for suffix in ('pdf', 'png'):
            path = output / f'{name}.{suffix}'
            metadata = {"Creator": None, "Producer": None,
                        "CreationDate": None, "ModDate": None} if suffix == 'pdf' else None
            fig.savefig(path, dpi=300, metadata=metadata)
            if paper.is_dir():
                shutil.copyfile(path, paper / path.name)
            assets.append(path)
        plt.close(fig)
    return assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--contract-only", action="store_true",
                        help="redraw only the explanatory contract diagram")
    args = parser.parse_args()
    producer = build_contract if args.contract_only else build_figures
    for path in producer(font_dir=args.font_dir):
        print("FIGURE_OK", path)


if __name__ == "__main__":
    main()
