#!/usr/bin/env python3
"""Reproduce two vector manuscript plots from retained audit exports only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    passes = {
        (row["method"], row["attack"]): 100.0 * float(row["attacked_test_pass_rate"])
        for row in rows
    }
    components = json.loads((root / "analysis/edit_component_audit.json")
                            .read_text(encoding="utf-8"))
    coverage = {
        (row["method"], row["edit"]): 100.0 * row["application_coverage"]
        for row in components
    }
    for method in METHODS:
        for edit in order:
            if (method, edit) not in passes or (method, edit) not in coverage:
                raise ValueError(f"Missing retained method/edit cell: {method}/{edit}")
    return order, passes, coverage


def build_figures(root: Path = ROOT, font_dir: Path | None = None) -> list[Path]:
    configure_fonts(font_dir)
    order, passes, coverage = figure_data(root)
    output = root / "figures"
    paper = root.parent / "paper/tosem/figures"
    output.mkdir(parents=True, exist_ok=True)
    if paper.parent.is_dir():
        paper.mkdir(parents=True, exist_ok=True)
    assets = []
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
        ax.set(xlim=xlim, ylim=(6.45, -0.45), xlabel=xlabel)
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
        path = output / f"{name}.pdf"
        fig.savefig(path, metadata={"Creator": None, "Producer": None,
                                    "CreationDate": None, "ModDate": None})
        plt.close(fig)
        if paper.is_dir():
            shutil.copyfile(path, paper / path.name)
        assets.append(path)
    return assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    for path in build_figures(font_dir=args.font_dir):
        print("FIGURE_OK", path)


if __name__ == "__main__":
    main()
