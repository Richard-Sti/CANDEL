#!/usr/bin/env python
"""Paper figure: TRGBH0 single-field smoothed model-variant grid.

Two stacked panels sharing the variant axis: field-median H0 distributions
(top) and matched-field harmonic evidence relative to the best variant
(bottom), over the 80 Manticore realisations of each variant.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(p for p in SCRIPT_DIR.parents if p.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401,E402
from plot_trgbh0_single_smoothed_sets_diagnostics import (  # noqa: E402
    by_field_lnz, grouped_by_set, load_rows, matched_fields)
from trgbh0_plot_style import (PAPER_FIGURE_DIR, ROOT,  # noqa: E402
                               save_pdf_png, set_paper_rc)

DEFAULT_RESULTS_DIR = (
    ROOT / "results" / "TRGBH0_paper" / "single_fields_smoothed")
DEFAULT_OUT = (
    SCRIPT_DIR.parents[1] / "output" / "trgbh0_single_smoothed_sets"
    / "trgbh0_single_smoothed_grid.pdf")
LN10 = np.log(10.0)
H0_LABEL = r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
# Canonical variant order and compact display labels.
ORDER = [
    ("R4 Gauss", "R4\nGauss"),
    ("R4 Gauss sky", "R4\nGauss\nsky"),
    ("R4 Gauss Vmono", "R4\nGauss\nVmono"),
    ("R4 Gauss Vmono sky", "R4\nGauss\nVmono\nsky"),
    ("R4 Stud sky", "R4\nStud\nsky"),
    ("R4 Stud sky beta", "R4\nStud\nsky\n$\\beta$"),
    ("R4 Stud Vmono sky", "R4\nStud\nVmono\nsky"),
    ("R8 Gauss sky", "R8\nGauss\nsky"),
    ("R8 Stud sky", "R8\nStud\nsky"),
]
PLANCK_H0 = 67.4
SHOES_H0 = 73.0


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--paper-figdir", type=Path, default=None,
        help="If set, also copy the PDF into this paper Figures directory.")
    return parser.parse_args()


def build_figure(groups, out_pdf):
    labels = [lab for lab, _ in ORDER if lab in groups]
    disp = [d for lab, d in ORDER if lab in groups]
    pos = np.arange(len(labels))
    fields = matched_fields(groups)
    lnz_map = by_field_lnz(groups)
    mean_lnz = {lab: np.mean([lnz_map[lab][f] for f in fields])
                for lab in labels}
    best = max(mean_lnz, key=mean_lnz.get)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc()
        fig, (ax_h0, ax_z) = plt.subplots(
            2, 1, figsize=(7.0, 5.0), sharex=True,
            constrained_layout=True, height_ratios=(1.25, 1.0))

        medians = [np.asarray([r["H0_q50"] for r in groups[lab]], float)
                   for lab in labels]
        ax_h0.axhspan(PLANCK_H0 - 0.5, PLANCK_H0 + 0.5, color="0.7",
                      alpha=0.30, lw=0)
        ax_h0.axhspan(SHOES_H0 - 1.0, SHOES_H0 + 1.0, color="#ef476f",
                      alpha=0.16, lw=0)
        ax_h0.axhline(PLANCK_H0, color="0.45", lw=0.7, ls="--")
        ax_h0.axhline(SHOES_H0, color="#ef476f", lw=0.7, ls="--")
        parts = ax_h0.violinplot(medians, positions=pos, widths=0.8,
                                 showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#473198")
            body.set_edgecolor("none")
            body.set_alpha(0.32)
        for i, vals in enumerate(medians):
            jitter = np.linspace(-0.16, 0.16, len(vals))
            ax_h0.scatter(i + jitter, vals, s=5, color="#473198",
                          alpha=0.34, edgecolor="none")
            q16, q50, q84 = np.percentile(vals, [16, 50, 84])
            ax_h0.errorbar(i, q50, yerr=[[q50 - q16], [q84 - q50]], fmt="o",
                           color="black", ms=3.6, capsize=2.4, zorder=5)
        ax_h0.text(0.012, 0.96, "Planck", transform=ax_h0.transAxes,
                   ha="left", va="top", fontsize=6.0, color="0.4",
                   fontstyle="italic")
        ax_h0.text(0.012, 0.04, "SH0ES", transform=ax_h0.transAxes,
                   ha="left", va="bottom", fontsize=6.0, color="#ef476f")
        ax_h0.set_ylabel(H0_LABEL)

        for f in fields:
            dz = np.asarray([(lnz_map[lab][f] - lnz_map[best][f]) / LN10
                             for lab in labels])
            ax_z.plot(pos, dz, color="0.6", lw=0.4, alpha=0.30)
        mean_dz = [(mean_lnz[lab] - mean_lnz[best]) / LN10 for lab in labels]
        std_dz = [np.std([(lnz_map[lab][f] - lnz_map[best][f]) / LN10
                          for f in fields], ddof=1) for lab in labels]
        ax_z.errorbar(pos, mean_dz, yerr=std_dz, color="#473198", marker="o",
                      ms=4.0, lw=1.1, capsize=2.4, zorder=5)
        ax_z.axhline(0.0, color="0.35", lw=0.75, ls="--")
        ax_z.set_ylabel(r"$\Delta\log_{10} Z_{\rm harm}$")
        ax_z.set_xticks(pos)
        ax_z.set_xticklabels(disp, fontsize=6.0)
        return save_pdf_png(fig, out_pdf)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results_dir)
    groups = grouped_by_set(rows)
    pdf, png = build_figure(groups, args.out)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    figdir = args.paper_figdir or PAPER_FIGURE_DIR
    if figdir is not None and Path(figdir).is_dir():
        dest = Path(figdir) / pdf.name
        dest.write_bytes(pdf.read_bytes())
        print(f"Copied {dest}")


if __name__ == "__main__":
    main()
