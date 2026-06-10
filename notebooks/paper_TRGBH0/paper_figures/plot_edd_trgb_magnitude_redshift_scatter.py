#!/usr/bin/env python
"""Create the EDD TRGB tip-magnitude versus CMB-frame velocity figure."""
import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/candel_mplconfig")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from edd_trgb_plot_data import OUTDIR, PAPER_RC, load_edd_trgb_plot_data
from trgbh0_plot_style import TRGBH0_COLOURS


OUTNAME = "edd_trgb_magnitude_redshift_scatter"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paper-figdir", type=Path, default=None,
        help="Optional directory to copy the generated PDF into.")
    return parser.parse_args()


def _axis_limits(values, pad_fraction=0.05):
    lo = np.min(values)
    hi = np.max(values)
    pad = pad_fraction * (hi - lo)
    return lo - pad, hi + pad


def make_figure(data):
    czcmb = data["czcmb"]
    mag = data["mag"]

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(3.35, 3.1))
        gs = fig.add_gridspec(
            2, 2, width_ratios=(4.0, 1.0), height_ratios=(1.0, 4.0),
            hspace=0.0, wspace=0.0)
        fig.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.94)
        ax_hist_cz = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0], sharex=ax_hist_cz)
        ax_hist_mag = fig.add_subplot(gs[1, 1], sharey=ax)

        cz_limits = _axis_limits(czcmb)
        mag_limits = _axis_limits(mag)

        ax.scatter(
            czcmb, mag, s=12, color=TRGBH0_COLOURS[0],
            edgecolor="black", linewidth=0.15, alpha=0.82)
        ax.axvline(0.0, color="0.45", lw=0.8, zorder=0)
        ax.set_xlim(*cz_limits)
        ax.set_ylim(*mag_limits)
        ax.set_xlabel(r"$cz_{\rm CMB}\ [\mathrm{km}\,\mathrm{s}^{-1}]$")
        ax.set_ylabel(r"$T_{814} - A_{814}\ [\mathrm{mag}]$")
        ax.text(
            0.04, 0.95, rf"$N={len(czcmb)}$",
            transform=ax.transAxes, ha="left", va="top")

        cz_bins = np.arange(-1000, 2400 + 200, 200)
        ax_hist_cz.hist(czcmb, bins=cz_bins, color=TRGBH0_COLOURS[0])
        ax_hist_cz.axvline(0.0, color="0.45", lw=0.8, zorder=0)
        ax_hist_cz.set_xlim(*cz_limits)

        mag_bins = np.arange(17.5, 28.1, 0.5)
        ax_hist_mag.hist(
            mag, bins=mag_bins, orientation="horizontal",
            color=TRGBH0_COLOURS[1])
        ax_hist_mag.set_ylim(*mag_limits)

        ax_hist_cz.tick_params(axis="x", labelbottom=False)
        ax_hist_cz.tick_params(axis="y", left=False, labelleft=False)
        ax_hist_mag.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_hist_mag.tick_params(axis="y", left=False, labelleft=False)
        ax_hist_cz.spines["top"].set_visible(False)
        ax_hist_cz.spines["right"].set_visible(False)
        ax_hist_cz.spines["left"].set_visible(False)
        ax_hist_mag.spines["top"].set_visible(False)
        ax_hist_mag.spines["right"].set_visible(False)
        ax_hist_mag.spines["bottom"].set_visible(False)

        fig.align_ylabels([ax])
    return fig


def save_figure(fig, paper_figdir=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".png"):
        out = OUTDIR / f"{OUTNAME}{suffix}"
        fig.savefig(out, dpi=250 if suffix == ".png" else None)
        outputs.append(out)
    if paper_figdir is not None:
        paper_figdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUTDIR / f"{OUTNAME}.pdf", paper_figdir / f"{OUTNAME}.pdf")
    return outputs


def main():
    args = parse_args()
    data = load_edd_trgb_plot_data(include_sky=False)
    fig = make_figure(data)
    outputs = save_figure(fig, args.paper_figdir)
    plt.close(fig)

    print(f"Plotted {len(data['czcmb'])} EDD TRGB hosts.")
    print(
        f"czcmb range = {np.min(data['czcmb']):.1f} to "
        f"{np.max(data['czcmb']):.1f} km/s")
    print(
        f"magnitude range = {np.min(data['mag']):.3f} to "
        f"{np.max(data['mag']):.3f} mag")
    for out in outputs:
        print(f"Wrote {out}")
    if args.paper_figdir is not None:
        print(f"Copied to {args.paper_figdir / f'{OUTNAME}.pdf'}")


if __name__ == "__main__":
    main()
