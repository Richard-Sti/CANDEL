#!/usr/bin/env python
"""KS diagnostics for EDD TRGB data-availability cuts."""
import argparse
import csv
import math
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/candel_mplconfig")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from edd_trgb_plot_data import DATA_FILE, PAPER_RC, save_figure
from trgbh0_plot_style import TRGBH0_COLOURS


OUTNAME = "edd_trgb_data_availability_ks.pdf"
ANCHORS = {"NGC4258", "NGC4258-DF6"}


def _float_or_nan(value):
    try:
        return float(value)
    except ValueError:
        return math.nan


def _load_rows():
    with DATA_FILE.open(newline="") as handle:
        next(handle)
        rows = []
        for row in csv.DictReader(handle):
            try:
                int(row["pgc"])
            except ValueError:
                continue
            rows.append(row)
    return rows


def _ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, values.size + 1) / values.size


def _finite_dm(rows):
    return np.asarray(
        [_float_or_nan(row["DM_tip"]) for row in rows
         if math.isfinite(_float_or_nan(row["DM_tip"]))],
        dtype=float,
    )


def build_samples():
    rows = [row for row in _load_rows() if row["Name"] not in ANCHORS]
    finite_mag = [
        row for row in rows
        if math.isfinite(_float_or_nan(row["T814"]) - _float_or_nan(row["A_814"]))
    ]
    missing_colour = [
        row for row in finite_mag
        if not math.isfinite(_float_or_nan(row["606-814"]))
    ]
    have_colour = [
        row for row in finite_mag
        if math.isfinite(_float_or_nan(row["606-814"]))
    ]
    missing_redshift = [
        row for row in have_colour
        if abs(_float_or_nan(row["Vcmb"])) >= 9999
    ]
    return {
        "parent": _finite_dm(finite_mag),
        "missing_colour": _finite_dm(missing_colour),
        "missing_redshift": _finite_dm(missing_redshift),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paper-figdir", type=Path, default=None,
        help="Optional directory to copy the generated PDF into.")
    return parser.parse_args()


def _plot_panel(ax_hist, ax_cdf, parent, subset, title, colour):
    bins = np.linspace(21.5, 31.0, 20)
    result = ks_2samp(parent, subset)

    ax_hist.hist(
        parent, bins=bins, histtype="step", density=True, color="0.25",
        linewidth=1.2, label=fr"$\mathrm{{Parent}}\ (N={parent.size})$")
    ax_hist.hist(
        subset, bins=bins, histtype="stepfilled", density=True, color=colour,
        alpha=0.35, linewidth=1.0,
        label=fr"$\mathrm{{Removed}}\ (N={subset.size})$")
    ax_hist.set_title(title)
    ax_hist.set_ylabel(r"$\mathrm{Density}$")
    ax_hist.legend(frameon=False, loc="upper left")
    ax_hist.text(
        0.97, 0.93,
        fr"$D={result.statistic:.3f}$" "\n" fr"$p={result.pvalue:.2g}$",
        transform=ax_hist.transAxes, ha="right", va="top")

    x_parent, y_parent = _ecdf(parent)
    x_subset, y_subset = _ecdf(subset)
    ax_cdf.step(x_parent, y_parent, where="post", color="0.25", linewidth=1.2)
    ax_cdf.step(x_subset, y_subset, where="post", color=colour, linewidth=1.5)
    ax_cdf.set_xlabel(r"$\mu_{\rm EDD}\ \mathrm{[mag]}$")
    ax_cdf.set_ylabel(r"$\mathrm{CDF}$")
    ax_cdf.set_ylim(0, 1.03)


def main():
    args = parse_args()
    samples = build_samples()
    parent = samples["parent"]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            2, 2, figsize=(7.0, 4.7), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.05]})

        _plot_panel(
            axes[0, 0], axes[1, 0], parent, samples["missing_colour"],
            r"$\mathrm{Missing\ F606W{-}F814W\ colour}$",
            TRGBH0_COLOURS[0])
        _plot_panel(
            axes[0, 1], axes[1, 1], parent, samples["missing_redshift"],
            r"$\mathrm{Missing\ redshift}$", TRGBH0_COLOURS[1])

        for ax in axes.ravel():
            ax.tick_params(direction="in", top=True, right=True)

        fig.tight_layout()
    out = save_figure(fig, OUTNAME, args.paper_figdir)
    plt.close(fig)
    print(f"Wrote {out}")
    if args.paper_figdir is not None:
        print(f"Copied to {args.paper_figdir}/{OUTNAME}")

    for name in ("missing_colour", "missing_redshift"):
        result = ks_2samp(parent, samples[name])
        print(
            f"{name}: N={samples[name].size}, "
            f"D={result.statistic:.6f}, p={result.pvalue:.6g}")


if __name__ == "__main__":
    main()
