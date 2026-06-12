#!/usr/bin/env python
"""Plot TRGBH0 H0 posteriors against SH0ES and Planck bands."""
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from scipy.stats import gaussian_kde

from trgbh0_plot_style import (
    FIGURE_DPI,
    OUTPUT_DIR,
    TRGBH0_COLOURS,
    TRGBH0_TABLE_RESULTS,
    paper_style,
    save_figure,
)


RESULTS = TRGBH0_TABLE_RESULTS
OUTDIR = OUTPUT_DIR
OUTNAME = "trgbh0_h0_comparison.pdf"

H0_COLOURS = {
    "density_sigv": TRGBH0_COLOURS[0],
    "student_t": TRGBH0_COLOURS[1],
    "planck": TRGBH0_COLOURS[2],
    "shoes": TRGBH0_COLOURS[3],
}

POSTERIORS = [
    (
        r"\texttt{Manticore}, Gaussian",
        RESULTS
        / "EDD_TRGB_rhoSmoothR4_MAS-PCS_sel-TRGB_magnitude_ManticoreLocalCOLA_main.hdf5",
        H0_COLOURS["density_sigv"],
        "-",
    ),
    (
        r"\texttt{Manticore}, Student-$t$",
        RESULTS
        / "EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_ManticoreLocalCOLA_main.hdf5",
        H0_COLOURS["student_t"],
        "--",
    ),
]


REFERENCE_BANDS = [
    ("Planck", 67.4, 0.5, H0_COLOURS["planck"]),
    ("SH0ES", 73.04, 1.04, H0_COLOURS["shoes"]),
]


def read_h0(path):
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["samples/H0"]).reshape(-1)


def kde_line(ax, samples, label, color, fill=False, ls="-", bw=1.0):
    samples = np.asarray(samples).reshape(-1)
    x = np.linspace(np.percentile(samples, 0.1), np.percentile(samples, 99.9),
                    500)
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    y = kde(x)
    ax.plot(x, y, color=color, ls=ls, label=label)
    if fill:
        ax.fill_between(x, 0, y, color=color, alpha=0.20)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    with paper_style(styles=("science",)):
        fig, ax = plt.subplots(figsize=(3.45, 2.55))
        for label, mean, sigma, color in REFERENCE_BANDS:
            ax.axvspan(
                mean - sigma, mean + sigma,
                color=color,
                alpha=0.18,
                lw=0,
                label=rf"{label} $\pm1\sigma$",
                zorder=0,
            )

        for label, path, color, ls in POSTERIORS:
            kde_line(
                ax, read_h0(path), label, color,
                fill=False, ls=ls, bw=1.5)

        ax.set_xlabel(
            r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel("Normalised PDF")
        ax.set_xlim(61.5, 75.8)
        ax.set_ylim(bottom=0)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=6.5,
            handlelength=1.6,
            columnspacing=1.0,
            frameon=False,
        )

        fig.tight_layout()
        save_figure(fig, OUTNAME, output_dir=OUTDIR, dpi=FIGURE_DPI,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {OUTDIR / OUTNAME}")


if __name__ == "__main__":
    main()
