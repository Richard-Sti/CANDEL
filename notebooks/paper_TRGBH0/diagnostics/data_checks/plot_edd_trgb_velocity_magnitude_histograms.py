#!/usr/bin/env python
"""Create EDD TRGB velocity and tip-magnitude histograms."""
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/candel_mplconfig")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from edd_trgb_plot_data import PAPER_RC, load_edd_trgb_plot_data, save_figure
from trgbh0_plot_style import TRGBH0_COLOURS


OUTNAME = "edd_trgb_velocity_magnitude_histograms.pdf"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paper-figdir", type=Path, default=None,
        help="Optional directory to copy the generated PDF into.")
    return parser.parse_args()


def make_figure(data, paper_figdir=None):
    with plt.rc_context(PAPER_RC):
        fig, (ax_cz, ax_mag) = plt.subplots(
            2, 1, figsize=(3.35, 3.1), constrained_layout=True,
            sharey=True)

        cz_bins = np.arange(-1000, 2400 + 200, 200)
        ax_cz.hist(data["czcmb"], bins=cz_bins, color=TRGBH0_COLOURS[0])
        ax_cz.set_ylabel(r"$\mathrm{Counts\ per\ bin}$")
        ax_cz.set_xlabel(r"$cz_{\rm CMB}\ [\mathrm{km}\,\mathrm{s}^{-1}]$")
        ax_cz.set_xlim(-1000, 2400)

        mag_bins = np.arange(17.5, 28.1, 0.5)
        ax_mag.hist(data["mag"], bins=mag_bins, color=TRGBH0_COLOURS[1])
        ax_mag.set_ylabel(r"$\mathrm{Counts\ per\ bin}$")
        ax_mag.set_xlabel(r"$T_{814} - A_{814}\ [\mathrm{mag}]$")
        ax_mag.set_xlim(17.5, 28.0)

    out = save_figure(fig, OUTNAME, paper_figdir)
    plt.close(fig)
    return out


def main():
    args = parse_args()
    out = make_figure(load_edd_trgb_plot_data(include_sky=False), args.paper_figdir)
    print(f"Wrote {out}")
    if args.paper_figdir is not None:
        print(f"Copied to {args.paper_figdir / OUTNAME}")


if __name__ == "__main__":
    main()
