#!/usr/bin/env python
"""Plot Galactic latitude distributions for the EDD TRGB hosts."""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from make_edd_trgb_ppc import CONFIG, load_observed_data
from trgbh0_plot_style import OUTPUT_DIR, paper_style, save_pdf_png

import candel

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/candel_mplconfig")


matplotlib.use("Agg")
import scienceplots  # noqa: E402,F401

OUTDIR = OUTPUT_DIR / "model_checks"
SIGNED_OUT = OUTDIR / "edd_trgb_host_galactic_latitude_b.pdf"
ABS_OUT = OUTDIR / "edd_trgb_host_galactic_latitude_abs_b.pdf"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plane-band", type=float, default=10.0,
        help="Shade the low-latitude band |b| below this value in degrees.")
    parser.add_argument(
        "--signed-output", type=Path, default=SIGNED_OUT,
        help="Output path for the signed-b distribution.")
    parser.add_argument(
        "--abs-output", type=Path, default=ABS_OUT,
        help="Output path for the absolute-|b| distribution.")
    return parser.parse_args()


def host_latitudes():
    config = candel.load_config(str(CONFIG), replace_los_prior=False)
    data = load_observed_data(config)
    _, b = candel.radec_to_galactic(data["RA_host"], data["dec_host"])
    return np.asarray(b, dtype=float)


def plot_signed_b(b, out, plane_band):
    bins = np.linspace(-90.0, 90.0, 37)
    with paper_style():
        fig, ax = plt.subplots(figsize=(3.42, 2.35), constrained_layout=True)
        ax.hist(b, bins=bins, histtype="stepfilled",
                color="#2C7FB8", alpha=0.25, edgecolor="#2C7FB8",
                linewidth=0.9)
        if plane_band > 0:
            ax.axvspan(-plane_band, plane_band, color="0.1", alpha=0.10,
                       linewidth=0)
            ax.axvline(-plane_band, color="0.25", linestyle=":",
                       linewidth=0.7)
            ax.axvline(plane_band, color="0.25", linestyle=":",
                       linewidth=0.7)
        ax.axvline(0.0, color="0.15", linewidth=0.7)
        ax.set_xlim(-90.0, 90.0)
        ax.set_xlabel(r"Galactic latitude $b$ [deg]")
        ax.set_ylabel("Number of hosts")
        ax.tick_params(direction="in", which="both", top=True, right=True)
    return save_pdf_png(fig, out)


def plot_abs_b(b, out, plane_band):
    abs_b = np.abs(b)
    bins = np.linspace(0.0, 90.0, 19)
    with paper_style():
        fig, ax = plt.subplots(figsize=(3.42, 2.35), constrained_layout=True)
        ax.hist(abs_b, bins=bins, histtype="stepfilled",
                color="#EF476F", alpha=0.25, edgecolor="#EF476F",
                linewidth=0.9)
        if plane_band > 0:
            ax.axvspan(0.0, plane_band, color="0.1", alpha=0.10,
                       linewidth=0)
            ax.axvline(plane_band, color="0.25", linestyle=":",
                       linewidth=0.7)
        ax.set_xlim(0.0, 90.0)
        ax.set_xlabel(r"Absolute Galactic latitude $|b|$ [deg]")
        ax.set_ylabel("Number of hosts")
        ax.tick_params(direction="in", which="both", top=True, right=True)
    return save_pdf_png(fig, out)


def main():
    args = parse_args()
    b = host_latitudes()
    signed_pdf, signed_png = plot_signed_b(
        b, args.signed_output, args.plane_band)
    abs_pdf, abs_png = plot_abs_b(b, args.abs_output, args.plane_band)

    abs_b = np.abs(b)
    print(f"n_hosts: {len(b)}")
    print(f"b median: {np.median(b):.2f} deg")
    print(f"|b| median: {np.median(abs_b):.2f} deg")
    if args.plane_band > 0:
        frac = np.mean(abs_b < args.plane_band)
        print(f"fraction with |b| < {args.plane_band:g} deg: {frac:.3f}")
    print(f"wrote: {signed_pdf}")
    print(f"wrote: {signed_png}")
    print(f"wrote: {abs_pdf}")
    print(f"wrote: {abs_png}")


if __name__ == "__main__":
    main()
