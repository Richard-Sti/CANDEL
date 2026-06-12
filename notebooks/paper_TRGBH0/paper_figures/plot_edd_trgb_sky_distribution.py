#!/usr/bin/env python
"""Create the EDD TRGB sky-distribution figure."""
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
from astropy import units as u
from astropy.coordinates import SkyCoord

from edd_trgb_plot_data import PAPER_RC, load_edd_trgb_plot_data, save_figure


OUTNAME = "edd_trgb_sky_distribution.pdf"
SKY_AXIS_FONTSIZE = 8
SKY_TICK_FONTSIZE = 7
SKY_LABEL_COLOUR = "#7b3294"
CLUSTERS = {
    "Virgo": {
        "centre": (187.70593, 12.39112),
        "marker": "*",
        "size": 42,
        "text_offset": (4, 4),
        "text_ha": "left",
        "text_va": "bottom",
    },
    "Fornax": {
        "centre": (54.62125, -35.45066),
        "marker": "D",
        "size": 24,
        "text_offset": (5, -11),
        "text_ha": "center",
        "text_va": "top",
    },
    "Cen A": {
        "galactic": (309.515874, 19.417325),
        "marker": "^",
        "size": 26,
        "text_offset": (4, 3),
        "text_ha": "left",
        "text_va": "bottom",
    },
    "UMa": {
        "galactic": (145.0, 65.0),
        "marker": "s",
        "size": 22,
        "text_offset": (4, -3),
        "text_ha": "left",
        "text_va": "top",
    },
    "M31": {
        "galactic": (121.174322, -21.572969),
        "marker": "P",
        "size": 28,
        "text_offset": (4, -3),
        "text_ha": "left",
        "text_va": "top",
    },
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


def _wrap_mollweide_longitude(l_deg):
    """Return Galactic longitude in radians, centred on l=0 deg."""
    lon = np.remainder(l_deg + 180.0, 360.0) - 180.0
    return -np.deg2rad(lon)


def cluster_positions():
    out = {}
    for name, props in CLUSTERS.items():
        if "galactic" in props:
            ell, b = props["galactic"]
        else:
            ra, dec = props["centre"]
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
            ell, b = coord.l.deg, coord.b.deg
        out[name] = {
            **props,
            "lon": _wrap_mollweide_longitude(ell),
            "lat": np.deg2rad(b),
        }
    return out


def make_figure(data, paper_figdir=None):
    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(3.35, 2.7), constrained_layout=True)
        ax_sky = fig.add_subplot(111, projection="mollweide")
        lon = _wrap_mollweide_longitude(data["ell"])
        lat = np.deg2rad(data["b"])
        norm = mpl.colors.TwoSlopeNorm(vmin=-800, vcenter=0, vmax=2200)
        scat = ax_sky.scatter(
            lon, lat, c=data["czcmb"], s=9, cmap="coolwarm", norm=norm,
            edgecolor="black", linewidth=0.12, alpha=0.82)
        legend_handles = []
        legend_labels = []
        for name, props in cluster_positions().items():
            handle = ax_sky.scatter(
                props["lon"], props["lat"], marker=props["marker"],
                s=props["size"], facecolor="#ffd60a",
                edgecolor="black", linewidth=0.55,
                zorder=5)
            legend_handles.append(handle)
            legend_labels.append(name)
        ax_sky.legend(
            legend_handles, legend_labels, ncol=3, frameon=False,
            loc="lower center", bbox_to_anchor=(0.5, 1.02),
            fontsize=SKY_AXIS_FONTSIZE, handletextpad=0.4,
            columnspacing=1.2, labelcolor=SKY_LABEL_COLOUR)
        ax_sky.grid(alpha=0.45)
        ax_sky.set_xlabel(r"$\ell$", fontsize=SKY_AXIS_FONTSIZE)
        ax_sky.set_ylabel(r"$b$", fontsize=SKY_AXIS_FONTSIZE)
        ax_sky.set_xticks(np.deg2rad([0, 120, -120]))
        ax_sky.set_xticklabels([])
        for x, label in zip(
                np.deg2rad([0, 120, -120]),
                [r"$0^\circ$", r"$240^\circ$", r"$120^\circ$"]):
            ax_sky.text(
                x, np.deg2rad(-6), label, ha="center", va="top",
                fontsize=SKY_TICK_FONTSIZE, color="black")
        ax_sky.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
        ax_sky.set_yticklabels([
            r"$-60^\circ$", r"$-30^\circ$", r"$0^\circ$",
            r"$30^\circ$", r"$60^\circ$"])
        ax_sky.tick_params(labelsize=SKY_TICK_FONTSIZE)
        for label in ax_sky.get_yticklabels():
            label.set_x(0.012)
        cbar = fig.colorbar(scat, ax=ax_sky, orientation="horizontal", pad=0.08,
                            fraction=0.06, aspect=30)
        cbar.set_label(
            r"$cz_{\rm CMB}\ [\mathrm{km}\,\mathrm{s}^{-1}]$",
            fontsize=SKY_AXIS_FONTSIZE)
        cbar.ax.tick_params(labelsize=SKY_TICK_FONTSIZE)

    out = save_figure(fig, OUTNAME, paper_figdir)
    plt.close(fig)
    return out


def main():
    args = parse_args()
    out = make_figure(load_edd_trgb_plot_data(), args.paper_figdir)
    print(f"Wrote {out}")
    if args.paper_figdir is not None:
        print(f"Copied to {args.paper_figdir / OUTNAME}")


if __name__ == "__main__":
    main()
