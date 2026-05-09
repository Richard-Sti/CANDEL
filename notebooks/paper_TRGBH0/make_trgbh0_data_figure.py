#!/usr/bin/env python
"""Create the EDD TRGB data-summary figure for the TRGBH0 paper."""
import argparse
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from astropy import units as u
from astropy.coordinates import SkyCoord

from candel import pvdata
from candel.util import SPEED_OF_LIGHT
from trgbh0_plot_style import TRGBH0_COLOURS


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
DATA_ROOT = ROOT / "data" / "EDD_TRGB"
OUTDIR = ROOT / "notebooks" / "paper_TRGBH0"
HIST_OUTNAME = "TRGBH0_data_histograms.pdf"
SKY_OUTNAME = "TRGBH0_sky_distribution.pdf"
SKY_AXIS_FONTSIZE = 8
SKY_TICK_FONTSIZE = 7
SKY_ANNOTATION_FONTSIZE = 9
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
    "M31": {
        "galactic": (121.17, -21.57),
        "marker": "X",
        "size": 28,
        "text_offset": (-4, -5),
        "text_ha": "right",
        "text_va": "top",
    },
    "UMa": {
        "galactic": (145.0, 65.0),
        "marker": "s",
        "size": 22,
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
        help="Optional directory to copy the generated PDFs into.")
    return parser.parse_args()


def _wrap_mollweide_longitude(l_deg):
    """Return Galactic longitude in radians, centred on l=0 deg."""
    lon = np.remainder(l_deg + 180.0, 360.0) - 180.0
    return -np.deg2rad(lon)


def load_data():
    data = pvdata.load_EDD_TRGB(str(DATA_ROOT))
    coords = SkyCoord(ra=data["RA"] * u.deg, dec=data["dec"] * u.deg)
    gal = coords.galactic
    return {
        "czcmb": np.asarray(data["zcmb"]) * SPEED_OF_LIGHT,
        "mag": np.asarray(data["mag"]),
        "ell": np.asarray(gal.l.deg),
        "b": np.asarray(gal.b.deg),
    }


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


def _save_figure(fig, outname, paper_figdir=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / outname
    fig.savefig(out)
    if paper_figdir is not None:
        paper_figdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, paper_figdir / outname)
    plt.close(fig)
    return out


def make_histogram_figure(data, paper_figdir=None):
    with plt.style.context("science"):
        fig, (ax_cz, ax_mag) = plt.subplots(
            2, 1, figsize=(3.35, 3.1), constrained_layout=True,
            sharey=True)
        cz = data["czcmb"]
        mag = data["mag"]

        cz_bins = np.arange(-1000, 2400 + 200, 200)
        ax_cz.hist(cz, bins=cz_bins, color=TRGBH0_COLOURS[0])
        ax_cz.set_ylabel("Counts per bin")
        ax_cz.set_xlabel(r"$cz_{\rm CMB}\ [\mathrm{km}\,\mathrm{s}^{-1}]$")
        ax_cz.set_xlim(-1000, 2400)

        mag_bins = np.arange(17.5, 28.1, 0.5)
        ax_mag.hist(mag, bins=mag_bins, color=TRGBH0_COLOURS[1])
        ax_mag.set_ylabel("Counts per bin")
        ax_mag.set_xlabel(r"$T_{814} - A_{814}\ [\mathrm{mag}]$")
        ax_mag.set_xlim(17.5, 28.0)

    return _save_figure(fig, HIST_OUTNAME, paper_figdir)


def make_sky_figure(data, paper_figdir=None):
    with plt.style.context("science"):
        fig = plt.figure(figsize=(3.35, 2.7), constrained_layout=True)
        ax_sky = fig.add_subplot(111, projection="mollweide")
        lon = _wrap_mollweide_longitude(data["ell"])
        lat = np.deg2rad(data["b"])
        norm = mpl.colors.TwoSlopeNorm(vmin=-800, vcenter=0, vmax=2200)
        scat = ax_sky.scatter(
            lon, lat, c=data["czcmb"], s=7, cmap="coolwarm", norm=norm,
            edgecolor="black", linewidth=0.12, alpha=0.82)
        for name, props in cluster_positions().items():
            ax_sky.scatter(
                props["lon"], props["lat"], marker=props["marker"],
                s=props["size"], facecolor="#ffd60a",
                edgecolor="black", linewidth=0.55,
                zorder=5)
            ax_sky.annotate(
                name, xy=(props["lon"], props["lat"]),
                xytext=props["text_offset"],
                textcoords="offset points", fontsize=SKY_ANNOTATION_FONTSIZE,
                color=SKY_LABEL_COLOUR, fontweight="bold",
                ha=props["text_ha"], va=props["text_va"], zorder=6,
                clip_on=False)
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

    return _save_figure(fig, SKY_OUTNAME, paper_figdir)


def main():
    args = parse_args()
    data = load_data()
    for outname, maker in [
            (HIST_OUTNAME, make_histogram_figure),
            (SKY_OUTNAME, make_sky_figure)]:
        out = maker(data, args.paper_figdir)
        print(f"Wrote {out}")
        if args.paper_figdir is not None:
            print(f"Copied to {args.paper_figdir / outname}")


if __name__ == "__main__":
    main()
