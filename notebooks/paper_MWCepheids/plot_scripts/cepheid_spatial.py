#!/usr/bin/env python3
"""Plot MW Cepheid sample in Galactocentric projections (face-on + edge-on)."""
import os
import sys

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import importlib  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402, F401

from candel import load_config  # noqa: E402
from candel.model.mwcepheids import get_drimmel_arm_traces  # noqa: E402
from candel.pvdata import CepheidData, to_mwcepheids_config  # noqa: E402

importlib.reload(scienceplots)

DELTA_PI = -0.01  # residual parallax zero-point [mas]
R_SUN = 8.122      # Solar Galactocentric distance [kpc]


def main():
    config_path = os.path.join(
        REPO_ROOT, "scripts", "runs", "configs", "config_MWCepheids.toml")
    config = to_mwcepheids_config(
        load_config(config_path, replace_los_prior=False))

    # Load all stars (both C22 and C27)
    config_all = dict(config)
    config_all["data"] = dict(config["data"])
    config_all["data"].pop("which_subset", None)
    data = CepheidData(config_all)

    ell = np.asarray(data.ell)
    b = np.asarray(data.b)
    pi = np.asarray(data.pi_EDR3)

    pi_corr = pi + DELTA_PI
    valid = pi_corr > 0.01
    d = np.where(valid, 1.0 / pi_corr, np.nan)

    ell_r = np.deg2rad(ell)
    b_r = np.deg2rad(b)
    x_hc = d * np.cos(b_r) * np.cos(ell_r)
    y_hc = d * np.cos(b_r) * np.sin(ell_r)
    z_hc = d * np.sin(b_r)

    x_gc = x_hc - R_SUN
    y_gc = y_hc
    z_gc = z_hc
    R_gc = np.sqrt(x_gc**2 + y_gc**2)

    is_c22 = np.asarray(data.is_c22)
    is_c27 = np.asarray(data.is_c27)
    m = valid

    # Spiral arm traces
    arms_xy = get_drimmel_arm_traces(R_sun=R_SUN)

    with plt.style.context("science"):
        fig, axes = plt.subplots(1, 2, figsize=(7.48, 2.6),
                                 gridspec_kw={"width_ratios": [1, 1]})

        col_c22 = "#8B0000"  # dark red
        col_c27 = "#6B8E23"  # olive green
        kw = dict(s=12, alpha=0.85, edgecolors="none", zorder=3)
        sun_kw = dict(marker="*", ms=10, color="gold", zorder=5,
                      markeredgecolor="k", markeredgewidth=0.4,
                      linestyle="none")

        # Face-on: x_gc vs y_gc
        ax = axes[0]
        for i, (xa, ya) in enumerate(arms_xy):
            ax.plot(xa, ya, '.', ms=0.8, alpha=0.5, color='0.45',
                    zorder=1)
        ax.plot([], [], 'o', ms=4, color='0.6', label="Spiral arms")
        ax.scatter(x_gc[m & is_c22], y_gc[m & is_c22],
                   c=col_c22, label="C22", **kw)
        ax.scatter(x_gc[m & is_c27], y_gc[m & is_c27],
                   c=col_c27, label="C27", marker="^", **kw)
        ax.plot(0, 0, "P", ms=6, mew=0.8, color="k", zorder=5,
                linestyle="none", label="GC")
        ax.plot(-R_SUN, 0, label=r"Sun", **sun_kw)
        ax.set_xlim(-17, 7.5)
        ax.set_ylim(-12, 12)
        ax.set_aspect("equal")
        ax.set_anchor("E")
        ax.set_xlabel(r"$x_\mathrm{GC}$ [kpc]")
        ax.set_ylabel(r"$y_\mathrm{GC}$ [kpc]")
        ax.legend(fontsize=7, loc="upper left", ncols=2)

        # Edge-on: R_gc vs z_gc
        ax = axes[1]
        ax.scatter(R_gc[m & is_c22], z_gc[m & is_c22],
                   c=col_c22, label="C22", **kw)
        ax.scatter(R_gc[m & is_c27], z_gc[m & is_c27],
                   c=col_c27, label="C27", marker="^", **kw)
        ax.axhline(0, color="0.7", lw=0.5, zorder=1)
        ax.plot(R_SUN, 0, label=r"Sun", **sun_kw)
        ax.set_xlabel(r"$R_\mathrm{GC}$ [kpc]")
        ax.set_ylabel(r"$z_\mathrm{GC}$ [kpc]")

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.18)
        fout = os.path.join(os.path.dirname(__file__), "cepheid_spatial.pdf")
        fig.savefig(fout, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fout}")


if __name__ == "__main__":
    main()
