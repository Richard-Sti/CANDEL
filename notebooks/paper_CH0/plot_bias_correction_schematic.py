# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Illustrate external bias corrections for a magnitude-limited SN sample.

The figure is a slide schematic rather than an analysis product.  It first
draws a parent Hubble diagram and applies a soft apparent-magnitude limit.
The detected SNe then receive deliberately exaggerated distance-modulus
corrections before the final likelihood is evaluated.

Examples:
    python plot_bias_correction_schematic.py
    python plot_bias_correction_schematic.py --formats pdf
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from os import makedirs
from os.path import join


def _heavy_imports():
    """Defer plotting imports so ``--help`` returns instantly."""
    global np, plt

    import matplotlib
    matplotlib.use("Agg")

    import numpy as np
    import matplotlib.pyplot as plt


PLOTS_DIR = "/Users/rstiskalek/Projects/CANDEL/plots/paper_CH0"

C_LIGHT = 299792.458
H0 = 70.0
Q0 = -0.55
M_SN = -19.25
M_LIM = 18.0
SCATTER_MU = 0.24
SELECTION_WIDTH = 0.12
BIASED_SCATTER = 0.105
CORRECTED_SCATTER = 0.115

Z_MIN = 0.012
Z_MAX = 0.11

COL_TRUTH = "#202020"
COL_OBS = "#1e42b9"
COL_CORR = "#87193d"
COL_MISSING = "#b6b6b6"
COL_LIMIT = "#d42a29"


def distance_modulus(z):
    """Low-redshift luminosity-distance approximation in magnitudes."""
    d_lum_mpc = C_LIGHT / H0 * z * (1.0 + 0.5 * (1.0 - Q0) * z)
    return 5.0 * np.log10(d_lum_mpc) + 25.0


def draw_parent_sample(n_parent, seed):
    """Draw a volume-weighted parent SN sample with intrinsic scatter."""
    rng = np.random.default_rng(seed)
    z = (rng.uniform(size=n_parent) * (Z_MAX**3 - Z_MIN**3) + Z_MIN**3)
    z = z**(1.0 / 3.0)
    mu_true = distance_modulus(z)
    mu_hat = mu_true + rng.normal(0.0, SCATTER_MU, size=n_parent)
    apparent_mag = mu_hat + M_SN
    p_detect = 1.0 / (
        1.0 + np.exp((apparent_mag - M_LIM) / SELECTION_WIDTH)
    )
    detected = rng.uniform(size=n_parent) < p_detect
    return z, mu_true, mu_hat, detected


def assigned_mu_correction(z):
    """Schematic positive correction applied to detected distance moduli."""
    turn_on = 1.0 / (1.0 + np.exp(-(z - 0.055) / 0.012))
    return 0.04 + 0.50 * turn_on


def plot_bias_correction_schematic(n_parent, seed, savedir, formats):
    _heavy_imports()

    z, mu_true, mu_hat, detected = draw_parent_sample(n_parent, seed)
    z_obs = z[detected]
    mu_true_obs = mu_true[detected]
    rng = np.random.default_rng(seed + 10_000)
    mu_corr_delta = assigned_mu_correction(z_obs)
    mu_obs = mu_true_obs - mu_corr_delta + rng.normal(0.0, BIASED_SCATTER,
                                                      size=len(z_obs))
    mu_corr = mu_true_obs + rng.normal(0.0, CORRECTED_SCATTER,
                                       size=len(z_obs))

    z_grid = np.linspace(Z_MIN, Z_MAX, 400)
    mu_grid = distance_modulus(z_grid)
    mu_limit = M_LIM - M_SN

    with plt.style.context("default"):
        fig = _make_figure(
            z, mu_hat, detected, z_obs, mu_obs, mu_corr,
            z_grid, mu_grid, mu_limit)

    makedirs(savedir, exist_ok=True)
    outpaths = []
    for fmt in formats:
        out = join(savedir, f"bias_correction_schematic.{fmt}")
        fig.savefig(out, dpi=450, bbox_inches="tight")
        outpaths.append(out)
    plt.close(fig)
    return outpaths


def _make_figure(
        z, mu_hat, detected, z_obs, mu_obs, mu_corr,
        z_grid, mu_grid, mu_limit):
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    fig, ax = plt.subplots(figsize=(4.7, 3.0), constrained_layout=True)

    missing = ~detected
    ax.fill_between(
        z_grid, mu_limit, 39.0, color=COL_LIMIT, alpha=0.08, lw=0)
    ax.scatter(
        z[missing], mu_hat[missing], s=22, marker="x", lw=0.8,
        color=COL_MISSING, alpha=0.55, label="Missing parent SNe")
    ax.plot(
        z_grid, mu_grid, color=COL_TRUTH, lw=1.7,
        label="Unbiased relation")
    ax.axhline(
        mu_limit, color=COL_LIMIT, ls="--", lw=1.2,
        label=r"Magnitude limit")
    ax.scatter(
        z_obs, mu_obs, s=28, color=COL_OBS, alpha=0.92,
        edgecolor="white", linewidth=0.35, label=r"Biased $\hat{\mu}$")

    for x, y0, y1 in zip(z_obs, mu_obs, mu_corr):
        ax.annotate(
            "", xy=(x, y1), xytext=(x, y0),
            arrowprops=dict(
                arrowstyle="-|>", lw=1.0, color=COL_CORR, alpha=0.65,
                shrinkA=3, shrinkB=4, mutation_scale=8))

    ax.scatter(
        z_obs, mu_corr, s=36, marker="o", color=COL_CORR, alpha=0.88,
        edgecolor="white", linewidth=0.35, label=r"Assigned $\mu_{\rm corr}$")

    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_ylim(33.4, 38.6)
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"Distance modulus $\mu$")
    ax.tick_params(axis="both")
    ax.grid(alpha=0.18, lw=0.5)
    ax.legend(loc="lower right", fontsize=7.2, frameon=False)
    return fig


def main():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--n-parent", type=int, default=90,
                        help="Number of simulated parent SNe. Default: 90.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed. Default: 7.")
    parser.add_argument("--savedir", default=PLOTS_DIR,
                        help=f"Output directory. Default: {PLOTS_DIR}.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"],
                        choices=["pdf", "png", "svg"],
                        help="Output formats. Default: pdf png.")
    args = parser.parse_args()

    outpaths = plot_bias_correction_schematic(
        args.n_parent, args.seed, args.savedir, args.formats)
    for out in outpaths:
        print(f"saved {out}")


if __name__ == "__main__":
    main()
