# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Posterior-predictive check of the CF4 TFR W1 redshift distribution.

Drives ``candel.mock.gen_TFR_mock`` at the posterior means of the W1 linear-bias
chain (2M++ density + r^2 exp[-(r/R)^q] empirical distance prior + Vext +
sigma_v) and compares the mock zcmb distribution to the observed sample.

Examples:
    python ppc_zcmb.py
    python ppc_zcmb.py --n-mock-factor 20 --n-reals 50
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from os.path import dirname, abspath, join
from sys import path as sys_path


def _heavy_imports():
    """Defer slow imports so ``--help`` is instant."""
    global np, h5py, plt, candel
    global gen_TFR_mock, name2field_loader, load_CF4_data
    global Distance2Distmod, Distance2Redshift

    import matplotlib
    matplotlib.use("Agg")

    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    sys_path.insert(0, "/Users/rstiskalek/Projects/candel")
    import candel
    from candel.mock import gen_TFR_mock
    from candel.field import name2field_loader
    from candel.pvdata import load_CF4_data
    from candel.cosmo.cosmography import (
        Distance2Distmod, Distance2Redshift)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SCRIPT_DIR = dirname(abspath(__file__))
PLOTS_DIR = "/Users/rstiskalek/Projects/CANDEL/plots/S8"

CHAIN = ("/Users/rstiskalek/Projects/CANDEL/results/S8/"
         "precomputed_los_Carrick2015_CF4_W1_linear.hdf5")
DENSITY_PATH = ("/Users/rstiskalek/Projects/CANDEL/data/fields/"
                "carrick2015_twompp_density.npy")
VELOCITY_PATH = ("/Users/rstiskalek/Projects/CANDEL/data/fields/"
                 "carrick2015_twompp_velocity.npy")
CF4_ROOT = "/Users/rstiskalek/Projects/CANDEL/data/CF4"

# CF4 W1 selection (matches the inference config).
B_MIN = 7.5
ZCMB_MAX = 0.05
ETA_MIN = -0.3

# Cosmography. Carrick2015 assumes Om=0.3.
OM = 0.3
R_GRID_MIN = 0.001
R_GRID_MAX = 251.0
R_GRID_N = 251

PARAM_KEYS = [
    "a_TFR", "b_TFR", "c_TFR", "sigma_int", "sigma_v",
    "beta", "b1",
    "eta_prior_mean", "eta_prior_std",
    "R_dist_emp", "q_dist_emp",
    "Vext_mag", "Vext_ell", "Vext_b",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def posterior_means(path, keys):
    with h5py.File(path, "r") as f:
        return {k: float(np.mean(f["samples"][k][...])) for k in keys}


def load_observed_zcmb():
    data = load_CF4_data(
        CF4_ROOT, which_band="w1", best_mag_quality=True,
        eta_min=ETA_MIN, zcmb_min=None, zcmb_max=ZCMB_MAX, b_min=B_MIN,
        remove_outliers=True, calibration=None, dust_model=None,
        return_all=False)
    return np.asarray(data["zcmb"])


def gen_one_mock(nsamples, pm, r_grid, field_loader, r2distmod, r2z, seed):
    return gen_TFR_mock(
        nsamples=nsamples, r_grid=r_grid,
        Vext_mag=pm["Vext_mag"], Vext_ell=pm["Vext_ell"],
        Vext_b=pm["Vext_b"],
        sigma_v=pm["sigma_v"], beta=pm["beta"],
        a_TFR=pm["a_TFR"], b_TFR=pm["b_TFR"], c_TFR=pm["c_TFR"],
        sigma_int=pm["sigma_int"],
        zeropoint_dipole_mag=None, zeropoint_dipole_ell=None,
        zeropoint_dipole_b=None,
        h=1.0, e_mag=0.05,
        eta_prior_mean=pm["eta_prior_mean"],
        eta_prior_std=pm["eta_prior_std"], e_eta=0.05,
        b_min=B_MIN, zcmb_max=ZCMB_MAX,
        R=pm["R_dist_emp"], p=2.0, n=pm["q_dist_emp"],
        field_loader=field_loader,
        r2distmod=r2distmod, r2z=r2z,
        Om=OM, seed=seed, verbose=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--n-mock-factor", type=int, default=10,
                        help="Per realisation, draw N_obs * factor samples "
                             "from the empirical prior, then keep N_obs after "
                             "the zcmb_max cut. Default: 10.")
    parser.add_argument("--n-reals", type=int, default=30,
                        help="Number of mock realisations for the band. "
                             "Default: 30.")
    parser.add_argument("--savedir", default=PLOTS_DIR)
    args = parser.parse_args()

    _heavy_imports()

    pm = posterior_means(CHAIN, PARAM_KEYS)
    print("Posterior means:")
    for k in PARAM_KEYS:
        print(f"  {k:<18s} = {pm[k]:.4f}")

    loader_cls = name2field_loader("Carrick2015")
    field_loader = loader_cls(path_density=DENSITY_PATH,
                              path_velocity=VELOCITY_PATH)
    r_grid = np.linspace(R_GRID_MIN, R_GRID_MAX, R_GRID_N)
    r2distmod = Distance2Distmod(Om0=OM)
    r2z = Distance2Redshift(Om0=OM)

    zobs = load_observed_zcmb()
    n_obs = len(zobs)
    print(f"observed CF4 W1 sample: {n_obs} galaxies")

    bins = np.histogram_bin_edges(zobs, bins="auto")
    counts = np.empty((args.n_reals, len(bins) - 1))
    for k in range(args.n_reals):
        data = gen_one_mock(
            nsamples=args.n_mock_factor * n_obs, pm=pm, r_grid=r_grid,
            field_loader=field_loader, r2distmod=r2distmod, r2z=r2z,
            seed=k)
        z = np.asarray(data["zcmb"])
        if len(z) >= n_obs:
            z = z[:n_obs]
        counts[k] = np.histogram(z, bins=bins, density=True)[0]
        print(f"  mock {k + 1}/{args.n_reals}: kept {len(z)} of "
              f"{args.n_mock_factor * n_obs}", flush=True)

    lo, hi = np.percentile(counts, [16, 84], axis=0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.fill_between(centers, lo, hi, step="mid", alpha=0.35,
                        color="#1e42b9", label="PPC")
        ax.hist(zobs, bins=bins, density=True, histtype="step",
                color="k", lw=1.4, label=r"CF4 TFR $W1$")
        ax.set_xlim(1e-3, ZCMB_MAX)
        ax.set_xlabel(r"$z_{\rm cmb}$")
        ax.set_ylabel(r"Normalised PDF")
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        out = join(args.savedir, "ppc_W1_zcmb.pdf")
        fig.savefig(out, dpi=450, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
