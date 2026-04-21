# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Toy joint H0 inference from per-galaxy NSS distance posteriors.

Loads D_c posterior samples from single-galaxy NSS runs, fits a Gaussian
KDE to each, and runs a joint numpyro model that shares H0, sigma_pec,
and a phenomenological selection function across all galaxies.

The per-galaxy distance constraint is approximated by the KDE of the NSS
D_c marginal posterior, with the volumetric D^2 prior divided out.

Always runs both with and without selection, and produces a single GetDist
comparison corner plot overlaying the two posteriors.

Usage:
    python scripts/megamaser/toy_joint_H0.py [--num-warmup 1000] [--num-samples 4000]
"""
import argparse
import logging
import os

from h5py import File as H5File
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from getdist import MCSamples, plots
from jax import random
from numpyro import factor
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
import seaborn as sns
from scipy.stats import gaussian_kde, norm

from candel.cosmo.cosmography import Distance2Redshift
from candel.model.integration import ln_simpson
from candel.model.utils import Maxwell
from candel.util import (SPEED_OF_LIGHT, fsection, load_config, read_samples)

# -----------------------------------------------------------------------
# Hard-coded paths and galaxy data
# -----------------------------------------------------------------------

RESULT_ROOT = "results/Megamaser"
CONFIG_PATH = "scripts/megamaser/config_maser.toml"

# Hard-coded NSS result files per galaxy
NSS_FILES = {
    "CGCG074-064": f"{RESULT_ROOT}/CGCG074-064_nss_Dflat_noclump.hdf5",
    "NGC5765b": f"{RESULT_ROOT}/NGC5765b_nss_Dflat_noclump.hdf5",
    "NGC6264": f"{RESULT_ROOT}/NGC6264_nss_Dflat_noclump.hdf5",
    "NGC6323": f"{RESULT_ROOT}/NGC6323_nss_Dflat_noclump.hdf5",
    "UGC3789": f"{RESULT_ROOT}/UGC3789_nss_Dflat_noclump.hdf5",
}

# Load v_sys_obs, D_lo, D_hi from config
_cfg = load_config(CONFIG_PATH)
_gal_cfg = _cfg["model"]["galaxies"]

GALAXIES = {}
for name, nss_file in NSS_FILES.items():
    gc = _gal_cfg[name]
    GALAXIES[name] = {
        "nss_file": nss_file,
        "v_sys_obs": gc["v_sys_obs"],
        "D_lo": gc["D_lo"],
        "D_hi": gc["D_hi"],
    }

# LaTeX labels for the corner plot
PARAM_LABELS = {
    "H0": r"H_0 \; [\mathrm{km\,s^{-1}\,Mpc^{-1}}]",
    "sigma_pec": r"\sigma_\mathrm{pec} \; [\mathrm{km\,s^{-1}}]",
    "D_lim": r"D_\mathrm{lim} \; [\mathrm{Mpc}]",
    "D_width": r"D_\mathrm{width} \; [\mathrm{Mpc}]",
    "CGCG074-064_D_c": r"D_\mathrm{CGCG074} \; [\mathrm{Mpc}]",
    "NGC5765b_D_c": r"D_\mathrm{N5765b} \; [\mathrm{Mpc}]",
    "NGC6264_D_c": r"D_\mathrm{N6264} \; [\mathrm{Mpc}]",
    "NGC6323_D_c": r"D_\mathrm{N6323} \; [\mathrm{Mpc}]",
    "UGC3789_D_c": r"D_\mathrm{U3789} \; [\mathrm{Mpc}]",
}


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def build_log_distance_likelihood(kde, D_lo, D_hi, n_grid=1024):
    """Build a tabulated log-likelihood of D_c from the NSS posterior.

    The NSS posterior used a flat prior on D_c, so the posterior IS the
    likelihood: log L(D) = log KDE(D) + const.

    Returns (D_grid, log_L_grid) as jnp arrays.
    """
    D_grid = np.linspace(D_lo, D_hi, n_grid)
    log_L = kde.logpdf(D_grid)
    log_L -= log_L.max()
    return jnp.asarray(D_grid), jnp.asarray(log_L)


# -----------------------------------------------------------------------
# numpyro model
# -----------------------------------------------------------------------

# Selection integral grid (fixed, independent of sampled parameters)
_D_SEL = jnp.linspace(1.0, 400.0, 1001)
_LOG_D2_SEL = 2.0 * jnp.log(_D_SEL)


def model(galaxy_data, distance2redshift, use_selection, priors,
          flat_dist=False):
    H0 = numpyro.sample("H0", priors["H0"])
    sigma_pec = numpyro.sample("sigma_pec", priors["sigma_pec"])
    h = H0 / 100.0

    if use_selection:
        D_lim = numpyro.sample("D_lim", priors["D_lim"])
        D_width = numpyro.sample("D_width", priors["D_width"])

        # Selection normalisation
        log_sel_grid = jax.scipy.stats.norm.logcdf(
            (D_lim - _D_SEL) / D_width)
        log_vol = _LOG_D2_SEL if not flat_dist else 0.0
        log_Z_sel = ln_simpson(log_sel_grid + log_vol, _D_SEL)

    for gd in galaxy_data:
        name = gd["name"]
        v_sys = gd["v_sys_obs"]
        D_lo, D_hi = gd["D_lo"], gd["D_hi"]

        D_c = numpyro.sample(f"{name}_D_c", dist.Uniform(D_lo, D_hi))

        # Volumetric D^2 prior (skip for flat distance prior)
        if not flat_dist:
            factor(f"{name}_Dvol", 2.0 * jnp.log(D_c))

        # Distance likelihood via interpolation on precomputed grid
        log_L_D = jnp.interp(D_c, gd["D_grid"], gd["log_L_grid"],
                             left=-1e10, right=-1e10)
        factor(f"{name}_ll_D", log_L_D)

        # Redshift likelihood (full log-pdf so sigma_pec is constrained)
        z_cosmo = distance2redshift(jnp.atleast_1d(D_c), h=h).squeeze()
        cz = SPEED_OF_LIGHT * z_cosmo
        ll_cz = dist.Normal(cz, sigma_pec).log_prob(v_sys)
        factor(f"{name}_ll_cz", ll_cz)

        # Selection
        if use_selection:
            log_sel = jax.scipy.stats.norm.logcdf(
                (D_lim - D_c) / D_width)
            factor(f"{name}_sel", log_sel - log_Z_sel)


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def _build_mcsample(samples, keys, label):
    """Build a GetDist MCSamples from a dict of numpy arrays."""
    names, param_labels, columns = [], [], []
    for k in keys:
        if k not in samples:
            continue
        v = samples[k]
        if v.max() - v.min() == 0:
            continue
        names.append(k)
        param_labels.append(PARAM_LABELS.get(k, k))
        columns.append(v)

    data = np.column_stack(columns)
    return MCSamples(samples=data, names=names, labels=param_labels,
                     label=label,
                     settings={"smooth_scale_2D": 0.35,
                               "smooth_scale_1D": 0.3})


def make_corner(samples_nosel, samples_sel, fpath):
    """Overlay both posteriors on a single GetDist triangle plot."""
    # Suppress getdist bandwidth/binning warnings — the fallback bandwidth
    # and default fine_bins are adequate, just noisy for correlated params.
    logging.getLogger("root").setLevel(logging.ERROR)

    try:
        import scienceplots  # noqa
        style = "science"
    except ImportError:
        style = "default"

    # Shared keys + selection params (nosel samples just won't have them)
    keys = (["H0", "sigma_pec"]
            + [f"{n}_D_c" for n in GALAXIES]
            + ["D_lim", "D_width"])

    gd_nosel = _build_mcsample(samples_nosel, keys, "No selection")
    gd_sel = _build_mcsample(samples_sel, keys, "With selection")

    # Legend labels with H0 summary
    H0_nosel = samples_nosel["H0"]
    H0_sel = samples_sel["H0"]
    label_nosel = (r"No selection: $H_0 = "
                   f"{H0_nosel.mean():.1f}"
                   r"_{-" f"{H0_nosel.mean() - np.percentile(H0_nosel, 16):.1f}"
                   r"}^{+" f"{np.percentile(H0_nosel, 84) - H0_nosel.mean():.1f}"
                   r"}$")
    label_sel = (r"With selection: $H_0 = "
                 f"{H0_sel.mean():.1f}"
                 r"_{-" f"{H0_sel.mean() - np.percentile(H0_sel, 16):.1f}"
                 r"}^{+" f"{np.percentile(H0_sel, 84) - H0_sel.mean():.1f}"
                 r"}$")

    fontsize = 22
    settings = plots.GetDistPlotSettings()
    settings.lab_fontsize = fontsize
    settings.axes_fontsize = fontsize - 1
    settings.legend_fontsize = fontsize
    settings.title_limit_fontsize = fontsize - 1
    with plt.style.context(style):
        g = plots.get_subplot_plotter(settings=settings)
        g.triangle_plot(
            [gd_nosel, gd_sel],
            params=keys,
            filled=True,
            legend_labels=[label_nosel, label_sel],
            legend_loc="upper right",
        )
        g.export(fpath, dpi=450)
    plt.close()
    print(f"Saved corner plot to {os.path.abspath(fpath)}")


def make_H0_1d(samples_nosel, samples_sel, fpath):
    """1D H0 posterior comparison with Planck and SH0ES."""
    try:
        import scienceplots  # noqa
        style = "science"
    except ImportError:
        style = "default"

    H0_nosel = samples_nosel["H0"]
    H0_sel = samples_sel["H0"]

    cols = ["#87193d", "#1e42b9", "#d42a29", "#05dd6b", "#ee35d5"]
    bw = 2.0

    # MNRAS one-column width: 84 mm ≈ 3.307 in
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(3.307, 2.5))

        sns.kdeplot(H0_nosel, ax=ax, fill=True, label="No selection",
                    color=cols[0], bw_adjust=bw)
        sns.kdeplot(H0_sel, ax=ax, fill=True, label="With selection",
                    color=cols[3], bw_adjust=bw)

        # SH0ES and Planck ±1σ bands
        ax.axvspan(73.04 - 1.04, 73.04 + 1.04,
                   alpha=0.25, color=cols[2], label="SH0ES", zorder=-1)
        ax.axvspan(67.4 - 0.5, 67.4 + 0.5,
                   alpha=0.25, color=cols[1], label="Planck", zorder=-1)

        ax.set_xlabel(
            r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel("Normalized PDF")
        ax.set_xlim(57, 83)

        # Split legend above figure: posteriors left, references right
        handles, labels_all = ax.get_legend_handles_labels()
        main_h, main_l, ref_h, ref_l = [], [], [], []
        for h, l in zip(handles, labels_all):
            if l in ["SH0ES", "Planck"]:
                ref_h.append(h)
                ref_l.append(l)
            else:
                main_h.append(h)
                main_l.append(l)

        fig.tight_layout()
        leg1 = ax.legend(main_h, main_l, loc="lower left",
                         bbox_to_anchor=(0.0, 1.01), ncol=1, frameon=False)
        ax.add_artist(leg1)
        ax.legend(ref_h, ref_l, loc="lower right",
                  bbox_to_anchor=(1.0, 1.01), ncol=1, frameon=False)

        fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved H0 1D plot to {os.path.abspath(fpath)}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run_nuts(galaxy_data, d2z, use_selection, priors,
             num_warmup, num_samples, num_chains, seed,
             flat_dist=False):
    """Run NUTS for one configuration, return samples dict."""
    dist_tag = "flat" if flat_dist else "volume"
    sel_tag = "with selection" if use_selection else "no selection"
    fsection(f"NUTS ({sel_tag}, {dist_tag} D prior)")

    kernel = NUTS(model, dense_mass=True,
                  init_strategy=init_to_median(num_samples=100))
    mcmc = MCMC(kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                chain_method="vectorized")
    mcmc.run(random.PRNGKey(seed), galaxy_data, d2z, use_selection, priors,
             flat_dist)
    mcmc.print_summary()

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

    H0 = samples["H0"]
    sp = samples["sigma_pec"]
    print(f"\nH0 = {H0.mean():.1f} +/- {H0.std():.1f} km/s/Mpc")
    print(f"  16/84 = [{np.percentile(H0, 16):.1f}, "
          f"{np.percentile(H0, 84):.1f}]")
    print(f"sigma_pec = {sp.mean():.0f} +/- {sp.std():.0f} km/s")

    if use_selection:
        print(f"D_lim = {samples['D_lim'].mean():.1f} "
              f"+/- {samples['D_lim'].std():.1f}")
        print(f"D_width = {samples['D_width'].mean():.1f} "
              f"+/- {samples['D_width'].std():.1f}")

    for name in GALAXIES:
        D = samples[f"{name}_D_c"]
        print(f"  {name:15s}: D_c = {D.mean():.1f} +/- {D.std():.1f}")

    # Save to HDF5
    sel_suffix = "sel" if use_selection else "nosel"
    dist_suffix = "Dflat" if flat_dist else "Dvol"
    fpath = f"{RESULT_ROOT}/toy_joint_H0_{sel_suffix}_{dist_suffix}.hdf5"
    with H5File(fpath, "w") as f:
        grp = f.create_group("samples")
        for k, v in samples.items():
            grp.create_dataset(k, data=v)
    print(f"Saved samples to {os.path.abspath(fpath)}")

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=4000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--n-grid", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flat-dist", action="store_true",
                        help="Use flat D prior instead of volumetric D^2")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}", flush=True)

    # ---- Priors (all in one place) ----
    sigma_pec_mean = 250.0  # km/s
    maxwell_scale = sigma_pec_mean / (2.0 * np.sqrt(2.0 / np.pi))

    priors = {
        "H0": dist.Uniform(10.0, 200.0),
        "sigma_pec": Maxwell(maxwell_scale),
        "D_lim": dist.Uniform(15.0, 1000.0),
        "D_width": dist.Uniform(15.0, 500.0),
    }

    # ---- Load KDEs and build grids ----
    galaxy_data = []
    for name, info in GALAXIES.items():
        print(f"Loading {name}...", flush=True)
        D_c = read_samples("", info["nss_file"], "D_c")
        kde = gaussian_kde(D_c)
        D_grid, log_L_grid = build_log_distance_likelihood(
            kde, info["D_lo"], info["D_hi"], n_grid=args.n_grid)
        galaxy_data.append({
            "name": name,
            "v_sys_obs": info["v_sys_obs"],
            "D_lo": info["D_lo"],
            "D_hi": info["D_hi"],
            "D_grid": D_grid,
            "log_L_grid": log_L_grid,
        })

    d2z = Distance2Redshift(Om0=0.315)

    if args.flat_dist:
        print("Using FLAT distance prior (p(D) = const)")
    else:
        print("Using VOLUMETRIC distance prior (p(D) ∝ D²)")

    # ---- Run both configurations ----
    samples_nosel = run_nuts(galaxy_data, d2z, use_selection=False,
                             priors=priors,
                             num_warmup=args.num_warmup,
                             num_samples=args.num_samples,
                             num_chains=args.num_chains,
                             seed=args.seed,
                             flat_dist=args.flat_dist)
    samples_sel = run_nuts(galaxy_data, d2z, use_selection=True,
                           priors=priors,
                           num_warmup=args.num_warmup,
                           num_samples=args.num_samples,
                           num_chains=args.num_chains,
                           seed=args.seed + 1,
                           flat_dist=args.flat_dist)

    # ---- Plots ----
    dist_suffix = "Dflat" if args.flat_dist else "Dvol"
    make_corner(samples_nosel, samples_sel,
                f"{RESULT_ROOT}/toy_joint_H0_corner_{dist_suffix}.png")
    make_H0_1d(samples_nosel, samples_sel,
               f"{RESULT_ROOT}/toy_joint_H0_1d_{dist_suffix}.png")


if __name__ == "__main__":
    main()
