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

By default runs no selection, distance selection, and redshift selection,
and produces GetDist comparison plots overlaying the posteriors.

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
from scipy.stats import gaussian_kde

from candel.cosmo.cosmography import Distance2Redshift
from candel.model.integration import ln_simpson
from candel.model.utils import Maxwell
from candel.util import (SPEED_OF_LIGHT, fsection, load_config, read_samples,
                         results_path)

# -----------------------------------------------------------------------
# Hard-coded paths and galaxy data
# -----------------------------------------------------------------------

RESULT_ROOT = results_path("results/Megamaser")
NSS_ROOT = f"{RESULT_ROOT}/paper_MCP"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_maser.toml")

# Hard-coded NSS result files per galaxy
NSS_FILES = {
    "CGCG074-064": f"{NSS_ROOT}/CGCG074-064_nss_Dflat_mode2.hdf5",
    "NGC5765b": f"{NSS_ROOT}/NGC5765b_nss_Dflat_mode2.hdf5",
    "NGC6264": f"{NSS_ROOT}/NGC6264_nss_Dflat_mode2.hdf5",
    "NGC6323": f"{NSS_ROOT}/NGC6323_nss_Dflat_mode2.hdf5",
    "UGC3789": f"{NSS_ROOT}/UGC3789_nss_Dflat_mode2.hdf5",
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
    "cz_lim_selection": r"cz_\mathrm{lim} \; [\mathrm{km\,s^{-1}}]",
    "cz_lim_selection_width": (
        r"cz_\mathrm{width} \; [\mathrm{km\,s^{-1}}]"),
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


def model(galaxy_data, distance2redshift, use_selection, selection_mode,
          priors, flat_dist=False):
    H0 = numpyro.sample("H0", priors["H0"])
    sigma_pec = numpyro.sample("sigma_pec", priors["sigma_pec"])
    h = H0 / 100.0

    if use_selection and selection_mode == "distance":
        D_lim = numpyro.sample("D_lim", priors["D_lim"])
        D_width = numpyro.sample("D_width", priors["D_width"])

        # Selection normalisation
        log_sel_grid = jax.scipy.stats.norm.logcdf(
            (D_lim - _D_SEL) / D_width)
        log_vol = _LOG_D2_SEL if not flat_dist else 0.0
        log_Z_sel = ln_simpson(log_sel_grid + log_vol, _D_SEL)
    elif use_selection and selection_mode == "redshift":
        cz_lim = numpyro.sample(
            "cz_lim_selection", priors["cz_lim_selection"])
        cz_width = numpyro.sample(
            "cz_lim_selection_width",
            priors["cz_lim_selection_width"])

        z_sel = distance2redshift(_D_SEL, h=h)
        cz_sel = SPEED_OF_LIGHT * z_sel
        cz_width_eff = jnp.sqrt(sigma_pec**2 + cz_width**2)
        log_sel_grid = jax.scipy.stats.norm.logcdf(
            (cz_lim - cz_sel) / cz_width_eff)
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
        if use_selection and selection_mode == "distance":
            log_sel = jax.scipy.stats.norm.logcdf(
                (D_lim - D_c) / D_width)
            factor(f"{name}_sel", log_sel - log_Z_sel)
        elif use_selection and selection_mode == "redshift":
            log_sel = jax.scipy.stats.norm.logcdf(
                (cz_lim - v_sys) / cz_width)
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


def _plot_keys(selection_mode):
    """Return parameters to include in the corner plot."""
    keys = ["H0", "sigma_pec"] + [f"{n}_D_c" for n in GALAXIES]
    if selection_mode in ("all", "none"):
        return keys
    if selection_mode == "redshift":
        return keys + ["cz_lim_selection", "cz_lim_selection_width"]
    return keys + ["D_lim", "D_width"]


def _h0_label(samples, label):
    H0 = samples["H0"]
    return (fr"{label}: $H_0 = "
            f"{H0.mean():.1f}"
            r"_{-" f"{H0.mean() - np.percentile(H0, 16):.1f}"
            r"}^{+" f"{np.percentile(H0, 84) - H0.mean():.1f}"
            r"}$")


def make_corner(sample_sets, selection_mode, fpath):
    """Overlay selected posteriors on a single GetDist triangle plot."""
    # Suppress getdist bandwidth/binning warnings — the fallback bandwidth
    # and default fine_bins are adequate, just noisy for correlated params.
    logging.getLogger("root").setLevel(logging.ERROR)

    try:
        import scienceplots  # noqa
        style = "science"
    except ImportError:
        style = "default"

    keys = _plot_keys(selection_mode)
    gd_samples = [
        _build_mcsample(samples, keys, label)
        for label, samples in sample_sets
    ]
    legend_labels = [
        _h0_label(samples, label) for label, samples in sample_sets
    ]

    fontsize = 22
    settings = plots.GetDistPlotSettings()
    settings.lab_fontsize = fontsize
    settings.axes_fontsize = fontsize - 1
    settings.legend_fontsize = fontsize
    settings.title_limit_fontsize = fontsize - 1
    with plt.style.context(style):
        g = plots.get_subplot_plotter(settings=settings)
        g.triangle_plot(
            gd_samples,
            params=keys,
            filled=True,
            legend_labels=legend_labels,
            legend_loc="upper right",
        )
        g.export(fpath, dpi=450)
    plt.close()
    print(f"Saved corner plot to {os.path.abspath(fpath)}")


def make_H0_1d(sample_sets, fpath):
    """1D H0 posterior comparison with Planck and SH0ES."""
    try:
        import scienceplots  # noqa
        style = "science"
    except ImportError:
        style = "default"
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    cols = ["#87193d", "#1e42b9", "#d42a29", "#05dd6b", "#ee35d5"]
    bw = 2.0

    # MNRAS one-column width: 84 mm ≈ 3.307 in
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(3.307, 2.5))

        if sns is None:
            x = np.linspace(57, 83, 512)
            for (label, samples), color in zip(sample_sets, cols):
                H0 = samples["H0"]
                y = gaussian_kde(H0, bw_method=bw)(x)
                ax.plot(x, y, label=label, color=color)
                ax.fill_between(x, y, alpha=0.35, color=color)
        else:
            for (label, samples), color in zip(sample_sets, cols):
                sns.kdeplot(samples["H0"], ax=ax, fill=True, label=label,
                            color=color, bw_adjust=bw)

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

def run_nuts(galaxy_data, d2z, use_selection, selection_mode, priors,
             num_warmup, num_samples, num_chains, seed,
             flat_dist=False):
    """Run NUTS for one configuration, return samples dict."""
    dist_tag = "flat" if flat_dist else "volume"
    sel_tag = (f"with {selection_mode} selection"
               if use_selection else "no selection")
    fsection(f"NUTS ({sel_tag}, {dist_tag} D prior)")

    kernel = NUTS(model, dense_mass=True,
                  init_strategy=init_to_median(num_samples=100))
    mcmc = MCMC(kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                chain_method="vectorized")
    mcmc.run(random.PRNGKey(seed), galaxy_data, d2z, use_selection,
             selection_mode, priors, flat_dist)
    mcmc.print_summary()

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

    H0 = samples["H0"]
    sp = samples["sigma_pec"]
    print(f"\nH0 = {H0.mean():.1f} +/- {H0.std():.1f} km/s/Mpc")
    print(f"  16/84 = [{np.percentile(H0, 16):.1f}, "
          f"{np.percentile(H0, 84):.1f}]")
    print(f"sigma_pec = {sp.mean():.0f} +/- {sp.std():.0f} km/s")

    if use_selection and selection_mode == "distance":
        print(f"D_lim = {samples['D_lim'].mean():.1f} "
              f"+/- {samples['D_lim'].std():.1f}")
        print(f"D_width = {samples['D_width'].mean():.1f} "
              f"+/- {samples['D_width'].std():.1f}")
    elif use_selection and selection_mode == "redshift":
        cz_lim = samples["cz_lim_selection"]
        cz_width = samples["cz_lim_selection_width"]
        print(f"cz_lim_selection = {cz_lim.mean():.0f} "
              f"+/- {cz_lim.std():.0f} km/s")
        print(f"cz_lim_selection_width = {cz_width.mean():.0f} "
              f"+/- {cz_width.std():.0f} km/s")

    for name in GALAXIES:
        D = samples[f"{name}_D_c"]
        print(f"  {name:15s}: D_c = {D.mean():.1f} +/- {D.std():.1f}")

    # Save to HDF5
    if use_selection:
        sel_suffix = "sel" if selection_mode == "distance" else "zsel"
    else:
        sel_suffix = "nosel"
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
    parser.add_argument("--selection", choices=["none", "distance",
                                                "redshift"],
                        default=None,
                        help=("Run only one configuration. Omit to run all: "
                              "no selection, distance selection, and "
                              "redshift selection."))
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
        "cz_lim_selection": dist.Uniform(500.0, 20000.0),
        "cz_lim_selection_width": dist.Uniform(50.0, 10000.0),
    }

    # ---- Load KDEs and build grids ----
    galaxy_data = []
    for name, info in GALAXIES.items():
        print(f"Loading {name}...", flush=True)
        D_c = read_samples("", info["nss_file"], "D_c")
        D_q16, D_med, D_q84 = np.percentile(D_c, [16, 50, 84])
        print(f"  D_c = {D_med:.2f} "
              f"-{D_med - D_q16:.2f} +{D_q84 - D_med:.2f} Mpc",
              flush=True)
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

    # ---- Run requested configurations ----
    run_modes = (
        ["none", "distance", "redshift"]
        if args.selection is None else [args.selection])
    labels = {
        "none": "No selection",
        "distance": "Distance selection",
        "redshift": "Redshift selection",
    }
    sample_sets = []
    for i, run_mode in enumerate(run_modes):
        use_selection = run_mode != "none"
        selection_mode = "distance" if run_mode == "none" else run_mode
        sample_sets.append(
            (labels[run_mode],
             run_nuts(galaxy_data, d2z,
                      use_selection=use_selection,
                      selection_mode=selection_mode,
                      priors=priors,
                      num_warmup=args.num_warmup,
                      num_samples=args.num_samples,
                      num_chains=args.num_chains,
                      seed=args.seed + i,
                      flat_dist=args.flat_dist)))

    # ---- Plots ----
    dist_suffix = "Dflat" if args.flat_dist else "Dvol"
    selection_plot_mode = "all" if args.selection is None else args.selection
    sel_plot_suffix = {
        "all": "_all",
        "none": "_nosel",
        "distance": "",
        "redshift": "_zsel",
    }[selection_plot_mode]
    make_corner(sample_sets, selection_plot_mode,
                f"{RESULT_ROOT}/toy_joint_H0_corner"
                f"{sel_plot_suffix}_{dist_suffix}.png")
    make_H0_1d(sample_sets,
               f"{RESULT_ROOT}/toy_joint_H0_1d"
               f"{sel_plot_suffix}_{dist_suffix}.png")


if __name__ == "__main__":
    main()
