#!/usr/bin/env python3
"""Precompute A_H extinction grids and save to the cache file.

Queries the dust map for MC sightlines and observed C22 stars on the
distance grid, then saves the result as an .npz file that the model
loads at init time. Run this once (or when config changes) to avoid
querying the dust map during model setup.

Usage:
    python scripts/preprocess/MWCepheids/precompute_mwcepheids_extinction.py
    python scripts/preprocess/MWCepheids/precompute_mwcepheids_extinction.py \
        --config scripts/runs/configs/config_MWCepheids.toml
"""
import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from candel.model.mwcepheids import sample_disk_sightlines
from candel.pvdata import CepheidData, query_AH_grid, to_mwcepheids_config
from candel.util import load_config


def plot_extinction(d_grid, AH_obs, AH_obs_valid, AH_mc,
                    AH_mc_valid, AH_max, fname):
    """Plot LOS extinction profiles for observed and MC sightlines."""
    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4))

    # --- Observed stars ---
    ax = axes[0]
    for i in range(AH_obs.shape[0]):
        m = AH_obs_valid[i]
        if not m.any():
            continue
        ax.plot(d_grid[m], AH_obs[i][m],
                lw=0.5, alpha=0.6)
    ax.axhline(AH_max, ls="--", color="k", lw=1,
               label=rf"$A_H = {AH_max}$")
    ax.set_xlabel("Distance [kpc]")
    ax.set_ylabel(r"$A_H$ [mag]")
    ax.set_title(
        f"Observed C22 ({AH_obs.shape[0]} stars)")
    ax.legend()

    # --- MC sightlines (subsample for readability) ---
    ax = axes[1]
    n_show = min(100, AH_mc.shape[0])
    for i in range(n_show):
        m = AH_mc_valid[i]
        if not m.any():
            continue
        ax.plot(d_grid[m], AH_mc[i][m],
                lw=0.3, alpha=0.4)
    ax.axhline(AH_max, ls="--", color="k", lw=1,
               label=rf"$A_H = {AH_max}$")
    ax.set_xlabel("Distance [kpc]")
    ax.set_ylabel(r"$A_H$ [mag]")
    ax.set_title(
        f"MC sightlines ({n_show}/{AH_mc.shape[0]})")
    ax.legend()

    # --- Fraction of MC sightlines exceeding AH_max ---
    ax = axes[2]
    # At each distance, fraction with A_H > AH_max
    # (only count valid sightlines)
    n_valid = AH_mc_valid.sum(axis=0)
    n_exceed = ((AH_mc > AH_max) & AH_mc_valid).sum(
        axis=0)
    # Avoid division by zero
    safe = n_valid > 0
    frac = np.zeros(len(d_grid))
    frac[safe] = n_exceed[safe] / n_valid[safe]

    ax.plot(d_grid, frac, color="C3", lw=1.5)
    ax.set_xlabel("Distance [kpc]")
    ax.set_ylabel(
        rf"Fraction with $A_H > {AH_max}$")
    ax.set_title("MC excluded fraction")
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {fname}")


def plot_sightlines(ell_dw, b_dw, ell_min, ell_max, b_min, b_max,
                    obs_data, fname):
    """Compare disk-weighted vs uniform sightline distributions.

    ``obs_data`` is a dict ``{campaign: (ell, b)}`` overlaid on histograms.
    """
    n = len(ell_dw)
    rng_comp = np.random.default_rng(12345)

    # Uniform sample: flat in ell, uniform in sin(b)
    ell_uni = rng_comp.uniform(ell_min, ell_max, n)
    sin_b_min = np.sin(np.radians(b_min))
    sin_b_max = np.sin(np.radians(b_max))
    b_uni = np.degrees(
        np.arcsin(rng_comp.uniform(sin_b_min, sin_b_max, n)))

    with plt.style.context("science"):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Top left: uniform scatter
        ax = axes[0, 0]
        ax.scatter(ell_uni, b_uni, s=1, alpha=0.3)
        ax.set_xlabel(r"$\ell$ [deg]")
        ax.set_ylabel(r"$b$ [deg]")
        ax.set_title(f"Uniform ($n = {n}$)")
        ax.set_xlim(ell_min, ell_max)
        ax.set_ylim(b_min, b_max)

        # Top right: disk-weighted scatter
        ax = axes[0, 1]
        ax.scatter(ell_dw, b_dw, s=1, alpha=0.3)
        ax.set_xlabel(r"$\ell$ [deg]")
        ax.set_ylabel(r"$b$ [deg]")
        ax.set_title(f"Disk-weighted ($n = {n}$)")
        ax.set_xlim(ell_min, ell_max)
        ax.set_ylim(b_min, b_max)

        # Bottom left: latitude histograms
        ax = axes[1, 0]
        bins_b = np.linspace(b_min, b_max, 61)
        ax.hist(b_uni, bins=bins_b, density=True, alpha=0.5,
                label="Uniform")
        ax.hist(b_dw, bins=bins_b, density=True, alpha=0.5,
                label="Disk-weighted")
        for camp, (ell_c, b_c) in obs_data.items():
            ax.hist(b_c, bins=bins_b, density=True, histtype="step",
                    lw=1.5, label=camp)
        ax.set_xlabel(r"$b$ [deg]")
        ax.set_ylabel("Density")
        ax.legend()

        # Bottom right: longitude histograms
        ax = axes[1, 1]
        bins_ell = np.linspace(ell_min, ell_max, 61)
        ax.hist(ell_uni, bins=bins_ell, density=True, alpha=0.5,
                label="Uniform")
        ax.hist(ell_dw, bins=bins_ell, density=True, alpha=0.5,
                label="Disk-weighted")
        for camp, (ell_c, b_c) in obs_data.items():
            ax.hist(ell_c, bins=bins_ell, density=True, histtype="step",
                    lw=1.5, label=camp)
        ax.set_xlabel(r"$\ell$ [deg]")
        ax.set_ylabel("Density")
        ax.legend()

        fig.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Plot saved to {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute A_H extinction grids for the forward model.")
    repo_root = Path(__file__).resolve().parents[3]
    default_config = repo_root / "scripts" / "runs" / "configs" / (
        "config_MWCepheids.toml")
    parser.add_argument("--config", default=default_config,
                        type=Path, help="Path to CANDEL run config TOML file.")
    args = parser.parse_args()

    config = to_mwcepheids_config(
        load_config(args.config, replace_los_prior=False))
    model_cfg = config["model"]

    # --- Read config values (must match model.py exactly) ---
    c22_cfg = model_cfg.get("C22", {})
    sel_cfg = c22_cfg.get("selection", {})
    dust_map = sel_cfg.get("dust_map", "bayestar")
    n_mc = model_cfg.get("n_mc_dustmap",
                         model_cfg.get("n_mc_selection", 50))
    mc_seed = model_cfg.get("mc_seed", 42)

    # Distance grid
    d_min = c22_cfg.get("d_min", 0.1)
    d_max = c22_cfg.get("d_max", 8.0)
    d_spacing = c22_cfg.get("d_spacing", 0.01)
    n_grid = int((d_max - d_min) / d_spacing) + 1
    if n_grid % 2 == 0:
        n_grid += 1
    d_grid = np.asarray(jnp.linspace(d_min, d_max, n_grid))

    # MC sightlines: disk-weighted sampling + dust-map rejection.
    # Oversample to survive rejection of sightlines without coverage.
    rng = np.random.default_rng(mc_seed)
    mc_cfg = model_cfg.get("selection_mc", {})
    ell_min = mc_cfg.get("ell_min", 0.0)
    ell_max = mc_cfg.get("ell_max", 360.0)
    b_min = mc_cfg.get("b_min", -90.0)
    b_max = mc_cfg.get("b_max", 90.0)

    disk_cfg = model_cfg.get("disk_prior", {})
    R_d = disk_cfg.get("R_d", 2.5)
    z_d = disk_cfg.get("z_d", 0.1)
    R_sun = disk_cfg.get("R_sun", 8.122)

    n_draw = max(n_mc * 3, n_mc + 500)
    ell_draw, b_draw = sample_disk_sightlines(
        n_draw, ell_min, ell_max, b_min, b_max,
        d_min, d_max, R_d, z_d, R_sun, rng)

    # Load observed C22 star coordinates (all stars, no exclusions)
    config_c22 = dict(config)
    config_c22["data"] = dict(config["data"], which_subset="C22",
                              exclude_stars=[])
    data = CepheidData(config_c22)
    ell_obs = np.asarray(data.ell)
    b_obs = np.asarray(data.b)

    # Cache path (same logic as _extinction_cache_path)
    local_paths = config.get("local", {}).get("paths", {})
    data_dir = local_paths.get("data", ".")
    fname = (f"AH_cache_{dust_map}"
             f"_seed{mc_seed}.npz")
    cache_path = os.path.join(data_dir, fname)

    print(f"Dust map       : {dust_map}")
    print(f"MC draw        : {n_draw} sightlines (seed={mc_seed})")
    print(f"Obs stars (C22): {len(ell_obs)}")
    print(f"Distance grid  : [{d_min}, {d_max}] kpc, n={n_grid}")
    print(f"Cache file     : {cache_path}")

    # Query dust map for all drawn sightlines
    print(f"Querying dust map '{dust_map}' for MC sightlines ...")
    AH_draw, AH_draw_valid, AH_draw_std = query_AH_grid(
        ell_draw, b_draw, d_grid, map_name=dust_map, return_std=True)

    # Reject sightlines with no dust map coverage
    has_coverage = AH_draw_valid.any(axis=1)
    n_valid = has_coverage.sum()
    n_rejected = n_draw - n_valid
    print(f"MC rejection   : {n_rejected}/{n_draw} sightlines "
          f"rejected (no dust coverage)")
    if n_valid < n_mc:
        raise RuntimeError(
            f"Only {n_valid}/{n_draw} MC sightlines have dust map "
            f"coverage (need {n_mc}). Increase oversample or "
            f"widen sky region bounds.")

    # Keep all valid sightlines (model slices to n_mc_selection)
    idx_valid = np.where(has_coverage)[0]
    ell_mc = ell_draw[idx_valid]
    b_mc = b_draw[idx_valid]
    AH_mc = AH_draw[idx_valid]
    AH_mc_valid = AH_draw_valid[idx_valid]
    AH_mc_std = AH_draw_std[idx_valid]
    print(f"MC valid       : {len(ell_mc)} sightlines saved")

    print(f"Querying dust map '{dust_map}' for observed stars ...")
    AH_obs, AH_obs_valid, AH_obs_std = query_AH_grid(
        ell_obs, b_obs, d_grid, map_name=dust_map, return_std=True)

    # Save
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        d_grid=d_grid,
        AH_mc_grid=AH_mc, AH_mc_valid=AH_mc_valid, AH_mc_std=AH_mc_std,
        AH_obs_profiles=AH_obs, AH_obs_valid=AH_obs_valid,
        AH_obs_std=AH_obs_std,
        ell_mc=ell_mc, b_mc=b_mc,
        ell_obs=ell_obs, b_obs=b_obs,
    )

    # Print coverage stats
    mc_cov = np.mean(AH_mc_valid)
    obs_cov = np.mean(AH_obs_valid)
    obs_star_valid = np.sum(AH_obs_valid, axis=1) > 0.5 * n_grid
    print(f"MC grid coverage : {mc_cov:.1%}")
    print(f"Obs grid coverage: {obs_cov:.1%}")
    print(f"Valid obs stars  : {int(np.sum(obs_star_valid))}/{len(ell_obs)}")

    # Report A_H std statistics (dust map posterior uncertainty)
    mc_std_valid = AH_mc_std[AH_mc_valid]
    obs_std_valid = AH_obs_std[AH_obs_valid]
    if len(mc_std_valid) > 0:
        print(f"MC A_H std       : "
              f"median={np.median(mc_std_valid):.4f}, "
              f"mean={np.mean(mc_std_valid):.4f}, "
              f"max={np.max(mc_std_valid):.4f} mag")
    if len(obs_std_valid) > 0:
        print(f"Obs A_H std      : "
              f"median={np.median(obs_std_valid):.4f}, "
              f"mean={np.mean(obs_std_valid):.4f}, "
              f"max={np.max(obs_std_valid):.4f} mag")

    print(f"Saved to {cache_path}")

    # Plot
    obs_data = {"C22": (ell_obs, b_obs)}
    sightline_fname = os.path.join(data_dir, "sightline_distribution.png")
    plot_sightlines(ell_mc, b_mc, ell_min, ell_max, b_min, b_max,
                    obs_data, sightline_fname)

    AH_max = sel_cfg.get("AH_max", 0.4)
    plot_fname = os.path.join(
        data_dir, "AH_extinction_profiles.png")
    plot_extinction(
        d_grid, AH_obs, AH_obs_valid,
        AH_mc, AH_mc_valid, AH_max, plot_fname)


if __name__ == "__main__":
    main()
