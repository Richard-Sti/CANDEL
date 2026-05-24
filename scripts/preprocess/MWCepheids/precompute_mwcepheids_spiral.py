#!/usr/bin/env python3
"""Precompute per-arm squared-distance profiles and save to cache.

Computes the squared distance from each arm's nearest trace point on distance
grids for MC sightlines and observed stars per campaign. The spiral factor
    (1 - f) + f * sum_arms exp(-dist_sq_k / (2 sigma^2))
is computed at runtime with sampled f and sigma.

If an extinction cache exists (and AH selection is active), MC sightline
coordinates are loaded from there (to match the rejection-sampled coords).
Otherwise, random sightlines are generated with the same seed and bounds
as the model would use.

Usage:
    python scripts/preprocess/MWCepheids/precompute_mwcepheids_spiral.py
    python scripts/preprocess/MWCepheids/precompute_mwcepheids_spiral.py \
        --config scripts/runs/configs/config_MWCepheids.toml
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from candel.model.mwcepheids import (
    compute_dist_sq_per_arm,
    get_drimmel_arm_traces,
    sample_disk_sightlines,
)
from candel.pvdata import CepheidData, to_mwcepheids_config
from candel.util import load_config


def plot_spiral_profiles(campaigns_data, mc_dist_sq_per_arm, mc_d_grid,
                         arm_xgc, arm_ygc, R_sun, R_d, z_d,
                         arm_frac, width, delta_pi, fname):
    """Plot spiral factor profiles with Cepheid distance markers.

    Left y-axis: spiral number density factor.
    Right y-axis: disk prior x spiral factor (normalized).
    """
    base, ext = os.path.splitext(fname)
    ncols = 4
    inv_2s2 = 1.0 / (2 * width**2)

    def _spiral_factor(dsq_per_arm, f=arm_frac):
        """dsq_per_arm: (n_arms, n_los, n_grid) or (n_arms, n_grid)."""
        gauss_sum = np.sum(np.exp(-dsq_per_arm * inv_2s2), axis=0)
        return (1 - f) + f * gauss_sum

    def _disk_prior(d_grid, ell_deg, b_deg):
        """Unnormalized disk prior: d^2 exp(-R_GC/R_d) exp(-|z|/z_d)."""
        ell_r = np.deg2rad(ell_deg)
        b_r = np.deg2rad(b_deg)
        x = d_grid * np.cos(b_r) * np.cos(ell_r)
        y = d_grid * np.cos(b_r) * np.sin(ell_r)
        z = d_grid * np.sin(b_r)
        R_GC = np.sqrt((R_sun - x)**2 + y**2)
        return d_grid**2 * np.exp(-R_GC / R_d - np.abs(z) / z_d)

    mc_profiles = _spiral_factor(mc_dist_sq_per_arm)

    with plt.style.context("science"):
        # --- Per-campaign figures: one panel per star ---
        for campaign, cdata in campaigns_data.items():
            d_grid = cdata["d_grid"]
            obs_dsq = cdata["obs_dist_sq_per_arm"]
            spiral = _spiral_factor(obs_dsq)
            pi_edr3 = cdata["pi_EDR3"]
            ell_obs = cdata["ell"]
            b_obs = cdata["b"]
            n_stars = spiral.shape[0]
            if n_stars == 0:
                continue

            pi_corr = np.asarray(pi_edr3) + delta_pi
            valid = pi_corr > 0.01
            d_approx = np.where(valid, 1.0 / pi_corr, np.nan)

            nrows = int(np.ceil(n_stars / ncols))
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows),
                squeeze=False)

            for j in range(n_stars):
                ax = axes[j // ncols, j % ncols]
                ax.plot(d_grid, spiral[j], lw=0.8, color="C0",
                        label="Spiral factor")

                # Right axis: disk prior x spiral (normalized)
                ax2 = ax.twinx()
                disk = _disk_prior(d_grid, ell_obs[j], b_obs[j])
                combined = disk * spiral[j]
                combined /= np.trapezoid(combined, d_grid)
                ax2.plot(d_grid, combined, lw=0.8, color="C1",
                         label="Disk × spiral")
                ax2.tick_params(labelsize=7, colors="C1")

                if valid[j] and d_grid[0] <= d_approx[j] <= d_grid[-1]:
                    ax.axvline(d_approx[j], ls=":", color="red", lw=1.2,
                               alpha=0.7)
                ax.set_title(f"Star {j}", fontsize=8)
                ax.tick_params(labelsize=7, colors="C0")

                if j == 0:
                    h1, l1 = ax.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right")

            for j in range(n_stars, nrows * ncols):
                axes[j // ncols, j % ncols].set_visible(False)

            for ax in axes[-1, :]:
                if ax.get_visible():
                    ax.set_xlabel(r"$d$ [kpc]", fontsize=8)
            for ax in axes[:, 0]:
                ax.set_ylabel("Spiral factor", fontsize=8, color="C0")

            fig.suptitle(f"{campaign} ({n_stars} stars)", fontsize=11)
            fig.tight_layout()
            camp_fname = f"{base}_{campaign}{ext}"
            fig.savefig(camp_fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved to {camp_fname}")

        # --- Summary figure: MC sightlines + Galactocentric map ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        ax = axes[0]
        n_show = min(200, mc_profiles.shape[0])
        for i in range(n_show):
            ax.plot(mc_d_grid, mc_profiles[i], lw=0.3, alpha=0.3)
        ax.set_xlabel(r"$d$ [kpc]")
        ax.set_ylabel("Spiral factor")
        ax.set_title(f"MC sightlines ({n_show}/{mc_profiles.shape[0]})")

        ax = axes[1]
        ax.plot(arm_xgc, arm_ygc, '.', ms=0.5, alpha=0.3, color='C7',
                label='Arm traces')
        ax.plot(-R_sun, 0, '*', ms=10, color='gold', zorder=5,
                label=r'Sun')

        for campaign, cdata in campaigns_data.items():
            pi_corr = np.asarray(cdata["pi_EDR3"]) + delta_pi
            valid = pi_corr > 0.01
            d_approx = np.where(valid, 1.0 / pi_corr, np.nan)

            ell_rad = np.deg2rad(np.asarray(cdata["ell"]))
            b_rad = np.deg2rad(np.asarray(cdata["b"]))
            x_gc = (d_approx * np.cos(b_rad) * np.cos(ell_rad) - R_sun)
            y_gc = d_approx * np.cos(b_rad) * np.sin(ell_rad)

            m = valid & np.isfinite(d_approx)
            ax.scatter(x_gc[m], y_gc[m], s=15, zorder=4,
                       label=f'{campaign} ({int(m.sum())} stars)')

        ax.set_xlabel(r"$x_\mathrm{GC}$ [kpc]")
        ax.set_ylabel(r"$y_\mathrm{GC}$ [kpc]")
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)

        fig.tight_layout()
        summary_fname = f"{base}_summary{ext}"
        fig.savefig(summary_fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {summary_fname}")


def plot_sightlines(ell_dw, b_dw, ell_min, ell_max, b_min, b_max,
                    obs_data, fname):
    """Compare disk-weighted vs uniform sightline distributions.

    ``obs_data`` is a dict ``{campaign: (ell, b)}`` overlaid on histograms.
    """
    n = len(ell_dw)
    rng_comp = np.random.default_rng(12345)
    ell_u = rng_comp.uniform(ell_min, ell_max, n)
    sin_b_min, sin_b_max = np.sin(np.deg2rad(b_min)), np.sin(np.deg2rad(b_max))
    b_u = np.rad2deg(np.arcsin(rng_comp.uniform(sin_b_min, sin_b_max, n)))

    with plt.style.context("science"):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))

        # Top left: uniform scatter
        axes[0, 0].scatter(ell_u, b_u, s=0.3, alpha=0.3)
        axes[0, 0].set_xlabel(r"$\ell$ [deg]")
        axes[0, 0].set_ylabel(r"$b$ [deg]")
        axes[0, 0].set_title(f"Uniform ($N = {n}$)")

        # Top right: disk-weighted scatter
        axes[0, 1].scatter(ell_dw, b_dw, s=0.3, alpha=0.3)
        axes[0, 1].set_xlabel(r"$\ell$ [deg]")
        axes[0, 1].set_ylabel(r"$b$ [deg]")
        axes[0, 1].set_title(f"Disk-weighted ($N = {n}$)")

        # Bottom left: latitude histograms
        b_bins = np.linspace(b_min, b_max, 61)
        axes[1, 0].hist(b_u, bins=b_bins, density=True, alpha=0.5,
                        label="Uniform")
        axes[1, 0].hist(b_dw, bins=b_bins, density=True, alpha=0.5,
                        label="Disk-weighted")
        for camp, (ell_c, b_c) in obs_data.items():
            axes[1, 0].hist(b_c, bins=b_bins, density=True, histtype="step",
                            lw=1.5, label=camp)
        axes[1, 0].set_xlabel(r"$b$ [deg]")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].legend()

        # Bottom right: longitude histograms
        ell_bins = np.linspace(ell_min, ell_max, 61)
        axes[1, 1].hist(ell_u, bins=ell_bins, density=True, alpha=0.5,
                        label="Uniform")
        axes[1, 1].hist(ell_dw, bins=ell_bins, density=True, alpha=0.5,
                        label="Disk-weighted")
        for camp, (ell_c, b_c) in obs_data.items():
            axes[1, 1].hist(ell_c, bins=ell_bins, density=True,
                            histtype="step", lw=1.5, label=camp)
        axes[1, 1].set_xlabel(r"$\ell$ [deg]")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].legend()

        fig.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-arm squared-distance profiles.")
    repo_root = Path(__file__).resolve().parents[3]
    default_config = repo_root / "scripts" / "runs" / "configs" / (
        "config_MWCepheids.toml")
    parser.add_argument("--config", default=default_config,
                        type=Path, help="Path to CANDEL run config TOML file.")
    args = parser.parse_args()

    config = to_mwcepheids_config(
        load_config(args.config, replace_los_prior=False))
    model_cfg = config["model"]

    # --- Read spiral arm config ---
    spiral_cfg = model_cfg.get("spiral_arms", {})
    ds = spiral_cfg.get("ds", None)
    use_extrapolated = spiral_cfg.get("use_extrapolated", True)

    # For plotting only: read a representative prior value.
    priors_cfg = model_cfg.get("priors", {})
    frac_prior = priors_cfg.get("spiral_arm_frac", {})
    width_prior = priors_cfg.get("spiral_width", {})
    plot_arm_frac = frac_prior.get(
        "value", frac_prior.get("loc", frac_prior.get("mean", 0.5)))
    plot_width = width_prior.get(
        "value", width_prior.get("loc", width_prior.get("mean", 0.3)))

    disk_cfg = model_cfg.get("disk_prior", {})
    R_d = disk_cfg.get("R_d", 2.5)
    z_d = disk_cfg.get("z_d", 0.1)
    R_sun = disk_cfg.get("R_sun", 8.122)

    mc_seed = model_cfg.get("mc_seed", 42)
    n_mc = model_cfg.get("n_mc_dustmap",
                         model_cfg.get("n_mc_selection", 50))
    delta_pi = -0.014

    # --- Load arm traces ---
    arms_xy = get_drimmel_arm_traces(
        R_sun=R_sun, use_extrapolated=use_extrapolated, ds=ds)
    arm_xgc = np.concatenate([x for x, y in arms_xy])
    arm_ygc = np.concatenate([y for x, y in arms_xy])

    # --- Load data and split by campaign (all stars, no exclusions) ---
    config_all = dict(config)
    config_all["data"] = dict(config["data"])
    config_all["data"].pop("which_subset", None)
    config_all["data"]["exclude_stars"] = []
    data_full = CepheidData(config_all)

    if data_full.campaign is not None:
        campaigns = {data_full.campaign: data_full}
    else:
        campaigns = data_full.split_by_campaign()

    # --- MC sightlines ---
    c22_cfg = model_cfg.get("C22", {})
    sel_cfg = c22_cfg.get("selection", {})
    AH_active = sel_cfg.get("apply_AH", False)
    dust_map = sel_cfg.get("dust_map", "bayestar")
    local_paths = config.get("local", {}).get("paths", {})
    data_dir = local_paths.get("data", ".")

    extinction_cache = os.path.join(
        data_dir, f"AH_cache_{dust_map}_seed{mc_seed}.npz")

    mc_cfg = model_cfg.get("selection_mc", {})
    sl_ell_min = mc_cfg.get("ell_min", 0.0)
    sl_ell_max = mc_cfg.get("ell_max", 360.0)
    sl_b_min = mc_cfg.get("b_min", -90.0)
    sl_b_max = mc_cfg.get("b_max", 90.0)

    # --- Distance grids and per-campaign MC sightlines ---
    d_grids = {}
    ell_mc_dict = {}
    b_mc_dict = {}
    mc_sources = {}

    # Per-campaign sky RNGs via SeedSequence (matches model.py)
    ss = np.random.SeedSequence(mc_seed)
    ss_sky = ss.spawn(2)[0]
    campaigns_sorted = sorted(campaigns.keys())
    sky_rngs = {c: np.random.default_rng(s)
                for c, s in zip(campaigns_sorted,
                                ss_sky.spawn(len(campaigns_sorted)))}

    for campaign in campaigns_sorted:
        camp_cfg = model_cfg.get(campaign, {})
        cd_min = camp_cfg.get("d_min", 0.1)
        cd_max = camp_cfg.get("d_max", 10.0)
        n_grid = max(501, int((cd_max - cd_min) / 0.01) + 1)
        if n_grid % 2 == 0:
            n_grid += 1
        d_grids[campaign] = np.linspace(cd_min, cd_max, n_grid)

        # C22 sightlines from extinction cache if available
        if (campaign == "C22" and AH_active
                and os.path.exists(extinction_cache)):
            print(f"Loading {campaign} MC coords from extinction cache")
            ext_cached = np.load(extinction_cache)
            n_cached = len(ext_cached["ell_mc"])
            if n_cached < n_mc:
                raise ValueError(
                    f"Extinction cache has {n_cached} MC sightlines "
                    f"but need {n_mc}. Re-run precompute_extinction.py.")
            ell_mc_dict[campaign] = ext_cached["ell_mc"][:n_mc]
            b_mc_dict[campaign] = ext_cached["b_mc"][:n_mc]
            mc_sources[campaign] = "extinction_cache"
        else:
            ell_mc, b_mc = sample_disk_sightlines(
                n_mc, sl_ell_min, sl_ell_max, sl_b_min, sl_b_max,
                cd_min, cd_max, R_d, z_d, R_sun, sky_rngs[campaign])
            ell_mc_dict[campaign] = ell_mc
            b_mc_dict[campaign] = b_mc
            mc_sources[campaign] = "disk_weighted"

    # Cache path
    cache_fname = f"spiral_cache_seed{mc_seed}.npz"
    cache_path = os.path.join(data_dir, cache_fname)

    print(f"R_sun            : {R_sun} kpc")
    print(f"R_d, z_d         : {R_d}, {z_d} kpc")
    print(f"ds               : {ds}")
    print(f"N arms           : {len(arms_xy)}")
    for c in campaigns_sorted:
        cd_min = d_grids[c][0]
        cd_max = d_grids[c][-1]
        print(f"MC {c:14s}: {n_mc} ({mc_sources[c]}), "
              f"d=[{cd_min:.1f}, {cd_max:.1f}] kpc")
    print(f"Campaigns        : {campaigns_sorted}")
    print(f"Cache file       : {cache_path}")

    # --- Compute per-arm dist_sq profiles ---
    save_dict = {
        "R_sun": R_sun,
        "R_d": R_d,
        "z_d": z_d,
    }

    mc_plot_grid = None
    mc_plot_dsq = None

    for campaign, d_grid in d_grids.items():
        ell_mc = ell_mc_dict[campaign]
        b_mc = b_mc_dict[campaign]

        print(f"\n--- {campaign} ---")
        print(f"  Distance grid: [{d_grid[0]:.2f}, {d_grid[-1]:.2f}] kpc, "
              f"n={len(d_grid)}")

        # Save per-campaign MC sightlines
        save_dict[f"{campaign}_ell_mc"] = ell_mc
        save_dict[f"{campaign}_b_mc"] = b_mc

        # MC dist_sq_per_arm
        print(f"  Computing MC dist_sq_per_arm ({n_mc} sightlines, "
              f"{len(arms_xy)} arms) ...")
        mc_dsq = compute_dist_sq_per_arm(
            ell_mc, b_mc, d_grid, arms_xy, R_sun)
        save_dict[f"{campaign}_mc_dist_sq_per_arm"] = mc_dsq
        save_dict[f"{campaign}_d_grid"] = d_grid
        print(f"  MC dist_sq_per_arm: shape={mc_dsq.shape}, "
              f"min={mc_dsq.min():.4f}, max={mc_dsq.max():.4f}")

        if mc_plot_grid is None or len(d_grid) > len(mc_plot_grid):
            mc_plot_grid = d_grid
            mc_plot_dsq = mc_dsq

        # Observed star dist_sq_per_arm
        camp_data = campaigns[campaign]
        if camp_data.ell is None:
            print(f"  No Galactic coordinates for {campaign}, skipping obs")
            continue

        ell_obs = np.asarray(camp_data.ell)
        b_obs = np.asarray(camp_data.b)
        print(f"  Computing obs dist_sq_per_arm ({len(ell_obs)} stars) ...")

        obs_dsq = compute_dist_sq_per_arm(
            ell_obs, b_obs, d_grid, arms_xy, R_sun)
        print(f"  Obs dist_sq_per_arm: shape={obs_dsq.shape}, "
              f"min={obs_dsq.min():.4f}, max={obs_dsq.max():.4f}")

        save_dict[f"{campaign}_obs_dist_sq_per_arm"] = obs_dsq
        save_dict[f"{campaign}_ell_obs"] = ell_obs
        save_dict[f"{campaign}_b_obs"] = b_obs

    # --- Save ---
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, **save_dict)
    print(f"\nSaved to {cache_path}")

    # --- Plot ---
    plot_data = {}
    for campaign, camp_data in campaigns.items():
        if camp_data.ell is None:
            continue
        key = f"{campaign}_obs_dist_sq_per_arm"
        if key not in save_dict:
            continue
        plot_data[campaign] = {
            "d_grid": d_grids[campaign],
            "obs_dist_sq_per_arm": save_dict[key],
            "ell": np.asarray(camp_data.ell),
            "b": np.asarray(camp_data.b),
            "pi_EDR3": np.asarray(camp_data.pi_EDR3),
        }

    obs_data = {c: (np.asarray(d.ell), np.asarray(d.b))
                for c, d in campaigns.items() if d.ell is not None}
    for campaign in campaigns_sorted:
        sightline_fname = os.path.join(
            data_dir, f"spiral_sightline_{campaign}.png")
        plot_sightlines(
            ell_mc_dict[campaign], b_mc_dict[campaign],
            sl_ell_min, sl_ell_max, sl_b_min, sl_b_max,
            obs_data, sightline_fname)

    plot_fname = os.path.join(data_dir, "spiral_profiles.png")
    plot_spiral_profiles(
        plot_data, mc_plot_dsq, mc_plot_grid,
        arm_xgc, arm_ygc, R_sun, R_d, z_d, plot_arm_frac, plot_width,
        delta_pi, plot_fname)


if __name__ == "__main__":
    main()
