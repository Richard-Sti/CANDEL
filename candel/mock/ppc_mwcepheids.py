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
"""Posterior predictive check generation for MW Cepheids."""
import logging
import os

import numpy as np
from tqdm.auto import tqdm

from ..pvdata.dust import postprocess_extinction_profiles
from ..model.mwcepheids.selection import C22SelectionConfig, C27SelectionConfig

logger = logging.getLogger(__name__)

def _generate_ppc_campaign(samples, data, config, rng):
    """Generate PPC samples for a single campaign via rejection sampling.

    Parameters
    ----------
    samples : dict
        Posterior samples from MCMC.
    data : CepheidData
        Data for a single campaign (data.campaign must be "C22" or "C27").
    config : dict
        Full configuration dictionary.
    rng : numpy.random.Generator
        Random number generator (mutated in place).

    Returns
    -------
    mW_sim : ndarray
        Simulated Wesenheit magnitudes (accepted after selection).
    pi_sim : ndarray
        Simulated parallaxes (accepted after selection).
    logP_sim : ndarray
        Simulated log-periods (accepted after selection).
    mW_obs : ndarray
        Observed magnitudes (filtered by pi_cut if applicable).
    pi_obs : ndarray
        Observed parallaxes (filtered by pi_cut if applicable).
    logP_obs : ndarray
        Observed log-periods.
    """
    model_cfg = config.get("model", {})
    distance_prior = model_cfg.get("distance_prior", "disk")

    # Disk prior parameters ("volume" is disk with R_d, z_d -> inf)
    disk_cfg = model_cfg.get("disk_prior", {})
    if distance_prior == "volume":
        distance_prior = "disk"
        disk_R_d = 1e10
        disk_z_d = 1e10
    else:
        disk_R_d = disk_cfg.get("R_d", 2.5)
        disk_z_d = disk_cfg.get("z_d", 0.1)
    disk_R_sun = disk_cfg.get("R_sun", 8.122)

    campaign = data.campaign
    camp_cfg = model_cfg.get(campaign, {})

    # Parse selection configs
    sel_c22 = C22SelectionConfig.from_config(model_cfg.get("C22", {}))
    sel_c27 = C27SelectionConfig.from_config(model_cfg.get("C27", {}))

    is_c22_data = campaign == "C22"
    is_c27_data = campaign == "C27"

    apply_mW_upper_sel = sel_c22.mW.apply and is_c22_data
    apply_AH_sel = sel_c22.AH.apply and is_c22_data
    apply_pi_sel_c22 = sel_c22.pi.apply and is_c22_data
    apply_logP_sel = sel_c22.logP.apply and is_c22_data
    apply_pi_selection = sel_c27.pi.apply and is_c27_data
    apply_mW_lower_sel = sel_c27.mW.apply and is_c27_data

    # Distance bounds from campaign config (with data-derived fallback)
    d_min_cfg = camp_cfg.get("d_min", 0.1)
    d_max_cfg = camp_cfg.get("d_max", 10.0)
    pi_obs_arr = np.asarray(data.pi_EDR3)
    d_obs_arr = 1.0 / pi_obs_arr
    d_min = max(d_min_cfg, 0.5 * np.min(d_obs_arr))
    d_max = min(d_max_cfg, 2.0 * np.max(d_obs_arr))

    any_sel = (apply_mW_upper_sel or apply_AH_sel or apply_pi_sel_c22
               or apply_logP_sel or apply_pi_selection
               or apply_mW_lower_sel)
    logger.info(f"PPC ({campaign}): d = [{d_min:.2f}, {d_max:.2f}] kpc, "
                f"prior = {distance_prior}")
    if apply_mW_upper_sel:
        thr = "infer" if sel_c22.mW.infer_threshold else sel_c22.mW.threshold
        logger.info(f"  C22 mW upper selection (mW_max = {thr})")
    if apply_AH_sel:
        logger.info(f"  C22 AH extinction selection "
                    f"(AH_max = {sel_c22.AH.threshold}, "
                    f"dust_map = {sel_c22.dust_map})")
    if apply_pi_sel_c22:
        thr = ("infer" if sel_c22.pi.infer_threshold
               else sel_c22.pi.threshold)
        logger.info(f"  C22 pi selection (pi_min = {thr}, "
                    f"smooth = {sel_c22.pi_smooth})")
    if apply_logP_sel:
        thr = ("infer" if sel_c22.logP.infer_threshold
               else sel_c22.logP.threshold)
        logger.info(f"  C22 logP selection (logP_min = {thr})")
    if apply_pi_selection:
        thr = "infer" if sel_c27.pi.infer_threshold else sel_c27.pi.threshold
        logger.info(f"  C27 pi selection (pi_min = {thr})")
    if apply_mW_lower_sel:
        thr = "infer" if sel_c27.mW.infer_threshold else sel_c27.mW.threshold
        logger.info(f"  C27 mW lower selection (mW_min = {thr})")
    if not any_sel:
        logger.info("  No selection applied")

    # Extract posterior samples
    n_samples = len(samples["M_H_1"])
    n_stars = data.n_stars

    M_H_1 = np.asarray(samples["M_H_1"])
    b_W = np.asarray(samples["b_W"])
    Z_W = np.asarray(samples["Z_W"])
    delta_pi = np.asarray(samples["delta_pi"])
    f_pi = (np.asarray(samples["f_pi"]) if "f_pi" in samples
            else np.ones(n_samples))

    # Per-campaign or shared intrinsic scatter
    sig_key = f"sigma_int_{campaign}"
    sigma_int = np.asarray(
        samples[sig_key] if sig_key in samples else samples["sigma_int"])

    # Population hyperparameters (for drawing logP and [O/H]_true)
    mu_logP_samples = np.asarray(samples[f"mu_logP_{campaign}"])
    sigma_logP_samples = np.asarray(samples[f"sigma_logP_{campaign}"])
    mu_OH_samples = np.asarray(samples[f"mu_OH_{campaign}"])
    sigma_OH_samples = np.asarray(samples[f"sigma_OH_{campaign}"])

    def _sel_param(sel_cfg, sample_key):
        if sel_cfg.infer_threshold:
            return np.asarray(samples[sample_key])
        return np.full(n_samples, float(sel_cfg.threshold))

    def _sel_width(sel_cfg, sample_key):
        if sel_cfg.infer_width:
            return np.asarray(samples[sample_key])
        return np.full(n_samples, float(sel_cfg.width))

    # Selection parameters
    if apply_mW_upper_sel:
        mW_max_samples = _sel_param(sel_c22.mW, "mW_max_C22")
        mW_width_c22_samples = _sel_width(sel_c22.mW, "mW_width_C22")
    if apply_pi_sel_c22:
        pi_min_c22_samples = _sel_param(sel_c22.pi, "pi_min_C22")
    if apply_pi_selection:
        pi_min_samples = _sel_param(sel_c27.pi, "pi_min_C27")
    if apply_AH_sel:
        AH_max_samples = _sel_param(sel_c22.AH, "AH_max_C22")
    if apply_mW_lower_sel:
        mW_min_samples = _sel_param(sel_c27.mW, "mW_min_C27")
    if apply_logP_sel:
        logP_min_samples = _sel_param(sel_c22.logP, "logP_min_C22")
        logP_width_samples = _sel_width(sel_c22.logP, "logP_width_C22")

    # Data arrays (measurement errors resampled from observed stars)
    mW_H_err = np.asarray(data.mW_H_err)
    pi_EDR3_err = np.asarray(data.pi_EDR3_err)

    # Load MC sightlines from caches (spiral first, then extinction).
    mc_seed = model_cfg.get("mc_seed", 42)
    dust_map = sel_c22.dust_map
    data_dir = config.get("local", {}).get("paths", {}).get("data", ".")

    ell_mc_cache = None
    b_mc_cache = None
    spiral_dsq_profiles = None
    spiral_d_grid = None

    # Spiral arm configuration
    apply_spirals = model_cfg.get("spiral_arms", {}).get("apply", False)

    # Try spiral cache first
    spiral_cache_path = os.path.join(
        data_dir, f"spiral_cache_seed{mc_seed}.npz")
    if os.path.exists(spiral_cache_path):
        cached_sp = np.load(spiral_cache_path)
        key_ell = f"{campaign}_ell_mc"
        key_b = f"{campaign}_b_mc"
        if key_ell in cached_sp:
            ell_mc_cache = cached_sp[key_ell]
            b_mc_cache = cached_sp[key_b]
            logger.info(f"  Loaded {len(ell_mc_cache)} MC sightlines "
                        f"from spiral cache")

        # Load dist_sq_per_arm profiles for spiral rejection
        key_dsq = f"{campaign}_mc_dist_sq_per_arm"
        key_dg = f"{campaign}_d_grid"
        if apply_spirals and key_dsq in cached_sp:
            spiral_dsq_profiles = cached_sp[key_dsq]
            spiral_d_grid = cached_sp[key_dg]
            logger.info(f"  Loaded spiral dist_sq profiles "
                        f"({spiral_dsq_profiles.shape[0]} arms, "
                        f"{spiral_dsq_profiles.shape[1]} sightlines)")

    # Try extinction cache (C22 only)
    ext_cache_path = os.path.join(
        data_dir, f"AH_cache_{dust_map}_seed{mc_seed}.npz")
    if is_c22_data and os.path.exists(ext_cache_path):
        cached = np.load(ext_cache_path)

        if ell_mc_cache is None:
            ell_mc_cache = cached["ell_mc"]
            b_mc_cache = cached["b_mc"]
            logger.info(f"  Loaded {len(ell_mc_cache)} MC sightlines "
                        f"from extinction cache")

        if apply_AH_sel:
            AH_mc_raw = cached["AH_mc_grid"]
            d_grid_cache = cached["d_grid"]
            AH_mc_profiles = postprocess_extinction_profiles(AH_mc_raw)

    n_mc_cached = len(ell_mc_cache) if ell_mc_cache is not None else 0

    if apply_spirals and spiral_dsq_profiles is None:
        logger.warning("  Spiral arms enabled but no spiral cache found "
                       "for PPC — ignoring spiral modulation")
        apply_spirals = False

    # Spiral posterior samples
    if apply_spirals:
        arm_frac_samples = np.asarray(
            samples.get("spiral_arm_frac", np.zeros(n_samples)))
        arm_width_samples = np.asarray(
            samples.get("spiral_width", np.full(n_samples, 0.3)))
        n_arms = spiral_dsq_profiles.shape[0]

    def _draw_distance_prior(n, ell_vals=None, b_vals=None):
        """Draw n distances from the prior."""
        u = rng.uniform(0, 1, size=n)
        if distance_prior == "disk":
            if ell_vals is None or b_vals is None:
                raise ValueError("Disk prior requires ell and b values")
            R_d, z_d, R_0 = disk_R_d, disk_z_d, disk_R_sun
            ell_rad = np.deg2rad(ell_vals)
            b_rad = np.deg2rad(b_vals)
            cos_ell = np.cos(ell_rad)
            cos_b = np.cos(b_rad)
            sin_b = np.sin(b_rad)

            d_prop = (d_min**3 + u * (d_max**3 - d_min**3)) ** (1.0 / 3.0)
            x = d_prop * cos_b * cos_ell
            y = d_prop * cos_b * np.sin(ell_rad)
            R_GC = np.sqrt((R_0 - x)**2 + y**2)
            z = d_prop * sin_b
            log_disk = -R_GC / R_d - np.abs(z) / z_d
            log_disk_max = 0.0
            accept_prob = np.exp(log_disk - log_disk_max)
            accept = rng.uniform(size=n) < accept_prob
            max_iter = 100
            for _ in range(max_iter):
                if accept.all():
                    break
                n_rej = (~accept).sum()
                u_new = rng.uniform(size=n_rej)
                d_new = (d_min**3 + u_new * (d_max**3 - d_min**3)) ** (1/3)
                x_new = d_new * cos_b[~accept] * cos_ell[~accept]
                R_GC_new = np.sqrt(
                    (R_0 - x_new)**2
                    + (d_new * cos_b[~accept]
                       * np.sin(ell_rad[~accept]))**2)
                z_new = d_new * sin_b[~accept]
                log_disk_new = -R_GC_new / R_d - np.abs(z_new) / z_d
                acc_new = rng.uniform(size=n_rej) < np.exp(log_disk_new)
                d_prop[~accept] = np.where(
                    acc_new, d_new, d_prop[~accept])
                accept[~accept] = acc_new
            return d_prop
        elif distance_prior == "distance_modulus":
            return d_min * np.exp(u * np.log(d_max / d_min))
        else:  # flat
            return d_min + u * (d_max - d_min)

    n_total = n_samples * n_stars
    batch_size = n_total * 5

    # Prepare sky coordinates for disk prior
    use_cached_sky = distance_prior == "disk" and ell_mc_cache is not None
    if distance_prior == "disk" and not use_cached_sky:
        if data.ell is None or data.b is None:
            raise ValueError(
                "Disk prior requires Galactic coordinates (ell, b) in data.")
        ell_obs_sky = np.asarray(data.ell)
        b_obs_sky = np.asarray(data.b)

    # Generate candidates in batches until we have enough accepted
    mW_acc_list, pi_acc_list, logP_acc_list = [], [], []
    n_accepted = 0
    n_generated = 0
    max_batches = 50

    pbar = tqdm(total=n_total, desc=f"PPC ({campaign})", unit="samples")

    for batch_i in range(max_batches):
        n_batch = batch_size

        # Resample measurement errors from observed stars
        idx_star = rng.choice(n_stars, n_batch)
        mW_err_cand = mW_H_err[idx_star]
        pi_err_cand = pi_EDR3_err[idx_star]

        # Draw sky coordinates and distances
        if distance_prior == "disk":
            if use_cached_sky:
                idx_sky = rng.choice(n_mc_cached, n_batch)
                ell_cand = ell_mc_cache[idx_sky]
                b_cand = b_mc_cache[idx_sky]
            else:
                idx_sky = rng.choice(len(ell_obs_sky), n_batch)
                ell_cand = ell_obs_sky[idx_sky]
                b_cand = b_obs_sky[idx_sky]
            d_cand = _draw_distance_prior(n_batch, ell_cand, b_cand)
        else:
            idx_sky = None
            d_cand = _draw_distance_prior(n_batch)

        post_idx = rng.integers(0, n_samples, n_batch)

        # Draw logP and [O/H]_true from population priors
        logP_cand = rng.normal(mu_logP_samples[post_idx],
                               sigma_logP_samples[post_idx])
        OH_cand = rng.normal(mu_OH_samples[post_idx],
                             sigma_OH_samples[post_idx])

        # Spiral arm rejection
        if apply_spirals and distance_prior == "disk":
            arm_frac = arm_frac_samples[post_idx]
            arm_width = arm_width_samples[post_idx]
            inv_2s2 = 1.0 / (2 * arm_width**2)

            d_idx_sp = np.searchsorted(spiral_d_grid, d_cand) - 1
            d_idx_sp = np.clip(d_idx_sp, 0, len(spiral_d_grid) - 2)
            dd = spiral_d_grid[d_idx_sp + 1] - spiral_d_grid[d_idx_sp]
            t_sp = np.clip(
                (d_cand - spiral_d_grid[d_idx_sp]) / dd, 0, 1)
            arange_n = np.arange(n_batch)

            gauss_sum = np.zeros(n_batch)
            for k in range(n_arms):
                profiles_k = spiral_dsq_profiles[k, idx_sky, :]
                dsq_at_d = (profiles_k[arange_n, d_idx_sp] * (1 - t_sp)
                            + profiles_k[arange_n, d_idx_sp + 1] * t_sp)
                gauss_sum += np.exp(-dsq_at_d * inv_2s2)

            spiral_factor = (1 - arm_frac) + arm_frac * gauss_sum
            spiral_max = 1 + arm_frac * (n_arms - 1)
            accept_sp = rng.uniform(size=n_batch) < (
                spiral_factor / np.maximum(spiral_max, 1e-10))

            idx_sky = idx_sky[accept_sp]
            d_cand = d_cand[accept_sp]
            post_idx = post_idx[accept_sp]
            logP_cand = logP_cand[accept_sp]
            OH_cand = OH_cand[accept_sp]
            mW_err_cand = mW_err_cand[accept_sp]
            pi_err_cand = pi_err_cand[accept_sp]
            n_batch = int(accept_sp.sum())

        # Compute observables
        M_pred = (M_H_1[post_idx] + b_W[post_idx] * (logP_cand - 1.0)
                  + Z_W[post_idx] * OH_cand)
        mu_cand = 5.0 * np.log10(d_cand) + 10.0
        mW_true = M_pred + mu_cand
        pi_true = 1.0 / d_cand - delta_pi[post_idx]

        sigma_mW_obs = np.sqrt(mW_err_cand**2 + sigma_int[post_idx]**2)
        mW_cand = rng.normal(mW_true, sigma_mW_obs)
        pi_cand = rng.normal(pi_true, f_pi[post_idx] * pi_err_cand)

        # Selection cuts
        accepted = np.ones(n_batch, dtype=bool)

        if apply_pi_selection:
            if sel_c27.pi_smooth and sel_c27.pi.width > 0:
                eff_pi_min = (pi_min_samples[post_idx]
                              - rng.normal(0, sel_c27.pi.width, n_batch))
            else:
                eff_pi_min = pi_min_samples[post_idx]
            accepted &= pi_cand > eff_pi_min

        if apply_pi_sel_c22:
            if sel_c22.pi_smooth and sel_c22.pi.width > 0:
                eff_pi_min = (pi_min_c22_samples[post_idx]
                              - rng.normal(0, sel_c22.pi.width, n_batch))
            else:
                eff_pi_min = pi_min_c22_samples[post_idx]
            accepted &= pi_cand > eff_pi_min

        if apply_mW_upper_sel:
            mW_width = mW_width_c22_samples[post_idx]
            eff_mW_max = mW_max_samples[post_idx] + rng.normal(0, mW_width)
            accepted &= mW_cand < eff_mW_max

        if apply_AH_sel:
            profiles_cand = AH_mc_profiles[idx_sky]
            d_idx_ah = np.searchsorted(d_grid_cache, d_cand) - 1
            d_idx_ah = np.clip(d_idx_ah, 0, len(d_grid_cache) - 2)
            t_ah = np.clip(
                (d_cand - d_grid_cache[d_idx_ah])
                / (d_grid_cache[d_idx_ah + 1] - d_grid_cache[d_idx_ah]),
                0.0, 1.0)
            arange_ah = np.arange(n_batch)
            AH_at_d = (profiles_cand[arange_ah, d_idx_ah] * (1 - t_ah)
                       + profiles_cand[arange_ah, d_idx_ah + 1] * t_ah)
            eff_AH_max = AH_max_samples[post_idx] + rng.normal(
                0, sel_c22.AH.width, n_batch)
            accepted &= AH_at_d < eff_AH_max

        if apply_mW_lower_sel:
            if sel_c27.mW.width > 0:
                eff_mW_min = (mW_min_samples[post_idx]
                              - rng.normal(0, sel_c27.mW.width, n_batch))
            else:
                eff_mW_min = mW_min_samples[post_idx]
            accepted &= mW_cand > eff_mW_min

        if apply_logP_sel:
            logP_w = logP_width_samples[post_idx]
            eff_logP_min = (logP_min_samples[post_idx]
                            + rng.normal(0, 1, n_batch) * logP_w)
            accepted &= logP_cand > eff_logP_min

        mW_acc_list.append(mW_cand[accepted])
        pi_acc_list.append(pi_cand[accepted])
        logP_acc_list.append(logP_cand[accepted])
        n_new = int(accepted.sum())
        n_accepted += n_new
        n_generated += n_batch
        pbar.update(min(n_new, n_total - (n_accepted - n_new)))

        if n_accepted >= n_total:
            break

        # Adapt batch size based on observed acceptance rate
        acc_rate = n_accepted / n_generated
        if acc_rate > 0:
            n_remaining = n_total - n_accepted
            batch_size = int(1.5 * n_remaining / acc_rate)
            batch_size = max(batch_size, 1000)

    pbar.close()

    mW_sim = np.concatenate(mW_acc_list)[:n_total]
    pi_sim = np.concatenate(pi_acc_list)[:n_total]
    logP_sim = np.concatenate(logP_acc_list)[:n_total]

    acc_rate = n_accepted / max(n_generated, 1)
    logger.info(f"  PPC acceptance: {n_accepted}/{n_generated} "
                f"({acc_rate:.1%}), batches={batch_i + 1}")

    if n_accepted < n_total:
        logger.warning(f"  Only {n_accepted}/{n_total} samples accepted "
                       f"after {max_batches} batches")

    mW_obs = np.asarray(data.mW_H)
    pi_obs = np.asarray(data.pi_EDR3)
    logP_obs = np.asarray(data.logP)

    return mW_sim, pi_sim, logP_sim, mW_obs, pi_obs, logP_obs


###############################################################################
# Public API
###############################################################################


def generate_ppc(samples, data, config, seed=42, n_ppc=None):
    """Generate PPC samples for all campaigns.

    Parameters
    ----------
    samples : dict
        Posterior samples from MCMC.
    data : CepheidData
        Loaded Cepheid data (single campaign or combined).
    config : dict
        Configuration dictionary.
    seed : int, optional
        Random seed for reproducibility.
    n_ppc : int, optional
        Downsample simulated data to this many points. If None, uses all.

    Returns
    -------
    result : dict
        Keys: ``mW_sim``, ``pi_sim``, ``logP_sim``, ``mW_obs``, ``pi_obs``,
        ``logP_obs``. Observed arrays are from the data; simulated arrays
        are from the posterior predictive.
    """
    rng = np.random.default_rng(seed)

    if data.campaign is None:
        datasets = data.split_by_campaign()
    else:
        datasets = {data.campaign: data}

    camp_results = {}
    for camp_name, camp_data in datasets.items():
        mW_s, pi_s, logP_s, mW_o, pi_o, logP_o = _generate_ppc_campaign(
            samples, camp_data, config, rng)
        camp_results[camp_name] = {
            "mW_sim": mW_s, "pi_sim": pi_s, "logP_sim": logP_s,
            "mW_obs": mW_o, "pi_obs": pi_o, "logP_obs": logP_o,
        }

    result = {}
    for key in ("mW_sim", "pi_sim", "logP_sim",
                "mW_obs", "pi_obs", "logP_obs"):
        result[key] = np.concatenate(
            [r[key] for r in camp_results.values()])

    if n_ppc is not None and n_ppc < len(result["mW_sim"]):
        idx = rng.choice(len(result["mW_sim"]), size=n_ppc, replace=False)
        for key in ("mW_sim", "pi_sim", "logP_sim"):
            result[key] = result[key][idx]

    return result
