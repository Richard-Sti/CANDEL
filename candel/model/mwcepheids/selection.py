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
"""Selection function configuration and utilities."""
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.scipy.special import log_ndtr, logsumexp, ndtr

from ..integration import ln_simpson_uniform
from ..utils import get_named_or_shared, load_priors, sample_prior

# Gauss-Legendre nodes/weights for bivariate normal CDF quadrature
_GL_NODES_16, _GL_WEIGHTS_16 = np.polynomial.legendre.leggauss(16)
_GL_NODES_16 = jnp.array(_GL_NODES_16)
_GL_WEIGHTS_16 = jnp.array(_GL_WEIGHTS_16)


@dataclass
class SelectionConfig:
    """Configuration for magnitude/parallax selection.

    Parameters
    ----------
    apply : bool
        Whether to apply selection correction.
    threshold : float or None
        Selection threshold value (mW_max, mW_min, or pi_min).
    infer_threshold : bool
        Whether to infer threshold from prior.
    width : float
        Selection width (added in quadrature to measurement error).
    infer_width : bool
        Whether to infer width from prior.
    priors : dict
        Prior distributions for inferred parameters.
    is_upper : bool
        True for upper limit (mW < threshold), False for lower limit.
    """
    apply: bool = False
    threshold: Optional[float] = None
    infer_threshold: bool = False
    width: float = 0.0
    infer_width: bool = False
    priors: dict = field(default_factory=dict)
    is_upper: bool = True  # True for mW < mW_max, False for mW > mW_min

    @classmethod
    def from_config(cls, cfg, threshold_key, width_key=None,
                    default_threshold=None, is_upper=True):
        """Create SelectionConfig from a config dictionary.

        Parameters
        ----------
        cfg : dict
            Selection configuration section.
        threshold_key : str
            Key for threshold value (e.g., "mW_max", "mW_min", "pi_min").
        width_key : str, optional
            Key for width value (e.g., "mW_width", "pi_width").
        default_threshold : float, optional
            Default threshold if not in config.
        is_upper : bool
            True for upper limit selection.
        """
        apply = cfg.get("apply", False)

        # Parse threshold
        threshold_cfg = cfg.get(threshold_key, default_threshold)
        if threshold_cfg == "infer":
            infer_threshold = True
            threshold = None
        else:
            infer_threshold = False
            threshold = (float(threshold_cfg) if threshold_cfg is not None
                         else None)

        # Parse width
        if width_key:
            width_cfg = cfg.get(width_key, 0.0)
            if width_cfg == "infer":
                infer_width = True
                width = 0.0
            else:
                infer_width = False
                width = float(width_cfg)
        else:
            infer_width = False
            width = 0.0

        # Load priors
        priors_cfg = cfg.get("priors", {})
        priors = load_priors(priors_cfg)[0] if priors_cfg else {}

        return cls(
            apply=apply,
            threshold=threshold,
            infer_threshold=infer_threshold,
            width=width,
            infer_width=infer_width,
            priors=priors,
            is_upper=is_upper,
        )


@dataclass
class C22SelectionConfig:
    """C22 selection config (magnitude, extinction, parallax, period)."""
    mW: SelectionConfig = field(default_factory=SelectionConfig)
    AH: SelectionConfig = field(default_factory=SelectionConfig)
    pi: SelectionConfig = field(default_factory=SelectionConfig)
    logP: SelectionConfig = field(default_factory=SelectionConfig)
    pi_smooth: bool = False
    dust_map: str = "bayestar"
    delta_pi_max: float = 0.1

    @classmethod
    def from_config(cls, cfg):
        """Create from model.C22.selection config."""
        selection_cfg = cfg.get("selection", {})
        mW = SelectionConfig.from_config(
            selection_cfg,
            threshold_key="mW_max",
            width_key="mW_width",
            default_threshold=7.736,
            is_upper=True,
        )
        mW.apply = selection_cfg.get("apply_mW", False)

        # Extinction selection: A_H < AH_max
        AH = SelectionConfig.from_config(
            selection_cfg,
            threshold_key="AH_max",
            width_key="AH_width",
            default_threshold=0.4,
            is_upper=True,
        )
        AH.apply = selection_cfg.get("apply_AH", False)
        # Guard against division by zero in log_ndtr
        if AH.apply and AH.width == 0.0:
            AH.width = 0.01

        # Parallax selection: pi > pi_min
        pi = SelectionConfig.from_config(
            selection_cfg,
            threshold_key="pi_min",
            width_key="pi_width",
            default_threshold=None,
            is_upper=False,
        )
        pi.apply = selection_cfg.get("apply_pi", False)

        # Period selection: logP > logP_min
        logP = SelectionConfig.from_config(
            selection_cfg,
            threshold_key="logP_min",
            width_key="logP_width",
            default_threshold=None,
            is_upper=False,
        )
        logP.apply = selection_cfg.get("apply_logP", False)

        dust_map = selection_cfg.get("dust_map", "bayestar")

        return cls(
            mW=mW, AH=AH, pi=pi, logP=logP,
            pi_smooth=selection_cfg.get("pi_smooth", False),
            dust_map=dust_map,
            delta_pi_max=selection_cfg.get("delta_pi_max", 0.1),
        )


@dataclass
class C27SelectionConfig:
    """C27 selection configuration (parallax and optional mW lower limit)."""
    pi: SelectionConfig = field(default_factory=SelectionConfig)
    mW: SelectionConfig = field(default_factory=SelectionConfig)
    pi_smooth: bool = False
    delta_pi_max: float = 0.1

    @classmethod
    def from_config(cls, cfg):
        """Create from model.C27.selection config."""
        selection_cfg = cfg.get("selection", {})

        # Parallax selection
        pi = SelectionConfig.from_config(
            selection_cfg,
            threshold_key="pi_min",
            width_key="pi_width",
            default_threshold=0.8,
            is_upper=False,
        )
        pi.apply = selection_cfg.get("apply_pi", False)

        # mW lower limit selection
        mW = SelectionConfig.from_config(
            selection_cfg,
            threshold_key="mW_min",
            width_key="mW_width",
            default_threshold=None,
            is_upper=False,
        )
        mW.apply = selection_cfg.get("apply_mW", False)

        return cls(
            pi=pi,
            mW=mW,
            pi_smooth=selection_cfg.get("pi_smooth", False),
            delta_pi_max=selection_cfg.get("delta_pi_max", 0.1),
        )


@dataclass
class SelectionMCData:
    """Precomputed per-campaign arrays for selection evaluation."""
    d_grid: object
    dx: float
    mu_grid: object
    inv_d_grid: object
    median_sigma_m: float
    median_sigma_pi: float
    mc_log_dist_prior: object
    mc_log_dist_norm: object
    n_mc: int
    AH_mc_grid: object = None
    AH_mc_valid: object = None
    AH_obs_profiles: object = None
    AH_obs_star_valid: object = None
    spiral_dist_sq_per_arm_grid: object = None


def log_probit_selection(obs, threshold, width, is_upper=True):
    """Log probit selection factor: Phi((obs - threshold) / width).

    Parameters
    ----------
    obs : array
        Observed values.
    threshold : float
        Selection threshold.
    width : float
        Probit transition width (> 0).
    is_upper : bool
        If True, selects obs < threshold. If False, selects obs > threshold.

    Returns
    -------
    array
        Log selection probability.
    """
    if is_upper:
        return log_ndtr((threshold - obs) / width)
    else:
        return log_ndtr((obs - threshold) / width)


def log_AH_selection(AH, AH_max, AH_width=0.01):
    """Log P(A_H < AH_max) — smooth step via probit.

    Parameters
    ----------
    AH : array
        H-band extinction values.
    AH_max : float
        Maximum allowed extinction.
    AH_width : float
        Smoothing width (probit scale; 0.01 ~ hard cut).

    Returns
    -------
    array
        Log selection probability.
    """
    return log_ndtr((AH_max - AH) / AH_width)


def log_phi2_cdf(x1, x2, rho):
    """Log bivariate normal CDF via Gauss-Legendre quadrature.

    Uses the integral representation:
        Phi_2(x1, x2; rho) = Phi(x1)*Phi(x2)
            + int_0^rho phi_2(x1, x2; t) dt

    Parameters
    ----------
    x1 : array
        First argument (can be array).
    x2 : float
        Second argument (scalar).
    rho : float
        Correlation coefficient.

    Returns
    -------
    array
        Log Phi_2(x1, x2; rho), same shape as x1.
    """
    # Map quadrature from [-1, 1] to [0, rho]
    t = 0.5 * rho * (_GL_NODES_16 + 1)
    w = 0.5 * rho * _GL_WEIGHTS_16

    # Bivariate normal density at each quadrature point
    det = 1 - t**2
    exponent = -(x1[..., None]**2 - 2 * t * x1[..., None] * x2
                 + x2**2) / (2 * det)
    density = jnp.exp(exponent) / (2 * jnp.pi * jnp.sqrt(det))
    integral = jnp.sum(w * density, axis=-1)

    result = ndtr(x1) * ndtr(x2) + integral
    return jnp.log(jnp.maximum(result, 1e-30))


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def spiral_log_factor(dist_sq_per_arm, arm_frac, width):
    """Compute spiral log-factor from precomputed per-arm dist_sq.

    Parameters
    ----------
    dist_sq_per_arm : array, shape (n_arms, ...)
        Squared distance to nearest trace point per arm.
    arm_frac : float
        Fraction of density in spiral arms.
    width : float
        Gaussian arm width sigma [kpc].
    """
    gauss_sum = jnp.sum(
        jnp.exp(-dist_sq_per_arm / (2 * width**2)), axis=0)
    return jnp.log((1 - arm_frac) + arm_frac * gauss_sum)


# -------------------------------------------------------------------------
# Selection sampling and evaluation
# -------------------------------------------------------------------------

def sample_selection_params(sel_c22, sel_c27, campaigns):
    """Sample selection parameters for C22 and C27.

    All parameter keys carry a campaign suffix (``_C22`` / ``_C27``)
    to avoid namespace collisions.
    """
    sel_params = {}

    # C22 parameters
    if sel_c22.mW.apply and "C22" in campaigns:
        if sel_c22.mW.infer_threshold:
            sel_params["mW_max_C22"] = sample_prior(
                "mW_max_C22", sel_c22.mW.priors["mW_max"])
        else:
            sel_params["mW_max_C22"] = sel_c22.mW.threshold

        if sel_c22.mW.infer_width:
            sel_params["mW_width_C22"] = sample_prior(
                "mW_width_C22", sel_c22.mW.priors["mW_width"])
        else:
            sel_params["mW_width_C22"] = sel_c22.mW.width

    if sel_c22.AH.apply and "C22" in campaigns:
        if sel_c22.AH.infer_threshold:
            sel_params["AH_max_C22"] = sample_prior(
                "AH_max_C22", sel_c22.AH.priors["AH_max"])
        else:
            sel_params["AH_max_C22"] = sel_c22.AH.threshold

    if sel_c22.pi.apply and "C22" in campaigns:
        if sel_c22.pi.infer_threshold:
            sel_params["pi_min_C22"] = sample_prior(
                "pi_min_C22", sel_c22.pi.priors["pi_min"])
        else:
            sel_params["pi_min_C22"] = sel_c22.pi.threshold

    if sel_c22.logP.apply and "C22" in campaigns:
        if sel_c22.logP.infer_threshold:
            sel_params["logP_min_C22"] = sample_prior(
                "logP_min_C22", sel_c22.logP.priors["logP_min"])
        else:
            sel_params["logP_min_C22"] = sel_c22.logP.threshold

        if sel_c22.logP.infer_width:
            sel_params["logP_width_C22"] = sample_prior(
                "logP_width_C22", sel_c22.logP.priors["logP_width"])
        else:
            sel_params["logP_width_C22"] = sel_c22.logP.width

    # C27 parameters
    if sel_c27.pi.apply and "C27" in campaigns:
        if sel_c27.pi.infer_threshold:
            sel_params["pi_min_C27"] = sample_prior(
                "pi_min_C27", sel_c27.pi.priors["pi_min"])
        else:
            sel_params["pi_min_C27"] = sel_c27.pi.threshold

        # C27 mW selection is only reachable when pi selection is active
        if sel_c27.mW.apply:
            if sel_c27.mW.infer_threshold:
                sel_params["mW_min_C27"] = sample_prior(
                    "mW_min_C27", sel_c27.mW.priors["mW_min"])
            else:
                sel_params["mW_min_C27"] = sel_c27.mW.threshold

    return sel_params


def log_selection_norm(campaign, mc_data, sel_c22, sel_c27,
                       params, apply_spiral):
    """Analytical selection normalisation.

    Marginalises over (logP, [O/H]) analytically using the multivariate
    normal CDF identity, and over sightlines (ell, b) via MC.
    Uses median measurement errors per campaign (sigma_m, sigma_pi).
    """
    sigma_int = get_named_or_shared("sigma_int", campaign, params)
    sigma_m = mc_data.median_sigma_m
    sigma_pi = params["f_pi"] * mc_data.median_sigma_pi

    # Population hyperparameters
    mu_logP = params[f"mu_logP_{campaign}"]
    sigma_logP = params[f"sigma_logP_{campaign}"]
    mu_OH = params[f"mu_OH_{campaign}"]
    sigma_OH = params[f"sigma_OH_{campaign}"]

    # Predicted magnitude at population means
    m_hat = (params["M_H_1"] + params["b_W"] * (mu_logP - 1)
             + params["Z_W"] * mu_OH + mc_data.mu_grid)

    # Fully inflated variance (mag + intrinsic + OH + logP)
    sigma_2_sq = (sigma_m**2 + sigma_int**2
                  + params["Z_W"]**2 * sigma_OH**2
                  + params["b_W"]**2 * sigma_logP**2)

    # Q index contribution (MW campaigns only)
    if "c_W" in params:
        mu_Q = params[f"mu_Q_{campaign}"]
        sigma_Q = params[f"sigma_Q_{campaign}"]
        m_hat = m_hat + params["c_W"] * mu_Q
        sigma_2_sq = sigma_2_sq + params["c_W"]**2 * sigma_Q**2

    if campaign == "C22":
        return _log_sel_norm_C22(
            mc_data, m_hat, sigma_2_sq, sigma_pi,
            mu_logP, sigma_logP, sel_c22, params, apply_spiral)
    else:
        return _log_sel_norm_C27(
            mc_data, m_hat, sigma_2_sq, sigma_pi,
            sel_c27, params, apply_spiral)


def _get_prior_and_norm(mc_data, params, apply_spiral):
    """Get distance log-prior (optionally with spirals) and log-norm.

    The log-norm returned here is the **disk-only** per-sightline
    normalisation Z_disk(u_j), regardless of whether spirals are
    active.  This is because MC sightlines are drawn proportional to
    Z_disk (the disk column density), so Z_disk is the correct
    importance-sampling denominator.

    When spirals are active, the integrand includes the spiral factor
    (disk x spiral), but the denominator stays Z_disk.  Together with
    the unnormalised per-host radial integral (see
    ``_likelihood_campaign_marg_d``), this ensures that Z_total cancels
    between the numerator and the selection normalisation::

        S_unnorm = Z_total * E_q[ int P(sel) p_unnorm dr / Z_disk(u) ]

    and the per-host term is log int L_i p_unnorm dr, so Z_total
    cancels in ell_i = (per-host) - log S_unnorm.
    """
    log_prior = mc_data.mc_log_dist_prior
    if apply_spiral and mc_data.spiral_dist_sq_per_arm_grid is not None:
        spiral_factor = spiral_log_factor(
            mc_data.spiral_dist_sq_per_arm_grid,
            params["spiral_arm_frac"], params["spiral_width"])
        log_prior = log_prior + spiral_factor
    log_norm = mc_data.mc_log_dist_norm
    return log_prior, log_norm


def _mc_average(ln_integrand, mc_data, log_norm):
    """Integrate over d per sightline, then MC-average.

    Computes logmeanexp_j[ log int f(r, u_j) dr - log Z_disk(u_j) ],
    where the subtraction of Z_disk corrects for the importance
    sampling (sightlines drawn proportional to disk column density).
    """
    ln_Z_per_mc = ln_simpson_uniform(ln_integrand, mc_data.dx, axis=-1)
    K = ln_Z_per_mc.shape[0]
    return logsumexp(ln_Z_per_mc - log_norm) - jnp.log(K)


def _log_sel_norm_C27(mc_data, m_hat, sigma_2_sq, sigma_pi,
                      sel_c27, params, apply_spiral):
    """C27 analytical selection normalisation (K=2, factorises)."""
    log_P_det = jnp.zeros_like(mc_data.mu_grid)

    if sel_c27.mW.apply and "mW_min_C27" in params:
        w_m = sel_c27.mW.width
        h1 = (m_hat - params["mW_min_C27"]) / jnp.sqrt(w_m**2 + sigma_2_sq)
        log_P_det = log_P_det + log_ndtr(h1)

    if sel_c27.pi.apply:
        w_pi = sel_c27.pi.width if sel_c27.pi_smooth else 0.0
        delta_pi = jnp.clip(params["delta_pi"],
                            -sel_c27.delta_pi_max,
                            sel_c27.delta_pi_max)
        pi_pred = mc_data.inv_d_grid - delta_pi
        h2 = (pi_pred - params["pi_min_C27"]) / jnp.sqrt(
            w_pi**2 + sigma_pi**2)
        log_P_det = log_P_det + log_ndtr(h2)

    log_prior, log_norm = _get_prior_and_norm(mc_data, params, apply_spiral)
    ln_integrand = log_prior + log_P_det[None, :]
    return _mc_average(ln_integrand, mc_data, log_norm)


def _log_sel_norm_C22(mc_data, m_hat, sigma_2_sq, sigma_pi,
                      mu_logP, sigma_logP, sel_c22, params, apply_spiral):
    """C22 analytical selection normalisation (K=3, uses Phi_2)."""
    log_P_det = jnp.zeros_like(mc_data.mu_grid)

    # h2: parallax lower limit
    if sel_c22.pi.apply:
        w_pi = sel_c22.pi.width if sel_c22.pi_smooth else 0.0
        delta_pi = jnp.clip(params["delta_pi"],
                            -sel_c22.delta_pi_max,
                            sel_c22.delta_pi_max)
        pi_pred = mc_data.inv_d_grid - delta_pi
        h2 = (pi_pred - params["pi_min_C22"]) / jnp.sqrt(
            w_pi**2 + sigma_pi**2)
        log_P_det = log_P_det + log_ndtr(h2)

    has_mW = sel_c22.mW.apply
    has_logP = sel_c22.logP.apply

    if has_mW and has_logP:
        w_m = params.get("mW_width_C22", 0.0)
        denom_m = jnp.sqrt(w_m**2 + sigma_2_sq)
        h1 = (params["mW_max_C22"] - m_hat) / denom_m

        w_logP = params.get("logP_width_C22", sel_c22.logP.width)
        denom_logP = jnp.sqrt(w_logP**2 + sigma_logP**2)
        h3 = (mu_logP - params["logP_min_C22"]) / denom_logP

        R13 = -params["b_W"] * sigma_logP**2 / (denom_m * denom_logP)
        log_P_det = log_P_det + log_phi2_cdf(h1, h3, R13)

    elif has_mW:
        w_m = params.get("mW_width_C22", 0.0)
        h1 = (params["mW_max_C22"] - m_hat) / jnp.sqrt(w_m**2 + sigma_2_sq)
        log_P_det = log_P_det + log_ndtr(h1)

    elif has_logP:
        w_logP = params.get("logP_width_C22", sel_c22.logP.width)
        h3 = (mu_logP - params["logP_min_C22"]) / jnp.sqrt(
            w_logP**2 + sigma_logP**2)
        log_P_det = log_P_det + log_ndtr(h3)

    log_prior, log_norm = _get_prior_and_norm(mc_data, params, apply_spiral)

    # Extinction selection (per-sightline)
    if sel_c22.AH.apply and mc_data.AH_mc_grid is not None:
        log_P_AH = jnp.where(
            mc_data.AH_mc_valid,
            log_AH_selection(mc_data.AH_mc_grid, params["AH_max_C22"],
                             sel_c22.AH.width),
            0.0)
    else:
        log_P_AH = 0.0

    ln_integrand = log_prior + log_P_det[None, :] + log_P_AH
    return _mc_average(ln_integrand, mc_data, log_norm)


def selection_correction(campaign, d, data, params,
                         mc_data, sel_c22, sel_c27, apply_spiral,
                         marg_d=False):
    """Truncated likelihood correction for selection effects.

    Per-star factors use observed data (pi_obs, mW_obs) — not the
    latent distance d — since the selection was applied to observables.
    Exception: A_H(d) is deterministic given d and constrains d directly.

    Adds two numpyro.factor terms to the log-joint:
      (1) sum_i log P(selected_i | obs_i, theta)   -- per-star
      (2) -N * log E[P(selected | theta)]           -- normalization
    """
    c22_sel_active = (sel_c22.mW.apply or sel_c22.AH.apply
                      or sel_c22.pi.apply or sel_c22.logP.apply)
    if campaign == "C22" and c22_sel_active:
        # --- (1) Per-star selection probability ---
        log_P_sel = jnp.zeros(data.n_stars)

        if sel_c22.mW.apply:
            mW_max = params["mW_max_C22"]
            mW_width = params.get("mW_width_C22", 0.0)
            # width=0 means hard cut: all observed stars already pass,
            # so the per-star factor is 1 (log=0) and only the
            # normalisation integral contributes.
            if mW_width > 0:
                log_P_sel = log_P_sel + log_probit_selection(
                    data.mW_H, mW_max, mW_width, is_upper=True)

        if sel_c22.AH.apply and not marg_d:
            AH_max = params["AH_max_C22"]
            AH_width = sel_c22.AH.width
            # A_H(d) is deterministic — legitimate constraint on d.
            # When d is None (marginalised mode), AH is inside the
            # distance integral and handled by distance_marg.
            AH_at_d = jax.vmap(
                lambda profile, di: jnp.interp(
                    di, mc_data.d_grid, profile)
            )(mc_data.AH_obs_profiles, d)

            log_P_AH = jnp.where(
                mc_data.AH_obs_star_valid,
                log_AH_selection(AH_at_d, AH_max, AH_width),
                0.0)
            log_P_sel = log_P_sel + log_P_AH

        if sel_c22.pi.apply:
            pi_min = params["pi_min_C22"]
            pi_w = (sel_c22.pi.width
                    if sel_c22.pi_smooth else 0.0)
            # width=0: hard cut, per-star factor skipped (see above)
            if pi_w > 0:
                log_P_sel = log_P_sel + log_probit_selection(
                    data.pi_EDR3, pi_min, pi_w, is_upper=False)

        if sel_c22.logP.apply:
            logP_min = params["logP_min_C22"]
            logP_w = params.get("logP_width_C22", sel_c22.logP.width)
            if logP_w > 0:
                log_P_sel = log_P_sel + log_probit_selection(
                    data.logP, logP_min, logP_w, is_upper=False)

        numpyro.factor(f"sel_prob_{campaign}", jnp.sum(log_P_sel))

        # --- (2) Selection normalization (analytical) ---
        ln_Z_S = log_selection_norm(
            "C22", mc_data, sel_c22, sel_c27,
            params, apply_spiral)
        numpyro.factor(f"sel_norm_{campaign}", -data.n_stars * ln_Z_S)

    # C27 selection requires pi.apply; mW selection alone does not
    # activate the correction (mW is always nested under pi.apply).
    elif campaign == "C27" and sel_c27.pi.apply:
        # --- (1) Per-star selection probability ---
        log_P_sel = jnp.zeros(data.n_stars)

        pi_min = params["pi_min_C27"]
        pi_w = sel_c27.pi.width if sel_c27.pi_smooth else 0
        # width=0: hard cut, per-star factor skipped (see C22 comment)
        if pi_w > 0:
            log_P_sel = log_P_sel + log_probit_selection(
                data.pi_EDR3, pi_min, pi_w, is_upper=False)

        if sel_c27.mW.apply and "mW_min_C27" in params:
            mW_min = params["mW_min_C27"]
            mW_w = sel_c27.mW.width
            if mW_w > 0:
                log_P_sel = log_P_sel + log_probit_selection(
                    data.mW_H, mW_min, mW_w, is_upper=False)

        numpyro.factor(f"sel_prob_{campaign}", jnp.sum(log_P_sel))

        # --- (2) Selection normalization (analytical) ---
        ln_Z_S = log_selection_norm(
            "C27", mc_data, sel_c22, sel_c27,
            params, apply_spiral)
        numpyro.factor(f"sel_norm_{campaign}", -data.n_stars * ln_Z_S)
