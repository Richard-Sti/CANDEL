# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Minimal NumPyro model for CCHP TRGB distance calibrators to infer H0.

WARNING: This module is under development and likely incorrect. Use with
caution.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr
from numpyro import factor, handlers, plate, sample
from numpyro.distributions import MultivariateNormal, Normal, Uniform

from ...cosmo.cosmography import (Distance2Distmod, Distance2Redshift,
                                  Distmod2Distance, Distmod2Redshift,
                                  LogGrad_Distmod2ComovingDistance,
                                  Redshift2Distance)
from ...util import (fprint, get_nested, load_config, radec_to_cartesian,
                     replace_prior_with_delta)
from ..interp import LOSInterpolator
from ..pv_utils import lp_galaxy_bias, rsample, sample_galaxy_bias
from ..simpson import ln_simpson
from ..utils import (load_priors, log_prob_integrand_sel, logmeanexp,
                     normal_logpdf_var, predict_cz)
from .model_CSP import (CSPModel, CSPSelection, compute_per_source_selection,
                        extract_csp_median_errors, log1mexp)


def logsumexp_by_group(logp, idx, n_groups=None):
    """
    Compute per-group logsumexp given per-item logp and integer group indices.
    This sums (in probability space) the likelihoods within each group.
    """
    if n_groups is None:
        n_groups = int(jnp.max(idx)) + 1

    # Per-group baselines for numerical stability
    baseline = jnp.full(n_groups, -jnp.inf)
    baseline = baseline.at[idx].max(logp)

    # Broadcast baseline to per-item
    baseline_per_item = baseline[idx]
    exp_shifted = jnp.exp(logp - baseline_per_item)

    sum_exp = jnp.zeros(n_groups, dtype=logp.dtype)
    sum_exp = sum_exp.at[idx].add(exp_shifted)

    return jnp.log(sum_exp) + baseline


def logmeanexp_by_group(logp, idx, n_groups=None):
    """
    Compute per-group log-mean-exp given per-item logp and integer group
    indices.
    """
    if n_groups is None:
        n_groups = int(jnp.max(idx)) + 1
    counts = jnp.bincount(idx, length=n_groups)
    return logsumexp_by_group(logp, idx, n_groups) - jnp.log(counts)


def log_at_least_one_selected(log_p_sel_i, idx, n_groups=None):
    """
    Compute log P(at least one SN selected) per host.

    For host j with SNe i in group j:
        P(at least one) = 1 - prod_i(1 - p_i)
        log P(at least one) = log(1 - exp(sum_i log(1 - p_i)))
                            = log1mexp(sum_i log1mexp(log_p_i))

    Parameters
    ----------
    log_p_sel_i : array (n_sn,)
        Log selection probability for each SN.
    idx : array (n_sn,)
        Group index mapping each SN to its host.
    n_groups : int, optional
        Number of groups (hosts).

    Returns
    -------
    log_p_host : array (n_groups,)
        Log probability that at least one SN is selected per host.
    """
    if n_groups is None:
        n_groups = int(jnp.max(idx)) + 1

    # log(1 - p_i) for each SN
    log_one_minus_p_i = log1mexp(log_p_sel_i)

    # Sum per host: log(prod_i (1 - p_i)) = sum_i log(1 - p_i)
    log_prod_one_minus_p = jnp.zeros(n_groups, dtype=log_p_sel_i.dtype)
    log_prod_one_minus_p = log_prod_one_minus_p.at[idx].add(log_one_minus_p_i)

    # Clip away from 0 to avoid log1mexp(0) = -inf
    # When sum ≈ 0, all p_i ≈ 0, so P(at least one) ≈ sum(p_i) via Taylor
    log_prod_one_minus_p = jnp.minimum(log_prod_one_minus_p, -1e-7)

    # log(1 - prod(1 - p_i)) = log P(at least one selected)
    return log1mexp(log_prod_one_minus_p)


@dataclass
class CCHPTRGBSelectionContext:
    """Data container for selection computation dependencies."""
    # Interpolators (can be None if not using reconstruction)
    f_rand_los_delta: Optional[object]
    f_rand_los_velocity: Optional[object]
    # Random LOS unit vectors, shape (n_rand_los, 3)
    rhat_rand_los: jnp.ndarray
    # Radial grid for integration
    r_host_range: jnp.ndarray
    log_prior_r_grid: jnp.ndarray
    # Cosmography functions
    distance2redshift: object
    distance2distmod: object
    # Observed data for per-source selection
    cz_cmb: jnp.ndarray
    m_Bprime: jnp.ndarray
    # Median errors for selection smoothing
    sigma_cz_sel: float
    sigma_SN_sel: float
    # Flags
    use_reconstruction: bool
    which_bias: str
    quadratic_bias_delta0: float = 0.0


class CCHPTRGBSelectionComputation:
    """Compute selection corrections for the CCHP model."""

    def __init__(self, ctx):
        self.ctx = ctx

    def log_S_cz(self, lp_r, Vpec, H0, sigma_v, cz_lim, cz_width):
        """Probability of detection term if redshift-truncated."""
        ctx = self.ctx
        zcosmo = ctx.distance2redshift(ctx.r_host_range, h=H0 / 100)
        cz_r = predict_cz(zcosmo[None, None, :], Vpec)

        sigma_v = jnp.asarray(sigma_v)
        while sigma_v.ndim < cz_r.ndim:
            sigma_v = sigma_v[..., None]
        sigma_v = jnp.broadcast_to(sigma_v, cz_r.shape)

        log_prob = log_prob_integrand_sel(cz_r, sigma_v, cz_lim, cz_width)
        return ln_simpson(
            lp_r + log_prob, x=ctx.r_host_range[None, None, :], axis=-1)

    def log_S_SN_mag(self, lp_r, M_B, H0, mag_lim, mag_width):
        """Probability of detection term if supernova magnitude-truncated."""
        ctx = self.ctx
        mag = ctx.distance2distmod(ctx.r_host_range, h=H0 / 100) + M_B
        log_prob = log_prob_integrand_sel(
            mag[None, None, :], ctx.sigma_SN_sel, mag_lim, mag_width)
        return ln_simpson(
            lp_r + log_prob, x=ctx.r_host_range[None, None, :], axis=-1)

    def correction_redshift(self, h, sigma_v, beta, bias_params, Vext, H0,
                            cz_lim, cz_width):
        """Compute redshift selection correction using random LOS."""
        ctx = self.ctx
        lp_r_grid = ctx.log_prior_r_grid[None, None, :]

        # Project Vext onto random LOS directions
        # Works for rhat_rand_los both (n_los, 3) and (n_sims, n_los, 3)
        Vext_rad_rand = jnp.sum(Vext[None, :] * ctx.rhat_rand_los, axis=-1)

        if ctx.use_reconstruction:
            rand_los_delta_grid = \
                ctx.f_rand_los_delta.interp_many_steps_per_galaxy(
                    ctx.r_host_range * h)
            lp_r_grid += lp_galaxy_bias(
                rand_los_delta_grid, jnp.log1p(rand_los_delta_grid),
                bias_params, ctx.which_bias,
                ctx.quadratic_bias_delta0)

            Vpec_grid = beta * \
                ctx.f_rand_los_velocity.interp_many_steps_per_galaxy(
                    ctx.r_host_range * h)
            Vpec_grid = Vpec_grid + Vext_rad_rand[None, :, None]
        else:
            Vpec_grid = Vext_rad_rand[None, :, None]

        # Marginalized selection over random LOS, shape (nfields, n_rand_los)
        sigma_cz_sel = jnp.sqrt(ctx.sigma_cz_sel**2 + sigma_v**2)
        log_S = self.log_S_cz(
            lp_r_grid, Vpec_grid, H0, sigma_cz_sel, cz_lim, cz_width)

        # Average over random LOS, shape (nfields,)
        log_S = logmeanexp(log_S, axis=1)

        # Per-source selection factor, shape (ngal,)
        log_p_sel_i = log_ndtr((cz_lim - ctx.cz_cmb) / cz_width)

        return log_S, log_p_sel_i

    def correction_SN_mag(self, h, bias_params, M_B, H0, mag_lim, mag_width):
        """Compute SN magnitude selection correction using random LOS."""
        ctx = self.ctx
        lp_r_grid = ctx.log_prior_r_grid[None, None, :]

        if ctx.use_reconstruction:
            rand_los_delta_grid = \
                ctx.f_rand_los_delta.interp_many_steps_per_galaxy(
                    ctx.r_host_range * h)
            lp_r_grid += lp_galaxy_bias(
                rand_los_delta_grid, jnp.log1p(rand_los_delta_grid),
                bias_params, ctx.which_bias,
                ctx.quadratic_bias_delta0)

        # Marginalized selection over random LOS, shape (nfields, n_rand_los)
        log_S = self.log_S_SN_mag(lp_r_grid, M_B, H0, mag_lim, mag_width)

        # Average over random LOS, shape (nfields,)
        log_S = logmeanexp(log_S, axis=1)

        # Per-source selection factor, shape (ngal,)
        log_p_sel_i = log_ndtr((mag_lim - ctx.m_Bprime) / mag_width)

        return log_S, log_p_sel_i


class BaseCCHPModel(ABC):
    """
    Base class for the CCHP model, providing common functionality and
    configuration loading.
    """

    def __init__(self, config_path, data):
        config = load_config(config_path, replace_los_prior=False)

        # Set unused parameter priors to delta functions.
        config = self.replace_priors(config)

        # Unpack and set the priors.
        priors = config["model"]["priors"]
        self.priors, self.prior_dist_name = load_priors(priors)

        self.config = config
        self.use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        self.which_bias = get_nested(config, "model/which_bias", "linear")
        self.quadratic_bias_delta0 = get_nested(
            config, "pv_model/quadratic_bias_delta0", 0.0)
        self.which_selection = get_nested(
            config, "model/which_selection", None)
        if self.use_reconstruction:
            recon_name = get_nested(config, "io/which_host_los", "unspecified")
            fprint(f"Using reconstruction: {recon_name}")
        fprint(f"which_bias set to {self.which_bias}")
        fprint(f"selection set to {self.which_selection}")
        self.use_cz_likelihood = get_nested(
            config, "model/use_cz_likelihood", True)
        if not self.use_cz_likelihood:
            fprint("TRGB cz likelihood DISABLED")
        self.mag_lim_SN = get_nested(config, "model/mag_lim_SN", None)
        self.mag_lim_SN_width = get_nested(
            config, "model/mag_lim_SN_width", None)

        self.cz_lim_selection = get_nested(
            config, "model/cz_lim_selection", 2000.0)
        self.cz_lim_selection_width = get_nested(
            config, "model/cz_lim_selection_width", 300.0)

        # Selection inference settings
        self.infer_sel = get_nested(config, "model/infer_sel", False)
        if self.which_selection not in (None, "none"):
            fprint(f"infer_sel set to {self.infer_sel}")
        self.cz_lim_selection_min = get_nested(
            config, "model/cz_lim_selection_min", 500.0)
        self.cz_lim_selection_max = get_nested(
            config, "model/cz_lim_selection_max", 10000.0)
        self.cz_lim_selection_width_min = get_nested(
            config, "model/cz_lim_selection_width_min", 50.0)
        self.cz_lim_selection_width_max = get_nested(
            config, "model/cz_lim_selection_width_max", 2000.0)
        self.mag_lim_SN_min = get_nested(
            config, "model/mag_lim_SN_min", 10.0)
        self.mag_lim_SN_max = get_nested(
            config, "model/mag_lim_SN_max", 18.0)
        self.mag_lim_SN_width_min = get_nested(
            config, "model/mag_lim_SN_width_min", 0.05)
        self.mag_lim_SN_width_max = get_nested(
            config, "model/mag_lim_SN_width_max", 2.0)

        if self.which_selection == "SN_magnitude":
            if self.infer_sel:
                fprint(f"SN selection: inferring mag_lim_SN in "
                       f"[{self.mag_lim_SN_min}, {self.mag_lim_SN_max}], "
                       f"width in [{self.mag_lim_SN_width_min}, "
                       f"{self.mag_lim_SN_width_max}]")
            else:
                fprint(f"SN selection: mag_lim_SN={self.mag_lim_SN}, "
                       f"width={self.mag_lim_SN_width}")
        elif self.which_selection == "redshift":
            if self.infer_sel:
                fprint(f"Redshift selection: inferring cz_lim in "
                       f"[{self.cz_lim_selection_min}, "
                       f"{self.cz_lim_selection_max}], "
                       f"width in [{self.cz_lim_selection_width_min}, "
                       f"{self.cz_lim_selection_width_max}]")
            else:
                fprint(f"Redshift selection: cz_lim={self.cz_lim_selection}, "
                       f"width={self.cz_lim_selection_width}")
        elif self.which_selection == "CSP":
            if get_nested(config, "model/use_stretch_gmm", False):
                raise NotImplementedError(
                    "CSP selection with GMM for stretch is not yet supported.")
            if self.use_reconstruction:
                raise NotImplementedError(
                    "CSP selection with reconstruction is not yet supported.")
            fprint("CSP selection: using stretch and color from matched CSP")

        # Load and possibly mask input data (depends on which_selection)
        self.set_data(data)

        # Initialize the interpolators
        self.Om = get_nested(config, "model/Om", 0.3)
        self.distmod2distance = Distmod2Distance(Om0=self.Om)
        self.distmod2redshift = Distmod2Redshift(Om0=self.Om)
        self.distance2distmod_scalar = Distance2Distmod(
            Om0=self.Om, is_scalar=True)
        self.distance2distmod = Distance2Distmod(Om0=self.Om)
        self.distance2redshift = Distance2Redshift(Om0=self.Om)
        self.log_grad_distmod2comoving_distance = LogGrad_Distmod2ComovingDistance(Om0=self.Om)  # noqa

        self.num_hosts = self.mag_obs.shape[0]

        # Distance modulus limits for sampling
        self.distmod_limits = config.get("model", {}).get(
            "distmod_limits", [25.0, 35.0])
        fprint(f"distmod_limits set to {self.distmod_limits}")
        self.distmod_limits_LMC = config["model"].get(
            "distmod_limits_LMC", self.distmod_limits)
        self.distmod_limits_N4258 = config["model"].get(
            "distmod_limits_N4258", self.distmod_limits)
        self.has_host_los = False
        self.num_fields = 1

        r0_decay_scale = get_nested(config, "io/los_r0_decay_scale", 5.0)
        fprint(f"los_r0_decay_scale set to {r0_decay_scale}")
        if "host_los_velocity" in data and "host_los_r" in data:
            self.has_host_los = True
            self.num_fields = data["host_los_velocity"].shape[0]
            fprint(f"Number of LOS field realisations: {self.num_fields}")
            if "host_los_density" in data:
                los_delta = jnp.asarray(data["host_los_density"]) - 1.0
                self.f_host_los_delta = LOSInterpolator(
                    data["host_los_r"],
                    los_delta,
                    r0_decay_scale=r0_decay_scale,
                    extrap_constant=0.0,
                )
            self.f_host_los_velocity = LOSInterpolator(
                data["host_los_r"],
                jnp.asarray(data["host_los_velocity"]),
                r0_decay_scale=r0_decay_scale,
                extrap_constant=0.0,
            )
        if self.use_reconstruction and not self.has_host_los:
            raise ValueError(
                "`use_reconstruction = True` but no host LOS data provided.")

        # Load random LOS interpolators if available (for selection modelling)
        self.has_rand_los = False
        self.num_rand_los = 1
        if "rand_los_velocity" in data and "rand_los_r" in data:
            self.has_rand_los = True
            self.num_rand_los = data["rand_los_velocity"].shape[1]
            fprint(f"Number of random LOS: {self.num_rand_los}")

            rand_los_delta = jnp.asarray(data["rand_los_density"]) - 1.0
            self.f_rand_los_delta = LOSInterpolator(
                data["rand_los_r"],
                rand_los_delta,
                r0_decay_scale=r0_decay_scale,
                extrap_constant=0.0,
            )
            self.f_rand_los_velocity = LOSInterpolator(
                data["rand_los_r"],
                jnp.asarray(data["rand_los_velocity"]),
                r0_decay_scale=r0_decay_scale,
                extrap_constant=0.0,
            )

            # Unit vectors for random LOS directions
            ra = np.asarray(data["rand_los_RA"])
            dec = np.asarray(data["rand_los_dec"])
            if ra.ndim == 1:
                rhat = radec_to_cartesian(ra, dec)
            else:
                # Per-realisation (n_sims, n_gal) → (n_sims, n_gal, 3)
                ra_rad = np.deg2rad(ra)
                dec_rad = np.deg2rad(dec)
                cos_dec = np.cos(dec_rad)
                rhat = np.stack([
                    cos_dec * np.cos(ra_rad),
                    cos_dec * np.sin(ra_rad),
                    np.sin(dec_rad),
                ], axis=-1)
            # axis=-1 works for both (n_gal, 3) and (n_sims, n_gal, 3)
            n = jnp.linalg.norm(rhat, axis=-1, keepdims=True)
            self.rhat_rand_los = jnp.asarray(
                rhat / np.where(n == 0.0, 1.0, n))

        # Set up radial range for volume prior normalization
        r_limits_malmquist = get_nested(
            config, "model/r_limits_malmquist", [0.01, 350])
        num_points_malmquist = get_nested(
            config, "model/num_points_malmquist", 251)

        # Auto r_limits from observed cz
        is_auto = (isinstance(r_limits_malmquist, str)
                   and r_limits_malmquist.startswith("auto"))
        if is_auto:
            # Parse h from "auto" or "auto_0.7" format
            if "_" in r_limits_malmquist:
                h_auto = float(r_limits_malmquist.split("_")[1])
            else:
                h_auto = 1.0
            cz_obs = jnp.asarray(data["cz_cmb"])
            cz_obs_lim = [float(jnp.min(cz_obs)), float(jnp.max(cz_obs))]
            redshift2distance = Redshift2Distance(Om0=self.Om)
            r_from_cz = redshift2distance(
                jnp.array(cz_obs_lim), h=h_auto, is_velocity=True)
            # Buffer is 25% or at least 15 Mpc, final r_min >= 0.15 Mpc
            r_min_raw = float(r_from_cz[0])
            r_max_raw = float(r_from_cz[1])
            buffer_low = max(r_min_raw * 0.25, 15.0)
            buffer_high = max(r_max_raw * 0.25, 15.0)
            r_limits_malmquist = [max(r_min_raw - buffer_low, 0.15),
                                  r_max_raw + buffer_high]
            fprint(f"auto r_limits_malmquist (h={h_auto}): "
                   f"[{r_limits_malmquist[0]:.1f}, "
                   f"{r_limits_malmquist[1]:.1f}] Mpc "
                   f"(buffer: -{buffer_low:.1f}, +{buffer_high:.1f} Mpc)")

        r_range = jnp.linspace(
            r_limits_malmquist[0], r_limits_malmquist[1],
            num_points_malmquist)

        fprint(f"setting radial range from {r_limits_malmquist[0]:.1f} to "
               f"{r_limits_malmquist[1]:.1f} Mpc with {num_points_malmquist} "
               "points for the CCHP host galaxies.")
        self.r_host_range = r_range
        self.Rmax = jnp.max(self.r_host_range)

        # Precompute log prior on the fixed radial grid (used multiple times)
        self._log_prior_r_grid = self.log_prior_distance(self.r_host_range)

        # Validate that random LOS is available when selection is used with
        # reconstruction
        if self.which_selection not in (None, "none"):
            if self.use_reconstruction and not self.has_rand_los:
                raise ValueError(
                    "Selection modelling with reconstruction requires random "
                    "LOS data (auto-loaded when use_reconstruction=true).")

            # When not using reconstruction but applying selection, create
            # dummy homogeneous random LOS (like SH0ES)
            if not self.use_reconstruction:
                fprint("Creating dummy random LOS for selection (no recon).")
                self.num_rand_los = 1
                self.rhat_rand_los = jnp.zeros((1, 3))

        # Build selection context and computation if selection is enabled
        self.selection = None
        self.csp_selection = None
        if self.which_selection not in (None, "none"):
            if self.which_selection == "CSP":
                # CSP selection uses CSPSelection class
                cz_lim = (100.0, self.cz_lim_selection)
                use_cz_selection = False  # Disable redshift selection for TRGB
                # Read grid settings from config
                r_lim = get_nested(
                    config, "model/r_limits_selection", [5, 300])
                n_r = get_nested(config, "model/n_r_selection", 51)
                n_s = get_nested(config, "model/n_s_selection", 31)
                n_BV = get_nested(config, "model/n_BV_selection", 31)
                pad = get_nested(config, "model/auto_limits_nsigma", 5.0)
                # Get observed limits from data
                s_obs = self.CSP_obs_vec[:, 1]
                BV_obs = self.CSP_obs_vec[:, 2]
                csp_errors = extract_csp_median_errors(self.CSP_cov)
                s_obs_lim = (float(jnp.min(s_obs)), float(jnp.max(s_obs)))
                BV_obs_lim = (float(jnp.min(BV_obs)), float(jnp.max(BV_obs)))
                # True (latent) limits: observed range ± padding for error
                s_lim = (s_obs_lim[0] - pad * csp_errors["sigma_s"],
                         s_obs_lim[1] + pad * csp_errors["sigma_s"])
                BV_lim = (BV_obs_lim[0] - pad * csp_errors["sigma_BV"],
                          BV_obs_lim[1] + pad * csp_errors["sigma_BV"])
                fprint(f"TRGB-CSP auto s_limits (true): [{s_lim[0]:.3f}, "
                       f"{s_lim[1]:.3f}] (obs ± {pad}σ)")
                fprint(f"TRGB-CSP auto BV_limits (true): [{BV_lim[0]:.3f}, "
                       f"{BV_lim[1]:.3f}] (obs ± {pad}σ)")
                fprint(f"TRGB-CSP observed limits: s=[{s_obs_lim[0]:.3f}, "
                       f"{s_obs_lim[1]:.3f}], BV=[{BV_obs_lim[0]:.3f}, "
                       f"{BV_obs_lim[1]:.3f}]")
                # Auto r_lim from observed cz
                if isinstance(r_lim, str) and r_lim.startswith("auto"):
                    # Parse h from "auto" or "auto_0.7" format
                    if "_" in r_lim:
                        h_auto = float(r_lim.split("_")[1])
                    else:
                        h_auto = 1.0
                    cz_obs = self.cz_cmb  # per-SN cz in km/s
                    cz_obs_lim = [float(jnp.min(cz_obs)),
                                  float(jnp.max(cz_obs))]
                    redshift2distance = Redshift2Distance(Om0=self.Om)
                    r_from_cz = redshift2distance(
                        jnp.array(cz_obs_lim), h=h_auto, is_velocity=True)
                    # Buffer is 25% or at least 15 Mpc, final r_min >= 0.15 Mpc
                    r_min_raw = float(r_from_cz[0])
                    r_max_raw = float(r_from_cz[1])
                    buffer_low = max(r_min_raw * 0.25, 15.0)
                    buffer_high = max(r_max_raw * 0.25, 15.0)
                    r_lim = [max(r_min_raw - buffer_low, 0.15),
                             r_max_raw + buffer_high]
                    fprint(f"TRGB-CSP auto r_limits (h={h_auto}): "
                           f"[{r_lim[0]:.1f}, {r_lim[1]:.1f}] Mpc "
                           f"(buffer: -{buffer_low:.1f}, "
                           f"+{buffer_high:.1f} Mpc)")
                # Build grids
                r_grid = jnp.linspace(r_lim[0], r_lim[1], n_r)
                s_grid = jnp.linspace(s_lim[0], s_lim[1], n_s)
                BV_grid = jnp.linspace(BV_lim[0], BV_lim[1], n_BV)
                self.csp_selection = CSPSelection(
                    r_grid=r_grid, s_grid=s_grid, BV_grid=BV_grid,
                    distance2redshift=self.distance2redshift,
                    distance2distmod=self.distance2distmod,
                    cz_lim=cz_lim, s_obs_lim=s_obs_lim, BV_obs_lim=BV_obs_lim,
                    use_cz_selection=use_cz_selection)
                n_eval = n_r * n_s * n_BV
                fprint(f"CSP selection grid: r=[{r_lim[0]:.1f}, "
                       f"{r_lim[1]:.1f}] ({n_r}), "
                       f"s=[{s_lim[0]:.3f}, {s_lim[1]:.3f}] ({n_s}), "
                       f"BV=[{BV_lim[0]:.2f}, {BV_lim[1]:.2f}] ({n_BV})")
                if use_cz_selection:
                    fprint(f"CSP selection cz=[{cz_lim[0]:.0f}, "
                           f"{cz_lim[1]:.0f}] km/s")
                fprint(f"CSP selection: use_cz_selection={use_cz_selection}, "
                       f"total evaluations: {n_eval}")
                # Store median measurement errors for selection
                self.CSP_sigma_m = csp_errors["sigma_m"]
                self.CSP_sigma_s = csp_errors["sigma_s"]
                self.CSP_sigma_BV = csp_errors["sigma_BV"]
                self.CSP_rho_ms = csp_errors["rho_ms"]
                self.CSP_rho_mBV = csp_errors["rho_mBV"]
                self.CSP_rho_sBV = csp_errors["rho_sBV"]
            else:
                ctx = CCHPTRGBSelectionContext(
                    f_rand_los_delta=getattr(self, "f_rand_los_delta", None),
                    f_rand_los_velocity=getattr(
                        self, "f_rand_los_velocity", None),
                    rhat_rand_los=self.rhat_rand_los,
                    r_host_range=self.r_host_range,
                    log_prior_r_grid=self._log_prior_r_grid,
                    distance2redshift=self.distance2redshift,
                    distance2distmod=self.distance2distmod,
                    cz_cmb=self.cz_cmb,
                    m_Bprime=self.m_Bprime,
                    sigma_cz_sel=self.sigma_cz_sel,
                    sigma_SN_sel=self.sigma_SN_sel,
                    use_reconstruction=self.use_reconstruction,
                    which_bias=self.which_bias,
                    quadratic_bias_delta0=self.quadratic_bias_delta0,
                )
                self.selection = CCHPTRGBSelectionComputation(ctx)

    def log_prior_distance(self, r):
        """Unnormalized uniform-in-volume distance prior: p(r) ~ r^2."""
        return 2.0 * jnp.log(r)

    def replace_priors(self, config):
        """Replace priors on parameters that are not used in the model."""
        if not get_nested(config, "model/use_reconstruction", False):
            replace_prior_with_delta(config, "beta", 0.0, verbose=False)
            replace_prior_with_delta(config, "b1", 1.0, verbose=False)
            replace_prior_with_delta(config, "b2", 0.0, verbose=False)
            replace_prior_with_delta(config, "alpha", 1.0, verbose=False)
            replace_prior_with_delta(config, "delta_b1", 0.0, verbose=False)
            replace_prior_with_delta(config, "alpha_low", 1.0, verbose=False)
            replace_prior_with_delta(config, "alpha_high", 1.0, verbose=False)
            replace_prior_with_delta(config, "log_rho_t", 0.0, verbose=False)
        which_sel = get_nested(config, "model/which_selection", None)
        if which_sel not in ("SN_magnitude", "CSP"):
            replace_prior_with_delta(config, "M_B", -18.5, verbose=False)
        if which_sel != "CSP":
            replace_prior_with_delta(config, "alpha_tripp", 0.7, verbose=False)
            replace_prior_with_delta(config, "beta_tripp", 2.5, verbose=False)
        which_bias = get_nested(config, "model/which_bias", "linear")
        if which_bias not in ["linear", "quadratic"]:
            replace_prior_with_delta(config, "b1", 1.0, verbose=False)
        if which_bias != "quadratic":
            replace_prior_with_delta(config, "b2", 0.0, verbose=False)
        return config

    def set_data(self, data):
        """Convert data to JAX arrays and set as attributes."""
        mag_obs = jnp.asarray(data["mag_TRGB"])
        e_mag_TRGB = jnp.asarray(data["e_mag_TRGB"])
        cz_cmb = jnp.asarray(data["cz_cmb"])
        e_czcmb = jnp.asarray(data["e_czcmb"])
        m_Bprime = jnp.asarray(data["m_Bprime"])
        e_m_Bprime = jnp.asarray(data["sigma_Bprime"])
        RA = jnp.asarray(data["RA"])
        DEC = jnp.asarray(data["DEC"])
        galaxies = np.asarray(data["Galaxy"])

        if self.which_selection == "SN_magnitude":
            sn_mask = jnp.isfinite(m_Bprime) & jnp.isfinite(e_m_Bprime)
            n_masked = int(jnp.sum(~sn_mask))
            fprint(f"SN magnitude selection: masking {n_masked} hosts "
                   "without finite SN photometry.")
            mag_obs = mag_obs[sn_mask]
            e_mag_TRGB = e_mag_TRGB[sn_mask]
            cz_cmb = cz_cmb[sn_mask]
            e_czcmb = e_czcmb[sn_mask]
            m_Bprime = m_Bprime[sn_mask]
            e_m_Bprime = e_m_Bprime[sn_mask]
            RA = RA[sn_mask]
            DEC = DEC[sn_mask]
            galaxies = galaxies[np.array(sn_mask)]
            # Also mask LOS data if provided.
            if ("host_los_density" in data
                    and data["host_los_density"] is not None):
                data["host_los_density"] = data["host_los_density"][
                    :, sn_mask, :]  # noqa
            if ("host_los_velocity" in data
                    and data["host_los_velocity"] is not None):
                data["host_los_velocity"] = data["host_los_velocity"][
                    :, sn_mask, :]  # noqa
            if "host_los_r" in data and data["host_los_r"] is not None:
                data["host_los_r"] = data["host_los_r"]

        # CSP selection: require matched CSP data
        self.CSP_obs_vec = None
        self.CSP_cov = None
        if self.which_selection == "CSP":
            if "CSP_obs_vec" not in data:
                raise ValueError(
                    "CSP selection requires load_CSP_matches=true in config.")
            # Mask to hosts with CSP matches (finite obs_vec)
            csp_mask = jnp.isfinite(data["CSP_obs_vec"][:, 0])
            n_masked = int(jnp.sum(~csp_mask))
            fprint(f"CSP selection: masking {n_masked} hosts without "
                   "CSP matches.")
            mag_obs = mag_obs[csp_mask]
            e_mag_TRGB = e_mag_TRGB[csp_mask]
            cz_cmb = cz_cmb[csp_mask]
            e_czcmb = e_czcmb[csp_mask]
            m_Bprime = m_Bprime[csp_mask]
            e_m_Bprime = e_m_Bprime[csp_mask]
            RA = RA[csp_mask]
            DEC = DEC[csp_mask]
            galaxies = galaxies[np.array(csp_mask)]
            # Store CSP data
            self.CSP_obs_vec = jnp.asarray(data["CSP_obs_vec"][csp_mask])
            self.CSP_cov = jnp.asarray(data["CSP_cov"][csp_mask])
            # Also mask LOS data if provided.
            if ("host_los_density" in data
                    and data["host_los_density"] is not None):
                data["host_los_density"] = data["host_los_density"][
                    :, csp_mask, :]
            if ("host_los_velocity" in data
                    and data["host_los_velocity"] is not None):
                data["host_los_velocity"] = data["host_los_velocity"][
                    :, csp_mask, :]

        self.mag_obs = mag_obs
        self.e_mag_TRGB = e_mag_TRGB
        self.cz_cmb = cz_cmb
        self.e_czcmb = e_czcmb
        self.m_Bprime = m_Bprime
        self.e_m_Bprime = e_m_Bprime
        self.RA = RA
        self.DEC = DEC
        # Group indices for repeated galaxies (multiple SNe per host).
        galaxies_unique, inverse = np.unique(galaxies, return_inverse=True)
        num_groups = galaxies_unique.shape[0]
        fprint(f"{len(galaxies)} SNe in {num_groups} unique hosts.")
        group_mask = np.zeros((num_groups, len(galaxies)), dtype=bool)
        for i in range(num_groups):
            group_mask[i, inverse == i] = True
        self.group_mask = jnp.asarray(group_mask)
        self.num_groups = num_groups
        self.group_index = jnp.asarray(inverse)
        self.group_counts = jnp.sum(self.group_mask, axis=1)
        self.sigma_SN_sel = jnp.median(self.e_m_Bprime)
        self.sigma_cz_sel = jnp.median(self.e_czcmb)

        # Host-level arrays (one value per unique host, not per SN).
        # TRGB magnitude and cz are properties of the host, not the SN.
        first_idx = np.array([np.where(inverse == i)[0][0]
                              for i in range(num_groups)])
        self.first_idx = jnp.asarray(first_idx)
        self.mag_obs_host = jnp.asarray(mag_obs[first_idx])
        self.e_mag_TRGB_host = jnp.asarray(e_mag_TRGB[first_idx])
        self.cz_cmb_host = jnp.asarray(cz_cmb[first_idx])
        self.e_czcmb_host = jnp.asarray(e_czcmb[first_idx])

        # Convert RA/Dec to Cartesian coordinates (per SN entry)
        rhat = radec_to_cartesian(self.RA, self.DEC)
        n = jnp.linalg.norm(rhat, axis=1, keepdims=True)
        self.rhat = rhat / jnp.where(n == 0.0, 1.0, n)
        # Host-level rhat (one per unique host)
        self.rhat_host = self.rhat[first_idx]

        self.mu_LMC_anchor = data["mu_LMC_anchor"]
        self.e_mu_LMC_anchor = data["e_mu_LMC_anchor"]
        self.mag_LMC_TRGB = data["mag_LMC_TRGB"]
        self.e_mag_LMC_TRGB = data["e_mag_LMC_TRGB"]
        self.mu_N4258_anchor = data["mu_N4258_anchor"]
        self.e_mu_N4258_anchor = data["e_mu_N4258_anchor"]
        self.mag_N4258_TRGB = data["mag_N4258_TRGB"]
        self.e_mag_N4258_TRGB = data["e_mag_N4258_TRGB"]

    def sample_host_distmod(self):
        """
        Sample distance moduli for host galaxies with a uniform prior in
        distance modulus. The r^2 volume prior is added via factor.
        """
        dist = Uniform(*self.distmod_limits)

        with plate("hosts", self.num_groups):
            mu_host = sample("mu_host", dist)

        # Anchors (LMC, NGC 4258) with custom limits if provided
        mu_LMC = sample("mu_LMC", Uniform(*self.distmod_limits_LMC))
        mu_N4258 = sample("mu_N4258", Uniform(*self.distmod_limits_N4258))

        # Anchor likelihoods
        sample("mu_LMC_ll",
               Normal(mu_LMC, self.e_mu_LMC_anchor),
               obs=self.mu_LMC_anchor)
        sample("mu_N4258_ll",
               Normal(mu_N4258, self.e_mu_N4258_anchor),
               obs=self.mu_N4258_anchor)

        return mu_host, mu_LMC, mu_N4258


class CCHPTRGBModel(BaseCCHPModel):
    """
    Forward model the TRGB magnitudes and host redshifts to infer the Hubble
    constant.

    Data structure:
    - Each row is one SN, multiple SNe can be in the same host galaxy
    - TRGB magnitude and cz are host properties (counted once per host)
    - SN observables (m_Bprime, CSP m/s/BV) are independent per SN

    Likelihood structure:
    - Host-level: volume prior, TRGB magnitude, cz (one per unique host)
    - SN-level: SN magnitude or CSP likelihood (summed over all SNe)
    - Selection correction: -N_host * log p(S=1|Lambda)
    """

    def __init__(self, config_path, data):
        super().__init__(config_path, data)

    def _log_prior_with_reconstruction(self, r, mu_host, h, bias_params):
        """Compute log prior on distance with reconstruction normalization."""
        lp_host_dist = self.log_prior_distance(r)
        lp_host_dist += self.log_grad_distmod2comoving_distance(mu_host, h=h)

        if self.use_reconstruction:
            rh_host = r * h
            los_delta_host = self.f_host_los_delta(rh_host)
            # lp_host_dist broadcasts to (nfields, ngal)
            lp_host_dist += lp_galaxy_bias(
                los_delta_host, jnp.log1p(los_delta_host),
                bias_params, self.which_bias,
                self.quadratic_bias_delta0)

            lp_grid = self._log_prior_r_grid[None, None, :]
            los_delta_grid = self.f_host_los_delta.interp_many_steps_per_galaxy(  # noqa
                self.r_host_range * h)
            lp_grid += lp_galaxy_bias(
                los_delta_grid, jnp.log1p(los_delta_grid),
                bias_params, self.which_bias,
                self.quadratic_bias_delta0)

            lp_grid_norm = ln_simpson(
                lp_grid, x=self.r_host_range[None, None, :], axis=-1)
            # Shape is (nfields, ngal)
            return lp_host_dist - lp_grid_norm
        else:
            # Shape is (1, ngal)
            return lp_host_dist[None, :]

    def _sample_selection_params(self):
        """Sample selection parameters if inferring selection."""
        cz_lim, cz_width, mag_lim, mag_width = None, None, None, None
        if self.infer_sel:
            if self.which_selection == "redshift":
                cz_lim = sample(
                    "cz_lim_selection",
                    Uniform(self.cz_lim_selection_min,
                            self.cz_lim_selection_max))
                cz_width = sample(
                    "cz_lim_selection_width",
                    Uniform(self.cz_lim_selection_width_min,
                            self.cz_lim_selection_width_max))
            elif self.which_selection == "SN_magnitude":
                mag_lim = sample(
                    "mag_lim_SN",
                    Uniform(self.mag_lim_SN_min, self.mag_lim_SN_max))
                mag_width = sample(
                    "mag_lim_SN_width",
                    Uniform(self.mag_lim_SN_width_min,
                            self.mag_lim_SN_width_max))
        return cz_lim, cz_width, mag_lim, mag_width

    def __call__(self, shared_params=None):
        H0 = rsample("H0", self.priors["H0"], shared_params)
        M_TRGB = rsample("M_TRGB", self.priors["M_TRGB"], shared_params)
        sigma_int = rsample("sigma_int", self.priors["sigma_int"],
                            shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)
        beta = rsample("beta", self.priors["beta"], shared_params)
        Vext = rsample("Vext", self.priors["Vext"], shared_params)
        M_B = rsample("M_B", self.priors["M_B"], shared_params)

        # Sample selection parameters only for redshift/SN_magnitude selection
        cz_lim_selection, cz_lim_selection_width = None, None
        mag_lim_SN, mag_lim_SN_width = None, None
        if self.which_selection in ("redshift", "SN_magnitude"):
            (cz_lim_selection, cz_lim_selection_width, mag_lim_SN,
             mag_lim_SN_width) = self._sample_selection_params()

        h = H0 / 100.0
        # Sample distance moduli per unique host galaxy
        mu_host, mu_LMC, mu_N4258 = self.sample_host_distmod()
        bias_params = sample_galaxy_bias(
            self.priors, self.which_bias, beta=beta, Om=self.Om)

        # Convert to comoving distances in Mpc (per unique host)
        r_host = self.distmod2distance(mu_host, h=h)
        z_cosmo_host = self.distance2redshift(r_host, h=h)

        # Broadcast to per-SN for interpolator compatibility
        mu_per_sn = mu_host[self.group_index]
        r_per_sn = r_host[self.group_index]

        # =====================================================================
        # Host-level likelihood (one per unique host galaxy)
        # =====================================================================
        # Volume prior with reconstruction normalization. LOS interpolator is
        # built per-SN, so we compute per-SN then extract.
        logp_prior_per_sn = self._log_prior_with_reconstruction(
            r_per_sn, mu_per_sn, h, bias_params)
        # Extract per-host (shape: nfields, num_groups)
        logp_host = logp_prior_per_sn[:, self.first_idx]

        var_tot_mag = self.e_mag_TRGB_host**2 + sigma_int**2
        var_tot_cz = self.e_czcmb_host**2 + sigma_v**2

        # TRGB magnitude likelihood (once per host), shape: (1, num_groups)
        logp_host += normal_logpdf_var(
            self.mag_obs_host, mu_host + M_TRGB, var_tot_mag)[None, :]

        # Project Vext along LOS (per host)
        Vext_rad_host = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)

        # Predicted redshifts per host, shape: (n_field, num_groups)
        if self.use_reconstruction:
            # Vpec interpolator is built per-SN, so compute per-SN then extract
            Vpec_per_sn = self.f_host_los_velocity(r_per_sn)
            Vpec_host = Vpec_per_sn[:, self.first_idx]
            Vpec_tot = Vext_rad_host[None, :] + beta * Vpec_host
            cz_th_host = predict_cz(z_cosmo_host[None, :], Vpec_tot)
        else:
            cz_th_host = predict_cz(
                z_cosmo_host[None, :], Vext_rad_host[None, :])

        # cz likelihood (once per host), shape: (n_field, num_groups)
        if self.use_cz_likelihood:
            ll_cz = normal_logpdf_var(self.cz_cmb_host, cz_th_host, var_tot_cz)
            logp_host += ll_cz

        # =====================================================================
        # SN-level likelihood (independent per SN, log-likelihoods summed)
        # =====================================================================
        logp_sn = jnp.zeros((1, len(self.group_index)))

        # Add the SN magnitude likelihood if using SN magnitude selection
        if self.which_selection == "SN_magnitude":
            logp_sn += normal_logpdf_var(
                self.m_Bprime, mu_per_sn + M_B,
                self.e_m_Bprime**2)[None, :]

        # Selection modelling
        if self.which_selection in (None, "none"):
            pass
        elif self.which_selection == "redshift":
            cz_lim = (cz_lim_selection if cz_lim_selection is not None
                      else self.cz_lim_selection)
            cz_width = (cz_lim_selection_width
                        if cz_lim_selection_width is not None
                        else self.cz_lim_selection_width)
            log_S, log_p_sel_i = self.selection.correction_redshift(
                h, sigma_v, beta, bias_params, Vext, H0, cz_lim, cz_width)
            # Selection correction: -N_host * log p(S=1|Lambda)
            # Subtract per host; summing over hosts gives -N * log_S
            logp_host -= log_S[:, None]
            # Per-source selection (sum over SNe)
            logp_sn += log_p_sel_i[None, :]
        elif self.which_selection == "SN_magnitude":
            m_lim = mag_lim_SN if mag_lim_SN is not None else self.mag_lim_SN
            m_width = (mag_lim_SN_width if mag_lim_SN_width is not None
                       else self.mag_lim_SN_width)
            log_S, log_p_sel_i = self.selection.correction_SN_mag(
                h, bias_params, M_B, H0, m_lim, m_width)
            # Selection correction: -N_host * log p(S=1|Lambda)
            # Subtract per host; summing over hosts gives -N * log_S
            logp_host -= log_S[:, None]
            # Per-source selection (sum over SNe)
            logp_sn += log_p_sel_i[None, :]
        elif self.which_selection == "CSP":
            # Sample Tripp standardization parameters
            alpha_tripp = rsample(
                "alpha_tripp", self.priors["alpha_tripp"], shared_params)
            beta_tripp = rsample(
                "beta_tripp", self.priors["beta_tripp"], shared_params)
            sigma_int_SN = rsample(
                "sigma_int_SN", self.priors["sigma_int_SN"], shared_params)
            # Sample CSP population parameters (may come from shared_params)
            mu_s = rsample("mu_s", self.priors["mu_s"], shared_params)
            sigma_s = rsample(
                "sigma_s", self.priors["sigma_s"], shared_params)
            mu_BV = rsample("mu_BV", self.priors["mu_BV"], shared_params)
            sigma_BV = rsample(
                "sigma_BV", self.priors["sigma_BV"], shared_params)
            rho_pop = rsample(
                "rho_pop", self.priors["rho_pop"], shared_params)
            # Sample selection parameters
            m_lim = rsample("m_lim", self.priors["m_lim"], shared_params)
            alpha_sel = rsample(
                "alpha_sel", self.priors["alpha_sel"], shared_params)
            beta_sel = rsample(
                "beta_sel", self.priors["beta_sel"], shared_params)
            sigma_sel = rsample(
                "sigma_sel", self.priors["sigma_sel"], shared_params)

            # CSP magnitude likelihood with full covariance
            # Sample true (s, BV) from population hyperprior for each SN
            pop_mean = jnp.array([mu_s, mu_BV])
            pop_cov = jnp.array([
                [sigma_s**2, rho_pop * sigma_s * sigma_BV],
                [rho_pop * sigma_s * sigma_BV, sigma_BV**2]])
            n_csp = self.CSP_obs_vec.shape[0]
            pop_dist = MultivariateNormal(pop_mean, pop_cov)
            with plate("csp_latent", n_csp):
                x_latent = sample("x_latent", pop_dist)
            s_true = x_latent[:, 0]
            BV_true = x_latent[:, 1]

            # True magnitude: m_true = M_B + mu - alpha*(s-1) + beta*BV
            m_true = (M_B + mu_per_sn - alpha_tripp * (s_true - 1)
                      + beta_tripp * BV_true)

            # Log-likelihood of observed (m, s, BV) | true using full cov
            # Add intrinsic scatter to magnitude variance
            cov_obs = self.CSP_cov.at[:, 0, 0].add(sigma_int_SN**2)
            mean_vec = jnp.stack([m_true, s_true, BV_true], axis=-1)
            L_cov = jnp.linalg.cholesky(cov_obs)
            log_det_cov = jnp.sum(
                jnp.log(jnp.diagonal(L_cov, axis1=-2, axis2=-1)), axis=-1)
            diff = self.CSP_obs_vec - mean_vec
            z = jnp.linalg.solve(L_cov, diff[..., None])[..., 0]
            ll_obs_mvn = -0.5 * (3 * jnp.log(2 * jnp.pi) + 2 * log_det_cov
                                 + jnp.sum(z**2, axis=-1))
            # SN likelihood (per SN)
            logp_sn += ll_obs_mvn[None, :]

            # Global selection probability
            # Total scatter on observed magnitude = intrinsic + measurement
            sigma_m_total = jnp.sqrt(sigma_int_SN**2 + self.CSP_sigma_m**2)
            log_p_sel = self.csp_selection(
                M_B, alpha_tripp, beta_tripp, sigma_m_total,
                mu_s, sigma_s, mu_BV, sigma_BV, rho_pop,
                m_lim, alpha_sel, beta_sel, sigma_sel,
                self.CSP_sigma_s, self.CSP_sigma_BV,
                self.CSP_rho_ms, self.CSP_rho_mBV, self.CSP_rho_sBV,
                sigma_v, h=h)
            # Selection correction: -N_host * log p(S=1|Lambda)
            # Subtract per host; summing over hosts gives -N * log_p_sel
            logp_host -= log_p_sel

            # Per-source selection (sum over SNe)
            log_p_sel_i = compute_per_source_selection(
                self.CSP_obs_vec, m_lim, alpha_sel, beta_sel, sigma_sel)
            logp_sn += log_p_sel_i[None, :]
        else:
            raise ValueError(
                f"Unknown selection '{self.which_selection}'. "
                "Use 'redshift', 'SN_magnitude', 'CSP', or 'none'.")

        # =====================================================================
        # Combine host and SN likelihoods
        # =====================================================================
        # Average host likelihood over field realisations, sum over hosts
        logp_host = logmeanexp(logp_host, axis=0)
        logp_host = jnp.sum(logp_host)

        # Sum SN likelihoods (product over independent SNe)
        # No field averaging needed for SN likelihood (no reconstruction dep)
        logp_sn = jnp.sum(logp_sn)

        logp_tot = logp_host + logp_sn

        # Anchor TRGB magnitudes
        logp_tot += normal_logpdf_var(
            self.mag_LMC_TRGB, M_TRGB + mu_LMC, self.e_mag_LMC_TRGB**2)
        logp_tot += normal_logpdf_var(
            self.mag_N4258_TRGB, M_TRGB + mu_N4258, self.e_mag_N4258_TRGB**2)

        factor("ll_total", logp_tot)


###############################################################################
#                    Joint TRGB + CSP Model for H0 inference                  #
###############################################################################


class JointTRGBCSPModel:
    """
    Joint model combining TRGB calibrators and CSP SNe for H0 inference.

    Shared parameters between models:
    - Cosmology: H0, Vext, sigma_v, beta
    - SN standardization: M_B, alpha_tripp, beta_tripp
    - Population: mu_s, sigma_s, mu_BV, sigma_BV, rho_pop
    - Selection: m_lim, alpha_sel, beta_sel, sigma_sel

    For overlapping hosts (SNe in both TRGB and CSP samples), distances
    are sampled once by the TRGB model and shared with the CSP likelihood.
    """

    shared_param_names = [
        # Cosmology
        "H0", "Vext", "sigma_v", "beta",
        # SN standardization
        "M_B", "alpha_tripp", "beta_tripp",  # "sigma_int_SN",
        # Population hyperparameters
        "mu_s", "sigma_s", "mu_BV", "sigma_BV", "rho_pop",
    ]

    def __init__(self, config_path, trgb_data, csp_data):
        """
        Initialize the joint model.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        trgb_data : object
            TRGB data container (from load_CCHP_from_config with CSP matches).
        csp_data : dict
            CSP data dict (from load_CSP_from_config). Must be disjoint from
            TRGB hosts.
        """
        # Check for overlap between TRGB and CSP samples (should be disjoint)
        trgb_sn_names = set(trgb_data["SN"])
        # CSP names have "SN" prefix, strip for comparison
        csp_sn_names = set(
            n[2:] if n.startswith("SN") else n for n in csp_data["sn"])
        overlap = trgb_sn_names & csp_sn_names
        if overlap:
            raise ValueError(
                f"CSP sample overlaps with TRGB hosts: {overlap}. "
                "Use a CSP sample that excludes TRGB hosts.")

        n_trgb_sne = len(trgb_data["SN"])
        n_trgb_hosts = len(np.unique(trgb_data["Galaxy"]))
        n_csp = len(csp_data["sn"])
        fprint(f"Joint model: {n_trgb_sne} TRGB SNe in {n_trgb_hosts} hosts, "
               f"{n_csp} CSP SNe (disjoint)")

        # Wrap CSP data in PVDataFrame (drops strings, converts to JAX)
        from ...pvdata import PVDataFrame
        csp_data = PVDataFrame(csp_data)

        # Initialize submodels
        self.trgb_model = CCHPTRGBModel(config_path, trgb_data)
        self.csp_model = CSPModel(config_path)

        # Validate and store CSP data
        self.csp_model.validate_data(csp_data)
        self.csp_data = csp_data

        # Store config
        self.config = self.trgb_model.config

        # Disable evidence computation
        fprint("setting `compute_evidence` to False.")
        self.config["inference"]["compute_evidence"] = False

    def _sample_shared_params(self):
        """Sample parameters shared between TRGB and CSP models."""
        shared = {}
        # Use TRGB model priors for shared params
        priors = self.trgb_model.priors
        for name in self.shared_param_names:
            if name in priors:
                shared[name] = rsample(name, priors[name])
        return shared

    def __call__(self):
        """NumPyro model for joint MCMC inference."""
        # Sample shared parameters once
        shared_params = self._sample_shared_params()
        H0 = shared_params["H0"]
        h = H0 / 100.0

        # Run TRGB model (computes likelihood for anchored SNe)
        with handlers.scope(prefix="TRGB"):
            self.trgb_model(shared_params=shared_params)

        # Run CSP model (all SNe, disjoint from TRGB)
        with handlers.scope(prefix="CSP"):
            self.csp_model(self.csp_data, shared_params=shared_params, h=h)
