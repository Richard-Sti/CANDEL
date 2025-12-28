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
"""
from abc import ABC
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr
from numpyro import factor, plate, sample
from numpyro.distributions import Normal, Uniform

from ..cosmography import (Distance2Distmod, Distance2Redshift,
                           Distmod2Distance, Distmod2Redshift,
                           LogGrad_Distmod2ComovingDistance)
from ..util import (fprint, get_nested, load_config, radec_to_cartesian,
                    replace_prior_with_delta)
from .interp import LOSInterpolator
from .model import (load_priors, lp_galaxy_bias, predict_cz, rsample,
                    sample_galaxy_bias)
from .model_SH0ES import log_prob_integrand_sel, logmeanexp
from .simpson import ln_simpson


def logmeanexp_by_group(logp, idx, n_groups=None):
    """
    Compute per-group log-mean-exp given per-item logp and integer group
    indices.
    """
    if n_groups is None:
        n_groups = int(jnp.max(idx)) + 1
    counts = jnp.bincount(idx, length=n_groups)

    # Per-group baselines for numerical stability
    baseline = jnp.full(n_groups, -jnp.inf)
    baseline = baseline.at[idx].max(logp)

    # Broadcast baseline to per-item
    baseline_per_item = baseline[idx]
    exp_shifted = jnp.exp(logp - baseline_per_item)

    sum_exp = jnp.zeros(n_groups, dtype=logp.dtype)
    sum_exp = sum_exp.at[idx].add(exp_shifted)

    logsum = jnp.log(sum_exp) + baseline
    return logsum - jnp.log(counts)


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

    def log_S_SN_mag(self, lp_r, M_SN, H0, mag_lim, mag_width):
        """Probability of detection term if supernova magnitude-truncated."""
        ctx = self.ctx
        mag = ctx.distance2distmod(ctx.r_host_range, h=H0 / 100) + M_SN
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
        Vext_rad_rand = jnp.sum(Vext[None, :] * ctx.rhat_rand_los, axis=1)

        if ctx.use_reconstruction:
            rand_los_delta_grid = \
                ctx.f_rand_los_delta.interp_many_steps_per_galaxy(
                    ctx.r_host_range * h)
            lp_r_grid += lp_galaxy_bias(
                rand_los_delta_grid, jnp.log1p(rand_los_delta_grid),
                bias_params, ctx.which_bias)

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

    def correction_SN_mag(self, h, bias_params, M_SN, H0, mag_lim, mag_width):
        """Compute SN magnitude selection correction using random LOS."""
        ctx = self.ctx
        lp_r_grid = ctx.log_prior_r_grid[None, None, :]

        if ctx.use_reconstruction:
            rand_los_delta_grid = \
                ctx.f_rand_los_delta.interp_many_steps_per_galaxy(
                    ctx.r_host_range * h)
            lp_r_grid += lp_galaxy_bias(
                rand_los_delta_grid, jnp.log1p(rand_los_delta_grid),
                bias_params, ctx.which_bias)

        # Marginalized selection over random LOS, shape (nfields, n_rand_los)
        log_S = self.log_S_SN_mag(lp_r_grid, M_SN, H0, mag_lim, mag_width)

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
        self.which_selection = get_nested(
            config, "model/which_selection", None)
        if self.use_reconstruction:
            recon_name = get_nested(config, "io/which_host_los", "unspecified")
            fprint(f"Using reconstruction: {recon_name}")
        fprint(f"which_bias set to {self.which_bias}")
        fprint(f"selection set to {self.which_selection}")
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
        if get_nested(config, "io/load_rand_los", False):
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
                rhat = radec_to_cartesian(
                    jnp.asarray(data["rand_los_RA"]),
                    jnp.asarray(data["rand_los_dec"]))
                n = jnp.linalg.norm(rhat, axis=1, keepdims=True)
                self.rhat_rand_los = rhat / jnp.where(n == 0.0, 1.0, n)

        # Set up radial range for volume prior normalization
        r_limits_malmquist = get_nested(
            config, "model/r_limits_malmquist", [0.01, 350])
        num_points_malmquist = get_nested(
            config, "model/num_points_malmquist", 251)
        r_range = jnp.linspace(
            r_limits_malmquist[0], r_limits_malmquist[1],
            num_points_malmquist)

        fprint(f"setting radial range from {r_limits_malmquist[0]} to "
               f"{r_limits_malmquist[1]} Mpc with {num_points_malmquist} "
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
                    "LOS data. Set `load_rand_los = true` in config.")

            # When not using reconstruction but applying selection, create
            # dummy homogeneous random LOS (like SH0ES)
            if not self.use_reconstruction:
                fprint("Creating dummy random LOS for selection (no recon).")
                self.num_rand_los = 1
                self.rhat_rand_los = jnp.zeros((1, 3))

        # Build selection context and computation if selection is enabled
        self.selection = None
        if self.which_selection not in (None, "none"):
            ctx = CCHPTRGBSelectionContext(
                f_rand_los_delta=getattr(self, "f_rand_los_delta", None),
                f_rand_los_velocity=getattr(self, "f_rand_los_velocity", None),
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
            )
            self.selection = CCHPTRGBSelectionComputation(ctx)

    def log_prior_distance(self, r):
        """Log prior on physical distance (volume prior, r^2)."""
        return 2.0 * jnp.log(r) - 3.0 * jnp.log(self.Rmax) + jnp.log(3.0)

    def replace_priors(self, config):
        """Replace priors on parameters that are not used in the model."""
        if not get_nested(config, "model/use_reconstruction", False):
            replace_prior_with_delta(config, "beta", 0.0, verbose=False)
            replace_prior_with_delta(config, "b1", 1.0, verbose=False)
            replace_prior_with_delta(config, "alpha", 1.0, verbose=False)
            replace_prior_with_delta(config, "delta_b1", 0.0, verbose=False)
            replace_prior_with_delta(config, "alpha_low", 1.0, verbose=False)
            replace_prior_with_delta(config, "alpha_high", 1.0, verbose=False)
            replace_prior_with_delta(config, "log_rho_t", 0.0, verbose=False)
        if get_nested(config, "model/which_selection", None) != "SN_magnitude":
            replace_prior_with_delta(config, "M_SN", -19.25, verbose=False)
        if get_nested(config, "model/which_bias", "linear") != "linear":
            replace_prior_with_delta(config, "b1", 1.0, verbose=False)
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

        self.mag_obs = mag_obs
        self.e_mag_TRGB = e_mag_TRGB
        self.cz_cmb = cz_cmb
        self.e_czcmb = e_czcmb
        self.m_Bprime = m_Bprime
        self.e_m_Bprime = e_m_Bprime
        self.RA = RA
        self.DEC = DEC
        # Group indices for repeated galaxies.
        galaxies_unique, inverse = np.unique(galaxies, return_inverse=True)
        num_groups = galaxies_unique.shape[0]
        fprint(f"{len(galaxies)} hosts, {num_groups} unique after grouping.")
        group_mask = np.zeros((num_groups, len(galaxies)), dtype=bool)
        for i in range(num_groups):
            group_mask[i, inverse == i] = True
        self.group_mask = jnp.asarray(group_mask)
        self.num_groups = num_groups
        self.group_index = jnp.asarray(inverse)
        self.group_counts = jnp.sum(self.group_mask, axis=1)
        self.sigma_SN_sel = jnp.median(self.e_m_Bprime)
        self.sigma_cz_sel = jnp.median(self.e_czcmb)

        # Convert RA/Dec to Cartesian coordinates
        fprint("Converting host RA/dec to Cartesian coordinates.")
        rhat = radec_to_cartesian(self.RA, self.DEC)
        n = jnp.linalg.norm(rhat, axis=1, keepdims=True)
        self.rhat_host = rhat / jnp.where(n == 0.0, 1.0, n)

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

    Duplicate objects per galaxy are handled by sampling one distance per
    unique host and averaging per-galaxy likelihoods within groups via
    log-mean-exp.
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
                bias_params, self.which_bias)

            lp_grid = self._log_prior_r_grid[None, None, :]
            los_delta_grid = self.f_host_los_delta.interp_many_steps_per_galaxy(  # noqa
                self.r_host_range * h)
            lp_grid += lp_galaxy_bias(
                los_delta_grid, jnp.log1p(los_delta_grid),
                bias_params, self.which_bias)

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
        M_SN = rsample("M_SN", self.priors["M_SN"], shared_params)

        (cz_lim_selection, cz_lim_selection_width, mag_lim_SN,
         mag_lim_SN_width) = self._sample_selection_params()

        h = H0 / 100.0
        # Sample distance moduli per each group
        mu_host_group, mu_LMC, mu_N4258 = self.sample_host_distmod()
        bias_params = sample_galaxy_bias(
            self.priors, self.which_bias, beta=beta, Om=self.Om)

        # Convert to comoving distances in Mpc (grouped hosts)
        r_group = self.distmod2distance(mu_host_group, h=h)
        z_cosmo_group = self.distance2redshift(r_group, h=h)

        # Propagate to per-galaxy distance moduli and distances.
        mu_host = mu_host_group[self.group_index]
        r = r_group[self.group_index]
        z_cosmo = z_cosmo_group[self.group_index]

        # Volume prior with reconstruction normalization
        logp_tot = self._log_prior_with_reconstruction(
            r, mu_host, h, bias_params)

        sigma_tot_mag = jnp.sqrt(self.e_mag_TRGB**2 + sigma_int**2)
        sigma_tot_cz = jnp.sqrt(self.e_czcmb**2 + sigma_v**2)

        # Add the magnitude likelihood, `(1, n_gal)`
        logp_tot += Normal(
            mu_host + M_TRGB, sigma_tot_mag).log_prob(self.mag_obs)[None, :]

        # Project Vext along LOS
        Vext_rad_host = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)

        # Get the predicted redshifts, (n_field, n_gal)
        if self.use_reconstruction:
            Vpec = self.f_host_los_velocity(r)
            cz_th_all = predict_cz(
                z_cosmo[None, :], Vext_rad_host[None, :] + beta * Vpec)
        else:
            cz_th_all = predict_cz(z_cosmo[None, :], Vext_rad_host[None, :])

        # Add the SN magnitude likelihood if using SN magnitude selection,
        # `(1, n_gal)`
        if self.which_selection == "SN_magnitude":
            logp_tot += Normal(
                mu_host + M_SN,
                self.e_m_Bprime).log_prob(self.m_Bprime)[None, :]

        # Selection modelling using random LOS.
        # log_S has shape (nfields,), log_p_sel_i has shape (ngal,)
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
            logp_tot -= log_S[:, None]
            logp_tot += log_p_sel_i[None, :]
        elif self.which_selection == "SN_magnitude":
            m_lim = mag_lim_SN if mag_lim_SN is not None else self.mag_lim_SN
            m_width = (mag_lim_SN_width if mag_lim_SN_width is not None
                       else self.mag_lim_SN_width)
            log_S, log_p_sel_i = self.selection.correction_SN_mag(
                h, bias_params, M_SN, H0, m_lim, m_width)
            logp_tot -= log_S[:, None]
            logp_tot += log_p_sel_i[None, :]
        else:
            raise ValueError(
                f"Unknown selection '{self.which_selection}'. "
                "Use 'redshift', 'SN_magnitude' or 'none'.")

        # Redshift likelihood, `(nfields, n_gal)`
        logp_tot += Normal(
            cz_th_all, sigma_tot_cz).log_prob(self.cz_cmb)

        # Average over field realisations
        logp_tot = logmeanexp(logp_tot, axis=0)

        # Average over multiple per-galaxy entries (same galaxy observed
        # multiple times)
        logp_tot = logmeanexp_by_group(
            logp_tot, self.group_index, n_groups=self.num_groups)

        # Sum over all groups
        logp_tot = jnp.sum(logp_tot)

        # Anchor TRGB magnitudes
        logp_tot += Normal(
            M_TRGB + mu_LMC, self.e_mag_LMC_TRGB).log_prob(
            self.mag_LMC_TRGB)
        logp_tot += Normal(
            M_TRGB + mu_N4258, self.e_mag_N4258_TRGB).log_prob(
            self.mag_N4258_TRGB)

        factor("ll_total", logp_tot)
