# Copyright (C) 2025 Richard Stiskalek
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
"""Pantheon+ forward model."""
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpyro import factor, plate, sample
from numpyro.distributions import MultivariateNormal, Uniform

from ..util import fprint
from .base_pv import BasePVModel
from .pv_utils import (add_sigma_mag_to_lane_cov, lp_galaxy_bias, rsample,
                       sample_distance_prior_volume, sample_galaxy_bias,
                       sigma_v_from_density)
from .utils import normal_logpdf_var, predict_cz


class PantheonPlusModel(BasePVModel):
    """
    Pantheon+ forward model, the distance is numerically marginalized out at
    each MCMC step instead of being sampled as a latent variable.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if self.which_Vext != "constant":
            raise NotImplementedError("Only constant Vext is implemented for "
                                      "the `PantheonPlusModel`.")

        if self.track_log_density_per_sample:
            raise NotImplementedError(
                "`track_log_density_per_sample` is not implemented "
                "for `PantheonPlusModel`.")

        if self.which_distance_prior != "empirical":
            raise ValueError(
                f"PantheonPlusModel only supports empirical distance prior, "
                f"got '{self.which_distance_prior}'.")

        fprint("setting `compute_evidence` to False.")
        self.compute_evidence = False

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for `PantheonPlusModel`.")

        # Sample the SN parameters.
        M = rsample("M", self.priors["SN_absmag"], shared_params)
        dM = rsample(
            "zeropoint_dipole", self.priors["zeropoint_dipole"], shared_params)
        M = M + jnp.sum(dM * data["rhat"], axis=1)
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)

        # For the Lane covariance we sample the Tripp params.
        if data.with_lane_covmat:
            alpha_SN = rsample(
                "SN_alpha", self.priors["SN_alpha"], shared_params)
            beta_SN = rsample("SN_beta", self.priors["SN_beta"], shared_params)
            x1 = data["x1"]
            c = data["c"]

        kwargs_dist = sample_distance_prior_volume(self.priors)

        # Sample velocity field parameters.
        Vext = rsample("Vext", self.priors["Vext"], shared_params)
        if self.density_dependent_sigma_v:
            sigma_v_low = rsample(
                "sigma_v_low", self.priors["sigma_v_low"], shared_params)
            sigma_v_high = rsample(
                "sigma_v_high", self.priors["sigma_v_high"], shared_params)
            log_sigma_v_rho_t = rsample(
                "log_sigma_v_rho_t", self.priors["log_sigma_v_rho_t"],
                shared_params)
            sigma_v_k = rsample(
                "sigma_v_k", self.priors["sigma_v_k"], shared_params)
        else:
            sigma_v = rsample(
                "sigma_v", self.priors["sigma_v"], shared_params)
        # Radially-project Vext
        Vext_rad = jnp.sum(data["rhat"] * Vext[None, :], axis=1)

        # Remaining parameters
        beta = rsample("beta", self.priors["beta"], shared_params)
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params,
            Om=self.Om, beta=beta)
        self._validate_volume_normalized_prior_data(data)

        # For the distance marginalization, h is not sampled. A grid is still
        # required to normalize the inhomogeneous Malmquist bias distribution.
        h = 1.
        r_grid = data["r_grid"] / h
        Rmax = r_grid[-1]

        # Sample the radial distance to each galaxy, `(n_galaxies)`.
        with plate("plate_distance", nsamples):
            r = sample("r_latent", Uniform(0, Rmax))

        mu = self.distance2distmod(r, h=h)  # (n_gal,)

        # Pointwise volume-normalized empirical prior along each source LOS:
        # log n(r,θ_s) + log f(r) + 2 log r - log N_field.
        r_safe = jnp.clip(r, a_min=1e-12)
        log_f = -jnp.exp(
            kwargs_dist["q"] * (jnp.log(r_safe) - jnp.log(kwargs_dist["R"])))
        lp_dist = (log_f + 2.0 * jnp.log(r_safe))[None, :]

        if data.with_lane_covmat:
            # Lane covariance is 3N x 3N where the values in the data vector
            # are `magnitude residual, 0, 0` repeated for all hosts.
            C = add_sigma_mag_to_lane_cov(sigma_int, data["mag_covmat"])

            # Compute the magnitude residuals.
            M_eff = (M - alpha_SN * x1 + beta_SN * c)         # (n_gal,)
            dx = data["mag"] - (mu + M_eff)

            # Compute the magnitude difference vector
            # [mag_res_i, 0, 0, mag_res_i + 1, 0, 0, etc...]
            dX = jnp.zeros((3 * dx.size,), dtype=dx.dtype)
            dX = dX.at[0::3].set(dx)
            # Finally, track the likelihood of the magnitudes
            sample(
                "mag_obs", MultivariateNormal(dX, C), obs=jnp.zeros_like(dX))
        else:
            # Track the likelihood of the predicted magnitudes, add any
            # intrinsic scatter to the covariance matrix.
            C = (data["mag_covmat"]
                 + jnp.eye(data["mag_covmat"].shape[0]) * sigma_int**2)
            sample("mag_obs", MultivariateNormal(mu + M, C), obs=data["mag"])

        # Evaluate the radial velocity and the galaxy bias at the sampled
        # distances, `(n_field, n_gal,)`, then apply the per-field 3D
        # normalizer shared by all LOS.
        Vrad = beta * data.f_los_velocity(r)
        lp_dist += lp_galaxy_bias(
            data.f_los_delta(r), data.f_los_log_density(r),
            bias_params, self.galaxy_bias,
            self.quadratic_bias_delta0)
        lp_dist -= self._compute_volume_log_N(
            data, kwargs_dist, bias_params)[:, None]

        with plate("plate_redshift", nsamples):
            # Predicted redshift, `(n_field, n_galaxies)`
            czpred = predict_cz(
                self.distance2redshift(r, h=h)[None, :],
                Vrad + Vext_rad[None, :])
            # Compute the redshift likelihood, and add the distance prior
            if self.density_dependent_sigma_v:
                delta_at_r = data.f_los_delta(r)
                sigma_v_r = sigma_v_from_density(
                    delta_at_r, sigma_v_low, sigma_v_high,
                    log_sigma_v_rho_t, sigma_v_k)
                ll = normal_logpdf_var(
                    data["czcmb"][None, :], czpred, sigma_v_r**2)
            else:
                ll = normal_logpdf_var(
                    data["czcmb"][None, :], czpred, sigma_v**2)
            ll += lp_dist
            # Average over field realizations and track
            factor("ll_obs", logsumexp(ll, axis=0) - jnp.log(data.num_fields))
