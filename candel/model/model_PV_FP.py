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
"""Fundamental Plane (FP) forward model."""
import jax.numpy as jnp
from numpyro import plate, sample
from numpyro.distributions import Normal, Uniform

from ..cosmography import Distance2LogAngDist
from .base_pv import BasePVModel
from .pv_utils import marginalise_2d_latent, rsample


class FPModel(BasePVModel):
    """
    A FP model where the distance modulus μ is integrated out using a grid,
    instead of being sampled as a latent variable. The true (log σ, log I)
    are analytically marginalised out using the Gaussian conjugacy of the
    FP relation.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if self.track_log_density_per_sample:
            raise NotImplementedError(
                "`track_log_density_per_sample` is not implemented "
                "for `FPModel`.")

        self.distance2logangdist = Distance2LogAngDist(Om0=self.Om)

        if self.which_distance_prior != "empirical":
            raise ValueError(
                f"FPModel only supports empirical distance prior, got "
                f"'{self.which_distance_prior}'.")

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for `FPModel`.")

        # Sample the FP parameters.
        a_FP = rsample("a_FP", self.priors["FP_a"], shared_params)
        b_FP = rsample("b_FP", self.priors["FP_b"], shared_params)
        c_FP = rsample("c_FP", self.priors["FP_c"], shared_params)
        sigma_log_theta = rsample(
            "sigma_log_theta", self.priors["sigma_log_theta"], shared_params)

        kwargs_dist, h, Vext, sigma_v, beta, bias_params = \
            self._sample_common_params(shared_params)

        logs_prior_mean = sample(
            "logs_prior_mean", Uniform(data["min_logs"], data["max_logs"]))
        logs_prior_std = sample(
            "logs_prior_std",
            Uniform(0, data["max_logs"] - data["min_logs"]))

        logI_prior_mean = sample(
            "logI_prior_mean", Uniform(data["min_logI"], data["max_logI"]))
        logI_prior_std = sample(
            "logI_prior_std",
            Uniform(0.0, data["max_logI"] - data["min_logI"]))
        rho = sample("rho_corr", Uniform(-1.0, 1.0))

        with plate("data", nsamples):
            r_grid = data["r_grid"] / h
            logda_grid = self.distance2logangdist(r_grid)

            lp_dist, Vrad = self._setup_lp_dist_and_Vrad(
                data, r_grid, kwargs_dist, beta, bias_params)

            ll_cz = self._compute_ll_cz(
                data, r_grid, h, Vext, sigma_v, Vrad)

            # Marginalise (log σ, log I) analytically; f = (a, b) for FP
            f_dot_mu1, fSf, log_ev_obs = marginalise_2d_latent(
                logs_prior_std, logI_prior_std, rho,
                logs_prior_mean, logI_prior_mean,
                data["e2_logs"], data["e2_logI"], data["logs"], data["logI"],
                a_FP, b_FP)

            sigma_eff = jnp.sqrt(
                fSf + sigma_log_theta**2 + data["e2_log_theta_eff"])

            # Predictive for θ, (n_gal, n_r)
            log_theta_pred = (
                (f_dot_mu1 + c_FP - 3)[:, None] - logda_grid[None, :])
            ll_theta = Normal(
                log_theta_pred, sigma_eff[:, None]).log_prob(
                    data["log_theta_eff"][:, None])

            ll = ll_theta[None, ...] + lp_dist + ll_cz
            ll = self._marginalize_over_r(ll, r_grid)
            ll += log_ev_obs

            self._average_fields_and_factor(ll, data)
