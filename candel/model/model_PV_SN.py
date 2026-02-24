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
"""Supernova forward model."""
import jax.numpy as jnp
from numpyro import plate, sample
from numpyro.distributions import Normal, Uniform

from .base_pv import BasePVModel
from .pv_utils import marginalise_2d_latent, rsample


class SNModel(BasePVModel):
    """
    A SNe forward model: the distance is numerically marginalized at each MCMC
    step. The Tripp coefficients are sampled. The true (x1, c) are
    analytically marginalised out using the Gaussian conjugacy of the
    Tripp relation.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if self.track_log_density_per_sample:
            raise NotImplementedError(
                "`track_log_density_per_sample` is not implemented "
                "for `SNModel`.")

        if self.which_distance_prior != "empirical":
            raise ValueError(
                f"SNModel only supports empirical distance prior, got "
                f"'{self.which_distance_prior}'.")

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for `SNModel`.")

        # --- Tripp (SALT2) parameters ---
        M_SN = rsample("SN_absmag", self.priors["SN_absmag"], shared_params)
        alpha_SN = rsample("SN_alpha", self.priors["SN_alpha"], shared_params)
        beta_SN = rsample("SN_beta", self.priors["SN_beta"], shared_params)
        dM = rsample(
            "zeropoint_dipole", self.priors["zeropoint_dipole"], shared_params)
        M_SN = M_SN + jnp.sum(dM * data["rhat"], axis=1)
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)

        kwargs_dist, h, Vext, sigma_v, beta, bias_params = \
            self._sample_common_params(shared_params)

        # Hyperpriors for (x1, c)
        x1_prior_mean = sample(
            "x1_prior_mean", Uniform(data["min_x1"], data["max_x1"]))
        x1_prior_std = sample(
            "x1_prior_std",
            Uniform(0.0, data["max_x1"] - data["min_x1"]))
        c_prior_mean = sample(
            "c_prior_mean", Uniform(data["min_c"], data["max_c"]))
        c_prior_std = sample(
            "c_prior_std", Uniform(0.0, data["max_c"] - data["min_c"]))
        rho = sample("rho_corr", Uniform(-1, 1))

        with plate("data", nsamples):
            r_grid = data["r_grid"] / h

            lp_dist, Vrad = self._setup_lp_dist_and_Vrad(
                data, r_grid, kwargs_dist, beta, bias_params)

            ll_cz = self._compute_ll_cz(data, r_grid, h, Vext, sigma_v, Vrad)

            # Marginalise (x1, c) analytically; f = (-α, β) for Tripp
            f_dot_mu1, fSf, log_ev_obs = marginalise_2d_latent(
                x1_prior_std, c_prior_std, rho,
                x1_prior_mean, c_prior_mean,
                data["e2_x1"], data["e2_c"], data["x1"], data["c"],
                -alpha_SN, beta_SN)

            sigma_eff = jnp.sqrt(fSf + sigma_int**2 + data["e2_mag"])

            # Magnitude predictive, (n_gal, n_r)
            mu_r = self.distance2distmod(r_grid, h=h)[None, :]
            m_pred = mu_r + (M_SN + f_dot_mu1)[:, None]
            ll_mag = Normal(
                m_pred, sigma_eff[:, None]).log_prob(
                    data["mag"][:, None])

            ll = ll_mag[None, ...] + lp_dist + ll_cz
            ll = self._marginalize_over_r(ll, r_grid)
            ll += log_ev_obs

            self._average_fields_and_factor(ll, data)
