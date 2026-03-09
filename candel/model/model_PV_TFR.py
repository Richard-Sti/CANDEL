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
"""TFR (Tully-Fisher relation) forward model."""
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpyro import factor, plate, sample
from numpyro.distributions import Normal, Uniform

from ..util import fprint
from .base_pv import BasePVModel
from .pv_utils import (gauss_hermite_log_weights, get_absmag_TFR,
                       log_p_S_TFR_eta, rsample)


class TFRModel(BasePVModel):
    """
    A TFR forward model, distance is numerically marginalized out at each MCMC
    step instead of being sampled as a latent variable.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if not self.marginalize_eta:
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

        if self.which_distance_prior != "empirical":
            raise ValueError(
                f"TFRModel only supports empirical distance prior, got "
                f"'{self.which_distance_prior}'.")

        if self.marginalize_eta:
            n_gh = (self.eta_grid_kwargs["n_grid"]
                    if self.eta_grid_kwargs else 5)
            self._gh_nodes, self._gh_log_w = gauss_hermite_log_weights(n_gh)
            fprint(f"using Gauss-Hermite quadrature with {n_gh} nodes "
                   f"and Laplace centering for eta marginalization.")

    def __call__(self, data, shared_params=None):
        nsamples = len(data)
        # Initialize log density tracker; not to be tracked by NumPyro with
        # `factor`.
        if self.track_log_density_per_sample:
            log_density_per_sample = jnp.zeros(nsamples)

        # Sample the TFR parameters.
        a_TFR = rsample("a_TFR", self.priors["TFR_zeropoint"], shared_params)
        b_TFR = rsample("b_TFR", self.priors["TFR_slope"], shared_params)
        c_TFR = rsample("c_TFR", self.priors["TFR_curvature"], shared_params)
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)
        a_TFR_dipole = rsample(
            "zeropoint_dipole", self.priors["zeropoint_dipole"], shared_params)
        a_TFR = a_TFR + jnp.sum(a_TFR_dipole * data["rhat"], axis=1)
        kwargs_dist, h, Vext, sigma_v, beta, bias_params = \
            self._sample_common_params(shared_params)

        if data.sample_dust:
            Rdust = rsample("R_dust", self.priors["Rdust"], shared_params)
            Ab = Rdust * data["ebv"]
        else:
            Ab = 0.

        eta_prior_mean = sample(
            "eta_prior_mean", Uniform(data["min_eta"], data["max_eta"]))
        eta_prior_std = sample(
            "eta_prior_std", Uniform(0, data["max_eta"] - data["min_eta"]))

        with plate("data", nsamples):
            if self.marginalize_eta:
                var_h = eta_prior_std**2
                var_o = data["e_eta"]**2
                prec = 1.0 / var_h + 1.0 / var_o
                mu_c = (eta_prior_mean / var_h
                        + data["eta"] / var_o) / prec
                sigma_c = 1.0 / jnp.sqrt(prec)
                log_Z_eta = Normal(
                    eta_prior_mean,
                    jnp.sqrt(var_h + var_o)).log_prob(data["eta"])
            else:
                # Sample the galaxy linewidth from a Gaussian hyperprior.
                eta = sample(
                    "eta_latent", Normal(eta_prior_mean, eta_prior_std))
                sample("eta", Normal(eta, data["e_eta"]), obs=data["eta"])

                if self.track_log_density_per_sample:
                    log_density_per_sample += Normal(
                        eta_prior_mean, eta_prior_std).log_prob(eta)
                    log_density_per_sample += Normal(
                        eta, data["e_eta"]).log_prob(data["eta"])

            if data.add_eta_truncation:
                neglog_pS = -log_p_S_TFR_eta(
                    eta_prior_mean, eta_prior_std, data["e_eta"],
                    data.eta_min, data.eta_max)

                factor("neg_log_S_eta", neglog_pS)
                if self.track_log_density_per_sample:
                    log_density_per_sample += neglog_pS

            e_mag = jnp.sqrt(sigma_int**2 + data["e2_mag"])

            r_grid = data["r_grid"] / h

            lp_dist, Vrad = self._setup_lp_dist_and_Vrad(
                data, r_grid, kwargs_dist, beta, bias_params)

            ll_cz = self._compute_ll_cz(
                data, r_grid, h, Vext, sigma_v, Vrad)

            # Likelihood of the observed magnitudes.
            if self.marginalize_eta:
                mu_grid = self.distance2distmod(r_grid, h=h)

                M_c = get_absmag_TFR(mu_c, a_TFR, b_TFR, c_TFR)
                M_prime_c = b_TFR + jnp.where(
                    mu_c > 0, 2 * c_TFR * mu_c, 0.0)

                sigma_eff_sq = e_mag**2 + (M_prime_c * sigma_c)**2
                sigma_star = sigma_c * e_mag / jnp.sqrt(sigma_eff_sq)

                R = ((data["mag"] - Ab) - M_c)[:, None] - mu_grid[None, :]
                delta_mu = (R * M_prime_c[:, None] * sigma_c[:, None]**2
                            / sigma_eff_sq[:, None])

                # d_s2x = delta_mu + sqrt(2)*sigma_star*x_gh
                # eta_nodes = mu_c + d_s2x (avoid materializing both)
                sqrt2_sigma_star_x = (jnp.sqrt(2.0)
                                      * sigma_star[:, None, None]
                                      * self._gh_nodes[None, None, :])
                d_s2x = delta_mu[:, :, None] + sqrt2_sigma_star_x
                eta_nodes = mu_c[:, None, None] + d_s2x

                M_eta = get_absmag_TFR(
                    eta_nodes, a_TFR[:, None, None], b_TFR, c_TFR)

                ll_mag = Normal(
                    mu_grid[None, :, None] + M_eta,
                    e_mag[:, None, None]).log_prob(
                        (data["mag"] - Ab)[:, None, None])

                log_ratio = (
                    jnp.log(sigma_star / sigma_c)[:, None, None]
                    + self._gh_nodes**2
                    - 0.5 * d_s2x**2
                    / sigma_c[:, None, None]**2)

                ll_eta = logsumexp(
                    ll_mag + log_ratio + self._gh_log_w[None, None, :],
                    axis=-1)
                ll_eta += log_Z_eta[:, None]

                ll = (ll_cz + lp_dist) + ll_eta[None, ...]
                ll = self._marginalize_over_r(ll, r_grid, data)
            else:
                ll_mag = Normal(
                    self.distance2distmod(r_grid, h=h)[None, :] +
                    get_absmag_TFR(eta, a_TFR, b_TFR, c_TFR)[:, None],
                    e_mag[:, None]).log_prob(
                        (data["mag"] - Ab)[:, None])[None, ...]
                ll = ll_cz + ll_mag + lp_dist
                ll = self._marginalize_over_r(ll, r_grid, data)

            self._average_fields_and_factor(
                ll, data,
                log_density_per_sample
                if self.track_log_density_per_sample else None)
