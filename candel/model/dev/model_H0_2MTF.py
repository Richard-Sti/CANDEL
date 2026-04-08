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
"""
2MTF K-band TFR forward model.

Forward-models observed redshifts and K-band apparent magnitudes for a
magnitude-limited TFR sample with linewidth cuts. Distances are numerically
marginalized, and the true linewidths are marginalized via Gauss-Hermite
quadrature.

The distance prior is r^2 (uniform in volume), optionally modulated by
inhomogeneous Malmquist bias from a reconstruction.

The selection function accounts for:
  - Apparent magnitude limit (hard or sigmoid)
  - Linewidth cuts on observed eta (using a representative eta error)
and depends only on model parameters, not on per-host observed data.
"""
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as norm_jax
from numpyro import factor, sample
from numpyro.distributions import Normal, Uniform

from ...util import fprint, get_nested
from ..base_model import H0ModelBase
from ..pv_utils import (gauss_hermite_log_weights, get_absmag_TFR,
                        lp_galaxy_bias, rsample, sample_galaxy_bias)
from ..integration import ln_simpson_precomputed
from ..utils import log_prob_integrand_sel, logmeanexp, predict_cz


class EDD2MTFModel(H0ModelBase):
    """2MTF K-band TFR forward model with magnitude and linewidth selection."""

    # ------------------------------------------------------------------
    #  Phase 1: model physics
    # ------------------------------------------------------------------

    # _replace_unused_priors inherited from H0ModelBase

    def _load_selection_thresholds(self):
        config = self.config
        self.mag_lim = get_nested(config, "model/mag_lim", 11.25)
        self.mag_lim_width = get_nested(config, "model/mag_lim_width", None)
        self.eta_min_sel = get_nested(config, "model/eta_min_sel", None)
        self.eta_max_sel = get_nested(config, "model/eta_max_sel", None)
        fprint(f"mag_lim={self.mag_lim}, mag_lim_width={self.mag_lim_width}")
        fprint(f"eta_min_sel={self.eta_min_sel}, "
               f"eta_max_sel={self.eta_max_sel}")

    def _load_model_flags(self):
        super()._load_model_flags()
        self.marginalize_eta = get_nested(
            self.config, "model/marginalize_eta", True)
        fprint(f"marginalize_eta={self.marginalize_eta}")
        n_gh = get_nested(self.config, "model/n_gauss_hermite", 5)
        self._gh_nodes, self._gh_log_w = gauss_hermite_log_weights(n_gh)
        fprint(f"Gauss-Hermite quadrature with {n_gh} nodes.")

        # Fixed eta grid for selection integration — covers a wider range
        # than the eta cuts so that P(eta_obs in range | eta_true) is
        # properly modeled near the boundaries.
        n_eta_sel = get_nested(self.config, "model/n_eta_sel_grid", 101)
        self._eta_sel_grid = jnp.linspace(-1.0, 1.0, n_eta_sel)
        fprint(f"eta selection grid: {n_eta_sel} points on [-1, 1].")

    # ------------------------------------------------------------------
    #  Phase 2: data loading
    # ------------------------------------------------------------------

    def _load_data(self, data):
        super()._load_data(data)
        self.num_hosts = len(self.mag)
        fprint(f"loaded {self.num_hosts} 2MTF host galaxies.")

    def _set_data_arrays(self, data):
        super()._set_data_arrays(data, skip_keys=("host_names",))

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_config(self):
        if self.use_reconstruction and not self.has_host_los:
            raise ValueError(
                "`use_reconstruction` requires host LOS interpolators.")
        if self.use_reconstruction and not self.has_rand_los:
            raise ValueError(
                "`use_reconstruction` requires random LOS interpolators "
                "for the global selection function.")

    # ------------------------------------------------------------------
    #  Selection function (depends only on model params, not data)
    # ------------------------------------------------------------------

    def _log_S_selection(self, lp_r, a_TFR, b_TFR, c_TFR,
                         eta_mean, eta_std, sigma_int, H0,
                         mu_grid=None):
        """Selection fraction: integrate P(sel | r, eta_true) over (r, eta).

        P(sel) = int p(r) int P(mag_sel | r, eta) P(eta_sel | eta)
                              p(eta | hyperprior) deta dr

        where P(eta_sel | eta_true) = Phi((eta_max - eta)/e_eta)
                                    - Phi((eta_min - eta)/e_eta)
        accounts for observed-eta noise at the selection boundaries.
        """
        r_grid = self.r_host_range
        if mu_grid is None:
            mu_grid = self.distance2distmod(r_grid, h=H0 / 100)

        e_eff = jnp.sqrt(sigma_int**2 + self.e2_mag_median)

        eta_grid = self._eta_sel_grid
        n_eta = len(eta_grid)

        # Gaussian hyperprior weight: p(eta_true | eta_mean, eta_std)
        log_p_eta_prior = -0.5 * ((eta_grid - eta_mean) / eta_std)**2 \
            - jnp.log(eta_std) - 0.5 * jnp.log(2 * jnp.pi)

        # P(eta_obs in [eta_min, eta_max] | eta_true, e_eta_median)
        log_p_eta_sel = jnp.zeros(n_eta)
        e_eta_rep = self.e_eta_median
        if self.eta_min_sel is not None and self.eta_max_sel is not None:
            cdf_hi = norm_jax.cdf(
                (self.eta_max_sel - eta_grid) / e_eta_rep)
            cdf_lo = norm_jax.cdf(
                (self.eta_min_sel - eta_grid) / e_eta_rep)
            log_p_eta_sel = jnp.log(jnp.clip(cdf_hi - cdf_lo, 1e-30))
        elif self.eta_min_sel is not None:
            log_p_eta_sel = norm_jax.logcdf(
                (eta_grid - self.eta_min_sel) / e_eta_rep)
        elif self.eta_max_sel is not None:
            log_p_eta_sel = norm_jax.logcdf(
                (self.eta_max_sel - eta_grid) / e_eta_rep)

        M_eta = get_absmag_TFR(eta_grid, a_TFR, b_TFR, c_TFR)

        # m_true at each (eta, r): shape (n_eta, n_r)
        m_pred = mu_grid[None, :] + M_eta[:, None]

        # P(m_obs < mag_lim | m_true, sigma_int)
        log_p_mag_sel = log_prob_integrand_sel(
            m_pred, e_eff, self.mag_lim, self.mag_lim_width)

        # Combine: p(eta) * P(mag_sel) * P(eta_sel) on the eta grid
        integrand = (log_p_mag_sel
                     + log_p_eta_prior[:, None]
                     + log_p_eta_sel[:, None])

        # Trapezoid rule in log-space over eta
        d_eta = eta_grid[1] - eta_grid[0]
        log_trap_w = jnp.full(n_eta, jnp.log(d_eta))
        log_trap_w = log_trap_w.at[0].add(jnp.log(0.5))
        log_trap_w = log_trap_w.at[-1].add(jnp.log(0.5))
        log_sel_r = logsumexp(
            integrand + log_trap_w[:, None], axis=0)

        return ln_simpson_precomputed(
            lp_r + log_sel_r[None, :], self._simpson_log_w, axis=-1)

    # ------------------------------------------------------------------
    #  Forward model
    # ------------------------------------------------------------------

    def __call__(self):
        # --- Global parameters ---
        H0 = rsample("H0", self.priors["H0"])
        a_TFR = rsample("a_TFR", self.priors["a_TFR"])
        b_TFR = rsample("b_TFR", self.priors["b_TFR"])
        c_TFR = rsample("c_TFR", self.priors["c_TFR"])
        sigma_int = rsample("sigma_int", self.priors["sigma_int"])
        sigma_v = rsample("sigma_v", self.priors["sigma_v"])
        Vext = rsample("Vext", self.priors["Vext"])
        beta = rsample("beta", self.priors["beta"])
        bias_params = sample_galaxy_bias(
            self.priors, self.which_bias, beta=beta, Om=self.Om)

        h = H0 / 100

        # Linewidth hyperprior
        eta_mean = sample("eta_mean", Uniform(
            self.eta_min_sel if self.eta_min_sel is not None else -1.0,
            self.eta_max_sel if self.eta_max_sel is not None else 1.0))
        eta_std = sample("eta_std", Uniform(0.01, 0.5))

        # --- Pre-compute cosmographic grids ---
        r_grid = self.r_host_range
        mu_grid = self.distance2distmod(r_grid, h=h)
        z_grid = self.distance2redshift(r_grid, h=h)

        # --- Distance prior: r^2 (uniform in volume) ---
        lp_r = self.log_prior_distance(r_grid)

        # --- Per-host likelihood ---
        Vext_rad_host = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)
        log_w = self._simpson_log_w
        sigma_mag = jnp.sqrt(self.e2_mag + sigma_int**2)

        # Marginalize eta_true via Gauss-Hermite quadrature with
        # Laplace centering (same as TFRModel)
        var_h = eta_std**2
        var_o = self.e2_eta
        prec = 1.0 / var_h + 1.0 / var_o
        mu_c = (eta_mean / var_h + self.eta / var_o) / prec
        sigma_c = 1.0 / jnp.sqrt(prec)

        # Evidence p(eta_obs | eta_mean, eta_std)
        log_Z_eta = Normal(
            eta_mean,
            jnp.sqrt(var_h + var_o)).log_prob(self.eta)

        # Laplace-centered GH quadrature (matches PV TFR model)
        M_c = get_absmag_TFR(mu_c, a_TFR, b_TFR, c_TFR)
        M_prime_c = b_TFR + jnp.where(
            mu_c > 0, 2 * c_TFR * mu_c, 0.0)

        sigma_eff_sq = sigma_mag**2 + (M_prime_c * sigma_c)**2
        sigma_star = sigma_c * sigma_mag / jnp.sqrt(sigma_eff_sq)

        R = (self.mag - M_c)[:, None] - mu_grid[None, :]
        delta_mu = (R * M_prime_c[:, None] * sigma_c[:, None]**2
                    / sigma_eff_sq[:, None])

        # GH nodes at Laplace-shifted center: (n_hosts, n_r, n_gh)
        mu_star = mu_c[:, None] + delta_mu
        eta_nodes = (mu_star[:, :, None]
                     + jnp.sqrt(2.0) * sigma_star[:, None, None]
                     * self._gh_nodes[None, None, :])

        M_eta = get_absmag_TFR(eta_nodes, a_TFR, b_TFR, c_TFR)

        # ll_mag: (n_hosts, n_r, n_gh)
        ll_mag = Normal(
            mu_grid[None, :, None] + M_eta,
            sigma_mag[:, None, None]).log_prob(
                self.mag[:, None, None])

        d_s2x = (delta_mu[:, :, None]
                 + jnp.sqrt(2.0) * sigma_star[:, None, None]
                 * self._gh_nodes[None, None, :])
        log_ratio = (
            jnp.log(sigma_star / sigma_c)[:, None, None]
            + self._gh_nodes**2
            - 0.5 * d_s2x**2
            / sigma_c[:, None, None]**2)

        # Sum over GH nodes -> (n_hosts, n_r)
        ll_eta = logsumexp(
            ll_mag + log_ratio + self._gh_log_w[None, None, :],
            axis=-1)
        ll_eta += log_Z_eta[:, None]

        if self.use_reconstruction:
            rh_grid = r_grid * h
            delta_grid = self.f_host_los_delta.interp_many(rh_grid)
            log_rho = (jnp.log(1 + delta_grid)
                       if "linear" not in self.which_bias else None)
            lp_bias = lp_galaxy_bias(
                delta_grid, log_rho, bias_params, self.which_bias)

            lp_dist = lp_r[None, None, :] + lp_bias

            # Global selection from random LOS (unnormalized prior).
            lp_rand_dist_grid = lp_r[None, None, :]
            rand_delta = \
                self.f_rand_los_delta.interp_many_steps_per_galaxy(
                    r_grid * h)
            log_rho_rand = (jnp.log(1 + rand_delta)
                            if "linear" not in self.which_bias else None)
            lp_rand_dist_grid = lp_rand_dist_grid + lp_galaxy_bias(
                rand_delta, log_rho_rand, bias_params, self.which_bias)

            log_S = logmeanexp(self._log_S_selection(
                lp_rand_dist_grid, a_TFR, b_TFR, c_TFR,
                eta_mean, eta_std, sigma_int, H0, mu_grid=mu_grid),
                axis=-1)

            Vpec_grid = beta * self.f_host_los_velocity.interp_many(rh_grid)
            Vpec_grid += Vext_rad_host[None, :, None]
            cz_pred = predict_cz(z_grid[None, None, :], Vpec_grid)
            ll_cz = Normal(
                cz_pred, sigma_v).log_prob(self.czcmb[None, :, None])

            lp_dist_w = lp_dist + log_w
            ll_host = logsumexp(
                lp_dist_w + ll_eta[None, :, :] + ll_cz,
                axis=-1) - log_S[:, None]
            ll_host = logmeanexp(ll_host, axis=0)
        else:
            # Homogeneous selection (same for all hosts)
            log_S = self._log_S_selection(
                lp_r[None, :], a_TFR, b_TFR, c_TFR,
                eta_mean, eta_std, sigma_int, H0, mu_grid=mu_grid)

            lp_dist = lp_r[None, :]
            log_norm = ln_simpson_precomputed(
                lp_dist, log_w, axis=-1)

            cz_pred = predict_cz(
                z_grid[None, :], Vext_rad_host[:, None])
            ll_cz = Normal(
                cz_pred, sigma_v).log_prob(self.czcmb[:, None])

            ll_host = ln_simpson_precomputed(
                lp_dist + ll_eta + ll_cz,
                log_w, axis=-1) - log_norm
            ll_host -= log_S[0]

        factor("ll_host", jnp.sum(ll_host))
