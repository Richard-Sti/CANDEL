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
Combined TRGB + 2MTF distance ladder forward model.

Each host galaxy has both TRGB and TFR observables. The per-host likelihood
integrates both magnitude likelihoods over a shared distance grid:

    ll_host_i = log ∫ L_TRGB(r) × L_TFR(r) × L_cz(r) × π(r) dr

where L_TFR(r) includes Gauss-Hermite quadrature over the true linewidth η.

Selection is TRGB magnitude only (TRGB has a much smaller volume than
K-band TFR, so it is the binding selection).
"""
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as norm_jax
from numpyro import factor, sample
from numpyro.distributions import Normal, Uniform

from ..util import fprint, get_nested, replace_prior_with_delta
from .base_model import H0ModelBase
from .pv_utils import (gauss_hermite_log_weights, get_absmag_TFR,
                       lp_galaxy_bias, rsample, sample_galaxy_bias)
from .simpson import ln_simpson_precomputed
from .utils import logmeanexp, predict_cz


class TRGB2MTFModel(H0ModelBase):
    """Combined TRGB + 2MTF forward model with shared distance grid."""

    # ------------------------------------------------------------------
    #  Phase 1: model physics
    # ------------------------------------------------------------------

    def _replace_unused_priors(self, config):
        use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        if not use_reconstruction:
            replace_prior_with_delta(config, "beta", 0.0)
        config = self._replace_bias_priors(config)
        return config

    def _load_selection_thresholds(self):
        config = self.config
        priors = config.setdefault(
            "model", {}).setdefault("priors", {})
        which_sel = get_nested(config, "model/which_selection", None)

        if which_sel == "TRGB_magnitude":
            active = {"mag_lim_TRGB", "mag_lim_TRGB_width"}
        else:
            active = set()

        spec = {
            "mag_lim_TRGB": None,
            "mag_lim_TRGB_width": None,
        }
        for name, default in spec.items():
            if name not in active:
                setattr(self, name, None)
                setattr(self, f"_infer_{name}", False)
                continue

            raw = get_nested(config, f"model/{name}", default)
            if raw == "infer":
                p = priors.get(name)
                if p is None:
                    raise ValueError(
                        f"`{name}` set to 'infer' but no "
                        f"prior [model.priors.{name}] found.")
                setattr(self, name, None)
                setattr(self, f"_infer_{name}", True)
                fprint(f"{name} will be inferred.")
            else:
                setattr(self, name, raw)
                setattr(self, f"_infer_{name}", False)

    def _load_model_flags(self):
        super()._load_model_flags()
        # GH quadrature for TFR eta marginalization
        n_gh = get_nested(self.config, "model/n_gauss_hermite", 5)
        self._gh_nodes, self._gh_log_w = gauss_hermite_log_weights(n_gh)
        fprint(f"Gauss-Hermite quadrature with {n_gh} nodes.")

    # ------------------------------------------------------------------
    #  Phase 2: data loading
    # ------------------------------------------------------------------

    def _load_data(self, data):
        super()._load_data(data)
        self.num_hosts = len(self.mag_obs)
        fprint(f"loaded {self.num_hosts} TRGB+2MTF host galaxies.")

    def _set_data_arrays(self, data):
        super()._set_data_arrays(data, skip_keys=("host_names",))

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_config(self):
        if self.use_reconstruction and not self.has_host_los:
            raise ValueError(
                "`use_reconstruction` requires host LOS interpolators.")

        allowed_selection = ["TRGB_magnitude", None]
        if self.which_selection not in allowed_selection:
            raise ValueError(
                f"Unknown `which_selection`: {self.which_selection}. "
                f"Expected one of {allowed_selection}.")

        if self.apply_sel and self.use_reconstruction \
                and not self.has_rand_los:
            raise ValueError(
                "Selection with reconstruction requires random LOS.")

        if self.which_selection == "TRGB_magnitude":
            if self.mag_lim_TRGB is None \
                    and not self._infer_mag_lim_TRGB:
                raise ValueError(
                    "`mag_lim_TRGB` must be set or 'infer' "
                    "for TRGB_magnitude selection.")

    # ------------------------------------------------------------------
    #  Selection functions
    # ------------------------------------------------------------------

    def _resolve_threshold(self, name):
        if getattr(self, f"_infer_{name}"):
            return rsample(name, self.priors[name])
        return getattr(self, name)

    # ------------------------------------------------------------------
    #  Forward model
    # ------------------------------------------------------------------

    def __call__(self):
        # --- Shared parameters ---
        H0 = rsample("H0", self.priors["H0"])
        sigma_v = rsample("sigma_v", self.priors["sigma_v"])
        Vext = rsample("Vext", self.priors["Vext"])
        beta = rsample("beta", self.priors["beta"])
        bias_params = sample_galaxy_bias(
            self.priors, self.which_bias, beta=beta, Om=self.Om)

        h = H0 / 100

        # --- TRGB-specific parameters ---
        M_TRGB = rsample("M_TRGB", self.priors["M_TRGB"])
        sigma_int_TRGB = rsample(
            "sigma_int_TRGB", self.priors["sigma_int_TRGB"])

        # --- TFR-specific parameters ---
        a_TFR = rsample("a_TFR", self.priors["a_TFR"])
        b_TFR = rsample("b_TFR", self.priors["b_TFR"])
        c_TFR = rsample("c_TFR", self.priors["c_TFR"])
        sigma_int_TFR = rsample(
            "sigma_int_TFR", self.priors["sigma_int_TFR"])

        # Linewidth hyperprior
        eta_mean = sample("eta_mean", Uniform(
            self.eta_min_sel if hasattr(self, 'eta_min_sel')
            and self.eta_min_sel is not None else -1.0,
            self.eta_max_sel if hasattr(self, 'eta_max_sel')
            and self.eta_max_sel is not None else 1.0))
        eta_std = sample("eta_std", Uniform(0.01, 0.5))

        # --- Distance moduli (anchors) ---
        dist = Uniform(*self.distmod_limits)
        mu_LMC = sample("mu_LMC", dist)
        mu_N4258 = sample("mu_N4258", dist)

        # --- Geometric anchor constraints ---
        sample("mu_LMC_ll",
               Normal(mu_LMC, self.e_mu_LMC_anchor),
               obs=self.mu_LMC_anchor)
        sample("mu_N4258_ll",
               Normal(mu_N4258, self.e_mu_N4258_anchor),
               obs=self.mu_N4258_anchor)

        # --- TRGB magnitude anchor constraints ---
        factor("mag_LMC_TRGB_ll",
               Normal(M_TRGB + mu_LMC, self.e_mag_LMC_TRGB).log_prob(
                   self.mag_LMC_TRGB))
        factor("mag_N4258_TRGB_ll",
               Normal(M_TRGB + mu_N4258, self.e_mag_N4258_TRGB).log_prob(
                   self.mag_N4258_TRGB))

        # --- Anchor distance prior ---
        mu_anchors = jnp.array([mu_LMC, mu_N4258])
        r_anchors = self.distmod2distance(mu_anchors, h=h)
        lp_anchor_dist = self.log_prior_distance(r_anchors)
        lp_anchor_dist += self.log_grad_distmod2comoving_distance(
            mu_anchors, h=h)
        factor("lp_anchor_dist", lp_anchor_dist)

        # --- Pre-compute cosmographic grids ---
        r_grid = self.r_host_range
        lp_r = self.log_prior_distance(r_grid)
        Vext_rad_host = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)

        mu_grid = self.distance2distmod(r_grid, h=h)
        z_grid = self.distance2redshift(r_grid, h=h)

        # --- TRGB magnitude selection ---
        log_S = None
        if self.which_selection == "TRGB_magnitude":
            mag_lim = self._resolve_threshold("mag_lim_TRGB")
            mag_width = self._resolve_threshold("mag_lim_TRGB_width")

            factor("ll_sel_per_object", jnp.sum(
                norm_jax.logcdf(
                    (mag_lim - self.mag_obs) / mag_width)))

            e_eff = jnp.sqrt(self.e2_mag_obs_median + sigma_int_TRGB**2)
            lp_rand_dist_grid = lp_r[None, None, :]
            if self.use_reconstruction:
                rand_delta = \
                    self.f_rand_los_delta.interp_many_steps_per_galaxy(
                        self.r_host_range * h)
                log_rho = (jnp.log(1 + rand_delta)
                           if "linear" not in self.which_bias else None)
                lp_rand_dist_grid = lp_rand_dist_grid + lp_galaxy_bias(
                    rand_delta, log_rho, bias_params, self.which_bias)

            log_S = logmeanexp(self.log_S_mag(
                lp_rand_dist_grid, M_TRGB, H0, e_eff,
                mag_lim, mag_width, mu_grid=mu_grid), axis=-1)

        # --- Per-host distance integration ---
        self._call_marginalized(
            h, M_TRGB, sigma_int_TRGB,
            a_TFR, b_TFR, c_TFR, sigma_int_TFR,
            eta_mean, eta_std,
            sigma_v, beta, bias_params,
            Vext_rad_host, r_grid, lp_r, log_S,
            mu_grid=mu_grid, z_grid=z_grid)

    # ------------------------------------------------------------------
    #  Distance marginalization
    # ------------------------------------------------------------------

    def _call_marginalized(self, h, M_TRGB, sigma_int_TRGB,
                           a_TFR, b_TFR, c_TFR, sigma_int_TFR,
                           eta_mean, eta_std,
                           sigma_v, beta, bias_params,
                           Vext_rad_host, r_grid, lp_r, log_S,
                           mu_grid=None, z_grid=None):
        if mu_grid is None:
            mu_grid = self.distance2distmod(r_grid, h=h)
        if z_grid is None:
            z_grid = self.distance2redshift(r_grid, h=h)

        log_w = self._simpson_log_w

        # --- TRGB likelihood on grid: (n_hosts, n_r) ---
        sigma_mag_TRGB = jnp.sqrt(self.e2_mag_obs + sigma_int_TRGB**2)
        ll_TRGB = Normal(
            M_TRGB + mu_grid[None, :],
            sigma_mag_TRGB[:, None]).log_prob(self.mag_obs[:, None])

        # --- TFR likelihood on grid via GH quadrature: (n_hosts, n_r) ---
        sigma_mag_TFR = jnp.sqrt(self.e2_mag_TFR + sigma_int_TFR**2)

        var_h = eta_std**2
        var_o = self.e2_eta
        prec = 1.0 / var_h + 1.0 / var_o
        mu_c = (eta_mean / var_h + self.eta / var_o) / prec
        sigma_c = 1.0 / jnp.sqrt(prec)

        # Evidence p(eta_obs | eta_mean, eta_std)
        log_Z_eta = Normal(
            eta_mean,
            jnp.sqrt(var_h + var_o)).log_prob(self.eta)

        # Laplace-centered GH quadrature
        M_c = get_absmag_TFR(mu_c, a_TFR, b_TFR, c_TFR)
        M_prime_c = b_TFR + jnp.where(mu_c > 0, 2 * c_TFR * mu_c, 0.0)

        sigma_eff_sq = sigma_mag_TFR**2 + (M_prime_c * sigma_c)**2
        sigma_star = sigma_c * sigma_mag_TFR / jnp.sqrt(sigma_eff_sq)

        R = (self.mag_TFR - M_c)[:, None] - mu_grid[None, :]
        delta_mu = (R * M_prime_c[:, None] * sigma_c[:, None]**2
                    / sigma_eff_sq[:, None])

        # GH nodes: (n_hosts, n_r, n_gh)
        mu_star = mu_c[:, None] + delta_mu
        eta_nodes = (mu_star[:, :, None]
                     + jnp.sqrt(2.0) * sigma_star[:, None, None]
                     * self._gh_nodes[None, None, :])

        M_eta = get_absmag_TFR(eta_nodes, a_TFR, b_TFR, c_TFR)

        ll_mag_TFR = Normal(
            mu_grid[None, :, None] + M_eta,
            sigma_mag_TFR[:, None, None]).log_prob(
                self.mag_TFR[:, None, None])

        d_s2x = (delta_mu[:, :, None]
                 + jnp.sqrt(2.0) * sigma_star[:, None, None]
                 * self._gh_nodes[None, None, :])
        log_ratio = (
            jnp.log(sigma_star / sigma_c)[:, None, None]
            + self._gh_nodes**2
            - 0.5 * d_s2x**2
            / sigma_c[:, None, None]**2)

        # Sum over GH nodes -> (n_hosts, n_r)
        ll_TFR = logsumexp(
            ll_mag_TFR + log_ratio + self._gh_log_w[None, None, :],
            axis=-1)
        ll_TFR += log_Z_eta[:, None]

        # --- Redshift likelihood + integration ---
        e_cz = jnp.sqrt(self.e2_czcmb + sigma_v**2)

        if self.use_reconstruction:
            rh_grid = r_grid * h

            delta_grid = self.f_host_los_delta.interp_many(rh_grid)
            log_rho = (jnp.log(1 + delta_grid)
                       if "linear" not in self.which_bias else None)
            lp_bias = lp_galaxy_bias(
                delta_grid, log_rho, bias_params, self.which_bias)

            lp_dist = lp_r[None, None, :] + lp_bias

            Vpec_grid = beta * self.f_host_los_velocity.interp_many(rh_grid)
            Vpec_grid += Vext_rad_host[None, :, None]
            cz_pred = predict_cz(z_grid[None, None, :], Vpec_grid)
            ll_cz = Normal(
                cz_pred, e_cz[None, :, None]).log_prob(
                    self.czcmb[None, :, None])

            lp_dist_w = lp_dist + log_w

            # Joint integration: TRGB × TFR × cz × prior
            ll_host = logsumexp(
                lp_dist_w + ll_TRGB[None, :, :] + ll_TFR[None, :, :]
                + ll_cz,
                axis=-1)

            if self.apply_sel:
                ll_host -= log_S[:, None]
            ll_host = logmeanexp(ll_host, axis=0)
        else:
            lp_dist = lp_r[None, :]
            log_normalizer = ln_simpson_precomputed(
                lp_dist, log_w, axis=-1)

            cz_pred = predict_cz(
                z_grid[None, :], Vext_rad_host[:, None])
            ll_cz = Normal(
                cz_pred, e_cz[:, None]).log_prob(self.czcmb[:, None])

            # Joint integration: TRGB × TFR × cz × prior
            ll_host = ln_simpson_precomputed(
                lp_dist + ll_TRGB + ll_TFR + ll_cz,
                log_w, axis=-1) - log_normalizer

            if self.apply_sel:
                ll_host -= log_S[0]

        factor("ll_host", jnp.sum(ll_host))
