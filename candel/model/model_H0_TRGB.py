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
"""TRGB-calibrated H0 forward model for EDD TRGB distance indicators."""
import jax.numpy as jnp
from numpyro import factor, plate, sample
from numpyro.distributions import Normal, Uniform

from ..util import fprint, get_nested, replace_prior_with_delta
from .base_model import H0ModelBase
from .pv_utils import lp_galaxy_bias, rsample, sample_galaxy_bias
from .simpson import ln_simpson_precomputed
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as norm_jax

from .utils import logmeanexp, predict_cz


class TRGBModel(H0ModelBase):
    """
    TRGB-calibrated H0 model for EDD TRGB distance indicators with
    inhomogeneous Malmquist bias correction and peculiar velocity modeling.
    """

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

        # Only load thresholds relevant to the active selection.
        if which_sel == "TRGB_magnitude":
            active = {"mag_lim_TRGB", "mag_lim_TRGB_width"}
        elif which_sel == "redshift":
            active = {"cz_lim_selection", "cz_lim_selection_width"}
        else:
            active = set()

        spec = {
            "cz_lim_selection":       None,
            "cz_lim_selection_width":  None,
            "mag_lim_TRGB":           None,
            "mag_lim_TRGB_width":     None,
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
        self.marginalize_distance = get_nested(
            self.config, "model/marginalize_distance", True)
        fprint(f"marginalize_distance set to {self.marginalize_distance}")

    # ------------------------------------------------------------------
    #  Phase 2: data loading
    # ------------------------------------------------------------------

    def _load_data(self, data):
        super()._load_data(data)
        self.num_hosts = len(self.mag_obs)
        fprint(f"loaded {self.num_hosts} TRGB host galaxies.")

    def _set_data_arrays(self, data):
        super()._set_data_arrays(data, skip_keys=("host_names",))

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_config(self):
        if self.use_reconstruction and not self.has_host_los:
            raise ValueError(
                "`use_reconstruction` requires host LOS interpolators.")

        allowed_selection = ["TRGB_magnitude", "redshift", None]
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
        if self.which_selection == "redshift":
            if self.cz_lim_selection is None \
                    and not self._infer_cz_lim_selection:
                raise ValueError(
                    "`cz_lim_selection` must be set or "
                    "'infer' for redshift selection.")

    # ------------------------------------------------------------------
    #  Selection functions
    # ------------------------------------------------------------------

    def _resolve_threshold(self, name):
        """Return the threshold value, sampling if flagged."""
        if getattr(self, f"_infer_{name}"):
            return rsample(name, self.priors[name])
        return getattr(self, name)

    def log_S_TRGB_mag(self, lp_r, M_TRGB, H0, sigma_int,
                       mag_lim, mag_width, mu_grid=None):
        """Selection correction for TRGB magnitude limit."""
        e_eff = jnp.sqrt(self.e2_mag_median + sigma_int**2)
        return self.log_S_mag(
            lp_r, M_TRGB, H0, e_eff,
            mag_lim, mag_width, mu_grid=mu_grid)

    # ------------------------------------------------------------------
    #  Forward model
    # ------------------------------------------------------------------

    def __call__(self):
        # --- Global parameters ---
        H0 = rsample("H0", self.priors["H0"])
        M_TRGB = rsample("M_TRGB", self.priors["M_TRGB"])
        sigma_int = rsample("sigma_int", self.priors["sigma_int"])
        sigma_v = rsample("sigma_v", self.priors["sigma_v"])
        Vext = rsample("Vext", self.priors["Vext"])
        beta = rsample("beta", self.priors["beta"])
        bias_params = sample_galaxy_bias(
            self.priors, self.which_bias, beta=beta, Om=self.Om)

        h = H0 / 100

        # --- Distance moduli ---
        dist = Uniform(*self.distmod_limits)
        mu_LMC = sample("mu_LMC", dist)
        mu_N4258 = sample("mu_N4258", dist)

        if not self.marginalize_distance:
            with plate("hosts", self.num_hosts):
                mu_host = sample("mu_host", dist)

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

        # --- Selection function ---
        r_grid = self.r_host_range
        lp_r = self.log_prior_distance(r_grid)
        Vext_rad_host = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)

        # Pre-compute cosmographic grids once
        mu_grid = self.distance2distmod(r_grid, h=h)
        z_grid = self.distance2redshift(r_grid, h=h)

        log_S = None
        if self.which_selection == "TRGB_magnitude":
            mag_lim = self._resolve_threshold("mag_lim_TRGB")
            mag_width = self._resolve_threshold("mag_lim_TRGB_width")

            factor("ll_sel_per_object", jnp.sum(
                norm_jax.logcdf((mag_lim - self.mag_obs) / mag_width)))

            # Global S using JOINT prior π(r,û) ∝ r²(1+b₁δ):
            # p(S=1|Λ) = ∫∫ Φ(r) r²(1+b₁δ) dΩ dr / Z_total
            # MC: (1/N) Σ_j ∫ Φ(r) r²(1+b₁δ_j) dr
            # Note: _apply_rand_reconstruction normalizes per-LOS,
            # so we add log_Z back to get the unnormalized integral.
            e_eff = jnp.sqrt(self.e2_mag_median + sigma_int**2)
            lp_sel_grid = lp_r[None, None, :]
            lp_rand_dist_grid, _ = \
                self._prepare_selection_grid(lp_sel_grid, Vext)
            if self.use_reconstruction:
                lp_rand_dist_grid, _, _, log_Z_rand = \
                    self._apply_rand_reconstruction(
                        lp_rand_dist_grid, h, bias_params)
            else:
                log_Z_rand = 0.

            # log ∫ Φ π_norm dr (per random LOS)
            log_S_norm = self.log_S_mag(
                lp_rand_dist_grid, M_TRGB, H0, e_eff,
                mag_lim, mag_width, mu_grid=mu_grid)
            # Unnormalize: log ∫ Φ r²(1+b₁δ_j) dr = log_S_norm + log_Z
            log_S = logmeanexp(log_S_norm + log_Z_rand, axis=-1)

        elif self.which_selection == "redshift":
            cz_lim = self._resolve_threshold("cz_lim_selection")
            cz_width = self._resolve_threshold("cz_lim_selection_width")

            factor("ll_sel_per_object", jnp.sum(
                norm_jax.logcdf((cz_lim - self.czcmb) / cz_width)))

            lp_sel_grid = lp_r[None, None, :]
            lp_rand_dist_grid, Vext_rad_rand = \
                self._prepare_selection_grid(lp_sel_grid, Vext)
            if self.use_reconstruction:
                lp_rand_dist_grid, _, rand_los_Vpec_grid, _ = \
                    self._apply_rand_reconstruction(
                        lp_rand_dist_grid, h, bias_params)
            else:
                rand_los_Vpec_grid = 0.

            log_S = self.log_S_cz(
                lp_rand_dist_grid,
                Vext_rad_rand[None, :, None]
                + beta * rand_los_Vpec_grid,
                H0, sigma_v, cz_lim, cz_width)
            log_S = logmeanexp(log_S, axis=-1)

        # --- Per-host distance handling ---
        e_cz = jnp.sqrt(self.e2_czcmb + sigma_v**2)

        if self.marginalize_distance:
            self._call_marginalized(
                h, M_TRGB, sigma_int, sigma_v, beta, bias_params,
                Vext_rad_host, r_grid, lp_r, e_cz, log_S,
                mu_grid=mu_grid, z_grid=z_grid)
        else:
            self._call_sampled(
                mu_host, h, M_TRGB, sigma_int, sigma_v, beta,
                bias_params, Vext_rad_host, r_grid, lp_r, e_cz,
                log_S)

    # ------------------------------------------------------------------
    #  Distance marginalization path
    # ------------------------------------------------------------------

    def _call_marginalized(self, h, M_TRGB, sigma_int, sigma_v, beta,
                           bias_params, Vext_rad_host, r_grid, lp_r,
                           e_cz, log_S,
                           mu_grid=None, z_grid=None):
        if mu_grid is None:
            mu_grid = self.distance2distmod(r_grid, h=h)
        if z_grid is None:
            z_grid = self.distance2redshift(r_grid, h=h)

        log_w = self._simpson_log_w

        sigma_mag = jnp.sqrt(self.e2_mag_obs + sigma_int**2)
        ll_mag = Normal(
            M_TRGB + mu_grid[None, :],
            sigma_mag[:, None]).log_prob(self.mag_obs[:, None])

        if self.use_reconstruction:
            rh_grid = r_grid * h

            delta_grid = self.f_host_los_delta.interp_many(rh_grid)
            log_rho = (jnp.log(1 + delta_grid)
                       if "linear" not in self.which_bias else None)
            lp_bias = lp_galaxy_bias(
                delta_grid, log_rho,
                bias_params, self.which_bias)

            # Unnormalized log distance prior (volume + bias)
            lp_dist = lp_r[None, None, :] + lp_bias

            # Redshift likelihood on grid
            Vpec_grid = beta * self.f_host_los_velocity.interp_many(
                rh_grid)
            Vpec_grid += Vext_rad_host[None, :, None]
            cz_pred = predict_cz(z_grid[None, None, :], Vpec_grid)
            ll_cz = Normal(
                cz_pred, e_cz[None, :, None]).log_prob(
                    self.czcmb[None, :, None])

            lp_dist_w = lp_dist + log_w

            # Unnormalized distance integral: log ∫ L_i r²(1+b₁δ_i) dr
            # We do NOT subtract log_normalizer (= log Z_i) because the
            # angular prior π(ℓ,b) ∝ Z(ℓ,b) cancels it. The b₁ constraint
            # comes from Z_i variation across hosts.
            ll_host = logsumexp(
                lp_dist_w + ll_mag[None, :, :] + ll_cz,
                axis=-1)

            if self.apply_sel:
                ll_host -= log_S[:, None]
            ll_host = logmeanexp(ll_host, axis=0)
        else:
            # Distance prior (volume only)
            lp_dist = lp_r[None, :]
            log_normalizer = ln_simpson_precomputed(
                lp_dist, log_w, axis=-1)

            # Redshift likelihood on grid
            cz_pred = predict_cz(
                z_grid[None, :], Vext_rad_host[:, None])
            ll_cz = Normal(
                cz_pred, e_cz[:, None]).log_prob(self.czcmb[:, None])

            # Marginalize over distance
            ll_host = ln_simpson_precomputed(
                lp_dist + ll_mag + ll_cz,
                log_w, axis=-1) - log_normalizer

            if self.apply_sel:
                ll_host -= log_S[0]

        factor("ll_host", jnp.sum(ll_host))

    # ------------------------------------------------------------------
    #  Distance sampling path
    # ------------------------------------------------------------------

    def _call_sampled(self, mu_host, h, M_TRGB, sigma_int, sigma_v, beta,
                      bias_params, Vext_rad_host, r_grid, lp_r, e_cz,
                      log_S):
        # Magnitude likelihood at sampled mu_host
        sigma_mag = jnp.sqrt(self.e2_mag_obs + sigma_int**2)
        factor("ll_mag", jnp.sum(
            Normal(M_TRGB + mu_host, sigma_mag).log_prob(self.mag_obs)))

        # Distance prior at sampled point
        r_host = self.distmod2distance(mu_host, h=h)
        lp_host_dist = self.log_prior_distance(r_host)
        lp_host_dist += self.log_grad_distmod2comoving_distance(
            mu_host, h=h)

        lp_host_dist_grid = lp_r[None, None, :]

        if self.use_reconstruction:
            ll_reconstruction, _, _, rh_host = \
                self._apply_host_reconstruction(
                    lp_host_dist, lp_host_dist_grid, r_host, h,
                    bias_params)

            if self.apply_sel:
                ll_reconstruction -= log_S[:, None]

            z_cosmo = self.distmod2redshift(mu_host, h=h)
            Vpec = beta * self.f_host_los_velocity(rh_host)
            Vpec += Vext_rad_host[None, :]
            cz_pred = predict_cz(z_cosmo[None, :], Vpec)

            ll_reconstruction += Normal(
                cz_pred, e_cz[None, :]).log_prob(
                    self.czcmb[None, :])
            ll_reconstruction = logmeanexp(ll_reconstruction, axis=0)
            factor("ll_reconstruction", ll_reconstruction)
        else:
            self._no_reconstruction_fallback(
                lp_host_dist, lp_host_dist_grid)

            if self.apply_sel:
                factor("neg_log_S_correction",
                       -log_S[0] * self.num_hosts)

            z_cosmo = self.distmod2redshift(mu_host, h=h)
            cz_pred = predict_cz(z_cosmo, Vext_rad_host)
            factor("ll_cz", jnp.sum(
                Normal(cz_pred, e_cz).log_prob(self.czcmb)))
