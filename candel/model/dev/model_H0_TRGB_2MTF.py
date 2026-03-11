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
Combined TRGB + 2MTF distance ladder forward model with two groups.

Group 1 (TRGB): all TRGB hosts. A subset also has TFR observables
(the overlap with 2MTF). For overlap hosts the per-host likelihood is:

    ll_i = log int L_TRGB(r) * L_TFR(r) * L_cz(r) * pi(r) dr

For TRGB-only hosts the TFR factor is omitted.
Selection: TRGB magnitude.

Group 2 (TFR-only): 2MTF hosts with no TRGB data. Per-host likelihood:

    ll_i = log int L_TFR(r) * L_cz(r) * pi(r) dr

Selection: TFR magnitude + linewidth.

Both groups share calibration parameters (H0, sigma_v, Vext, beta,
TFR slope/zero-point, eta hyperprior).
"""
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as norm_jax
from numpyro import factor, sample
from numpyro.distributions import Normal, Uniform

from ...util import fprint, fsection, get_nested, replace_prior_with_delta
from ..base_model import H0ModelBase
from ..interp import LOSInterpolator
from ..pv_utils import (gauss_hermite_log_weights, get_absmag_TFR,
                        lp_galaxy_bias, rsample, sample_galaxy_bias)
from ..simpson import ln_simpson_precomputed
from ..utils import (log_prob_integrand_sel, logmeanexp, normal_logpdf_var,
                     predict_cz)


class TRGB2MTFModel(H0ModelBase):
    """Combined TRGB + 2MTF model with two galaxy groups.

    WARNING: this model does not account for the selection mechanism that
    determines which galaxies have both TRGB and TFR data (the overlap).
    It implicitly assumes the overlap is fully determined by the TRGB
    selection — i.e., that having TFR data is independent of galaxy
    properties beyond those already captured by the TRGB selection
    function. If the overlap requires an additional selection criterion
    (e.g., K-band brightness, measurable linewidth), a joint selection
    model for the overlap subset would be needed.
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(self, config_path, data_trgb, data_tfr):
        self._pending_tfr_data = data_tfr
        super().__init__(config_path, data_trgb)

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

        # --- TRGB selection thresholds ---
        which_sel_trgb = get_nested(
            config, "model/which_selection_trgb", "TRGB_magnitude")
        if which_sel_trgb == "TRGB_magnitude":
            active_trgb = {"mag_lim_TRGB", "mag_lim_TRGB_width"}
        else:
            active_trgb = set()

        spec_trgb = {
            "mag_lim_TRGB": None,
            "mag_lim_TRGB_width": None,
        }
        for name, default in spec_trgb.items():
            if name not in active_trgb:
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

        # --- TFR selection thresholds ---
        self.mag_lim_TFR = get_nested(config, "model/mag_lim_TFR", 11.25)
        self.mag_lim_TFR_width = get_nested(
            config, "model/mag_lim_TFR_width", None)
        self.eta_min_sel = get_nested(config, "model/eta_min_sel", None)
        self.eta_max_sel = get_nested(config, "model/eta_max_sel", None)
        fprint(f"TFR selection: mag_lim={self.mag_lim_TFR}, "
               f"mag_lim_width={self.mag_lim_TFR_width}")
        fprint(f"  eta_min_sel={self.eta_min_sel}, "
               f"eta_max_sel={self.eta_max_sel}")

    def _load_model_flags(self):
        super()._load_model_flags()
        # Per-group selection flags
        self.which_selection_trgb = get_nested(
            self.config, "model/which_selection_trgb", "TRGB_magnitude")
        self.which_selection_tfr = get_nested(
            self.config, "model/which_selection_tfr", "magnitude")
        self.apply_sel_trgb = self.which_selection_trgb is not None
        self.apply_sel_tfr = self.which_selection_tfr is not None
        # Override base apply_sel so random LOS setup triggers correctly
        self.apply_sel = self.apply_sel_trgb or self.apply_sel_tfr
        fprint(f"TRGB selection: {self.which_selection_trgb}")
        fprint(f"TFR selection: {self.which_selection_tfr}")

        # GH quadrature for TFR eta marginalization
        n_gh = get_nested(self.config, "model/n_gauss_hermite", 5)
        self._gh_nodes, self._gh_log_w = gauss_hermite_log_weights(n_gh)
        fprint(f"Gauss-Hermite quadrature with {n_gh} nodes.")

        # Eta grid for TFR selection integration
        n_eta_sel = get_nested(self.config, "model/n_eta_sel_grid", 101)
        self._eta_sel_grid = jnp.linspace(-1.0, 1.0, n_eta_sel)

    # ------------------------------------------------------------------
    #  Phase 2: data loading
    # ------------------------------------------------------------------

    def _load_data(self, data):
        # TRGB group via base class
        super()._load_data(data)
        self.n_trgb = len(self.mag_obs)
        self._any_overlap = bool(np.any(data.get("has_TFR", False)))
        fprint(f"loaded {self.n_trgb} TRGB host galaxies.")

        # TFR-only group
        data_tfr = self._pending_tfr_data
        del self._pending_tfr_data
        self._load_tfr_group(data_tfr)

    def _set_data_arrays(self, data):
        super()._set_data_arrays(data, skip_keys=("host_names",))

    def _load_tfr_group(self, data_tfr):
        """Load the TFR-only galaxy group."""
        fsection("TFR-only group")
        self.n_tfr = len(data_tfr["mag"])
        fprint(f"loaded {self.n_tfr} TFR-only host galaxies.")

        # Store TFR arrays
        self.tfr_mag = jnp.asarray(data_tfr["mag"])
        self.tfr_e2_mag = jnp.asarray(data_tfr["e_mag"])**2
        self.tfr_e2_mag_median = float(data_tfr["e_mag_median"])**2
        self.tfr_eta = jnp.asarray(data_tfr["eta"])
        self.tfr_e2_eta = jnp.asarray(data_tfr["e_eta"])**2
        self.tfr_e_eta_median = float(data_tfr["e_eta_median"])
        self.tfr_czcmb = jnp.asarray(data_tfr["czcmb"])
        self.tfr_e2_czcmb = jnp.asarray(data_tfr["e_czcmb"])**2

        # Sky directions
        rhat = np.column_stack([
            np.cos(np.deg2rad(data_tfr["dec_host"]))
            * np.cos(np.deg2rad(data_tfr["RA_host"])),
            np.cos(np.deg2rad(data_tfr["dec_host"]))
            * np.sin(np.deg2rad(data_tfr["RA_host"])),
            np.sin(np.deg2rad(data_tfr["dec_host"])),
        ])
        n = np.linalg.norm(rhat, axis=1, keepdims=True)
        self.tfr_rhat = jnp.asarray(rhat / np.where(n == 0, 1, n))

        # LOS interpolators
        self.has_tfr_los = False
        r0 = get_nested(self.config, "io/los_r0_decay_scale", 5)
        if "host_los_density" in data_tfr:
            los_delta = data_tfr["host_los_density"] - 1
            los_velocity = data_tfr["host_los_velocity"]
            los_r = data_tfr["host_los_r"]
            self.f_tfr_los_delta = LOSInterpolator(
                los_r, los_delta,
                extrap_constant=0., r0_decay_scale=r0)
            self.f_tfr_los_velocity = LOSInterpolator(
                los_r, los_velocity,
                extrap_constant=0., r0_decay_scale=r0)
            self.has_tfr_los = True
            fprint(f"loaded TFR host LOS for {los_delta.shape[1]} galaxies.")

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_config(self):
        if self.use_reconstruction and not self.has_host_los:
            raise ValueError(
                "`use_reconstruction` requires TRGB host LOS.")
        if self.use_reconstruction and not self.has_tfr_los:
            raise ValueError(
                "`use_reconstruction` requires TFR host LOS.")

        if self.which_selection_trgb == "TRGB_magnitude":
            if self.mag_lim_TRGB is None \
                    and not self._infer_mag_lim_TRGB:
                raise ValueError(
                    "`mag_lim_TRGB` must be set or 'infer' "
                    "for TRGB_magnitude selection.")

        if self.apply_sel_trgb and self.use_reconstruction \
                and not self.has_rand_los:
            raise ValueError(
                "TRGB selection with reconstruction requires random LOS.")
        if self.apply_sel_tfr and self.use_reconstruction \
                and not self.has_rand_los:
            raise ValueError(
                "TFR selection with reconstruction requires random LOS.")

    # ------------------------------------------------------------------
    #  Selection helpers
    # ------------------------------------------------------------------

    def _resolve_threshold(self, name):
        if getattr(self, f"_infer_{name}"):
            return rsample(name, self.priors[name])
        return getattr(self, name)

    def _log_S_tfr_selection(self, lp_r, a_TFR, b_TFR, c_TFR,
                             eta_mean, eta_std, sigma_int_TFR, H0,
                             mu_grid=None):
        """TFR selection fraction (magnitude + linewidth)."""
        r_grid = self.r_host_range
        if mu_grid is None:
            mu_grid = self.distance2distmod(r_grid, h=H0 / 100)

        e_eff = jnp.sqrt(sigma_int_TFR**2 + self.tfr_e2_mag_median)

        eta_grid = self._eta_sel_grid
        n_eta = len(eta_grid)

        log_p_eta_prior = -0.5 * ((eta_grid - eta_mean) / eta_std)**2 \
            - jnp.log(eta_std) - 0.5 * jnp.log(2 * jnp.pi)

        log_p_eta_sel = jnp.zeros(n_eta)
        e_eta_rep = self.tfr_e_eta_median
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
        m_pred = mu_grid[None, :] + M_eta[:, None]

        log_p_mag_sel = log_prob_integrand_sel(
            m_pred, e_eff, self.mag_lim_TFR, self.mag_lim_TFR_width)

        integrand = (log_p_mag_sel
                     + log_p_eta_prior[:, None]
                     + log_p_eta_sel[:, None])

        d_eta = eta_grid[1] - eta_grid[0]
        log_trap_w = jnp.full(n_eta, jnp.log(d_eta))
        log_trap_w = log_trap_w.at[0].add(jnp.log(0.5))
        log_trap_w = log_trap_w.at[-1].add(jnp.log(0.5))
        log_sel_r = logsumexp(
            integrand + log_trap_w[:, None], axis=0)

        return ln_simpson_precomputed(
            lp_r + log_sel_r[None, :], self._simpson_log_w, axis=-1)

    # ------------------------------------------------------------------
    #  GH quadrature for TFR likelihood on distance grid
    # ------------------------------------------------------------------

    def _tfr_ll_on_grid(self, mag, e2_mag, eta, e2_eta,
                        a_TFR, b_TFR, c_TFR, sigma_int_TFR,
                        eta_mean, eta_std, mu_grid):
        """Compute TFR log-likelihood on distance grid: (n_hosts, n_r)."""
        var_mag = e2_mag + sigma_int_TFR**2

        var_h = eta_std**2
        var_o = e2_eta
        prec = 1.0 / var_h + 1.0 / var_o
        mu_c = (eta_mean / var_h + eta / var_o) / prec
        sigma_c = 1.0 / jnp.sqrt(prec)

        log_Z_eta = normal_logpdf_var(eta, eta_mean, var_h + var_o)

        M_c = get_absmag_TFR(mu_c, a_TFR, b_TFR, c_TFR)
        M_prime_c = b_TFR + jnp.where(mu_c > 0, 2 * c_TFR * mu_c, 0.0)

        sigma_eff_sq = var_mag + (M_prime_c * sigma_c)**2
        sigma_star = sigma_c * jnp.sqrt(var_mag) / jnp.sqrt(sigma_eff_sq)

        R = (mag - M_c)[:, None] - mu_grid[None, :]
        delta_mu = (R * M_prime_c[:, None] * sigma_c[:, None]**2
                    / sigma_eff_sq[:, None])

        mu_star = mu_c[:, None] + delta_mu
        eta_nodes = (mu_star[:, :, None]
                     + jnp.sqrt(2.0) * sigma_star[:, None, None]
                     * self._gh_nodes[None, None, :])

        M_eta = get_absmag_TFR(eta_nodes, a_TFR, b_TFR, c_TFR)

        ll_mag = normal_logpdf_var(
            mag[:, None, None],
            mu_grid[None, :, None] + M_eta,
            var_mag[:, None, None])

        d_s2x = (delta_mu[:, :, None]
                 + jnp.sqrt(2.0) * sigma_star[:, None, None]
                 * self._gh_nodes[None, None, :])
        log_ratio = (
            jnp.log(sigma_star / sigma_c)[:, None, None]
            + self._gh_nodes**2
            - 0.5 * d_s2x**2
            / sigma_c[:, None, None]**2)

        ll_tfr = logsumexp(
            ll_mag + log_ratio + self._gh_log_w[None, None, :],
            axis=-1)
        ll_tfr += log_Z_eta[:, None]

        return ll_tfr

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
            self.eta_min_sel if self.eta_min_sel is not None else -1.0,
            self.eta_max_sel if self.eta_max_sel is not None else 1.0))
        eta_std = sample("eta_std", Uniform(0.01, 0.5))

        # --- Anchor distance moduli ---
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
               normal_logpdf_var(self.mag_LMC_TRGB,
                                 M_TRGB + mu_LMC,
                                 self.e_mag_LMC_TRGB**2))
        factor("mag_N4258_TRGB_ll",
               normal_logpdf_var(self.mag_N4258_TRGB,
                                 M_TRGB + mu_N4258,
                                 self.e_mag_N4258_TRGB**2))

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
        mu_grid = self.distance2distmod(r_grid, h=h)
        z_grid = self.distance2redshift(r_grid, h=h)

        # =================================================================
        #  Group 1: TRGB hosts (with optional TFR overlap)
        # =================================================================
        self._call_trgb_group(
            h, M_TRGB, sigma_int_TRGB,
            a_TFR, b_TFR, c_TFR, sigma_int_TFR,
            eta_mean, eta_std,
            sigma_v, beta, bias_params, Vext,
            r_grid, lp_r, mu_grid, z_grid)

        # =================================================================
        #  Group 2: TFR-only hosts
        # =================================================================
        self._call_tfr_group(
            h, a_TFR, b_TFR, c_TFR, sigma_int_TFR,
            eta_mean, eta_std,
            sigma_v, beta, bias_params, Vext,
            r_grid, lp_r, mu_grid, z_grid)

    # ------------------------------------------------------------------
    #  Group 1: TRGB hosts
    # ------------------------------------------------------------------

    def _call_trgb_group(self, h, M_TRGB, sigma_int_TRGB,
                         a_TFR, b_TFR, c_TFR, sigma_int_TFR,
                         eta_mean, eta_std,
                         sigma_v, beta, bias_params, Vext,
                         r_grid, lp_r, mu_grid, z_grid):
        log_w = self._simpson_log_w
        Vext_rad = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)
        e2_cz = self.e2_czcmb + sigma_v**2

        # TRGB magnitude likelihood: (n_trgb, n_r)
        var_mag_TRGB = self.e2_mag_obs + sigma_int_TRGB**2
        ll_TRGB = normal_logpdf_var(
            self.mag_obs[:, None],
            M_TRGB + mu_grid[None, :],
            var_mag_TRGB[:, None])

        # TFR likelihood for overlap hosts: (n_trgb, n_r)
        # For hosts without TFR data, this contributes 0.
        ll_TFR = jnp.zeros_like(ll_TRGB)
        if self._any_overlap:
            ll_TFR_raw = self._tfr_ll_on_grid(
                self.mag_TFR, self.e2_mag_TFR, self.eta_trgb, self.e2_eta_trgb,
                a_TFR, b_TFR, c_TFR, sigma_int_TFR,
                eta_mean, eta_std, mu_grid)
            ll_TFR = jnp.where(self.has_TFR[:, None], ll_TFR_raw, 0.0)

        # TRGB selection
        log_S_trgb = None
        if self.which_selection_trgb == "TRGB_magnitude":
            mag_lim = self._resolve_threshold("mag_lim_TRGB")
            mag_width = self._resolve_threshold("mag_lim_TRGB_width")

            factor("ll_sel_per_object_trgb", jnp.sum(
                norm_jax.logcdf(
                    (mag_lim - self.mag_obs) / mag_width)))

            e_eff = jnp.sqrt(self.e2_mag_obs_median + sigma_int_TRGB**2)
            lp_rand_dist_grid = lp_r[None, None, :]
            if self.use_reconstruction:
                rand_delta = \
                    self.f_rand_los_delta.interp_many_steps_per_galaxy(
                        r_grid * h)
                log_rho = (jnp.log(1 + rand_delta)
                           if "linear" not in self.which_bias else None)
                lp_rand_dist_grid = lp_rand_dist_grid + lp_galaxy_bias(
                    rand_delta, log_rho, bias_params, self.which_bias)

            log_S_trgb = logmeanexp(self.log_S_mag(
                lp_rand_dist_grid, M_TRGB, h * 100, e_eff,
                mag_lim, mag_width, mu_grid=mu_grid), axis=-1)

        # Distance integration
        if self.use_reconstruction:
            rh_grid = r_grid * h
            delta_grid = self.f_host_los_delta.interp_many(rh_grid)
            log_rho = (jnp.log(1 + delta_grid)
                       if "linear" not in self.which_bias else None)
            lp_bias = lp_galaxy_bias(
                delta_grid, log_rho, bias_params, self.which_bias)
            lp_dist = lp_r[None, None, :] + lp_bias

            Vpec_grid = beta * self.f_host_los_velocity.interp_many(rh_grid)
            Vpec_grid += Vext_rad[None, :, None]
            cz_pred = predict_cz(z_grid[None, None, :], Vpec_grid)
            ll_cz = normal_logpdf_var(
                self.czcmb[None, :, None],
                cz_pred, e2_cz[None, :, None])

            lp_dist_w = lp_dist + log_w
            ll_host = logsumexp(
                lp_dist_w + ll_TRGB[None, :, :] + ll_TFR[None, :, :]
                + ll_cz, axis=-1)

            if self.apply_sel_trgb:
                ll_host -= log_S_trgb[:, None]
            ll_host = logmeanexp(ll_host, axis=0)
        else:
            lp_dist = lp_r[None, :]
            log_normalizer = ln_simpson_precomputed(
                lp_dist, log_w, axis=-1)

            cz_pred = predict_cz(
                z_grid[None, :], Vext_rad[:, None])
            ll_cz = normal_logpdf_var(
                self.czcmb[:, None], cz_pred, e2_cz[:, None])

            ll_host = ln_simpson_precomputed(
                lp_dist + ll_TRGB + ll_TFR + ll_cz,
                log_w, axis=-1) - log_normalizer

            if self.apply_sel_trgb:
                ll_host -= log_S_trgb[0]

        factor("ll_host_trgb", jnp.sum(ll_host))

    # ------------------------------------------------------------------
    #  Group 2: TFR-only hosts
    # ------------------------------------------------------------------

    def _call_tfr_group(self, h, a_TFR, b_TFR, c_TFR, sigma_int_TFR,
                        eta_mean, eta_std,
                        sigma_v, beta, bias_params, Vext,
                        r_grid, lp_r, mu_grid, z_grid):
        if self.n_tfr == 0:
            return

        log_w = self._simpson_log_w
        Vext_rad = jnp.sum(Vext[None, :] * self.tfr_rhat, axis=1)

        # TFR likelihood on grid: (n_tfr, n_r)
        ll_TFR = self._tfr_ll_on_grid(
            self.tfr_mag, self.tfr_e2_mag,
            self.tfr_eta, self.tfr_e2_eta,
            a_TFR, b_TFR, c_TFR, sigma_int_TFR,
            eta_mean, eta_std, mu_grid)

        if self.use_reconstruction:
            rh_grid = r_grid * h
            delta_grid = self.f_tfr_los_delta.interp_many(rh_grid)
            log_rho = (jnp.log(1 + delta_grid)
                       if "linear" not in self.which_bias else None)
            lp_bias = lp_galaxy_bias(
                delta_grid, log_rho, bias_params, self.which_bias)
            lp_dist = lp_r[None, None, :] + lp_bias

            # TFR selection from random LOS
            lp_rand_dist_grid = lp_r[None, None, :]
            rand_delta = \
                self.f_rand_los_delta.interp_many_steps_per_galaxy(
                    r_grid * h)
            log_rho_rand = (jnp.log(1 + rand_delta)
                            if "linear" not in self.which_bias else None)
            lp_rand_dist_grid = lp_rand_dist_grid + lp_galaxy_bias(
                rand_delta, log_rho_rand, bias_params, self.which_bias)

            log_S_tfr = logmeanexp(self._log_S_tfr_selection(
                lp_rand_dist_grid, a_TFR, b_TFR, c_TFR,
                eta_mean, eta_std, sigma_int_TFR, h * 100,
                mu_grid=mu_grid), axis=-1)

            Vpec_grid = beta * self.f_tfr_los_velocity.interp_many(rh_grid)
            Vpec_grid += Vext_rad[None, :, None]
            cz_pred = predict_cz(z_grid[None, None, :], Vpec_grid)
            ll_cz = normal_logpdf_var(
                self.tfr_czcmb[None, :, None], cz_pred, sigma_v**2)

            lp_dist_w = lp_dist + log_w
            ll_host = logsumexp(
                lp_dist_w + ll_TFR[None, :, :] + ll_cz,
                axis=-1) - log_S_tfr[:, None]
            ll_host = logmeanexp(ll_host, axis=0)
        else:
            log_S_tfr = self._log_S_tfr_selection(
                lp_r[None, :], a_TFR, b_TFR, c_TFR,
                eta_mean, eta_std, sigma_int_TFR, h * 100,
                mu_grid=mu_grid)

            lp_dist = lp_r[None, :]
            log_norm = ln_simpson_precomputed(
                lp_dist, log_w, axis=-1)

            cz_pred = predict_cz(
                z_grid[None, :], Vext_rad[:, None])
            ll_cz = normal_logpdf_var(
                self.tfr_czcmb[:, None], cz_pred, sigma_v**2)

            ll_host = ln_simpson_precomputed(
                lp_dist + ll_TFR + ll_cz,
                log_w, axis=-1) - log_norm
            ll_host -= log_S_tfr[0]

        factor("ll_host_tfr", jnp.sum(ll_host))
