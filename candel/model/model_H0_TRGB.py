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
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as norm_jax
from numpyro import factor, sample
from numpyro.distributions import Normal, Uniform

from ..util import fprint, get_nested, replace_prior_with_delta
from .base_model import H0ModelBase
from .integration import ln_simpson_precomputed
from .pv_utils import (lp_galaxy_bias, octupole_radial, quadrupole_radial,
                       rsample, sample_galaxy_bias, sample_octupole,
                       sample_quadrupole, sigmoid_monopole_radial)
from .utils import logmeanexp, normal_logpdf_var, predict_cz


class TRGBModel(H0ModelBase):
    """
    TRGB-calibrated H0 model for EDD TRGB distance indicators with
    inhomogeneous Malmquist bias correction and peculiar velocity modeling.
    """

    # ------------------------------------------------------------------
    #  Phase 1: model physics
    # ------------------------------------------------------------------

    def _replace_unused_priors(self, config):
        config = super()._replace_unused_priors(config)

        which_sel = get_nested(config, "model/which_selection", None)
        if which_sel != "SN_magnitude":
            replace_prior_with_delta(
                config, "M_B", -19.0, verbose=False)

        return config

    def _load_selection_thresholds(self):
        active_map = {
            "TRGB_magnitude": {"mag_lim_TRGB", "mag_lim_TRGB_width"},
            "redshift": {"cz_lim_selection", "cz_lim_selection_width"},
            "SN_magnitude": {"mag_lim_SN", "mag_lim_SN_width"},
        }
        spec = {
            "cz_lim_selection":       None,
            "cz_lim_selection_width":  None,
            "mag_lim_TRGB":           None,
            "mag_lim_TRGB_width":     None,
            "mag_lim_SN":             None,
            "mag_lim_SN_width":       None,
        }
        super()._load_selection_thresholds(active_map, spec)

    # ------------------------------------------------------------------
    #  Phase 2: data loading
    # ------------------------------------------------------------------

    def _load_data(self, data):
        # Extract SN-level data before super() processes the data dict
        self._has_sn_data = ("m_Bprime" in data
                             and data["m_Bprime"] is not None)
        if self._has_sn_data:
            self._sn_group_index = jnp.asarray(
                data.pop("sn_group_index"))
            self._m_Bprime = jnp.asarray(data.pop("m_Bprime"))
            self._e_m_Bprime = jnp.asarray(data.pop("e_m_Bprime"))
            self._e2_m_Bprime = self._e_m_Bprime ** 2
            self._e_m_Bprime_median = float(
                data.pop("e_m_Bprime_median"))
            n_sn = len(self._m_Bprime)
            n_hosts = len(np.unique(np.asarray(self._sn_group_index)))
            fprint(f"loaded {n_sn} SNe across {n_hosts} hosts "
                   f"for SN magnitude data.")

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

        allowed_selection = [
            "TRGB_magnitude", "redshift", "SN_magnitude", None]
        if self.which_selection not in allowed_selection:
            raise ValueError(
                f"Unknown `which_selection`: {self.which_selection}. "
                f"Expected one of {allowed_selection}.")

        if self.which_selection == "SN_magnitude" \
                and not self._has_sn_data:
            raise ValueError(
                "SN_magnitude selection requires SN data "
                "(m_Bprime, e_m_Bprime) in the data dict.")

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
        if self.which_selection == "SN_magnitude":
            if self.mag_lim_SN is None \
                    and not self._infer_mag_lim_SN:
                raise ValueError(
                    "`mag_lim_SN` must be set or 'infer' "
                    "for SN_magnitude selection.")

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
        Vext_quad = None
        if self.use_Vext_quadrupole:
            Vext_quad = sample_quadrupole(
                "Vext_quad", *self.Vext_quad_mag_range)
        Vext_oct = None
        if self.use_Vext_octupole:
            Vext_oct = sample_octupole(
                "Vext_oct", *self.Vext_oct_mag_range)
        Vext_mono = None
        if self.which_Vext_monopole == "constant":
            Vext_mono = rsample("Vext_mono", self.priors["Vext_mono"])
        elif self.which_Vext_monopole == "sigmoid":
            Vext_mono_left = rsample(
                "Vext_mono_left", self.priors["Vext_mono_left"])
            Vext_mono_rt = rsample(
                "Vext_mono_rt", self.priors["Vext_mono_rt"])
            Vext_mono_angle = rsample(
                "Vext_mono_angle", self.priors["Vext_mono_angle"])
            Vext_mono = (Vext_mono_left, Vext_mono_rt, Vext_mono_angle)
        beta = rsample("beta", self.priors["beta"])
        bias_params = sample_galaxy_bias(
            self.priors, self.which_bias, beta=beta, Om=self.Om)

        h = H0 / 100

        # --- Distance moduli ---
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
        # mu_anchors = jnp.array([mu_LMC, mu_N4258])
        # r_anchors = self.distmod2distance(mu_anchors, h=h)
        # lp_anchor_dist = self.log_prior_distance(r_anchors)
        # lp_anchor_dist += self.log_grad_distmod2comoving_distance(
        #     mu_anchors, h=h)
        # factor("lp_anchor_dist", lp_anchor_dist)

        # --- Per-host cosmographic grids (fine grid) ---
        r_grid = self.r_host_range
        lp_r = self.log_prior_distance(r_grid)
        Vext_rad_host = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)
        Vext_mono_host_grid = None
        if isinstance(Vext_mono, tuple):
            V_left, r_t, angle = Vext_mono
            k = jnp.tan(angle)
            Vext_mono_host_grid = sigmoid_monopole_radial(
                V_left, r_t, k, r_grid)
        elif Vext_mono is not None:
            Vext_mono_host_grid = jnp.broadcast_to(
                Vext_mono, r_grid.shape)
        if Vext_quad is not None:
            Q_mag, q1_hat, q2_hat = Vext_quad
            Vext_rad_host = Vext_rad_host + quadrupole_radial(
                Q_mag, q1_hat, q2_hat, self.rhat_host)
        if Vext_oct is not None:
            O_mag, o1_hat, o2_hat, o3_hat = Vext_oct
            Vext_rad_host = Vext_rad_host + octupole_radial(
                O_mag, o1_hat, o2_hat, o3_hat, self.rhat_host)
        mu_grid = self.distance2distmod(r_grid, h=h)
        z_grid = self.distance2redshift(r_grid, h=h)

        # --- Selection function (coarse grid) ---
        r_sel = self.r_sel_range
        lp_r_sel = self.log_prior_distance(r_sel)

        log_S = None
        ll_sn_host = None

        if self.which_selection == "TRGB_magnitude":
            mag_lim = self._resolve_threshold("mag_lim_TRGB")
            mag_width = self._resolve_threshold("mag_lim_TRGB_width")

            factor("ll_sel_per_object", jnp.sum(
                norm_jax.logcdf((mag_lim - self.mag_obs) / mag_width)))

            e_eff = jnp.sqrt(self.e2_mag_median + sigma_int**2)
            lp_rand_dist_sel = lp_r_sel[None, None, :]
            if self.use_reconstruction:
                rand_delta = \
                    self.f_rand_los_delta.interp_many_steps_per_galaxy(
                        r_sel * h)
                log_rho = (jnp.log(1 + rand_delta)
                           if "linear" not in self.which_bias else None)
                lp_rand_dist_sel = lp_rand_dist_sel + lp_galaxy_bias(
                    rand_delta, log_rho, bias_params, self.which_bias)

            mu_grid_sel = self.distance2distmod(r_sel, h=h)
            log_S = logmeanexp(self.log_S_mag(
                lp_rand_dist_sel, M_TRGB, H0, e_eff,
                mag_lim, mag_width, mu_grid=mu_grid_sel), axis=-1)

        elif self.which_selection == "redshift":
            cz_lim = self._resolve_threshold("cz_lim_selection")
            cz_width = self._resolve_threshold("cz_lim_selection_width")

            factor("ll_sel_per_object", jnp.sum(
                norm_jax.logcdf((cz_lim - self.czcmb) / cz_width)))

            lp_rand_dist_sel = lp_r_sel[None, None, :]
            # Works for rhat_rand_los both (n_los, 3) and (n_sims, n_los, 3)
            Vext_rad_rand = jnp.sum(
                Vext[None, :] * self.rhat_rand_los, axis=-1)
            if Vext_quad is not None:
                Q_mag, q1_hat, q2_hat = Vext_quad
                Vext_rad_rand = Vext_rad_rand + quadrupole_radial(
                    Q_mag, q1_hat, q2_hat, self.rhat_rand_los)
            if Vext_oct is not None:
                O_mag, o1_hat, o2_hat, o3_hat = Vext_oct
                Vext_rad_rand = Vext_rad_rand + octupole_radial(
                    O_mag, o1_hat, o2_hat, o3_hat, self.rhat_rand_los)
            if self.use_reconstruction:
                rand_delta = \
                    self.f_rand_los_delta.interp_many_steps_per_galaxy(
                        r_sel * h)
                log_rho = (jnp.log(1 + rand_delta)
                           if "linear" not in self.which_bias
                           else None)
                lp_rand_dist_sel = lp_rand_dist_sel + lp_galaxy_bias(
                    rand_delta, log_rho, bias_params, self.which_bias)
                rand_los_Vpec_sel = \
                    self.f_rand_los_velocity.interp_many_steps_per_galaxy(
                        r_sel * h)
            else:
                rand_los_Vpec_sel = 0.

            Vpec_sel = (Vext_rad_rand[None, :, None]
                        + beta * rand_los_Vpec_sel)
            if isinstance(Vext_mono, tuple):
                V_left, r_t, angle = Vext_mono
                k = jnp.tan(angle)
                Vpec_sel = Vpec_sel + sigmoid_monopole_radial(
                    V_left, r_t, k, r_sel)[None, None, :]
            elif Vext_mono is not None:
                Vpec_sel = Vpec_sel + Vext_mono
            log_S = logmeanexp(self.log_S_cz(
                lp_rand_dist_sel, Vpec_sel,
                H0, sigma_v, cz_lim, cz_width), axis=-1)

        elif self.which_selection == "SN_magnitude":
            M_B = rsample("M_B", self.priors["M_B"])
            mag_lim = self._resolve_threshold("mag_lim_SN")
            mag_width = self._resolve_threshold("mag_lim_SN_width")

            # Per-SN selection probability
            factor("ll_sel_per_object", jnp.sum(
                norm_jax.logcdf(
                    (mag_lim - self._m_Bprime) / mag_width)))

            # Population selection integral using M_B
            lp_rand_dist_sel = lp_r_sel[None, None, :]
            if self.use_reconstruction:
                rand_delta = \
                    self.f_rand_los_delta.interp_many_steps_per_galaxy(
                        r_sel * h)
                log_rho = (jnp.log(1 + rand_delta)
                           if "linear" not in self.which_bias
                           else None)
                lp_rand_dist_sel = lp_rand_dist_sel + lp_galaxy_bias(
                    rand_delta, log_rho, bias_params, self.which_bias)

            mu_grid_sel = self.distance2distmod(r_sel, h=h)
            log_S = logmeanexp(self.log_S_mag(
                lp_rand_dist_sel, M_B, H0,
                self._e_m_Bprime_median,
                mag_lim, mag_width, mu_grid=mu_grid_sel), axis=-1)

            # SN magnitude likelihood on distance grid
            # Per-SN: (n_sn, n_grid)
            ll_sn_per = normal_logpdf_var(
                self._m_Bprime[:, None],
                M_B + mu_grid[None, :],
                self._e2_m_Bprime[:, None])
            # Sum SNe per host: (n_hosts, n_grid)
            ll_sn_host = jnp.zeros(
                (self.num_hosts, len(r_grid)))
            ll_sn_host = ll_sn_host.at[
                self._sn_group_index].add(ll_sn_per)

        # --- Per-host distance handling ---
        e2_cz = self.e2_czcmb + sigma_v**2

        self._call_marginalized(
            h, M_TRGB, sigma_int, sigma_v, beta, bias_params,
            Vext_rad_host, r_grid, lp_r, e2_cz, log_S,
            mu_grid=mu_grid, z_grid=z_grid,
            ll_sn_host=ll_sn_host,
            Vext_mono_host_grid=Vext_mono_host_grid)

    # ------------------------------------------------------------------
    #  Distance marginalization path
    # ------------------------------------------------------------------

    def _call_marginalized(self, h, M_TRGB, sigma_int, sigma_v, beta,
                           bias_params, Vext_rad_host, r_grid, lp_r,
                           e2_cz, log_S,
                           mu_grid=None, z_grid=None,
                           ll_sn_host=None,
                           Vext_mono_host_grid=None):
        if mu_grid is None:
            mu_grid = self.distance2distmod(r_grid, h=h)
        if z_grid is None:
            z_grid = self.distance2redshift(r_grid, h=h)

        log_w = self._simpson_log_w

        e2_mag = self.e2_mag_obs + sigma_int**2
        ll_mag = normal_logpdf_var(
            self.mag_obs[:, None],
            M_TRGB + mu_grid[None, :],
            e2_mag[:, None])

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
            if Vext_mono_host_grid is not None:
                Vpec_grid += Vext_mono_host_grid[None, None, :]
            cz_pred = predict_cz(z_grid[None, None, :], Vpec_grid)
            ll_cz = normal_logpdf_var(
                self.czcmb[None, :, None], cz_pred,
                e2_cz[None, :, None])

            lp_dist_w = lp_dist + log_w

            # Unnormalized distance integral: log ∫ L_i r²(1+b₁δ_i) dr
            # We do NOT subtract log_normalizer (= log Z_i) because the
            # angular prior π(ℓ,b) ∝ Z(ℓ,b) cancels it. The b₁ constraint
            # comes from Z_i variation across hosts.
            integrand = lp_dist_w + ll_mag[None, :, :] + ll_cz
            if ll_sn_host is not None:
                integrand = integrand + ll_sn_host[None, :, :]
            ll_host = logsumexp(integrand, axis=-1)

            if self.apply_sel:
                ll_host -= log_S[:, None]
            ll_host = logmeanexp(ll_host, axis=0)
        else:
            # Distance prior (volume only)
            lp_dist = lp_r[None, :]
            log_normalizer = ln_simpson_precomputed(
                lp_dist, log_w, axis=-1)

            # Redshift likelihood on grid
            Vpec_no_recon = Vext_rad_host[:, None]
            if Vext_mono_host_grid is not None:
                Vpec_no_recon = Vpec_no_recon + Vext_mono_host_grid[None, :]
            cz_pred = predict_cz(z_grid[None, :], Vpec_no_recon)
            ll_cz = normal_logpdf_var(
                self.czcmb[:, None], cz_pred, e2_cz[:, None])

            # Marginalize over distance
            integrand = lp_dist + ll_mag + ll_cz
            if ll_sn_host is not None:
                integrand = integrand + ll_sn_host
            ll_host = ln_simpson_precomputed(
                integrand, log_w, axis=-1) - log_normalizer

            if self.apply_sel:
                ll_host -= log_S[0]

        factor("ll_host", jnp.sum(ll_host))
