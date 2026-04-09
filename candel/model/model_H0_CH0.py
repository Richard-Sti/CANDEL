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
"""Cepheid-calibrated H0 (CH0) forward model in JAX."""
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm as norm_jax
from numpyro import deterministic, factor, plate, sample
from numpyro.distributions import MultivariateNormal, Normal, Uniform

from ..util import fprint, get_nested, replace_prior_with_delta
from .base_model import H0ModelBase
from .integration import ln_simpson_precomputed, simpson_log_weights
from .pv_utils import (lp_galaxy_bias, octupole_radial, quadrupole_radial,
                       rsample, sample_galaxy_bias, sample_octupole,
                       sample_quadrupole, sigmoid_monopole_radial)
from .utils import (log_prob_integrand_sel, logmeanexp, mvn_logpdf_cholesky,
                    normal_logpdf_var, predict_cz)

###############################################################################
#                          Base CH0 model                                     #
###############################################################################


class CH0Model(H0ModelBase):
    """
    Base class for Cepheid-calibrated H0 models, handling configuration,
    data loading, and numerical grid setup.
    """

    # ------------------------------------------------------------------
    #  Phase 1: model physics configuration
    # ------------------------------------------------------------------

    def _replace_unused_priors(self, config):
        config = super()._replace_unused_priors(config)

        use_Cepheid_host_redshift = get_nested(
            config, "model/use_Cepheid_host_redshift", False)
        use_PV_covmat_scaling = get_nested(
            config, "model/use_PV_covmat_scaling", False)
        which_selection = get_nested(
            config, "model/which_selection", None)

        if which_selection not in ["SN_magnitude", "SN_magnitude_redshift"]:  # noqa
            replace_prior_with_delta(config, "M_B", -19.25)

        if not use_Cepheid_host_redshift:
            replace_prior_with_delta(config, "H0", 73.04)
            replace_prior_with_delta(config, "Vext", [0., 0., 0.])
            replace_prior_with_delta(config, "sigma_v", 100.0)

        if not use_PV_covmat_scaling:
            replace_prior_with_delta(config, "A_covmat", 1.0)

        return config

    def _load_selection_thresholds(self):
        active_map = {
            "redshift": {"cz_lim_selection", "cz_lim_selection_width"},
            "SN_magnitude": {"mag_lim_SN", "mag_lim_SN_width"},
            "SN_magnitude_redshift": {
                "cz_lim_selection", "cz_lim_selection_width",
                "mag_lim_SN", "mag_lim_SN_width"},
        }
        spec = {
            "cz_lim_selection": 3300.0,
            "cz_lim_selection_width": None,
            "mag_lim_SN": 14.0,
            "mag_lim_SN_width": None,
        }
        super()._load_selection_thresholds(active_map, spec)

    def _load_model_flags(self):
        super()._load_model_flags()
        config = self.config
        self.use_Cepheid_host_redshift = get_nested(
            config, "model/use_Cepheid_host_redshift", False)
        fprint(f"use_Cepheid_host_redshift set to {self.use_Cepheid_host_redshift}")  # noqa
        self.use_uniform_mu_host_priors = get_nested(
            config, "model/use_uniform_mu_host_priors", True)
        fprint(f"use_uniform_mu_host_priors set to {self.use_uniform_mu_host_priors}")  # noqa
        self.use_fiducial_Cepheid_host_PV_covariance = get_nested(
            config, "model/use_fiducial_Cepheid_host_PV_covariance", True)
        fprint(f"use_fiducial_Cepheid_host_PV_covariance set to {self.use_fiducial_Cepheid_host_PV_covariance}")  # noqa
        self.use_PV_covmat_scaling = get_nested(
            config, "model/use_PV_covmat_scaling", False)
        fprint(f"use_PV_covmat_scaling set to {self.use_PV_covmat_scaling}")
        self.use_density_dependent_sigma_v = get_nested(
            config, "model/use_density_dependent_sigma_v", False)
        fprint("use_density_dependent_sigma_v set to "
               f"{self.use_density_dependent_sigma_v}")
        self.track_host_velocity = get_nested(
            config, "model/track_host_velocity", False)
        fprint(f"track_host_velocity set to {self.track_host_velocity}")
        self.weight_selection_by_covmat_Neff = get_nested(
            config, "model/weight_selection_by_covmat_Neff", False)
        fprint(f"weight_selection_by_covmat_Neff set to "
               f"{self.weight_selection_by_covmat_Neff}")

    # ------------------------------------------------------------------
    #  Phase 2: data loading
    # ------------------------------------------------------------------

    def _load_data(self, data):
        super()._load_data(data)

    def _set_data_arrays(self, data):
        skip = ("q_names", "host_map", "host_names")
        super()._set_data_arrays(data, skip_keys=skip)

    def _setup_malmquist_grid(self):
        """CH0 samples distance moduli directly — only the selection grid
        is needed, not the host integration grid."""
        config = self.config
        r_min = 0.01
        r_max_sel = get_nested(config, "model/r_max_selection", 70)
        dr = get_nested(config, "model/dr_malmquist", 0.5)

        num_pts = int(np.round((r_max_sel - r_min) / dr)) + 1
        num_pts = max(num_pts, 3)
        if num_pts % 2 == 0:
            num_pts += 1

        self.r_sel_range = jnp.linspace(r_min, r_max_sel, num_pts)
        self._simpson_log_w_sel = simpson_log_weights(self.r_sel_range)
        dr_actual = float((r_max_sel - r_min) / (num_pts - 1))
        fprint(f"selection grid: {num_pts} points over "
               f"[{r_min}, {r_max_sel}] Mpc (dr={dr_actual:.3f}).")

        self._lp_sel_dist_grid = self.log_prior_distance(
            self.r_sel_range)[None, None, :]

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_config(self):
        if self.use_reconstruction and self.use_fiducial_Cepheid_host_PV_covariance:  # noqa
            raise ValueError(
                "Cannot use `use_reconstruction` and "
                "`use_fiducial_Cepheid_host_PV_covariance` at the same time.")

        if self.use_reconstruction and not self.has_host_los:
            raise ValueError("Option `use_reconstruction` requires host LOS "
                             "interpolators to be available. Please provide "
                             "`host_los_r` and `host_los_density` "
                             "in the data.")

        if self.use_density_dependent_sigma_v and not self.use_reconstruction:
            raise ValueError(
                "`use_density_dependent_sigma_v` requires "
                "`use_reconstruction` to be set to True.")

        if self.use_density_dependent_sigma_v:
            required = ["sigma_v_low", "sigma_v_high",
                        "log_sigma_v_rho_t", "sigma_v_k"]
            missing = [k for k in required if k not in self.priors]
            if missing:
                raise ValueError(
                    "Missing priors for density-dependent sigma_v: "
                    f"{', '.join(missing)}.")

        allowed_selection = [
            "redshift", "SN_magnitude",
            "SN_magnitude_redshift", None]
        if self.which_selection not in allowed_selection:
            raise ValueError(
                f"Unknown `which_selection`: {self.which_selection}. "
                f"Expected one of {allowed_selection}.")

        if self.which_selection in ["redshift", "SN_magnitude_redshift"] and not self.use_Cepheid_host_redshift:  # noqa
            raise ValueError(
                "If `which_selection` is set to 'redshift', "
                "`use_Cepheid_host_redshift` must be set to True.")

        if self.apply_sel and self.use_uniform_mu_host_priors:
            raise ValueError(
                "If `which_selection` is set, "
                "`use_uniform_mu_host_priors` must be set to False.")

        if self.apply_sel and self.use_reconstruction and not self.has_rand_los:  # noqa
            raise ValueError(
                "If `which_selection` is set and `use_reconstruction` is "
                "True, `has_rand_los` must be set to True.")

        if not self.use_fiducial_Cepheid_host_PV_covariance and self.weight_selection_by_covmat_Neff:  # noqa
            raise ValueError(
                "Cannot use `weight_selection_by_covmat_Neff` without "
                "`use_fiducial_Cepheid_host_PV_covariance` set to True.")

    # ------------------------------------------------------------------
    #  Sampling helpers
    # ------------------------------------------------------------------

    def sample_host_distmod(self):
        """
        Sample distance moduli for host galaxies, with a uniform prior in the
        distance modulus. Includes geometric anchor information for NGC 4258,
        the LMC, and M31.
        """
        dist = Uniform(*self.distmod_limits)

        with plate("hosts", self.num_hosts):
            mu_host = sample("mu_host", dist)

        mu_N4258 = sample("mu_N4258", dist)
        mu_LMC = sample("mu_LMC", dist)
        mu_M31 = sample("mu_M31", dist)

        factor("mu_N4258_ll",
               normal_logpdf_var(mu_N4258, self.mu_N4258_anchor,
                                 self.e2_mu_N4258_anchor))
        factor("mu_LMC_ll",
               normal_logpdf_var(mu_LMC, self.mu_LMC_anchor,
                                 self.e2_mu_LMC_anchor))

        return mu_host, mu_N4258, mu_LMC, mu_M31

    # ------------------------------------------------------------------
    #  Selection functions
    # ------------------------------------------------------------------

    def log_S_cz(self, lp_r, Vpec, H0, sigma_v, cz_lim, cz_width):
        """Probability of detection term if redshift-truncated."""
        return super().log_S_cz(
            lp_r, Vpec, H0, sigma_v, cz_lim, cz_width)

    def log_S_SN_mag(self, lp_r, M_SN, H0, mag_lim, mag_width):
        """Probability of detection term if supernova magnitude-truncated."""
        return self.log_S_mag(
            lp_r, M_SN, H0,
            self.mean_std_mag_SN_unique_Cepheid_host,
            mag_lim, mag_width)

    def log_S_SN_mag_cz(self, lp_r, Vpec, M_SN, H0, sigma_v,
                        mag_lim, mag_width, cz_lim, cz_width):
        """
        Probability of detection term if supernova magnitude and
        redshift-truncated.
        """
        h = H0 / 100
        zcosmo = self.distance2redshift(self.r_sel_range, h=h)
        cz_r = predict_cz(zcosmo[None, None, :], Vpec)
        mag = self.distance2distmod(self.r_sel_range, h=h) + M_SN

        sigma_v = jnp.asarray(sigma_v)
        while sigma_v.ndim < cz_r.ndim:
            sigma_v = sigma_v[None, ...]
        sigma_v = jnp.broadcast_to(sigma_v, cz_r.shape)
        log_prob = log_prob_integrand_sel(
            mag[None, None, :], self.mean_std_mag_SN_unique_Cepheid_host,
            mag_lim, mag_width)
        log_prob += log_prob_integrand_sel(
            cz_r, sigma_v, cz_lim, cz_width)
        return ln_simpson_precomputed(
            lp_r + log_prob, self._simpson_log_w_sel, axis=-1)

    def sigma_v_from_density(self, delta, sigma_v_low, sigma_v_high,
                             log_rho_t, k):
        """Map overdensity to sigma_v through a sigmoid in log density."""
        rho = jnp.clip(1.0 + delta, a_min=1e-6)
        log_rho = jnp.log(rho)
        return sigma_v_low + (sigma_v_high - sigma_v_low) / (
            1.0 + jnp.exp(-k * (log_rho - log_rho_t)))

    def __call__(self):
        # Hubble constant
        H0 = rsample("H0", self.priors["H0"])

        # CPLR calibration
        M_W = rsample("M_W", self.priors["M_W"])
        b_W = rsample("b_W", self.priors["b_W"])
        Z_W = rsample("Z_W", self.priors["Z_W"])

        # SN calibration
        M_B = rsample("M_B", self.priors["M_B"])

        # Velocity field calibration
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
        if self.use_density_dependent_sigma_v:
            sigma_v_low = rsample("sigma_v_low", self.priors["sigma_v_low"])
            sigma_v_high = rsample("sigma_v_high", self.priors["sigma_v_high"])
            log_sigma_v_rho_t = rsample(
                "log_sigma_v_rho_t", self.priors["log_sigma_v_rho_t"])
            sigma_v_k = rsample("sigma_v_k", self.priors["sigma_v_k"])
            sigma_v_base = 0.5 * (sigma_v_low + sigma_v_high)
        else:
            sigma_v = rsample("sigma_v", self.priors["sigma_v"])
            sigma_v_base = sigma_v
        A_covmat = rsample("A_covmat", self.priors["A_covmat"])
        beta = rsample("beta", self.priors["beta"])

        # Galaxy bias parameters
        bias_params = sample_galaxy_bias(
            self.priors, self.which_bias, beta=beta, Om=self.Om)

        def map_sigma_v(delta):
            if self.use_density_dependent_sigma_v:
                return self.sigma_v_from_density(
                    delta, sigma_v_low, sigma_v_high, log_sigma_v_rho_t,
                    sigma_v_k)
            return jnp.broadcast_to(sigma_v_base, delta.shape)

        h = H0 / 100
        Vext_rad_host = self.rhat_host @ Vext
        if Vext_quad is not None:
            Q_mag, q1_hat, q2_hat = Vext_quad
            Vext_rad_host = Vext_rad_host + quadrupole_radial(
                Q_mag, q1_hat, q2_hat, self.rhat_host)
        if Vext_oct is not None:
            O_mag, o1_hat, o2_hat, o3_hat = Vext_oct
            Vext_rad_host = Vext_rad_host + octupole_radial(
                O_mag, o1_hat, o2_hat, o3_hat, self.rhat_host)

        # HST and Gaia zero-point calibration of MW Cepheids.
        factor("M_W_HST",
               normal_logpdf_var(self.M_HST, M_W, self.e2_M_HST))
        factor("M_W_Gaia",
               normal_logpdf_var(self.M_Gaia, M_W, self.e2_M_Gaia))

        mu_host, mu_N4258, mu_LMC, mu_M31 = self.sample_host_distmod()

        # Distance moduli for Cepheids, with per-Cepheid dZP correction.
        dZP = sample("dZP", Normal(0, self.sigma_grnd))
        mu_host_cepheid = jnp.concatenate(
            [mu_host,
             jnp.array([mu_N4258, mu_LMC, mu_M31])]
            )

        # Distance moduli without any corrections.
        mu_host_all = jnp.concatenate(
            [mu_host, jnp.array([mu_N4258, mu_LMC, mu_M31])]
            )

        # Comoving distances to all hosts in Mpc and in Mpc / h.
        r_host_all = self.distmod2distance(mu_host_all, h=h)
        r_host = r_host_all[:self.num_hosts]

        if isinstance(Vext_mono, tuple):
            V_left, r_t, angle = Vext_mono
            k = jnp.tan(angle)
            Vext_rad_host = Vext_rad_host + sigmoid_monopole_radial(
                V_left, r_t, k, r_host)
        elif Vext_mono is not None:
            Vext_rad_host = Vext_rad_host + Vext_mono

        # Do we use a r^2 prior on the host distance moduli?
        if self.use_uniform_mu_host_priors:
            lp_host_dist = jnp.zeros(self.num_hosts)
        else:
            lp_all_host_dist = self.log_prior_distance(r_host_all)
            lp_all_host_dist += self.log_grad_distmod2comoving_distance(
                mu_host_all, h=h)
            lp_host_dist = lp_all_host_dist[:self.num_hosts]

            lp_anchor_dist = lp_all_host_dist[self.num_hosts:]
            factor("lp_anchor_dist", lp_anchor_dist)

        lp_rand_dist_grid, Vext_rad_rand, Vext_mono_sel = \
            self._prepare_selection_grid(
                self._lp_sel_dist_grid, Vext, Vext_quad, Vext_oct,
                Vext_mono)

        if self.use_reconstruction:
            ll_reconstruction, los_delta_host, rh_host = \
                self._apply_host_reconstruction(
                    lp_host_dist, r_host, h, bias_params)
            sigma_v_host = map_sigma_v(los_delta_host)
            if self.apply_sel:
                rand_los_delta_grid = \
                    self.f_rand_los_delta.interp_many_steps_per_galaxy(
                        self.r_sel_range * h)
                log_rho = (jnp.log(1 + rand_los_delta_grid)
                           if "linear" not in self.which_bias
                           else None)
                lp_rand_dist_grid += lp_galaxy_bias(
                    rand_los_delta_grid, log_rho,
                    bias_params, self.which_bias)
                sigma_v_selection = map_sigma_v(rand_los_delta_grid)
                _needs_vel = self.which_selection in [
                    "redshift", "SN_magnitude_redshift"]
                if _needs_vel:
                    rand_los_Vpec_grid = \
                        self.f_rand_los_velocity\
                        .interp_many_steps_per_galaxy(
                            self.r_sel_range * h)
                else:
                    rand_los_Vpec_grid = 0.
            else:
                rand_los_Vpec_grid = 0.
        else:
            rand_los_Vpec_grid = 0.
            self._no_reconstruction_fallback(lp_host_dist)
            sigma_v_host = map_sigma_v(jnp.zeros((1, self.num_hosts)))
            if self.apply_sel:
                sigma_v_selection = map_sigma_v(
                    jnp.zeros((1, self.num_rand_los,
                               self.r_sel_range.size)))

        # SN magnitude likelihood (shared by SN_magnitude* selections)
        if self.which_selection in ["SN_magnitude", "SN_magnitude_redshift"]:
            mag_SN = (self.L_SN_unique_Cepheid_host_dist @ mu_host_all) + M_B
            factor(
                "ll_SN",
                mvn_logpdf_cholesky(
                    self.mag_SN_unique_Cepheid_host, mag_SN,
                    self.L_SN_unique_Cepheid_host))

        # Per-object selection probability + population selection integral
        if self.which_selection == "redshift":
            cz_lim = self._resolve_threshold("cz_lim_selection")
            cz_width = self._resolve_threshold("cz_lim_selection_width")

            factor("ll_sel_per_object", jnp.sum(
                norm_jax.logcdf(
                    (cz_lim - self.czcmb_cepheid_host) / cz_width)))

            Vpec_sel = (Vext_rad_rand[None, ..., None]
                        + beta * rand_los_Vpec_grid)
            if Vext_mono_sel is not None:
                Vpec_sel = Vpec_sel + Vext_mono_sel[None, None, :]
            log_S = self.log_S_cz(
                lp_rand_dist_grid, Vpec_sel,
                H0, sigma_v_selection, cz_lim, cz_width)

            if self.weight_selection_by_covmat_Neff:
                log_S *= self.Neff_PV_covmat_cepheid_host / self.num_hosts

        elif self.which_selection == "SN_magnitude":
            mag_lim = self._resolve_threshold("mag_lim_SN")
            mag_width = self._resolve_threshold("mag_lim_SN_width")

            factor("ll_sel_per_object", jnp.sum(
                norm_jax.logcdf(
                    (mag_lim - self.mag_SN_unique_Cepheid_host) / mag_width)))

            log_S = self.log_S_SN_mag(
                lp_rand_dist_grid, M_B, H0, mag_lim, mag_width)

            if self.weight_selection_by_covmat_Neff:
                log_S *= self.Neff_C_SN_unique_Cepheid_host / self.num_hosts

        elif self.which_selection == "SN_magnitude_redshift":
            cz_lim = self._resolve_threshold("cz_lim_selection")
            cz_width = self._resolve_threshold("cz_lim_selection_width")
            mag_lim = self._resolve_threshold("mag_lim_SN")
            mag_width = self._resolve_threshold("mag_lim_SN_width")

            factor("ll_sel_per_object", jnp.sum(
                norm_jax.logcdf(
                    (cz_lim - self.czcmb_cepheid_host) / cz_width)
                + norm_jax.logcdf(
                    (mag_lim - self.mag_SN_unique_Cepheid_host) / mag_width)))

            Vpec_sel = (Vext_rad_rand[None, ..., None]
                        + beta * rand_los_Vpec_grid)
            if Vext_mono_sel is not None:
                Vpec_sel = Vpec_sel + Vext_mono_sel[None, None, :]
            log_S = self.log_S_SN_mag_cz(
                lp_rand_dist_grid, Vpec_sel,
                M_B, H0, sigma_v_selection,
                mag_lim, mag_width, cz_lim, cz_width)

            if self.weight_selection_by_covmat_Neff:
                log_S *= self.Neff_PV_covmat_cepheid_host / self.num_hosts
        else:
            log_S = jnp.zeros((1, self.num_hosts))

        # axis=-1 averages over random LOS.
        log_S = logmeanexp(log_S, axis=-1)
        # Ensure log_S is (n_fields,) for broadcasting with
        # ll_reconstruction (n_fields, n_hosts).
        log_S = log_S.reshape(-1)

        if self.use_reconstruction:
            ll_reconstruction -= log_S[:, None]
        else:
            factor("neg_log_S_correction", -log_S * self.num_hosts)

        # Now assign these host distances to each Cepheid.
        mu_cepheid = (self.L_Cepheid_host_dist @ mu_host_cepheid
                      ).at[self.idx_dZP].add(dZP)

        mag_cepheid = mu_cepheid + M_W + b_W * self.logP + Z_W * self.OH
        factor(
            "ll_cepheid",
            mvn_logpdf_cholesky(self.mag_cepheid, mag_cepheid, self.L_Cepheid)
            )

        if self.use_Cepheid_host_redshift:
            z_cosmo = self.distmod2redshift(mu_host, h=h)
            if self.use_reconstruction:
                e2_cz = (
                    self.e2_czcmb_cepheid_host[None, :] + sigma_v_host**2)
            else:
                e2_cz = self.e2_czcmb_cepheid_host + sigma_v_host[0]**2

            if self.use_fiducial_Cepheid_host_PV_covariance:
                cz_pred = predict_cz(z_cosmo, Vext_rad_host)
                C = A_covmat * self.PV_covmat_cepheid_host
                C = C.at[jnp.diag_indices(len(e2_cz))].add(e2_cz)
                sample("cz_pred", MultivariateNormal(cz_pred, C),
                       obs=self.czcmb_cepheid_host)
            elif self.use_reconstruction:
                Vpec = beta * self.f_host_los_velocity(rh_host)
                Vpec += Vext_rad_host[None, :]
                cz_pred = predict_cz(z_cosmo[None, :], Vpec)

                if self.track_host_velocity:
                    deterministic("Vpec_host_skipZ", Vpec)

                ll_reconstruction += normal_logpdf_var(
                    self.czcmb_cepheid_host[None, :], cz_pred, e2_cz)

                ll_reconstruction = logmeanexp(ll_reconstruction, axis=0)
                factor("ll_reconstruction", ll_reconstruction)
            else:
                cz_pred = predict_cz(z_cosmo, Vext_rad_host)
                factor("cz_pred",
                       normal_logpdf_var(self.czcmb_cepheid_host,
                                         cz_pred, e2_cz).sum())
