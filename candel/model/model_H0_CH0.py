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
from numpyro import deterministic, factor, plate, sample
from numpyro.distributions import MultivariateNormal, Normal, Uniform

from ..util import fprint, get_nested, replace_prior_with_delta
from .base_model import H0ModelBase
from .pv_utils import lp_galaxy_bias, rsample, sample_galaxy_bias
from .simpson import ln_simpson_precomputed
from .utils import (log_prob_integrand_sel, logmeanexp,
                    mvn_logpdf_cholesky, normal_logpdf_var, predict_cz)

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
        """Replace priors on parameters not used in the model."""
        use_Cepheid_host_redshift = get_nested(
            config, "model/use_Cepheid_host_redshift", False)
        use_PV_covmat_scaling = get_nested(
            config, "model/use_PV_covmat_scaling", False)
        use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
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

        if not use_reconstruction:
            replace_prior_with_delta(config, "beta", 0.0)

        config = self._replace_bias_priors(config)
        return config

    def _load_selection_thresholds(self):
        config = self.config

        spec = {
            "cz_lim_selection": 3300.0,
            "cz_lim_selection_width": None,
            "mag_lim_SN": 14.0,
            "mag_lim_SN_width": None,
        }
        for name, default in spec.items():
            val = get_nested(config, f"model/{name}", default)
            if val == "infer":
                raise ValueError(
                    f"CH0 model does not support inferring `{name}`.")
            setattr(self, name, val)

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
        self._precompute_cepheid_stats()

    def _set_data_arrays(self, data):
        skip = ("q_names", "host_map", "host_names")
        super()._set_data_arrays(data, skip_keys=skip)

    def _precompute_cepheid_stats(self):
        self.mean_logP = jnp.mean(self.logP)
        self.mean_OH = jnp.mean(self.OH)

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

        sample("mu_N4258_ll",
               Normal(self.mu_N4258_anchor, self.e_mu_N4258_anchor),
               obs=mu_N4258)
        sample("mu_LMC_ll",
               Normal(self.mu_LMC_anchor, self.e_mu_LMC_anchor),
               obs=mu_LMC)

        return mu_host, mu_N4258, mu_LMC, mu_M31

    # ------------------------------------------------------------------
    #  Selection functions
    # ------------------------------------------------------------------

    def log_S_cz(self, lp_r, Vpec, H0, sigma_v):
        """Probability of detection term if redshift-truncated."""
        return super().log_S_cz(
            lp_r, Vpec, H0, sigma_v,
            self.cz_lim_selection,
            self.cz_lim_selection_width)

    def log_S_SN_mag(self, lp_r, M_SN, H0):
        """Probability of detection term if supernova magnitude-truncated."""
        return self.log_S_mag(
            lp_r, M_SN, H0,
            self.mean_std_mag_SN_unique_Cepheid_host,
            self.mag_lim_SN, self.mag_lim_SN_width)

    def log_S_SN_mag_cz(self, lp_r, Vpec, M_SN, H0, sigma_v):
        """
        Probability of detection term if supernova magnitude and
        redshift-truncated.
        """
        zcosmo = self.distance2redshift(self.r_sel_range, h=H0 / 100)
        cz_r = predict_cz(zcosmo[None, None, :], Vpec)
        mag = self.distance2distmod(self.r_sel_range, h=H0 / 100) + M_SN

        sigma_v = jnp.asarray(sigma_v)
        while sigma_v.ndim < cz_r.ndim:
            sigma_v = sigma_v[..., None]
        sigma_v = jnp.broadcast_to(sigma_v, cz_r.shape)
        log_prob = log_prob_integrand_sel(
            mag[None, None, :], self.mean_std_mag_SN_unique_Cepheid_host,
            self.mag_lim_SN, self.mag_lim_SN_width)
        log_prob += log_prob_integrand_sel(
            cz_r, sigma_v, self.cz_lim_selection, self.cz_lim_selection_width)
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
        # Project Vext along the LOS to each host.
        Vext_rad_host = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)

        # HST and Gaia zero-point calibration of MW Cepheids.
        sample("M_W_HST", Normal(M_W, self.e_M_HST), obs=self.M_HST)
        sample("M_W_Gaia", Normal(M_W, self.e_M_Gaia), obs=self.M_Gaia)

        mu_host, mu_N4258, mu_LMC, mu_M31 = self.sample_host_distmod()

        # Distance moduli for Cepheids, with corrections for LMC.
        dZP = sample("dZP", Normal(0, self.sigma_grnd))
        mu_host_cepheid = jnp.concatenate(
            [mu_host,
             jnp.array([mu_N4258, mu_LMC + dZP, mu_M31])]
            )

        # Distance moduli without any corrections.
        mu_host_all = jnp.concatenate(
            [mu_host, jnp.array([mu_N4258, mu_LMC, mu_M31])]
            )

        # Comoving distances to all hosts in Mpc and in Mpc / h.
        r_host_all = self.distmod2distance(mu_host_all, h=h)
        r_host = r_host_all[:self.num_hosts]

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

        # Selection grid: built on r_sel_range (coarser, sufficient for
        # the smooth selection integrals).
        lp_sel_dist_grid = self.log_prior_distance(
            self.r_sel_range)[None, None, :]
        lp_rand_dist_grid, Vext_rad_rand = \
            self._prepare_selection_grid(lp_sel_dist_grid, Vext)

        sigma_v_host = None
        sigma_v_selection = None

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
                sigma_v_selection = map_sigma_v(
                    jnp.zeros(
                        (self.num_fields, self.num_rand_los,
                         self.r_sel_range.size)))
        else:
            rand_los_Vpec_grid = 0.
            self._no_reconstruction_fallback(lp_host_dist)
            sigma_v_host = map_sigma_v(jnp.zeros((1, self.num_hosts)))
            sigma_v_selection = map_sigma_v(
                jnp.zeros((1, self.num_rand_los, self.r_sel_range.size)))

        # Selection function (unnormalized prior — no log_Z correction)
        if self.which_selection == "redshift":
            log_S = self.log_S_cz(
                lp_rand_dist_grid,
                Vext_rad_rand[None, :, None] + beta * rand_los_Vpec_grid,
                H0, sigma_v_selection)

            if self.weight_selection_by_covmat_Neff:
                log_S *= self.Neff_PV_covmat_cepheid_host / self.num_hosts
        elif self.which_selection == "SN_magnitude":
            mu_SN = self.L_SN_unique_Cepheid_host_dist @ mu_host_all
            mag_SN = mu_SN + M_B

            log_S = self.log_S_SN_mag(lp_rand_dist_grid, M_B, H0)

            if self.weight_selection_by_covmat_Neff:
                log_S *= self.Neff_C_SN_unique_Cepheid_host / self.num_hosts

            factor(
                "ll_SN",
                mvn_logpdf_cholesky(
                    self.mag_SN_unique_Cepheid_host, mag_SN,
                    self.L_SN_unique_Cepheid_host)
                )
        elif self.which_selection == "SN_magnitude_redshift":
            log_S = self.log_S_SN_mag_cz(
                lp_rand_dist_grid,
                Vext_rad_rand[None, :, None] + beta * rand_los_Vpec_grid,
                M_B, H0, sigma_v_selection)

            mu_SN = self.L_SN_unique_Cepheid_host_dist @ mu_host_all
            mag_SN = mu_SN + M_B

            if self.weight_selection_by_covmat_Neff:
                log_S *= self.Neff_PV_covmat_cepheid_host / self.num_hosts

            factor(
                "ll_SN",
                mvn_logpdf_cholesky(
                    self.mag_SN_unique_Cepheid_host, mag_SN,
                    self.L_SN_unique_Cepheid_host)
                )
        else:
            log_S = jnp.zeros((1, self.num_hosts))

        log_S = logmeanexp(log_S, axis=-1)

        if self.use_reconstruction:
            ll_reconstruction -= log_S[:, None]
        else:
            factor("neg_log_S_correction", -log_S[0] * self.num_hosts)

        # Now assign these host distances to each Cepheid.
        mu_cepheid = self.L_Cepheid_host_dist @ mu_host_cepheid

        logP = self.logP
        OH = self.OH

        # Predict the Cepheid magnitudes and compute their likelihood.
        mag_cepheid = mu_cepheid + M_W + b_W * logP + Z_W * OH
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
