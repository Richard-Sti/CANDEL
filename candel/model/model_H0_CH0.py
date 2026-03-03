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
from jax.debug import print as jprint  # noqa
from jax.scipy.stats import norm as norm_jax
from numpyro import deterministic, factor, plate, sample
from numpyro.distributions import MultivariateNormal, Normal, Uniform

from ..cosmography import (Distance2Distmod, Distmod2Distance,
                           Distmod2Redshift, LogGrad_Distmod2ComovingDistance)
from ..util import (fprint, fsection, get_nested, radec_to_cartesian,
                    replace_prior_with_delta)
from .base_model import ModelBase
from .interp import LOSInterpolator
from .pv_utils import (lp_galaxy_bias, rsample, sample_distance_prior,
                       sample_galaxy_bias)
from .simpson import ln_simpson
from .utils import (log_prior_r_empirical, log_prob_integrand_sel, logmeanexp,
                    mvn_logpdf_cholesky, predict_cz)

###############################################################################
#                          Base CH0 model                                     #
###############################################################################


class CH0Model(ModelBase):
    """
    Base class for Cepheid-calibrated H0 models, handling configuration,
    data loading, and numerical grid setup.
    """

    def __init__(self, config_path, data):
        super().__init__(config_path)
        fsection("Model")

        # --- Model physics: priors, flags, thresholds ---
        self._configure_physics()

        # --- Data: LOS interpolators, arrays, Cepheid stats ---
        self._load_data(data)

        # --- Numerical infrastructure: cosmography, grids ---
        self._setup_grids()

        self._validate_config(data)

        fname_out = get_nested(self.config, "io/fname_output", None)
        if fname_out is not None:
            fprint(f"output will be saved to `{fname_out}`.")

    # ------------------------------------------------------------------
    #  Phase 1: model physics configuration
    # ------------------------------------------------------------------

    def _configure_physics(self):
        config = self.config
        config = self._replace_unused_priors(config)
        self.config = config
        self._load_and_set_priors()
        self._load_selection_thresholds()
        self._load_model_flags()

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
        which_distance_prior = get_nested(
            config, "model/which_distance_prior", "volume")

        if which_selection == "empirical":
            fprint("selected `empirical` selection. Switching the distance "
                   "prior.")
            which_selection = None
            config["model"]["which_selection"] = None
            which_distance_prior = "empirical"
            config["model"]["which_distance_prior"] = "empirical"

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

        if which_distance_prior != "empirical":
            fprint("not using empirical distance prior. Disabling "
                   "its parameters.")
            app = "dist_emp"
            replace_prior_with_delta(config, f"R_{app}", 1., verbose=False)
            replace_prior_with_delta(config, f"p_{app}", 2., verbose=False)
            replace_prior_with_delta(config, f"n_{app}", 1., verbose=False)
        return config

    def _load_selection_thresholds(self):
        config = self.config
        self.cz_lim_selection = get_nested(
            config, "model/cz_lim_selection", 3300.0)
        self.cz_lim_selection_width = get_nested(
            config, "model/cz_lim_selection_width", None)
        self.mag_lim_SN = get_nested(config, "model/mag_lim_SN", 14.0)
        self.mag_lim_SN_width = get_nested(
            config, "model/mag_lim_SN_width", None)
        self.mag_lim_Cepheid = get_nested(
            config, "model/mag_lim_Cepheid", 24.0)
        self.e_mag_Cepheid = get_nested(
            config, "model/e_mag_Cepheid", 0.1)

    def _load_model_flags(self):
        config = self.config
        self.which_distance_prior = get_nested(
            config, "model/which_distance_prior", "volume")
        fprint(f"which_distance_prior set to {self.which_distance_prior}")
        self.which_selection = get_nested(
            config, "model/which_selection", None)
        fprint(f"which_selection set to {self.which_selection}")
        self.num_hosts_selection_mag = get_nested(
            config, "model/num_hosts_selection_mag", None)
        fprint("num_hosts_selection_mag set to "
               f"{self.num_hosts_selection_mag}")
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
        self.use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        fprint(f"use_reconstruction set to {self.use_reconstruction}")
        self.use_density_dependent_sigma_v = get_nested(
            config, "model/use_density_dependent_sigma_v", False)
        fprint("use_density_dependent_sigma_v set to "
               f"{self.use_density_dependent_sigma_v}")
        self.track_host_velocity = get_nested(
            config, "model/track_host_velocity", False)
        fprint(f"track_host_velocity set to {self.track_host_velocity}")
        self.which_bias = get_nested(
            config, "model/which_bias", "linear")
        fprint(f"which_bias set to {self.which_bias}")
        self.weight_selection_by_covmat_Neff = get_nested(
            config, "model/weight_selection_by_covmat_Neff", False)
        fprint(f"weight_selection_by_covmat_Neff set to "
               f"{self.weight_selection_by_covmat_Neff}")

        self.apply_sel = self.which_selection is not None

    # ------------------------------------------------------------------
    #  Phase 2: data loading
    # ------------------------------------------------------------------

    def _load_data(self, data):
        self.has_host_los = False
        self.has_rand_los = False
        self.num_rand_los = 1
        self.num_fields = 1

        r0_decay_scale = get_nested(self.config, "io/los_r0_decay_scale", 5)
        if get_nested(self.config, "io/load_host_los"):
            self._load_los_interpolator(
                data, which="host", r0_decay_scale=r0_decay_scale)
        if get_nested(self.config, "io/load_rand_los"):
            self._load_los_interpolator(
                data, which="rand", r0_decay_scale=r0_decay_scale)

        self._set_data_arrays(data)
        self._precompute_cepheid_stats()

    def _set_data_arrays(self, data):
        keys_popped = []
        for key in list(data.keys()):
            if data[key] is None:
                keys_popped.append(key)
                del data[key]
        fprint("Popped the following keys with `None` values from data: "
               f"{', '.join(keys_popped)}")

        attrs_set = []
        for k, v in data.items():
            if k in ["q_names", "host_map", "host_names"]:
                continue

            if isinstance(v, np.ndarray):
                v = jnp.asarray(v)

            setattr(self, k, v)
            attrs_set.append(k)

            if k.startswith("e_"):
                k = k.replace("e_", "e2_")
                setattr(self, k, v * v)
                attrs_set.append(k)

        def _normalize_rows(x):
            n = jnp.linalg.norm(x, axis=1, keepdims=True)
            return x / jnp.where(n == 0.0, 1.0, n)

        specs = [
            ("rhat_host",     ("RA_host",     "dec_host"),     "host"),
            ("rhat_rand_los", ("rand_los_RA", "rand_los_dec"), "random LOS"),
        ]

        for attr, (ra_key, dec_key), label in specs:
            if ra_key in data and dec_key in data:
                fprint(f"Converting {label} RA/dec to Cartesian coordinates.")
                assert data[ra_key].ndim == 1 and data[dec_key].ndim == 1
                rhat = radec_to_cartesian(data[ra_key], data[dec_key])
                setattr(self, attr, _normalize_rows(rhat))
                attrs_set.append(attr)

        fprint(f"set the following attributes: {', '.join(attrs_set)}")

    def _load_los_interpolator(self, data, which="host", r0_decay_scale=5.):
        if which not in ("host", "rand"):
            raise ValueError("`which` must be either 'host' or 'rand'.")

        los_delta = data[f"{which}_los_density"] - 1
        los_velocity = data[f"{which}_los_velocity"]
        los_r = data[f"{which}_los_r"]

        fprint(f"loaded {which} galaxy LOS interpolators for "
               f"{los_delta.shape[1]} galaxies.")

        if which == "host" and "mask_host" in data:
            m = data["mask_host"]
            los_delta = los_delta[:, m, ...]
            los_velocity = los_velocity[:, m, ...]

        kwargs = {"r0_decay_scale": r0_decay_scale}

        setattr(self, f"has_{which}_los", True)
        setattr(self, f"f_{which}_los_delta",
                LOSInterpolator(
                    los_r, los_delta, extrap_constant=0., **kwargs))
        setattr(self, f"f_{which}_los_velocity",
                LOSInterpolator(
                    los_r, los_velocity, extrap_constant=0., **kwargs))

    def _precompute_cepheid_stats(self):
        self.mean_logP = jnp.mean(self.logP)
        self.mean_OH = jnp.mean(self.OH)
        self.logP_min = jnp.min(self.logP)
        self.logP_max = jnp.max(self.logP)
        self.OH_min = jnp.min(self.OH)
        self.OH_max = jnp.max(self.OH)

    # ------------------------------------------------------------------
    #  Phase 3: numerical grids and interpolation
    # ------------------------------------------------------------------

    def _setup_grids(self):
        self._setup_cosmography()
        self._setup_malmquist_grid()
        self._setup_random_los_grid()
        self._setup_fields_and_bias()

    def _setup_cosmography(self):
        self.distmod2redshift = Distmod2Redshift(Om0=self.Om)
        self.distmod2distance = Distmod2Distance(Om0=self.Om)
        self.distance2distmod_scalar = Distance2Distmod(
            Om0=self.Om, is_scalar=True)
        self.log_grad_distmod2comoving_distance = LogGrad_Distmod2ComovingDistance(Om0=self.Om)  # noqa
        self.distmod_limits = self.config["model"]["distmod_limits"]

    def _setup_malmquist_grid(self):
        config = self.config
        r_limits_malmquist = get_nested(
            config, "model/r_limits_malmquist", [0.01, 350])
        self._num_points_malmquist = get_nested(
            config, "model/num_points_malmquist", 251)
        self.r_host_range = jnp.linspace(
            r_limits_malmquist[0], r_limits_malmquist[1],
            self._num_points_malmquist)
        fprint(
            f"setting radial range from {r_limits_malmquist[0]}"
            f" to {r_limits_malmquist[1]} Mpc with "
            f"{self._num_points_malmquist} points for the "
            f"Cepheid host galaxies.")
        self.Rmax = jnp.max(self.r_host_range)

    def _setup_random_los_grid(self):
        if not self.use_reconstruction and self.apply_sel:
            fprint("overwriting the set of random LOS to a single LOS if not "
                   "using a reconstruction.")
            self.num_rand_los = 1
            self.rand_los_density = jnp.ones(
                (1, self.num_rand_los, self._num_points_malmquist))
            self.rand_los_velocity = jnp.zeros_like(self.rand_los_density)
            self.rhat_rand_los = jnp.zeros((self.num_rand_los, 3))
            self.rand_los_RA = None
            self.rand_los_dec = None

    def _setup_fields_and_bias(self):
        if self.use_reconstruction:
            self.br_min_clip = get_nested(
                self.config, "model/galaxy_bias_min_clip", 1e-5)
            self.num_fields = len(self.host_los_velocity)
            fprint(f"marginalizing over {self.num_fields} field realizations.")

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    def _validate_config(self, data):
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
            "redshift", "SN_magnitude", "Cepheid_magnitude",
            "SN_magnitude_redshift", "SN_magnitude_or_redshift_Nmag", None]
        if self.which_selection not in allowed_selection:
            raise ValueError(
                f"Unknown `which_selection`: {self.which_selection}. "
                "Expected one of ['redshift', 'SN_magnitude', "
                "'Cepheid_magnitude', 'SN_magnitude_redshift', "
                "'SN_magnitude_or_redshift_Nmag', None].")

        if self.which_selection in ["redshift", "SN_magnitude_redshift", "SN_magnitude_or_redshift_Nmag"] and not self.use_Cepheid_host_redshift:  # noqa
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

        if self.which_selection == "SN_magnitude_or_redshift_Nmag":
            if self.num_hosts_selection_mag is None:
                raise ValueError(
                    "Set `model/num_hosts_selection_mag` when using "
                    "`SN_magnitude_or_redshift_Nmag` selection.")
            if not (0 <= self.num_hosts_selection_mag <= self.num_hosts):
                raise ValueError(
                    "`num_hosts_selection_mag` must be between 0 and "
                    "`num_hosts`.")

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
        zcosmo = self.distance2redshift(self.r_host_range, h=H0 / 100)
        cz_r = predict_cz(zcosmo[None, None, :], Vpec)
        sigma_v = jnp.asarray(sigma_v)
        while sigma_v.ndim < cz_r.ndim:
            sigma_v = sigma_v[..., None]
        sigma_v = jnp.broadcast_to(sigma_v, cz_r.shape)
        log_prob = log_prob_integrand_sel(
            cz_r, sigma_v, self.cz_lim_selection, self.cz_lim_selection_width)
        return ln_simpson(
            lp_r + log_prob, x=self.r_host_range[None, None, :], axis=-1)

    def log_S_SN_mag(self, lp_r, M_SN, H0):
        """Probability of detection term if supernova magnitude-truncated."""
        mag = self.distance2distmod(self.r_host_range, h=H0 / 100) + M_SN

        log_prob = log_prob_integrand_sel(
            mag[None, None, :], self.mean_std_mag_SN_unique_Cepheid_host,
            self.mag_lim_SN, self.mag_lim_SN_width)

        return ln_simpson(
            lp_r + log_prob, x=self.r_host_range[None, None, :], axis=-1)

    def log_S_SN_mag_cz(self, lp_r, Vpec, M_SN, H0, sigma_v):
        """
        Probability of detection term if supernova magnitude and
        redshift-truncated.
        """
        zcosmo = self.distance2redshift(self.r_host_range, h=H0 / 100)
        cz_r = predict_cz(zcosmo[None, None, :], Vpec)
        mag = self.distance2distmod(self.r_host_range, h=H0 / 100) + M_SN

        sigma_v = jnp.asarray(sigma_v)
        while sigma_v.ndim < cz_r.ndim:
            sigma_v = sigma_v[..., None]
        sigma_v = jnp.broadcast_to(sigma_v, cz_r.shape)
        log_prob = log_prob_integrand_sel(
            mag[None, None, :], self.mean_std_mag_SN_unique_Cepheid_host,
            self.mag_lim_SN, self.mag_lim_SN_width)
        log_prob += log_prob_integrand_sel(
            cz_r, sigma_v, self.cz_lim_selection, self.cz_lim_selection_width)
        return ln_simpson(
            lp_r + log_prob, x=self.r_host_range[None, None, :], axis=-1)

    def log_S_Cepheid_mag(self, lp_r, M_W, b_W, Z_W, H0):
        """Probability of detection term if Cepheid magnitude-truncated."""
        raise NotImplementedError(
            "Cepheid selection is not understood well enough.")
        mu = self.distance2distmod(self.r_host_range, h=H0 / 100)
        mag = mu[None, :] + M_W + b_W * self.mean_logP + Z_W * self.mean_OH

        log_cdf = norm_jax.logcdf(
            (self.mag_lim_Cepheid - mag[None, ...]) / self.e_mag_Cepheid)
        return ln_simpson(
            lp_r + log_cdf, x=self.r_host_range[None, None, :], axis=-1)

    def log_prior_distance(self, r, **kwargs):
        """Log prior on the physical distance."""
        if self.which_distance_prior == "volume":
            return 2 * jnp.log(r) - 3 * jnp.log(self.Rmax) + jnp.log(3)
        elif self.which_distance_prior == "empirical":
            return log_prior_r_empirical(
                r, kwargs["R"], kwargs["p"], kwargs["n"], Rmax_grid=self.Rmax)
        else:
            raise ValueError(
                f"Unknown distance prior: `{self.which_distance_prior}`")

    def sigma_v_from_density(self, delta, sigma_v_low, sigma_v_high,
                             log_rho_t, k):
        """Map overdensity to sigma_v through a sigmoid in log density."""
        rho = jnp.clip(1.0 + delta, a_min=1e-6)
        log_rho = jnp.log(rho)
        return sigma_v_low + (sigma_v_high - sigma_v_low) / (
            1.0 + jnp.exp(-k * (log_rho - log_rho_t)))

    def __call__(self, ):
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

        # Empirical distance prior calibration
        kwargs_dist = sample_distance_prior(self.priors)

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
        if self.has_host_los:
            rh_host = r_host * h

        # Do we use a r^2 prior on the host distance moduli?
        if self.use_uniform_mu_host_priors:
            lp_host_dist = jnp.zeros(self.num_hosts)
        else:
            lp_all_host_dist = self.log_prior_distance(
                r_host_all, **kwargs_dist)
            lp_all_host_dist += self.log_grad_distmod2comoving_distance(
                mu_host_all, h=h)
            lp_host_dist = lp_all_host_dist[:self.num_hosts]

            lp_anchor_dist = lp_all_host_dist[self.num_hosts:]
            factor("lp_anchor_dist", lp_anchor_dist)

        # Prepare the grid of r^2 prior of shape (eventually)
        # `(n_fields, n_galaxies, n_steps)`.
        lp_host_dist_grid = self.log_prior_distance(
            self.r_host_range, **kwargs_dist)[None, None, :]

        sigma_v_host = None
        sigma_v_selection = None

        # Copy the homogeneous Malmquist bias for the random LOS
        if self.apply_sel:
            lp_rand_dist_grid = jnp.copy(lp_host_dist_grid)
            Vext_rad_rand = jnp.sum(Vext[None, :] * self.rhat_rand_los, axis=1)
        else:
            lp_rand_dist_grid = 0.
            Vext_rad_rand = 0.

        if self.use_reconstruction:
            lp_host_dist = lp_host_dist[None, :]
            los_delta_host = self.f_host_los_delta(rh_host)
            sigma_v_host = map_sigma_v(los_delta_host)

            lp_host_dist += lp_galaxy_bias(
                los_delta_host, jnp.log(1 + los_delta_host), bias_params,
                self.which_bias)

            los_delta_grid = self.f_host_los_delta.interp_many_steps_per_galaxy(  # noqa
                self.r_host_range * h)

            lp_host_dist_grid += lp_galaxy_bias(
                los_delta_grid, jnp.log(1 + los_delta_grid), bias_params,
                self.which_bias)

            lp_host_dist_norm = ln_simpson(
                lp_host_dist_grid, x=self.r_host_range[None, None, ...],
                axis=-1)

            ll_reconstruction = lp_host_dist - lp_host_dist_norm
            lp_host_dist_grid -= lp_host_dist_norm[:, :, None]

            if self.apply_sel:
                rand_los_delta_grid = self.f_rand_los_delta.interp_many_steps_per_galaxy(  # noqa
                    self.r_host_range * h)
                sigma_v_selection = map_sigma_v(rand_los_delta_grid)

                lp_rand_dist_grid += lp_galaxy_bias(
                    rand_los_delta_grid, jnp.log(1 + rand_los_delta_grid),
                    bias_params, self.which_bias)

                lp_rand_dist_grid -= ln_simpson(
                    lp_rand_dist_grid, x=self.r_host_range[None, None, ...],
                    axis=-1)[..., None]

                rand_los_Vpec_grid = self.f_rand_los_velocity.interp_many_steps_per_galaxy(  # noqa
                    self.r_host_range * h)
            else:
                rand_los_Vpec_grid = 0.
                sigma_v_selection = map_sigma_v(
                    jnp.zeros(
                        (self.num_fields, self.num_rand_los,
                         self.r_host_range.size)))
        else:
            rand_los_Vpec_grid = 0.
            lp_host_dist_grid = jnp.repeat(
                lp_host_dist_grid, self.num_hosts, axis=1)
            factor("lp_host_dist", lp_host_dist)
            sigma_v_host = map_sigma_v(jnp.zeros((1, self.num_hosts)))
            sigma_v_selection = map_sigma_v(
                jnp.zeros((1, self.num_rand_los, self.r_host_range.size)))

        # Selection function
        if self.which_selection == "redshift":
            log_S = self.log_S_cz(
                lp_rand_dist_grid,
                Vext_rad_rand[None, :, None] + beta * rand_los_Vpec_grid,
                H0, sigma_v_selection)

            if self.weight_selection_by_covmat_Neff:
                log_S *= self.Neff_PV_covmat_cepheid_host / self.num_hosts
        elif self.which_selection == "SN_magnitude_or_redshift_Nmag":
            log_S_cz = self.log_S_cz(
                lp_rand_dist_grid,
                Vext_rad_rand[None, :, None] + beta * rand_los_Vpec_grid,
                H0, sigma_v_selection)
            log_S_mag = self.log_S_SN_mag(lp_rand_dist_grid, M_B, H0)

            if self.weight_selection_by_covmat_Neff:
                raise NotImplementedError(
                    "Weighting by Neff not implemented for "
                    "`SN_magnitude_or_redshift_Nmag` selection.")

            mu_SN = self.L_SN_unique_Cepheid_host_dist @ mu_host_all
            mag_SN = mu_SN + M_B

            w_mag = self.num_hosts_selection_mag / self.num_hosts

            factor(
                "ll_SN",
                w_mag * mvn_logpdf_cholesky(
                    self.mag_SN_unique_Cepheid_host, mag_SN,
                    self.L_SN_unique_Cepheid_host)
                )
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
        elif self.which_selection == "Cepheid_magnitude":
            raise NotImplementedError(
                "Cepheid selection is not understood well enough.")
        else:
            log_S = jnp.zeros((1, self.num_hosts))

        # Average the selection term over the random line-of-sight
        if self.which_selection == "SN_magnitude_or_redshift_Nmag":
            log_S_cz = logmeanexp(log_S_cz, axis=-1)
            log_S_mag = logmeanexp(log_S_mag, axis=-1)
            w_mag = self.num_hosts_selection_mag / self.num_hosts
            w_cz = 1 - w_mag
            log_S = w_mag * log_S_mag + w_cz * log_S_cz
        else:
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
                e_cz = jnp.sqrt(e2_cz)

                if self.track_host_velocity:
                    deterministic("Vpec_host_skipZ", Vpec)

                ll_reconstruction += Normal(cz_pred, e_cz).log_prob(
                    self.czcmb_cepheid_host[None, :])

                ll_reconstruction = logmeanexp(ll_reconstruction, axis=0)
                factor("ll_reconstruction", ll_reconstruction)
            else:
                cz_pred = predict_cz(z_cosmo, Vext_rad_host)
                e_cz = jnp.sqrt(e2_cz)
                with plate("Cepheid_anchors_redshift", self.num_hosts):
                    sample("cz_pred", Normal(cz_pred, e_cz),
                           obs=self.czcmb_cepheid_host)
