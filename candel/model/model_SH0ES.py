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
"""Turning the SH0ES framework into a forward model in JAX."""
from abc import ABC

import jax.numpy as jnp
import numpy as np
from jax.debug import print as jprint  # noqa
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as norm_jax
from numpyro import factor, plate, sample
from numpyro.distributions import (HalfNormal, MultivariateNormal, Normal,
                                   Uniform)

from ..cosmography import (Distance2Distmod, Distance2Redshift,
                           Distmod2Distance, Distmod2Redshift,
                           LogGrad_Distmod2ComovingDistance)
from ..util import (fprint, get_nested, load_config, radec_to_cartesian,
                    replace_prior_with_delta)
from .interp import LOSInterpolator
from .model import (JeffreysPrior, load_priors, log_prior_r_empirical,
                    mvn_logpdf_cholesky, predict_cz, rsample)
from .simpson import ln_simpson


def log_integral_gauss_pdf_times_cdf(mu, sigma, t, w):
    """
    Log of ∫ N(x|mu, sigma^2) Φ((t - x)/w) dx.
    Closed form: Φ((mu - t)/sqrt(sigma^2 + w^2))
    """
    return norm_jax.logcdf((t - mu) / jnp.sqrt(sigma**2 + w**2))


def log_prob_integrand_sel(x, e_x, lim, lim_width):
    if lim_width is None:
        return norm_jax.logcdf((lim - x) / e_x)
    else:
        return log_integral_gauss_pdf_times_cdf(x, e_x, lim, lim_width)


class BaseSH0ESModel(ABC):
    """
    Base class for the SH0ES model, providing common functionality and
    configuration loading.
    """

    def __init__(self, config_path, data):
        config = load_config(config_path, replace_los_prior=False)

        # Set unused parameter priors to delta functions.
        config = self.replace_priors(config)
        # Unpack and set the priors.
        priors = config["model"]["priors"]
        self.priors, self.prior_dist_name = load_priors(priors)

        self.config = config

        # Load the host and random galaxy LOS interpolators if available.
        r0_decay_scale = get_nested(config, "io/los_r0_decay_scale", 5)
        if get_nested(config, "io/load_host_los"):
            self.get_los_interpolator(
                data, which="host", r0_decay_scale=r0_decay_scale)

        if get_nested(config, "io/load_rand_los"):
            self.get_los_interpolator(
                data, which="rand", r0_decay_scale=r0_decay_scale)

        # Load the data, set attributes, convert to JAX arrays and do
        # any other conversions.
        self.set_data(data)
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

        # Initialize the interpolators
        self.Om = get_nested(config, "model/Om0", 0.3)
        self.distmod2redshift = Distmod2Redshift(Om0=self.Om)
        self.distmod2distance = Distmod2Distance(Om0=self.Om)

        self.distance2distmod_scalar = Distance2Distmod(
            Om0=self.Om, is_scalar=True)
        self.distance2distmod = Distance2Distmod(Om0=self.Om)
        self.distance2redshift = Distance2Redshift(Om0=self.Om)

        self.log_grad_distmod2comoving_distance = LogGrad_Distmod2ComovingDistance(Om0=self.Om)  # noqa

        self.distmod_limits = self.config["model"]["distmod_limits"]

        self.use_SNe_HF_SH0ES = get_nested(config, "model/use_SNe_HF_SH0ES", False)  # noqa
        fprint(f"use_SNe_HF_SH0ES set to {self.use_SNe_HF_SH0ES}")
        self.use_SNe_HF_Bayes = get_nested(config, "model/use_SNe_HF_Bayes", False)  # noqa
        fprint(f"use_SNe_HF_Bayes set to {self.use_SNe_HF_Bayes}")

        if self.use_SNe_HF_SH0ES and self.use_SNe_HF_Bayes:
            raise ValueError(
                "Cannot use both `use_SNe_HF_SH0ES` and `use_SNe_HF_Bayes`.")

        self.use_MNR = get_nested(config, "model/use_MNR", False)
        fprint(f"use_MNR set to {self.use_MNR}")
        self.which_selection = get_nested(
            config, "model/which_selection", None)
        fprint(f"which_selection set to {self.which_selection}")
        self.use_Cepheid_host_redshift = get_nested(
            config, "model/use_Cepheid_host_redshift", False)
        fprint(f"use_Cepheid_host_redshift set to {self.use_Cepheid_host_redshift}")  # noqa
        self.use_uniform_mu_host_priors = get_nested(
            config, "model/use_uniform_mu_host_priors", True)
        fprint(f"use_uniform_mu_host_priors set to {self.use_uniform_mu_host_priors}")  # noqa
        self.which_distance_prior = get_nested(
            config, "model/which_distance_prior", "volume")
        fprint(f"which_distance_prior set to {self.which_distance_prior}")
        self.use_fiducial_Cepheid_host_PV_covariance = get_nested(
            config, "model/use_fiducial_Cepheid_host_PV_covariance", True)
        fprint(f"use_fiducial_Cepheid_host_PV_covariance set to {self.use_fiducial_Cepheid_host_PV_covariance}")  # noqa
        self.use_PV_covmat_scaling = get_nested(
            config, "model/use_PV_covmat_scaling", False)
        fprint(f"use_PV_covmat_scaling set to {self.use_PV_covmat_scaling}")
        self.use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        fprint(f"use_reconstruction set to {self.use_reconstruction}")
        self.which_bias = get_nested(
            config, "model/which_bias", "linear")
        fprint(f"which_bias set to {self.which_bias}")
        self.weight_selection_by_covmat_Neff = get_nested(
            config, "model/weight_selection_by_covmat_Neff", False)
        fprint(f"weight_selection_by_covmat_Neff set to "
               f"{self.weight_selection_by_covmat_Neff}")

        self.apply_sel = self.which_selection is not None

        if self.use_reconstruction and self.use_fiducial_Cepheid_host_PV_covariance:  # noqa
            raise ValueError(
                "Cannot use `use_reconstruction` and "
                "`use_fiducial_Cepheid_host_PV_covariance` at the same time.")

        if self.use_reconstruction and not self.has_host_los:
            raise ValueError("Option `use_reconstruction` requires host LOS "
                             "interpolators to be available. Please provide "
                             "`host_los_r` and `host_los_density` "
                             "in the data.")

        if self.which_selection not in ["redshift", "SN_magnitude", "Cepheid_magnitude", "SN_magnitude_redshift", None]:  # noqa
            raise ValueError(
                f"Unknown `which_selection`: {self.which_selection}. "
                "Expected one of ['redshift', 'SN_magnitude', 'Cepheid_magnitude', 'SN_magnitude_redshift', None].")  # noqa

        if self.which_selection in ["redshift", "SN_magnitude_redshift"] and not self.use_Cepheid_host_redshift:  # noqa
            raise ValueError(
                "If `which_selection` is set to 'redshift', "
                "`use_Cepheid_host_redshift` must be set to True.")

        if self.apply_sel and self.use_uniform_mu_host_priors:  # noqa
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

        r_limits_malmquist = get_nested(
            config, "model/r_limits_malmquist", [0.01, 350])
        num_points_malmquist = get_nested(
            config, "model/num_points_malmquist", 251)
        r_range = jnp.linspace(
            r_limits_malmquist[0], r_limits_malmquist[1],
            num_points_malmquist)

        fprint(f"setting radial range from {r_limits_malmquist[0]} to "
               f"{r_limits_malmquist[1]} Mpc with {num_points_malmquist} "
               f"points for the Cepheid host galaxies.")
        self.r_host_range = r_range
        self.Rmax = jnp.max(self.r_host_range)

        if not self.use_reconstruction and self.apply_sel:
            fprint("overwriting the set of random LOS to a single LOS if not "
                   "using a reconstruction.")
            self.num_rand_los = 1
            self.rand_los_density = jnp.ones(
                (1, self.num_rand_los, num_points_malmquist))
            self.rand_los_velocity = jnp.zeros_like(self.rand_los_density)
            # Set this one to zero, so that Vext is never propagated.
            self.rhat_rand_los = jnp.zeros((self.num_rand_los, 3))
            self.rand_los_RA = None
            self.rand_los_dec = None

        if self.use_reconstruction:
            self.br_min_clip = get_nested(
                config, "model/galaxy_bias_min_clip", 1e-5)

            self.num_fields = len(self.host_los_velocity)
            fprint(f"marginalizing over {self.num_fields} field realizations.")

        self.mean_logP = jnp.mean(self.logP)
        self.mean_OH = jnp.mean(self.OH)

        # Precompute min-max for MNR priors.
        self.logP_min = jnp.min(data["logP"])
        self.logP_max = jnp.max(data["logP"])

        self.OH_min = jnp.min(data["OH"])
        self.OH_max = jnp.max(data["OH"])

        if data["Cepheids_only"] and (self.use_SNe_HF_SH0ES or self.use_SNe_HF_Bayes):  # noqa
            raise ValueError(
                "Cannot use SNe_HF with Cepheids only data. Likely because of "
                "imposing a redshift threshold on the Cepheid hosts.")

        fname_out = get_nested(config, "io/fname_output", None)
        if fname_out is not None:
            fprint(f"output will be saved to `{fname_out}`.")

    def replace_priors(self, config):
        """Replace priors on parameters that are not used in the model."""
        use_SNe = (
            get_nested(config, "model/use_SNe_HF_SH0ES", False)
            or get_nested(config, "model/use_SNe_HF_Bayes", False))
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

        if not use_SNe and not which_selection in ["SN_magnitude", "SN_magnitude_redshift"]:  # noqa
            replace_prior_with_delta(config, "M_B", -19.25)

        if not (use_Cepheid_host_redshift or use_SNe):
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

    def set_data(self, data):
        keys_popped = []
        for key in list(data.keys()):
            if data[key] is None:
                keys_popped.append(key)
                del data[key]
        fprint("Popped the following keys with `None` values from data: "
               f"{', '.join(keys_popped)}")

        attrs_set = []
        for k, v in data.items():
            if k in ["q_names", "host_map"]:
                continue

            if isinstance(v, np.ndarray):
                v = jnp.asarray(v)

            setattr(self, k, v)
            attrs_set.append(k)

            if k.startswith("e_"):
                k = k.replace("e_", "e2_")
                setattr(self, k, v * v)
                attrs_set.append(k)

        def _normalize_rows(x: np.ndarray) -> np.ndarray:
            """Normalize each row vector in a 2D array to unit length."""
            n = jnp.linalg.norm(x, axis=1, keepdims=True)
            # Avoid division by zero by replacing zero-norm rows with ones
            return x / jnp.where(n == 0.0, 1.0, n)

        # Each entry defines: (attribute name, (RA key, Dec key))
        specs = [
            ("rhat_host",     ("RA_host",     "dec_host"),     "host"),
            ("rhat_rand_los", ("rand_los_RA", "rand_los_dec"), "random LOS"),
            ("rhat_SN_HF",    ("RA_SN_HF",    "dec_SN_HF"),    "SN_HF"),
        ]

        for attr, (ra_key, dec_key), label in specs:
            if ra_key in data and dec_key in data:
                fprint(f"Converting {label} RA/dec to Cartesian coordinates.")
                rhat = radec_to_cartesian(data[ra_key], data[dec_key])
                # Store normalized Cartesian unit vectors as attributes
                setattr(self, attr, _normalize_rows(rhat))
                attrs_set.append(attr)  # Keep track of set attributes

        fprint(f"set the following attributes: {', '.join(attrs_set)}")

    def get_los_interpolator(self, data, which="host", r0_decay_scale=5.):
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

    def sample_host_distmod(self):
        """
        Sample distance moduli for host galaxies, with a uniform prior in the
        distance modulus. The log PDF of the prior if r^2 gets added directly
        within the model. Includes the geometric anchor information for
        NGC 4258 and the LMC.
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


class SH0ESModel(BaseSH0ESModel):
    """SH0ES forward model for either Cepheid-only or Cepheid+SNe data."""
    def __init__(self, config_path, data):
        super().__init__(config_path, data)

    def log_S_cz(self, lp_r, Vpec, H0, sigma_v):
        """Probability of detection term if redshift-truncated."""
        # Cosmological redshift of shape `(n_steps,)`
        zcosmo = self.distance2redshift(self.r_host_range, h=H0 / 100)
        cz_r = predict_cz(zcosmo[None, None, :], Vpec)
        log_prob = log_prob_integrand_sel(
            cz_r, sigma_v, self.cz_lim_selection, self.cz_lim_selection_width)
        # The expected output is of the shape `(n_fields, n_galaxies,)`,
        return ln_simpson(
            lp_r + log_prob, x=self.r_host_range[None, None, :], axis=-1)

    def log_S_SN_mag(self, lp_r, M_SN, H0):
        """Probability of detection term if supernova magnitude-truncated."""
        mag = self.distance2distmod(self.r_host_range, h=H0 / 100) + M_SN

        log_prob = log_prob_integrand_sel(
            mag[None, None, :], self.mean_std_mag_SN_unique_Cepheid_host,
            self.mag_lim_SN, self.mag_lim_SN_width)

        # The expected output is of the shape `(n_fields, n_galaxies,)`,
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
            "Cepheid selection is not understood well enough..")
        mu = self.distance2distmod(self.r_host_range, h=H0 / 100)
        mag = mu[None, :] + M_W + b_W * self.mean_logP + Z_W * self.mean_OH

        log_cdf = norm_jax.logcdf(
            (self.mag_lim_Cepheid - mag[None, ...]) / self.e_mag_Cepheid)
        return ln_simpson(
            lp_r + log_cdf, x=self.r_host_range[None, None, :], axis=-1)

    def log_prior_distance(self, r, **kwargs):
        """Log prior on the (physical distance)."""
        if self.which_distance_prior == "volume":
            return 2 * jnp.log(r) - 3 * jnp.log(self.Rmax) + jnp.log(3)
        elif self.which_distance_prior == "empirical":
            return log_prior_r_empirical(
                r, kwargs["R"], kwargs["p"], kwargs["n"], Rmax_grid=self.Rmax)
        else:
            raise ValueError(
                f"Unknown distance prior: `{self.which_distance_prior}`")

    def log_galaxy_bias(self, delta, beta):
        if self.which_bias == "linear":
            b1 = self.Om ** 0.55 / beta
            return jnp.log(jnp.clip(1 + b1 * delta, self.br_min_clip))
        elif self.which_bias == "powerlaw":
            # Neyrinck+2014 model.
            alpha = 0.65
            rho_exp = 0.4
            eps = 1.5

            x = 1 + delta
            return alpha * jnp.log(x) - (x / rho_exp)**(-eps)
        else:
            raise ValueError(
                f"Unknown galaxy bias model: `{self.which_bias}`.")

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
        sigma_v = rsample("sigma_v", self.priors["sigma_v"])
        A_covmat = rsample("A_covmat", self.priors["A_covmat"])
        beta = rsample("beta", self.priors["beta"])
        # Empirical distance prior calibration
        R_dist_emp = rsample("R_dist_emp", self.priors["R_dist_emp"])
        p_dist_emp = rsample("p_dist_emp", self.priors["p_dist_emp"])
        n_dist_emp = rsample("n_dist_emp", self.priors["n_dist_emp"])
        kwargs_dist = {"R": R_dist_emp, "p": p_dist_emp, "n": n_dist_emp}

        h = H0 / 100
        # Project Vext along the LOS to each host.
        Vext_rad_host = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)

        # HST and Gaia zero-point calibration of MW Cepheids.
        sample("M_W_HST", Normal(M_W, self.e_M_HST), obs=self.M_HST)
        sample("M_W_Gaia", Normal(M_W, self.e_M_Gaia), obs=self.M_Gaia)

        mu_host, mu_N4258, mu_LMC, mu_M31 = self.sample_host_distmod()

        # Distannce moduli for Cepheids, with corrections for LMC.
        dZP = sample("dZP", Normal(0, self.sigma_grnd))
        mu_host_cepheid = jnp.concatenate(
            [mu_host,
             jnp.array([mu_N4258, mu_LMC + dZP, mu_M31])]
            )

        # Distances moduli without any corrections.
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
            # We will add the log-likelihood of this either below or together
            # with the reconstruction likelihood.
            lp_all_host_dist = self.log_prior_distance(
                r_host_all, **kwargs_dist)
            lp_all_host_dist += self.log_grad_distmod2comoving_distance(
                mu_host_all, h=h)
            lp_host_dist = lp_all_host_dist[:self.num_hosts]

            # This one can be added already now.
            lp_anchor_dist = lp_all_host_dist[self.num_hosts:]
            factor("lp_anchor_dist", lp_anchor_dist)

        # Prepare the grid of r^2 prior of shape (eventually)
        # `(n_fields, n_galaxies, n_steps)`.
        lp_host_dist_grid = self.log_prior_distance(
            self.r_host_range, **kwargs_dist)[None, None, :]

        # Copy the homogeneous Malmquist bias for the random LOS
        if self.apply_sel:
            lp_rand_dist_grid = jnp.copy(lp_host_dist_grid)
            Vext_rad_rand = jnp.sum(Vext[None, :] * self.rhat_rand_los, axis=1)
        else:
            lp_rand_dist_grid = 0.
            Vext_rad_rand = 0.

        if self.use_reconstruction:
            lp_host_dist = lp_host_dist[None, :]
            # Compute LOS delta from reconstruction at host distances (Mpc/h),
            # shape is (n_fields, n_galaxies).
            los_delta_host = self.f_host_los_delta(rh_host)

            # Add the inhomogeneous Malmquist bias (n_fields, n_galaxies)
            lp_host_dist += self.log_galaxy_bias(los_delta_host, beta=beta)

            # Evaluate LOS overdensity over a radial grid in Mpc / h:
            # `(n_fields, n_galaxies, n_steps)``
            los_delta_grid = self.f_host_los_delta.interp_many_steps_per_galaxy(  # noqa
                self.r_host_range * h)

            # Compute integrand for normalization
            # Shape: (n_fields, n_galaxies, n_steps)
            lp_host_dist_grid += self.log_galaxy_bias(
                los_delta_grid, beta=beta)

            # Simpson integral over radial steps, per field and galaxy
            lp_host_dist_norm = ln_simpson(
                lp_host_dist_grid, x=self.r_host_range[None, None, ...],
                axis=-1)

            ll_reconstruction = lp_host_dist - lp_host_dist_norm
            lp_host_dist_grid -= lp_host_dist_norm[:, :, None]

            # Precompute the LOS distance prior and peculiar velocity for the
            # random LOS which are used to model selection.
            if self.apply_sel:
                # Evaluate the LOS density
                rand_los_delta_grid = self.f_rand_los_delta.interp_many_steps_per_galaxy(  # noqa
                    self.r_host_range * h)
                # Compute the inhomogeneous Malmquist bias
                # (previously computed homogeneous)
                lp_rand_dist_grid += self.log_galaxy_bias(
                    rand_los_delta_grid, beta=beta)
                # Compute the normalization constant
                lp_rand_dist_grid -= ln_simpson(
                    lp_rand_dist_grid, x=self.r_host_range[None, None, ...],
                    axis=-1)[..., None]

                rand_los_Vpec_grid = self.f_rand_los_velocity.interp_many_steps_per_galaxy(  # noqa
                    self.r_host_range * h)
            else:
                rand_los_Vpec_grid = 0.
        else:
            rand_los_Vpec_grid = 0.
            # Repeat the grid over all host galaxies.
            lp_host_dist_grid = jnp.repeat(
                lp_host_dist_grid, self.num_hosts, axis=1)
            # Track the distance prior already now if not using any
            # reconstruction, otherwise it is done later as it is averaged
            # together with the redshift likelihood.
            factor("lp_host_dist", lp_host_dist)

        # Selection function of shape `(n_fields, n_random_los)` calculated
        # for the *random* LOS. Average over the randoms will be taken below.
        if self.which_selection == "redshift":
            log_S = self.log_S_cz(
                lp_rand_dist_grid,
                Vext_rad_rand[None, :, None] + beta * rand_los_Vpec_grid,
                H0, sigma_v)

            if self.weight_selection_by_covmat_Neff:
                log_S *= self.Neff_PV_covmat_cepheid_host / self.num_hosts
        elif self.which_selection == "SN_magnitude":
            # Assign distance moduli to the SN hosts.
            mu_SN = self.L_SN_unique_Cepheid_host_dist @ mu_host_all
            mag_SN = mu_SN + M_B

            log_S = self.log_S_SN_mag(lp_rand_dist_grid, M_B, H0)

            if self.weight_selection_by_covmat_Neff:
                log_S *= self.Neff_C_SN_unique_Cepheid_host / self.num_hosts

            # Since the selection is in supernova apparent magnitude, must
            # constrain their absolute magnitude and thus also forward model
            # the supernova apparent magnitudes.
            factor(
                "ll_SN",
                mvn_logpdf_cholesky(
                    self.mag_SN_unique_Cepheid_host, mag_SN,
                    self.L_SN_unique_Cepheid_host)
                )

            # # If testing a diagonal covariance..
            # with plate("plate_ll_SN", self.num_hosts):
            #     sample(
            #         "ll_SN",
            #         Normal(mag_SN,
            #                np.sqrt(np.diag(self.C_SN_unique_Cepheid_host))),
            #         obs=self.mag_SN_unique_Cepheid_host)

        elif self.which_selection == "SN_magnitude_redshift":
            log_S = self.log_S_SN_mag_cz(
                lp_rand_dist_grid,
                Vext_rad_rand[None, :, None] + beta * rand_los_Vpec_grid,
                M_B, H0, sigma_v)

            # Assign distance moduli to the SN hosts.
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
            log_S = self.log_S_Cepheid_mag(
                lp_host_dist_grid, M_W, b_W, Z_W, H0)

            if self.weight_selection_by_covmat_Neff:
                log_S *= self.Neff_C_Cepheid / self.num_hosts
        else:
            log_S = jnp.zeros((1, self.num_hosts))

        # Average the selection term over the random line-of-sight.
        log_S = logsumexp(log_S, axis=-1) - jnp.log(self.num_rand_los)

        if self.use_reconstruction:
            # Subtract it per each host
            ll_reconstruction -= log_S[:, None]
        else:
            # If not using a reconstruction, can already start tracking the
            # selection function here. Since the shape is `(1, )`,
            # we can slice and factor.
            factor("neg_log_S_correction", -log_S[0] * self.num_hosts)

        # Now assign these host distances to each Cepheid.
        mu_cepheid = self.L_Cepheid_host_dist @ mu_host_cepheid

        if self.use_MNR:
            # Global hyperpriors for host-level logP mean and std
            mean_logP_all = sample(
                "mean_logP_all", Uniform(self.logP_min, self.logP_max))
            std_logP_all = sample(
                "std_logP_all", Uniform(1e-5, self.logP_max - self.logP_min))
            mean_std_logP = sample(
                "mean_std_logP", Uniform(1e-5, self.logP_max - self.logP_min))

            # Global hyperprior for host-level OH mean and std
            mean_OH_all = sample(
                "mean_OH_all", Uniform(self.OH_min, self.OH_max))
            std_OH_all = sample(
                "std_OH_all", Uniform(1e-5, self.OH_max - self.OH_min))
            mean_std_OH = sample(
                "mean_std_OH", Uniform(1e-5, self.OH_max - self.OH_min))

            # Per-host parameters
            with plate("MNR_Cepheid", len(mu_host_all)):
                mean_logP_per_host = sample(
                    "mean_logP_per_host", Normal(mean_logP_all, std_logP_all))
                std_logP_per_host = sample(
                    "std_logP_per_host", HalfNormal(mean_std_logP))

                mean_OH_per_host = sample(
                    "mean_OH_per_host", Normal(mean_OH_all, std_OH_all))
                std_OH_per_host = sample(
                    "std_OH_per_host", HalfNormal(mean_std_OH))

            # Per-Cepheid parameters
            mean_logP = self.L_Cepheid_host_dist @ mean_logP_per_host
            std_logP = self.L_Cepheid_host_dist @ std_logP_per_host

            mean_OH = self.L_Cepheid_host_dist @ mean_OH_per_host
            std_OH = self.L_Cepheid_host_dist @ std_OH_per_host

            with plate("Cepheid_true_params", self.num_cepheids):
                logP = sample(
                    "logP", Normal(mean_logP, std_logP), obs=self.logP)
                OH = sample(
                    "OH", Normal(mean_OH, std_OH), obs=self.OH)
        else:
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
            e2_cz = self.e2_czcmb_cepheid_host + sigma_v**2

            if self.use_fiducial_Cepheid_host_PV_covariance:
                cz_pred = predict_cz(z_cosmo, Vext_rad_host)
                # Because we're adding sigma_v^2 to the diagonal, we cannot
                # use the Cholesky factorization of the covariance matrix.
                C = A_covmat * self.PV_covmat_cepheid_host
                C = C.at[jnp.diag_indices(len(e2_cz))].add(e2_cz)
                sample("cz_pred", MultivariateNormal(cz_pred, C),
                       obs=self.czcmb_cepheid_host)
            elif self.use_reconstruction:
                # The reconstruction is assumed to be in Mpc / h. The shape
                # becomes `(n_fields, n_galaxies)`
                Vpec = beta * self.f_host_los_velocity(rh_host)
                Vpec += Vext_rad_host[None, :]
                cz_pred = predict_cz(z_cosmo[None, :], Vpec)
                e_cz = jnp.sqrt(e2_cz)

                ll_reconstruction += Normal(
                    cz_pred, e_cz[None, :]).log_prob(
                        self.czcmb_cepheid_host[None, :])

                # Here compute the average log-density of the Cepheid hosts,
                # averaged over the field realizations, so that the final
                # shape is `(n_galaxies,)`.
                ll_reconstruction = logsumexp(ll_reconstruction, axis=0)
                ll_reconstruction -= jnp.log(self.num_fields)
                factor("ll_reconstruction", ll_reconstruction)
            else:
                cz_pred = predict_cz(z_cosmo, Vext_rad_host)
                e_cz = jnp.sqrt(e2_cz)
                with plate("Cepheid_anchors_redshift", self.num_hosts):
                    sample("cz_pred", Normal(cz_pred, e_cz),
                           obs=self.czcmb_cepheid_host)

        # Distances to the host that have both supernovae and Cepheids
        if self.use_SNe_HF_SH0ES:
            mu_SN_Cepheid = self.L_SN_Cepheid_dist @ mu_host_all
            mag_SN_Cepheid = mu_SN_Cepheid + M_B

            Y_SN_flow = jnp.ones(self.num_flow_SN) * (M_B - 5 * jnp.log10(H0))
            Y_SN = jnp.concatenate([mag_SN_Cepheid, Y_SN_flow])
            factor(
                "ll_SN",
                mvn_logpdf_cholesky(self.Y_SN, Y_SN, self.L_SN)
                )
        elif self.use_SNe_HF_Bayes:
            raise NotImplementedError(
                "Bayesian SNe HF model is not fully implemented yet.")

            # Distances to the host that have both supernovae and Cepheids
            # and their true apparent magnitudes.
            mu_SN_Cepheid = self.L_SN_Cepheid_dist @ mu_host_all
            mag_true_SN_Cepheid = mu_SN_Cepheid + M_B

            # TODO: add a prior for this in the config. Can we use something
            # without infinite mass at zero?
            e_mu = sample("e_mu", JeffreysPrior(0.001, 0.5))

            # Sample the true apparent magnitudes of the Cepheid hosts, from
            # a r^2 prior effectively.
            # TODO: this needs to  be added/fixed
            # with plate("SN_mag", self.num_SN_HF):
            #     mag_true_HF = sample(
            #         "mag_true",
            #         MagnitudeDistribution(5, 25, self.Y_SN[77:], e_mu))
            # Replace eventually... placeholder
            mag_true_HF = 0 + e_mu

            mag_true_SN = jnp.concatenate([mag_true_SN_Cepheid, mag_true_HF])

            factor(
                "ll_SN_HF",
                mvn_logpdf_cholesky(self.Y_SN, mag_true_SN, self.L_SN)
                )

            mu_SN = mag_true_HF - M_B

            # Now the redshift likelihood
            Vext_rad_SN = jnp.sum(Vext[None, :] * self.rhat_SN_HF, axis=1)
            cz_pred_SN = predict_cz(
                self.distmod2redshift(mu_SN, h=h), Vext_rad_SN)
            e_cz = jnp.sqrt(self.e2_czcmb_SN_HF + sigma_v**2)
            with plate("SN_redshift", self.num_SN_HF):
                sample("cz_pred2", Normal(cz_pred_SN, e_cz),
                       obs=self.czcmb_SN_HF)
