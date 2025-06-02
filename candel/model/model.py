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
"""Distance ladder and velocity field probabilistic models."""
import hashlib
import json
from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from jax import random, vmap
from jax.lax import cond
from jax.scipy.stats import norm
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from numpyro import deterministic, factor, handlers, plate, sample
from numpyro.distributions import (Delta, Distribution, MultivariateNormal,
                                   Normal, ProjectedNormal, TruncatedNormal,
                                   Uniform, constraints)
from numpyro.distributions.util import validate_sample
from numpyro.handlers import reparam
from numpyro.infer.reparam import ProjectedNormalReparam
from quadax import simpson

from ..cosmography import (Distance2Distmod, Distmod2Distance,
                           Distmod2Redshift,
                           LogAngularDiameterDistance2Distmod,
                           LogGrad_Distmod2ComovingDistance, Redshift2Distance)
from ..util import SPEED_OF_LIGHT, fprint, load_config
from .magnitude_selection import log_magnitude_selection
from .simpson import ln_simpson

###############################################################################
#                         Configuration file checks                           #
###############################################################################


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    elif hasattr(obj, 'item') and isinstance(obj.item(), (int, float, bool, str)):  # noqa
        return obj.item()
    else:
        return obj


def config_hash(cfg):
    safe_cfg = make_json_safe(cfg)
    json_str = json.dumps(safe_cfg, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


###############################################################################
#                                Priors                                       #
###############################################################################


class JeffreysPrior(Uniform):
    """
    Wrapper around Uniform that keeps Uniform sampling but overrides
    log_prob to behave like a Jeffreys prior.
    """

    def log_prob(self, value):
        in_bounds = (value >= self.low) & (value <= self.high)
        return jnp.where(in_bounds, -jnp.log(value), -jnp.inf)


class MagnitudeDistribution(Distribution):
    """
    Distribution with unnormalized log-prob ∝ 10^{0.6 x}, sampled via Normal.
    """
    reparametrized_params = ["xmin", "xmax"]
    support = constraints.real  # change to interval if you want hard clipping

    def __init__(self, xmin, xmax, mag_sample, e_mag_sample,
                 validate_args=None):
        self.xmin = xmin
        self.xmax = xmax
        self.mag_sample = mag_sample
        self.e_mag_sample = e_mag_sample
        self._lambda = 0.6 * jnp.log(10)

        batch_shape = jnp.shape(mag_sample)
        super().__init__(batch_shape=batch_shape, event_shape=(),
                         validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        u = random.normal(key, shape=sample_shape + self.batch_shape)
        return self.mag_sample + self.e_mag_sample * u

    @validate_sample
    def log_prob(self, value):
        in_bounds = (value >= self.xmin) & (value <= self.xmax)
        logp = self._lambda * value
        return jnp.where(in_bounds, logp, -jnp.inf)


def sample_vector(name, mag_min, mag_max):
    """
    Sample a 3D vector uniformly in direction and magnitude.

    NOTE: Be cautious when computing evidences or comparing models with and
    without this vector. In such cases, you *must* use `sample_vector_fixed`
    to ensure consistent prior volume handling.
    """
    # Sample unit direction vector from isotropic ProjectedNormal
    with reparam(config={f"xdir_{name}_skipZ": ProjectedNormalReparam()}):
        xdir = sample(f"xdir_{name}_skipZ", ProjectedNormal(jnp.zeros(3)))

    # Extract spherical coordinates from the unit vector
    cos_theta = deterministic(f"{name}_cos_theta", xdir[2])
    sin_theta = jnp.sqrt(1 - cos_theta**2)

    phi = jnp.arctan2(xdir[1], xdir[0])
    phi = cond(phi < 0, lambda x: x + 2 * jnp.pi, lambda x: x, phi)
    phi = deterministic(f"{name}_phi", phi)

    mag = sample(f"{name}_mag", Uniform(mag_min, mag_max))

    return mag * jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta
    ])


def sample_vector_fixed(name, mag_min, mag_max):
    """
    Sample a 3D vector but without accounting for continuity and poles.

    This enforces that all sampled points have the same contribution to
    `log_density` which is not the case for the `sample_vector` function
    because the unit vectors are drawn.
    """
    phi = sample(f"{name}_phi", Uniform(0, 2 * jnp.pi))
    cos_theta = sample(f"{name}_cos_theta", Uniform(-1, 1))
    sin_theta = jnp.sqrt(1 - cos_theta**2)

    mag = sample(f"{name}_mag", Uniform(mag_min, mag_max))

    return mag * jnp.array(
        [sin_theta * jnp.cos(phi),
         sin_theta * jnp.sin(phi),
         cos_theta]
        )


def sample_spline_radial_vector(name, nval, low, high):
    """
    Sample a radial vector approximated as a spline with `n` knots spherical
    coordinates. The magnitude is sampled uniformly and the direction is
    sampled uniformly on the unit sphere.
    """
    with plate(f"{name}_plate", nval):
        phi = sample(f"{name}_phi", Uniform(0, 2 * jnp.pi))
        cos_theta = sample(f"{name}_cos_theta", Uniform(-1, 1))
        sin_theta = jnp.sqrt(1 - cos_theta**2)

        mag = sample(f"{name}_mag", Uniform(low, high))

    return mag[:, None] * jnp.asarray([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta]).T


def interp_spline_radial_vector(rq, bin_values, **kwargs):
    """Interpolate delta radial vectors using JAX-compatible splines."""
    x = jnp.asarray(kwargs["rknot"])
    k = kwargs.get("k", 3)
    endpoints = kwargs.get("endpoints", "not-a-knot")

    def spline_eval(y):
        spline = InterpolatedUnivariateSpline(x, y, k=k, endpoints=endpoints)
        return spline(rq)

    return vmap(spline_eval)(bin_values.T)


def load_priors(config_priors):
    """Load a dictionary of NumPyro distributions from a TOML file."""
    _DIST_MAP = {
        "normal": lambda p: Normal(p["loc"], p["scale"]),
        "uniform": lambda p: Uniform(p["low"], p["high"]),
        "delta": lambda p: Delta(p["value"]),
        "jeffreys": lambda p: JeffreysPrior(p["low"], p["high"]),
        "vector_uniform": lambda p: {"type": "vector_uniform", "low": p["low"], "high": p["high"]},  # noqa
        "vector_uniform_fixed": lambda p: {"type": "vector_uniform_fixed", "low": p["low"], "high": p["high"],},  # noqa
        "vector_radial_spline_uniform": lambda p: {"type": "vector_radial_spline_uniform", "nval": len(p["rknot"]), "low": p["low"], "high": p["high"]},  # noqa
    }
    priors = {}
    prior_dist_name = {}
    for name, spec in config_priors.items():
        dist_name = spec.pop("dist", None)
        if dist_name not in _DIST_MAP:
            raise ValueError(
                f"Unsupported distribution '{dist_name}' for '{name}'")

        if dist_name == "delta":
            spec["value"] = jnp.asarray(spec["value"])

        priors[name] = _DIST_MAP[dist_name](spec)
        prior_dist_name[name] = dist_name

    return priors, prior_dist_name


###############################################################################
#                           Sampling utilities                                #
###############################################################################


def _rsample(name, dist):
    """
    Samples from `dist` unless it is a delta function or vector directive.
    """
    if name == "TFR_zeropoint_dipole" and dist.get("dist") == "delta":
        return jnp.zeros(3)

    if isinstance(dist, Delta):
        return deterministic(name, dist.v)

    if isinstance(dist, dict) and dist.get("type") == "vector_uniform":
        return sample_vector(name, dist["low"], dist["high"])

    if isinstance(dist, dict) and dist.get("type") == "vector_uniform_fixed":
        return sample_vector_fixed(name, dist["low"], dist["high"])

    if isinstance(dist, dict) and dist.get("type") == "vector_radial_spline_uniform":  # noqa
        return sample_spline_radial_vector(
            name, dist["nval"], dist["low"], dist["high"], )

    return sample(name, dist)


def rsample(name, dist, shared_params):
    """Sample a parameter from `dist`, unless provided in `shared_params`."""
    if shared_params is not None and name in shared_params:
        return shared_params[name]
    return _rsample(name, dist)


def log_norm_pmu(mu_TFR, sigma_mu, distmod2distance, num_points=30,
                 num_sigma=5, low_clip=20., high_clip=40., h=1.0):
    """
    Computation of log ∫ r^2 * p(mu | mu_TFR, sigma_mu) dr. The regular grid
    over which the integral is computed is defined uniformly in distance
    modulus.
    """
    # Get the limits in distance modulus
    delta = num_sigma * sigma_mu
    lo = jnp.clip(mu_TFR - delta, low_clip, high_clip)
    hi = jnp.clip(mu_TFR + delta, low_clip, high_clip)

    # shape: (ngalaxies, num_points)
    mu_grid = jnp.linspace(0.0, 1.0, num_points)
    # shape (n, num_points)
    mu_grid = lo[:, None] + (hi - lo)[:, None] * mu_grid
    # same shape
    r_grid = distmod2distance(mu_grid, h=h)

    weights = (
        2 * jnp.log(r_grid)
        + norm.logpdf(mu_grid, loc=mu_TFR[:, None], scale=sigma_mu[:, None])
    )

    return jnp.where(
        lo == hi,
        0.0,
        jnp.log(simpson(jnp.exp(weights), x=r_grid, axis=-1))
        )


def log_norm_pmu_im(mu_TFR, sigma_mu, bias_params, distmod2distance,
                    los_interp, galaxy_bias, num_points=30, num_sigma=5,
                    low_clip=20., high_clip=40., h=1.0):
    """
    Computation of log ∫ r^2 * rho * p(mu | mu_TFR, sigma_mu) dr. The regular
    grid over which the integral is computed is defined uniformly in distance
    modulus.
    """
    delta = num_sigma * sigma_mu
    lo = jnp.clip(mu_TFR - delta, low_clip, high_clip)
    hi = jnp.clip(mu_TFR + delta, low_clip, high_clip)

    # shape: (n, num_points)
    mu_grid = jnp.linspace(0.0, 1.0, num_points)
    mu_grid = lo[:, None] + (hi - lo)[:, None] * mu_grid

    # These are of shape (n, num_points)
    r_grid = distmod2distance(mu_grid, h=h)

    weights = (
        2 * jnp.log(r_grid)
        + norm.logpdf(mu_grid, loc=mu_TFR[:, None], scale=sigma_mu[:, None])
    )

    if galaxy_bias == "powerlaw":
        log_rho_grid = los_interp.interp_many_steps_per_galaxy(r_grid * h)
        weights += bias_params[0] * log_rho_grid
    elif galaxy_bias == "linear":
        delta_grid = los_interp.interp_many_steps_per_galaxy(r_grid * h)
        weights += jnp.log(jnp.clip(1 + bias_params[0] * delta_grid, 1e-5))
    else:
        raise ValueError(f"Invalid galaxy bias model '{galaxy_bias}'.")

    return jnp.where(
        lo == hi,
        0.0,
        jnp.log(simpson(jnp.exp(weights), x=r_grid, axis=-1))
        )


def make_mag_grid(mag, e_mag, num_sigma=3, num_points=100):
    """Make a magnitude grid for the TFR model."""
    mu_start = mag - num_sigma * e_mag
    mu_end = mag + num_sigma * e_mag
    return jnp.linspace(mu_start, mu_end, num_points).T


def get_muTFR(mag, eta, a_TFR, b_TFR, c_TFR=0.0):
    curvature_correction = jnp.where(eta > 0, c_TFR * eta**2, 0.0)
    return mag - (a_TFR + b_TFR * eta + curvature_correction)


def get_linear_sigma_mu_TFR(data, sigma_mu, b_TFR, c_TFR):
    return jnp.sqrt(
        data["e2_mag"]
        + (b_TFR + 2 * jnp.where(
            data["eta"] > 0, c_TFR, 0) * data["eta"])**2 * data["e2_eta"]
        + sigma_mu**2)


###############################################################################
#                               Base models                                   #
###############################################################################


class BaseModel(ABC):
    """Base class for all PV models. """

    def __init__(self, config_path):
        config = load_config(config_path)

        kind = config["pv_model"]["kind"]
        kind_allowed = ["Vext", "Vext_radial"]
        if kind not in kind_allowed and not kind.startswith("precomputed_los_"):  # noqa
            raise ValueError(
                f"Invalid kind '{kind}'. Must be one of {kind_allowed} or "
                "start with 'precomputed_los_'.")

        # Initialize plenty of interpolators for distance and redshift
        self.distmod2redshift = Distmod2Redshift()
        self.distmod2distance = Distmod2Distance()
        self.distance2distmod = Distance2Distmod()
        self.redshift2distance = Redshift2Distance()
        self.log_grad_distmod2distance = LogGrad_Distmod2ComovingDistance()

        priors = config["model"]["priors"]

        self.with_radial_Vext = kind == "Vext_radial"
        if self.with_radial_Vext:
            d = priors["Vext_radial"]
            fprint(f"using radial `Vext` with spline knots at {d['rknot']}")
            self.with_radial_Vext = True
            self.kwargs_radial_Vext = {
                key: d[key] for key in ["rknot", "k", "endpoints"]}
        else:
            self.with_radial_Vext = False
            self.kwargs_radial_Vext = {}

        self.priors, self.prior_dist_name = load_priors(priors)
        self.num_norm_kwargs = config["model"]["mu_norm"]
        self.mag_grid_kwargs = config["model"]["mag_grid"]

        self.use_MNR = config["pv_model"]["use_MNR"]

        self.galaxy_bias = config["pv_model"]["galaxy_bias"]
        if self.galaxy_bias not in ["powerlaw", "linear"]:
            raise ValueError(
                f"Invalid galaxy bias model '{self.galaxy_bias}'. "
                "Choose either 'powerlaw'.")

        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


def sample_galaxy_bias(priors, galaxy_bias, shared_params=None):
    """
    Sample a vector of galaxy bias parameters based on the specified model.
    """
    if galaxy_bias == "powerlaw":
        alpha = rsample("alpha", priors["alpha"], shared_params)
        bias_params = [alpha,]
    elif galaxy_bias == "linear":
        b1 = rsample("b1", priors["b1"], shared_params)
        bias_params = [b1,]
    else:
        raise ValueError(f"Invalid galaxy bias model '{galaxy_bias}'.")

    return bias_params


def lp_galaxy_bias(delta, log_rho, bias_params, galaxy_bias):
    """
    Given the galaxy bias probabibility, given some density and a bias model.
    """
    if galaxy_bias == "powerlaw":
        lp = bias_params[0] * log_rho
    elif galaxy_bias == "linear":
        lp = jnp.log(jnp.clip(1 + bias_params[0] * delta, 1e-5))
    else:
        raise ValueError(f"Invalid galaxy bias model '{galaxy_bias}'.")

    return lp


def compute_Vext_radial(data, Vext, with_radial_Vext=False, **kwargs_radial):
    """Compute the line-of-sight projection of the external velocity."""
    if with_radial_Vext:
        # Shape (3, n_rbins)
        Vext = interp_spline_radial_vector(
            data["los_r"], Vext, **kwargs_radial)

        # Shape (n_galaxies, n_rbins)
        Vext_rad = jnp.sum(
            data["rhat"][..., None] * Vext[None, ...], axis=1)
    else:
        # Shape (n_galaxies, 1)
        Vext_rad = jnp.sum(data["rhat"] * Vext[None, :], axis=1)[:, None]

    return Vext_rad


###############################################################################
#                              TFR models                                     #
###############################################################################


class TFRModel(BaseModel):
    """
    A TFR model that samples the distance modulus but fixes the true
    apparent magnitude and linewidth to the observed values.
    """

    def __init__(self, config_path,):
        super().__init__(config_path)
        if self.config["inference"]["compute_evidence"]:
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for `TFRModel`.")

        # Sample the TFR parameters.
        a_TFR = rsample("a_TFR_h", self.priors["TFR_zeropoint"], shared_params)
        b_TFR = rsample("b_TFR", self.priors["TFR_slope"], shared_params)
        c_TFR = rsample("c_TFR", self.priors["TFR_curvature"], shared_params)
        sigma_mu = rsample("sigma_mu", self.priors["sigma_mu"], shared_params)
        a_TFR_dipole = rsample(
            "a_TFR_dipole", self.priors["TFR_zeropoint_dipole"], shared_params)

        # Sample the velocity field parameters.
        if self.with_radial_Vext:
            raise NotImplementedError(
                "Radial Vext is not implemented for TFRModel.")
        else:
            Vext = rsample("Vext", self.priors["Vext"], shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)

        # Remainining parameters
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params)
        beta = rsample("beta", self.priors["beta"], shared_params)
        h = rsample("h", self.priors["h"], shared_params)

        a_TFR = deterministic("a_TFR", a_TFR + 5 * jnp.log10(h))

        if self.use_MNR:
            eta_prior_mean = sample(
                "eta_prior_mean", Uniform(data["min_eta"], data["max_eta"]))
            eta_prior_std = sample(
                "eta_prior_std", Uniform(0, data["max_eta"] - data["min_eta"]))

        with plate("data", nsamples):
            if self.use_MNR:
                # Magnitude hyperprior and selection.
                mag = sample(
                    "mag_latent",
                    MagnitudeDistribution(**data.mag_dist_kwargs,))
                if data.add_mag_selection:
                    # Magnitude selection at the true magnitude values.
                    log_Fm = log_magnitude_selection(
                        mag, **data.mag_selection_kwargs)

                    # Magnitude selection normalization.
                    mag_grid = make_mag_grid(
                        mag, data["e_mag"], **self.mag_grid_kwargs)
                    log_pmag_norm = (
                        + Normal(mag_grid, data["e_mag"][:, None]).log_prob(mag[:, None])  # noqa
                        + log_magnitude_selection(
                            mag_grid, **data.mag_selection_kwargs)
                        )

                    log_Fm -= ln_simpson(log_pmag_norm, x=mag_grid, axis=-1)
                    factor("mag_norm", log_Fm)

                sample("mag_obs", Normal(mag, data["e_mag"]), obs=data["mag"])

                # Linewidth hyperprior and selection.
                eta = sample(
                    "eta_latent", Normal(eta_prior_mean, eta_prior_std))
                if data.add_eta_truncation:
                    sample("eta_obs", TruncatedNormal(
                        eta, data["e_eta"], low=data.eta_min,
                        high=data.eta_max), obs=data["eta"])
                else:
                    sample("eta_obs", Normal(
                        eta, data["e_eta"]), obs=data["eta"])
                sigma_mu = jnp.ones_like(mag) * sigma_mu
            else:
                mag = data["mag"]
                eta = data["eta"]
                sigma_mu = get_linear_sigma_mu_TFR(
                    data, sigma_mu, b_TFR, c_TFR)

            a_TFR = a_TFR + jnp.sum(a_TFR_dipole * data["rhat"], axis=1)
            mu_TFR = get_muTFR(mag, eta, a_TFR, b_TFR, c_TFR)

            mu = sample("mu_latent", Normal(mu_TFR, sigma_mu))
            r = self.distmod2distance(mu, h=h)

            # Homogeneous & inhomogeneous Malmquist bias
            log_pmu = 2 * jnp.log(r) + self.log_grad_distmod2distance(mu, h=h)
            if data.has_precomputed_los:
                # The field is in Mpc / h, so convert the distance modulus
                # back to Mpc / h in case that h != 1.
                Vrad = beta * data.f_los_velocity(r * h)

                if self.galaxy_bias == "powerlaw":
                    alpha = bias_params[0]
                    log_pmu += alpha * data.f_los_log_density(r * h)
                    log_pmu_norm = log_norm_pmu_im(
                        mu_TFR, sigma_mu, bias_params, self.distmod2distance,
                        data.f_los_log_density, self.galaxy_bias,
                        **self.num_norm_kwargs, h=h)
                elif self.galaxy_bias == "linear":
                    b1 = bias_params[0]
                    log_pmu += jnp.log(
                        jnp.clip(1 + b1 * data.f_los_delta(r * h), 1e-5))
                    log_pmu_norm = log_norm_pmu_im(
                        mu_TFR, sigma_mu, bias_params, self.distmod2distance,
                        data.f_los_delta, self.galaxy_bias,
                        **self.num_norm_kwargs, h=h)
                else:
                    raise ValueError(
                        f"Invalid galaxy bias model '{self.galaxy_bias}'.")

            else:
                Vrad = 0.
                # Homogeneous Malmquist bias
                log_pmu_norm = log_norm_pmu(
                    mu_TFR, sigma_mu, self.distmod2distance,
                    **self.num_norm_kwargs, h=h)

            factor("mu_norm", log_pmu - log_pmu_norm)

            Vext_rad = jnp.sum(data["rhat"] * Vext, axis=1)
            zpec = (Vrad + Vext_rad) / SPEED_OF_LIGHT
            zcmb = self.distmod2redshift(mu, h=h)
            czpred = SPEED_OF_LIGHT * ((1 + zcmb) * (1 + zpec) - 1)

            sample("obs", Normal(czpred, sigma_v), obs=data["czcmb"])

        if data.num_calibrators > 0:
            mu_calibration = mu[data["is_calibrator"]]
            with plate("calibrators", len(mu_calibration)):
                sample(
                    "mu_cal",
                    MultivariateNormal(mu_calibration, data["C_mu_cal"]),
                    obs=data["mu_cal"])


class TFRModel_DistMarg(BaseModel):
    """
    A TFR model where the distance modulus μ is integrated out using a grid,
    instead of being sampled as a latent variable.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if self.use_MNR:
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        # Sample the TFR parameters.
        a_TFR = rsample("a_TFR", self.priors["TFR_zeropoint"], shared_params)
        b_TFR = rsample("b_TFR", self.priors["TFR_slope"], shared_params)
        c_TFR = rsample("c_TFR", self.priors["TFR_curvature"], shared_params)
        sigma_mu = rsample("sigma_mu", self.priors["sigma_mu"], shared_params)
        a_TFR_dipole = rsample(
            "a_TFR_dipole", self.priors["TFR_zeropoint_dipole"], shared_params)

        if data.sample_dust:
            Rdust = rsample("R_dust", self.priors["Rdust"], shared_params)
            Ab = Rdust * data["ebv"]
        else:
            Ab = 0.

        # Sample velocity field parameters.
        if self.with_radial_Vext:
            Vext = rsample(
                "Vext_rad", self.priors["Vext_radial"], shared_params)
        else:
            Vext = rsample("Vext", self.priors["Vext"], shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)

        # Remaining parameters
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params)
        beta = rsample("beta", self.priors["beta"], shared_params)

        # For the distance marginalization, h is not sampled.
        h = 1.

        if self.use_MNR:
            eta_prior_mean = sample(
                "eta_prior_mean", Uniform(data["min_eta"], data["max_eta"]))
            eta_prior_std = sample(
                "eta_prior_std", Uniform(0, data["max_eta"] - data["min_eta"]))

        with plate("data", nsamples):
            if self.use_MNR:
                # Magnitude hyperprior and selection, note the optional dust
                # correction.
                mag = sample(
                    "mag_latent",
                    MagnitudeDistribution(**data.mag_dist_kwargs,))

                if data.add_mag_selection:
                    # Magnitude selection at the true magnitude values.
                    log_Fm = log_magnitude_selection(
                        mag, **data.mag_selection_kwargs)

                    # Magnitude selection normalization.
                    mag_grid = make_mag_grid(
                        mag, data["e_mag"], **self.mag_grid_kwargs)
                    log_pmag_norm = (
                        + Normal(mag_grid, data["e_mag"][:, None]).log_prob(mag[:, None])  # noqa
                        + log_magnitude_selection(
                            mag_grid, **data.mag_selection_kwargs)
                        )

                    log_Fm -= ln_simpson(log_pmag_norm, x=mag_grid, axis=-1)
                    factor("mag_norm", log_Fm)

                # Correct for Milky Way extinction by subtracting Ab. If Ab
                # is nonzero, the data is assumed to be uncorrected.
                # This correction is applied only when comparing to observed
                # values (MNR parameters are sampled from an isotropic).
                sample(
                    "mag_obs",
                    Normal(mag + Ab, data["e_mag"]), obs=data["mag"])

                # Linewidth hyperprior and selection.
                eta = sample(
                    "eta_latent", Normal(eta_prior_mean, eta_prior_std))
                if data.add_eta_truncation:
                    sample("eta_obs", TruncatedNormal(
                        eta, data["e_eta"], low=data.eta_min,
                        high=data.eta_max), obs=data["eta"])
                else:
                    sample("eta_obs", Normal(
                        eta, data["e_eta"]), obs=data["eta"])

                sigma_mu = jnp.ones_like(mag) * sigma_mu
            else:
                mag = data["mag"] - Ab
                eta = data["eta"]
                sigma_mu = get_linear_sigma_mu_TFR(
                    data, sigma_mu, b_TFR, c_TFR)

            a_TFR = a_TFR + jnp.sum(a_TFR_dipole * data["rhat"], axis=1)
            mu_TFR = get_muTFR(mag, eta, a_TFR, b_TFR, c_TFR)

            r_grid = data["los_r"][None, :] / h
            mu_grid = self.distance2distmod(r_grid, h=h)

            ll = 2 * jnp.log(r_grid)
            ll += Normal(mu_TFR[:, None], sigma_mu[:, None]).log_prob(mu_grid)

            if data.has_precomputed_los:
                # The shape is `(n_galaxies, num_steps.)`
                Vrad = beta * data["los_velocity"]
                ll += lp_galaxy_bias(
                    data["los_delta"], data["los_log_density"], bias_params,
                    self.galaxy_bias)
            else:
                Vrad = 0.

            ll_norm = ln_simpson(ll, x=r_grid, axis=-1)

            Vext_rad = compute_Vext_radial(
                data, Vext, with_radial_Vext=self.with_radial_Vext,
                **self.kwargs_radial_Vext)
            zpec = (Vrad + Vext_rad) / SPEED_OF_LIGHT
            zcmb = self.distmod2redshift(mu_grid, h=h)
            czpred = SPEED_OF_LIGHT * ((1 + zcmb) * (1 + zpec) - 1)

            ll += Normal(czpred, sigma_v).log_prob(data["czcmb"][:, None])
            ll = ln_simpson(ll, x=r_grid, axis=-1) - ll_norm

            factor("obs", ll)


###############################################################################
#                           Supernova models                                 #
###############################################################################


class PantheonPlusModel_DistMarg(BaseModel):
    """
    Pantheon+ model with numerical distance marginalisation. Uses the
    precomputed magnitude covariance matrix.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if not self.use_MNR:
            raise ValueError(
                "The PantheonPlus model requires the MNR model to be used. "
                "Please set `use_MNR` to True in the config file.")

        fprint("setting `compute_evidence` to False.")
        self.config["inference"]["compute_evidence"] = False

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for `PantheonPlusModel`.")

        # Sample the SN parameters.
        M = rsample("M", self.priors["SN_absmag"], shared_params)
        dM = rsample(
            "M_dipole", self.priors["SN_absmag_dipole"], shared_params)
        sigma_mu = rsample("sigma_mu", self.priors["sigma_mu"], shared_params)

        # Sample velocity field parameters.
        if self.with_radial_Vext:
            Vext = rsample(
                "Vext_rad", self.priors["Vext_radial"], shared_params)
        else:
            Vext = rsample("Vext", self.priors["Vext"], shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)

        # Remaining parameters
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params)
        beta = rsample("beta", self.priors["beta"], shared_params)

        # For the distance marginalization, h is not sampled.
        h = 1.

        with plate("data", nsamples):
            # Magnitude hyperprior and selection.
            mag = sample("mag_latent",
                         MagnitudeDistribution(**data.mag_dist_kwargs,))
            # This can be sped up using decomposition.
            sample(
                "mag_obs", MultivariateNormal(mag, data["mag_covmat"]),
                obs=data["mag"])
            sigma_mu = jnp.ones_like(mag) * sigma_mu

            M = M + jnp.sum(dM * data["rhat"], axis=1)
            mu_SN = mag - M

            r_grid = data["los_r"][None, :] / h
            mu_grid = self.distance2distmod(r_grid, h=h)

            ll = 2 * jnp.log(r_grid)
            ll += Normal(mu_SN[:, None], sigma_mu[:, None]).log_prob(mu_grid)

            if data.has_precomputed_los:
                # The shape is `(n_galaxies, num_steps.)`
                Vrad = beta * data["los_velocity"]
                ll += lp_galaxy_bias(
                    data["los_delta"], data["los_log_density"], bias_params,
                    self.galaxy_bias)
            else:
                Vrad = 0.

            ll_norm = ln_simpson(ll, x=r_grid, axis=-1)

            Vext_rad = compute_Vext_radial(
                data, Vext, with_radial_Vext=self.with_radial_Vext,
                **self.kwargs_radial_Vext)
            zpec = (Vrad + Vext_rad) / SPEED_OF_LIGHT
            zcmb = self.distmod2redshift(mu_grid, h=h)
            czpred = SPEED_OF_LIGHT * ((1 + zcmb) * (1 + zpec) - 1)

            ll += Normal(czpred, sigma_v).log_prob(data["czcmb"][:, None])
            ll = ln_simpson(ll, x=r_grid, axis=-1) - ll_norm

            factor("obs", ll)


###############################################################################
#                           Clusters models                                 #
###############################################################################


def get_Ez(z, Om):
    """
    Compute the E(z) function for a flat universe with matter density Om.
    """
    return jnp.sqrt(Om * (1 + z)**3 + (1 - Om))


def mu_from_CL_calibration(theta, c, log_dL_grid, log_dA_grid, mu_grid):
    """
    Solve for which distance modulus solves the equation
    `log_dL - c * log_dA = theta`. Takes advantage of pre-computing `mu`
    and `log_dL` and `log_dA` grids.
    """

    y = log_dL_grid - c * log_dA_grid
    res = y - theta
    return jnp.interp(0.0, res, mu_grid)


class Clusters_DistMarg(BaseModel):
    """
    Cluster L-T-Y scaling relation peculiar velocity model with distance
    marginalization.
    """

    def __init__(self, config_path):
        super().__init__(config_path)

        which_relation = self.config["io"]["Clusters"]["which_relation"]
        if which_relation not in ["LT", "LY", "LTY", "YT"]:
            raise ValueError(f"Invalid scaling relation '{which_relation}'. "
                             "Choose either 'LT' or 'LY' or 'LTY'.")

        self.which_relation = which_relation
        self.sample_T = "T" in which_relation
        self.sample_Y = "Y" in which_relation
        self.sample_F = "L" in which_relation

        # Later make this choice more flexible.
        self.Om = 0.3

        # Disable priors only if the variable is not used in the relation.
        used = set(which_relation)
        prior_info = {
            "T": ("sample_T", "logT", "CL_B"),
            "Y": ("sample_Y", "logY", "CL_C"),
        }

        for var, (attr, name, key) in prior_info.items():
            if var not in used and not getattr(self, attr, False):
                fprint(
                    f"`{name}` is not used in the model. Disabling its prior.")
                self.priors[key] = Delta(jnp.asarray(0.0))
                self.prior_dist_name[key] = "delta"

        if self.use_MNR:
            raise NotImplementedError(
                "MNR for clusters is not implemented yet. Please set "
                "`use_MNR` to False in the config file.")
            # fprint("setting `compute_evidence` to False.")
            # self.config["inference"]["compute_evidence"] = False

        # If C is being sampled, then we need to precompute the grids to
        # relate `logDL - c * logDA = theta` to `mu`. Note that these are
        # h = 1 and with a hard-cded Om = 0.3 cosmology.
        if which_relation[0] == 'L':
            d = self.config["io"]["Clusters"]
            z_grid = np.linspace(
                d["zcmb_mapping_min"], d["zcmb_mapping_max"],
                d["zcmb_mapping_num_points"])
            fprint("precomputing the distance grid for z in "
                   f"[{d['zcmb_mapping_min']}, {d['zcmb_mapping_max']}] with "
                   f"{d['zcmb_mapping_num_points']} points.")

            cs = FlatLambdaCDM(H0=100, Om0=self.Om)
            mu_grid = cs.distmod(z_grid).value
            log_dA_grid = jnp.log10(cs.angular_diameter_distance(z_grid).value)
            log_dL_grid = jnp.log10(cs.luminosity_distance(z_grid).value)
            mu_grid = jnp.asarray(mu_grid)

            f_partial = partial(
                mu_from_CL_calibration, log_dL_grid=log_dL_grid,
                log_dA_grid=log_dA_grid, mu_grid=mu_grid)
            self.mu_from_LTY_calibration = vmap(
                f_partial, in_axes=(0, None), out_axes=0)

        if which_relation == "YT":
            self.logangdist2distmod = LogAngularDiameterDistance2Distmod()

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for `TFRModel`.")

        # Sample the cluster scaling parameters.
        A = rsample("A_CL", self.priors["CL_A"], shared_params)
        B = rsample("B_CL", self.priors["CL_B"], shared_params)
        C = rsample("C_CL", self.priors["CL_C"], shared_params)
        sigma_mu = rsample("sigma_mu", self.priors["sigma_mu"], shared_params)

        # Sample velocity field parameters.
        if self.with_radial_Vext:
            Vext = rsample(
                "Vext_rad", self.priors["Vext_radial"], shared_params)
        else:
            Vext = rsample("Vext", self.priors["Vext"], shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)
        # Remaining parameters
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params)
        beta = rsample("beta", self.priors["beta"], shared_params)

        # For the distance marginalization, h is not sampled. Careful because
        # the mapping to `mu` above assumes h = 1.
        h = 1.

        with plate("data", nsamples):
            if self.use_MNR:
                raise NotImplementedError(
                    "MNR for clusters is not implemented yet. Please set "
                    "`use_MNR` to False in the config file.")
            else:
                sigma_mu2 = jnp.ones(nsamples) * sigma_mu**2
                rel = self.which_relation[0]

                # Fixed contributions depending on relation type
                if rel == "L":
                    # LogL = A + B * logT + C * logY
                    logF = data["logF"]
                    sigma_mu2 += data["e2_logF"]
                elif rel == "Y":
                    # logY = A + B * logT + C * logF
                    logY = data["logY"]
                    sigma_mu2 += data["e2_logY"]
                else:
                    raise ValueError(
                        f"Invalid scaling relation '{self.which_relation}'.")

                # Conditional contributions based on sampling flags
                if self.sample_T:
                    logT = data["logT"]
                    sigma_mu2 += B**2 * data["e2_logT"]

                if self.sample_Y and rel == "L":
                    logY = data["logY"]
                    sigma_mu2 += C**2 * data["e2_logY"]

                if self.sample_F and rel == "Y":
                    logF = data["logF"]
                    sigma_mu2 += C**2 * data["e2_logF"]

                sigma_mu = jnp.sqrt(sigma_mu2)

            # This should depend on the cosmological redshift, but the
            # corrections are small and subdominant to the noise in the cluster
            # scaling relations.
            Ez = get_Ez(data["zcmb"], Om=self.Om)

            if self.which_relation == "LT":
                logL_pred = jnp.log10(Ez) + A + B * logT
                mu_cluster = 2.5 * (logL_pred - logF) + 25
            elif self.which_relation[0] == "L":
                theta = 0.5 * (jnp.log10(Ez) + A + B * logT + C * logY - logF)
                mu_cluster = self.mu_from_LTY_calibration(theta, C)
            elif self.which_relation == "YT":
                logDA = 0.5 * (jnp.log10(Ez) + A + B * logT - logY)
                mu_cluster = self.logangdist2distmod(logDA, h=h)
            else:
                raise ValueError(
                    f"Invalid scaling relation '{self.which_relation}'.")

            # From now on it is standard calculations.
            r_grid = data["los_r"][None, :] / h
            mu_grid = self.distance2distmod(r_grid, h=h)

            ll = 2 * jnp.log(r_grid)
            ll += Normal(
                mu_cluster[:, None], sigma_mu[:, None]).log_prob(mu_grid)

            if data.has_precomputed_los:
                # The shape is `(n_galaxies, num_steps.)`
                Vrad = beta * data["los_velocity"]
                ll += lp_galaxy_bias(
                    data["los_delta"], data["los_log_density"], bias_params,
                    self.galaxy_bias)
            else:
                Vrad = 0.

            ll_norm = ln_simpson(ll, x=r_grid, axis=-1)

            Vext_rad = compute_Vext_radial(
                data, Vext, with_radial_Vext=self.with_radial_Vext,
                **self.kwargs_radial_Vext)
            zpec = (Vrad + Vext_rad) / SPEED_OF_LIGHT
            zcmb = self.distmod2redshift(mu_grid, h=h)
            czpred = SPEED_OF_LIGHT * ((1 + zcmb) * (1 + zpec) - 1)

            ll += Normal(czpred, sigma_v).log_prob(data["czcmb"][:, None])
            ll = ln_simpson(ll, x=r_grid, axis=-1) - ll_norm

            factor("obs", ll)


###############################################################################
#                                FP models                                    #
###############################################################################


class FPModel_DistMarg(BaseModel):
    """
    A FP model where the distance modulus μ is integrated out using a grid,
    instead of being sampled as a latent variable.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if self.use_MNR:
            raise RuntimeError(
                "MNR for FP is not implemented yet. Please set "
                "`use_MNR` to False in the config file.")
            # fprint("setting `compute_evidence` to False.")
            # self.config["inference"]["compute_evidence"] = False

        self.logangdist2distmod = LogAngularDiameterDistance2Distmod()

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for `TFRModel`.")

        # Sample the TFR parameters.
        a_FP = rsample("a_FP", self.priors["FP_a"], shared_params)
        b_FP = rsample("b_FP", self.priors["FP_b"], shared_params)
        c_FP = rsample("c_FP", self.priors["FP_c"], shared_params)
        sigma_mu = rsample("sigma_mu", self.priors["sigma_mu"], shared_params)

        # Sample velocity field parameters.
        if self.with_radial_Vext:
            Vext = rsample(
                "Vext_rad", self.priors["Vext_radial"], shared_params)
        else:
            Vext = rsample("Vext", self.priors["Vext"], shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)

        # Remaining parameters
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params)
        beta = rsample("beta", self.priors["beta"], shared_params)

        # For the distance marginalization, h is not sampled.
        h = 1.

        if self.use_MNR:
            raise NotImplementedError("MNR for FP is not implemented yet.")

        with plate("data", nsamples):
            if self.use_MNR:
                raise NotImplementedError("MNR for FP is not implemented yet.")
            else:
                logs = data["logs"]
                logI = data["logI"]
                logtheta = data["log_theta_eff"]
                sigma_mu = jnp.ones_like(logs) * sigma_mu

            logdA = a_FP * logs + b_FP * logI + c_FP - logtheta - 3
            mu_FP = self.logangdist2distmod(logdA, h=h)

            r_grid = data["los_r"][None, :] / h
            mu_grid = self.distance2distmod(r_grid, h=h)

            ll = 2 * jnp.log(r_grid)
            ll += Normal(mu_FP[:, None], sigma_mu[:, None]).log_prob(mu_grid)

            if data.has_precomputed_los:
                # The shape is `(n_galaxies, num_steps.)`
                Vrad = beta * data["los_velocity"]
                ll += lp_galaxy_bias(
                    data["los_delta"], data["los_log_density"], bias_params,
                    self.galaxy_bias)
            else:
                Vrad = 0.

            ll_norm = ln_simpson(ll, x=r_grid, axis=-1)

            Vext_rad = compute_Vext_radial(
                data, Vext, with_radial_Vext=self.with_radial_Vext,
                **self.kwargs_radial_Vext)
            zpec = (Vrad + Vext_rad) / SPEED_OF_LIGHT
            zcmb = self.distmod2redshift(mu_grid, h=h)

            czpred = SPEED_OF_LIGHT * ((1 + zcmb) * (1 + zpec) - 1)

            ll += Normal(czpred, sigma_v).log_prob(data["czcmb"][:, None])
            ll = ln_simpson(ll, x=r_grid, axis=-1) - ll_norm

            factor("obs", ll)


###############################################################################
#                               Joint model                                   #
###############################################################################

class JointPVModel:
    """
    A joint probabilistic velocity (PV) model that runs multiple submodels
    (e.g., TFR models) on independent datasets, while sharing a subset of
    parameters across all submodels.
    """

    def __init__(self, submodels, shared_param_names):
        self.submodels = submodels
        self.shared_param_names = shared_param_names

        # Check that all submodels have the same config.
        ref_hash = config_hash(submodels[0].config)
        for i, model in enumerate(submodels[1:], start=1):
            if config_hash(model.config) != ref_hash:
                raise ValueError(f"Submodel {i} has a different config hash.")

        self.config = submodels[0].config
        self.with_radial_Vext = submodels[0].with_radial_Vext

    def _sample_shared_params(self, priors):
        shared = {}
        for name in self.shared_param_names:
            shared[name] = _rsample(name, priors[name])
        return shared

    def __call__(self, data):
        assert len(data) == len(self.submodels)
        shared_params = self._sample_shared_params(self.submodels[0].priors)

        for i, (submodel, data_i) in enumerate(zip(self.submodels, data)):
            name = data_i.name if data_i is not None else f"dataset_{i}"
            with handlers.scope(prefix=name):
                submodel(data_i, shared_params=shared_params)
