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
import tomllib
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import vmap
from jax.lax import cond
from jax.scipy.stats import norm
from numpyro import deterministic, factor, plate, sample
from numpyro.distributions import Delta, Normal, ProjectedNormal, Uniform
from numpyro.handlers import reparam
from numpyro.infer.reparam import ProjectedNormalReparam
from quadax import simpson

from .cosmography import (Distmod2Distance, Distmod2Redshift,
                          LogGrad_Distmod2ComovingDistance)
from .util import SPEED_OF_LIGHT

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


def load_priors(config_priors):
    """Load a dictionary of NumPyro distributions from a TOML file."""
    _DIST_MAP = {
        "normal": lambda p: Normal(p["loc"], p["scale"]),
        "uniform": lambda p: Uniform(p["low"], p["high"]),
        "delta": lambda p: Delta(p["value"]),
        "jeffreys": lambda p: JeffreysPrior(p["low"], p["high"]),
        "vector_uniform": lambda p: {"type": "vector_uniform", "low": p["low"], "high": p["high"],}  # noqa
    }
    priors = {}
    for name, spec in config_priors.items():
        dist_name = spec.pop("dist", None)
        if dist_name not in _DIST_MAP:
            raise ValueError(
                f"Unsupported distribution '{dist_name}' for '{name}'")
        priors[name] = _DIST_MAP[dist_name](spec)

    return priors


###############################################################################
#                           Sampling utilities                                #
###############################################################################


def rsample(name, dist):
    """
    Samples from `dist` unless it is a delta function or vector directive.
    """
    if isinstance(dist, Delta):
        return deterministic(name, dist.v)

    if isinstance(dist, dict) and dist.get("type") == "vector_uniform":
        return sample_vector(name, dist["low"], dist["high"])

    return sample(name, dist)


def norm_pmu_homogeneous(mu_TFR, sigma_mu, distmod2distance, num_points=30,
                         num_sigma=5):
    """
    Calculate the integral of `r^2 * p(mu | mu_TFR, sigma_mu)` over the r.
    There is no Jacobian because it cancels as `|dr / dmu| * dmu`.
    """
    def f(mu_tfr_i, sigma_mu_i):
        delta = num_sigma * sigma_mu_i
        mu_grid = jnp.linspace(mu_tfr_i - delta, mu_tfr_i + delta, num_points)
        r = distmod2distance(mu_grid)
        weights = r**2 * norm.pdf(mu_grid, loc=mu_tfr_i, scale=sigma_mu_i)
        return simpson(weights, x=mu_grid)

    return vmap(f)(mu_TFR, sigma_mu)


def get_muTFR(mag, eta, a_TFR, b_TFR, c_TFR=0.0):
    curvature_correction = jnp.where(eta > 0, c_TFR * eta**2, 0.0)
    return mag - (a_TFR + b_TFR * eta + curvature_correction)


def get_linear_sigma_mu_TFR(data, sigma_mu, b_TFR, c_TFR):
    return jnp.sqrt(
        data["e2_mag"]
        + (b_TFR + 2 * jnp.where(
            data["eta"] > 0, c_TFR, 0) * data["eta"]) * data["e2_eta"]
        + sigma_mu**2)


###############################################################################
#                                 Models                                      #
###############################################################################


class BaseModel(ABC):
    """Base class for all models. """

    def __init__(self, config_path):
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        self.distmod2redshift = Distmod2Redshift()
        self.distmod2distance = Distmod2Distance()
        self.log_grad_distmod2distance = LogGrad_Distmod2ComovingDistance()

        self.priors = load_priors(config["model"]["priors"])
        self.num_norm_kwargs = config["model"]["mu_norm"]

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SimpleTFRModel(BaseModel):
    """
    A simple TFR model that samples the distance modulus but fixes the true
    apparent magnitude and linewidth to the observed values.
    """

    def __call__(self, data,):
        nsamples = len(data)

        # Sample the TFR parameters.
        a_TFR = rsample("a_TFR", self.priors["TFR_zeropoint"])
        b_TFR = rsample("b_TFR", self.priors["TFR_slope"])
        c_TFR = rsample("c_TFR", self.priors["TFR_curvature"])
        sigma_mu = rsample("sigma_mu", self.priors["TFR_scatter"])

        # Sample the velocity field parameters.
        Vext = rsample("Vext", self.priors["Vext"])[None, :]
        sigma_v = rsample("sigma_v", self.priors["sigma_v"])

        with plate("data", nsamples):
            mu_TFR = get_muTFR(data["mag"], data["eta"], a_TFR, b_TFR, c_TFR)

            sigma_mu = jnp.sqrt(
                data["e2_mag"] + b_TFR**2 * data["e2_eta"] + sigma_mu**2)
            mu = sample("mu_latent", Normal(mu_TFR, sigma_mu))
            r = self.distmod2distance(mu)

            # Homogeneous Malmquist bias
            log_drdmu = self.log_grad_distmod2distance(mu)
            pmu_norm = norm_pmu_homogeneous(
                mu_TFR, sigma_mu, self.distmod2distance,
                **self.num_norm_kwargs)
            factor("mu_norm", 2 * jnp.log(r) + log_drdmu - jnp.log(pmu_norm))

            zpec = jnp.sum(data["rhat"] * Vext, axis=1) / SPEED_OF_LIGHT
            zcmb = self.distmod2redshift(mu)
            czpred = SPEED_OF_LIGHT * ((1 + zcmb) * (1 + zpec) - 1)

            sample("obs", Normal(czpred, sigma_v), obs=data["czcmb"])
