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

import jax.numpy as jnp
from jax.lax import cond
from numpyro import deterministic, plate, sample
from numpyro.distributions import Delta, Normal, ProjectedNormal, Uniform
from numpyro.handlers import reparam
from numpyro.infer.reparam import ProjectedNormalReparam

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


def load_priors(path):
    """Load a dictionary of NumPyro distributions from a TOML file."""
    with open(path, "rb") as f:
        config = tomllib.load(f)

    _DIST_MAP = {
        "normal": lambda p: Normal(p["loc"], p["scale"]),
        "uniform": lambda p: Uniform(p["low"], p["high"]),
        "delta": lambda p: Delta(p["value"]),
        "jeffreys": lambda p: JeffreysPrior(p["low"], p["high"]),
        "vector_uniform": lambda p: {"type": "vector_uniform", "low": p["low"], "high": p["high"],}  # noqa
    }
    priors = {}
    for name, spec in config.items():
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


###############################################################################
#                                 Models                                      #
###############################################################################


def simple_TFR_model(data, priors, distmod2redshift):
    nsamples = len(data)

    # Sample the TFR parameters.
    a_TFR = rsample("a_TFR", priors["TFR_zeropoint"])
    b_TFR = rsample("b_TFR", priors["TFR_slope"])

    # Sample the velocity field parameters.
    Vext = rsample("Vext", priors["Vext"])[None, :]
    sigma_v = rsample("sigma_v", priors["sigma_v"])

    with plate("data", nsamples):
        mu = data["mag"] - (a_TFR + b_TFR * data["eta"])
        zpec = jnp.sum(data["rhat"] * Vext, axis=1) / SPEED_OF_LIGHT
        zcmb = distmod2redshift(mu)
        czpred = SPEED_OF_LIGHT * ((1 + zcmb) * (1 + zpec) - 1)

        sample("obs", Normal(czpred, sigma_v), obs=data["czcmb"])
