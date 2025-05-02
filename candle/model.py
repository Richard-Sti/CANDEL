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

import tomllib

import jax.numpy as jnp
from numpyro import deterministic, plate, sample
from numpyro.distributions import Delta, Normal, Uniform

###############################################################################
#                              Load priors                                    #
###############################################################################


class JeffreysPrior(Uniform):
    """
    Wrapper around Uniform that keeps Uniform sampling but overrides
    log_prob to behave like a Jeffreys prior.
    """

    def log_prob(self, value):
        in_bounds = (value >= self.low) & (value <= self.high)
        return jnp.where(in_bounds, -jnp.log(value), -jnp.inf)


def load_priors(path):
    """Load a dictionary of NumPyro distributions from a TOML file."""
    with open(path, "rb") as f:
        config = tomllib.load(f)

    _DIST_MAP = {
        "normal": lambda p: Normal(p["loc"], p["scale"]),
        "uniform": lambda p: Uniform(p["low"], p["high"]),
        "delta": lambda p: Delta(p["value"]),
        "jeffreys": lambda p: JeffreysPrior(p["low"], p["high"]),
    }

    priors = {}
    for name, spec in config.items():
        dist_name = spec.pop("dist", None)
        if dist_name not in _DIST_MAP:
            raise ValueError(
                f"Unsupported distribution '{dist_name}' for '{name}'")
        priors[name] = _DIST_MAP[dist_name](spec)

    return priors


def rsample(name, dist):
    """
    Samples from `dist` unless it is a delta function, in which case returns a
    deterministic value.
    """
    if isinstance(dist, Delta):
        return deterministic(name, dist.v)
    return sample(name, dist)


###############################################################################
#                                 Models                                      #
###############################################################################


def simple_TFR_model(data, priors):
    nsamples = len(data)

    a_TFR = rsample("a_TFR", priors["TFR_zeropoint"])
    b_TFR = rsample("b_TFR", priors["TFR_slope"])
    sigma_v = rsample("sigma_v", priors["sigma_v"])

    with plate("data", nsamples):

        mu = data["mag"] - (a_TFR + b_TFR * data["eta"])

        dist = jnp.power(10, (mu - 25) / 5)

        czcmb = 100 * dist

        sample("obs", Normal(czcmb, sigma_v), obs=data["czcmb"])
