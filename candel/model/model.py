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
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax.lax import cond
from jax.scipy.stats import norm
from numpyro import deterministic, factor, plate, sample
from numpyro.distributions import (Delta, MultivariateNormal, Normal,
                                   ProjectedNormal, Uniform)
from numpyro.handlers import reparam
from numpyro.infer.reparam import ProjectedNormalReparam
from quadax import simpson

from ..cosmography import (Distmod2Distance, Distmod2Redshift,
                           LogGrad_Distmod2ComovingDistance)
from ..util import SPEED_OF_LIGHT, fprint, load_config
from .simpson import ln_simpson

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


def load_priors(config_priors):
    """Load a dictionary of NumPyro distributions from a TOML file."""
    _DIST_MAP = {
        "normal": lambda p: Normal(p["loc"], p["scale"]),
        "uniform": lambda p: Uniform(p["low"], p["high"]),
        "delta": lambda p: Delta(p["value"]),
        "jeffreys": lambda p: JeffreysPrior(p["low"], p["high"]),
        "vector_uniform": lambda p: {"type": "vector_uniform", "low": p["low"], "high": p["high"]},  # noqa
        "vector_uniform_fixed": lambda p: {"type": "vector_uniform_fixed", "low": p["low"], "high": p["high"],}  # noqa
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


def rsample(name, dist):
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

    return sample(name, dist)


def log_norm_pmu(mu_TFR, sigma_mu, distmod2distance, num_points=30,
                 num_sigma=5, low_clip=20., high_clip=40., h=1.0):
    """Computation of log ∫ r^2 * p(mu | mu_TFR, sigma_mu) dr"""
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


def log_norm_pmu_im(mu_TFR, sigma_mu, alpha, distmod2distance, los_interp,
                    num_points=30, num_sigma=5, low_clip=20., high_clip=40.,
                    h=1.0):
    """Computation of log ∫ r^2 * rho * p(mu | mu_TFR, sigma_mu) dr."""
    delta = num_sigma * sigma_mu
    lo = jnp.clip(mu_TFR - delta, low_clip, high_clip)
    hi = jnp.clip(mu_TFR + delta, low_clip, high_clip)

    # shape: (n, num_points)
    mu_grid = jnp.linspace(0.0, 1.0, num_points)
    mu_grid = lo[:, None] + (hi - lo)[:, None] * mu_grid

    # These are of shape (n, num_points)
    r_grid = distmod2distance(mu_grid, h=h)
    log_rho_grid = los_interp.interp_many_steps_per_galaxy(r_grid * h)

    weights = (
        2 * jnp.log(r_grid)
        + alpha * log_rho_grid
        + norm.logpdf(mu_grid, loc=mu_TFR[:, None], scale=sigma_mu[:, None])
    )

    return jnp.where(
        lo == hi,
        0.0,
        jnp.log(simpson(jnp.exp(weights), x=r_grid, axis=-1))
        )


def make_mu_grid(mu_TFR, num_points=51, half_width=1.5, low_clip=20.,
                 high_clip=40.,):
    """Generate a grid of `mu` values centered on each `mu_TFR`"""
    lo = jnp.clip((mu_TFR - half_width).ravel(), low_clip, high_clip)
    hi = jnp.clip((mu_TFR + half_width).ravel(), low_clip, high_clip)
    return jnp.linspace(lo, hi, num_points, axis=-1)


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

        self.distmod2redshift = Distmod2Redshift()
        self.distmod2distance = Distmod2Distance()
        self.log_grad_distmod2distance = LogGrad_Distmod2ComovingDistance()

        self.priors, self.prior_dist_name = load_priors(
            config["model"]["priors"])
        self.num_norm_kwargs = config["model"]["mu_norm"]
        self.mu_grid_kwargs = config["model"]["mu_grid"]

        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


###############################################################################
#                              TFR models                                     #
###############################################################################


class SimpleTFRModel(BaseModel):
    """
    A simple TFR model that samples the distance modulus but fixes the true
    apparent magnitude and linewidth to the observed values.
    """

    def __init__(self, config_path):
        super().__init__(config_path)
        if self.config["inference"]["compute_evidence"]:
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

    def __call__(self, data,):
        nsamples = len(data)

        # Sample the TFR parameters.
        a_TFR = rsample("a_TFR_h", self.priors["TFR_zeropoint"])
        b_TFR = rsample("b_TFR", self.priors["TFR_slope"])
        c_TFR = rsample("c_TFR", self.priors["TFR_curvature"])
        sigma_mu = rsample("sigma_mu", self.priors["TFR_scatter"])

        a_TFR_dipole = rsample(
            "a_TFR_dipole", self.priors["TFR_zeropoint_dipole"])

        # Sample the velocity field parameters.
        Vext = rsample("Vext", self.priors["Vext"])[None, :]
        sigma_v = rsample("sigma_v", self.priors["sigma_v"])

        # Remainining parameters
        alpha = rsample("alpha", self.priors["alpha"])
        beta = rsample("beta", self.priors["beta"])
        h = rsample("h", self.priors["h"])

        a_TFR = deterministic("a_TFR", a_TFR + 5 * jnp.log10(h))

        with plate("data", nsamples):
            a_TFR = a_TFR + jnp.sum(a_TFR_dipole * data["rhat"], axis=1)
            mu_TFR = get_muTFR(data["mag"], data["eta"], a_TFR, b_TFR, c_TFR)
            sigma_mu = get_linear_sigma_mu_TFR(data, sigma_mu, b_TFR, c_TFR)

            mu = sample("mu_latent", Normal(mu_TFR, sigma_mu))
            r = self.distmod2distance(mu, h=h)

            # Homogeneous & inhomogeneous Malmquist bias
            log_pmu = 2 * jnp.log(r) + self.log_grad_distmod2distance(mu, h=1)
            if data.has_precomputed_los:
                # The field is in Mpc / h, so convert the distance modulus
                # back to Mpc / h in case that h != 1.
                Vrad = beta * data.f_los_velocity(r * h)

                # Inhomogeneous Malmquist bias
                log_pmu += alpha * data.f_los_log_density(r * h)
                log_pmu_norm = log_norm_pmu_im(
                    mu_TFR, sigma_mu, alpha, self.distmod2distance,
                    data.f_los_log_density, **self.num_norm_kwargs, h=h)
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


class SimpleTFRModel_DistMarg(BaseModel):
    """
    A TFR model where the distance modulus μ is integrated out using a grid,
    instead of being sampled as a latent variable.
    """

    def __call__(self, data):
        nsamples = len(data)

        # Sample the TFR parameters.
        a_TFR = rsample("a_TFR", self.priors["TFR_zeropoint"])
        b_TFR = rsample("b_TFR", self.priors["TFR_slope"])
        c_TFR = rsample("c_TFR", self.priors["TFR_curvature"])
        sigma_mu = rsample("sigma_mu", self.priors["TFR_scatter"])
        a_TFR_dipole = rsample(
            "a_TFR_dipole", self.priors["TFR_zeropoint_dipole"])

        # Sample velocity field parameters.
        Vext = rsample("Vext", self.priors["Vext"])
        sigma_v = rsample("sigma_v", self.priors["sigma_v"])

        # # Remaining parameters
        alpha = rsample("alpha", self.priors["alpha"])
        beta = rsample("beta", self.priors["beta"])

        with plate("data", nsamples):
            a_TFR = a_TFR + jnp.sum(a_TFR_dipole * data["rhat"], axis=1)
            mu_TFR = get_muTFR(data["mag"], data["eta"], a_TFR, b_TFR, c_TFR)
            sigma_mu = get_linear_sigma_mu_TFR(data, sigma_mu, b_TFR, c_TFR)

            mu_grid = make_mu_grid(mu_TFR, **self.mu_grid_kwargs)
            r_grid = self.distmod2distance(mu_grid)

            ll = 2 * jnp.log(r_grid)
            ll += Normal(mu_TFR[:, None], sigma_mu[:, None]).log_prob(mu_grid)

            if data.has_precomputed_los:
                # The shape is `(n_galaxies, num_steps.)`
                Vrad = data.f_los_velocity.interp_many_steps_per_galaxy(r_grid)
                Vrad *= beta

                log_rho = data.f_los_log_density.interp_many_steps_per_galaxy(
                    r_grid)
                ll += alpha * log_rho
            else:
                Vrad = 0.

            ll -= ln_simpson(ll, x=r_grid, axis=-1)[:, None]

            Vext_rad = jnp.sum(data["rhat"] * Vext[None, :], axis=1)

            zpec = (Vrad + Vext_rad[:, None]) / SPEED_OF_LIGHT
            zcmb = self.distmod2redshift(mu_grid)
            czpred = SPEED_OF_LIGHT * ((1 + zcmb) * (1 + zpec) - 1)

            ll += Normal(czpred, sigma_v).log_prob(data["czcmb"][:, None])
            ll = ln_simpson(ll, x=r_grid, axis=-1)

            factor("obs", ll)
