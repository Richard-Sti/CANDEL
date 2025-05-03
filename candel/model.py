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

from .cosmography import (Distmod2Distance, Distmod2Redshift,
                          LogGrad_Distmod2ComovingDistance)
from .simpson import ln_simpson
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


def log_norm_pmu_homogeneous(mu_TFR, sigma_mu, distmod2distance, num_points=30,
                             num_sigma=5):
    """
    Calculate the integral of `r^2 * p(mu | mu_TFR, sigma_mu)` over the r.
    There is no Jacobian because it cancels as `|dr / dmu| * dmu`.
    """
    def f(mu_tfr_i, sigma_mu_i):
        delta = num_sigma * sigma_mu_i
        mu_grid = jnp.linspace(mu_tfr_i - delta, mu_tfr_i + delta, num_points)
        r_grid = distmod2distance(mu_grid)

        weights = (
            + 2 * jnp.log(r_grid)
            + norm.logpdf(mu_grid, loc=mu_tfr_i, scale=sigma_mu_i)
            )

        return ln_simpson(weights, x=r_grid, axis=-1)

    return vmap(f)(mu_TFR, sigma_mu)


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
        self.mu_grid_kwargs = config["model"]["mu_grid"]

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
            sigma_mu = get_linear_sigma_mu_TFR(data, sigma_mu, b_TFR, c_TFR)

            mu = sample("mu_latent", Normal(mu_TFR, sigma_mu))
            r = self.distmod2distance(mu)

            # Homogeneous Malmquist bias
            log_drdmu = self.log_grad_distmod2distance(mu)
            log_pmu_norm = log_norm_pmu_homogeneous(
                mu_TFR, sigma_mu, self.distmod2distance,
                **self.num_norm_kwargs)
            factor("mu_norm", 2 * jnp.log(r) + log_drdmu - log_pmu_norm)

            zpec = jnp.sum(data["rhat"] * Vext, axis=1) / SPEED_OF_LIGHT
            zcmb = self.distmod2redshift(mu)
            czpred = SPEED_OF_LIGHT * ((1 + zcmb) * (1 + zpec) - 1)

            sample("obs", Normal(czpred, sigma_v), obs=data["czcmb"])


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

        # Sample velocity field parameters.
        Vext = rsample("Vext", self.priors["Vext"])[None, :]
        sigma_v = rsample("sigma_v", self.priors["sigma_v"])

        with plate("data", nsamples):
            mu_TFR = get_muTFR(data["mag"], data["eta"], a_TFR, b_TFR, c_TFR)
            sigma_mu = get_linear_sigma_mu_TFR(data, sigma_mu, b_TFR, c_TFR)

            mu_grid = make_mu_grid(mu_TFR, **self.mu_grid_kwargs)
            r_grid = self.distmod2distance(mu_grid)

            zpec = jnp.sum(data["rhat"] * Vext, axis=1)[:, None] / SPEED_OF_LIGHT  # noqa
            zcmb = self.distmod2redshift(mu_grid)
            czpred = SPEED_OF_LIGHT * ((1 + zcmb) * (1 + zpec) - 1)

            ll = 2 * jnp.log(r_grid)
            ll += Normal(mu_TFR[:, None], sigma_mu[:, None]).log_prob(mu_grid)
            ll -= ln_simpson(ll, x=r_grid, axis=-1)[:, None]

            ll += Normal(czpred, sigma_v).log_prob(data["czcmb"][:, None])
            ll = ln_simpson(ll, x=r_grid, axis=-1)

            factor("obs", ll)



# def BIC_AIC(samples, log_likelihood, ndata):
#     """
#     Get the BIC/AIC of HMC samples from a Numpyro model.
#
#     Parameters
#     ----------
#     samples: dict
#         Dictionary of samples from the Numpyro MCMC object.
#     log_likelihood: numpy array
#         Log likelihood values of the samples.
#     ndata: int
#         Number of data points.
#
#     Returns
#     -------
#     BIC, AIC: floats
#     """
#     kmax = np.argmax(log_likelihood)
#
#     # How many parameters?
#     nparam = 0
#     for key, val in samples.items():
#         if "_deterministic" in key or "_skipZ" in key:
#             continue
#
#         if val.ndim == 1:
#             nparam += 1
#         else:
#             # The first dimension is the number of steps.
#             nparam += np.prod(val.shape[1:])
#
#     BIC = nparam * np.log(ndata) - 2 * log_likelihood[kmax]
#     AIC = 2 * nparam - 2 * log_likelihood[kmax]
#
#     return float(BIC), float(AIC)
#
#
# def dict_samples_to_array(samples, exclude_deterministic=False):
#     """Convert a dictionary of samples to a 2-dimensional array."""
#     data = []
#     names = []
#
#     for key, value in samples.items():
#         if exclude_deterministic and ("_deterministic" in key or "_skipZ" in key):  # noqa
#             continue
#
#         if value.ndim == 1:
#             data.append(value)
#             names.append(key)
#         elif value.ndim == 2:
#             for i in range(value.shape[-1]):
#                 data.append(value[:, i])
#                 names.append(f"{key}_{i}")
#         elif value.ndim == 3:
#             for i in range(value.shape[-1]):
#                 for j in range(value.shape[-2]):
#                     data.append(value[:, j, i])
#                     names.append(f"{key}_{j}_{i}")
#         else:
#             raise ValueError("Invalid dimensionality of samples to stack.")
#
#     return np.vstack(data).T, names
#
#
# def harmonic_evidence(samples, log_posterior, temperature=0.8, epochs_num=20,
#                       return_flow_samples=True, verbose=True):
#     """
#     Calculate the evidence using the `harmonic` package. The model has a few
#     more hyperparameters that are set to defaults now.
#
#     Parameters
#     ----------
#     samples: 3-dimensional array
#         MCMC samples of shape `(nchains, nsamples, ndim)`.
#     log_posterior: 2-dimensional array
#         Log posterior values of shape `(nchains, nsamples)`.
#     temperature: float, optional
#         Temperature of the `harmonic` model.
#     epochs_num: int, optional
#         Number of epochs for training the model.
#     return_flow_samples: bool, optional
#         Whether to return the flow samples.
#     verbose: bool, optional
#         Whether to print progress.
#
#     Returns
#     -------
#     ln_inv_evidence, err_ln_inv_evidence: float and tuple of floats
#         The log inverse evidence and its error.
#     flow_samples: 2-dimensional array, optional
#         Flow samples of shape `(nsamples, ndim)`. To check their agreement
#         with the input samples.
#     """
#     try:
#         import harmonic as hm
#     except ImportError:
#         raise ImportError("The `harmonic` package is required to calculate the evidence.") from None  # noqa
#
#     # Do some standard checks of inputs.
#     if samples.ndim != 3:
#         raise ValueError("The samples must be a 3-dimensional array of shape `(nchains, nsamples, ndim)`.")  # noqa
#
#     if log_posterior.ndim != 2 and log_posterior.shape[:2] != samples.shape[:2]:                             # noqa
#         raise ValueError("The log posterior must be a 2-dimensional array of shape `(nchains, nsamples)`.")  # noqa
#
#     ndim = samples.shape[-1]
#     chains = hm.Chains(ndim)
#     chains.add_chains_3d(samples, log_posterior)
#     chains_train, chains_infer = hm.utils.split_data(
#         chains, training_proportion=0.5)
#
#     # This has a few more hyperparameters that are set to defaults now.
#     model = hm.model.RQSplineModel(
#         ndim, standardize=True, temperature=temperature)
#     model.fit(chains_train.samples, epochs=epochs_num, verbose=verbose)
#
#     ev = hm.Evidence(chains_infer.nchains, model)
#     ev.add_chains(chains_infer)
#     ln_inv_evidence = ev.ln_evidence_inv
#     err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()
#
#     if return_flow_samples:
#         samples = samples.reshape((-1, ndim))
#         samp_num = samples.shape[0]
#         flow_samples = model.sample(samp_num)
#
#         return ln_inv_evidence, err_ln_inv_evidence, flow_samples
#
#     return ln_inv_evidence, err_ln_inv_evidence
#
#
# def laplace_evidence(samples, log_posterior):
#     """
#     Calculate the evidence using the Laplace approximation. Calculates
#     the mean and error of the log evidence estimated from the chains.
#
#     Parameters
#     ----------
#     samples: 3-dimensional array
#         MCMC samples of shape `(nchains, nsamples, ndim)`.
#     log_posterior: 2-dimensional array
#         Log posterior values of shape `(nchains, nsamples)`.
#
#     Returns
#     -------
#     mean_ln_inv_evidence, err_ln_inv_evidence: two floats
#     """
#     nchains = len(samples)
#     ndim = samples.shape[-1]
#     logZ = np.full(nchains, np.nan)
#
#     for n in range(nchains):
#         C = np.cov(samples[0], rowvar=False)
#         lp_max = np.max(log_posterior[n])
#
#         logZ[n] = (lp_max + 0.5 * (np.sum(np.log(np.linalg.eigvalsh(C)))
#                                    + ndim * np.log(2 * np.pi)))
#
#     return np.mean(logZ), np.std(logZ)
#
#
