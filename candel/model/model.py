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
"""
TFR, FP, SNe ... forward models (typically no absolute distance calibration).
"""
import hashlib
import json
from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from jax import vmap
from jax.debug import print as jprint  # noqa
from jax.lax import cond
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammainc, gammaln, logsumexp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from numpyro import deterministic, factor, handlers, plate, sample
from numpyro.distributions import (Delta, MultivariateNormal, Normal,
                                   ProjectedNormal, TruncatedNormal, Uniform)
from numpyro.handlers import reparam
from numpyro.infer.reparam import ProjectedNormalReparam

from ..cosmography import (Distance2Distmod, Distance2Distmod_withOm,
                           Distance2LogAngDist, Distance2Redshift,
                           Distance2Redshift_withOm,
                           LogAngularDiameterDistance2Distmod)
from ..util import SPEED_OF_LIGHT, fprint, get_nested, load_config
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
#                            Useful functions                                 #
###############################################################################


def predict_cz(zcosmo, Vrad):
    return SPEED_OF_LIGHT * ((1 + zcosmo) * (1 + Vrad / SPEED_OF_LIGHT) - 1)


def mvn_logpdf_cholesky(y, mu, L):
    """
    Log-pdf of a multivariate normal using Cholesky factor L (lower
    triangular).
    """
    z = solve_triangular(L, y - mu, lower=True)
    log_det = jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (len(y) * jnp.log(2 * jnp.pi) + 2 * log_det + jnp.dot(z, z))


###############################################################################
#                                Priors                                       #
###############################################################################

def log_prior_r_empirical(r, R, p, n, Rmax_grid, Rmax_truncate=None):
    """
    Log of the (empirical) truncated prior:
        π(r) ∝ r^p * exp(-(r/R)^n),   0 < r ≤ Rmax
    Normalized by Z = [R^(1+p) * γ(a, x)] / n with a = (1+p)/n, x = (Rmax/R)^n
    """
    if Rmax_truncate is None:
        Rmax = Rmax_grid
    else:
        Rmax = jnp.minimum(Rmax_grid, Rmax_truncate)

    a = (1.0 + p) / n
    x = (Rmax / R) ** n

    # log γ(a, x) = log Γ(a) + log P(a, x), P = regularized lower γ
    log_gamma_lower = (
        gammaln(a) + jnp.log(jnp.clip(gammainc(a, x), 1e-300, 1.0)))
    log_norm = (1.0 + p) * jnp.log(R) - jnp.log(n) + log_gamma_lower

    logpdf = p * jnp.log(r) - (r / R)**n - log_norm
    valid = (r > 0) & (r <= Rmax)
    return jnp.where(valid, logpdf, -jnp.inf)


class JeffreysPrior(Uniform):
    """
    Wrapper around Uniform that keeps Uniform sampling but overrides
    log_prob to behave like a Jeffreys prior.

    Sometimes this is also called a reference prior, or a scale-invariant
    prior.
    """

    def log_prob(self, value):
        in_bounds = (value >= self.low) & (value <= self.high)
        return jnp.where(in_bounds, -jnp.log(value), -jnp.inf)


def sample_vector_components_uniform(name, low, high):
    """
    Sample a 3D vector by drawing each Cartesian component independently
    from a uniform distribution over [xmin, xmax].
    """
    x = sample(f"{name}_x", Uniform(low, high))
    y = sample(f"{name}_y", Uniform(low, high))
    z = sample(f"{name}_z", Uniform(low, high))
    return jnp.array([x, y, z])


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
        "truncated_normal": lambda p: TruncatedNormal(p["mean"], p["scale"], low=p.get("low", None), high= p.get("high", None)),  # noqa
        "uniform": lambda p: Uniform(p["low"], p["high"]),
        "delta": lambda p: Delta(p["value"]),
        "jeffreys": lambda p: JeffreysPrior(p["low"], p["high"]),
        "vector_uniform": lambda p: {"type": "vector_uniform", "low": p["low"], "high": p["high"]},  # noqa
        "vector_uniform_fixed": lambda p: {"type": "vector_uniform_fixed", "low": p["low"], "high": p["high"],},  # noqa
        "vector_radial_spline_uniform": lambda p: {"type": "vector_radial_spline_uniform", "nval": len(p["rknot"]), "low": p["low"], "high": p["high"]},  # noqa
        "vector_components_uniform": lambda p: {"type": "vector_components_uniform", "low": p["low"], "high": p["high"],},  # noqa
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
    if name == "zeropoint_dipole" and dist.get("dist") == "delta":
        return jnp.zeros(3)

    if isinstance(dist, Delta):
        return deterministic(name, dist.v)

    if isinstance(dist, dict) and dist.get("type") == "vector_uniform":
        return sample_vector(name, dist["low"], dist["high"])

    if isinstance(dist, dict) and dist.get("type") == "vector_uniform_fixed":
        return sample_vector_fixed(name, dist["low"], dist["high"])

    if isinstance(dist, dict) and dist.get("type") == "vector_components_uniform":  # noqa
        return sample_vector_components_uniform(
            name, dist["low"], dist["high"])

    if isinstance(dist, dict) and dist.get("type") == "vector_radial_spline_uniform":  # noqa
        return sample_spline_radial_vector(
            name, dist["nval"], dist["low"], dist["high"], )

    return sample(name, dist)


def rsample(name, dist, shared_params=None):
    """Sample a parameter from `dist`, unless provided in `shared_params`."""
    if shared_params is not None and name in shared_params:
        return shared_params[name]
    return _rsample(name, dist)


def get_absmag_TFR(eta, a_TFR, b_TFR, c_TFR=0.0):
    return a_TFR + b_TFR * eta + jnp.where(eta > 0, c_TFR * eta**2, 0.0)


def get_linear_sigma_mag_TFR(data, sigma_int, b_TFR, c_TFR):
    return jnp.sqrt(
        data["e2_mag"]
        + (b_TFR + 2 * jnp.where(
            data["eta"] > 0, c_TFR, 0) * data["eta"])**2 * data["e2_eta"]
        + sigma_int**2)


def make_adaptive_grid(x_obs, e_x, k_sigma, n_grid):
    """
    Construct an adaptive uniform grid for a latent variable x.

    The grid is centred on the observed value x_obs with extent
    ± k_sigma * e_x, where e_x is the measurement uncertainty, and the output
    shape is `(n_obj, n_grid)`.
    """
    half_span = k_sigma * e_x
    a = x_obs - half_span
    b = x_obs + half_span
    eps = jnp.finfo(jnp.float32).eps
    b = jnp.maximum(b, a + eps)
    t = jnp.linspace(0.0, 1.0, n_grid)
    return a[:, None] + (b - a)[:, None] * t[None, :]


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

        # Initialize interpolators for distance and redshift
        self.Om = get_nested(config, "model/Om", 0.3)
        self.distance2distmod = Distance2Distmod(Om0=self.Om)
        self.distance2redshift = Distance2Redshift(Om0=self.Om)

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
        self.use_MNR = get_nested(config, "model/use_MNR", False)
        self.marginalize_eta = get_nested(
            config, "model/marginalize_eta", True)
        if self.marginalize_eta:
            self.eta_grid_kwargs = config["model"]["eta_grid"]
            fprint(
                "marginalizing eta with "
                f"k_sigma = {self.eta_grid_kwargs['k_sigma']} and "
                f"n_grid = {self.eta_grid_kwargs['n_grid']}.")

        self.galaxy_bias = config["pv_model"]["galaxy_bias"]
        if self.galaxy_bias not in ["powerlaw", "linear", "linear_from_beta"]:
            raise ValueError(
                f"Invalid galaxy bias model '{self.galaxy_bias}'.")

        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


def sample_galaxy_bias(priors, galaxy_bias, shared_params=None, **kwargs):
    """
    Sample a vector of galaxy bias parameters based on the specified model.
    """
    if galaxy_bias == "powerlaw":
        alpha = rsample("alpha", priors["alpha"], shared_params)
        bias_params = [alpha,]
    elif galaxy_bias == "linear":
        b1 = rsample("b1", priors["b1"], shared_params)
        bias_params = [b1,]
    elif galaxy_bias == "linear_from_beta":
        b1 = kwargs["Om"]**0.55 / kwargs["beta"]
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
    elif galaxy_bias in ["linear", "linear_from_beta"]:
        lp = jnp.log(jnp.clip(1 + bias_params[0] * delta, 1e-5))
    else:
        raise ValueError(f"Invalid galaxy bias model '{galaxy_bias}'.")

    return lp


def compute_Vext_radial(data, r_grid, Vext, with_radial_Vext=False,
                        **kwargs_radial):
    """
    Compute the line-of-sight projection of the external velocity.

    Promote the final output to shape `(n_field, n_gal, n_rbins)`.
    """
    if with_radial_Vext:
        # Shape (3, n_rbins)
        Vext = interp_spline_radial_vector(r_grid, Vext, **kwargs_radial)

        Vext_rad = jnp.sum(
            data["rhat"][..., None] * Vext[None, ...], axis=1)[None, ...]
    else:
        Vext_rad = jnp.sum(data["rhat"] * Vext[None, :], axis=1)[None, :, None]

    return Vext_rad


###############################################################################
#                              TFR models                                     #
###############################################################################

class TFRModel(BaseModel):
    """
    A TFR forward model, distance is numerically marginalized out at each MCMC
    step instead of being sampled as a latent variable.
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
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)
        a_TFR_dipole = rsample(
            "zeropoint_dipole", self.priors["zeropoint_dipole"], shared_params)
        a_TFR = a_TFR + jnp.sum(a_TFR_dipole * data["rhat"], axis=1)

        # For the distance marginalization, h is not sampled.
        h = 1.

        R_dist_emp = rsample("R_dist_emp", self.priors["R_dist_emp"])
        p_dist_emp = rsample("p_dist_emp", self.priors["p_dist_emp"])
        n_dist_emp = rsample("n_dist_emp", self.priors["n_dist_emp"])
        kwargs_dist = {"R": R_dist_emp, "p": p_dist_emp, "n": n_dist_emp}

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
        beta = rsample("beta", self.priors["beta"], shared_params)
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params, Om=self.Om,
            beta=beta)

        if self.use_MNR:
            eta_prior_mean = sample(
                "eta_prior_mean", Uniform(data["min_eta"], data["max_eta"]))
            eta_prior_std = sample(
                "eta_prior_std", Uniform(0, data["max_eta"] - data["min_eta"]))

        with plate("data", nsamples):
            if self.use_MNR:
                if self.marginalize_eta:
                    # 2D grid of eta, `(n_gal, n_eta_grid)`
                    eta_grid = make_adaptive_grid(
                        data["eta"], data["e_eta"], k_sigma=4, n_grid=11)
                    # Evaluate the corresponding MNR hyperprior for the grid
                    lp_eta = Normal(
                        eta_prior_mean, eta_prior_std).log_prob(eta_grid)

                    # Evaluate the likelihood of the observed data given the
                    # grid, `(n_gal, n_eta_grid)`
                    if data.add_eta_truncation:
                        lp_eta += TruncatedNormal(
                            eta_grid, data["e_eta"][:, None], low=data.eta_min,
                            high=data.eta_max).log_prob(data["eta"][:, None])
                    else:
                        lp_eta += Normal(
                            eta_grid, data["e_eta"][:, None]).log_prob(
                                data["eta"][:, None])
                else:
                    # Sample the galaxy linewidth from a Gaussian hyperprior.
                    eta = sample(
                        "eta_latent", Normal(eta_prior_mean, eta_prior_std))
                    # Track the likelihood of the observed linewidths.
                    if data.add_eta_truncation:
                        sample("eta", TruncatedNormal(
                            eta, data["e_eta"], low=data.eta_min,
                            high=data.eta_max), obs=data["eta"])
                    else:
                        sample("eta", Normal(
                            eta, data["e_eta"]), obs=data["eta"])

                e_mag = jnp.sqrt(sigma_int**2 + data["e2_mag"])
            else:
                eta = data["eta"]
                e_mag = get_linear_sigma_mag_TFR(data, sigma_int, b_TFR, c_TFR)

            r_grid = data["r_grid"] / h

            # Log-prior on the galaxy distance, `(n_field, n_gal, n_step)`
            lp_dist = log_prior_r_empirical(
                r_grid, **kwargs_dist, Rmax_grid=r_grid[-1])[None, None, :]

            if data.has_precomputed_los:
                # Reconstruction LOS velocity `(n_field, n_gal, n_step)`
                Vrad = beta * data["los_velocity_r_grid"]
                # Add inhomogeneous Malmquist bias and normalize the r prior
                lp_dist += lp_galaxy_bias(
                    data["los_delta_r_grid"],
                    data["los_log_density_r_grid"],
                    bias_params, self.galaxy_bias
                    )
                lp_dist -= ln_simpson(
                    lp_dist, x=r_grid[None, None, :], axis=-1)[..., None]
            else:
                Vrad = 0.

            # Likelihood of the observed redshifts, `(n_field, n_gal, n_rbins)`
            Vext_rad = compute_Vext_radial(
                data, r_grid, Vext, with_radial_Vext=self.with_radial_Vext,
                **self.kwargs_radial_Vext)
            czpred = predict_cz(
                self.distance2redshift(r_grid, h=h)[None, None, :],
                Vrad + Vext_rad)
            ll_cz = Normal(czpred, sigma_v).log_prob(
                data["czcmb"][None, :, None])

            # Likelihood of the observed magnitudes.
            if self.use_MNR and self.marginalize_eta:
                # Absolute magnitude grid, `(n_gal, n_eta_grid)
                M_eta = get_absmag_TFR(eta_grid, a_TFR[:, None], b_TFR, c_TFR)
                # Radial grid converted to distance modulus, `(n_gal,)`
                mu_grid = self.distance2distmod(r_grid, h=h)

                # Likelihood of magnitudes `(n_gal, n_rbin, n_eta_grid)`
                ll_mag = Normal(
                    mu_grid[None, :, None] + M_eta[:, None, :],
                    e_mag[:, None, None]).log_prob(
                        (data["mag"] - Ab)[:, None, None])

                # Add the log-prior and log-likelihood of linewidth, the shape
                # remains `(n_gal, n_rbin, n_eta_grid)`
                if self.marginalize_eta:
                    ll_mag += lp_eta[:, None, :]

                # Add all ... `(n_field, n_gal, nrbins, n_eta_grid)`
                ll = (ll_cz + lp_dist)[..., None] + ll_mag[None, ...]
                # Marginalize over the eta grid, `(n_field, n_gal, nrbins)`
                ll = ln_simpson(ll, x=eta_grid[None, :, None, :], axis=-1)
                # Marginalize over the radial distance `(n_field, n_gal)`
                ll = ln_simpson(ll, x=r_grid[None, None, :], axis=-1)
            else:
                # Likelihood ... `(n_field, n_gal, n_grid)`
                ll_mag = Normal(
                    self.distance2distmod(r_grid, h=h)[None, :] +
                    get_absmag_TFR(eta, a_TFR, b_TFR, c_TFR)[:, None],
                    e_mag[:, None]).log_prob(
                        (data["mag"] - Ab)[:, None])[None, ...]
                ll = ll_cz + ll_mag + lp_dist
                # Marginalise over the radial distance, `(n_field, n_gal)`
                ll = ln_simpson(ll, x=r_grid[None, None, :], axis=-1)

            # Average over realizations and track the log-density.
            factor("ll_obs",
                   logsumexp(ll, axis=0) - jnp.log(data.num_fields))


###############################################################################
#                           Supernova models                                 #
###############################################################################


class SNModel(BaseModel):
    """
    A SNe forward model: the distance is numerically marginalized at each MCMC
    step. The Tripp coefficients are sampled. In the MNR branch, x1 and c are
    sampled explicitly.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if self.use_MNR:
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for `SNModel`.")

        # --- Tripp (SALT2) parameters ---
        M_SN = rsample("SN_absmag", self.priors["SN_absmag"], shared_params)
        alpha_SN = rsample("SN_alpha", self.priors["SN_alpha"], shared_params)
        beta_SN = rsample("SN_beta", self.priors["SN_beta"], shared_params)
        dM = rsample(
            "zeropoint_dipole", self.priors["zeropoint_dipole"], shared_params)
        M_SN = M_SN + jnp.sum(dM * data["rhat"], axis=1)
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)

        # Distance marginalization does not sample h.
        h = 1.0

        # Empirical p(r) hyperparameters
        R_dist_emp = rsample("R_dist_emp", self.priors["R_dist_emp"])
        p_dist_emp = rsample("p_dist_emp", self.priors["p_dist_emp"])
        n_dist_emp = rsample("n_dist_emp", self.priors["n_dist_emp"])
        kwargs_dist = {"R": R_dist_emp, "p": p_dist_emp, "n": n_dist_emp}

        # --- Velocity field / selection nuisance ---
        if self.with_radial_Vext:
            Vext = rsample("Vext_rad", self.priors["Vext_radial"],
                           shared_params)
        else:
            Vext = rsample("Vext", self.priors["Vext"], shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)
        beta = rsample("beta", self.priors["beta"], shared_params)

        bias_params = sample_galaxy_bias(self.priors, self.galaxy_bias,
                                         shared_params, Om=self.Om, beta=beta)

        # Optional MNR hyperpriors for x1, c (only used if self.use_MNR)
        if self.use_MNR:
            x1_prior_mean = sample(
                "x1_prior_mean", Uniform(data["min_x1"], data["max_x1"]))
            x1_prior_std = sample(
                "x1_prior_std",
                Uniform(0.0, data["max_x1"] - data["min_x1"]))
            c_prior_mean = sample(
                "c_prior_mean", Uniform(data["min_c"], data["max_c"]))
            c_prior_std = sample(
                "c_prior_std", Uniform(0.0, data["max_c"] - data["min_c"]))

        with plate("data", nsamples):
            if self.use_MNR:
                # Sample latent x1, c and condition on their measurements.
                x1 = sample("x1_latent", Normal(x1_prior_mean, x1_prior_std))
                c = sample("c_latent", Normal(c_prior_mean, c_prior_std))

                sample("x1_obs", Normal(x1, data["e_x1"]), obs=data["x1"])
                sample("c_obs", Normal(c, data["e_c"]), obs=data["c"])

                # Magnitude scatter does NOT re-propagate x1/c obs errors.
                e_mag = jnp.sqrt(sigma_int**2 + data["e2_mag"])
            else:
                # Use measured x1, c; propagate their errors into μ.
                x1 = data["x1"]
                c = data["c"]
                e_mag = jnp.sqrt(
                    data["e2_mag"] + (alpha_SN**2) * data["e2_x1"]
                    + (beta_SN**2) * data["e2_c"] + sigma_int**2)

            # ----- r grid -----
            r_grid = data["r_grid"] / h  # (n_r,)

            # Log-prior on r: (n_field, n_gal, n_r)
            lp_dist = log_prior_r_empirical(
                r_grid, **kwargs_dist, Rmax_grid=r_grid[-1])[None, None, :]

            if data.has_precomputed_los:
                Vrad = beta * data["los_velocity_r_grid"]
                lp_dist += lp_galaxy_bias(
                    data["los_delta_r_grid"],
                    data["los_log_density_r_grid"],
                    bias_params, self.galaxy_bias)

                lp_dist -= ln_simpson(
                    lp_dist, x=r_grid[None, None, :], axis=-1)[..., None]
            else:
                Vrad = 0.0

            # Predicted cz (n_field, n_gal, n_r)
            Vext_rad = compute_Vext_radial(
                data, r_grid, Vext, with_radial_Vext=self.with_radial_Vext,
                **self.kwargs_radial_Vext)
            czpred = predict_cz(
                self.distance2redshift(r_grid, h=h)[None, None, :],
                Vrad + Vext_rad)
            ll_cz = Normal(czpred, sigma_v).log_prob(
                data["czcmb"][None, :, None])

            # ----- Magnitude likelihood -----
            # Tripp: m_obs ~ Normal( μ(r) + M - α x1 + β c , e_mag )
            mu_r = self.distance2distmod(r_grid, h=h)[None, :]   # (1, n_r)
            M_eff = (M_SN - alpha_SN * x1 + beta_SN * c)         # (n_gal,)
            ll_mag = Normal(
                loc=mu_r + M_eff[:, None], scale=e_mag[:, None]).log_prob(
                    data["mag"][:, None])[None, ...]    # (1,n_gal,n_r)

            # Combine and integrate over r
            ll = ll_cz + ll_mag + lp_dist
            ll = ln_simpson(ll, x=r_grid[None, None, :], axis=-1)

            # Average over realizations and contribute to joint density
            factor("ll_obs",
                   logsumexp(ll, axis=0) - jnp.log(data.num_fields))


def add_sigma_mag_to_lane_cov(sigma_mag, Sigma_d):
    """
    Add the intrinsic magnitude scatter to the Lane covariance matrix. It is
    added along the diagonal and the indices corresponding to the magnitude
    residuals.
    """
    n3 = Sigma_d.shape[0]
    N = n3 // 3
    idx = 3 * jnp.arange(N)
    diag_D = jnp.zeros(n3).at[idx].set(sigma_mag**2)
    return Sigma_d + jnp.diag(diag_D)


class PantheonPlusModel(BaseModel):
    """
    Pantheon+ forward model, the distance is numerically marginalized out at
    each MCMC step instead of being sampled as a latent variable.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if not self.use_MNR:
            raise ValueError(
                "The PantheonPlus model requires the MNR model to be used. "
                "Please set `use_MNR` to True in the config file.")

        if self.with_radial_Vext:
            raise ValueError("Radial velocity extension is not supported "
                             "for `PantheonPlusModel`")

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
            "zeropoint_dipole", self.priors["zeropoint_dipole"], shared_params)
        M = M + jnp.sum(dM * data["rhat"], axis=1)
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)

        # For the Lane covariance we sample the Tripp params.
        if data.with_lane_covmat:
            alpha_SN = rsample(
                "SN_alpha", self.priors["SN_alpha"], shared_params)
            beta_SN = rsample("SN_beta", self.priors["SN_beta"], shared_params)
            x1 = data["x1"]
            c = data["c"]

        R_dist_emp = rsample("R_dist_emp", self.priors["R_dist_emp"])
        p_dist_emp = rsample("p_dist_emp", self.priors["p_dist_emp"])
        n_dist_emp = rsample("n_dist_emp", self.priors["n_dist_emp"])
        kwargs_dist = {"R": R_dist_emp, "p": p_dist_emp, "n": n_dist_emp}

        # Sample velocity field parameters.
        Vext = rsample("Vext", self.priors["Vext"], shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)
        # Radially-project Vext
        Vext_rad = jnp.sum(data["rhat"] * Vext[None, :], axis=1)

        # Remaining parameters
        beta = rsample("beta", self.priors["beta"], shared_params)
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params,
            Om=self.Om, beta=beta)

        # For the distance marginalization, h is not sampled. Watch out for
        # the width of the uniform distribution of h is sampled. A grid
        # is still required to normalize the inhomogeneous Malmquist bias
        # distribution.
        h = 1.
        r_grid = data["r_grid"] / h
        Rmax = r_grid[-1]

        # Sample the radial distance to each galaxy, `(n_galaxies)`.
        with plate("plate_distance", nsamples):
            r = sample("r_latent", Uniform(0, Rmax))

        mu = self.distance2distmod(r, h=h)  # (n_gal,)

        # Precompute the homogeneous distance prior, `(n_field, n_galaxies)`
        lp_dist = log_prior_r_empirical(
            r, **kwargs_dist, Rmax_grid=Rmax)[None, :]

        if data.with_lane_covmat:
            # Lane covariance is 3N x 3N where the values in the data vector
            # are `magnitude residual, 0, 0` repeated for all hosts.
            C = add_sigma_mag_to_lane_cov(sigma_int, data["mag_covmat"])

            # Compute the magnitude residuals.
            M_eff = (M - alpha_SN * x1 + beta_SN * c)         # (n_gal,)
            dx = data["mag"] - (mu + M_eff)

            # Compute the magnitude difference vector
            # [mag_res_i, 0, 0, mag_res_i + 1, 0, 0, etc...]
            dX = jnp.zeros((3 * dx.size,), dtype=dx.dtype)
            dX = dX.at[0::3].set(dx)
            # Finally, track the likelihood of the magnitudes
            sample(
                "mag_obs", MultivariateNormal(dX, C), obs=jnp.zeros_like(dX))
        else:
            # Track the likelihood of the predicted magnitudes, add any
            # intrinsic scatter to the covariance matrix.
            C = (data["mag_covmat"]
                 + jnp.eye(data["mag_covmat"].shape[0]) * sigma_int**2)
            sample("mag_obs", MultivariateNormal(mu + M, C), obs=data["mag"])

        if data.has_precomputed_los:
            # Evaluate the radial velocity and the galaxy bias at the sampled
            # distances, `(n_field, n_gal,)`
            Vrad = beta * data.f_los_velocity(r)
            # Inhomogeneous Malmquist bias contribution.
            lp_dist += lp_galaxy_bias(
                data.f_los_delta(r), data.f_los_log_density(r),
                bias_params, self.galaxy_bias)

            # Distance prior normalization grid, `(n_field, n_gal, n_rbin)`
            lp_dist_norm = log_prior_r_empirical(
                r_grid, **kwargs_dist, Rmax_grid=Rmax)[None, None, :]
            lp_dist_norm += lp_galaxy_bias(
                data["los_delta_r_grid"],
                data["los_log_density_r_grid"],
                bias_params, self.galaxy_bias)
            # Finally integrate over the radial bins and normalize.
            lp_dist -= ln_simpson(
                lp_dist_norm, x=r_grid[None, None, :], axis=-1)
        else:
            Vrad = 0.

        with plate("plate_redshift", nsamples):
            # Predicted redshift, `(n_field, n_galaxies)`
            czpred = predict_cz(
                self.distance2redshift(r, h=h)[None, :],
                Vrad + Vext_rad[None, :])
            # Compute the redshift likelihood, and add the distance prior
            ll = Normal(czpred, sigma_v).log_prob(data["czcmb"][None, :])
            ll += lp_dist
            # Average the over field realizations and track
            factor("ll_obs", logsumexp(ll, axis=0) - jnp.log(data.num_fields))


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


class ClustersModel(BaseModel):
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
        raise NotImplementedError("Must be updated after chatting with Tariq.")
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for "
                "`ClustersModel`.")

        # Sample the cluster scaling parameters.
        A = rsample("A_CL", self.priors["CL_A"], shared_params)
        B = rsample("B_CL", self.priors["CL_B"], shared_params)
        C = rsample("C_CL", self.priors["CL_C"], shared_params)
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)

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
                sigma_int2 = jnp.ones(nsamples) * sigma_int**2
                rel = self.which_relation[0]

                # Fixed contributions depending on relation type
                if rel == "L":
                    # LogL = A + B * logT + C * logY
                    logF = data["logF"]
                    sigma_int2 += data["e2_logF"]
                elif rel == "Y":
                    # logY = A + B * logT + C * logF
                    logY = data["logY"]
                    sigma_int2 += data["e2_logY"]
                else:
                    raise ValueError(
                        f"Invalid scaling relation '{self.which_relation}'.")

                # Conditional contributions based on sampling flags
                if self.sample_T:
                    logT = data["logT"]
                    sigma_int2 += B**2 * data["e2_logT"]

                if self.sample_Y and rel == "L":
                    logY = data["logY"]
                    sigma_int2 += C**2 * data["e2_logY"]

                if self.sample_F and rel == "Y":
                    logF = data["logF"]
                    sigma_int2 += C**2 * data["e2_logF"]

                sigma_int = jnp.sqrt(sigma_int2)

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
                mu_cluster[:, None], sigma_int[:, None]).log_prob(mu_grid)

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


class FPModel(BaseModel):
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
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

        self.distance2logangdist = Distance2LogAngDist(Om0=self.Om)

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for `FPModel`.")

        # Sample the FP parameters.
        a_FP = rsample("a_FP", self.priors["FP_a"], shared_params)
        b_FP = rsample("b_FP", self.priors["FP_b"], shared_params)
        c_FP = rsample("c_FP", self.priors["FP_c"], shared_params)
        sigma_log_theta = rsample(
            "sigma_log_theta", self.priors["sigma_log_theta"], shared_params)

        # For the distance marginalization, h is not sampled.
        h = 1.

        R_dist_emp = rsample("R_dist_emp", self.priors["R_dist_emp"])
        p_dist_emp = rsample("p_dist_emp", self.priors["p_dist_emp"])
        n_dist_emp = rsample("n_dist_emp", self.priors["n_dist_emp"])
        kwargs_dist = {"R": R_dist_emp, "p": p_dist_emp, "n": n_dist_emp}

        # Sample velocity field parameters.
        if self.with_radial_Vext:
            Vext = rsample(
                "Vext_rad", self.priors["Vext_radial"], shared_params)
        else:
            Vext = rsample("Vext", self.priors["Vext"], shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)

        # Remaining parameters
        beta = rsample("beta", self.priors["beta"], shared_params)
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params, Om=self.Om,
            beta=beta)

        if self.use_MNR:
            raise NotImplementedError("MNR for FP is not implemented yet.")

        with plate("data", nsamples):
            if self.use_MNR:
                raise NotImplementedError("MNR for FP is not implemented yet.")
            else:
                logs = data["logs"]
                logI = data["logI"]

                sigma_log_theta = jnp.sqrt(
                    + sigma_log_theta**2
                    + a_FP**2 * data["e2_logs"] + b_FP**2 * data["e2_logI"]
                    )

            r_grid = data["r_grid"] / h
            logda_grid = self.distance2logangdist(r_grid)

            # Homogeneous Malmqusit distance prior, `(n_field, n_gal, n_rbin)`
            lp_dist = log_prior_r_empirical(
                r_grid, **kwargs_dist, Rmax_grid=r_grid[-1])[None, None, :]

            # Predict the angular galaxy size, `(n_gal, n_rbin)``
            log_theta_eff = (
                + (a_FP * logs + b_FP * logI + c_FP - 3)[:, None]
                - logda_grid[None, :]
                )

            # Likelihood of the obs ang sizes, `(n_field, n_gal, n_rbin)`
            ll = Normal(log_theta_eff, sigma_log_theta[:, None]).log_prob(
                data["log_theta_eff"][:, None])[None, ...]

            if data.has_precomputed_los:
                # Reconstruction LOS velocity `(n_field, n_gal, n_step)`
                Vrad = beta * data["los_velocity_r_grid"]
                # Add inhomogeneous Malmquist bias and normalize the r prior
                lp_dist += lp_galaxy_bias(
                    data["los_delta_r_grid"],
                    data["los_log_density_r_grid"],
                    bias_params, self.galaxy_bias
                    )
                lp_dist -= ln_simpson(
                    lp_dist, x=r_grid[None, None, :], axis=-1)[..., None]
            else:
                Vrad = 0.

            # Add the distance prior to the tracked likelihood
            ll += lp_dist
            # Likelihood of the observed redshifts, `(n_field, n_gal, n_rbins)`
            Vext_rad = compute_Vext_radial(
                data, r_grid, Vext, with_radial_Vext=self.with_radial_Vext,
                **self.kwargs_radial_Vext)
            czpred = predict_cz(
                self.distance2redshift(r_grid, h=h)[None, None, :],
                Vrad + Vext_rad)
            ll += Normal(czpred, sigma_v).log_prob(
                data["czcmb"][None, :, None])

            # Marginalise over the radial distance, average over realisations
            # and track the log-density.
            ll = ln_simpson(ll, x=r_grid[None, None, :], axis=-1)
            factor("ll_obs", logsumexp(ll, axis=0) - jnp.log(data.num_fields))


###############################################################################
#                       Calibrated-distance model                             #
###############################################################################

class CalibratedDistanceModel_DistMarg(BaseModel):
    """
    A calibrated distance model, where the task is to forward-model a set of
    precomputed galaxy distance moduli, typically done e.g. in CF4, while
    sampling the Hubble constant.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if self.use_MNR:
            raise RuntimeError(
                "MNR for FP is not implemented yet. Please set "
                "`use_MNR` to False in the config file.")
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

        if self.prior_dist_name["h"] == "delta":
            raise ValueError(
                "Must sample `h` for `CalibratedDistanceModel_DistMarg`. "
                "Currently set to a delta-function prior.")

        self.distance2distmod_with_Om = Distance2Distmod_withOm()
        self.distance2redshift_with_Om = Distance2Redshift_withOm()

    def __call__(self, data, shared_params=None):
        nsamples = len(data)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for `FPModel`.")

        # Sample the FP parameters.
        h = rsample("h", self.priors["h"], shared_params)
        Om = rsample("Om", self.priors["Om"], shared_params)
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)

        R_dist_emp = rsample("R_dist_emp", self.priors["R_dist_emp"])
        p_dist_emp = rsample("p_dist_emp", self.priors["p_dist_emp"])
        n_dist_emp = rsample("n_dist_emp", self.priors["n_dist_emp"])
        kwargs_dist = {"R": R_dist_emp, "p": p_dist_emp, "n": n_dist_emp}

        # Sample velocity field parameters.
        if self.with_radial_Vext:
            Vext = rsample(
                "Vext_rad", self.priors["Vext_radial"], shared_params)
        else:
            Vext = rsample("Vext", self.priors["Vext"], shared_params)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)

        # Remaining parameters
        beta = rsample("beta", self.priors["beta"], shared_params)
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params, Om=Om,
            beta=beta)

        if self.use_MNR:
            raise NotImplementedError("MNR for FP is not implemented yet.")

        with plate("data", nsamples):
            if self.use_MNR:
                raise NotImplementedError("MNR for FP is not implemented yet.")

            # Convert the distance grid from `Mpc / h` to `Mpc``
            r_grid = data["r_grid"] / h
            mu_grid = self.distance2distmod_with_Om(r_grid, Om=Om, h=h)

            # Homogeneous Malmqusit distance prior, `(n_field, n_gal, n_rbin)`
            lp_dist = log_prior_r_empirical(
                r_grid, **kwargs_dist, Rmax_grid=r_grid[-1])[None, None, :]

            # Likelihood of the 'obs' dist moduli, `(n_field, n_gal, n_rbin)`
            sigma_mu = jnp.sqrt(sigma_int**2 + data["e2_mu"])
            ll = Normal(
                mu_grid[None, :], sigma_mu[:, None]).log_prob(
                data["mu"][:, None])[None, ...]

            if data.has_precomputed_los:
                # Reconstruction LOS velocity `(n_field, n_gal, n_step)`
                Vrad = beta * data["los_velocity_r_grid"]
                # Add inhomogeneous Malmquist bias and normalize the r prior
                lp_dist += lp_galaxy_bias(
                    data["los_delta_r_grid"],
                    data["los_log_density_r_grid"],
                    bias_params, self.galaxy_bias
                    )
                lp_dist -= ln_simpson(
                    lp_dist, x=r_grid[None, None, :], axis=-1)[..., None]
            else:
                Vrad = 0.

            # Add the distance prior to the tracked likelihood
            ll += lp_dist
            # Likelihood of the observed redshifts, `(n_field, n_gal, n_rbins)`
            Vext_rad = compute_Vext_radial(
                data, r_grid, Vext, with_radial_Vext=self.with_radial_Vext,
                **self.kwargs_radial_Vext)
            czpred = predict_cz(
                self.distance2redshift_with_Om(
                    r_grid, Om=Om, h=h)[None, None, :], Vrad + Vext_rad)
            ll += Normal(czpred, sigma_v).log_prob(
                data["czcmb"][None, :, None])

            # Marginalise over the radial distance, average over realisations
            # and track the log-density.
            ll = ln_simpson(ll, x=r_grid[None, None, :], axis=-1)
            factor("ll_obs", logsumexp(ll, axis=0) - jnp.log(data.num_fields))


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
