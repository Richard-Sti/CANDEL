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
General utility functions for PV and H0 forward models: configuration,
physics, priors, and SH0ES helpers.
"""
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import betainc, gammaln, logsumexp
from jax.scipy.stats import norm as norm_jax
from numpy.polynomial.hermite import hermgauss as _hermgauss
from numpyro.distributions import (Delta, Distribution, Gamma, LogUniform,
                                   Normal, TruncatedNormal, Uniform,
                                   constraints)

from ..util import SPEED_OF_LIGHT

# Gauss-Hermite nodes for Student-t selection integrals via log-space
# saddle-point quadrature. int f(x) exp(-x^2) dx ~ sum_i w_i f(x_i).
_N_GH_SEL = 32
_GH_SEL_NODES_NP, _GH_SEL_WEIGHTS_NP = _hermgauss(_N_GH_SEL)
_GH_SEL_NODES = jnp.asarray(_GH_SEL_NODES_NP)
_GH_SEL_LOG_WEIGHTS = jnp.log(jnp.asarray(_GH_SEL_WEIGHTS_NP))

###############################################################################
#                         Configuration file checks                           #
###############################################################################


# Config sub-sections that must match across submodels in a joint
# inference. Anything outside these (per-catalogue io entries,
# inference-control flags like compute_evidence) may legitimately
# differ between submodels.
JOINT_RELEVANT_SECTIONS = ("model", "pv_model")


def _leaf_eq(a, b):
    """Equality that tolerates numpy/jax arrays.

    Their ``==`` is elementwise.
    """
    if hasattr(a, "__array__") or hasattr(b, "__array__"):
        try:
            return bool(np.array_equal(a, b))
        except Exception:
            return False
    return a == b


def _dict_path_diffs(a, b, prefix):
    """Return list of (path, val_a, val_b) leaf disagreements."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            sub = f"{prefix}/{k}"
            if k not in a:
                out.append((sub, "<missing>", b[k]))
            elif k not in b:
                out.append((sub, a[k], "<missing>"))
            else:
                out.extend(_dict_path_diffs(a[k], b[k], sub))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(prefix, a, b)]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_dict_path_diffs(x, y, f"{prefix}[{i}]"))
        return out
    if _leaf_eq(a, b):
        return []
    return [(prefix, a, b)]


def joint_config_mismatch(ref_cfg, other_cfg,
                          sections=JOINT_RELEVANT_SECTIONS):
    """List joint-incompatible differences between two config dicts.

    Returns ``[(key_path, ref_value, other_value), ...]`` restricted to
    the sections that determine joint forward-model behaviour. An empty
    list means the two configs agree on every joint-relevant key.
    """
    diffs = []
    for section in sections:
        diffs.extend(_dict_path_diffs(
            ref_cfg.get(section, {}), other_cfg.get(section, {}), section))
    return diffs

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


def normal_logpdf_var(x, mean, var):
    """Log-pdf of a normal distribution parameterized by variance."""
    d = x - mean
    return -0.5 * (jnp.log(2 * jnp.pi * var) + d * d / var)


def student_t_logpdf_var(x, mean, var, nu):
    """Log-pdf of a Student-t parameterized by variance (=scale^2).

    `var` is the scale parameter squared, not the distribution
    variance (which is var * nu / (nu - 2) for nu > 2).
    """
    d = x - mean
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * jnp.log(nu * jnp.pi * var)
            - (nu + 1) / 2 * jnp.log1p(d * d / (nu * var)))


def student_t_logcdf(t, nu):
    """Log-CDF of a standardized Student-t distribution (loc=0, scale=1)."""
    t2 = t * t
    w = nu / (nu + t2)
    Ix = betainc(nu / 2, 0.5, w)
    return jnp.where(t < 0,
                     jnp.log(0.5) + jnp.log(Ix),
                     jnp.log1p(-0.5 * Ix))


###############################################################################
#                                Priors                                       #
###############################################################################


def smoothclip_nr(nr, tau):
    """Smooth zero-clipping for the number density."""
    return 0.5 * (nr + jnp.sqrt(nr**2 + tau**2))


class SineAngle(Distribution):
    r"""Sine-weighted angle prior for disk inclination.

    Isotropic random orientation gives P(i) ∝ sin(i) for the inclination
    angle i ∈ [low, high] (in radians internally, but the distribution
    works in DEGREES to match the maser model convention).

    PDF:
        f(i) = sin(i) / (cos(low) - cos(high)),  low <= i <= high
    where i, low, high are in degrees.
    """
    arg_constraints = {"low": constraints.real, "high": constraints.real}
    reparametrized_params = ["low", "high"]

    def __init__(self, low=0.0, high=180.0, validate_args=None):
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(self.low), jnp.shape(self.high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def sample(self, key, sample_shape=()):
        # CDF inversion: cos(i) ~ Uniform(cos(high), cos(low))
        shape = sample_shape + self.batch_shape
        lo_rad = jnp.deg2rad(self.low)
        hi_rad = jnp.deg2rad(self.high)
        u = random.uniform(key, shape)
        cos_i = jnp.cos(hi_rad) + u * (jnp.cos(lo_rad) - jnp.cos(hi_rad))
        return jnp.rad2deg(jnp.arccos(cos_i))

    def log_prob(self, value):
        lo_rad = jnp.deg2rad(self.low)
        hi_rad = jnp.deg2rad(self.high)
        val_rad = jnp.deg2rad(value)
        log_norm = jnp.log(jnp.abs(jnp.cos(lo_rad) - jnp.cos(hi_rad)))
        # sin(i) in radians, but value is in degrees so need deg->rad Jacobian
        # f(i_deg) = sin(i_rad) / norm * (pi/180)
        lp = jnp.log(jnp.sin(val_rad)) - log_norm + jnp.log(jnp.pi / 180)
        in_bounds = (value >= self.low) & (value <= self.high)
        return jnp.where(in_bounds, lp, -jnp.inf)


class VolumePrior(Distribution):
    r"""Volumetric distance prior p(D) \propto D^2 on [low, high].

    Normalised PDF: f(D) = 3 D^2 / (high^3 - low^3).
    CDF inversion: D = (u * (high^3 - low^3) + low^3)^{1/3}.
    """
    arg_constraints = {"low": constraints.positive,
                       "high": constraints.positive}
    reparametrized_params = ["low", "high"]

    def __init__(self, low, high, validate_args=None):
        self.low = jnp.asarray(low, dtype=float)
        self.high = jnp.asarray(high, dtype=float)
        self._d3_diff = self.high**3 - self.low**3
        self._log_norm = jnp.log(self._d3_diff / 3)
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(self.low), jnp.shape(self.high))
        super().__init__(batch_shape=batch_shape,
                         validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        u = random.uniform(key, shape)
        return (u * self._d3_diff + self.low**3)**(1 / 3)

    def log_prob(self, value):
        in_bounds = (value >= self.low) & (value <= self.high)
        return jnp.where(in_bounds, 2 * jnp.log(value) - self._log_norm,
                         -jnp.inf)


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


class Maxwell(Distribution):
    r"""
    Maxwell–Boltzmann (speed) distribution in 3D.

    PDF:
        f(x; a) = sqrt(2/pi) * x^2 * exp(-x^2 / (2 a^2)) / a^3,  x >= 0,  a > 0
    where `a = sqrt(kT/m)` is the scale parameter.

    Args:
        scale (float or array): positive scale parameter `a`.

    Notes:
        This is the chi distribution with k=3 degrees of freedom and scale `a`.
    """
    arg_constraints = {"scale": constraints.positive}
    support = constraints.nonnegative
    reparametrized_params = ["scale"]

    def __init__(self, scale=1.0, validate_args=None):
        self.scale = jnp.asarray(scale)
        batch_shape = jnp.shape(self.scale)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        z = random.normal(key, shape + (3,)) * self.scale[..., None]
        return jnp.linalg.norm(z, axis=-1)

    def log_prob(self, value):
        a = self.scale
        x = jnp.asarray(value)

        in_support = (x >= 0)
        # log(sqrt(2/pi)) = 0.5*(log 2 - log pi)
        log_c = 0.5 * (jnp.log(2.0) - jnp.log(jnp.pi))
        lp = (log_c + 2.0 * jnp.log(x)
              - (x * x) / (2.0 * a * a) - 3.0 * jnp.log(a))
        return jnp.where(in_support, lp, -jnp.inf)


def load_priors(config_priors):
    """Load a dictionary of NumPyro distributions from a TOML file."""
    _DIST_MAP = {
        "normal": lambda p: Normal(p["loc"], p["scale"]),
        "truncated_normal": lambda p: TruncatedNormal(p["mean"], p["scale"], low=p.get("low", None), high= p.get("high", None)),  # noqa
        "uniform": lambda p: Uniform(p["low"], p["high"]),
        "log_uniform": lambda p: LogUniform(p["low"], p["high"]),
        "delta": lambda p: Delta(p["value"]),
        "jeffreys": lambda p: JeffreysPrior(p["low"], p["high"]),
        "volume": lambda p: VolumePrior(p["low"], p["high"]),
        "gamma": lambda p: Gamma(p["concentration"], p["rate"]),
        "maxwell": lambda p: Maxwell(p["scale"]),
        "sine_angle": lambda p: SineAngle(
            p.get("low", 0.0), p.get("high", 180.0)),
        "vector_uniform": lambda p: {"type": "vector_uniform", "low": p["low"], "high": p["high"]},  # noqa
        "vector_uniform_fixed": lambda p: {"type": "vector_uniform_fixed", "low": p["low"], "high": p["high"],},  # noqa
        "vector_radial_uniform": lambda p: {"type": "vector_radial_uniform", "nval": len(p["rknot"]), "low": p["low"], "high": p["high"]},  # noqa
        "vector_components_uniform": lambda p: {"type": "vector_components_uniform", "low": p["low"], "high": p["high"],},  # noqa
        "vector_radialmag_uniform": lambda p: {"type": "vector_radialmag_uniform", "nval": len(p["rknot"]), "low": p["low"], "high": p["high"]},  # noqa
        "quadrupole": lambda p: {"type": "quadrupole", "low": p["low"], "high": p["high"]},  # noqa
        "octupole": lambda p: {"type": "octupole", "low": p["low"], "high": p["high"]},  # noqa
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
#                         SH0ES utility functions                             #
###############################################################################


def log_integral_gauss_pdf_times_cdf(mu, sigma, t, w):
    """
    Log of ∫ N(x|mu, sigma^2) Φ((t - x)/w) dx.
    Closed form: Φ((mu - t)/sqrt(sigma^2 + w^2))
    """
    return norm_jax.logcdf((t - mu) / jnp.sqrt(sigma**2 + w**2))


def _student_t_sel_gh_weights(nu):
    """GH quadrature nodes (tau) and log-weights for Student-t selection.

    The Student-t is a Gaussian scale mixture: t_nu = int N(0,1/tau)
    Gamma(tau|a,a) dtau with a=nu/2. Substituting s=log(tau) and expanding
    around the saddle point (s=0, curvature a) gives a GH quadrature.
    Fully differentiable w.r.t. nu.
    """
    a = nu / 2
    sigma_s = 1.0 / jnp.sqrt(a)
    s = jnp.sqrt(2.0) * sigma_s * _GH_SEL_NODES
    tau = jnp.exp(s)

    # log [Gamma(exp(s)|a,a) * exp(s)] = a*log(a) - gammaln(a) + a*s - a*exp(s)
    log_gamma_s = a * jnp.log(a) - gammaln(a) + a * s - a * tau

    # GH: int h(s) ds = sigma*sqrt(2) * sum_i w_i * h(s_i) * exp(x_i^2)
    log_w = (_GH_SEL_LOG_WEIGHTS + _GH_SEL_NODES**2
             + jnp.log(jnp.sqrt(2.0) * sigma_s) + log_gamma_s)
    return tau, log_w


def log_prob_integrand_sel(x, e_x, lim, lim_width, nu_cz=None):
    x = jnp.asarray(x)
    e_x = jnp.asarray(e_x)
    if nu_cz is None:
        if lim_width is None:
            return norm_jax.logcdf((lim - x) / e_x)
        else:
            return log_integral_gauss_pdf_times_cdf(x, e_x, lim, lim_width)
    else:
        tau, log_w = _student_t_sel_gh_weights(nu_cz)
        if lim_width is None:
            log_cdf = norm_jax.logcdf(
                (lim - x)[..., None] / e_x[..., None] * jnp.sqrt(tau))
        else:
            var_eff = e_x[..., None]**2 / tau + lim_width**2
            log_cdf = norm_jax.logcdf(
                (lim - x[..., None]) / jnp.sqrt(var_eff))
        return logsumexp(log_w + log_cdf, axis=-1)


def logmeanexp(x, axis=None, denom=None):
    """Stable log(mean(exp(x))) with optional explicit denominator."""
    denom = x.shape[axis] if denom is None else denom
    return logsumexp(x, axis=axis) - jnp.log(denom)


def logweightedmeanexp(log_x, log_w, axis=-1):
    """log(sum(W_j * x_j)) where W_j = exp(log_w_j) / sum(exp(log_w_k))."""
    return logsumexp(log_x + log_w, axis=axis) - logsumexp(log_w, axis=axis)
