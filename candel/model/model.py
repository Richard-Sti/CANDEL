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
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
from jax import random, vmap
from jax.debug import print as jprint  # noqa
from jax.lax import cond
from interpax import interp1d
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammainc, gammaln, logsumexp
from jax.scipy.stats.norm import cdf as jax_norm_cdf
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from numpyro import deterministic, factor, handlers, plate, sample
from numpyro.distributions import (Delta, Distribution, MultivariateNormal,
                                   Normal, ProjectedNormal, TruncatedNormal,
                                   Uniform, constraints)
from numpyro.handlers import reparam
from numpyro.infer.reparam import ProjectedNormalReparam

from ..cosmography import (Distance2Distmod, Distance2Distmod_withOm,
                           Distance2LogAngDist, Distance2LogLumDist,
                           Distance2Redshift, Distance2Redshift_withOm,
                           Redshift2Distance)
from ..util import (SPEED_OF_LIGHT, fprint, galactic_to_radec_cartesian,
                    get_nested, load_config)
from .simpson import ln_simpson

###############################################################################
#                         Configuration file checks                           #
###############################################################################

def smoothclip_nr(nr, tau):
    """Smooth zero-clipping for the number density."""
    return 0.5 * (nr + jnp.sqrt(nr**2 + tau**2))

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


def logpdf_mvn2_broadcast(x1, x2, m1, m2, v11, v22, v12):
    """
    Elementwise log N2 with broadcasting:
    cov = [[v11, v12],[v12, v22]].
    All inputs can broadcast to a common shape.
    """
    two_pi = 2.0 * jnp.pi
    det = v11 * v22 - v12 * v12
    inv11 =  v22 / det
    inv22 =  v11 / det
    inv12 = -v12 / det
    dx1 = x1 - m1
    dx2 = x2 - m2
    quad = inv11*dx1*dx1 + 2.0*inv12*dx1*dx2 + inv22*dx2*dx2
    return -0.5*(2.0*jnp.log(two_pi) + jnp.log(det) + quad)


def _delta_a_to_frac(delta_a):
    """Convert a zeropoint shift ΔA to fractional H change."""
    delta_a = jnp.asarray(delta_a)
    return jnp.power(10.0, 0.5 * delta_a) - 1.0


def _frac_to_mag(frac):
    """Convert fractional δH to magnitude ΔA: ΔA = 2·log10(1 + δH)."""
    frac = jnp.asarray(frac)
    return 2.0 * jnp.log10(1.0 + frac)

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


def sample_vector_fixed(name, mag_min, mag_max, direction=None):
    """
    Sample a 3D vector but without accounting for continuity and poles.

    This enforces that all sampled points have the same contribution to
    `log_density` which is not the case for the `sample_vector` function
    because the unit vectors are drawn.
    """
    if direction is not None:
        phi, cos_theta = _direction_from_galactic(direction)
    else:
        phi = sample(f"{name}_phi", Uniform(0, 2 * jnp.pi))
        cos_theta = sample(f"{name}_cos_theta", Uniform(-1, 1))
    sin_theta = jnp.sqrt(1 - cos_theta**2)

    mag = sample(f"{name}_mag", Uniform(mag_min, mag_max))

    return mag * jnp.array(
        [sin_theta * jnp.cos(phi),
         sin_theta * jnp.sin(phi),
         cos_theta]
        )


def sample_quadrupole_fixed(name, mag_min, mag_max):
    """
    Sample a quadrupole but without accounting for continuity and poles.
    
    This enforces that all sampled points have the same contribution to
    `log_density` which is not the case for the `sample_quadrupole` function
    because the unit vectors are drawn.
    """
    phi1 = sample(f"{name}_phi1", Uniform(0, 2 * jnp.pi))
    cos_theta1 = sample(f"{name}_cos_theta1", Uniform(-1, 1))
    sin_theta1 = jnp.sqrt(1 - cos_theta1**2)
    phi2 = sample(f"{name}_phi2", Uniform(0, 2 * jnp.pi))
    cos_theta2 = sample(f"{name}_cos_theta2", Uniform(-1, 1))
    sin_theta2 = jnp.sqrt(1 - cos_theta2**2)

    mag = sample(f"{name}_mag", Uniform(mag_min, mag_max))

    vector1 = jnp.array(
        [sin_theta1 * jnp.cos(phi1),
         sin_theta1 * jnp.sin(phi1),
         cos_theta1]
        )
    vector2 = jnp.array(
        [sin_theta2 * jnp.cos(phi2),
         sin_theta2 * jnp.sin(phi2),
         cos_theta2]
        )
    return jnp.sqrt(mag) * jnp.array([vector1, vector2]).T



def sample_spline_radial_vector(name, nval, low, high, half_sky=False):
    """
    Sample a radial vector at `nval` knots: direction ~ isotropic,
    magnitude ~ Uniform(low, high). Returns an array of shape (nval, 3).

    If half_sky=True, restrict cos_theta to [0, 1] (northern hemisphere)
    to break the sign degeneracy. Use with signed magnitude to cover all
    physical directions.
    """
    with plate(f"{name}_plate", nval):
        phi = sample(f"{name}_phi", Uniform(0.0, 2.0 * jnp.pi))
        if half_sky:
            cos_theta = sample(f"{name}_cos_theta", Uniform(0.0, 1.0))
        else:
            cos_theta = sample(f"{name}_cos_theta", Uniform(-1.0, 1.0))
        sin_theta = jnp.sqrt(jnp.clip(1.0 - cos_theta**2, 0.0, 1.0))

        mag = sample(f"{name}_mag", Uniform(low, high))

        # Unit direction vector
        u = jnp.stack(
            (sin_theta * jnp.cos(phi),
             sin_theta * jnp.sin(phi),
             cos_theta),
            axis=-1
        )

    return mag[..., None] * u


def _direction_from_galactic(direction):
    """
    Convert galactic (ell, b) in degrees to spherical angles in the ICRS frame
    used internally. We rotate to ICRS Cartesian and then extract (phi, cos_theta).

    This is a JAX-compatible version that doesn't use astropy.
    """
    # Handle both scalar JAX traced values and numpy arrays
    ell_deg = direction["ell"]
    b_deg = direction["b"]

    # Check if inputs are JAX traced values (from numpyro sampling)
    is_jax_traced = hasattr(ell_deg, 'shape') and not isinstance(ell_deg, np.ndarray)

    if is_jax_traced or isinstance(ell_deg, (jnp.ndarray, float, int)):
        # JAX-compatible path: do the rotation manually
        ell_rad = jnp.deg2rad(ell_deg)
        b_rad = jnp.deg2rad(b_deg)

        # Galactic Cartesian coordinates
        cos_b = jnp.cos(b_rad)
        x_gal = cos_b * jnp.cos(ell_rad)
        y_gal = cos_b * jnp.sin(ell_rad)
        z_gal = jnp.sin(b_rad)

        # Rotation matrix from galactic to ICRS (from astropy)
        # Columns are ICRS coords of galactic X, Y, Z unit vectors
        R = jnp.array([
            [-0.0548756577,  0.4941094372, -0.8676661376],
            [-0.8734370520, -0.4448297212, -0.1980763373],
            [-0.4838350736,  0.7469821840,  0.4559838137],
        ])

        # Apply rotation
        x = R[0, 0] * x_gal + R[0, 1] * y_gal + R[0, 2] * z_gal
        y = R[1, 0] * x_gal + R[1, 1] * y_gal + R[1, 2] * z_gal
        z = R[2, 0] * x_gal + R[2, 1] * y_gal + R[2, 2] * z_gal
    else:
        # NumPy path: use astropy for accuracy
        xyz = jnp.asarray(
            galactic_to_radec_cartesian(ell_deg, b_deg),
            dtype=jnp.float32,
        )
        x, y, z = xyz

    norm = jnp.sqrt(x * x + y * y + z * z) + 1e-12
    phi = jnp.arctan2(y, x)
    phi = jnp.where(phi < 0, phi + 2 * jnp.pi, phi)
    cos_theta = jnp.clip(z / norm, -1.0, 1.0)
    return phi, cos_theta


def sample_radialmag_vector(name, nval, low, high, max_modulus=None,
                            direction=None, smoothness_scale=None,
                            smoothness_threshold=None,
                            sample_galactic=False, half_sky=False,
                            fixed_knots=None):
    """
    Sample a vector whose magnitude varies at `nval` knots but a direction
    is shared by all knots and sampled isotropically on the sky. The magnitude
    is sampled ~ Uniform(low, high).

    If `max_modulus` is provided (sequence of length `nval`), magnitudes in
    each bin are drawn from `[-abs(max_modulus_i), abs(max_modulus_i)]`.

    If `smoothness_scale` is provided (in km/s), a smoothness prior is applied
    that penalizes differences between adjacent knots. If `smoothness_threshold`
    is also provided, differences within the threshold are not penalized (flat
    region), and only the excess beyond threshold is penalized with a Gaussian.

    If `sample_galactic` is True, the direction is sampled directly in galactic
    coordinates (ell, b) rather than ICRS spherical coordinates.

    If `half_sky` is True or "north", the galactic latitude is restricted to
    [0, 90°] (northern hemisphere) to break the sign degeneracy. If `half_sky`
    is "south", restricts to [-90°, 0°] (southern hemisphere). Implies
    `sample_galactic=True`.

    If `fixed_knots` is provided (dict mapping knot index to fixed value),
    those knots are fixed at the specified values instead of being sampled.

    Returns the tuple (mag, rhat), where `mag` has shape (nval,) and `rhat`
    has shape (3,).
    """
    # half_sky implies sample_galactic
    # Normalize half_sky to False, "north", or "south"
    if half_sky is True or half_sky == "north":
        half_sky = "north"
        sample_galactic = True
    elif half_sky == "south":
        sample_galactic = True
    else:
        half_sky = False

    # Convert scalar bounds to per-knot arrays.
    low_arr = jnp.asarray(low)
    high_arr = jnp.asarray(high)

    if low_arr.ndim == 0:
        low_arr = jnp.full((nval,), low_arr)
    if high_arr.ndim == 0:
        high_arr = jnp.full((nval,), high_arr)

    if max_modulus is not None:
        max_arr = jnp.abs(jnp.asarray(max_modulus))
        if max_arr.shape[0] != nval:
            raise ValueError(
                f"`max_modulus` must have length {nval}, "
                f"got {max_arr.shape[0]}")
        low_arr = -max_arr
        high_arr = max_arr

    if direction is not None:
        phi, cos_theta = _direction_from_galactic(direction)
    elif sample_galactic:
        # Sample directly in galactic coordinates
        ell_rad = sample(f"{name}_ell", Uniform(0, 2 * jnp.pi))
        # half_sky restricts to one galactic hemisphere
        if half_sky == "north":
            sin_b_low, sin_b_high = 0.0, 1.0
        elif half_sky == "south":
            sin_b_low, sin_b_high = -1.0, 0.0
        else:
            sin_b_low, sin_b_high = -1.0, 1.0
        sin_b = sample(f"{name}_sin_b", Uniform(sin_b_low, sin_b_high))
        b_rad = jnp.arcsin(sin_b)
        # Convert to ICRS for internal use
        phi, cos_theta = _direction_from_galactic({
            "ell": jnp.degrees(ell_rad),
            "b": jnp.degrees(b_rad)
        })
    else:
        phi = sample(f"{name}_phi", Uniform(0, 2 * jnp.pi))
        cos_theta = sample(f"{name}_cos_theta", Uniform(-1, 1))
    sin_theta = jnp.sqrt(1 - cos_theta**2)

    # Handle fixed knots
    if fixed_knots is not None and len(fixed_knots) > 0:
        # Convert string keys to integers (TOML parses numeric keys as strings)
        fixed_knots = {int(k): v for k, v in fixed_knots.items()}
        # Identify which knots to sample vs fix
        fixed_indices = set(fixed_knots.keys())
        sample_indices = [i for i in range(nval) if i not in fixed_indices]
        n_sample = len(sample_indices)

        if n_sample > 0:
            # Sample only the non-fixed knots
            sample_low = jnp.array([low_arr[i] for i in sample_indices])
            sample_high = jnp.array([high_arr[i] for i in sample_indices])
            with plate(f"{name}_plate", n_sample):
                sampled_mag = sample(f"{name}_mag", Uniform(sample_low, sample_high))

            # Build full mag array by combining fixed and sampled values
            mag_list = []
            sample_idx = 0
            for i in range(nval):
                if i in fixed_indices:
                    mag_list.append(fixed_knots[i])
                else:
                    mag_list.append(sampled_mag[sample_idx])
                    sample_idx += 1
            mag = jnp.array(mag_list)
        else:
            # All knots are fixed
            mag = jnp.array([fixed_knots[i] for i in range(nval)])
    else:
        with plate(f"{name}_plate", nval):
            mag = sample(f"{name}_mag", Uniform(low_arr, high_arr))

    # Apply smoothness prior on differences between adjacent knots
    if smoothness_scale is not None and smoothness_scale > 0:
        diffs = jnp.diff(mag)
        threshold = smoothness_threshold if smoothness_threshold is not None else 0.0
        # Soft threshold: no penalty within threshold, Gaussian beyond
        excess = jnp.maximum(jnp.abs(diffs) - threshold, 0.0)
        smoothness_logp = -0.5 * jnp.sum((excess / smoothness_scale) ** 2)
        factor(f"{name}_smoothness", smoothness_logp)

    rhat = jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta
        ])

    return mag, rhat


def h0_percent_to_bulkflow(r, percent, *, H0=100.0, q0=-0.53):
    """
    Convert a fractional H0 dipole (in percent) to an equivalent bulk-flow
    magnitude evaluated at radius `r` (array-like).

    Matches the form used in `plot_Vext_radmag`: delta * (H0 r +
    q0 H0^2 r^2 / c).
    """
    frac = percent / 100.0
    r = jnp.asarray(r)
    return frac * (H0 * r + q0 * (H0**2) * r**2 / SPEED_OF_LIGHT)


def sample_radial_spline_uniform_fixed_direction(name, nval, low, high):
    """
    Sample velocity vectors with a FIXED direction across all radial knots,
    but VARIABLE magnitude per knot.
    
    This is a fixed-direction variant of `sample_radial_spline_uniform()`.
    While that function samples independent 3D vectors at each knot, this function
    constrains all knots to share the same direction but allows different magnitudes.
    
    Useful for modeling bulk flows or jets where the direction is constant
    but the velocity magnitude varies with radius. Can be used with either:
    - which_Vext="radial" for spline interpolation between knots
    - which_Vext="radial_binned" for piecewise constant bins
    
    Parameters
    ----------
    name : str
        Parameter name prefix
    nval : int
        Number of radial knots (for spline) or bins (for piecewise constant)
    low : float
        Lower bound for magnitude
    high : float
        Upper bound for magnitude
    
    Returns
    -------
    vectors : array_like
        Array of shape (nval, 3) with same direction but different magnitudes
    """
    # Sample ONE direction (shared across all knots/bins)
    phi = sample(f"{name}_direction_phi", Uniform(0.0, 2.0 * jnp.pi))
    cos_theta = sample(f"{name}_direction_cos_theta", Uniform(-1.0, 1.0))
    sin_theta = jnp.sqrt(jnp.clip(1.0 - cos_theta**2, 0.0, 1.0))
    
    # Unit direction vector (same for all knots/bins)
    u = jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta
    ])
    
    # Sample DIFFERENT magnitudes for each knot/bin
    with plate(f"{name}_mag_plate", nval):
        mag = sample(f"{name}_mag", Uniform(low, high))
    
    # Broadcast: (nval,) * (3,) -> (nval, 3)
    return mag[:, None] * u[None, :]


def interp_spline_radial_vector(rq, bin_values, **kwargs):
    """Spline interp with constant extrapolation at boundaries."""
    x = jnp.asarray(kwargs["rknot"])
    k = kwargs.get("k", 3)
    endpoints = kwargs.get("endpoints", "not-a-knot")

    rq = jnp.asarray(rq)
    x0, x1 = x[0], x[-1]

    def spline_eval(y):
        s = InterpolatedUnivariateSpline(x, jnp.asarray(y),
                                         k=k, endpoints=endpoints)
        vals = s(rq)
        y0, y1 = y[0], y[-1]
        vals = jnp.where(rq < x0, y0, vals)
        vals = jnp.where(rq > x1, y1, vals)
        return vals

    return vmap(spline_eval)(bin_values.T)


def _slerp(u0, u1, t, eps=1e-8):
    """Spherical linear interpolation between unit vectors u0 and u1."""
    dot = jnp.clip(jnp.dot(u0, u1), -1.0, 1.0)
    theta = jnp.arccos(dot)
    sin_th = jnp.sin(theta)

    def slerp_core(_):
        a = jnp.sin((1.0 - t) * theta) / sin_th
        b = jnp.sin(t * theta) / sin_th
        return a * u0 + b * u1

    def lerp_norm(_):
        # Fall back to normalized linear interpolation for nearly parallel vectors
        v = (1.0 - t) * u0 + t * u1
        n = jnp.linalg.norm(v)
        return jnp.where(n > 0.0, v / n, u0)

    return cond(sin_th < eps, lerp_norm, slerp_core, operand=None)


def interp_cartesian_vector(rq, v_knot, **kwargs):
    """
    Interpolate a 3D Cartesian vector field as a function of radius.

    Uses SLERP for direction interpolation (stays on unit sphere) and
    interpax for magnitude interpolation. This prevents magnitude overshoot
    that can occur with component-wise spline interpolation.

    Parameters
    ----------
    rq : array
        Query radii at which to interpolate.
    v_knot : array of shape (K, 3)
        Velocity vectors at K radial knots.
    **kwargs : dict
        Must contain 'rknot' (radial knot positions).
        Optional 'method' for magnitude interpolation (default 'cubic').

    Returns
    -------
    array of shape (3, len(rq))
        Interpolated velocity vectors (transposed for compatibility).
    """
    rknot = jnp.asarray(kwargs["rknot"])
    method = kwargs.get("method", "cubic")

    rq = jnp.asarray(rq)
    y = jnp.asarray(v_knot)  # (K, 3)
    K = y.shape[0]

    # Handle 2D rq (n_gal, n_los) during z-space iteration with LOS data
    input_shape = rq.shape
    is_2d = rq.ndim == 2
    if is_2d:
        rq_flat = rq.ravel()
    else:
        rq_flat = rq

    # Compute magnitudes and unit directions at knots
    mk = jnp.linalg.norm(y, axis=-1)  # (K,)
    mk_safe = jnp.where(mk > 0.0, mk, 1.0)
    uk = y / mk_safe[:, None]  # (K, 3)

    # Interpolate magnitude using interpax
    x0, x1 = rknot[0], rknot[-1]
    m_r = interp1d(rq_flat, rknot, mk, method=method)
    # Constant extrapolation at boundaries
    m_r = jnp.where(rq_flat < x0, mk[0], m_r)
    m_r = jnp.where(rq_flat > x1, mk[-1], m_r)

    # Interpolate direction using SLERP
    def dir_at_r(r):
        i = jnp.clip(jnp.searchsorted(rknot, r, side="right") - 1, 0, K - 2)
        xl, xr = rknot[i], rknot[i + 1]
        t = jnp.where(xr > xl, (r - xl) / (xr - xl), 0.0)
        return _slerp(uk[i], uk[i + 1], t)

    u_r = vmap(dir_at_r)(rq_flat)  # (R, 3)
    # Constant extrapolation at boundaries
    u_r = jnp.where((rq_flat < x0)[:, None], uk[0], u_r)
    u_r = jnp.where((rq_flat > x1)[:, None], uk[-1], u_r)

    # Combine magnitude and direction
    result = m_r[:, None] * u_r  # (R, 3)

    if is_2d:
        # Reshape to (n_gal, n_los, 3) then transpose to (3, n_gal, n_los)
        result = result.reshape(input_shape + (3,))
        return jnp.moveaxis(result, -1, 0)
    else:
        return result.T  # (3, R) to match interp_spline_radial_vector output


def load_priors(config_priors):
    """Load a dictionary of NumPyro distributions from a TOML file."""
    _DIST_MAP = {
        "normal": lambda p: Normal(p["loc"], p["scale"]),
        "truncated_normal": lambda p: TruncatedNormal(p["mean"], p["scale"], low=p.get("low", None), high= p.get("high", None)),  # noqa
        "uniform": lambda p: Uniform(p["low"], p["high"]),
        "array_uniform": lambda p: {
            "type": "array_uniform",
            "low": p["low"],
            "high": p["high"],
            "nval": p.get("nval"),
        },
        "delta": lambda p: Delta(p["value"]),
        "jeffreys": lambda p: JeffreysPrior(p["low"], p["high"]),
        "maxwell": lambda p: Maxwell(p["scale"]),
        "vector_uniform": lambda p: {"type": "vector_uniform", "low": p["low"], "high": p["high"]},  # noqa
        "vector_uniform_fixed": lambda p: {
            "type": "vector_uniform_fixed",
            "low": p["low"],
            "high": p["high"],
            # Optional fixed direction override (deg) as {"ell": ..., "b": ...}
            "direction": p.get("direction"),
        },
        "vector_radial_uniform": lambda p: {"type": "vector_radial_uniform", "nval": len(p["rknot"]), "low": p["low"], "high": p["high"], "half_sky": p.get("half_sky", False)},  # noqa
        "vector_radial_spline_uniform": lambda p: {"type": "vector_radial_uniform", "nval": len(p["rknot"]), "low": p["low"], "high": p["high"], "half_sky": p.get("half_sky", False)},  # noqa (alias for vector_radial_uniform)
        "vector_radial_spline_uniform_fixed_direction": lambda p: {"type": "vector_radial_spline_uniform_fixed_direction", "nval": len(p["rknot"]) if "rknot" in p else None, "low": p["low"], "high": p["high"]},  # noqa
        "vector_components_uniform": lambda p: {"type": "vector_components_uniform", "low": p["low"], "high": p["high"],},  # noqa
        "quadrupole_uniform_fixed": lambda p: {"type": "quadrupole_uniform_fixed", "low": p["low"], "high": p["high"],},  # noqa
        "vector_radialmag_uniform": lambda p: {
            "type": "vector_radialmag_uniform",
            "nval": len(p["rknot"]),
            "low": p["low"],
            "high": p["high"],
            # Optional fixed direction override (deg) as {"ell": ..., "b": ...}
            "direction": p.get("direction"),
            # Optional per-knot max drawn from either `max_modulus` or an
            # equivalent H0 dipole percent.
            "max_modulus": (
                h0_percent_to_bulkflow(
                    p["rknot"],
                    p["h0_dipole_percent"],
                ) if "h0_dipole_percent" in p else p.get("max_modulus")
            ),
            # Optional smoothness prior (km/s scale for penalizing knot differences)
            "smoothness_scale": p.get("smoothness_scale"),
            "smoothness_threshold": p.get("smoothness_threshold"),
            # Sample in galactic coordinates (ell, b) instead of ICRS
            "sample_galactic": p.get("sample_galactic", False),
            # Restrict ell to [0, 180°] to break sign degeneracy (implies sample_galactic)
            "half_sky": p.get("half_sky", False),
            # Optional dict mapping knot index -> fixed value (e.g., {0: 0.0})
            "fixed_knots": p.get("fixed_knots"),
        },  # noqa
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
    if isinstance(dist, Delta) and name == "zeropoint_dipole":
        return dist.v

    if isinstance(dist, Delta):
        return deterministic(name, dist.v)

    if isinstance(dist, dict) and dist.get("type") == "vector_uniform":
        return sample_vector(name, dist["low"], dist["high"])

    if isinstance(dist, dict) and dist.get("type") == "vector_uniform_fixed":
        return sample_vector_fixed(
            name, dist["low"], dist["high"], direction=dist.get("direction"))

    if isinstance(dist, dict) and dist.get("type") == "vector_components_uniform":  # noqa
        return sample_vector_components_uniform(
            name, dist["low"], dist["high"])
    
    # New quadrupole sampling
    if isinstance(dist, dict) and dist.get("type") == "quadrupole_uniform_fixed":  # noqa
        return sample_quadrupole_fixed(name, dist["low"], dist["high"])

    if isinstance(dist, dict) and dist.get("type") == "vector_radial_uniform":  # noqa
        return sample_spline_radial_vector(
            name, dist["nval"], dist["low"], dist["high"],
            half_sky=dist.get("half_sky", False))

    if isinstance(dist, dict) and dist.get("type") == "vector_radial_spline_uniform":  # noqa
        return sample_spline_radial_vector(
            name, dist["nval"], dist["low"], dist["high"],
            half_sky=dist.get("half_sky", False))
    
    if isinstance(dist, dict) and dist.get("type") == "vector_radial_spline_uniform_fixed_direction":  # noqa
        return sample_radial_spline_uniform_fixed_direction(
            name, dist["nval"], dist["low"], dist["high"])
    
    if isinstance(dist, dict) and dist.get("type") == "vector_radialmag_uniform":  # noqa
        return sample_radialmag_vector(
            name, dist["nval"], dist["low"], dist["high"],
            max_modulus=dist.get("max_modulus"),
            direction=dist.get("direction"),
            smoothness_scale=dist.get("smoothness_scale"),
            smoothness_threshold=dist.get("smoothness_threshold"),
            sample_galactic=dist.get("sample_galactic", False),
            half_sky=dist.get("half_sky", False),
            fixed_knots=dist.get("fixed_knots"))

    if isinstance(dist, dict) and dist.get("type") == "array_uniform":
        nval = dist.get("nval")
        if nval is None:
            raise ValueError(f"`nval` must be set for array_uniform '{name}'")
        with plate(f"{name}_plate", nval):
            return sample(name, Uniform(dist["low"], dist["high"]))


    return sample(name, dist)


def rsample(name, dist, shared_params=None):
    """Sample a parameter from `dist`, unless provided in `shared_params`.

    If shared_params is provided and name is not yet in it, the sampled value
    is stored for future calls (enabling sharing across submodels).
    """
    if shared_params is not None:
        if name in shared_params:
            return shared_params[name]
        # Sample and store for future reuse
        value = _rsample(name, dist)
        shared_params[name] = value
        return value
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

        kind = get_nested(config, "pv_model/kind", "")
        kind_allowed = ["Vext", "Vext_radial"]
        if kind not in kind_allowed and not kind.startswith("precomputed_los_"):  # noqa
            raise ValueError(
                f"Invalid kind '{kind}'. Must be one of {kind_allowed} or "
                "start with 'precomputed_los_'.")

        self.track_log_density_per_sample = get_nested(
            config, "inference/track_log_density_per_sample", False)
        self.save_distances = get_nested(
            config, "inference/save_distances", False)

        # Initialize interpolators for distance and redshift
        self.Om = get_nested(config, "model/Om", 0.3)
        self.distance2distmod = Distance2Distmod(Om0=self.Om)
        self.distance2redshift = Distance2Redshift(Om0=self.Om)

        priors = config["model"]["priors"]

        self.which_Vext = get_nested(config, "pv_model/which_Vext", "constant")

        # Map which_Vext values to prior key names (some differ)
        vext_prior_key_map = {
            "radial": "Vext_radial",
            "radial_magnitude": "Vext_radmag",
        }

        if self.which_Vext in ["radial", "radial_magnitude"]:
            prior_key = vext_prior_key_map[self.which_Vext]
            d = priors[prior_key]
            rknot = d.get("rknot")
            if rknot is None:
                raise KeyError(
                    f"`model/priors/Vext_{self.which_Vext}` must define `rknot`.")
            fprint(f"using radial `Vext` with spline knots at {rknot}")
            # Always carry knots and optionally pass through interpolation kwargs.
            self.kwargs_Vext = {"rknot": rknot}
            for opt_key in ("method", "k", "endpoints"):
                if opt_key in d:
                    self.kwargs_Vext[opt_key] = d[opt_key]
            if self.which_Vext == "radial_magnitude":
                # Magnitude interpolation defaults to cubic if unspecified.
                self.kwargs_Vext.setdefault("method", "cubic")
        elif self.which_Vext == "radial_binned":
            bin_edges = get_nested(config, "pv_model/Vext_radial_bin_edges", None)
            if bin_edges is None:
                raise ValueError(
                    "Must specify `Vext_radial_bin_edges` in config when "
                    "`which_Vext = 'radial_binned'`.")
            n_bins = len(bin_edges) - 1
            fprint(f"using radial_binned `Vext` with {n_bins} bins: {bin_edges}.")
            self.kwargs_Vext = {
                "n_bins": n_bins,
                "bin_edges": bin_edges}
        elif self.which_Vext == "per_pix":
            nside = get_nested(config, "pv_model/Vext_per_pix_nside", None)
            if nside is None:
                raise ValueError(
                    "Must specify `Vext_per_pix_nside` in config when "
                    "`which_Vext = 'per_pix'`.")
            if not (nside > 0 and ((nside & (nside - 1)) == 0)):
                raise ValueError(
                    f"Invalid nside={nside} in "
                    f"which_Vext = '{self.which_Vext}'. "
                    "Must be a positive power of 2.")
            fprint(f"using per-pixel `Vext` at nside={nside}.")
            npix = 12 * nside**2
            self.kwargs_Vext = {
                "nside": nside, "npix": npix,
                "Q": jnp.asarray(sumzero_basis(npix))}
        elif self.which_Vext == "constant":
            self.which_Vext = "constant"
            self.kwargs_Vext = {}
        else:
            raise ValueError(f"Invalid which_Vext '{self.which_Vext}'.")

        self.priors, self.prior_dist_name = load_priors(priors)
        self.use_MNR = get_nested(config, "model/use_MNR", False)
        self.marginalize_eta = get_nested(
            config, "model/marginalize_eta", True)
        if self.marginalize_eta:
            self.eta_grid_kwargs = config["model"]["eta_grid"]
            fprint(
                "marginalizing eta with "
                f"k_sigma = {self.eta_grid_kwargs['k_sigma']} and "
                f"n_grid = {self.eta_grid_kwargs['n_grid']} (if TFR).")

        self.galaxy_bias = config["pv_model"]["galaxy_bias"]
        if self.galaxy_bias not in ["unity", "powerlaw", "linear",
                                    "linear_from_beta",
                                    "linear_from_beta_stochastic",
                                    "double_powerlaw"]:
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
    if galaxy_bias == "unity":
        return [1.,]
    if galaxy_bias == "powerlaw":
        alpha = rsample("alpha", priors["alpha"], shared_params)
        bias_params = [alpha,]
    elif galaxy_bias == "linear":
        b1 = rsample("b1", priors["b1"], shared_params)
        bias_params = [b1,]
    elif galaxy_bias == "linear_from_beta":
        b1 = kwargs["Om"]**0.55 / kwargs["beta"]
        bias_params = [b1,]
    elif galaxy_bias == "linear_from_beta_stochastic":
        b1_mean = kwargs["Om"]**0.55 / kwargs["beta"]
        delta_b1 = rsample("delta_b1_skipZ", priors["delta_b1"], shared_params)
        b1 = deterministic("b1", b1_mean + delta_b1)
        bias_params = [b1,]
    elif galaxy_bias == "double_powerlaw":
        alpha_low = rsample("alpha_low", priors["alpha_low"], shared_params)
        alpha_high = rsample("alpha_high", priors["alpha_high"], shared_params)
        log_rho_t = rsample("log_rho_t", priors["log_rho_t"], shared_params)
        bias_params = [alpha_low, alpha_high, log_rho_t]

    else:
        raise ValueError(f"Invalid galaxy bias model '{galaxy_bias}'.")

    return bias_params


def lp_galaxy_bias(delta, log_rho, bias_params, galaxy_bias):
    """
    Given the galaxy bias probabibility, given some density and a bias model.
    """
    if galaxy_bias == "powerlaw":
        lp = bias_params[0] * log_rho
    elif galaxy_bias == "double_powerlaw":
        alpha_low, alpha_high, log_rho_t = bias_params
        log_x = log_rho - log_rho_t
        lp = (alpha_low * log_x
              + (alpha_high - alpha_low) * jnp.logaddexp(0.0, log_x))
    elif "linear" in galaxy_bias or galaxy_bias == "unity":
        lp = jnp.log(smoothclip_nr(1 + bias_params[0] * delta, tau=0.1))
    else:
        raise ValueError(f"Invalid galaxy bias model '{galaxy_bias}'.")

    return lp


def compute_Vext_radial(data, r_grid, Vext, which_Vext, **kwargs_Vext):
    """
    Compute the line-of-sight projection of the external velocity.

    Promote the final output to shape `(n_field, n_gal, n_rbins)`.
    """
    if which_Vext == "radial":
        # Use SLERP-based interpolation to prevent magnitude overshoot
        Vext = interp_cartesian_vector(r_grid, Vext, **kwargs_Vext)
        if Vext.ndim == 2:
            # 1D case: Vext is (3, n_rbins)
            # rhat is (n_gal, 3), result is (n_gal, n_rbins)
            Vext_rad = jnp.sum(
                data["rhat"][..., None] * Vext[None, ...], axis=1)[None, ...]
        else:
            # 2D case: Vext is (3, n_gal, n_los) from z-space iteration
            # rhat is (n_gal, 3), result is (n_gal, n_los)
            Vext_rad = jnp.einsum("gi,igk->gk", data["rhat"], Vext)[None, ...]
    elif which_Vext == "radial_binned":
        # Vext shape: (n_bins * 3,) representing (n_bins, 3) flattened
        n_bins = kwargs_Vext["n_bins"]
        Vext_3d = Vext.reshape(n_bins, 3)  # Shape: (n_bins, 3)
        # Project each bin's velocity onto LOS
        # data["rhat"]: (n_gal, 3), C_radial_bin: (n_gal, n_bins)
        # Vext_3d: (n_bins, 3)
        # Result: (n_gal,) = sum over component axis of (n_gal, n_bins) * (n_bins, 3) * (n_gal, 3)
        Vext_rad = jnp.sum(
            (data["C_radial_bin"] @ Vext_3d) * data["rhat"], axis=1)[None, :, None]
    elif which_Vext == "radial_magnitude":
        # Unpack the tuple of magnitude and direction.
        Vext_mag, rhat = Vext
        # Interpolate the magnitude as a function of radius.
        rknot = jnp.asarray(kwargs_Vext["rknot"])
        r_grid_clamped = jnp.clip(r_grid, rknot[0], rknot[-1])
        # Clamp instead of relying on interpax `extrap` because some versions
        # reject 0-D JAX scalars (e.g., Vext_mag[0]) as extrap values.

        # Handle both 1D r_grid (n_rbins,) and 2D r_grid (n_gal, n_rbins)
        # during z-space iteration.
        method = kwargs_Vext.get("method", "cubic")
        if method == "jax_linear":
            # Use JAX's built-in linear interpolation (faster on GPU)
            if r_grid_clamped.ndim == 1:
                Vext_mag_r = jnp.interp(r_grid_clamped, rknot, Vext_mag)
            else:
                # 2D case: flatten, interpolate, reshape
                orig_shape = r_grid_clamped.shape
                Vext_mag_r = jnp.interp(
                    r_grid_clamped.ravel(), rknot, Vext_mag
                ).reshape(orig_shape)
        else:
            # Use interpax for cubic or other methods
            if r_grid_clamped.ndim == 1:
                Vext_mag_r = interp1d(
                    r_grid_clamped, rknot, Vext_mag,
                    method=method, extrap=False)
            else:
                # 2D case: flatten, interpolate, reshape
                orig_shape = r_grid_clamped.shape
                Vext_mag_r = interp1d(
                    r_grid_clamped.ravel(), rknot, Vext_mag,
                    method=method, extrap=False
                ).reshape(orig_shape)

        # Project the LOS of each galaxy onto the dipole direction, shape
        # is (n_gal,).
        cos_theta = jnp.sum(data["rhat"] * rhat[None, :], axis=1)

        # Finally, the shape is (n_field, n_gal, n_rbins).
        if r_grid_clamped.ndim == 1:
            Vext_rad = (cos_theta[:, None] * Vext_mag_r[None, :])[None, :, :]
        else:
            # 2D case: Vext_mag_r is (n_gal, n_rbins), cos_theta is (n_gal,)
            Vext_rad = (cos_theta[:, None] * Vext_mag_r)[None, :, :]
    elif which_Vext == "per_pix":
        Vext_rad = (data["C_pix"] @ Vext)[None, :, None]
    elif which_Vext == "constant":
        Vext_rad = jnp.sum(data["rhat"] * Vext[None, :], axis=1)[None, :, None]
    else:
        raise ValueError(f"Invalid which_Vext '{which_Vext}'.")
    return Vext_rad


def compute_quadrupole_radial(data, quadrupole):
    """Compute the line-of-sight projection of the quadrupole.
    Q_rad = Q (q1.ni q2.ni - 1/3 q1.q2)
    where q1 and q2 are the quadrupole vectors and ni is the unit radial vector to the galaxy i.
    """
    Qq1 = quadrupole[:, 0]  # shape (3,)
    Qq2 = quadrupole[:, 1]  # shape (3,)
    
    dot1 = jnp.sum(data["rhat"] * Qq1, axis=1)  # (N, 1)
    dot2 = jnp.sum(data["rhat"] * Qq2, axis=1)  # (N, 1)

    Q_rad = dot1 * dot2  # (N, 1)
    Q_rad -= (1 / 3) * jnp.dot(Qq1, Qq2)  # scalar, broadcasted

    return Q_rad  # (N, 1)


def compute_los_zspace_to_rspace(
    data, los_r, z_grid, Vext, which_Vext, kwargs_Vext,
    redshift2distance, h, n_iterations=0
):
    """
    Map LOS quantities from z-space to r-space accounting for Vext.

    Uses (1 + z_obs) = (1 + z_cosmo) * (1 + Vext_rad/c) to convert
    observed redshift grid to cosmological redshift, then to real-space distance.

    Parameters
    ----------
    data : PVDataFrame
        Data container with galaxy directions (rhat).
    los_r : array (n_los,)
        r grid from LOS files (used for Vext evaluation).
    z_grid : array (n_los,)
        Redshift grid from LOS files.
    Vext : array
        External velocity parameters.
    which_Vext : str
        Vext mode (constant, radial, etc.).
    kwargs_Vext : dict
        Additional Vext parameters.
    redshift2distance : callable
        Redshift2Distance instance for z -> r conversion.
    h : float or array (n_gal,)
        Hubble parameter (per-galaxy for anisotropic H0).
    n_iterations : int
        Number of iterations to refine r_cosmo for radial Vext models.
        Only applies when which_Vext is 'radial' or 'radial_magnitude'.
        Default 0 means use the approximation (evaluate Vext at los_r).

    Returns
    -------
    r_cosmo : array (n_field, n_gal, n_los)
        Real-space distances corresponding to each z_grid point.
    """
    c = 299792.458  # km/s

    def _z_cosmo_to_r_cosmo(z_cosmo, h_val):
        """Convert z_cosmo to r_cosmo, handling scalar or per-galaxy h."""
        if jnp.ndim(h_val) == 0:
            # Scalar h - vectorized over all dimensions
            shape = z_cosmo.shape
            return redshift2distance(z_cosmo.ravel(), h=h_val).reshape(shape)
        else:
            # Per-galaxy h for anisotropic H0
            # h_val shape: (n_gal,), z_cosmo shape: (n_field, n_gal, n_los)
            # vmap over fields (axis 0), then over galaxies (axis 0 of inner)
            def _z_to_r_per_gal(z_line, h_gal):
                return redshift2distance(z_line, h=h_gal)

            def _process_field(z_field):
                # z_field: (n_gal, n_los), h_val: (n_gal,)
                return vmap(_z_to_r_per_gal, in_axes=(0, 0))(z_field, h_val)

            # vmap over field dimension
            return vmap(_process_field)(z_cosmo)  # (n_field, n_gal, n_los)

    # Initial computation: evaluate Vext at los_r (the approximation)
    r_eval = los_r
    Vext_rad = compute_Vext_radial(
        data, r_eval, Vext, which_Vext, **kwargs_Vext
    )  # (n_field, n_gal, n_los)

    # Convert z_obs to z_cosmo: (1 + z_obs) = (1 + z_cosmo) * (1 + Vext_rad/c)
    z_cosmo = (1 + z_grid[None, None, :]) / (1 + Vext_rad / c) - 1
    z_cosmo = jnp.maximum(z_cosmo, 1e-8)

    # Convert z_cosmo to r_cosmo (preserves all dimensions)
    r_cosmo = _z_cosmo_to_r_cosmo(z_cosmo, h)

    # Iterate to refine r_cosmo (only for radial Vext models)
    if n_iterations > 0 and which_Vext in ("radial", "radial_magnitude"):
        for _ in range(n_iterations):
            # Re-evaluate Vext at the current r_cosmo estimate
            # Use mean over fields as representative r_eval
            r_eval = jnp.mean(r_cosmo, axis=0)  # (n_gal, n_los)

            Vext_rad = compute_Vext_radial(
                data, r_eval, Vext, which_Vext, **kwargs_Vext
            )

            z_cosmo = (1 + z_grid[None, None, :]) / (1 + Vext_rad / c) - 1
            z_cosmo = jnp.maximum(z_cosmo, 1e-8)

            r_cosmo = _z_cosmo_to_r_cosmo(z_cosmo, h)

    return r_cosmo


def sample_distance_prior(priors, shared_params=None):
    """Sample hyperparameters describing the empirical distance prior."""
    return {
        "R": rsample("R_dist_emp", priors["R_dist_emp"], shared_params),
        "p": rsample("p_dist_emp", priors["p_dist_emp"], shared_params),
        "n": rsample("n_dist_emp", priors["n_dist_emp"], shared_params),
    }


def sumzero_basis(npix):
    """
    Return an orthonormal basis `(npix x (npix - 1))` for the subspace of
    vectors with zero sum.
    """
    one = jnp.ones((npix,)) / jnp.sqrt(npix)
    e1 = jnp.zeros((npix,)).at[0].set(1.0)
    v = one - e1
    beta = 2.0 / jnp.dot(v, v)
    H = jnp.eye(npix) - beta * jnp.outer(v, v)
    Q = H[:, 1:]
    return Q


def sample_Vext(priors, which_Vext, shared_params=None, kwargs_Vext={}):
    if which_Vext == "radial":
        Vext = rsample("Vext_radial", priors["Vext_radial"], shared_params)
    elif which_Vext == "radial_magnitude":
        Vext = rsample(
            "Vext_radmag", priors["Vext_radmag"],
            shared_params)
    elif which_Vext == "radial_binned":
        prior = priors["Vext_radial_binned"]
        
        # For vector_radial_spline_uniform_fixed_direction, inject n_bins if not provided
        if isinstance(prior, dict) and prior.get("type") == "vector_radial_spline_uniform_fixed_direction":
            if prior.get("nval") is None:
                prior["nval"] = kwargs_Vext["n_bins"]
            # Sample directly - the function handles the plate internally
            Vext = rsample("Vext_rad_bin", prior, shared_params)
            # Flatten to shape (n_bins * 3,) for consistency with compute_Vext_radial
            Vext = Vext.reshape(-1)
        else:
            # For other prior types (e.g., vector_uniform), sample one vector per bin
            with plate("Vext_rad_bin_plate", kwargs_Vext["n_bins"]):
                Vext = rsample("Vext_rad_bin", prior, shared_params)
            # Flatten to shape (n_bins * 3,) for consistency with compute_Vext_radial
            Vext = Vext.reshape(-1)
    elif which_Vext == "per_pix":
        # Sample scale parameter and (npix-1) standard normals, project onto
        # sum-to-zero subspace via orthonormal basis Q
        Vext_sigma = rsample("Vext_sigma", priors["Vext_sigma"], shared_params)
        npix = kwargs_Vext["npix"]
        with plate("Vext_pix_plate", npix - 1):
            u = rsample("Vext_pix_u", Normal(0., 1.), shared_params)
        # Compute Vext_pix inline (not as deterministic to avoid singular
        # covariance in evidence calculation). Store in shared_params for
        # sharing between submodels.
        Vext = Vext_sigma * (kwargs_Vext["Q"] @ u)
        if shared_params is not None:
            shared_params["Vext_pix"] = Vext
    elif which_Vext == "constant":
        Vext = rsample("Vext", priors["Vext"], shared_params)
    else:
        raise ValueError(f"Invalid which_Vext '{which_Vext}'.")

    return Vext


def sample_A_clusters(priors, which_zeropoint, shared_params=None, kwargs_zeropoint={}):
    """
    Sample zeropoint parameters for clusters, supporting per-pixel and radial binned variation.

    Note: Uses 'zeropoint_pix' as the sample name, with backward compatibility for 'A_pix'.

    Parameters
    ----------
    priors : dict
        Prior distributions.
    which_zeropoint : str
        Mode: "per_pix", "radial_binned", "radial_binned_dipole", or "constant".
    shared_params : dict, optional
        Shared parameters for joint inference.
    kwargs_zeropoint : dict, optional
        Additional arguments (e.g., n_bins for radial modes).
    """
    if which_zeropoint == "per_pix":
        # Prefer zeropoint_pix, fall back to legacy A_pix for backward compatibility
        prior_key = "zeropoint_pix" if "zeropoint_pix" in priors else "A_pix"
        zp_pix = rsample("zeropoint_pix", priors[prior_key], shared_params)
        return zp_pix
    elif which_zeropoint == "radial_binned":
        # Sample one zeropoint value per radial bin
        with plate("zeropoint_radial_bin_plate", kwargs_zeropoint["n_bins"]):
            zp_radial_bin = rsample("zeropoint_radial_bin", priors["A_LT"], shared_params)
        return zp_radial_bin
    elif which_zeropoint == "radial_binned_dipole":
        # Sample a 3D dipole vector per radial bin
        with plate("zeropoint_dipole_radial_bin_plate", kwargs_zeropoint["n_bins"]):
            zp_dipole_radial_bin = rsample(
                "zeropoint_dipole_radial_bin", priors["zeropoint_dipole"], shared_params)
        # Flatten to shape (n_bins * 3,) for consistency with compute_A_clusters_radial
        zp_dipole_radial_bin = zp_dipole_radial_bin.reshape(-1)
        return zp_dipole_radial_bin
    elif which_zeropoint == "constant":
        # Use regular zeropoint from priors (will be handled by rsample in main model)
        return None
    else:
        raise ValueError(f"Invalid which_zeropoint '{which_zeropoint}'.")


def compute_A_clusters_radial(data, zp_pix, which_zeropoint, **kwargs_zeropoint):
    """
    Compute the per-pixel or radial binned zeropoint variation.
    Returns per-galaxy zeropoint offset values: shape (n_gal,).

    Parameters
    ----------
    zp_pix : array
        Zeropoint values (per-pixel or per-bin).
    which_zeropoint : str
        Mode: "per_pix", "radial_binned", "radial_binned_dipole", or "constant".
    """
    if which_zeropoint == "per_pix":
        # zp_pix has shape (npix,), data["C_pix"] has shape (n_gal, npix)
        zp_radial = data["C_pix"] @ zp_pix  # shape (n_gal,)
        return zp_radial
    elif which_zeropoint == "radial_binned":
        # zp_pix has shape (n_bins,), data["C_A_radial_bin"] has shape (n_gal, n_bins)
        zp_radial = data["C_A_radial_bin"] @ zp_pix  # shape (n_gal,)
        return zp_radial
    elif which_zeropoint == "radial_binned_dipole":
        # zp_pix has shape (n_bins * 3,), need to reshape to (n_bins, 3)
        n_bins = kwargs_zeropoint["n_bins"]
        zp_dipole_3d = zp_pix.reshape(n_bins, 3)  # shape (n_bins, 3)
        # Project each bin's dipole onto line of sight for each galaxy
        # data["C_A_radial_bin"] @ zp_dipole_3d gives (n_gal, 3)
        zp_dipole_per_gal = data["C_A_radial_bin"] @ zp_dipole_3d  # shape (n_gal, 3)
        # Dot with rhat to get radial component
        zp_radial = jnp.sum(zp_dipole_per_gal * data["rhat"], axis=1)  # shape (n_gal,)
        return zp_radial
    elif which_zeropoint == "constant":
        return 0.0  # No per-pixel or radial variation
    else:
        raise ValueError(f"Invalid which_zeropoint '{which_zeropoint}'.")


###############################################################################
#                              TFR models                                     #
###############################################################################

def log_p_S_TFR_eta(eta_mean, w_eta, e_eta, eta_min, eta_max, ):
    """
    Compute the fraction of samples given a truncation in linewidth
    distribution, whose hyperprior is assumed to be Gaussian.
    """
    denom = jnp.sqrt(e_eta**2 + w_eta**2)
    if eta_min is not None and eta_max is not None:
        a = jax_norm_cdf((eta_max - eta_mean) / denom)
        b = jax_norm_cdf((eta_min - eta_mean) / denom)
        p = a - b
    elif eta_max is not None:
        p = jax_norm_cdf((eta_max - eta_mean) / denom)
    elif eta_min is not None:
        p = jax_norm_cdf((eta_mean - eta_min) / denom)
    else:
        raise ValueError("Invalid eta_min/eta_max configuration.")

    return jnp.log(jnp.clip(p, 1e-300, 1.0))  # guard against log(0)


class TFRModel(BaseModel):
    """
    A TFR forward model, distance is numerically marginalized out at each MCMC
    step instead of being sampled as a latent variable.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        if self.use_MNR and not self.marginalize_eta:
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

    def __call__(self, data, shared_params=None):
        nsamples = len(data)
        # Initialize log density tracker; not to be tracked by NumPyro with
        # `factor`.
        if self.track_log_density_per_sample:
            log_density_per_sample = jnp.zeros(nsamples)

        # Sample the TFR parameters.
        a_TFR = rsample("a_TFR", self.priors["TFR_zeropoint"], shared_params)
        b_TFR = rsample("b_TFR", self.priors["TFR_slope"], shared_params)
        c_TFR = rsample("c_TFR", self.priors["TFR_curvature"], shared_params)
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)
        a_TFR_dipole = rsample(
            "zeropoint_dipole", self.priors["zeropoint_dipole"], shared_params)
        a_TFR = a_TFR + jnp.sum(a_TFR_dipole * data["rhat"], axis=1)
        kwargs_dist = sample_distance_prior(self.priors, shared_params)

        # For the distance marginalization, h is not sampled.
        h = 1.

        if data.sample_dust:
            Rdust = rsample("R_dust", self.priors["Rdust"], shared_params)
            Ab = Rdust * data["ebv"]
        else:
            Ab = 0.

        # Sample velocity field parameters.
        Vext = sample_Vext(
            self.priors, self.which_Vext, shared_params, self.kwargs_Vext)
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
                    lp_eta += Normal(
                        eta_grid, data["e_eta"][:, None]).log_prob(
                            data["eta"][:, None])
                else:
                    # Sample the galaxy linewidth from a Gaussian hyperprior.
                    eta = sample(
                        "eta_latent", Normal(eta_prior_mean, eta_prior_std))
                    sample("eta", Normal(eta, data["e_eta"]), obs=data["eta"])

                    if self.track_log_density_per_sample:
                        log_density_per_sample += Normal(
                            eta_prior_mean, eta_prior_std).log_prob(eta)
                        log_density_per_sample += Normal(
                            eta, data["e_eta"]).log_prob(data["eta"])

                # Track the likelihood of the observed linewidths.
                if data.add_eta_truncation:
                    neglog_pS = -log_p_S_TFR_eta(
                        eta_prior_mean, eta_prior_std, data["e_eta"],
                        data.eta_min, data.eta_max)

                    factor("neg_log_S_eta", neglog_pS)
                    if self.track_log_density_per_sample:
                        log_density_per_sample += neglog_pS

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
                Vrad = beta * los_velocity_r_grid
                # Add inhomogeneous Malmquist bias and normalize the r prior
                lp_dist += lp_galaxy_bias(
                    los_delta_r_grid,
                    los_log_density_r_grid,
                    bias_params, self.galaxy_bias
                    )
                lp_dist -= ln_simpson(
                    lp_dist, x=r_grid[None, None, :], axis=-1)[..., None]
            else:
                Vrad = 0.

            # Likelihood of the observed redshifts, `(n_field, n_gal, n_rbins)`
            Vext_rad = compute_Vext_radial(
                data, r_grid, Vext, which_Vext=self.which_Vext,
                **self.kwargs_Vext)
            # deterministic("Vext_rad", Vext_rad[0, :, 0])
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
            ll = logsumexp(ll, axis=0) - jnp.log(data.num_fields)
            factor("ll_obs", ll)

            if self.track_log_density_per_sample:
                log_density_per_sample += ll
                deterministic("log_density_per_sample", log_density_per_sample)


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

        if self.track_log_density_per_sample:
            raise NotImplementedError(
                "`track_log_density_per_sample` is not implemented "
                "for `SNModel`.")

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
        kwargs_dist = sample_distance_prior(self.priors, shared_params)

        # --- Velocity field / selection nuisance ---
        Vext = sample_Vext(
            self.priors, self.which_Vext, shared_params, self.kwargs_Vext)
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
            rho = sample("rho_corr", Uniform(-1, 1))

            mu = jnp.array([x1_prior_mean, c_prior_mean])
            cov = jnp.array([
                [x1_prior_std**2, rho * x1_prior_std * c_prior_std],
                [rho * x1_prior_std * c_prior_std, c_prior_std**2]])

        with plate("data", nsamples):
            if self.use_MNR:
                x_latent = sample("x_latent", MultivariateNormal(mu, cov))
                x1 = x_latent[:, 0]
                c = x_latent[:, 1]

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
                data, r_grid, Vext, which_Vext=self.which_Vext,
                **self.kwargs_Vext)
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

        if self.which_Vext != "constant":
            raise NotImplementedError("Only constant Vext is implemented for "
                                      "the `PantheonPlusModel`.")

        if self.track_log_density_per_sample:
            raise NotImplementedError(
                "`track_log_density_per_sample` is not implemented "
                "for `PantheonPlusModel`.")

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

        kwargs_dist = sample_distance_prior(self.priors, shared_params)

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


DEFAULT_A_PRIOR = {"dist": "uniform", "low": -0.5, "high": 2.5}
DEFAULT_B_PRIOR = {"dist": "uniform", "low": 1.5, "high": 3.0}


def _ensure_scaling_priors(priors):
    """
    Ensure that the cluster scaling relation priors include the per-relation
    keys used by the modern models (A_LT, A_YT, ...). Falls back to the legacy
    CL_* entries or defaults when necessary and keeps compatibility by
    re-populating the CL_* entries if missing.
    """
    template_A = (priors.get("A_LT") or priors.get("A_YT")
                  or priors.get("CL_A") or DEFAULT_A_PRIOR)
    template_B = (priors.get("B_LT") or priors.get("B_YT")
                  or priors.get("CL_B") or DEFAULT_B_PRIOR)

    for name in ("A_LT", "A_YT"):
        if name not in priors:
            priors[name] = deepcopy(template_A)
    for name in ("B_LT", "B_YT"):
        if name not in priors:
            priors[name] = deepcopy(template_B)

    priors.setdefault("CL_A", deepcopy(priors["A_LT"]))
    priors.setdefault("CL_B", deepcopy(priors["B_LT"]))


class ClustersModel(BaseModel):
    """Cluster scaling relations with explicit distance marginalization."""

    _VALID_RELATIONS = {"LT", "LY", "LTY", "YT", "YTL", "LTYT"}

    def __init__(self, config_path):
        super().__init__(config_path)

        _ensure_scaling_priors(self.priors)
        (self.relation_by_catalogue,
         self.default_relation) = self._build_relation_lookup()
        self.used_relations = set(self.relation_by_catalogue.values())
        # Backwards compatibility for code that inspects this attribute.
        self.which_relation = self.default_relation
        self.apply_Ez_correction = bool(
            get_nested(self.config, "io/apply_Ez_correction", True))
        self.n_zspace_iterations = int(
            get_nested(self.config, "pv_model/n_zspace_iterations", 0))
        self.density_prior = bool(
            get_nested(self.config, "pv_model/density_prior", True))

        # Zeropoint parameterization: "fracH0" or "magnitude"
        # - "fracH0": prior is flat in fractional δH, converted to magnitude
        # - "magnitude": prior is flat in magnitude ΔA (legacy)
        self.zeropoint_parameterization = get_nested(
            self.config, "pv_model/zeropoint_parameterization", "magnitude")

        # Single shared nside for per-pixel anisotropy modes (zeropoint and H0)
        self.anisotropy_per_pix_nside = get_nested(
            self.config, "pv_model/anisotropy_per_pix_nside", 1)

        # Detect varying H0 parameters (cosmological anisotropy - affects z→r)
        self._has_varying_H0_dipole = self._prior_is_varying("H0_dipole")
        self._has_varying_H0_quad = self._prior_is_varying("H0_quad")
        self.which_H0 = get_nested(self.config, "pv_model/which_H0", "constant")
        self._has_varying_H0_pix = (self.which_H0 == "per_pix")
        self._has_varying_H0 = (
            self._has_varying_H0_dipole or
            self._has_varying_H0_quad or
            self._has_varying_H0_pix
        )

        # Detect varying zeropoint parameters (calibration - doesn't affect z→r)
        self._has_varying_zeropoint_dipole = self._prior_is_varying("zeropoint_dipole")
        self._has_varying_zeropoint_quad = self._prior_is_varying("zeropoint_quad")
        self.which_zeropoint = get_nested(
            self.config, "pv_model/which_zeropoint",
            get_nested(self.config, "pv_model/which_A", "constant"))  # Backward compat
        self._has_varying_zeropoint_pix = (self.which_zeropoint == "per_pix")

        # Detect varying Vext
        self._has_varying_Vext = (
            self.which_Vext != "constant" or
            self._prior_is_varying("Vext")
        )

        # Auto-detect if z-space mapping is needed
        # Required when H0 or Vext is varying (affects z→r conversion)
        self._needs_zspace = self._has_varying_H0 or self._has_varying_Vext

        # Backward compatibility: still support use_zspace config flag
        use_zspace_config = get_nested(self.config, "pv_model/use_zspace", None)
        if use_zspace_config is not None:
            self._needs_zspace = bool(use_zspace_config) or self._needs_zspace

        # Also support legacy stretch_los_with_zeropoint flag
        # If set, treat varying zeropoint as H0 anisotropy (affects z→r)
        self._legacy_stretch_mode = bool(
            get_nested(self.config, "pv_model/stretch_los_with_zeropoint", False))
        if self._legacy_stretch_mode and self._has_varying_zeropoint_dipole:
            self._needs_zspace = True

        # Backward compat attribute
        self.has_varying_zeropoint_dipole = self._has_varying_zeropoint_dipole

        if self._needs_zspace:
            self.redshift2distance = Redshift2Distance(Om0=self.Om)

        self.distance2logda = Distance2LogAngDist(Om0=self.Om, zmax_interp=1.0)
        self.distance2logdl = Distance2LogLumDist(Om0=self.Om, zmax_interp=1.0)

        if {"LT", "YT", "LTYT"} & self.used_relations:
            self.priors["CL_C"] = Delta(jnp.asarray(0.0))
        if "LY" in self.used_relations:
            self.priors["CL_B"] = Delta(jnp.asarray(0.0))

        # Configure per-pixel H0 anisotropy
        if self.which_H0 == "per_pix":
            nside = self.anisotropy_per_pix_nside
            npix = 12 * nside**2
            fprint(f"using per-pixel `H0` at nside={nside}.")
            self.kwargs_H0 = {"nside": nside, "npix": npix}
        else:
            self.kwargs_H0 = {}

        # Configuration for per-pixel zeropoint variation (similar to Vext)
        # Support both new 'which_zeropoint' and legacy 'which_A'
        self.which_A = self.which_zeropoint  # Backward compat alias

        if self.which_zeropoint == "per_pix":
            # Use shared anisotropy nside
            nside = self.anisotropy_per_pix_nside
            # Also check legacy config
            nside_legacy = get_nested(self.config, "pv_model/A_per_pix_nside", None)
            if nside_legacy is not None:
                nside = nside_legacy
            if not (nside > 0 and ((nside & (nside - 1)) == 0)):
                raise ValueError(
                    f"Invalid nside={nside} in "
                    f"which_zeropoint = '{self.which_zeropoint}'. "
                    "Must be a positive power of 2.")
            fprint(f"using per-pixel `zeropoint` at nside={nside}.")
            npix = 12 * nside**2
            self.kwargs_zeropoint = {
                "nside": nside, "npix": npix,
                "Q": jnp.asarray(sumzero_basis(npix))}
        elif self.which_zeropoint == "radial_binned":
            bin_edges = get_nested(self.config, "pv_model/A_radial_bin_edges", None)
            if bin_edges is None:
                raise ValueError(
                    "Must specify `A_radial_bin_edges` in config when "
                    "`which_zeropoint = 'radial_binned'`.")
            bin_edges = jnp.asarray(bin_edges)
            n_bins = len(bin_edges) - 1
            fprint(f"using radial binned `zeropoint` with {n_bins} bins.")
            self.kwargs_zeropoint = {"n_bins": n_bins, "bin_edges": bin_edges}
        elif self.which_zeropoint == "radial_binned_dipole":
            bin_edges = get_nested(self.config, "pv_model/A_radial_bin_edges", None)
            if bin_edges is None:
                raise ValueError(
                    "Must specify `A_radial_bin_edges` in config when "
                    "`which_zeropoint = 'radial_binned_dipole'`.")
            bin_edges = jnp.asarray(bin_edges)
            n_bins = len(bin_edges) - 1
            fprint(f"using radial binned dipole `zeropoint` with {n_bins} bins.")
            self.kwargs_zeropoint = {"n_bins": n_bins, "bin_edges": bin_edges}
        elif self.which_zeropoint == "constant":
            self.kwargs_zeropoint = {}
        else:
            raise ValueError(f"Invalid which_zeropoint '{self.which_zeropoint}'.")

        # Backward compat alias
        self.kwargs_A = self.kwargs_zeropoint

        if self.use_MNR:
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

    def _prior_is_varying(self, prior_name):
        """Check if a prior is varying (not a delta function)."""
        prior = self.priors.get(prior_name, None)
        if prior is None:
            return False
        if isinstance(prior, Delta):
            return False
        if isinstance(prior, dict):
            return prior.get("type") != "delta" and prior.get("dist") != "delta"
        return True

    def _validate_relation(self, relation):
        if relation not in self._VALID_RELATIONS:
            raise ValueError(
                f"Invalid scaling relation '{relation}'. "
                "Choose either 'LT', 'LY', 'LTY', 'YT', 'YTL', or 'LTYT'.")

    def _get_catalogue_names(self, io_cfg):
        names = io_cfg.get("catalogue_name", "Clusters")
        if isinstance(names, str) or names is None:
            return [names or "Clusters"]
        return list(names)

    def _build_relation_lookup(self):
        io_cfg = self.config.get("io", {})
        catalogue_names = self._get_catalogue_names(io_cfg)

        default_relation = io_cfg.get("Clusters", {}).get("which_relation")
        if default_relation is None:
            default_relation = "LT"
        self._validate_relation(default_relation)

        relations = {}
        for name in catalogue_names:
            section = io_cfg.get(name, {})
            relation = section.get("which_relation", default_relation)
            if relation is None:
                relation = default_relation
            self._validate_relation(relation)
            relations[name] = relation

        if not relations:
            relations["Clusters"] = default_relation
            catalogue_names = ["Clusters"]

        self.catalogue_names = list(catalogue_names)
        first = catalogue_names[0]
        return relations, relations.get(first, default_relation)

    def _relation_for_dataset(self, data):
        name = getattr(data, "name", None)
        if name in self.relation_by_catalogue:
            return self.relation_by_catalogue[name]
        return self.default_relation

    def __call__(self, data, shared_params=None):
        nsamples = len(data)
        if self.track_log_density_per_sample:
            log_density_per_sample = jnp.zeros(nsamples)

        relation = self._relation_for_dataset(data)
        # Keep attribute for backward compatibility (e.g., logging/introspection)
        self.which_relation = relation

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for "
                "`ClustersModel`.")

        # Sample the cluster scaling parameters.
        use_LT_branch = relation in ["LT", "LTY", "LTYT"]
        use_YT_branch = relation in ["YT", "YTL", "LTYT"]

        A_LT = B_LT = None
        A_YT = B_YT = None
        sigma_LT = sigma_YT = None
        A_dipole = None

        if use_LT_branch:
            A_LT = rsample("A_LT", self.priors["A_LT"], shared_params)
            B_LT = rsample("B_LT", self.priors["B_LT"], shared_params)
            sigma_LT = rsample("sigma_LT", self.priors["sigma_int"], shared_params)
        if use_YT_branch:
            A_YT = rsample("A_YT", self.priors["A_YT"], shared_params)
            B_YT = rsample("B_YT", self.priors["B_YT"], shared_params)
            sigma_YT = rsample("sigma_YT", self.priors["sigma_int"], shared_params)

        C = rsample("C_CL", self.priors["CL_C"], shared_params)

        # =================================================================
        # Sample H0 anisotropy parameters (always fractional δH)
        # These affect z→r mapping (cosmological anisotropy)
        # =================================================================
        H0_dipole = None
        H0_quad = None
        H0_pix = None
        h_per_gal = None  # Will be set if H0 is varying

        if "H0_dipole" in self.priors:
            H0_dipole = rsample("H0_dipole", self.priors["H0_dipole"], shared_params)
        if "H0_quad" in self.priors:
            H0_quad = rsample("H0_quad", self.priors["H0_quad"], shared_params)
        if self.which_H0 == "per_pix" and "H0_pix" in self.priors:
            H0_pix = rsample("H0_pix", self.priors["H0_pix"], shared_params)

        # =================================================================
        # Sample zeropoint parameters (calibration offsets)
        # These only affect the likelihood, not z→r mapping
        # =================================================================
        zp_dipole_sampled = None
        zp_quad_sampled = None
        zp_pix_sampled = None
        A_dipole = None  # For backward compatibility

        delta_A = None
        if self.which_zeropoint in ["per_pix", "radial_binned", "radial_binned_dipole"]:
            zp_pix_sampled = sample_A_clusters(
                self.priors, self.which_zeropoint, shared_params, self.kwargs_zeropoint)
            delta_A = compute_A_clusters_radial(
                data, zp_pix_sampled, self.which_zeropoint, **self.kwargs_zeropoint)
        else:
            # Traditional dipole/quadrupole approach
            zp_dipole_sampled = rsample(
                "zeropoint_dipole", self.priors["zeropoint_dipole"], shared_params)
            zp_quad_sampled = rsample(
                "zeropoint_quad", self.priors["zeropoint_quad"], shared_params)
            A_dipole = zp_dipole_sampled  # Backward compat alias

        # =================================================================
        # Compute effective zeropoint magnitude for likelihood
        # =================================================================
        # Determine the effective zeropoint based on the mode:
        # - dipH0: H0 affects z→r, auto-derive zeropoint from H0
        # - dipA: Zeropoint affects likelihood only
        # - Legacy stretch mode: Treat zeropoint as H0

        if self._has_varying_H0_dipole and H0_dipole is not None:
            # dipH0 run: H0 affects z→r mapping
            # Auto-derive zeropoint magnitude from H0 for the likelihood
            H0_mag = jnp.linalg.norm(H0_dipole)
            H0_dir = H0_dipole / jnp.maximum(H0_mag, 1e-30)
            zp_dipole_mag = _frac_to_mag(H0_mag) * H0_dir

            # Compute radial projection for likelihood
            zp_dipole_radial = jnp.sum(zp_dipole_mag * data["rhat"], axis=1)
            delta_A = zp_dipole_radial if delta_A is None else delta_A + zp_dipole_radial

        elif self._legacy_stretch_mode and self._has_varying_zeropoint_dipole:
            # Legacy stretch mode: treat varying zeropoint as H0 anisotropy
            # The z→r mapping will use the zeropoint, so no extra delta_A here
            # (handled in the z-space section below)
            if zp_dipole_sampled is not None:
                zp_dipole_radial = jnp.sum(zp_dipole_sampled * data["rhat"], axis=1)
                zp_quad_radial = compute_quadrupole_radial(data, zp_quad_sampled) if zp_quad_sampled is not None else 0.0
                delta_A = zp_dipole_radial + zp_quad_radial

        elif self._has_varying_zeropoint_dipole and zp_dipole_sampled is not None:
            # dipA run: zeropoint affects likelihood only
            if self.zeropoint_parameterization == "fracH0":
                # Sampled as fractional δH, convert to magnitude
                zp_frac_mag = jnp.linalg.norm(zp_dipole_sampled)
                zp_frac_dir = zp_dipole_sampled / jnp.maximum(zp_frac_mag, 1e-30)
                zp_dipole_mag = _frac_to_mag(zp_frac_mag) * zp_frac_dir
            else:
                # Sampled as magnitude, use directly
                zp_dipole_mag = zp_dipole_sampled

            zp_dipole_radial = jnp.sum(zp_dipole_mag * data["rhat"], axis=1)
            zp_quad_radial = 0.0
            if self._has_varying_zeropoint_quad and zp_quad_sampled is not None:
                if self.zeropoint_parameterization == "fracH0":
                    zp_quad_mag = _frac_to_mag(zp_quad_sampled)
                else:
                    zp_quad_mag = zp_quad_sampled
                zp_quad_radial = compute_quadrupole_radial(data, zp_quad_mag)
            delta_A = zp_dipole_radial + zp_quad_radial

        # Handle H0 quadrupole contribution to zeropoint
        if self._has_varying_H0_quad and H0_quad is not None:
            H0_quad_mag = _frac_to_mag(H0_quad)
            H0_quad_radial = compute_quadrupole_radial(data, H0_quad_mag)
            delta_A = H0_quad_radial if delta_A is None else delta_A + H0_quad_radial

        # Handle per-pixel H0 contribution to zeropoint
        if self._has_varying_H0_pix and H0_pix is not None:
            H0_pix_mag = _frac_to_mag(H0_pix)
            pix_delta = data["C_pix"] @ H0_pix_mag
            delta_A = pix_delta if delta_A is None else delta_A + pix_delta

        # Apply zeropoint offset to scaling relation intercepts
        if delta_A is not None:
            if A_LT is not None:
                A_LT = A_LT + delta_A
            if A_YT is not None:
                A_YT = A_YT + delta_A

        if relation in ["LT", "LTY"]:
            A = A_LT
            B = B_LT
        elif relation in ["YT", "YTL"]:
            A = A_YT
            B = B_YT
        elif relation == "LTYT":
            A = A_LT
            B = B_LT

        if relation == "LTYT":
            rho12 = sample("rho12", Uniform(-0.99, 0.99))  # avoid singular cov

        # For the distance marginalization, h is not sampled.
        h = 1.

        kwargs_dist = sample_distance_prior(self.priors, shared_params)

        def _broadcast_param(param, template):
            param = jnp.asarray(param)
            if param.ndim == 0:
                param = jnp.broadcast_to(param, template.shape)
            return param

        # Sample velocity field parameters.
        Vext = sample_Vext(
            self.priors, self.which_Vext, shared_params, self.kwargs_Vext)
    
        if self.which_Vext == "constant":
            Vext_quad = rsample("Vext_quad", self.priors["Vext_quad"], shared_params)

        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)

        # Remaining parameters
        beta = rsample("beta", self.priors["beta"], shared_params)
        if self.density_prior:
            bias_params = sample_galaxy_bias(
                self.priors, self.galaxy_bias, shared_params, Om=self.Om,
                beta=beta)
        else:
            bias_params = [1.]  # Dummy value, not used

        # MNR hyperpriors for cluster observables
        if self.use_MNR:
            def _shared_or_sample(name, dist):
                if shared_params is not None and name in shared_params:
                    return shared_params[name]
                value = sample(name, dist)
                if shared_params is not None:
                    shared_params[name] = value
                return value

            # All scaling relations need T hyperprior
            logT_prior_mean = _shared_or_sample(
                "logT_prior_mean",
                Uniform(data["min_logT"], data["max_logT"]),
            )
            logT_prior_std = _shared_or_sample(
                "logT_prior_std",
                Uniform(0.0, data["max_logT"] - data["min_logT"]),
            )

            if relation in ["LTY", "YTL"]:
                # These relations need bivariate MNR priors
                if relation == "LTY":
                    # T and Y bivariate prior
                    logY_prior_mean = sample(
                        "logY_prior_mean", Uniform(data["min_logY"], data["max_logY"]))
                    logY_prior_std = sample(
                        "logY_prior_std",
                        Uniform(0.0, data["max_logY"] - data["min_logY"]))
                    rho_TY = sample("rho_TY", Uniform(-1, 1))
                    
                    mu_TY = jnp.array([logT_prior_mean, logY_prior_mean])
                    cov_TY = jnp.array([
                        [logT_prior_std**2, rho_TY * logT_prior_std * logY_prior_std],
                        [rho_TY * logT_prior_std * logY_prior_std, logY_prior_std**2]])
                
                elif relation == "YTL":
                    # T and L bivariate prior  
                    logF_prior_mean = sample(
                        "logF_prior_mean", Uniform(data["min_logF"], data["max_logF"]))
                    logF_prior_std = sample(
                        "logF_prior_std",
                        Uniform(0.0, data["max_logF"] - data["min_logF"]))
                    rho_TF = sample("rho_TF", Uniform(-1, 1))
                    
                    mu_TF = jnp.array([logT_prior_mean, logF_prior_mean])
                    cov_TF = jnp.array([
                        [logT_prior_std**2, rho_TF * logT_prior_std * logF_prior_std],
                        [rho_TF * logT_prior_std * logF_prior_std, logF_prior_std**2]])

        with plate("data", nsamples):
            if self.use_MNR:
                if relation in ["LT", "YT", "LTYT"]:
                    # T-only MNR for these relations
                    logT = sample("logT_latent", Normal(logT_prior_mean, logT_prior_std))
                    sample("logT_obs", Normal(logT, data["e_logT"]), obs=data["logT"])
                    logY = data["logY"]
                    logF = data["logF"]
                
                elif relation == "LTY":
                    # T and Y bivariate MNR
                    x_latent = sample("x_latent_TY", MultivariateNormal(mu_TY, cov_TY))
                    logT = x_latent[:, 0]
                    logY = x_latent[:, 1]
                    
                    sample("logT_obs", Normal(logT, data["e_logT"]), obs=data["logT"])
                    sample("logY_obs", Normal(logY, data["e_logY"]), obs=data["logY"])
                    logF = data["logF"]
                
                elif relation == "YTL":
                    # T and L bivariate MNR
                    x_latent = sample("x_latent_TF", MultivariateNormal(mu_TF, cov_TF))
                    logT = x_latent[:, 0]
                    logF = x_latent[:, 1]
                    
                    sample("logT_obs", Normal(logT, data["e_logT"]), obs=data["logT"])
                    sample("logF_obs", Normal(logF, data["e_logF"]), obs=data["logF"])
                    logY = data["logY"]
            else:
                logT = data["logT"]
                logY = data["logY"]
                logF = data["logF"]

            logY_safe = logY
            e2_logY_safe = data["e2_logY"]
            if relation == "LT":
                logY_safe = jnp.where(jnp.isfinite(logY), logY, 0.0)
                e2_logY_safe = jnp.where(
                    jnp.isfinite(data["e2_logY"]), data["e2_logY"], 0.0)

            # Calculate intrinsic scatter - MNR doesn't propagate observational errors
            if self.use_MNR:
                if relation in ["LT", "LTY"]:
                    sigma_logF = jnp.sqrt(data["e2_logF"] + sigma_LT**2)
                if relation in ["YT", "YTL"]:
                    sigma_logY = jnp.sqrt(data["e2_logY"] + sigma_YT**2)
                if relation == "LTYT":
                    sigma_logF = jnp.sqrt(data["e2_logF"] + sigma_LT**2)
                    sigma_logY = jnp.sqrt(data["e2_logY"] + sigma_YT**2)
            else:
                # Non-MNR case: propagate observational errors
                if relation in ["LT", "LTY"]:
                    sigma_logF = jnp.sqrt(
                        data["e2_logF"] + sigma_LT**2
                        + B**2 * data["e2_logT"] + C**2 * e2_logY_safe)
                if relation in ["YT", "YTL"]:
                    sigma_logY = jnp.sqrt(
                        data["e2_logY"] + sigma_YT**2
                        + B**2 * data["e2_logT"] + C**2 * data["e2_logF"])

            r_grid = data["r_grid"] / h
            logdl_grid = self.distance2logdl(r_grid)
            logda_grid = self.distance2logda(r_grid)

            if data.has_precomputed_los:
                los_delta_r_grid = None
                los_velocity_r_grid = None
                los_log_density_r_grid = None

                if self._needs_zspace and "los_z" not in data.data:
                    raise ValueError(
                        "_needs_zspace=True but LOS file does not contain 'z' array. "
                        "Run scripts/preprocess/los_real2redshift.py to add it.")

                if self._needs_zspace and "los_z" in data.data:
                    # Z-space mode: map from z_grid to r_grid accounting for Vext/H0
                    z_grid_los = data["los_z"]
                    los_r = data["los_r"]

                    # =================================================================
                    # Compute h_per_gal for z→r conversion
                    # H0 anisotropy affects the conversion: r = c*z / H(θ)
                    # =================================================================
                    delta_h = jnp.zeros(data["rhat"].shape[0])  # (n_gal,)

                    # H0 dipole contribution (always fractional)
                    if self._has_varying_H0_dipole and H0_dipole is not None:
                        H0_dip_norm = jnp.linalg.norm(H0_dipole)
                        cos_theta = jnp.sum(
                            H0_dipole * data["rhat"], axis=1
                        ) / jnp.maximum(H0_dip_norm, 1e-30)
                        delta_h = delta_h + H0_dip_norm * cos_theta

                    # H0 quadrupole contribution (already fractional)
                    if self._has_varying_H0_quad and H0_quad is not None:
                        delta_h = delta_h + compute_quadrupole_radial(data, H0_quad)

                    # H0 per-pixel contribution (fractional)
                    if self._has_varying_H0_pix and H0_pix is not None:
                        delta_h = delta_h + data["C_pix"] @ H0_pix

                    # Legacy mode: treat varying zeropoint as H0 anisotropy
                    if self._legacy_stretch_mode and self._has_varying_zeropoint_dipole:
                        if A_dipole is not None:
                            A_norm = jnp.linalg.norm(A_dipole)
                            cos_theta = jnp.sum(
                                A_dipole * data["rhat"], axis=1
                            ) / jnp.maximum(A_norm, 1e-30)
                            legacy_delta = _delta_a_to_frac(A_norm)
                            delta_h = delta_h + legacy_delta * cos_theta

                    # Compute effective h per galaxy
                    h_per_gal = h * (1.0 + delta_h)

                    # Compute r_cosmo for each (field, galaxy, z_grid_point)
                    r_cosmo = compute_los_zspace_to_rspace(
                        data, los_r, z_grid_los, Vext, self.which_Vext,
                        self.kwargs_Vext, self.redshift2distance, h_per_gal,
                        n_iterations=self.n_zspace_iterations
                    )  # (n_field, n_gal, n_los) or (1, n_gal, n_los) for constant Vext

                    # Get LOS values on the original grid
                    los_delta_orig = data.f_los_delta.interp_many_steps_per_galaxy(los_r)
                    los_velocity_orig = data.f_los_velocity.interp_many_steps_per_galaxy(los_r)
                    los_log_density_orig = data.f_los_log_density.interp_many_steps_per_galaxy(los_r)

                    # Interpolate LOS quantities from r_cosmo positions to r_grid.
                    # r_cosmo is always (1, n_gal, n_los) since Vext is sampled,
                    # not field-dependent. Squeeze and share across fields.
                    r_cosmo_2d = r_cosmo[0]  # (n_gal, n_los)

                    def _interp_to_rgrid(los_values, r_cosmo_line):
                        """Interpolate los_values (on r_cosmo positions) onto r_grid."""
                        return jnp.interp(r_grid, r_cosmo_line, los_values)

                    def _interp_field(los_field):
                        """Interpolate one field's LOS values using shared r_cosmo."""
                        return vmap(_interp_to_rgrid, in_axes=(0, 0))(
                            los_field, r_cosmo_2d)

                    los_delta_r_grid = vmap(_interp_field)(los_delta_orig)
                    los_velocity_r_grid = vmap(_interp_field)(los_velocity_orig)
                    los_log_density_r_grid = vmap(_interp_field)(los_log_density_orig)

                else:
                    # No z-space mapping needed - use precomputed or interpolate directly
                    if "los_delta_r_grid" in data.data:
                        los_delta_r_grid = data["los_delta_r_grid"]
                        los_velocity_r_grid = data["los_velocity_r_grid"]
                        los_log_density_r_grid = data["los_log_density_r_grid"]
                    else:
                        los_delta_r_grid = data.f_los_delta.interp_many_steps_per_galaxy(r_grid)
                        los_velocity_r_grid = data.f_los_velocity.interp_many_steps_per_galaxy(r_grid)
                        los_log_density_r_grid = data.f_los_log_density.interp_many_steps_per_galaxy(r_grid)


            # Homogeneous Malmqusit distance prior, `(n_field, n_gal, n_rbin)`
            lp_dist = log_prior_r_empirical(
                r_grid, **kwargs_dist, Rmax_grid=r_grid[-1])[None, None, :]
            #lp_dist = 0.

            # Predict logF/logY incorporating (optional) cosmological E(z)
            if self.apply_Ez_correction:
                logEz_raw = jnp.log10(get_Ez(data["zcmb"], Om=self.Om))
            else:
                logEz_raw = jnp.zeros_like(data["zcmb"])
            logEz = _broadcast_param(logEz_raw, logT)
            logEz_LT = logEz
            logEz_YT = -logEz

            # Predict logF from the scaling relation, `(ngal, nrbin)``
            if relation in ["LT", "LTY"]:
                A_vec = _broadcast_param(A, logT)
                logF_pred = (
                    (logEz_LT + A_vec + B * logT)[:, None]
                    + C * (logY_safe[:, None] + 2 * logda_grid[None, :])
                    - jnp.log10(4 * jnp.pi) - 2 * logdl_grid[None, :]
                )
                # Likelihood of logF , `(n_field, n_gal, n_rbin)`
                ll = Normal(logF_pred, sigma_logF[:, None]).log_prob(
                    data["logF"][:, None])[None, ...]
            elif relation in ["YT", "YTL"]:
                A_vec = _broadcast_param(A, logT)
                logY_pred = (
                    (logEz_YT + A_vec + B * logT)[:, None]
                    + C * (logF[:, None] + 2 * logdl_grid[None, :]
                           + jnp.log10(4 * jnp.pi))
                    - 2 * logda_grid[None, :]
                )
                # Likelihood of logY , `(n_field, n_gal, n_rbin)`
                ll = Normal(logY_pred, sigma_logY[:, None]).log_prob(
                    data["logY"][:, None])[None, ...]
            elif relation == "LTYT":

                # --- Intrinsic means in log-space at fixed T ---
                A_LT_vec = _broadcast_param(A_LT, logT)
                A_YT_vec = _broadcast_param(A_YT, logT)
                mL = (logEz_LT + A_LT_vec + B_LT * logT)[:, None]   # (n_gal,)
                mY = (logEz_YT + A_YT_vec + B_YT * logT)[:, None]   # (n_gal,)

                # --- Map to observable means over the distance grid ---
                # logF = logL - log10(4π) - 2 log DL
                # logy = logY - 2 log DA
                mF = mL - jnp.log10(4 * jnp.pi) - 2.0 * logdl_grid[None, :]   # (n_gal, n_rbin)
                my = mY - 2.0 * logda_grid[None, :]                           # (n_gal, n_rbin)

                # --- Intrinsic covariance (logL, logY) at fixed T ---
                # sigma_LT: scatter in logL; sigma_YT: scatter in logY; rho12: intrinsic corr.

                # Total covariance in observable space = intrinsic + measurement
                # (no measurement cross-covariance)

                v11 = sigma_LT**2  + data["e2_logF"]          # (n_gal,)
                v22 = sigma_YT**2 + data["e2_logY"]          # (n_gal,)
                v12 = jnp.ones_like(v11) * rho12 * sigma_LT * sigma_YT           # scalar → broadcasts
                if self.use_MNR == False:
                    # Add measurement error propagation from T
                    v11 += B_LT**2 * data["e2_logT"]
                    v22 += B_YT**2 * data["e2_logT"]
                    v12 += B_LT * B_YT * data["e2_logT"]

                # Broadcast across the distance grid
                V11 = v11[:, None]   # (n_gal, 1)
                V22 = v22[:, None]   # (n_gal, 1)
                V12 = v12[:, None]   # (n_gal, 1)

                # Observations broadcast to (n_gal, n_rbin)
                xF = data["logF"][:, None]
                xy = data["logY"][:, None]

                # Joint log-likelihood per galaxy × per grid bin
                ll_joint = logpdf_mvn2_broadcast(xF, xy, mF, my, V11, V22, V12)  # (n_gal, n_rbin)
                ll = ll_joint[None, ...]  # (n_field=1, n_gal, n_rbin)
            else:
                raise ValueError(f"Invalid which_relation '{relation}'.")

            if data.has_precomputed_los:
                # Reconstruction LOS velocity `(n_field, n_gal, n_step)`
                Vrad = beta * los_velocity_r_grid
                # Add inhomogeneous Malmquist bias and normalize the r prior
                if self.density_prior:
                    lp_dist += lp_galaxy_bias(
                        los_delta_r_grid,
                        los_log_density_r_grid,
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
                data, r_grid, Vext, which_Vext=self.which_Vext,
                **self.kwargs_Vext)
                    
            if self.which_Vext == "constant":
                Vext_quad_rad = compute_quadrupole_radial(data, Vext_quad)
                Vext_rad += Vext_quad_rad[None, :, None]
            czpred = predict_cz(
                self.distance2redshift(r_grid, h=h)[None, None, :],
                Vrad + Vext_rad)
            ll += Normal(czpred, sigma_v).log_prob(
                data["czcmb"][None, :, None])

            if self.save_distances or self.track_log_density_per_sample:
                # Diagnostics: MAP and mean/std of r per field/galaxy
                logw = ll - jnp.max(ll, axis=-1, keepdims=True)
                w = jnp.exp(logw)
                w /= jnp.sum(w, axis=-1, keepdims=True)
                r_mean = jnp.sum(w * r_grid[None, None, :], axis=-1)
                r_var = jnp.sum(
                    w * (r_grid[None, None, :] - r_mean[..., None])**2, axis=-1)
                r_std = jnp.sqrt(r_var)
                r_map = r_grid[jnp.argmax(ll, axis=-1)]
                deterministic("r_map_skipZ", r_map)
                deterministic("r_mean_skipZ", r_mean)
                deterministic("r_std_skipZ", r_std)

            if self.track_log_density_per_sample:
                # Save distance prior (shape: n_field, n_gal, n_rbin)
                deterministic("lp_dist_skipZ", lp_dist)

                # Save redshift/cz likelihood components
                ll_cz = Normal(czpred, sigma_v).log_prob(
                    data["czcmb"][None, :, None])
                cz_nsigma = (data["czcmb"][None, :, None] - czpred) / sigma_v
                deterministic("ll_cz_skipZ", ll_cz)
                deterministic("cz_nsigma_skipZ", cz_nsigma)

                # Save scaling relation likelihood components (LT or YT only)
                if relation == "LT":
                    ll_cluster = Normal(logF_pred, sigma_logF[:, None]).log_prob(
                        data["logF"][:, None])
                    cluster_nsigma = (data["logF"][:, None] - logF_pred) / sigma_logF[:, None]
                    deterministic("ll_cluster_skipZ", ll_cluster[None, ...])
                    deterministic("cluster_nsigma_skipZ", cluster_nsigma[None, ...])
                elif relation == "YT":
                    ll_cluster = Normal(logY_pred, sigma_logY[:, None]).log_prob(
                        data["logY"][:, None])
                    cluster_nsigma = (data["logY"][:, None] - logY_pred) / sigma_logY[:, None]
                    deterministic("ll_cluster_skipZ", ll_cluster[None, ...])
                    deterministic("cluster_nsigma_skipZ", cluster_nsigma[None, ...])
                # LTYT skipped (bivariate gaussian)

            # Marginalise over the radial distance, average over realisations
            # and track the log-density.
            #deterministic("ll_skipZ", ll)
            ll = ln_simpson(ll, x=r_grid[None, None, :], axis=-1)
            ll = logsumexp(ll, axis=0) - jnp.log(data.num_fields)
            factor("ll_obs", ll)

            if self.track_log_density_per_sample:
                log_density_per_sample += ll
                deterministic("log_density_per_sample", log_density_per_sample)



class HybridClustersModel(BaseModel):
    """
    Cluster LTY scaling relation with explicit distance marginalization.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        self.which_relation = self.config["io"]["Clusters"]["which_relation"]
        if self.which_relation not in ["LT", "LY", "LTY"]:
            raise ValueError(
                f"Invalid scaling relation '{self.which_relation}'. "
                "Choose either 'LT' or 'LY' or 'LTY'.")

        self.redshift2distance = Redshift2Distance(Om0=self.Om)

        if self.which_relation == "LT":
            self.priors["CL_C"] = Delta(jnp.asarray(0.0))
        if self.which_relation == "LY":
            self.priors["CL_B"] = Delta(jnp.asarray(0.0))

        if self.use_MNR:
            raise NotImplementedError(
                "MNR for clusters is not implemented yet. Please set "
                "`use_MNR` to False in the config file.")
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

    def __call__(self, data, shared_params=None):
        nsamples = len(data)
        if self.track_log_density_per_sample:
            log_density_per_sample = jnp.zeros(nsamples)

        if data.sample_dust:
            raise NotImplementedError(
                "Dust sampling is not implemented for "
                "`ClustersModel`.")

        # Sample the cluster scaling parameters.
        A = rsample("A_CL", self.priors["CL_A"], shared_params)
        B = rsample("B_CL", self.priors["CL_B"], shared_params)
        sigma_int = rsample(
            "sigma_int", self.priors["sigma_int"], shared_params)

        # For the distance marginalization, h is not sampled.
        h = 1.

        # Sample velocity field parameters.
        if self.with_radial_Vext:
            Vext = rsample(
                "Vext_rad", self.priors["Vext_radial"], shared_params)
        else:
            Vext = rsample("Vext", self.priors["Vext"], shared_params)

        with plate("data", nsamples):
            if self.use_MNR:
                raise NotImplementedError(
                    "MNR for clusters is not implemented yet.")
            else:
                logT = data["logT"]
                sigma_logF = jnp.sqrt(
                    data["e2_logF"] + sigma_int**2
                    + B**2 * data["e2_logT"])

            Vext_rad = compute_Vext_radial(
            data, None, Vext, with_radial_Vext=self.with_radial_Vext,
            **self.kwargs_radial_Vext)[0,:,0]
            
            zcosmo = (1 + data["zcmb"]) / (1 + Vext_rad / SPEED_OF_LIGHT) - 1
            logdl = jnp.log10(self.redshift2distance(zcosmo, h=h) * (1 + zcosmo))

            # jprint("zcosmo{}", zcosmo.shape)
            # jprint("Vext_rad{}", Vext_rad.shape)


            logF_pred = (A + B * logT[:, None]
                         - jnp.log10(4 * jnp.pi) - 2 * logdl)

            # Likelihood of logF , `(n_field, n_gal, n_rbin)`
            ll = Normal(logF_pred, sigma_logF[:, None]).log_prob(
                data["logF"][:, None])[None, ...]

            factor("ll_obs", ll)

            if self.track_log_density_per_sample:
                log_density_per_sample += ll
                deterministic("log_density_per_sample", log_density_per_sample)


class MigkasModel(BaseModel):
    """
    The simple model of Migkas et al. from the FLAMINGO paper.
    """
    def __init__(self, config_path):
        super().__init__(config_path)

        self.redshift2distance = Redshift2Distance(Om0=self.Om)

    def __call__(self, data):
        
        nsamples = len(data['logT'])

        A = sample("A_CL", Uniform(-5, 5))
        B = sample("B_CL", Uniform(-5, 5))
        sigma = sample("sigma_int", Uniform(0, 1))

        Vext = rsample("Vext", self.priors["Vext"], shared_params=None)
        Vext_rad = compute_Vext_radial(data, None, Vext, with_radial_Vext=False)
        Vext_rad = Vext_rad[0,:,0]

        dH = rsample("dH0", self.priors["dH0"], shared_params=None)
        dH_rad = jnp.sum(data['rhat'] * dH[None, :], axis=1)
        h = 0.7 * (1 + dH_rad)
        
        #vpec  = jnp.sum(dipole_vector * data['rhat'],axis=1)   #
        zcosmo = (1 + data["zcmb"]) / (1 + Vext_rad / SPEED_OF_LIGHT) - 1

        dL = self.redshift2distance(zcosmo, h=h) * (1 + zcosmo)
        logL = data['logF'] + jnp.log10(4.0*jnp.pi) + 2.0*jnp.log10(dL)

        with plate("data", nsamples):
            chi2 = (logL - (A + B * data['logT']))**2 / (sigma**2 + data['e_logLx']**2 + (B * data['e_logT']**2))
            lnL = -0.5 * chi2  - 0.5 * jnp.log(sigma**2 + data['e_logLx']**2 + B**2 * data['e_logT']**2)

            factor("likelihood", lnL)


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
            fprint("setting `compute_evidence` to False.")
            self.config["inference"]["compute_evidence"] = False

        if self.track_log_density_per_sample:
            raise NotImplementedError(
                "`track_log_density_per_sample` is not implemented "
                "for `FPModel`.")

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

        kwargs_dist = sample_distance_prior(self.priors, shared_params)

        # Sample velocity field parameters.
        Vext = sample_Vext(
            self.priors, self.which_Vext, shared_params, self.kwargs_Vext)
        sigma_v = rsample("sigma_v", self.priors["sigma_v"], shared_params)

        # Remaining parameters
        beta = rsample("beta", self.priors["beta"], shared_params)
        bias_params = sample_galaxy_bias(
            self.priors, self.galaxy_bias, shared_params, Om=self.Om,
            beta=beta)

        if self.use_MNR:
            logs_prior_mean = sample(
                "logs_prior_mean", Uniform(data["min_logs"], data["max_logs"]))
            logs_prior_std = sample(
                "logs_prior_std",
                Uniform(0, data["max_logs"] - data["min_logs"]))

            logI_prior_mean = sample(
                "logI_prior_mean", Uniform(data["min_logI"], data["max_logI"]))
            logI_prior_std = sample(
                "logI_prior_std",
                Uniform(0.0, data["max_logI"] - data["min_logI"]))
            rho = sample("rho_corr", Uniform(-1.0, 1.0))

            mu = jnp.array([logs_prior_mean, logI_prior_mean])
            cov = jnp.array([
                [logs_prior_std**2, rho * logs_prior_std * logI_prior_std],
                [rho * logs_prior_std * logI_prior_std, logI_prior_std**2]])

        with plate("data", nsamples):
            if self.use_MNR:
                x_latent = sample("x_latent", MultivariateNormal(mu, cov))
                logs = x_latent[:, 0]
                logI = x_latent[:, 1]

                sample("logs_obs", Normal(logs, data["e_logs"]), obs=logs)
                sample("logI_obs", Normal(logI, data["e_logI"]), obs=logI)

                sigma_log_theta = jnp.sqrt(
                    sigma_log_theta**2 + data["e2_log_theta_eff"])
            else:
                logs = data["logs"]
                logI = data["logI"]

                sigma_log_theta = jnp.sqrt(
                    + sigma_log_theta**2
                    + data["e2_log_theta_eff"]
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
                data, r_grid, Vext, which_Vext=self.which_Vext,
                **self.kwargs_Vext)
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

        kwargs_dist = sample_distance_prior(self.priors, shared_params)

        # Sample velocity field parameters.
        Vext = sample_Vext(
            self.priors, self.which_Vext, shared_params, self.kwargs_Vext)
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
                data, r_grid, Vext, which_Vext=self.which_Vext,
                **self.kwargs_Vext)
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
        self.which_Vext = submodels[0].which_Vext
        self.kwargs_Vext = getattr(submodels[0], "kwargs_Vext", {})

    def _sample_shared_params(self, priors):
        shared = {}
        for name in self.shared_param_names:
            if name in priors:
                shared[name] = _rsample(name, priors[name])
            # Skip params not in priors - they'll be sampled on first use
            # via rsample and added to shared dict
        return shared

    def __call__(self, data):
        assert len(data) == len(self.submodels)
        shared_params = self._sample_shared_params(self.submodels[0].priors)

        # Sample Vext_pix_u BEFORE entering any scope to avoid prefix in name
        if self.which_Vext == "per_pix" and "Vext_pix_u" in self.shared_param_names:
            npix = self.kwargs_Vext["npix"]
            with plate("Vext_pix_plate", npix - 1):
                u = sample("Vext_pix_u", Normal(0., 1.))
            shared_params["Vext_pix_u"] = u

        for i, (submodel, data_i) in enumerate(zip(self.submodels, data)):
            name = data_i.name if data_i is not None else f"dataset_{i}"
            with handlers.scope(prefix=name):
                submodel(data_i, shared_params=shared_params)
