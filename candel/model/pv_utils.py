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
PV-specific sampling utilities: vector sampling, galaxy bias, external
velocity, distance priors, and TFR/SN helpers.
"""
import jax.numpy as jnp
from interpax import interp1d
from jax import vmap
from jax.lax import cond
from jax.scipy.stats.norm import cdf as jax_norm_cdf
from numpyro import deterministic, plate, sample
from numpyro.distributions import Delta, Normal, Uniform

from .utils import smoothclip_nr

###############################################################################
#                        Vector sampling utilities                            #
###############################################################################


def sample_vector_components_uniform(name, low, high):
    """
    Sample a 3D vector by drawing each Cartesian component independently
    from a uniform distribution over [xmin, xmax].
    """
    x = sample(f"{name}_x", Uniform(low, high))
    y = sample(f"{name}_y", Uniform(low, high))
    z = sample(f"{name}_z", Uniform(low, high))
    return jnp.array([x, y, z])


def sample_vector_fixed(name, mag_min, mag_max):
    """
    Sample a 3D vector with direction ~ isotropic, magnitude ~ Uniform.
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


def sample_radialmag_vector(name, nval, low, high):
    """
    Sample a vector whose magnitude varies at `nval` knots but a direction
    is shared by all knots and sampled isotropically on the sky. The magnitude
    is sampled ~ Uniform(low, high).

    Returns the tuple (mag, rhat), where `mag` has shape (nval,) and `rhat`
    has shape (3,).
    """
    phi = sample(f"{name}_phi", Uniform(0, 2 * jnp.pi))
    cos_theta = sample(f"{name}_cos_theta", Uniform(-1, 1))
    sin_theta = jnp.sqrt(1 - cos_theta**2)

    with plate(f"{name}_plate", nval):
        mag = sample(f"{name}_mag", Uniform(low, high))

    rhat = jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta
        ])

    return mag, rhat


def sample_radial_vector(name, nval, low, high):
    """
    Sample a radial vector at `nval` knots: direction ~ isotropic,
    magnitude ~ Uniform(low, high). Returns an array of shape (nval, 3).
    """
    with plate(f"{name}_plate", nval):
        phi = sample(f"{name}_phi", Uniform(0.0, 2.0 * jnp.pi))
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


###############################################################################
#                        SLERP / vector interpolation                         #
###############################################################################


def _slerp(u0, u1, t, eps=1e-8):
    dot = jnp.clip(jnp.dot(u0, u1), -1.0, 1.0)
    theta = jnp.arccos(dot)
    sin_th = jnp.sin(theta)

    def slerp_core(_):
        a = jnp.sin((1.0 - t) * theta) / sin_th
        b = jnp.sin(t * theta) / sin_th
        return a * u0 + b * u1

    def lerp_norm(_):
        v = (1.0 - t) * u0 + t * u1
        n = jnp.linalg.norm(v)
        return jnp.where(n > 0.0, v / n, u0)

    return cond(sin_th < eps, lerp_norm, slerp_core, operand=None)


def interp_cartesian_vector(rq, rknot, v_knot, method="cubic"):
    """Magnitude via interpax; direction via SLERP; constant extrapolation."""
    rq = jnp.asarray(rq)
    x = jnp.asarray(rknot)
    y = jnp.asarray(v_knot)            # (K, 3)
    K = y.shape[0]

    mk = jnp.linalg.norm(y, axis=-1)   # (K,)
    mk_safe = jnp.where(mk > 0.0, mk, 1.0)
    uk = y / mk_safe[:, None]

    m_r = interp1d(rq, x, mk, method=method)
    x0, x1 = x[0], x[-1]
    m_r = jnp.where(rq < x0, mk[0], m_r)
    m_r = jnp.where(rq > x1, mk[-1], m_r)

    def dir_at_r(r):
        i = jnp.clip(jnp.searchsorted(x, r, side="right") - 1, 0, K - 2)
        xl, xr = x[i], x[i + 1]
        t = jnp.where(xr > xl, (r - xl) / (xr - xl), 0.0)
        return _slerp(uk[i], uk[i + 1], t)

    u_r = vmap(dir_at_r)(rq)           # (R, 3)
    u_r = jnp.where((rq < x0)[:, None], uk[0], u_r)
    u_r = jnp.where((rq > x1)[:, None], uk[-1], u_r)

    return m_r[:, None] * u_r          # (R, 3)


###############################################################################
#                           Sampling utilities                                #
###############################################################################


def _rsample(name, dist):
    """
    Samples from `dist` unless it is a delta function or vector directive.
    """
    if isinstance(dist, Delta) and name == "zeropoint_dipole":
        return jnp.zeros(3)

    if isinstance(dist, Delta):
        return deterministic(name, dist.v)

    if isinstance(dist, dict) and dist.get("type") == "vector_uniform_fixed":
        return sample_vector_fixed(name, dist["low"], dist["high"])

    if isinstance(dist, dict) and dist.get("type") == "vector_components_uniform":  # noqa
        return sample_vector_components_uniform(
            name, dist["low"], dist["high"])

    if isinstance(dist, dict) and dist.get("type") == "vector_radial_uniform":
        return sample_radial_vector(
            name, dist["nval"], dist["low"], dist["high"], )

    if isinstance(dist, dict) and dist.get("type") == "vector_radialmag_uniform":  # noqa
        return sample_radialmag_vector(
            name, dist["nval"], dist["low"], dist["high"], )

    return sample(name, dist)


def rsample(name, dist, shared_params=None):
    """
    Retrieve a parameter value: use `shared_params[name]` when provided (for
    sharing/conditioning across submodels), otherwise draw from `dist`.
    """
    if shared_params is not None and name in shared_params:
        return shared_params[name]
    return _rsample(name, dist)


###############################################################################
#                           TFR helpers                                       #
###############################################################################


def marginalise_2d_latent(prior_std1, prior_std2, rho,
                          prior_mean1, prior_mean2,
                          e2_obs1, e2_obs2, obs1, obs2, f1, f2):
    """Analytically marginalise 2D Gaussian latent variables.

    Given a bivariate Gaussian hyperprior on latent x = (x1, x2) with mean
    (prior_mean1, prior_mean2) and covariance parameterised by (prior_std1,
    prior_std2, rho), independent Gaussian observations obs1, obs2 with
    variances e2_obs1, e2_obs2, and a linear projection f = (f1, f2),
    returns:

        f_dot_mu1 : fᵀ μ₁  (posterior mean projected onto f)
        fSf       : fᵀ Σ₁ f (posterior variance projected onto f)
        log_ev_obs: ln N(d_obs | μ_h, Σ_h + Σ_o)
    """
    # Hyperprior variances and covariance
    s_12 = prior_std1**2
    s_22 = prior_std2**2
    s_xc = rho * prior_std1 * prior_std2
    det_h = s_12 * s_22 - s_xc**2

    # Λ_h = Σ_h⁻¹
    Lh00 = s_22 / det_h
    Lh01 = -s_xc / det_h
    Lh11 = s_12 / det_h

    # Λ_h @ μ_h
    Lh_mu0 = Lh00 * prior_mean1 + Lh01 * prior_mean2
    Lh_mu1 = Lh01 * prior_mean1 + Lh11 * prior_mean2

    # Observation precisions
    tau1 = 1.0 / e2_obs1
    tau2 = 1.0 / e2_obs2

    # Posterior precision Λ₁ = Λ_h + Λ_o
    L11 = Lh00 + tau1
    L12 = Lh01
    L22 = Lh11 + tau2
    det_1 = L11 * L22 - L12**2

    # Posterior covariance Σ₁ = Λ₁⁻¹
    S11 = L22 / det_1
    S12 = -L12 / det_1
    S22 = L11 / det_1

    # Natural parameter η₁ = Λ_h μ_h + Λ_o d_obs
    eta1 = Lh_mu0 + tau1 * obs1
    eta2 = Lh_mu1 + tau2 * obs2

    # Posterior mean μ₁ = Σ₁ η₁
    mu1_1 = S11 * eta1 + S12 * eta2
    mu1_2 = S12 * eta1 + S22 * eta2

    # fᵀ μ₁ and fᵀ Σ₁ f
    f_dot_mu1 = f1 * mu1_1 + f2 * mu1_2
    fSf = f1**2 * S11 + 2 * f1 * f2 * S12 + f2**2 * S22

    # Evidence for obs: ln N(d_obs | μ_h, Σ_h + Σ_o)
    P11 = s_12 + e2_obs1
    P12 = s_xc
    P22 = s_22 + e2_obs2
    det_P = P11 * P22 - P12**2
    r1 = obs1 - prior_mean1
    r2 = obs2 - prior_mean2
    mahal = (P22 * r1**2 - 2 * P12 * r1 * r2 + P11 * r2**2) / det_P
    log_ev_obs = -0.5 * mahal - 0.5 * jnp.log(det_P) - jnp.log(2 * jnp.pi)

    return f_dot_mu1, fSf, log_ev_obs


def get_absmag_TFR(eta, a_TFR, b_TFR, c_TFR=0.0):
    """
    Tully–Fisher absolute magnitude with optional curvature:

        M(eta) = a + b * eta + c * eta^2 for eta > 0,
                 a + b * eta           otherwise.

    The quadratic term only applies on the high-width (eta > 0) side to
    capture potential curvature while keeping the low-width branch linear.
    """
    return a_TFR + b_TFR * eta + jnp.where(eta > 0, c_TFR * eta**2, 0.0)


def gauss_hermite_log_weights(n):
    """Gauss-Hermite nodes and log-weights for Gaussian-weighted integration.

    Returns (nodes, log_weights) such that::

        int f(x) N(x;mu,s) dx ~ sum_i exp(lw_i) * f(mu + sqrt(2)*s*x_i)
    """
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)
    log_w = jnp.log(jnp.asarray(w, dtype=jnp.float32)) - 0.5 * jnp.log(jnp.pi)
    return jnp.asarray(x, dtype=jnp.float32), log_w


###############################################################################
#                      Galaxy bias & velocity                                #
###############################################################################


def sigma_v_from_density(delta, sigma_v_low, sigma_v_high, log_rho_t, k):
    """Map overdensity to sigma_v through a sigmoid in log density."""
    rho = jnp.clip(1.0 + delta, a_min=1e-6)
    log_rho = jnp.log(rho)
    return sigma_v_low + (sigma_v_high - sigma_v_low) / (
        1.0 + jnp.exp(-k * (log_rho - log_rho_t)))


def sample_galaxy_bias(priors, galaxy_bias, shared_params=None, **kwargs):
    """
    Sample a vector of galaxy bias parameters based on the specified model.
    """
    if galaxy_bias == "unity":
        return [1.,]
    elif galaxy_bias == "powerlaw":
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
    elif galaxy_bias == "quadratic":
        b1 = rsample("b1", priors["b1"], shared_params)
        b2 = rsample("b2", priors["b2"], shared_params)
        bias_params = [b1, b2]
    elif galaxy_bias == "spline":
        knots_delta = kwargs["spline_bias_knots_delta"]
        import numpy as _np
        knots_log1pd = jnp.log(1 + jnp.array(knots_delta))
        pin_idx = int(_np.argmin(_np.abs(_np.array(knots_delta))))
        n_knots = len(knots_delta)
        # Sample N-1 free amplitudes, insert 0 at pinned knot
        amps = []
        for i in range(n_knots):
            if i == pin_idx:
                continue
            y_i = rsample(f"spline_bias_y_{i}", priors["spline_bias_y"],
                          shared_params)
            amps.append((i, y_i))
        all_amps = jnp.zeros(n_knots)
        for i, y_i in amps:
            all_amps = all_amps.at[i].set(y_i)
        bias_params = [knots_log1pd, all_amps]
    else:
        raise ValueError(f"Invalid galaxy bias model '{galaxy_bias}'.")

    return bias_params


def lp_galaxy_bias(delta, log_rho, bias_params, galaxy_bias,
                   quadratic_bias_delta0=0.0):
    """
    Log galaxy bias probability, given some density and a bias model.
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
    elif galaxy_bias == "quadratic":
        b1, b2 = bias_params
        d = delta - quadratic_bias_delta0
        lp = jnp.log(smoothclip_nr(1 + b1 * d + b2 * d**2, tau=0.1))
    elif galaxy_bias == "spline":
        knots_log1pd, amplitudes = bias_params
        shape = log_rho.shape
        x = jnp.clip(log_rho.ravel(), knots_log1pd[0], knots_log1pd[-1])
        lp = interp1d(x, knots_log1pd, amplitudes, method="cubic")
        lp = lp.reshape(shape)
    else:
        raise ValueError(f"Invalid galaxy bias model '{galaxy_bias}'.")

    return lp


def compute_Vext_radial(data, r_grid, Vext, which_Vext, **kwargs_Vext):
    """
    Compute the line-of-sight projection of the external velocity.

    Promote the final output to shape `(n_field, n_gal, n_rbins)`.
    """
    if which_Vext == "radial":
        # Shape (3, n_rbins)
        Vext = interp_cartesian_vector(r_grid, v_knot=Vext, **kwargs_Vext)

        Vext_rad = jnp.sum(
            data["rhat"][:, None, :] * Vext[None, :, :], axis=-1)[None, ...]
    elif which_Vext == "radial_magnitude":
        # Unpack the tuple of magnitude and direction.
        Vext_mag, rhat = Vext
        # Interpolate the magnitude as a function of radius, shape (n_rbins,).
        Vext_mag_r = interp1d(
            r_grid, kwargs_Vext["rknot"], Vext_mag,
            method=kwargs_Vext["method"], extrap=(Vext_mag[0], Vext_mag[-1]))
        # Clipping to only keep positive magnitudes.
        Vext_mag_r = smoothclip_nr(Vext_mag_r, tau=2.5)

        # Project the LOS of each galaxy onto the dipole direction, shape
        # is (n_gal,).
        cos_theta = jnp.sum(data["rhat"] * rhat[None, :], axis=1)

        # Finally, the shape is (n_field, n_gal, n_rbins).
        Vext_rad = (cos_theta[:, None] * Vext_mag_r[None, :])[None, :, :]
    elif which_Vext == "per_pix":
        Vext_rad = (data["C_pix"] @ Vext)[None, :, None]
    elif which_Vext == "constant":
        Vext_rad = jnp.sum(data["rhat"] * Vext[None, :], axis=1)[None, :, None]
    else:
        raise ValueError(f"Invalid which_Vext '{which_Vext}'.")
    return Vext_rad


def sample_distance_prior(priors):
    """Sample hyperparameters describing the empirical distance prior."""
    return {
        "R": rsample("R_dist_emp", priors["R_dist_emp"]),
        "p": rsample("p_dist_emp", priors["p_dist_emp"]),
        "n": rsample("n_dist_emp", priors["n_dist_emp"])
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
        Vext = rsample("Vext_rad", priors["Vext_radial"], shared_params)
    elif which_Vext == "radial_magnitude":
        Vext = rsample(
            "Vext_radmag", priors["Vext_radial_magnitude"],
            shared_params)
    elif which_Vext == "per_pix":
        Vext_sigma = rsample("Vext_sigma", priors["Vext_sigma"], shared_params)

        with plate("Vext_pix_plate", kwargs_Vext["npix"] - 1):
            u = rsample("Vext_pix_skipZ", Normal(0., 1.), shared_params)

        Vext = rsample(
            "Vext_pix",
            Delta(Vext_sigma * (kwargs_Vext["Q"] @ u)), shared_params)
    elif which_Vext == "constant":
        Vext = rsample("Vext", priors["Vext"], shared_params)
    else:
        raise ValueError(f"Invalid which_Vext '{which_Vext}'.")

    return Vext


###############################################################################
#                              TFR/SN helpers                                 #
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
