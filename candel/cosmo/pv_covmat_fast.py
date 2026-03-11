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
Fast PV covariance via velocity correlation tensor decomposition.

Instead of the O(N² × n_k × ℓ_max) Bessel-sum expansion, this uses the
Gorski (1988) velocity correlation tensor:

    Ψ_{ab}(s) = ψ_⊥(s)(δ_{ab} - ŝ_a ŝ_b) + ψ_∥(s) ŝ_a ŝ_b

where ψ_∥ and ψ_⊥ are 1D functions of scalar separation that can be
precomputed once and interpolated.

References
----------
- Gorski (1988), ApJL 332, L7
- Gorski, Davis, Strauss, White & Yahil (1989), ApJ 344, 1
- Strauss & Willick (1995), Phys. Rep. 261, 271
- Koda, Blake, Davis+ (2014), MNRAS 445, 4267
"""
import numpy as np
from scipy.integrate import simpson

###############################################################################
# NumPy precomputation (runs once before MCMC)
###############################################################################


def _j1_over_x(x):
    """Compute j_1(x)/x with Taylor expansion for small x.

    j_1(x)/x = sin(x)/x^3 - cos(x)/x^2

    For small x, use Taylor: j_1(x)/x ≈ 1/3 - x^2/30 + x^4/840.
    """
    out = np.empty_like(x)
    small = np.abs(x) < 0.1
    xlarge = x[~small]

    out[small] = (1.0 / 3 - x[small]**2 / 30 + x[small]**4 / 840)
    out[~small] = np.sin(xlarge) / xlarge**3 - np.cos(xlarge) / xlarge**2

    return out


def _j0(x):
    """Compute j_0(x) = sin(x)/x, with j_0(0) = 1."""
    return np.sinc(x / np.pi)


def precompute_psi_functions(k, Pk, dDdtau, s_max=600.0, n_s=20000):
    """Precompute ψ_∥(s) and ψ_⊥(s) on a 1D grid.

    Parameters
    ----------
    k : 1D array
        Wavenumbers in h/Mpc.
    Pk : 1D array
        Matter power spectrum P(k) at z=0 in (Mpc/h)^3.
    dDdtau : float
        dD/dτ (derivative of growth factor w.r.t. conformal time).
    s_max : float
        Maximum separation in Mpc/h.
    n_s : int
        Number of grid points.

    Returns
    -------
    s_grid : 1D array of shape (n_s,)
    psi_par : 1D array of shape (n_s,)
        ψ_∥(s) values.
    psi_perp : 1D array of shape (n_s,)
        ψ_⊥(s) values.
    sigma_v_sq_over3 : float
        σ²_v / 3 = ψ_∥(0) = ψ_⊥(0), the diagonal value.
    """
    s_grid = np.linspace(0, s_max, n_s)
    prefactor = dDdtau**2 / (2 * np.pi**2)

    # (n_s, n_k)
    ks = k[None, :] * s_grid[:, None]

    j0_ks = _j0(ks)
    j1_over_ks = _j1_over_x(ks)

    # ψ_∥(s) = prefactor * ∫ dk P(k) [j₀(ks) - 2 j₁(ks)/(ks)]
    integrand_par = Pk[None, :] * (j0_ks - 2 * j1_over_ks)
    psi_par = prefactor * simpson(integrand_par, x=k, axis=1)

    # ψ_⊥(s) = prefactor * ∫ dk P(k) j₁(ks)/(ks)
    integrand_perp = Pk[None, :] * j1_over_ks
    psi_perp = prefactor * simpson(integrand_perp, x=k, axis=1)

    sigma_v_sq_over3 = psi_par[0]

    return s_grid, psi_par, psi_perp, sigma_v_sq_over3


###############################################################################
# NumPy covariance assembly (for verification / CPU use)
###############################################################################


def assemble_pv_covariance_numpy(r, rhat, s_grid, psi_par, psi_perp,
                                 sigma_v_sq_over3):
    """Assemble the PV covariance matrix using NumPy.

    Parameters
    ----------
    r : 1D array (N,)
        Comoving distances in Mpc/h.
    rhat : 2D array (N, 3)
        Unit LOS vectors.
    s_grid, psi_par, psi_perp : 1D arrays
        Precomputed ψ functions from `precompute_psi_functions`.
    sigma_v_sq_over3 : float
        Diagonal value σ²_v/3.

    Returns
    -------
    C : 2D array (N, N)
    """
    from scipy.interpolate import interp1d as scipy_interp1d

    x = r[:, None] * rhat  # (N, 3)

    # Separation vectors and magnitude
    dx = x[None, :, :] - x[:, None, :]  # (N, N, 3)
    s = np.linalg.norm(dx, axis=-1)  # (N, N)

    s_safe = np.maximum(s, 1e-10)
    shat = dx / s_safe[:, :, None]  # (N, N, 3)

    # Dot products
    ri_dot_rj = np.einsum('ia,ja->ij', rhat, rhat)
    ri_dot_s = np.einsum('ia,ija->ij', rhat, shat)
    rj_dot_s = np.einsum('ja,ija->ij', rhat, shat)

    # Interpolate ψ functions
    f_par = scipy_interp1d(s_grid, psi_par, kind='cubic',
                           bounds_error=False, fill_value=0.0)
    f_perp = scipy_interp1d(s_grid, psi_perp, kind='cubic',
                            bounds_error=False, fill_value=0.0)
    psi_par_ij = f_par(s)
    psi_perp_ij = f_perp(s)

    # Assemble
    C = (psi_perp_ij * (ri_dot_rj - ri_dot_s * rj_dot_s)
         + psi_par_ij * ri_dot_s * rj_dot_s)

    # Fix diagonal
    np.fill_diagonal(C, sigma_v_sq_over3)

    return C


###############################################################################
# JAX covariance assembly (for use inside MCMC / JIT)
###############################################################################


def assemble_pv_covariance_jax(r, rhat, s_grid, psi_par, psi_perp,
                               sigma_v_sq_over3):
    """Assemble PV covariance matrix in JAX (JIT-compilable, differentiable).

    Uses the law of cosines to avoid materializing the (N,N,3) separation
    vector array, and fast linear interpolation on the uniform ψ grid.

    Parameters
    ----------
    r : JAX array (N,)
        Comoving distances in Mpc/h.
    rhat : JAX array (N, 3)
        Unit LOS vectors.
    s_grid : JAX array (n_s,)
        Precomputed separation grid (must be uniform spacing).
    psi_par : JAX array (n_s,)
        ψ_∥ values on the grid.
    psi_perp : JAX array (n_s,)
        ψ_⊥ values on the grid.
    sigma_v_sq_over3 : float or JAX scalar
        Diagonal value σ²_v/3.

    Returns
    -------
    C : JAX array (N, N)
    """
    import jax.numpy as jnp

    N = r.shape[0]
    ri_dot_rj = rhat @ rhat.T  # (N, N)

    ri = r[:, None]  # (N, 1)
    rj = r[None, :]  # (1, N)

    # Law of cosines: s² = r_i² + r_j² - 2 r_i r_j cos(θ_ij)
    s_sq = ri**2 + rj**2 - 2 * ri * rj * ri_dot_rj
    s = jnp.sqrt(jnp.maximum(s_sq, 1e-30))

    # Dot products without materializing (N,N,3) shat array:
    #   ri_dot_s = rhat_i . (x_j - x_i) / s = (r_j cos(θ) - r_i) / s
    #   rj_dot_s = rhat_j . (x_j - x_i) / s = (r_j - r_i cos(θ)) / s
    ri_dot_s = (rj * ri_dot_rj - ri) / s
    rj_dot_s = (rj - ri * ri_dot_rj) / s

    # Fast linear interpolation on the uniform ψ grid.
    ds = s_grid[1] - s_grid[0]
    s_clipped = jnp.clip(s, 0, s_grid[-1] - 1e-10)
    idx_f = s_clipped / ds
    idx = jnp.floor(idx_f).astype(jnp.int32)
    frac = idx_f - idx

    psi_par_ij = psi_par[idx] + frac * (psi_par[idx + 1] - psi_par[idx])
    psi_perp_ij = psi_perp[idx] + frac * (psi_perp[idx + 1] - psi_perp[idx])

    C = (psi_perp_ij * (ri_dot_rj - ri_dot_s * rj_dot_s)
         + psi_par_ij * ri_dot_s * rj_dot_s)

    C = C.at[jnp.diag_indices(N)].set(sigma_v_sq_over3)

    return C


###############################################################################
# Log-likelihood
###############################################################################


def pv_covariance_log_likelihood(v_obs, v_pred, C, beta, sigma_v):
    """Log-likelihood of observed PV given predicted PV and covariance.

    Computes:
        log p(v_obs | v_pred, C, β, σ_v)

    where C_total = β² C + σ_v² I.

    Parameters
    ----------
    v_obs : JAX array (N,)
    v_pred : JAX array (N,)
    C : JAX array (N, N)
        Covariance matrix from `assemble_pv_covariance_jax` (unit β).
    beta : JAX scalar
    sigma_v : JAX scalar

    Returns
    -------
    log_prob : JAX scalar
    """
    import jax.numpy as jnp

    N = v_obs.shape[0]
    C_total = beta**2 * C + sigma_v**2 * jnp.eye(N)
    L = jnp.linalg.cholesky(C_total)

    residual = v_obs - v_pred
    alpha = jnp.linalg.solve(L, residual)

    log_det = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    log_prob = -0.5 * (N * jnp.log(2 * jnp.pi) + log_det
                       + jnp.dot(alpha, alpha))
    return log_prob
