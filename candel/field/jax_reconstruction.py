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
JAX-jittable density field reconstruction for anisotropic H0 inference.

This module provides a fast, differentiable single-iteration density
reconstruction that outputs LOS density and velocity profiles for clusters.
"""
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import gammaincc, gamma as gamma_func
import numpy as np
import h5py

# Physical constants
H0_BAR = 100.0      # km/s/Mpc (fiducial)
C_LIGHT = 299792.458  # km/s
Q0 = -0.55          # Deceleration parameter

# Schechter LF defaults (2M++ values)
ALPHA_DEFAULT = -1.02
MSTAR_DEFAULT = -23.55
BIAS_CONST = 0.73
BIAS_SLOPE = 0.24


class PrecomputedData:
    """Container for precomputed reconstruction data.

    Registered as a JAX pytree for use in jitted functions.
    """
    # Separate arrays (dynamic) from scalars (static)
    _array_fields = [
        'coords', 'psi_3d', 'nonmasked', 'kernel_fft', 'k_grid', 'k_mag_safe',
        'cluster_rhat', 'los_r', 'los_positions',
        'gal_rhat', 'gal_z_obs', 'gal_K2Mpp', 'gal_m_b', 'gal_m_f',
        'clone_source_idx', 'clone_rhat'
    ]
    _static_fields = ['N', 'dx', 'RMAX', 'alpha', 'Mstar', 'cf', 'cb', 'beta',
                      'r0_decay_scale']

    def __init__(
        self,
        N: int,
        dx: float,
        RMAX: float,
        coords: jnp.ndarray,
        psi_3d: jnp.ndarray,
        nonmasked: jnp.ndarray,
        kernel_fft: jnp.ndarray,
        k_grid: jnp.ndarray,
        k_mag_safe: jnp.ndarray,
        cluster_rhat: jnp.ndarray,
        los_r: jnp.ndarray,
        los_positions: jnp.ndarray,
        gal_rhat: jnp.ndarray,
        gal_z_obs: jnp.ndarray,
        gal_K2Mpp: jnp.ndarray,
        gal_m_b: jnp.ndarray,
        gal_m_f: jnp.ndarray,
        clone_source_idx: jnp.ndarray,
        clone_rhat: jnp.ndarray,
        alpha: float,
        Mstar: float,
        cf: float,
        cb: float,
        beta: float,
        r0_decay_scale: float,
    ):
        # Grid
        self.N = N
        self.dx = dx
        self.RMAX = RMAX
        self.coords = coords
        self.psi_3d = psi_3d
        self.nonmasked = nonmasked

        # FFT data
        self.kernel_fft = kernel_fft
        self.k_grid = k_grid
        self.k_mag_safe = k_mag_safe

        # Cluster data
        self.cluster_rhat = cluster_rhat
        self.los_r = los_r
        self.los_positions = los_positions

        # Galaxy data
        self.gal_rhat = gal_rhat
        self.gal_z_obs = gal_z_obs
        self.gal_K2Mpp = gal_K2Mpp
        self.gal_m_b = gal_m_b
        self.gal_m_f = gal_m_f

        # ZoA cloning
        self.clone_source_idx = clone_source_idx
        self.clone_rhat = clone_rhat

        # Scalars
        self.alpha = alpha
        self.Mstar = Mstar
        self.cf = cf
        self.cb = cb
        self.beta = beta
        self.r0_decay_scale = r0_decay_scale


def _precomputed_flatten(p):
    """Flatten PrecomputedData for JAX pytree."""
    children = tuple(getattr(p, f) for f in PrecomputedData._array_fields)
    aux_data = tuple(getattr(p, f) for f in PrecomputedData._static_fields)
    return children, aux_data


def _precomputed_unflatten(aux_data, children):
    """Unflatten PrecomputedData for JAX pytree."""
    kwargs = dict(zip(PrecomputedData._array_fields, children))
    kwargs.update(dict(zip(PrecomputedData._static_fields, aux_data)))
    return PrecomputedData(**kwargs)


# Register as JAX pytree
jax.tree_util.register_pytree_node(
    PrecomputedData,
    _precomputed_flatten,
    _precomputed_unflatten
)


# =============================================================================
# Helper functions
# =============================================================================

def gamma_upper_jax(s, x):
    """Upper incomplete gamma function: Γ(s,x) = gammaincc(s,x) × Γ(s)."""
    return gammaincc(s, x) * gamma_func(s)


def compute_weights_jax(r_mpc, K2Mpp, m_b, m_f, alpha, Mstar, cf, cb):
    """
    Compute total weight = lumWeight × L/L* using full Schechter integral.

    Parameters
    ----------
    r_mpc : (n_gal,) array
        Distances in Mpc/h
    K2Mpp : (n_gal,) array
        K-band apparent magnitudes
    m_b, m_f : (n_gal,) arrays
        Bright and faint magnitude limits per galaxy
    alpha, Mstar : float
        Schechter LF parameters
    cf, cb : float
        Completeness fractions

    Returns
    -------
    w_total : (n_gal,) array
        Total weight = lumWeight × L/L*
    """
    # Distance modulus
    mu = 5.0 * jnp.log10(jnp.maximum(r_mpc, 0.1)) + 25.0

    # Absolute magnitude and L/L*
    M_abs = K2Mpp - mu
    L_over_Lstar = 10.0 ** (-0.4 * (M_abs - Mstar))

    # Magnitude limits in absolute terms
    Mb_iter = jnp.clip(m_b - mu, -26.0, -17.0)
    Mf_iter = jnp.clip(m_f - mu, -26.0, -17.0)

    # Schechter integral: Γ(α+2, x)
    a = alpha + 2.0

    def integral_lw(M_faint, M_bright):
        """Integral of L-weighted LF from M_bright to M_faint."""
        x_f = 10.0 ** (0.4 * (Mstar - M_faint))
        x_b = 10.0 ** (0.4 * (Mstar - M_bright))
        return gamma_upper_jax(a, x_f) - gamma_upper_jax(a, x_b)

    # Selection function (Carrick Eq. 8-9)
    numer = (cf * integral_lw(Mf_iter, Mb_iter) +
             cb * integral_lw(Mb_iter, jnp.full_like(Mb_iter, -26.0)))
    denom = integral_lw(jnp.array(-17.0), jnp.array(-26.0))

    getWeight = jnp.maximum(numer / denom, 1e-10)
    lumWeight = 1.0 / getWeight

    return lumWeight * L_over_Lstar


def apply_zoa_cloning(r_mpc, w_total, gal_rhat, clone_source_idx, clone_rhat):
    """
    Add ZoA clones with same distances but reflected directions.

    Parameters
    ----------
    r_mpc : (n_gal,) array
        Galaxy distances
    w_total : (n_gal,) array
        Galaxy weights
    gal_rhat : (n_gal, 3) array
        Galaxy unit direction vectors
    clone_source_idx : (n_clone,) int array
        Index of source galaxy for each clone
    clone_rhat : (n_clone, 3) array
        Unit vectors for cloned positions

    Returns
    -------
    r_all, w_all, rhat_all : arrays
        Concatenated original + clone data
    """
    # Get clone distances and weights from source galaxies
    r_clone = r_mpc[clone_source_idx]
    w_clone = w_total[clone_source_idx]

    # Concatenate original + clones
    r_all = jnp.concatenate([r_mpc, r_clone])
    w_all = jnp.concatenate([w_total, w_clone])
    rhat_all = jnp.concatenate([gal_rhat, clone_rhat])

    return r_all, w_all, rhat_all


def cic_deposit_jax(positions, weights, N, dx, RMAX):
    """
    Cloud-in-Cell deposition using JAX scatter operations.

    Parameters
    ----------
    positions : (n_particles, 3) array
        Cartesian positions in Mpc/h
    weights : (n_particles,) array
        Particle weights
    N : int
        Grid size
    dx : float
        Cell size in Mpc/h
    RMAX : float
        Half box size in Mpc/h

    Returns
    -------
    rho : (N, N, N) array
        Density field
    """
    # Convert physical to grid coordinates
    gc = (positions + RMAX) / dx - 0.5  # (n_particles, 3)
    i0 = jnp.floor(gc).astype(jnp.int32)
    f = gc - i0  # Fractional part

    # Initialize density grid
    rho = jnp.zeros((N, N, N))

    # CIC weights for each corner
    wx = jnp.stack([1.0 - f[:, 0], f[:, 0]], axis=1)  # (n, 2)
    wy = jnp.stack([1.0 - f[:, 1], f[:, 1]], axis=1)
    wz = jnp.stack([1.0 - f[:, 2], f[:, 2]], axis=1)

    # Loop over 8 corners
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                # Indices for this corner
                idx_i = jnp.clip(i0[:, 0] + di, 0, N - 1)
                idx_j = jnp.clip(i0[:, 1] + dj, 0, N - 1)
                idx_k = jnp.clip(i0[:, 2] + dk, 0, N - 1)

                # Combined weight
                w = weights * wx[:, di] * wy[:, dj] * wz[:, dk]

                # Scatter add
                rho = rho.at[idx_i, idx_j, idx_k].add(w)

    return rho


def gaussian_smooth_fft(delta, kernel_fft):
    """
    Gaussian smoothing via FFT convolution with precomputed kernel.

    Parameters
    ----------
    delta : (N, N, N) array
        Density contrast field
    kernel_fft : (N, N, N//2+1) complex array
        FFT of Gaussian kernel

    Returns
    -------
    smoothed : (N, N, N) array
        Smoothed density contrast
    """
    N = delta.shape[0]
    delta_fft = jnp.fft.rfftn(delta)
    smoothed_fft = delta_fft * kernel_fft
    return jnp.fft.irfftn(smoothed_fft, s=(N, N, N))


def velocity_from_density_fft(delta, beta, k_grid, k_mag_safe):
    """
    Compute velocity field from density via FFT.

    v_k = i * beta * H0 * delta_k * k / k²

    Parameters
    ----------
    delta : (N, N, N) array
        Smoothed density contrast
    beta : float
        RSD parameter (f/b ≈ 0.43)
    k_grid : (N, N, N//2+1, 3) array
        k-vectors
    k_mag_safe : (N, N, N//2+1) array
        |k| with k=0 set to safe value

    Returns
    -------
    velocity : (N, N, N, 3) array
        Velocity field in km/s
    """
    N = delta.shape[0]
    delta_k = jnp.fft.rfftn(delta)

    # v_k = i * beta * H0 * delta_k * k / k²
    factor = 1j * beta * H0_BAR / k_mag_safe**2  # (N, N, N//2+1)

    # Compute velocity in k-space for each component
    vx_k = factor * delta_k * k_grid[..., 0]
    vy_k = factor * delta_k * k_grid[..., 1]
    vz_k = factor * delta_k * k_grid[..., 2]

    # IFFT back to real space
    vx = jnp.fft.irfftn(vx_k, s=(N, N, N))
    vy = jnp.fft.irfftn(vy_k, s=(N, N, N))
    vz = jnp.fft.irfftn(vz_k, s=(N, N, N))

    return jnp.stack([vx, vy, vz], axis=-1)


def trilinear_interp_batch(field, positions, N, dx, RMAX):
    """
    Trilinear interpolation of 3D field at multiple positions.

    Parameters
    ----------
    field : (N, N, N) array
        3D field to interpolate
    positions : (..., 3) array
        Positions in Mpc/h (can be any shape with last dim = 3)
    N : int
        Grid size
    dx : float
        Cell size
    RMAX : float
        Half box size

    Returns
    -------
    values : (...) array
        Interpolated values at positions
    """
    # Convert to grid coordinates
    gc = (positions + RMAX) / dx  # (..., 3)
    i0 = jnp.floor(gc).astype(jnp.int32)
    f = gc - i0

    # Clip indices to valid range
    i0 = jnp.clip(i0, 0, N - 2)
    i1 = i0 + 1

    # Extract indices for each dimension
    i0x, i0y, i0z = i0[..., 0], i0[..., 1], i0[..., 2]
    i1x, i1y, i1z = i1[..., 0], i1[..., 1], i1[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

    # Gather 8 corner values
    v000 = field[i0x, i0y, i0z]
    v001 = field[i0x, i0y, i1z]
    v010 = field[i0x, i1y, i0z]
    v011 = field[i0x, i1y, i1z]
    v100 = field[i1x, i0y, i0z]
    v101 = field[i1x, i0y, i1z]
    v110 = field[i1x, i1y, i0z]
    v111 = field[i1x, i1y, i1z]

    # Trilinear interpolation
    result = (v000 * (1 - fx) * (1 - fy) * (1 - fz) +
              v001 * (1 - fx) * (1 - fy) * fz +
              v010 * (1 - fx) * fy * (1 - fz) +
              v011 * (1 - fx) * fy * fz +
              v100 * fx * (1 - fy) * (1 - fz) +
              v101 * fx * (1 - fy) * fz +
              v110 * fx * fy * (1 - fz) +
              v111 * fx * fy * fz)

    return result


# =============================================================================
# Main JIT function
# =============================================================================

@jit
def compute_los_profiles_jax(
    H0_dipole: jnp.ndarray,
    p: PrecomputedData,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute LOS density and velocity profiles for all clusters.

    This is the main JAX-jittable function for anisotropic H0 inference.

    Parameters
    ----------
    H0_dipole : (3,) array
        Dipole vector d = amplitude × unit_vector
        H0(n̂) = H0_bar × (1 + d · n̂)
    p : PrecomputedData
        Precomputed static data

    Returns
    -------
    los_density : (n_clusters, n_r) array
        LOS density profiles (1 + δ)
    los_velocity : (n_clusters, n_r) array
        LOS velocity profiles [km/s]
    """
    # 1. Galaxy distances from H0(n̂)
    H0_dir = H0_BAR * (1.0 + jnp.dot(p.gal_rhat, H0_dipole))
    r_mpc = (C_LIGHT / H0_dir) * (p.gal_z_obs - (1.0 + Q0) / 2.0 * p.gal_z_obs**2)

    # 2. Weights (full Schechter lumWeight × L/L*)
    w_total = compute_weights_jax(
        r_mpc, p.gal_K2Mpp, p.gal_m_b, p.gal_m_f,
        p.alpha, p.Mstar, p.cf, p.cb
    )

    # 3. ZoA cloning
    r_all, w_all, rhat_all = apply_zoa_cloning(
        r_mpc, w_total, p.gal_rhat,
        p.clone_source_idx, p.clone_rhat
    )

    # 4. Positions
    positions = r_all[:, None] * rhat_all  # (n_gal+n_clone, 3)

    # 5. CIC deposition
    rho = cic_deposit_jax(positions, w_all, p.N, p.dx, p.RMAX)

    # 6. Normalize
    rho_mean = jnp.sum(rho * p.nonmasked) / jnp.sum(p.nonmasked)
    delta = rho / jnp.maximum(rho_mean, 1e-10) - 1.0
    delta = jnp.where(p.nonmasked, delta, 0.0)

    # 7. Divide by psi (bias normalization)
    delta = delta / p.psi_3d
    delta = jnp.where(p.nonmasked, delta, 0.0)

    # 8. Gaussian smooth
    delta = gaussian_smooth_fft(delta, p.kernel_fft)

    # 9. Velocity field from density
    velocity = velocity_from_density_fft(delta, p.beta, p.k_grid, p.k_mag_safe)

    # 10. Extract LOS profiles via trilinear interpolation
    los_density = trilinear_interp_batch(1.0 + delta, p.los_positions,
                                          p.N, p.dx, p.RMAX)

    # Velocity: interpolate each component and dot with cluster_rhat
    vx = trilinear_interp_batch(velocity[..., 0], p.los_positions,
                                 p.N, p.dx, p.RMAX)
    vy = trilinear_interp_batch(velocity[..., 1], p.los_positions,
                                 p.N, p.dx, p.RMAX)
    vz = trilinear_interp_batch(velocity[..., 2], p.los_positions,
                                 p.N, p.dx, p.RMAX)

    # v_LOS = v · r̂_cluster  (broadcast over n_r)
    los_velocity = (vx * p.cluster_rhat[:, None, 0] +
                    vy * p.cluster_rhat[:, None, 1] +
                    vz * p.cluster_rhat[:, None, 2])

    # 11. Apply edge smoothing for r > RMAX
    # Exponential decay to bring delta → 0 and velocity → 0 beyond box edge
    # f(r) = f_interp * exp(-(r - RMAX) / r0) for r > RMAX
    delta_excess = jnp.maximum(p.los_r - p.RMAX, 0.0)  # (n_r,)
    decay = jnp.exp(-delta_excess / p.r0_decay_scale)  # (n_r,)

    # For density: (1+delta)_smoothed = 1 + (los_density - 1) * decay
    los_delta = los_density - 1.0
    los_density = 1.0 + los_delta * decay[None, :]

    # For velocity: v_smoothed = v * decay
    los_velocity = los_velocity * decay[None, :]

    return los_density, los_velocity


# =============================================================================
# Precomputation
# =============================================================================

def compute_psi_numpy(r_mpc, m_lim=11.5, alpha=ALPHA_DEFAULT, Mstar=MSTAR_DEFAULT):
    """
    Compute bias normalization psi(r) using numpy (for precomputation).

    psi(r) = b_const + b_slope × ⟨L/L*⟩(r)
    """
    from scipy.special import gammaincc as np_gammaincc
    from scipy.special import gamma as np_gamma

    def np_gamma_upper(s, x):
        return np_gammaincc(s, x) * np_gamma(s)

    r_safe = np.maximum(r_mpc, 0.1)
    mu = 5.0 * np.log10(r_safe) + 25.0
    M_lim = m_lim - mu
    x_lim = 10.0 ** (-0.4 * (M_lim - Mstar))

    a2 = alpha + 2.0
    a3 = alpha + 3.0
    gamma_a2 = np_gamma_upper(a2, x_lim)
    gamma_a3 = np_gamma_upper(a3, x_lim)
    gamma_a2 = np.where(np.abs(gamma_a2) < 1e-12, 1e-12, gamma_a2)

    L_mean = gamma_a3 / gamma_a2
    return BIAS_CONST + BIAS_SLOPE * L_mean


def gaussian_kernel_3d_fft(N, sigma_vox):
    """
    Compute FFT of 3D Gaussian kernel for convolution.

    Parameters
    ----------
    N : int
        Grid size
    sigma_vox : float
        Smoothing scale in voxels

    Returns
    -------
    kernel_fft : (N, N, N//2+1) complex array
        FFT of normalized Gaussian kernel
    """
    # Create centered kernel
    x = np.fft.fftfreq(N) * N
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    r2 = xx**2 + yy**2 + zz**2

    kernel = np.exp(-r2 / (2 * sigma_vox**2))
    kernel /= kernel.sum()  # Normalize

    return np.fft.rfftn(kernel)


def compute_k_grid(N, dx):
    """
    Compute k-vectors for velocity FFT.

    Parameters
    ----------
    N : int
        Grid size
    dx : float
        Cell size in Mpc/h

    Returns
    -------
    k_grid : (N, N, N//2+1, 3) array
        k-vectors
    k_mag_safe : (N, N, N//2+1) array
        |k| with k=0 set to 1.0 to avoid division by zero
    """
    # k in units of 1/Mpc/h
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    kz = np.fft.rfftfreq(N, d=dx) * 2 * np.pi

    kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k_grid = np.stack([kx_3d, ky_3d, kz_3d], axis=-1)

    k_mag = np.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)
    k_mag_safe = np.where(k_mag > 0, k_mag, 1.0)  # Avoid division by zero

    return k_grid, k_mag_safe


def ra_dec_to_galactic_rhat(ra_deg, dec_deg):
    """
    Convert RA/Dec to Galactic unit direction vectors.

    Parameters
    ----------
    ra_deg, dec_deg : (n,) arrays
        Right ascension and declination in degrees

    Returns
    -------
    rhat : (n, 3) array
        Unit vectors in Galactic Cartesian coordinates
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
    gal = c.galactic

    l_rad = gal.l.rad
    b_rad = gal.b.rad

    cos_b = np.cos(b_rad)
    rhat = np.stack([
        cos_b * np.cos(l_rad),
        cos_b * np.sin(l_rad),
        np.sin(b_rad)
    ], axis=1)

    return rhat


def lb_to_rhat(l_deg, b_deg):
    """
    Convert Galactic l, b to unit direction vectors.

    Parameters
    ----------
    l_deg, b_deg : (n,) arrays
        Galactic longitude and latitude in degrees

    Returns
    -------
    rhat : (n, 3) array
        Unit vectors in Galactic Cartesian coordinates
    """
    l_rad = np.deg2rad(l_deg)
    b_rad = np.deg2rad(b_deg)

    cos_b = np.cos(b_rad)
    rhat = np.stack([
        cos_b * np.cos(l_rad),
        cos_b * np.sin(l_rad),
        np.sin(b_rad)
    ], axis=1)

    return rhat


def precompute_reconstruction_data(
    cluster_file: str,
    galaxy_catalogue: dict,
    clone_source_idx: np.ndarray,
    clone_l_deg: np.ndarray,
    clone_b_deg: np.ndarray,
    N: int = 128,
    BOX_SIDE: float = 400.0,
    SIGMA_SMOOTH: float = 4.0,
    alpha: float = ALPHA_DEFAULT,
    Mstar: float = MSTAR_DEFAULT,
    cf: float = 1.0,
    cb: float = 1.0,
    beta: float = 0.43,
    r0_decay_scale: float = 5.0,
    r_grid: np.ndarray = None,
) -> PrecomputedData:
    """
    Precompute all static quantities for JAX reconstruction.

    Parameters
    ----------
    cluster_file : str
        Path to HDF5 file with cluster LOS data
    galaxy_catalogue : dict
        Galaxy data with keys: 'l_deg', 'b_deg', 'z_obs', 'K2Mpp', 'm_b', 'm_f'
    clone_source_idx : (n_clone,) int array
        Index of source galaxy for each ZoA clone
    clone_l_deg, clone_b_deg : (n_clone,) arrays
        Galactic coordinates for cloned positions
    N : int
        Grid resolution (default 128)
    BOX_SIDE : float
        Box size in Mpc/h (default 400)
    SIGMA_SMOOTH : float
        Gaussian smoothing scale in Mpc/h (default 4)
    alpha, Mstar : float
        Schechter LF parameters
    cf, cb : float
        Completeness fractions
    beta : float
        RSD parameter
    r0_decay_scale : float
        Edge smoothing decay scale in Mpc/h (default 5.0).
        For r > RMAX, delta and velocity decay exponentially:
        f(r) = f(RMAX) * exp(-(r - RMAX) / r0_decay_scale)
    r_grid : (n_r,) array, optional
        Custom radial grid for LOS output (e.g., Malmquist integration grid).
        If None, uses the 'r' array from cluster_file.

    Returns
    -------
    PrecomputedData
        Container with all precomputed arrays
    """
    RMAX = BOX_SIDE / 2.0
    dx = BOX_SIDE / N

    # Grid setup
    coords = np.linspace(-RMAX + dx/2, RMAX - dx/2, N)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
    rr = np.sqrt(xx**2 + yy**2 + zz**2)

    # Psi lookup (1D table, interpolated to 3D)
    r_psi = np.linspace(0.1, RMAX, 50)
    psi_table = compute_psi_numpy(r_psi, m_lim=11.5, alpha=alpha, Mstar=Mstar)
    psi_3d = np.interp(rr.ravel(), r_psi, psi_table).reshape(rr.shape)
    psi_3d = np.maximum(psi_3d, 0.1)

    # Nonmasked region (inside box)
    nonmasked = rr < RMAX

    # Gaussian kernel FFT
    sigma_vox = SIGMA_SMOOTH / dx
    kernel_fft = gaussian_kernel_3d_fft(N, sigma_vox)

    # k-grid for velocity FFT
    k_grid, k_mag_safe = compute_k_grid(N, dx)

    # Load cluster data
    with h5py.File(cluster_file, 'r') as f:
        cluster_RA = f['RA'][:]
        cluster_dec = f['dec'][:]
        # Use custom r_grid if provided, otherwise load from file
        if r_grid is None:
            los_r = f['r'][:]
        else:
            los_r = np.asarray(r_grid)

    # Cluster unit vectors
    cluster_rhat = ra_dec_to_galactic_rhat(cluster_RA, cluster_dec)

    # Precompute all LOS positions: (n_clusters, n_r, 3)
    los_positions = los_r[None, :, None] * cluster_rhat[:, None, :]

    # Galaxy data
    gal_rhat = lb_to_rhat(galaxy_catalogue['l_deg'], galaxy_catalogue['b_deg'])

    # Clone unit vectors
    clone_rhat = lb_to_rhat(clone_l_deg, clone_b_deg)

    return PrecomputedData(
        N=N,
        dx=dx,
        RMAX=RMAX,
        coords=jnp.array(coords, dtype=jnp.float32),
        psi_3d=jnp.array(psi_3d, dtype=jnp.float32),
        nonmasked=jnp.array(nonmasked),
        kernel_fft=jnp.array(kernel_fft),
        k_grid=jnp.array(k_grid, dtype=jnp.float32),
        k_mag_safe=jnp.array(k_mag_safe, dtype=jnp.float32),
        cluster_rhat=jnp.array(cluster_rhat, dtype=jnp.float32),
        los_r=jnp.array(los_r, dtype=jnp.float32),
        los_positions=jnp.array(los_positions, dtype=jnp.float32),
        gal_rhat=jnp.array(gal_rhat, dtype=jnp.float32),
        gal_z_obs=jnp.array(galaxy_catalogue['z_obs'], dtype=jnp.float32),
        gal_K2Mpp=jnp.array(galaxy_catalogue['K2Mpp'], dtype=jnp.float32),
        gal_m_b=jnp.array(galaxy_catalogue['m_b'], dtype=jnp.float32),
        gal_m_f=jnp.array(galaxy_catalogue['m_f'], dtype=jnp.float32),
        clone_source_idx=jnp.array(clone_source_idx, dtype=jnp.int32),
        clone_rhat=jnp.array(clone_rhat, dtype=jnp.float32),
        alpha=alpha,
        Mstar=Mstar,
        cf=cf,
        cb=cb,
        beta=beta,
        r0_decay_scale=r0_decay_scale,
    )
