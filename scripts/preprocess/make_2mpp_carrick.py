"""
Reproduce the Carrick2015 density/velocity field reconstruction with optional
anisotropic H0 dipole extension.

This implements the iterative RSD correction procedure from Carrick et al. (2015)
Sections 2.1-2.5, including:
- Luminosity weighting with Schechter LF
- Galaxy bias normalization
- Iterative beta adiabatic increase (0 -> 1)
- Distance averaging to suppress oscillations

References:
    Carrick et al. (2015): https://arxiv.org/abs/1504.04627

Ambiguities in the original paper (documented):
    - Grid resolution: 256^3 (Carrick uses 257^3)
    - FoF linking length: Use pre-computed groups from catalogue
    - Bias correction timing: After delta_g, before smoothing
    - "Previous 5 predictions": Last 5 iterations, not including current
    - 2MRS cutoff: Zero weight for m_lim < 12.0 AND r > 125 h^-1 Mpc
    - FFT zero-padding: Use 2x padding to avoid wrap-around artifacts (not specified in paper)
"""
from __future__ import annotations

import argparse
from collections import deque
import os
import time
from typing import Dict, Tuple, Optional
import numpy as np
from scipy import special as sps
import healpy as hp

from candel.util import galactic_to_radec, galactic_to_radec_cartesian, radec_to_cartesian


def galactic_to_galactic_cartesian(ell_deg, b_deg):
    """
    Convert galactic coordinates (ell, b) in degrees to Galactic Cartesian unit vectors.

    Returns (n_gal, 3) array where:
    - x points toward Galactic Center (ell=0, b=0)
    - y points in direction of Galactic rotation (ell=90, b=0)
    - z points toward North Galactic Pole (b=90)
    """
    ell_rad = np.deg2rad(ell_deg)
    b_rad = np.deg2rad(b_deg)

    cos_b = np.cos(b_rad)
    x = cos_b * np.cos(ell_rad)
    y = cos_b * np.sin(ell_rad)
    z = np.sin(b_rad)

    return np.stack([x, y, z], axis=-1)

# Reuse utilities from make_2mpp_zspace
from .make_2mpp_zspace import (
    _load_groups,
    _sample_completeness,
    _cic_deposit,
    _gaussian_smooth_fft,
    _as_float32,
    H0_MIN,
    JAX_AVAILABLE,
)

if JAX_AVAILABLE:
    import jax.numpy as jnp
else:
    import numpy as jnp

C_LIGHT = 299792.458  # km/s

# Carrick2015 LF parameters (Section 2.2)
ALPHA_DEFAULT = -0.85
MSTAR_DEFAULT = -23.25  # K-band, h=1
M_MIN_DEFAULT = -20.0   # Minimum absolute magnitude for L_min

# Bias parameters (Section 2.4, Westover 2007)
BIAS_CONST = 0.73
BIAS_SLOPE = 0.24

# Beta* parameter (Carrick Section 3.1): β* = f(Ω_m)/b* ≈ 0.43
BETA_STAR = 0.43

# Universal magnitude limit for ψ(r) computation (Equation 8)
# Use 11.5 (2MRS limit) or 12.5 (full 2M++ limit)
PSI_MLIM_UNIVERSAL = 11.5

# If True, use PSI_MLIM_UNIVERSAL (11.5) for both 2MRS and Deep regions
# If False, use 11.5 for 2MRS and 12.5 for Deep regions
PSI_USE_UNIFORM_MLIM = True

# Deceleration parameter for distance calculation
# q0 = Omega_m/2 - Omega_Lambda = 0.3/2 - 0.7 = -0.55 for standard LCDM
Q0_DEFAULT = -0.55


def _gamma_upper(s: float, x: np.ndarray) -> np.ndarray:
    """
    Upper incomplete gamma function: Gamma(s, x) = integral[x, inf] t^(s-1) exp(-t) dt.

    Uses recurrence relation for s <= 0.
    """
    x = np.asarray(x)
    if s > 0:
        return sps.gammaincc(s, x) * sps.gamma(s)
    # Recurrence for negative s
    k = int(np.ceil(1.0 - s))
    sp = s + k
    upper = sps.gammaincc(sp, x) * sps.gamma(sp)
    current_s = sp
    for _ in range(k):
        upper = (upper - np.power(x, current_s - 1) * np.exp(-x)) / (current_s - 1)
        current_s -= 1
    return upper


def _load_galaxies_with_mag(path: str) -> Dict[str, np.ndarray]:
    """
    Load 2M++ galaxy catalogue including K-band magnitude.

    Columns needed: l, b, K2M++, Vcmb, GID
    """
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Designation" in line and "|" in line:
            header_idx = i
            break

    if header_idx is not None:
        # Standard 2m++.txt format with | delimiter
        # Columns: 0=Designation, 1=RA, 2=Dec, 3=l, 4=b, 5=K2M++, 6=Vh, 7=Vcmb, 8=Verr, 9=GID
        data = np.genfromtxt(
            path,
            delimiter="|",
            skip_header=header_idx + 2,
            usecols=[3, 4, 5, 7, 9],
        )
        data = np.atleast_2d(data)
        l_deg = data[:, 0]
        b_deg = data[:, 1]
        K2Mpp = data[:, 2]
        vcmb = data[:, 3]
        gid_raw = data[:, 4]
        gid = np.where(np.isfinite(gid_raw), gid_raw, -1).astype(int)
        return {
            "l_deg": l_deg,
            "b_deg": b_deg,
            "K2Mpp": K2Mpp,
            "vcmb": vcmb,
            "gid": gid,
        }

    # Fallback: try generic parsing
    data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    if data.dtype.names is None:
        raise ValueError(f"Galaxy catalogue {path} is missing a header row.")

    name_map = {
        "gid": "gid", "groupid": "gid", "group_id": "gid",
        "l": "l_deg", "l_deg": "l_deg", "glon": "l_deg",
        "b": "b_deg", "b_deg": "b_deg", "glat": "b_deg",
        "vcmb": "vcmb", "v_cmb": "vcmb", "cz": "vcmb", "cz_cmb": "vcmb",
        "k2mpp": "K2Mpp", "k2m++": "K2Mpp", "kmag": "K2Mpp", "k_mag": "K2Mpp",
    }
    out = {}
    for name in data.dtype.names:
        key = name_map.get(name.strip().lower())
        if key is not None:
            out[key] = data[name]

    if "gid" not in out:
        out["gid"] = np.full_like(out["vcmb"], -1, dtype=int)

    missing = {key for key in ("l_deg", "b_deg", "vcmb", "K2Mpp") if key not in out}
    if missing:
        raise ValueError(f"Galaxy catalogue {path} missing columns: {sorted(missing)}")

    return out


def compute_selection_function(
    r_mpc: np.ndarray,
    m_lim: np.ndarray,
    alpha: float = ALPHA_DEFAULT,
    M_star: float = MSTAR_DEFAULT,
    M_min: float = M_MIN_DEFAULT,
) -> np.ndarray:
    """
    Compute selection function S(r) from Carrick Section 2.2.

    S(r) = Gamma(alpha+2, x_lim) / Gamma(alpha+2, x_min)

    This is the fraction of the luminosity function observable at distance r.

    Parameters
    ----------
    r_mpc : array
        Comoving distances in h^-1 Mpc
    m_lim : array
        Apparent magnitude limit (11.5 or 12.5)
    alpha : float
        Schechter function faint-end slope
    M_star : float
        Characteristic absolute magnitude
    M_min : float
        Minimum absolute magnitude (defines L_min)

    Returns
    -------
    S : array
        Selection function (0 to 1)
    """
    r_safe = np.maximum(r_mpc, 0.1)
    mu = 5.0 * np.log10(r_safe) + 25.0
    M_lim = m_lim - mu

    x_lim = 10.0 ** (-0.4 * (M_lim - M_star))
    x_min = 10.0 ** (-0.4 * (M_min - M_star))

    a = alpha + 2.0
    gamma_x_lim = _gamma_upper(a, x_lim)
    gamma_x_min = _gamma_upper(a, np.full_like(x_lim, x_min))

    # S(r) = fraction observable = Gamma(a, x_lim) / Gamma(a, x_min)
    gamma_x_min = np.where(np.abs(gamma_x_min) < 1e-12, 1e-12, gamma_x_min)
    S = gamma_x_lim / gamma_x_min

    # Clip to valid range
    S = np.clip(S, 1e-6, 1.0)

    return S


def compute_luminosity_weight(
    r_mpc: np.ndarray,
    m_lim: np.ndarray,
    alpha: float = ALPHA_DEFAULT,
    M_star: float = MSTAR_DEFAULT,
    M_min: float = M_MIN_DEFAULT,
) -> np.ndarray:
    """
    Compute luminosity weight w_L = 1/S(r) from Carrick Section 2.2.

    Returns
    -------
    w_L : array
        Luminosity selection weights (= 1/S(r))
    """
    S = compute_selection_function(r_mpc, m_lim, alpha, M_star, M_min)
    return 1.0 / S


def compute_bias_normalization(
    r_mpc: np.ndarray,
    m_lim: np.ndarray,
    alpha: float = ALPHA_DEFAULT,
    M_star: float = MSTAR_DEFAULT,
    b_const: float = BIAS_CONST,
    b_slope: float = BIAS_SLOPE,
) -> np.ndarray:
    """
    Compute luminosity-weighted effective bias psi(r) from Carrick Section 2.4.

    psi(r) = b_eff(r) / b* = b_const + b_slope * <L/L*>

    where <L/L*> = Gamma(alpha+3, x_lim) / Gamma(alpha+2, x_lim)

    Parameters
    ----------
    r_mpc : array
        Comoving distances in h^-1 Mpc
    m_lim : array
        Apparent magnitude limit
    alpha : float
        Schechter function faint-end slope
    M_star : float
        Characteristic absolute magnitude
    b_const, b_slope : float
        Bias relation parameters: b/b* = b_const + b_slope * L/L*

    Returns
    -------
    psi : array
        Bias normalization factors
    """
    r_safe = np.maximum(r_mpc, 0.1)
    mu = 5.0 * np.log10(r_safe) + 25.0
    M_lim = m_lim - mu
    x_lim = 10.0 ** (-0.4 * (M_lim - M_star))

    a2 = alpha + 2.0
    a3 = alpha + 3.0

    gamma_a2 = _gamma_upper(a2, x_lim)
    gamma_a3 = _gamma_upper(a3, x_lim)

    gamma_a2 = np.where(np.abs(gamma_a2) < 1e-12, 1e-12, gamma_a2)
    L_over_Lstar_mean = gamma_a3 / gamma_a2

    psi = b_const + b_slope * L_over_Lstar_mean

    return psi


def velocity_from_density_fft(
    delta: np.ndarray,
    beta: float,
    box_side_mpc: float,
    zero_pad: bool = True,
) -> np.ndarray:
    """
    Compute velocity field from density using FFT (Carrick Eq. 1).

    v(r) = (beta / 4*pi) * integral delta(r') * (r' - r) / |r' - r|^3 d^3r'

    In Fourier space: v_k = -i * beta * k / k^2 * delta_k

    Parameters
    ----------
    delta : array
        3D overdensity field, shape (N, N, N)
    beta : float
        Beta* parameter = f(Omega_m) / b*
    box_side_mpc : float
        Box side length in h^-1 Mpc
    zero_pad : bool
        Whether to zero-pad to avoid wrap-around artifacts

    Returns
    -------
    velocity : array
        3D velocity field, shape (N, N, N, 3) in km/s
    """
    N = delta.shape[0]

    if zero_pad:
        # Pad to 2x size to avoid wrap-around
        N_pad = 2 * N
        delta_padded = np.zeros((N_pad, N_pad, N_pad), dtype=np.float64)
        delta_padded[:N, :N, :N] = delta
        dx = box_side_mpc / N  # Original voxel size
        box_side_pad = box_side_mpc * 2
    else:
        delta_padded = delta
        N_pad = N
        dx = box_side_mpc / N
        box_side_pad = box_side_mpc

    dx_pad = box_side_pad / N_pad

    # FFT of density
    delta_k = np.fft.rfftn(delta_padded)

    # Wave vectors
    kx = 2.0 * np.pi * np.fft.fftfreq(N_pad, d=dx_pad)
    ky = 2.0 * np.pi * np.fft.fftfreq(N_pad, d=dx_pad)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N_pad, d=dx_pad)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0  # Avoid division by zero at k=0

    # Velocity in Fourier space (Carrick Eq. 1):
    # v(r) = (f / 4π) * integral[ δ(r') * (r'-r) / |r'-r|³ ] d³r'
    #
    # In Fourier space: v_k = +i * β * H0 * (k/k²) * δ_k
    # The 4π factors cancel: (f/4π) in real space × (4πi k/k²) from the FT = f × i k/k²
    #
    # Note: Carrick's β* = 0.43, but we iterate from 0 to 1 for convergence
    H0 = 100.0  # h km/s / Mpc
    prefactor = 1j * beta * H0 / K2

    vx_k = prefactor * KX * delta_k
    vy_k = prefactor * KY * delta_k
    vz_k = prefactor * KZ * delta_k

    # Zero out k=0 mode (mean velocity)
    vx_k[0, 0, 0] = 0.0
    vy_k[0, 0, 0] = 0.0
    vz_k[0, 0, 0] = 0.0

    # IFFT back to real space
    vx = np.fft.irfftn(vx_k, s=(N_pad, N_pad, N_pad))
    vy = np.fft.irfftn(vy_k, s=(N_pad, N_pad, N_pad))
    vz = np.fft.irfftn(vz_k, s=(N_pad, N_pad, N_pad))

    if zero_pad:
        # Crop back to original size
        vx = vx[:N, :N, :N]
        vy = vy[:N, :N, :N]
        vz = vz[:N, :N, :N]

    velocity = np.stack([vx, vy, vz], axis=-1).astype(np.float32)

    return velocity


def trilinear_interp_velocity(
    velocity_field: np.ndarray,
    positions: np.ndarray,
    box_side_mpc: float,
) -> np.ndarray:
    """
    Trilinear interpolation of velocity field at given positions.

    Parameters
    ----------
    velocity_field : array
        3D velocity field, shape (N, N, N, 3)
    positions : array
        Galaxy positions in h^-1 Mpc, shape (n_gal, 3)
    box_side_mpc : float
        Box side length in h^-1 Mpc

    Returns
    -------
    v_interp : array
        Interpolated velocities, shape (n_gal, 3)
    """
    N = velocity_field.shape[0]
    Rmax = box_side_mpc / 2.0
    dx = box_side_mpc / N

    # Convert positions to grid coordinates (observer at center)
    coords = (positions + Rmax) / dx

    # Get integer indices and fractional parts
    i0 = np.floor(coords).astype(np.int32)
    f = coords - i0

    # Clip to valid range
    i0 = np.clip(i0, 0, N - 2)
    i1 = i0 + 1

    # Trilinear weights
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    wx0, wx1 = 1.0 - fx, fx
    wy0, wy1 = 1.0 - fy, fy
    wz0, wz1 = 1.0 - fz, fz

    # Interpolate each velocity component
    v_interp = np.zeros((positions.shape[0], 3), dtype=np.float32)

    for c in range(3):
        field = velocity_field[:, :, :, c]

        v000 = field[i0[:, 0], i0[:, 1], i0[:, 2]]
        v100 = field[i1[:, 0], i0[:, 1], i0[:, 2]]
        v010 = field[i0[:, 0], i1[:, 1], i0[:, 2]]
        v110 = field[i1[:, 0], i1[:, 1], i0[:, 2]]
        v001 = field[i0[:, 0], i0[:, 1], i1[:, 2]]
        v101 = field[i1[:, 0], i0[:, 1], i1[:, 2]]
        v011 = field[i0[:, 0], i1[:, 1], i1[:, 2]]
        v111 = field[i1[:, 0], i1[:, 1], i1[:, 2]]

        vals = (
            v000 * wx0 * wy0 * wz0
            + v100 * wx1 * wy0 * wz0
            + v010 * wx0 * wy1 * wz0
            + v110 * wx1 * wy1 * wz0
            + v001 * wx0 * wy0 * wz1
            + v101 * wx1 * wy0 * wz1
            + v011 * wx0 * wy1 * wz1
            + v111 * wx1 * wy1 * wz1
        )
        v_interp[:, c] = vals

    return v_interp


def update_distances(
    z_obs: np.ndarray,
    v_los: np.ndarray,
    H0: float = 100.0,
    q0: float = Q0_DEFAULT,
) -> np.ndarray:
    """
    Update comoving distances using predicted LOS velocity.

    From Carrick Eqs. (2) and (3):
    (1 + z_obs) = (1 + z_cos) * (1 + v_pec/c)
    H0 * R = c * (z_cos - (1+q0)/2 * z_cos^2)

    Parameters
    ----------
    z_obs : array
        Observed redshifts
    v_los : array
        Predicted line-of-sight velocities in km/s
    H0 : float
        Hubble constant in km/s/Mpc
    q0 : float
        Deceleration parameter

    Returns
    -------
    R : array
        Updated comoving distances in h^-1 Mpc
    """
    # Cosmological redshift from observed redshift and peculiar velocity
    z_cos = (1.0 + z_obs) / (1.0 + v_los / C_LIGHT) - 1.0
    z_cos = np.maximum(z_cos, 1e-8)  # Avoid negative redshifts

    # Comoving distance from cosmological redshift
    R = (C_LIGHT / H0) * (z_cos - (1.0 + q0) / 2.0 * z_cos**2)

    return R


def compute_anisotropic_h0(
    rhat: np.ndarray,
    h_iso: float,
    dipole_ell: float,
    dipole_b: float,
    dipole_mag: float,
) -> np.ndarray:
    """
    Compute per-galaxy H0 with dipole anisotropy.

    From model.py: delta_H/H = 10^(0.5 * |d|) - 1
    Per-galaxy: H(n) = H_iso * (1 + delta_frac * cos_theta)

    Parameters
    ----------
    rhat : array
        Unit direction vectors, shape (n_gal, 3)
    h_iso : float
        Isotropic Hubble parameter (h = H0/100)
    dipole_ell : float
        Dipole galactic longitude in degrees
    dipole_b : float
        Dipole galactic latitude in degrees
    dipole_mag : float
        Dipole magnitude (in zeropoint magnitude units)

    Returns
    -------
    H_per_gal : array
        Per-galaxy H0 in km/s/Mpc
    """
    if np.abs(dipole_mag) < 1e-10:
        return np.full(len(rhat), h_iso * 100.0)

    # Dipole direction unit vector
    d_hat = galactic_to_radec_cartesian(dipole_ell, dipole_b)

    # Cosine of angle between galaxy direction and dipole
    cos_theta = np.sum(rhat * d_hat, axis=1)

    # Convert dipole magnitude to fractional H0 change
    # delta_frac = 10^(0.5 * dipole_mag) - 1
    delta_frac = np.power(10.0, 0.5 * dipole_mag) - 1.0

    H_per_gal = h_iso * 100.0 * (1.0 + delta_frac * cos_theta)

    # Ensure H0 doesn't go too low
    H_per_gal = np.maximum(H_per_gal, H0_MIN)

    return H_per_gal


def build_carrick_field(
    data_dir: str,
    N: int = 257,
    box_side_mpc: float = 400.0,
    sigma_mpc: float = 4.0,
    n_iterations: int = 101,
    n_avg: int = 5,
    r_2mrs_cutoff: float = 125.0,  # Carrick: "Galaxies from 2MRS with distances > 125 h^-1 Mpc are assigned weight zero"
    cmin: float = 0.5,
    alpha: float = ALPHA_DEFAULT,
    M_star: float = MSTAR_DEFAULT,
    M_min: float = M_MIN_DEFAULT,
    H0: float = 100.0,
    q0: float = Q0_DEFAULT,
    # Bias parameters
    b_const: float = BIAS_CONST,
    b_slope: float = BIAS_SLOPE,
    # FoF group handling
    sum_group_luminosities: bool = True,  # Sum all group members' luminosities at centroid
    # ZoA cloning (already in 2M++ catalogue, only needed during iteration)
    clone_zoa: bool = False,
    # Normalization method
    use_global_mean: bool = True,  # Per Carrick: "mean density within 200 h^-1 Mpc"
    # Anisotropic H0 parameters (hardcoded test values)
    aniso_h0: bool = False,
    dipole_ell: float = 140.34,
    dipole_b: float = 46.67,
    dipole_mag: float = 0.03,
    # Diagnostics
    plot_psi: bool = True,  # Plot psi(r) bias correction curve
    save_intermediate_iters: Optional[list] = None,  # List of iteration indices to save delta fields
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Build the Carrick2015-style density and velocity fields with iterative
    RSD correction.

    Parameters
    ----------
    data_dir : str
        Path to 2M++ data directory
    N : int
        Grid resolution (N^3)
    box_side_mpc : float
        Box side length in h^-1 Mpc
    sigma_mpc : float
        Gaussian smoothing scale in h^-1 Mpc
    n_iterations : int
        Number of beta iterations (typically 101: beta=0.00 to 1.00)
    n_avg : int
        Number of previous distance estimates to average
    r_2mrs_cutoff : float
        Distance cutoff for 2MRS galaxies in h^-1 Mpc
    cmin : float
        Minimum angular completeness (floor)
    alpha, M_star, M_min : float
        Schechter LF parameters
    H0, q0 : float
        Cosmological parameters
    aniso_h0 : bool
        Whether to use anisotropic H0 model
    dipole_ell, dipole_b, dipole_mag : float
        H0 dipole parameters (galactic coords and magnitude)
    verbose : bool
        Print progress information

    Returns
    -------
    result : dict
        Dictionary containing density field, velocity field, and metadata
    """
    # All distances in h^-1 Mpc (real space)
    Rmax_mpc = box_side_mpc / 2.0

    if verbose:
        print(f"Carrick2015 reconstruction: {N}^3 grid, {box_side_mpc} h^-1 Mpc box")
        print(f"Smoothing: {sigma_mpc} h^-1 Mpc, iterations: {n_iterations}")
        if aniso_h0:
            print(f"Anisotropic H0: ell={dipole_ell}, b={dipole_b}, mag={dipole_mag}")

    # Load groups
    group_path = os.path.join(data_dir, "2m++_groups.txt")
    if not os.path.exists(group_path):
        group_path = os.path.join(data_dir, "twompp_groups.txt")
    groups = _load_groups(group_path)
    group_gid = groups["gid"].astype(int)
    group_l = groups["l_deg"]
    group_b = groups["b_deg"]
    group_v = groups["vcmb"]
    group_id_to_idx = {int(gid): i for i, gid in enumerate(group_gid)}

    # Load galaxies with magnitudes
    galaxy_candidates = ["2m++.txt", "twompp.txt", "2m++_galaxies.txt"]
    galaxy_path = None
    for name in galaxy_candidates:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            galaxy_path = candidate
            break
    if galaxy_path is None:
        raise FileNotFoundError(f"No galaxy catalogue found in {data_dir}")

    galaxies = _load_galaxies_with_mag(galaxy_path)
    l_deg = galaxies["l_deg"].astype(np.float64)
    b_deg = galaxies["b_deg"].astype(np.float64)
    K2Mpp = galaxies["K2Mpp"].astype(np.float64)
    vcmb = galaxies["vcmb"].astype(np.float64)
    gid = galaxies["gid"].astype(int)

    # Filter bad values
    valid = np.isfinite(l_deg) & np.isfinite(b_deg) & np.isfinite(vcmb) & np.isfinite(K2Mpp)
    valid &= vcmb > 0.0

    l_deg = l_deg[valid]
    b_deg = b_deg[valid]
    K2Mpp = K2Mpp[valid]
    vcmb = vcmb[valid]
    gid = gid[valid]

    n_gal = len(vcmb)
    if verbose:
        print(f"Loaded {n_gal} galaxies after filtering")

    # ZoA Cloning: Fill Zone of Avoidance by cloning nearby galaxies
    # Per Carrick2015 §2.1:
    # - For |ℓ| > 30°: mask |b| < 5°, clone from 0° < |b| < 5° strips
    # - For |ℓ| < 30° (Galactic center): mask |b| < 10°, clone from 10° < |b| < 15° strips
    if clone_zoa:
        # Normalize longitude to [-180, 180]
        l_norm = np.mod(l_deg + 180, 360) - 180

        # Clone for |ℓ| > 30° region (narrow ZoA)
        near_gc = np.abs(l_norm) < 30  # Near Galactic center

        # Source strips: 0 < |b| < 5 for |ℓ| > 30
        source_narrow_n = (~near_gc) & (b_deg > 0) & (b_deg < 5)  # North strip
        source_narrow_s = (~near_gc) & (b_deg < 0) & (b_deg > -5)  # South strip

        # Source strips: 10 < |b| < 15 for |ℓ| < 30
        source_wide_n = near_gc & (b_deg > 10) & (b_deg < 15)  # North strip
        source_wide_s = near_gc & (b_deg < -10) & (b_deg > -15)  # South strip

        # Clone north strips to negative b (mirror)
        clones_l = []
        clones_b = []
        clones_K = []
        clones_v = []
        clones_gid = []

        # Narrow ZoA clones
        for src, target_b_sign in [(source_narrow_n, -1), (source_narrow_s, 1)]:
            if np.any(src):
                clones_l.append(l_deg[src])
                clones_b.append(target_b_sign * np.abs(b_deg[src]))  # Mirror to opposite side
                clones_K.append(K2Mpp[src])
                clones_v.append(vcmb[src])
                clones_gid.append(gid[src])

        # Wide ZoA clones (near GC)
        for src, target_b_offset in [(source_wide_n, -15), (source_wide_s, 15)]:
            if np.any(src):
                clones_l.append(l_deg[src])
                # Clone 10-15° strip to 0-5° in masked region (shift by ~12.5°)
                clones_b.append(np.sign(target_b_offset) * (np.abs(b_deg[src]) - 10))
                clones_K.append(K2Mpp[src])
                clones_v.append(vcmb[src])
                clones_gid.append(gid[src])

        if clones_l:
            n_clones = sum(len(c) for c in clones_l)
            l_deg = np.concatenate([l_deg] + clones_l)
            b_deg = np.concatenate([b_deg] + clones_b)
            K2Mpp = np.concatenate([K2Mpp] + clones_K)
            vcmb = np.concatenate([vcmb] + clones_v)
            gid = np.concatenate([gid] + clones_gid)
            if verbose:
                print(f"ZoA cloning: added {n_clones} cloned galaxies")

    n_gal = len(vcmb)

    # FoF Grouping: Collapse group members to common redshift (distance only)
    # Per Carrick2015 §2.5: "Objects were first grouped using the FoF algorithm,
    # and then placed at the mean of their group redshift distance to suppress
    # the Fingers-of-God effect."
    # NOTE: Only vcmb (distance) is collapsed - angular positions (l, b) are preserved.
    # This matches the behavior in aquila catalogues where group members have
    # identical r but different l, b.
    # Two options:
    #   sum_group_luminosities=True: Keep all members, sum luminosities at common distance
    #   sum_group_luminosities=False: Keep first member only (original behavior)
    is_grouped = gid >= 0
    n_in_groups = 0

    # Assign group redshift to grouped galaxies (keep original l, b)
    for i in range(n_gal):
        if is_grouped[i] and gid[i] in group_id_to_idx:
            idx = group_id_to_idx[gid[i]]
            # Only collapse distance (vcmb), preserve angular positions (l_deg, b_deg)
            vcmb[i] = group_v[idx]
            n_in_groups += 1

    if not sum_group_luminosities:
        # Original behavior: keep only first member per group
        seen_groups = set()
        keep = np.ones(n_gal, dtype=bool)
        for i in range(n_gal):
            if is_grouped[i] and gid[i] >= 0:
                if gid[i] in seen_groups:
                    keep[i] = False
                else:
                    seen_groups.add(gid[i])
        l_deg = l_deg[keep]
        b_deg = b_deg[keep]
        K2Mpp = K2Mpp[keep]
        vcmb = vcmb[keep]
        gid = gid[keep]

    # Re-filter: some groups have negative vcmb
    valid_after_groups = vcmb > 0.0
    l_deg = l_deg[valid_after_groups]
    b_deg = b_deg[valid_after_groups]
    K2Mpp = K2Mpp[valid_after_groups]
    vcmb = vcmb[valid_after_groups]
    gid = gid[valid_after_groups]

    n_gal = len(vcmb)
    n_unique_groups = len(set(gid[gid >= 0]))
    n_field = np.sum(gid < 0)
    if verbose:
        mode = "summing luminosities" if sum_group_luminosities else "first member only"
        print(f"FoF grouping ({mode}): {n_in_groups} in {n_unique_groups} groups + {n_field} field = {n_gal} tracers")

    # Load angular completeness maps
    map11 = hp.read_map(os.path.join(data_dir, "incompleteness_11_5.fits"), verbose=False)
    map12 = hp.read_map(os.path.join(data_dir, "incompleteness_12_5.fits"), verbose=False)
    m_lim, comp = _sample_completeness(l_deg, b_deg, map11, map12)

    # Angular completeness weight
    # Handle NaN and zero completeness values
    comp_safe = np.where(np.isfinite(comp) & (comp > 0), comp, cmin)
    w_ang = 1.0 / np.clip(comp_safe, cmin, 1.0)

    # Unit direction vectors in Galactic Cartesian (to match Carrick field coordinates)
    rhat = galactic_to_galactic_cartesian(l_deg, b_deg).astype(np.float64)

    # Observed redshifts
    z_obs = vcmb / C_LIGHT

    # Initial distances in h^-1 Mpc from Hubble flow with q0 correction
    # Carrick Eq. (3): H0*R = c*(z - (1+q0)/2 * z^2)
    z_initial = vcmb / C_LIGHT
    r_mpc = (C_LIGHT / H0) * (z_initial - (1.0 + q0) / 2.0 * z_initial**2)

    # For anisotropic H0, modify initial distances
    if aniso_h0:
        H_per_gal = compute_anisotropic_h0(rhat, H0 / 100.0, dipole_ell, dipole_b, dipole_mag)
        r_mpc = (C_LIGHT / H_per_gal) * (z_initial - (1.0 + q0) / 2.0 * z_initial**2)

    # Rolling buffer for distance averaging (suppress oscillations in triple-valued regions)
    distance_history = deque(maxlen=n_avg)

    # Beta values for iteration (adiabatically increase from 0 to β*)
    beta_values = np.linspace(0.0, BETA_STAR, n_iterations)

    # Storage for final fields
    delta_final = None
    velocity_final = None
    dx_mpc = None

    # Storage for intermediate fields (if requested)
    intermediate_deltas = {}
    if save_intermediate_iters is None:
        save_intermediate_iters = []

    t0 = time.time()

    for i_iter, beta in enumerate(beta_values):
        # Compute luminosity weights with current distances (in h^-1 Mpc)
        w_L = compute_luminosity_weight(r_mpc, m_lim, alpha, M_star, M_min)

        # Compute galaxy luminosities L/L* from apparent magnitude and distance
        # M_i = K2Mpp_i - distance_modulus = K2Mpp_i - 5*log10(r) - 25
        # L/L* = 10^(-0.4 * (M - M*))
        r_safe = np.maximum(r_mpc, 0.1)
        distance_modulus = 5.0 * np.log10(r_safe) + 25.0
        M_abs = K2Mpp - distance_modulus
        L_over_Lstar = np.power(10.0, -0.4 * (M_abs - M_star))

        # Total weight: w_ang * w_L * L/L* (luminosity-weighted density)
        # This deposits weighted luminosity, not just weighted counts
        w_total = w_ang * w_L * L_over_Lstar

        # Zero out 2MRS galaxies beyond 125 h^-1 Mpc
        # Carrick: "Galaxies from 2MRS with distances > 125 h^-1 Mpc are assigned weight zero"
        # 2MRS galaxies are those with K < 11.75 (the 2MRS survey limit)
        is_2mrs = K2Mpp < 11.75
        beyond_cutoff = r_mpc > r_2mrs_cutoff
        w_total[is_2mrs & beyond_cutoff] = 0.0

        # Also zero out galaxies beyond box
        w_total[r_mpc > Rmax_mpc] = 0.0

        # Positions in h^-1 Mpc (real space)
        positions_mpc = r_mpc[:, None] * rhat

        # CIC deposition in h^-1 Mpc (deposits weighted luminosity)
        rho, dx_mpc = _cic_deposit(
            _as_float32(positions_mpc),
            _as_float32(w_total),
            N,
            Rmax_mpc,
        )

        # Create mask for valid region (inside Rmax sphere)
        coords_1d = np.linspace(-Rmax_mpc, Rmax_mpc, N)
        xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        rr_grid = np.sqrt(xx**2 + yy**2 + zz**2)
        valid_mask = rr_grid <= Rmax_mpc

        # Create mask for non-zeroed regions
        # Zeroed regions = 2MRS-only angular directions (m_lim=11.5) AND r > 125 Mpc/h
        # These have no galaxy data due to the 2MRS cutoff
        if i_iter == 0:
            # Build 3D mask for non-zeroed regions (only need to do once)
            # For each voxel, check if its angular direction is 2MRS-only
            nside_coverage = hp.get_nside(map12)

            # Normalize grid positions to unit vectors
            rr_safe = np.maximum(rr_grid, 1e-10)
            ux = xx / rr_safe
            uy = yy / rr_safe
            uz = zz / rr_safe

            # Convert to theta, phi for HEALPix
            theta_grid = np.arccos(np.clip(uz, -1, 1))  # z = cos(theta)
            phi_grid = np.arctan2(uy, ux)  # in range [-pi, pi]
            phi_grid = np.where(phi_grid < 0, phi_grid + 2*np.pi, phi_grid)  # to [0, 2pi]

            # Get HEALPix pixel for each voxel
            pix_grid = hp.ang2pix(nside_coverage, theta_grid.ravel(), phi_grid.ravel())

            # Check if each voxel is in 2MRS-only region (map12 <= 0)
            is_2mrs_only_angular = (map12[pix_grid] <= 0).reshape(rr_grid.shape)

            # Zeroed regions: 2MRS-only AND r > 125
            zeroed_mask = is_2mrs_only_angular & (rr_grid > r_2mrs_cutoff)

            if verbose:
                n_zeroed = np.sum(zeroed_mask & valid_mask)
                n_valid = np.sum(valid_mask)
                print(f"  Non-zeroed region: {100*(1 - n_zeroed/n_valid):.1f}% of valid voxels")

        # Carrick2015 order (per Carrick's clarification):
        # 1. Mask regions: r > 200, or (2MRS-only AND r > 125)
        # 2. Compute rho_bar from non-masked regions only
        # 3. Compute delta = rho/rho_bar - 1, set delta = 0 for masked regions
        # 4. Apply bias scaling (m_lim=11.5 for 2MRS, 12.5 for deep) - should not change mean
        # 5. Smooth with Gaussian

        # Non-masked region: valid AND not zeroed
        nonmasked = valid_mask & ~zeroed_mask

        if use_global_mean:
            # Normalize using non-masked region only (Carrick's procedure)
            rho_mean = np.mean(rho[nonmasked])
            rho_mean = max(rho_mean, 1e-10)
            delta_g = rho / rho_mean - 1.0
            # Set masked regions to delta = 0
            delta_g = np.where(nonmasked, delta_g, 0.0)
        else:
            # Radial profile normalization (empirically better for LOS comparison)
            n_rbins = 40
            r_edges = np.linspace(0, Rmax_mpc, n_rbins + 1)
            r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

            rho_radial = np.zeros(n_rbins)
            for j in range(n_rbins):
                shell_mask = (rr_grid >= r_edges[j]) & (rr_grid < r_edges[j + 1])
                if np.any(shell_mask):
                    rho_radial[j] = np.mean(rho[shell_mask])

            # Smooth the radial profile to reduce noise
            from scipy.ndimage import gaussian_filter1d
            rho_radial_smooth = gaussian_filter1d(rho_radial, sigma=2.0)
            rho_radial_smooth = np.maximum(rho_radial_smooth, 1e-10)

            # Map radial profile to 3D grid
            rho_expected_3d = np.interp(
                rr_grid.ravel(), r_centers, rho_radial_smooth
            ).reshape(rr_grid.shape)
            rho_expected_3d = np.maximum(rho_expected_3d, 1e-10)

            delta_g = rho / rho_expected_3d - 1.0

        # Step 2: Apply bias normalization (Carrick Eq. 8): delta_g* = delta_g / psi(r)
        # psi(r) is computed using the appropriate magnitude limit for each region:
        #   - m_lim = 11.5 for 2MRS-only regions
        #   - m_lim = 12.5 for deep (6dF/SDSS) regions (if PSI_USE_UNIFORM_MLIM=False)
        # psi(r) = 0.73 + 0.24 * <L/L*>_L
        # where <L/L*>_L = Gamma(alpha+3, x_min) / Gamma(alpha+2, x_min)
        r_grid_flat = rr_grid.ravel()
        # Create m_lim array based on PSI_USE_UNIFORM_MLIM flag
        if PSI_USE_UNIFORM_MLIM:
            m_lim_3d = np.full(rr_grid.shape, PSI_MLIM_UNIVERSAL)
        else:
            m_lim_3d = np.where(is_2mrs_only_angular, 11.5, 12.5)
        psi_3d = compute_bias_normalization(
            r_grid_flat,
            m_lim_3d.ravel(),
            alpha, M_star, b_const, b_slope
        ).reshape(rr_grid.shape)
        psi_3d = np.maximum(psi_3d, 0.1)

        # Plot psi(r) on first iteration only
        if plot_psi and i_iter == 0:
            import matplotlib.pyplot as plt
            r_plot = np.linspace(1, Rmax_mpc, 200)
            psi_plot = compute_bias_normalization(
                r_plot,
                np.full_like(r_plot, PSI_MLIM_UNIVERSAL),
                alpha, M_star, b_const, b_slope
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(r_plot, psi_plot, 'b-', linewidth=2)
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('r [h⁻¹ Mpc]', fontsize=12)
            ax.set_ylabel('ψ(r)', fontsize=12)
            ax.set_title(f'Bias correction ψ(r) = {b_const} + {b_slope}×⟨L/L*⟩\n'
                        f'(m_lim={PSI_MLIM_UNIVERSAL}, α={alpha}, M*={M_star})')
            ax.set_xlim(0, Rmax_mpc)
            ax.grid(True, alpha=0.3)
            # Add key values as text
            psi_50 = compute_bias_normalization(np.array([50.0]), np.array([PSI_MLIM_UNIVERSAL]), alpha, M_star, b_const, b_slope)[0]
            psi_100 = compute_bias_normalization(np.array([100.0]), np.array([PSI_MLIM_UNIVERSAL]), alpha, M_star, b_const, b_slope)[0]
            psi_200 = compute_bias_normalization(np.array([200.0]), np.array([PSI_MLIM_UNIVERSAL]), alpha, M_star, b_const, b_slope)[0]
            ax.text(0.95, 0.05, f'ψ(50)={psi_50:.3f}\nψ(100)={psi_100:.3f}\nψ(200)={psi_200:.3f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            plt.savefig('psi_r_diagnostic.png', dpi=150)
            plt.close()
            print(f"Saved psi(r) plot to psi_r_diagnostic.png")

        # Apply bias normalization: delta_g* = delta_g / psi
        # Per Carrick: this should not change the mean of delta
        # (The mean is already 0 for non-masked regions by construction)
        delta_g_star = delta_g / psi_3d
        # Keep masked regions at 0
        delta_g_star = np.where(nonmasked, delta_g_star, 0.0)

        # Step 3: Gaussian smoothing AFTER bias correction (Carrick's order)
        delta_smooth = np.asarray(_gaussian_smooth_fft(
            _as_float32(delta_g_star),
            sigma_mpc,
            box_side_mpc,
        ))

        delta_final = delta_smooth.astype(np.float32)

        # Save intermediate field if requested
        if i_iter in save_intermediate_iters:
            intermediate_deltas[i_iter] = delta_final.copy()
            if verbose:
                print(f"  Saved delta field at iteration {i_iter}")

        if beta > 0:
            # Compute velocity field from smoothed, bias-corrected delta
            # This follows Carrick Eq. (7) using δ_g* smoothed
            # Beta is increased adiabatically from 0 to 1 during reconstruction
            velocity = velocity_from_density_fft(
                delta_final,  # Use smoothed field
                beta,
                box_side_mpc,
                zero_pad=True,
            )
            velocity_final = velocity

            # Interpolate velocity at galaxy positions
            v_at_gal = trilinear_interp_velocity(velocity, positions_mpc, box_side_mpc)

            # Line-of-sight velocity
            v_los = np.sum(v_at_gal * rhat, axis=1)

            # Update distances using RSD correction
            if aniso_h0:
                H_per_gal = compute_anisotropic_h0(rhat, H0 / 100.0, dipole_ell, dipole_b, dipole_mag)
                z_cos = (1.0 + z_obs) / (1.0 + v_los / C_LIGHT) - 1.0
                z_cos = np.maximum(z_cos, 1e-8)
                r_new = (C_LIGHT / H_per_gal) * (z_cos - (1.0 + q0) / 2.0 * z_cos**2)
            else:
                r_new = update_distances(z_obs, v_los, H0, q0)

            # Average previous distance estimates to suppress oscillations
            distance_history.append(r_new.copy())
            if len(distance_history) > 1:
                r_mpc = np.mean(np.array(distance_history), axis=0)
            else:
                r_mpc = r_new

        if verbose and (i_iter % 10 == 0 or i_iter == n_iterations - 1):
            elapsed = time.time() - t0
            print(f"  Iteration {i_iter}/{n_iterations-1}, beta={beta:.2f}, elapsed={elapsed:.1f}s")

    total_time = time.time() - t0
    if verbose:
        print(f"Reconstruction complete in {total_time:.1f}s")

    # Metadata
    if dx_mpc is None:
        dx_mpc = box_side_mpc / N
    metadata = {
        "box_side_mpc": float(box_side_mpc),
        "voxel_size_mpc": float(dx_mpc),
        "Rmax_mpc": float(Rmax_mpc),
        "N": int(N),
        "sigma_mpc": float(sigma_mpc),
        "n_iterations": int(n_iterations),
        "n_avg": int(n_avg),
        "alpha": float(alpha),
        "M_star": float(M_star),
        "M_min": float(M_min),
        "H0": float(H0),
        "q0": float(q0),
        "r_2mrs_cutoff": float(r_2mrs_cutoff),
        "cmin": float(cmin),
        "aniso_h0": bool(aniso_h0),
        "dipole_ell": float(dipole_ell) if aniso_h0 else None,
        "dipole_b": float(dipole_b) if aniso_h0 else None,
        "dipole_mag": float(dipole_mag) if aniso_h0 else None,
        "coord_convention": "Galactic Cartesian: x=cos(b)cos(l), y=cos(b)sin(l), z=sin(b)",
        "observer_position": "center of box",
        "ambiguities": [
            "Grid resolution: 256^3 (Carrick uses 257^3)",
            "FoF linking: pre-computed groups from catalogue",
            "Bias timing: after delta_g, before smoothing",
            "Distance averaging: last 5 iterations",
            "2MRS cutoff: m_lim < 12.0 AND r > 125",
            "FFT zero-padding: 2x for wrap-around avoidance",
        ],
    }

    return {
        "delta": np.asarray(delta_final, dtype=np.float32),
        "velocity": np.asarray(velocity_final, dtype=np.float32) if velocity_final is not None else None,
        "r_mpc_final": r_mpc.astype(np.float32),
        "intermediate_deltas": intermediate_deltas,
        "metadata": metadata,
    }


def compare_with_carrick(
    delta_ours: np.ndarray,
    carrick_path: str,
    rmax_compare: float = 200.0,
) -> Dict[str, float]:
    """
    Compare our reconstructed field with the reference Carrick field.

    Parameters
    ----------
    delta_ours : array
        Our overdensity field (N, N, N)
    carrick_path : str
        Path to Carrick density field .npy file
    rmax_compare : float
        Maximum radius for comparison in h^-1 Mpc

    Returns
    -------
    stats : dict
        Comparison statistics
    """
    # Load Carrick field (stored as overdensity delta, so density = 1 + delta)
    carrick_delta = np.load(carrick_path)

    N_ours = delta_ours.shape[0]
    N_carrick = carrick_delta.shape[0]

    # Our box is 400 h^-1 Mpc, Carrick is also 400 h^-1 Mpc
    box_side = 400.0

    # Create radial mask
    coords_ours = np.linspace(-box_side/2, box_side/2, N_ours)
    coords_carrick = np.linspace(-box_side/2, box_side/2, N_carrick)

    xx_o, yy_o, zz_o = np.meshgrid(coords_ours, coords_ours, coords_ours, indexing='ij')
    rr_ours = np.sqrt(xx_o**2 + yy_o**2 + zz_o**2)
    mask_ours = rr_ours < rmax_compare

    xx_c, yy_c, zz_c = np.meshgrid(coords_carrick, coords_carrick, coords_carrick, indexing='ij')
    rr_carrick = np.sqrt(xx_c**2 + yy_c**2 + zz_c**2)
    mask_carrick = rr_carrick < rmax_compare

    # Interpolate our field to Carrick grid for comparison
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (coords_ours, coords_ours, coords_ours),
        delta_ours,
        bounds_error=False,
        fill_value=0.0,
    )

    points = np.stack([xx_c.ravel(), yy_c.ravel(), zz_c.ravel()], axis=-1)
    delta_ours_interp = interp(points).reshape(N_carrick, N_carrick, N_carrick)

    # Compute log-space comparison
    log_ours = np.log10(np.clip(1.0 + delta_ours_interp[mask_carrick], 1e-5, None))
    log_carrick = np.log10(np.clip(1.0 + carrick_delta[mask_carrick], 1e-5, None))

    mean_abs_diff = np.mean(np.abs(log_ours - log_carrick))
    rms_diff = np.sqrt(np.mean((log_ours - log_carrick)**2))
    correlation = np.corrcoef(log_ours, log_carrick)[0, 1]

    return {
        "mean_abs_log_diff": float(mean_abs_diff),
        "rms_log_diff": float(rms_diff),
        "correlation": float(correlation),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Carrick2015 density/velocity field reconstruction."
    )
    parser.add_argument("--data-dir", default="data/2M++",
                        help="Path to 2M++ data directory")
    parser.add_argument("--output", default="carrick_field_reproduced.npz",
                        help="Output file path")
    parser.add_argument("--N", type=int, default=256,
                        help="Grid resolution")
    parser.add_argument("--box-side", type=float, default=400.0,
                        help="Box side length in h^-1 Mpc")
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Gaussian smoothing scale in h^-1 Mpc")
    parser.add_argument("--n-iterations", type=int, default=101,
                        help="Number of beta iterations")
    parser.add_argument("--n-avg", type=int, default=5,
                        help="Number of previous iterations to average")
    parser.add_argument("--r-2mrs-cutoff", type=float, default=125.0,
                        help="2MRS distance cutoff in h^-1 Mpc")
    parser.add_argument("--cmin", type=float, default=0.5,
                        help="Minimum angular completeness")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                        help="Schechter LF alpha")
    parser.add_argument("--M-star", type=float, default=MSTAR_DEFAULT,
                        help="Schechter LF M*")
    parser.add_argument("--aniso-h0", action="store_true",
                        help="Enable anisotropic H0 model")
    parser.add_argument("--compare-carrick", type=str, default=None,
                        help="Path to Carrick reference field for comparison")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    result = build_carrick_field(
        data_dir=args.data_dir,
        N=args.N,
        box_side_mpc=args.box_side,
        sigma_mpc=args.sigma,
        n_iterations=args.n_iterations,
        n_avg=args.n_avg,
        r_2mrs_cutoff=args.r_2mrs_cutoff,
        cmin=args.cmin,
        alpha=args.alpha,
        M_star=args.M_star,
        aniso_h0=args.aniso_h0,
        verbose=not args.quiet,
    )

    # Save results
    np.savez_compressed(
        args.output,
        delta=result["delta"],
        velocity=result["velocity"],
        r_mpc_final=result["r_mpc_final"],
        metadata=np.array(result["metadata"], dtype=object),
    )
    print(f"Saved output to {args.output}")

    # Compare with reference if provided
    if args.compare_carrick is not None:
        print("\nComparing with reference Carrick field...")
        stats = compare_with_carrick(result["delta"], args.compare_carrick)
        print(f"  Mean |log10(1+delta)| difference: {stats['mean_abs_log_diff']:.4f}")
        print(f"  RMS log difference: {stats['rms_log_diff']:.4f}")
        print(f"  Correlation: {stats['correlation']:.4f}")


if __name__ == "__main__":
    main()
