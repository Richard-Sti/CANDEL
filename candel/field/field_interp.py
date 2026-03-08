# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Scripts to load fields that have been previously interpolated at the line of
sight of galaxies and saved to HDF5 files, typically this is done with the
`csiborgtools` package.
"""
from warnings import warn

import numpy as np
from numba import njit, prange
from scipy.interpolate import RegularGridInterpolator

from ..util import (fprint, radec_to_cartesian, radec_to_galactic,
                    radec_to_supergalactic)


@njit(cache=True)
def _trilinear_interp_field(field_flat, pos_flat, grid_min, grid_step,
                            ngrid, fill_value):
    """Fused trilinear interpolation: index computation + weighted sum."""
    n = pos_flat.shape[0]
    s = np.intp(ngrid)
    s2 = s * s
    nmax = s - 2
    result = np.empty(n, dtype=np.float32)
    inv_step = np.float32(1.0 / grid_step)
    gmin = np.float32(grid_min)
    use_fill = not np.isnan(fill_value)

    for i in range(n):
        fx = (pos_flat[i, 0] - gmin) * inv_step
        fy = (pos_flat[i, 1] - gmin) * inv_step
        fz = (pos_flat[i, 2] - gmin) * inv_step

        ix0 = np.intp(np.floor(fx))
        iy0 = np.intp(np.floor(fy))
        iz0 = np.intp(np.floor(fz))

        oob = ix0 < 0 or iy0 < 0 or iz0 < 0 or \
            ix0 > nmax or iy0 > nmax or iz0 > nmax
        if oob and use_fill:
            result[i] = fill_value
            continue

        # Clamp for safety
        if ix0 < 0:
            ix0 = np.intp(0)
        elif ix0 > nmax:
            ix0 = nmax
        if iy0 < 0:
            iy0 = np.intp(0)
        elif iy0 > nmax:
            iy0 = nmax
        if iz0 < 0:
            iz0 = np.intp(0)
        elif iz0 > nmax:
            iz0 = nmax

        wx1 = np.float32(fx - ix0)
        wy1 = np.float32(fy - iy0)
        wz1 = np.float32(fz - iz0)
        wx0 = np.float32(1.0) - wx1
        wy0 = np.float32(1.0) - wy1
        wz0 = np.float32(1.0) - wz1

        ix1 = ix0 + np.intp(1)
        iy1 = iy0 + np.intp(1)
        iz1 = iz0 + np.intp(1)

        # Base flat index components
        bx0 = ix0 * s2
        bx1 = ix1 * s2
        by0 = iy0 * s
        by1 = iy1 * s

        result[i] = (wx0 * (wy0 * (wz0 * field_flat[bx0 + by0 + iz0]
                                    + wz1 * field_flat[bx0 + by0 + iz1])
                            + wy1 * (wz0 * field_flat[bx0 + by1 + iz0]
                                     + wz1 * field_flat[bx0 + by1 + iz1]))
                     + wx1 * (wy0 * (wz0 * field_flat[bx1 + by0 + iz0]
                                     + wz1 * field_flat[bx1 + by0 + iz1])
                              + wy1 * (wz0 * field_flat[bx1 + by1 + iz0]
                                       + wz1 * field_flat[bx1 + by1 + iz1])))
    return result


@njit(parallel=True, cache=True)
def _trilinear_interp_density_velocity(
        density_flat, vx_flat, vy_flat, vz_flat,
        pos_flat, rhat_rep, grid_min, grid_step, ngrid, fill_value):
    """Fused trilinear interpolation of density + 3 velocity components.

    `rhat_rep` has shape (n_flat, 3) — the unit vector for each query point,
    tiled to match pos_flat. The LOS velocity is computed inline as the dot
    product of interpolated velocity with rhat.
    """
    n = pos_flat.shape[0]
    s = np.intp(ngrid)
    s2 = s * s
    nmax = s - 2
    inv_step = np.float32(1.0 / grid_step)
    gmin = np.float32(grid_min)
    use_fill = not np.isnan(fill_value)

    out_density = np.empty(n, dtype=np.float32)
    out_los_vel = np.empty(n, dtype=np.float32)

    for i in prange(n):
        fx = (pos_flat[i, 0] - gmin) * inv_step
        fy = (pos_flat[i, 1] - gmin) * inv_step
        fz = (pos_flat[i, 2] - gmin) * inv_step

        ix0 = np.intp(np.floor(fx))
        iy0 = np.intp(np.floor(fy))
        iz0 = np.intp(np.floor(fz))

        oob = ix0 < 0 or iy0 < 0 or iz0 < 0 or \
            ix0 > nmax or iy0 > nmax or iz0 > nmax
        if oob and use_fill:
            out_density[i] = fill_value
            out_los_vel[i] = np.float32(0.0)
            continue

        if ix0 < 0:
            ix0 = np.intp(0)
        elif ix0 > nmax:
            ix0 = nmax
        if iy0 < 0:
            iy0 = np.intp(0)
        elif iy0 > nmax:
            iy0 = nmax
        if iz0 < 0:
            iz0 = np.intp(0)
        elif iz0 > nmax:
            iz0 = nmax

        wx1 = np.float32(fx - ix0)
        wy1 = np.float32(fy - iy0)
        wz1 = np.float32(fz - iz0)
        wx0 = np.float32(1.0) - wx1
        wy0 = np.float32(1.0) - wy1
        wz0 = np.float32(1.0) - wz1

        ix1 = ix0 + np.intp(1)
        iy1 = iy0 + np.intp(1)
        iz1 = iz0 + np.intp(1)

        bx0 = ix0 * s2
        bx1 = ix1 * s2
        by0 = iy0 * s
        by1 = iy1 * s

        # 8 corner weights
        w000 = wx0 * wy0 * wz0
        w100 = wx1 * wy0 * wz0
        w010 = wx0 * wy1 * wz0
        w110 = wx1 * wy1 * wz0
        w001 = wx0 * wy0 * wz1
        w101 = wx1 * wy0 * wz1
        w011 = wx0 * wy1 * wz1
        w111 = wx1 * wy1 * wz1

        # 8 flat indices
        i000 = bx0 + by0 + iz0
        i100 = bx1 + by0 + iz0
        i010 = bx0 + by1 + iz0
        i110 = bx1 + by1 + iz0
        i001 = bx0 + by0 + iz1
        i101 = bx1 + by0 + iz1
        i011 = bx0 + by1 + iz1
        i111 = bx1 + by1 + iz1

        out_density[i] = (w000 * density_flat[i000]
                          + w100 * density_flat[i100]
                          + w010 * density_flat[i010]
                          + w110 * density_flat[i110]
                          + w001 * density_flat[i001]
                          + w101 * density_flat[i101]
                          + w011 * density_flat[i011]
                          + w111 * density_flat[i111])

        vx_val = (w000 * vx_flat[i000] + w100 * vx_flat[i100]
                  + w010 * vx_flat[i010] + w110 * vx_flat[i110]
                  + w001 * vx_flat[i001] + w101 * vx_flat[i101]
                  + w011 * vx_flat[i011] + w111 * vx_flat[i111])
        vy_val = (w000 * vy_flat[i000] + w100 * vy_flat[i100]
                  + w010 * vy_flat[i010] + w110 * vy_flat[i110]
                  + w001 * vy_flat[i001] + w101 * vy_flat[i101]
                  + w011 * vy_flat[i011] + w111 * vy_flat[i111])
        vz_val = (w000 * vz_flat[i000] + w100 * vz_flat[i100]
                  + w010 * vz_flat[i010] + w110 * vz_flat[i110]
                  + w001 * vz_flat[i001] + w101 * vz_flat[i101]
                  + w011 * vz_flat[i011] + w111 * vz_flat[i111])

        out_los_vel[i] = (vx_val * rhat_rep[i, 0]
                          + vy_val * rhat_rep[i, 1]
                          + vz_val * rhat_rep[i, 2])

    return out_density, out_los_vel


def apply_gaussian_smoothing(field, smooth_scale, boxsize, make_copy=False):
    """
    Apply Gaussian smoothing to a 3D field using FFTs. Units of `smooth_scale`
    must match that of `boxsize`.
    """
    N = field.shape[0]
    if field.ndim != 3 or not (field.shape[1] == N and field.shape[2] == N):
        raise ValueError("`field` must be cubic with shape (N, N, N).")

    try:
        import smoothing_library as SL
        W_k = SL.FT_filter(boxsize, smooth_scale, N, "Gaussian", 1)
        return SL.field_smoothing(field, W_k, 1)
    except ImportError:
        warn("Pylians3 not found. Switching to NumPy FFT calculation.",
             UserWarning)

    x = field.copy() if make_copy else field
    dx = boxsize / N

    # Fourier-space grid
    kx = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    kz = np.fft.rfftfreq(N, d=dx) * 2.0 * np.pi
    kx3, ky3, kz3 = np.meshgrid(kx, kx, kz, indexing="ij")
    k2 = kx3**2 + ky3**2 + kz3**2

    # Gaussian filter in Fourier space
    Wk = np.exp(-0.5 * (smooth_scale**2) * k2)

    # FFT → filter → inverse FFT
    fhat = np.fft.rfftn(x)
    fhat *= Wk
    smoothed = np.fft.irfftn(fhat, s=(N, N, N))

    return smoothed.astype(field.dtype, copy=False)


def build_regular_interpolator(field, boxsize, fill_value=None):
    """A regular grid interpolator for a 3D field."""
    ngrid = field.shape[0]
    cellsize = boxsize / ngrid

    X = np.linspace(0.5 * cellsize, boxsize - 0.5 * cellsize, ngrid)
    Y, Z = X, X

    return RegularGridInterpolator(
        (X, Y, Z), field, fill_value=fill_value, bounds_error=False,
        method="linear",)


def interpolate_los_density_velocity(field_loader, r, RA, dec,
                                     smooth_target=None, verbose=True):
    """
    Interpolate the density and velocity fields along the line of sight
    specified by `RA` and `dec` at radial steps `r` from the observer. The
    former is expected in degrees, while the latter in `Mpc / h`.
    """
    if field_loader.coordinate_frame == "icrs":
        rhat = radec_to_cartesian(RA, dec)
    elif field_loader.coordinate_frame == "galactic":
        ell, b = radec_to_galactic(RA, dec)
        rhat = radec_to_cartesian(ell, b)
    elif field_loader.coordinate_frame == "supergalactic":
        ell, b = radec_to_supergalactic(RA, dec)
        rhat = radec_to_cartesian(ell, b)
    else:
        raise ValueError(
            f"Unknown coordinate frame: `{field_loader.coordinate_frame}`. "
            "Please add support for it.")

    rhat = rhat.astype(np.float32)

    # Precompute positions (n_r, n_gal, 3)
    n_r, n_gal = len(r), len(RA)
    pos = (field_loader.observer_pos[None, None, :]
           + r[:, None, None] * rhat[None, :, :]).astype(np.float32)
    pos_flat = pos.reshape(-1, 3)

    # Tile rhat to match pos_flat: (n_r, n_gal, 3) -> (n_r*n_gal, 3)
    rhat_rep = np.tile(rhat, (n_r, 1))

    # Load and prepare fields
    fprint("interpolating the density field...", verbose=verbose)
    eps = np.float32(1e-4)
    density = field_loader.load_density().astype(np.float32, copy=False)
    np.add(density, eps, out=density)
    np.log(density, out=density)
    fill_value = np.float32(np.log(1 + eps))

    ngrid = density.shape[0]
    cellsize = np.float32(field_loader.boxsize / ngrid)
    grid_min = np.float32(0.5 * cellsize)

    try:
        voxel_size = field_loader.effective_resolution
    except AttributeError:
        voxel_size = cellsize
    if smooth_target is not None:
        if smooth_target < voxel_size:
            raise ValueError(
                f"Target smoothing scale {smooth_target} is smaller than "
                f"the voxel size {voxel_size}.")

        smooth_scale = np.sqrt(smooth_target**2 - voxel_size**2)
        fprint(f"applying Gaussian smoothing with scale {smooth_scale:.1f} "
               f"Mpc/h to match target {smooth_target:.1f} Mpc/h.",
               verbose=verbose)
        density = apply_gaussian_smoothing(
            density, smooth_scale, field_loader.boxsize, make_copy=True)

    fprint("interpolating the velocity field...", verbose=verbose)
    velocity = field_loader.load_velocity()  # shape (3, ngrid, ngrid, ngrid)

    if smooth_target is not None:
        for i in range(3):
            velocity[i] = apply_gaussian_smoothing(
                velocity[i], smooth_scale, field_loader.boxsize,
                make_copy=True)

    # Ravel fields to 1D for flat indexing in the fused kernel.
    # density is already float32 & C-contiguous -> ravel is a view.
    density_flat = density.ravel()
    vx_flat = np.ascontiguousarray(velocity[0], dtype=np.float32).ravel()
    vy_flat = np.ascontiguousarray(velocity[1], dtype=np.float32).ravel()
    vz_flat = np.ascontiguousarray(velocity[2], dtype=np.float32).ravel()
    del velocity

    # Fused Numba kernel: density + velocity + LOS dot product in one pass
    los_density_flat, los_velocity_flat = _trilinear_interp_density_velocity(
        density_flat, vx_flat, vy_flat, vz_flat,
        pos_flat, rhat_rep, grid_min, cellsize, ngrid, fill_value)

    los_density = los_density_flat.reshape(n_r, n_gal)
    los_density = np.exp(los_density) - eps
    los_density = np.clip(los_density, eps, None)
    assert np.all(los_density > 0)

    los_velocity = los_velocity_flat.reshape(n_r, n_gal)
    assert np.all(np.isfinite(los_velocity))

    return los_density.T, los_velocity.T
