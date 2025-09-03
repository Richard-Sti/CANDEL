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
from scipy.interpolate import RegularGridInterpolator

from ..util import (fprint, radec_to_cartesian, radec_to_galactic,
                    radec_to_supergalactic)


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
        W_k = SL.FT_filter(boxsize, smooth_scale, N, "Gaussian")
        return SL.field_smoothing(field, W_k)
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
    pos = (field_loader.observer_pos[None, None, :]
           + r[:, None, None] * rhat[None, :, :]).astype(np.float32)
    pos_shape = pos.shape
    pos_flat = pos.reshape(-1, 3)

    # Interpolate density
    fprint("interpolating the density field...", verbose=verbose)
    eps = 1e-4
    density = np.log(field_loader.load_density() + eps)
    fill_value = np.log(1 + eps)

    try:
        voxel_size = field_loader.effective_resolution
    except AttributeError:
        voxel_size = field_loader.boxsize / density.shape[0]
    if smooth_target is not None:
        if smooth_target < voxel_size:
            raise ValueError(
                f"Target smoothing scale {smooth_target} is smaller than "
                f"than the voxel size {voxel_size}. Skipping smoothing.")

        smooth_scale = np.sqrt(smooth_target**2 - voxel_size**2)
        fprint(f"applying Gaussian smoothing with scale {smooth_scale:.1f} "
               f"Mpc/h to match target {smooth_target:.1f} Mpc/h.",
               verbose=verbose)
        density = apply_gaussian_smoothing(
            density, smooth_scale, field_loader.boxsize, make_copy=True)

    f_density = build_regular_interpolator(
        density, field_loader.boxsize, fill_value=fill_value)
    los_density = f_density(pos_flat).reshape(pos_shape[:2]).astype(np.float32)
    los_density = np.exp(los_density) - eps
    los_density = np.clip(los_density, eps, None)
    assert np.all(los_density > 0)

    # Interpolate velocity components one at a time
    fprint("interpolating the velocity field...", verbose=verbose)
    velocity = field_loader.load_velocity()  # shape (3, ngrid, ngrid, ngrid)
    v_interp = np.empty((pos_flat.shape[0], 3), dtype=np.float32)

    for i in range(3):
        if smooth_target is not None:
            velocity_i = apply_gaussian_smoothing(
                velocity[i], smooth_scale, field_loader.boxsize,
                make_copy=True)
        else:
            velocity_i = velocity[i]

        f_vel = build_regular_interpolator(velocity_i, field_loader.boxsize)
        v_interp[:, i] = f_vel(pos_flat)

    v_interp = v_interp.reshape(*pos_shape)  # (n_r, n_gal, 3)
    los_velocity = np.einsum('ijk,jk->ij', v_interp, rhat)
    assert np.all(np.isfinite(los_velocity))

    return los_density.T, los_velocity.T
