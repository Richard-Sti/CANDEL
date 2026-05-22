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
Utilities to interpolate 3D density and velocity fields along galaxy lines of
sight.
"""
import os
from warnings import warn

import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator

from ..util import (fprint, radec_to_cartesian, radec_to_galactic,
                    radec_to_supergalactic)

_NUMPY_GAUSSIAN_FALLBACK_MEMORY_FRACTION = 0.5
_PYLIANS_FALLBACK_NOTICE_PRINTED = False


@njit(cache=False)
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


def _validate_gaussian_smoothing_scale(field, smooth_scale, boxsize):
    """Validate a resolved Gaussian kernel scale and return grid metadata."""
    if field.ndim != 3:
        raise ValueError("`field` must be cubic with shape (N, N, N).")
    N = field.shape[0]
    if field.shape[1] != N or field.shape[2] != N:
        raise ValueError("`field` must be cubic with shape (N, N, N).")
    if not np.isfinite(smooth_scale) or smooth_scale <= 0:
        raise ValueError("`smooth_scale` must be finite and positive.")
    if not np.isfinite(boxsize) or boxsize <= 0:
        raise ValueError("`boxsize` must be finite and positive.")
    dx = float(boxsize) / N
    if smooth_scale <= dx:
        raise ValueError(
            f"Gaussian smoothing scale {smooth_scale:g} Mpc/h must exceed "
            f"the grid voxel size {dx:g} Mpc/h. Smoothing at or below the "
            "grid scale is not well defined for these field products.")
    return N, dx


def _discrete_periodic_gaussian_filter_fft(N, smooth_scale, boxsize):
    """Return the DFT of a periodic sampled Gaussian kernel."""
    R_grid = smooth_scale * N / boxsize
    axis = np.arange(N, dtype=np.float32)
    middle = N // 2
    axis[axis > middle] -= N
    r2 = (axis[:, None, None]**2
          + axis[None, :, None]**2
          + axis[None, None, :]**2)
    kernel = np.exp(-r2 / (2.0 * R_grid**2)).astype(np.float32)
    kernel /= np.sum(kernel, dtype=np.float64)
    return np.fft.rfftn(kernel, axes=(0, 1, 2)).astype(
        np.complex64, copy=False)


def _format_memory(num_bytes):
    """Return a compact binary-size string."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0


def _available_memory_bytes():
    """Best-effort estimate of currently available system memory."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return int(pages) * int(page_size)


def _numpy_gaussian_fallback_memory_bytes(N, dtype, make_copy):
    """Estimate peak temporary memory for the NumPy smoothing fallback."""
    n_real = int(N)**3
    n_fourier = int(N) * int(N) * (int(N) // 2 + 1)
    float32_bytes = np.dtype(np.float32).itemsize
    float64_bytes = np.dtype(np.float64).itemsize
    complex64_bytes = np.dtype(np.complex64).itemsize
    complex128_bytes = np.dtype(np.complex128).itemsize
    dtype_bytes = np.dtype(dtype).itemsize

    copy_bytes = n_real * dtype_bytes if make_copy else 0
    kernel_stage = (
        copy_bytes
        + 2 * n_real * float32_bytes
        + n_fourier * (complex128_bytes + complex64_bytes)
    )
    smoothing_stage = (
        copy_bytes
        + n_fourier * (2 * complex128_bytes + complex64_bytes)
        + n_real * (float64_bytes + dtype_bytes)
    )
    return int(1.25 * max(kernel_stage, smoothing_stage))


def _check_numpy_gaussian_fallback_memory(N, dtype, make_copy):
    """Raise if the NumPy fallback is likely to exceed memory headroom."""
    estimated = _numpy_gaussian_fallback_memory_bytes(N, dtype, make_copy)
    available = _available_memory_bytes()
    if available is None:
        return estimated, available

    budget = int(_NUMPY_GAUSSIAN_FALLBACK_MEMORY_FRACTION * available)
    if estimated > budget:
        raise RuntimeError(
            "Pylians `smoothing_library` is required for this Gaussian "
            "smoothing run. The NumPy fallback estimates "
            f"{_format_memory(estimated)} of temporary memory for a "
            f"{N}^3 grid, exceeding the available-memory headroom "
            f"({_format_memory(budget)} of currently available "
            f"{_format_memory(available)}).")
    return estimated, available


def _print_numpy_gaussian_fallback_notice(N, estimated, available):
    """Print a one-time notice when the Pylians smoothing path is absent."""
    global _PYLIANS_FALLBACK_NOTICE_PRINTED
    if _PYLIANS_FALLBACK_NOTICE_PRINTED:
        return
    if available is None:
        memory = f"estimated temporary memory {_format_memory(estimated)}"
    else:
        memory = (
            f"estimated temporary memory {_format_memory(estimated)}, "
            f"available {_format_memory(available)}")
    fprint(
        "Pylians `smoothing_library` is not available; using the NumPy "
        f"Gaussian smoothing fallback for a {N}^3 grid ({memory}).")
    _PYLIANS_FALLBACK_NOTICE_PRINTED = True


def apply_gaussian_smoothing(field, smooth_scale, boxsize, make_copy=False):
    """
    Apply periodic Gaussian smoothing to a 3D field using FFTs.

    The kernel follows Pylians' discrete-grid convention: sample the Gaussian
    on the periodic real-space grid, normalise it, Fourier transform that
    sampled kernel, then multiply the field FFT by it. Units of
    `smooth_scale` must match that of `boxsize`.
    """
    N, _ = _validate_gaussian_smoothing_scale(
        field, smooth_scale, boxsize)

    try:
        import smoothing_library as SL
        x = np.ascontiguousarray(field, dtype=np.float32)
        if make_copy:
            x = x.copy()
        W_k = SL.FT_filter(boxsize, smooth_scale, N, "Gaussian", 1)
        smoothed = SL.field_smoothing(x, W_k, 1)
        return smoothed.astype(field.dtype, copy=False)
    except ImportError:
        warn(
            "Optional `smoothing_library` (from Pylians3) not found; "
            "falling back to a NumPy implementation of the same sampled "
            "periodic Gaussian kernel. Install it with `pip install pylians` "
            "for the optimised path.",
            UserWarning, stacklevel=2)

    estimated, available = _check_numpy_gaussian_fallback_memory(
        N, field.dtype, make_copy)
    _print_numpy_gaussian_fallback_notice(N, estimated, available)
    x = field.copy() if make_copy else field
    Wk = _discrete_periodic_gaussian_filter_fft(N, smooth_scale, boxsize)
    fhat = np.fft.rfftn(x, axes=(0, 1, 2)) * Wk
    smoothed = np.fft.irfftn(fhat, s=(N, N, N), axes=(0, 1, 2))

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


def prepare_los_geometry(field_loader, r, RA, dec):
    """Compute LOS positions and unit vectors for interpolation."""
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
    n_r, n_gal = len(r), len(RA)
    pos = (field_loader.observer_pos[None, None, :]
           + r[:, None, None] * rhat[None, :, :]).astype(np.float32)
    pos_flat = pos.reshape(-1, 3)
    return pos_flat, rhat, n_r, n_gal


def _get_grid_params(field_loader, ngrid):
    """Return grid geometry from the loaded Cartesian field."""
    cellsize = np.float32(field_loader.boxsize / ngrid)
    grid_min = np.float32(0.5 * cellsize)
    return cellsize, grid_min


def _target_smoothing_to_kernel_scale(smooth_target, voxel_size):
    """Convert a target resolution to the extra Gaussian kernel scale."""
    if not np.isfinite(smooth_target) or smooth_target <= voxel_size:
        raise ValueError(
            f"Target smoothing scale {smooth_target} must be finite and "
            f"exceed the voxel size {voxel_size}.")
    smooth_scale = np.sqrt(smooth_target**2 - voxel_size**2)
    if smooth_scale <= voxel_size:
        min_target = np.sqrt(2.0) * voxel_size
        raise ValueError(
            f"Target smoothing scale {smooth_target:g} Mpc/h implies a "
            f"Gaussian kernel scale {smooth_scale:g} Mpc/h, which does not "
            f"exceed the voxel size {voxel_size:g} Mpc/h. Use target > "
            f"{min_target:g} Mpc/h.")
    return smooth_scale


def interpolate_los_density_velocity(field_loader, r, RA, dec,
                                     smooth_target=None, verbose=True,
                                     los_geometry=None,
                                     field_smoothing_scale=None,
                                     velocity_field_smoothing_scale=None):
    """
    Interpolate the density and velocity fields along the line of sight
    specified by `RA` and `dec` at radial steps `r` from the observer. The
    angular coordinates are expected in degrees, while `r` is in `Mpc / h`.

    Fields are loaded and interpolated one component at a time to limit
    peak memory usage (important for large grids like Manticore 1024^3).
    """
    if los_geometry is None:
        pos_flat, rhat, n_r, n_gal = prepare_los_geometry(
            field_loader, r, RA, dec)
    else:
        pos_flat, rhat, n_r, n_gal = los_geometry
    eps = np.float32(1e-4)
    fill_value = np.float32(np.log(1 + eps))

    # --- Density ---
    fprint("interpolating the density field...", verbose=verbose)
    density = field_loader.load_density().astype(np.float32, copy=False)
    ngrid = density.shape[0]
    cellsize, grid_min = _get_grid_params(field_loader, ngrid)

    rho_kernel_scale = None
    smoothing_inputs = [
        smooth_target is not None,
        field_smoothing_scale is not None,
    ]
    if sum(smoothing_inputs) > 1:
        raise ValueError(
            "`smooth_target` and "
            "`field_smoothing_scale` are mutually exclusive.")

    if smooth_target is not None:
        smooth_scale = _target_smoothing_to_kernel_scale(
            smooth_target, cellsize)
        rho_kernel_scale = smooth_scale
        fprint(f"applying Gaussian smoothing to density with scale "
               f"{smooth_scale:.1f} Mpc/h to match target "
               f"{smooth_target:.1f} Mpc/h.",
               verbose=verbose)
    elif field_smoothing_scale is not None:
        field_smoothing_scale = float(field_smoothing_scale)
        if (not np.isfinite(field_smoothing_scale)
                or field_smoothing_scale < 0):
            raise ValueError(
                "`field_smoothing_scale` must be finite and non-negative.")
        if field_smoothing_scale > 0:
            if field_smoothing_scale <= cellsize:
                raise ValueError(
                    "`field_smoothing_scale` must exceed the field voxel "
                    f"size {cellsize:g} Mpc/h; got "
                    f"{field_smoothing_scale:g} Mpc/h.")
            rho_kernel_scale = float(field_smoothing_scale)
            fprint(
                "applying Gaussian smoothing to density with scale "
                f"{rho_kernel_scale:.1f} Mpc/h.",
                verbose=verbose)

    velocity_smooth_scale = None
    if velocity_field_smoothing_scale is not None:
        velocity_field_smoothing_scale = float(velocity_field_smoothing_scale)
        if (not np.isfinite(velocity_field_smoothing_scale)
                or velocity_field_smoothing_scale < 0):
            raise ValueError(
                "`velocity_field_smoothing_scale` must be finite and "
                "non-negative.")
        if velocity_field_smoothing_scale > 0:
            if velocity_field_smoothing_scale <= cellsize:
                raise ValueError(
                    "`velocity_field_smoothing_scale` must exceed the field "
                    f"voxel size {cellsize:g} Mpc/h; got "
                    f"{velocity_field_smoothing_scale:g} Mpc/h.")
            velocity_smooth_scale = float(velocity_field_smoothing_scale)
            fprint(
                "applying Gaussian smoothing to velocity with scale "
                f"{velocity_smooth_scale:.1f} Mpc/h.",
                verbose=verbose)

    if rho_kernel_scale is not None:
        density = apply_gaussian_smoothing(
            density, rho_kernel_scale, field_loader.boxsize,
            make_copy=True)

    np.add(density, eps, out=density)
    np.log(density, out=density)
    los_density_flat = _trilinear_interp_field(
        density.ravel(), pos_flat, grid_min, cellsize, ngrid, fill_value)
    del density

    los_density = los_density_flat.reshape(n_r, n_gal)
    del los_density_flat
    los_density = np.exp(los_density) - eps
    los_density = np.clip(los_density, eps, None)
    assert np.all(los_density > 0)

    # --- Velocity (one component at a time) ---
    los_velocity = np.zeros((n_r, n_gal), dtype=np.float32)
    can_load_component = hasattr(field_loader, 'load_velocity_component')

    if can_load_component:
        for comp in range(3):
            fprint(f"interpolating velocity component {comp}...",
                   verbose=verbose)
            v_comp = field_loader.load_velocity_component(comp)
            if velocity_smooth_scale is not None:
                v_comp = apply_gaussian_smoothing(
                    v_comp, velocity_smooth_scale, field_loader.boxsize,
                    make_copy=True)
            v_flat = np.ascontiguousarray(v_comp, dtype=np.float32).ravel()
            del v_comp
            los_v_comp = _trilinear_interp_field(
                v_flat, pos_flat, grid_min, cellsize, ngrid, np.float32(0.0))
            del v_flat
            los_v_comp = los_v_comp.reshape(n_r, n_gal)
            los_v_comp *= rhat[None, :, comp]
            los_velocity += los_v_comp
            del los_v_comp
        if hasattr(field_loader, 'clear_velocity_cache'):
            field_loader.clear_velocity_cache()
    else:
        fprint("interpolating the velocity field...", verbose=verbose)
        velocity = field_loader.load_velocity()
        if velocity_smooth_scale is not None:
            for i in range(3):
                velocity[i] = apply_gaussian_smoothing(
                    velocity[i], velocity_smooth_scale, field_loader.boxsize,
                    make_copy=True)
        for comp in range(3):
            v_flat = np.ascontiguousarray(
                velocity[comp], dtype=np.float32).ravel()
            los_v_comp = _trilinear_interp_field(
                v_flat, pos_flat, grid_min, cellsize, ngrid,
                np.float32(0.0))
            los_v_comp = los_v_comp.reshape(n_r, n_gal)
            los_v_comp *= rhat[None, :, comp]
            los_velocity += los_v_comp
            del los_v_comp
        del velocity

    assert np.all(np.isfinite(los_velocity))

    return los_density.T, los_velocity.T
