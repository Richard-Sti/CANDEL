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
Data loading and preprocessing utilities for peculiar-velocity catalogues.

Provides dataframe-like containers, LOS interpolation helpers, covariance
assembly, and catalogue I/O wired to the project config files.
"""
import os
import shutil
import tempfile
from itertools import chain
from os.path import join

import numpy as np
from h5py import File
from jax import numpy as jnp

from ..field.field_interp import apply_gaussian_smoothing
from ..field.loader import field_allows_raw_product_reads, name2field_loader
from ..util import fprint, get_nested
from .field_cache import (_ArrayShapeOnly, _VOLUME_FIELD_CACHE_PREFIX,
                          _field_cache_dir_from_config,
                          _field_cache_enabled_from_config,
                          _field_cache_mpi_comm, _field_cache_path,
                          _field_source_metadata, _jsonable, _read_field_cache,
                          _read_field_cache_mpi_part,
                          _read_h0_volume_cache_superset,
                          _read_pv_volume_cache_superset, _write_field_cache,
                          _write_field_cache_mpi_part)
from .field_products import (field_smoothing_cache_payload,
                             field_smoothing_scale_from_config,
                             validate_field_smoothing_scale)

_SPHERE_RADIUS_DX_WARN_MIN = 15.0


def _field_cache_warmup_active():
    """Return whether a dedicated cache-warming command is running."""
    return os.environ.get("CANDEL_FIELD_CACHE_WARMUP", "0") == "1"


def _h0_volume_missing_cache_error(label, density_path,
                                   velocity_path=None):
    """Build an explicit error for required field-cache misses."""
    tried = [f"density: {density_path}"]
    if velocity_path is not None:
        tried.append(f"velocity: {velocity_path}")
    return RuntimeError(
        f"missing required warmed {label} cache. Tried "
        + "; ".join(tried)
        + ". Run scripts/preprocess/prepare_field_inputs.py --products "
          "cache for this task set before inference.")


def _density_unit_normalization(source):
    """Return the raw-density divisor needed to get dimensionless 1 + delta."""
    source_str = str(source)
    source_lower = source_str.lower()
    source_upper = source_str.upper()

    if "manticorelocalcola" in source_lower:
        return None
    if "manticore" in source_lower:
        return 0.306 * 275.4, "Manticore", "Om = 0.306"
    if source_upper == "CB1" or "_CB1" in source_upper:
        return 0.307 * 275.4, "CB1", "Om = 0.307"
    if source_upper == "CB2" or "_CB2" in source_upper:
        return 0.3111 * 275.4, "CB2", "Om = 0.3111"
    if source_upper == "HAMLET_V1" or "HAMLET_V1" in source_upper:
        return 0.3 * 275.4, "HAMLET_V1", "Om = 0.3"
    return None


def _reconstruction_omega_m(
        reconstruction_name=None, reconstruction_kwargs=None, fallback=0.3):
    """Return the matter density assumed by a configured reconstruction."""
    if reconstruction_kwargs:
        for key in ("Omega_m", "Om0", "Om", "omega_m"):
            if key in reconstruction_kwargs:
                return float(reconstruction_kwargs[key])
    if reconstruction_name is not None:
        raise ValueError(
            f"`io.reconstruction_main.{reconstruction_name}` must define "
            "`Om0` when that reconstruction is used.")
    return float(fallback)


def _extract_subcube(field_3d, observer_pos, dx, radius):
    """Extract a cubic sub-region centered on the observer.

    Returns the sub-field, the observer position in sub-cube coordinates,
    and the slice objects for extracting the same region from other arrays.
    All spatial quantities are in Mpc/h.
    """
    ngrid = field_3d.shape[0]
    slices = []
    new_obs = np.empty(3, dtype=np.float32)
    for axis in range(3):
        i_obs = observer_pos[axis] / dx
        i_lo = max(0, int(np.floor(i_obs - radius / dx)))
        i_hi = min(ngrid, int(np.ceil(i_obs + radius / dx)))
        slices.append(slice(i_lo, i_hi))
        new_obs[axis] = observer_pos[axis] - i_lo * dx
    sub = field_3d[slices[0], slices[1], slices[2]]
    return sub, new_obs, slices


def _field_loader_grid_shape(loader):
    """Return the native density-grid shape without reading it when possible."""
    if hasattr(loader, "ngrid"):
        ngrid = int(loader.ngrid)
        return (ngrid, ngrid, ngrid)
    rho = loader.load_density()
    return tuple(rho.shape)


def _subcube_shape_and_observer(shape, observer_pos, dx, radius):
    """Return subcube shape and observer position for a radius cut."""
    if radius is None:
        return tuple(shape), np.asarray(observer_pos, dtype=np.float32)
    new_shape = []
    new_obs = np.empty(3, dtype=np.float32)
    for axis in range(3):
        i_obs = observer_pos[axis] / dx
        i_lo = max(0, int(np.floor(i_obs - radius / dx)))
        i_hi = min(int(shape[axis]), int(np.ceil(i_obs + radius / dx)))
        new_shape.append(i_hi - i_lo)
        new_obs[axis] = observer_pos[axis] - i_lo * dx
    return tuple(new_shape), new_obs


def _sphere_voxel_weights(disp, radius, dx):
    """Fractional voxel volumes inside a sphere.

    Uses a continuous analytic ramp across the cell's radial extent. This is
    only a boundary-cell correction; interior cells have weight 1 and exterior
    cells have weight 0.
    """
    r = np.sqrt(disp[0]**2 + disp[1]**2 + disp[2]**2)
    r_safe = np.maximum(r, 0.25 * dx)
    half_width = 0.5 * dx * (
        np.abs(disp[0]) + np.abs(disp[1]) + np.abs(disp[2])) / r_safe
    half_width = np.maximum(half_width, 0.5 * dx)

    r_min = r - half_width
    r_max = r + half_width
    return np.clip((radius - r_min) / (r_max - r_min), 0.0, 1.0).astype(
        np.float32)


def _warn_coarse_sphere_radius(radius, dx, label):
    """Warn when sphere boundaries are poorly resolved by the voxel grid."""
    radius_over_dx = float(radius) / float(dx)
    if radius_over_dx < _SPHERE_RADIUS_DX_WARN_MIN:
        fprint(
            f"warning: `{label}` spans only {radius_over_dx:.1f} voxels; "
            "sphere boundary weights are approximate, so consider increasing "
            "the radius.")


def _validate_voxel_subsample_fraction(value, label):
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"`{label}` must be in (0, 1], got {value!r}.")
    return value


def _validate_voxel_subsample_seed(value, label):
    if not isinstance(value, int) or value < 0:
        raise ValueError(
            f"`{label}` must be a non-negative int, got {value!r}.")
    return value


def _choose_voxel_subsample_indices(n_voxels, fraction, seed):
    """Return deterministic voxel indices and the actual kept fraction."""
    if np.isclose(fraction, 1.0):
        return None, 1.0
    n_keep = int(np.ceil(float(fraction) * n_voxels))
    n_keep = min(max(n_keep, 1), n_voxels)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_voxels, size=n_keep, replace=False)), (
        n_keep / n_voxels)


def _h0_volume_cache_sampling_payload(fraction, seed):
    """Cache key entries for full-resolution H0 volume data."""
    return {"downsample": 1}


def _validate_h0_volume_supersampling(factor, radius, target_dx=None):
    """Validate inner-region supersampling settings for H0 volume data."""
    if isinstance(factor, bool) or int(factor) != factor:
        raise ValueError(
            "`model.selection_integral_supersample_factor` must be an "
            f"integer >= 1, got {factor!r}.")
    factor = int(factor)
    if factor < 1:
        raise ValueError(
            "`model.selection_integral_supersample_factor` must be >= 1, "
            f"got {factor!r}.")
    radius = float(radius)
    if radius < 0.0:
        raise ValueError(
            "`model.selection_integral_supersample_radius` must be >= 0, "
            f"got {radius!r}.")
    if target_dx is not None:
        target_dx = float(target_dx)
        if target_dx < 0.0:
            raise ValueError(
                "`model.selection_integral_supersample_target_dx` must be "
                f">= 0, got {target_dx!r}.")
        if np.isclose(target_dx, 0.0):
            target_dx = None
    return factor, radius, target_dx


def _h0_volume_supersampling_from_config(config):
    """Return validated H0 inner-region supersampling config."""
    return _validate_h0_volume_supersampling(
        get_nested(config, "model/selection_integral_supersample_factor", 1),
        get_nested(config, "model/selection_integral_supersample_radius", 0.0),
        get_nested(
            config, "model/selection_integral_supersample_target_dx", None))


def _h0_volume_resolved_supersample_factor(dx, factor, target_dx=None):
    """Resolve a target subcell size to the nearest integer split factor."""
    factor, _, target_dx = _validate_h0_volume_supersampling(
        factor, 0.0, target_dx)
    if target_dx is None:
        return factor
    dx = float(dx)
    if dx <= 0.0:
        raise ValueError(f"Native voxel size must be positive, got {dx!r}.")
    ratio = dx / target_dx
    lo = max(1, int(np.floor(ratio)))
    candidates = sorted({1, lo, lo + 1})
    return min(candidates, key=lambda f: (abs(dx / f - target_dx), -f))


def _field_loader_native_dx(loader):
    """Return a loader's native Cartesian voxel size in Mpc/h."""
    boxsize = float(loader.boxsize)
    if hasattr(loader, "ngrid"):
        return boxsize / int(loader.ngrid)

    for path_attr, keys in (
            ("fname", ("density", "overdensity")),
            ("file_path", ("density",))):
        if not hasattr(loader, path_attr):
            continue
        with File(getattr(loader, path_attr), "r") as f:
            for key in keys:
                if key in f:
                    return boxsize / int(f[key].shape[0])

    if hasattr(loader, "path_density"):
        arr = np.load(loader.path_density, mmap_mode="r")
        try:
            return boxsize / int(arr.shape[0])
        finally:
            del arr

    if hasattr(loader, "_density_path"):
        from astropy.io import fits
        with fits.open(loader._density_path, memmap=True) as hdul:
            return boxsize / int(hdul[0].data.shape[0])

    raise ValueError(
        f"Cannot infer native voxel size for {type(loader).__name__}. "
        "Set `model.selection_integral_supersample_target_dx = 0` to disable "
        "target-resolution supersampling for this field.")


def _h0_volume_cache_supersampling_payload(factor, radius, target_dx=None):
    """Cache key entries for H0 inner-region supersampling."""
    factor, radius, target_dx = _validate_h0_volume_supersampling(
        factor, radius, target_dx)
    if factor == 1 or np.isclose(radius, 0.0):
        return {}
    return {
        "supersample": {
            "factor": factor,
            "radius": radius,
            "method": "linear",
        },
    }


def _h0_volume_supersampling_cache_arrays(factor, radius):
    """Arrays stored in H0 volume caches to record supersampling settings."""
    factor, radius, _ = _validate_h0_volume_supersampling(factor, radius)
    active = factor > 1 and radius > 0.0
    return {
        "supersample_factor": np.asarray(factor if active else 1,
                                         dtype=np.int32),
        "supersample_radius": np.asarray(radius if active else 0.0,
                                         dtype=np.float32),
        "supersample_method": np.asarray("linear" if active else "none"),
    }


def _supersample_offsets_3d(factor, dx):
    """Return subcell centre offsets for a regular factor^3 subgrid."""
    one_d = ((np.arange(factor, dtype=np.float32) + 0.5) / factor
             - 0.5) * np.float32(dx)
    ox, oy, oz = np.meshgrid(one_d, one_d, one_d, indexing="ij")
    return np.column_stack((
        ox.reshape(-1), oy.reshape(-1), oz.reshape(-1))).astype(np.float32)


def _supersample_linear_interpolator(shape, sup_flat, offsets, dx):
    """Precompute trilinear interpolation indices for supersampled cells."""
    parent = np.unravel_index(sup_flat, shape)
    pos = [
        parent[axis][:, None] + offsets[None, :, axis] / dx
        for axis in range(3)
    ]

    indices = []
    weights = []
    for axis, n_axis in enumerate(shape):
        x = np.clip(pos[axis].reshape(-1), 0.0, n_axis - 1.0)
        lo = np.floor(x).astype(np.intp)
        hi = np.minimum(lo + 1, n_axis - 1)
        indices.extend([lo, hi])
        weights.append((x - lo).astype(np.float32))
    return tuple(indices), tuple(weights)


def _trilinear_interpolate_3d(field, indices, weights):
    """Evaluate a regular 3D field at precomputed trilinear positions."""
    i0, i1, j0, j1, k0, k1 = indices
    wx, wy, wz = weights
    field = np.asarray(field)

    c00 = field[i0, j0, k0] * (1.0 - wx) + field[i1, j0, k0] * wx
    c01 = field[i0, j0, k1] * (1.0 - wx) + field[i1, j0, k1] * wx
    c10 = field[i0, j1, k0] * (1.0 - wx) + field[i1, j1, k0] * wx
    c11 = field[i0, j1, k1] * (1.0 - wx) + field[i1, j1, k1] * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return (c0 * (1.0 - wz) + c1 * wz).astype(np.float32)


def _h0_volume_quadrature_geometry(
        log_r_grid, disp_ref, r_sub_ref, dx, geometry, radius,
        supersample_factor=1, supersample_radius=0.0, store_rhat=False):
    """Build flattened H0 volume-quadrature geometry arrays."""
    factor, supersample_radius, _ = _validate_h0_volume_supersampling(
        supersample_factor, supersample_radius)
    use_supersampling = factor > 1 and supersample_radius > 0.0

    if geometry == "sphere" and radius is not None:
        voxel_weight = _sphere_voxel_weights(disp_ref, radius, dx)
        voxel_mask = voxel_weight > 0.0
    else:
        voxel_weight = np.ones(r_sub_ref.shape, dtype=np.float32)
        voxel_mask = np.ones(r_sub_ref.shape, dtype=bool)

    flat_mask = voxel_mask.reshape(-1)
    if not use_supersampling:
        if geometry == "sphere" and radius is not None:
            flat_idx = np.flatnonzero(flat_mask)
            log_r_3d = np.asarray(log_r_grid).reshape(-1)[flat_idx]
            log_volume_weight_3d = np.log(
                voxel_weight.reshape(-1)[flat_idx]).astype(np.float32)
            rhat_fields = None
            if store_rhat:
                rhat_fields = []
                for comp in range(3):
                    rhat = (np.broadcast_to(
                        disp_ref[comp], r_sub_ref.shape) / r_sub_ref)
                    rhat_fields.append(rhat.reshape(-1)[flat_idx].astype(
                        np.float32))
        else:
            flat_idx = None
            log_r_3d = log_r_grid
            log_volume_weight_3d = None
            rhat_fields = None
            if store_rhat:
                rhat_fields = []
                for comp in range(3):
                    rhat = (np.broadcast_to(
                        disp_ref[comp], r_sub_ref.shape) / r_sub_ref)
                    rhat_fields.append(rhat.astype(np.float32))
        return {
            "voxel_mask": voxel_mask,
            "unsup_flat": flat_idx,
            "sup_flat": np.empty(0, dtype=np.int64),
            "n_subcells": 1,
            "log_r_3d": log_r_3d,
            "log_volume_weight_3d": log_volume_weight_3d,
            "rhat_fields": rhat_fields,
        }

    supersample_weight = _sphere_voxel_weights(
        disp_ref, supersample_radius, dx)
    supersample_mask = (supersample_weight > 0.0) & voxel_mask
    unsup_mask = voxel_mask & ~supersample_mask

    unsup_flat = np.flatnonzero(unsup_mask.reshape(-1))
    sup_flat = np.flatnonzero(supersample_mask.reshape(-1))
    n_offsets = factor ** 3
    n_unsup = len(unsup_flat)
    n_sup = len(sup_flat)
    n_out = n_unsup + n_sup * n_offsets

    flat_log_r = np.asarray(log_r_grid).reshape(-1)
    flat_weight = voxel_weight.reshape(-1)
    log_r_3d = np.empty(n_out, dtype=np.float32)
    log_volume_weight_3d = np.empty(n_out, dtype=np.float32)
    log_r_3d[:n_unsup] = flat_log_r[unsup_flat]
    log_volume_weight_3d[:n_unsup] = np.log(
        flat_weight[unsup_flat]).astype(np.float32)

    rhat_fields = None
    if store_rhat:
        rhat_fields = [
            np.empty(n_out, dtype=np.float32) for _ in range(3)]
        for comp in range(3):
            rhat = (np.broadcast_to(
                disp_ref[comp], r_sub_ref.shape) / r_sub_ref)
            rhat_fields[comp][:n_unsup] = rhat.reshape(-1)[unsup_flat]

    offsets = _supersample_offsets_3d(factor, dx)
    interp_indices, interp_weights = _supersample_linear_interpolator(
        r_sub_ref.shape, sup_flat, offsets, dx)
    sub_dx = dx / factor
    log_subcell_fraction = -3.0 * np.log(factor)
    flat_disp = [
        np.broadcast_to(d, r_sub_ref.shape).reshape(-1)
        for d in disp_ref]
    write = n_unsup
    chunk_size = max(1, min(n_sup, 64))
    for start in range(0, n_sup, chunk_size):
        stop = min(start + chunk_size, n_sup)
        parent_idx = sup_flat[start:stop]
        n_parent = len(parent_idx)
        sl = slice(write, write + n_parent * n_offsets)
        x = flat_disp[0][parent_idx, None] + offsets[None, :, 0]
        y = flat_disp[1][parent_idx, None] + offsets[None, :, 1]
        z = flat_disp[2][parent_idx, None] + offsets[None, :, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        r_safe = np.maximum(r, 0.25 * sub_dx)
        log_r_3d[sl] = np.log(r_safe).reshape(-1).astype(np.float32)
        if geometry == "sphere" and radius is not None:
            sub_weight = _sphere_voxel_weights((x, y, z), radius, sub_dx)
            log_weight = np.full(
                sub_weight.shape, -np.inf, dtype=np.float32)
            keep = sub_weight > 0.0
            log_weight[keep] = (
                np.log(sub_weight[keep]) + log_subcell_fraction)
            log_volume_weight_3d[sl] = log_weight.reshape(-1)
        else:
            log_volume_weight_3d[sl] = log_subcell_fraction
        if store_rhat:
            rhat_fields[0][sl] = (x / r_safe).reshape(-1).astype(np.float32)
            rhat_fields[1][sl] = (y / r_safe).reshape(-1).astype(np.float32)
            rhat_fields[2][sl] = (z / r_safe).reshape(-1).astype(np.float32)
        write = sl.stop

    return {
        "voxel_mask": voxel_mask,
        "unsup_flat": unsup_flat,
        "sup_flat": sup_flat,
        "n_subcells": n_offsets,
        "sup_interp_indices": interp_indices,
        "sup_interp_weights": interp_weights,
        "log_r_3d": log_r_3d,
        "log_volume_weight_3d": log_volume_weight_3d,
        "rhat_fields": rhat_fields,
    }


def _grid_displacements(shape, observer_pos, dx):
    """Return axis displacements and safe radius for a Cartesian grid."""
    ax = [(np.arange(shape[i], dtype=np.float32) + 0.5) * dx
          for i in range(3)]
    disp = [ax[0][:, None, None] - observer_pos[0],
            ax[1][None, :, None] - observer_pos[1],
            ax[2][None, None, :] - observer_pos[2]]
    r_grid = np.sqrt(disp[0]**2 + disp[1]**2 + disp[2]**2)
    r_grid = np.maximum(r_grid, 0.25 * dx)
    return disp, r_grid


def _max_grid_radius(shape, observer_pos, dx):
    """Return the farthest cell-centre radius on a Cartesian grid."""
    max_disp2 = 0.0
    for n_axis, obs_axis in zip(shape, observer_pos):
        lo = 0.5 * dx - float(obs_axis)
        hi = (int(n_axis) - 0.5) * dx - float(obs_axis)
        max_disp2 += max(abs(lo), abs(hi)) ** 2
    return float(np.sqrt(max_disp2))


def _expected_h0_volume_grid_from_loader(
        loader, geometry, subcube_radius, supersample_factor,
        supersample_radius):
    """Return the H0 volume radial grid and voxel-volume weight."""
    shape = _field_loader_grid_shape(loader)
    dx = float(loader.boxsize) / int(shape[0])
    obs = np.asarray(loader.observer_pos, dtype=np.float32)
    extract_radius = subcube_radius
    if geometry == "sphere" and subcube_radius is not None:
        extract_radius = subcube_radius + 0.5 * np.sqrt(3.0) * dx
    shape, obs = _subcube_shape_and_observer(
        shape, obs, dx, extract_radius)
    log_r_grid, log_dV = _volume_density_geometry(shape, obs, dx)
    disp, r_grid = _grid_displacements(shape, obs, dx)
    quad = _h0_volume_quadrature_geometry(
        log_r_grid, disp, r_grid, dx, geometry, subcube_radius,
        supersample_factor, supersample_radius, store_rhat=False)
    return np.exp(np.asarray(quad["log_r_3d"])).astype(np.float32), log_dV


def _expected_h0_volume_max_radius_from_loader(
        loader, geometry, subcube_radius, supersample_factor,
        supersample_radius):
    """Return the maximum H0 volume radius requested for one loader."""
    if geometry != "sphere":
        shape = _field_loader_grid_shape(loader)
        dx = float(loader.boxsize) / int(shape[0])
        obs = np.asarray(loader.observer_pos, dtype=np.float32)
        shape, obs = _subcube_shape_and_observer(
            shape, obs, dx, subcube_radius)
        return _max_grid_radius(shape, obs, dx)
    r_3d, _ = _expected_h0_volume_grid_from_loader(
        loader, geometry, subcube_radius, supersample_factor,
        supersample_radius)
    return float(np.max(r_3d))


def _expected_pv_volume_grid_from_loader(
        loader, downsample, subcube_radius, pad_subcube_boundary,
        geometry, radius, voxel_subsample_fraction, voxel_subsample_seed):
    """Return the PV volume radial grid requested for one loader."""
    shape = _field_loader_grid_shape(loader)
    dx = float(loader.boxsize) / int(shape[0])
    if downsample > 1:
        shape = tuple((int(n) - 1) // int(downsample) + 1 for n in shape)
        dx *= int(downsample)
    obs = np.asarray(loader.observer_pos, dtype=np.float32)
    if subcube_radius is not None:
        extract_radius = subcube_radius
        if pad_subcube_boundary:
            extract_radius += 0.5 * np.sqrt(3.0) * dx
        shape, obs = _subcube_shape_and_observer(
            shape, obs, dx, extract_radius)

    log_r_grid, log_dV = _volume_density_geometry(shape, obs, dx)
    if geometry == "sphere":
        disp, r_grid = _grid_displacements(shape, obs, dx)
        voxel_weight = _sphere_voxel_weights(disp, radius, dx)
        voxel_mask = voxel_weight > 0.0
        r_3d = np.exp(np.asarray(log_r_grid)[voxel_mask])
    else:
        r_3d = np.exp(np.asarray(log_r_grid))
    voxel_subsample_idx, actual_fraction = _choose_voxel_subsample_indices(
        int(np.size(r_3d)), voxel_subsample_fraction,
        voxel_subsample_seed)
    if voxel_subsample_idx is not None:
        r_3d = r_3d.reshape(-1)[voxel_subsample_idx]
        log_dV = log_dV - float(np.log(actual_fraction))
    return r_3d.astype(np.float32), log_dV


def _expected_pv_volume_max_radius_from_loader(
        loader, downsample, subcube_radius, pad_subcube_boundary,
        geometry, radius, voxel_subsample_fraction, voxel_subsample_seed):
    """Return the maximum PV volume radius requested for one loader."""
    if geometry != "sphere":
        shape = _field_loader_grid_shape(loader)
        dx = float(loader.boxsize) / int(shape[0])
        if downsample > 1:
            shape = tuple((int(n) - 1) // int(downsample) + 1 for n in shape)
            dx *= int(downsample)
        obs = np.asarray(loader.observer_pos, dtype=np.float32)
        if subcube_radius is not None:
            shape, obs = _subcube_shape_and_observer(
                shape, obs, dx, subcube_radius)
        n_voxels = int(np.prod(shape))
        voxel_subsample_idx, _ = _choose_voxel_subsample_indices(
            n_voxels, voxel_subsample_fraction, voxel_subsample_seed)
        if voxel_subsample_idx is None:
            return _max_grid_radius(shape, obs, dx)
        i, j, k = np.unravel_index(voxel_subsample_idx, shape)
        x = (i.astype(np.float32) + 0.5) * dx - obs[0]
        y = (j.astype(np.float32) + 0.5) * dx - obs[1]
        z = (k.astype(np.float32) + 0.5) * dx - obs[2]
        return float(np.max(np.sqrt(x**2 + y**2 + z**2)))
    r_3d, _ = _expected_pv_volume_grid_from_loader(
        loader, downsample, subcube_radius, pad_subcube_boundary,
        geometry, radius, voxel_subsample_fraction, voxel_subsample_seed)
    return float(np.max(r_3d))


def _h0_volume_apply_quadrature(field, quad):
    """Map a 3D field onto the H0 quadrature geometry."""
    if len(quad["sup_flat"]) == 0:
        if quad["unsup_flat"] is None:
            return np.asarray(field, dtype=np.float32)
        flat = np.asarray(field).reshape(-1)
        return flat[quad["unsup_flat"]].astype(np.float32)
    flat = np.asarray(field).reshape(-1)
    sup_values = _trilinear_interpolate_3d(
        field, quad["sup_interp_indices"], quad["sup_interp_weights"])
    return np.concatenate((
        flat[quad["unsup_flat"]],
        sup_values,
    )).astype(np.float32)


def _subsample_h0_volume_arrays(arrays, fraction, seed):
    """Apply random voxel thinning to full cached H0 volume arrays."""
    idx, actual = _choose_voxel_subsample_indices(
        int(np.size(arrays["r_3d"])), fraction, seed)
    if idx is None:
        return arrays, actual

    out = dict(arrays)
    for key in ("r_3d", "log_volume_weight_3d",
                "rhat_x_3d", "rhat_y_3d", "rhat_z_3d"):
        if key in out:
            out[key] = np.asarray(out[key]).reshape(-1)[idx]
    for key in ("rho_3d_fields", "vrad_3d_fields"):
        if key in out:
            arr = np.asarray(out[key])
            out[key] = arr.reshape((arr.shape[0], -1))[:, idx]
    out["log_dV_3d"] = float(out["log_dV_3d"]) - float(np.log(actual))
    return out, actual


def _precompute_cosmo_3d(log_r_3d, Om0, include_redshift=True):
    """Precompute h=1 distance modulus and optionally redshift on 3D grid.

    At runtime: mu(r, h) = mu_at_h1 - 5*log10(h), z_cosmo is h-independent.
    Both depend only on r in Mpc/h, which is fixed by the grid geometry.
    """
    from ..cosmo.cosmography import Distance2Distmod, Distance2Redshift

    r_flat = jnp.exp(jnp.asarray(log_r_3d).ravel())
    mu_flat = Distance2Distmod(Om0=Om0)(r_flat, h=1.0)
    if include_redshift:
        z_flat = Distance2Redshift(Om0=Om0)(r_flat, h=1.0)
    else:
        z_flat = None

    shape = log_r_3d.shape
    z_3d = None if z_flat is None else z_flat.reshape(shape)
    return mu_flat.reshape(shape), z_3d


def _h0_density_fields_from_rho(rho_fields, mode, copy=True):
    """Apply the runtime bias-dependent density representation."""
    if copy:
        rho_fields = np.array(rho_fields, dtype=np.float32, copy=True)
    else:
        rho_fields = np.asarray(rho_fields, dtype=np.float32)
        if not rho_fields.flags.writeable:
            rho_fields = np.array(rho_fields, dtype=np.float32, copy=True)

    if mode == "log_rho":
        np.log(rho_fields, out=rho_fields)
    else:
        np.subtract(rho_fields, 1.0, out=rho_fields)
    return rho_fields


def _h0_log_radius_from_r(r_3d, copy=True):
    """Return log radius, optionally reusing the input array's memory."""
    if copy:
        log_r_3d = np.array(r_3d, dtype=np.float32, copy=True)
    else:
        log_r_3d = np.asarray(r_3d, dtype=np.float32)
        if not log_r_3d.flags.writeable:
            log_r_3d = np.array(log_r_3d, dtype=np.float32, copy=True)
    np.log(log_r_3d, out=log_r_3d)
    return log_r_3d


def _h0_volume_runtime_result(
        rho_fields, r_3d, log_dV_3d, source_meta, mode, Om0,
        load_velocity, vrad_fields=None, log_volume_weight_3d=None,
        rhat_fields=None, copy_inputs=True):
    """Convert cached H0 volume arrays to the data dict used by models."""
    coord_frame = source_meta[0]["state"]["coordinate_frame"]
    log_r_3d = jnp.asarray(_h0_log_radius_from_r(r_3d, copy=copy_inputs))
    mu_at_h1_3d, zcosmo_3d = _precompute_cosmo_3d(
        log_r_3d, Om0, include_redshift=load_velocity)
    density_3d_fields = jnp.asarray(
        _h0_density_fields_from_rho(rho_fields, mode, copy=copy_inputs))
    del rho_fields, r_3d
    result = {
        "density_3d_fields": density_3d_fields,
        "log_r_3d": log_r_3d,
        "log_dV_3d": float(log_dV_3d),
        "mu_at_h1_3d": mu_at_h1_3d,
        "density_3d_mode": mode,
        "volume_density_batch_size": 1,
        "coordinate_frame_3d": coord_frame,
    }
    if zcosmo_3d is not None:
        result["zcosmo_3d"] = zcosmo_3d
    if log_volume_weight_3d is not None:
        result["log_volume_weight_3d"] = jnp.asarray(log_volume_weight_3d)
    if load_velocity:
        result["vrad_3d_fields"] = jnp.asarray(vrad_fields)
        for label in ("rhat_x_3d", "rhat_y_3d", "rhat_z_3d"):
            result[label] = jnp.asarray(rhat_fields[label])
    return result


def _cached_h0_volume_result(
        cached, source_meta, mode, Om0, load_velocity,
        voxel_subsample_fraction=1.0, voxel_subsample_seed=42):
    """Convert cached H0 base fields to the data dict used by models."""
    cached, actual_fraction = _subsample_h0_volume_arrays(
        cached, voxel_subsample_fraction, voxel_subsample_seed)
    if not np.isclose(actual_fraction, 1.0):
        fprint(
            f"  applying random voxel subsample f={actual_fraction:.6g} "
            f"({np.size(cached['r_3d']):,} kept).")
    rhat_fields = None
    rho_fields = cached.pop("rho_3d_fields")
    r_3d = cached.pop("r_3d")
    log_dV_3d = cached.pop("log_dV_3d")
    log_volume_weight_3d = cached.pop("log_volume_weight_3d", None)
    vrad_fields = None
    if load_velocity:
        vrad_fields = cached.pop("vrad_3d_fields")
        rhat_fields = {
            label: cached.pop(label)
            for label in ("rhat_x_3d", "rhat_y_3d", "rhat_z_3d")
        }
    return _h0_volume_runtime_result(
        rho_fields, r_3d, log_dV_3d, source_meta, mode, Om0, load_velocity,
        vrad_fields=vrad_fields, log_volume_weight_3d=log_volume_weight_3d,
        rhat_fields=rhat_fields, copy_inputs=False)


def _load_volume_data_for_H0_mpi(
        comm, field_name, field_kwargs, field_indices, galaxy_bias, Om0,
        subcube_radius, voxel_subsample_fraction, voxel_subsample_seed,
        load_velocity, geometry, density_cache_path, velocity_cache_path,
        source_meta, mode, density_required, velocity_required,
        supersample_factor=1, supersample_radius=0.0,
        field_smoothing_scale=None, max_radius=None):
    """Build one H0 volume cache file with fields split over MPI ranks."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        fprint(f"MPI field-cache warmup: splitting {len(field_indices)} "
               f"field(s) over {size} rank(s).")
        cache_dir = os.path.dirname(density_cache_path[0])
        os.makedirs(cache_dir, exist_ok=True)
        part_dir = tempfile.mkdtemp(
            prefix=".tmp_h0_volume_mpi_",
            dir=cache_dir)
    else:
        part_dir = None
    part_dir = comm.bcast(part_dir, root=0)

    local_parts = []
    part_keys = {
        "rho_3d_fields", "r_3d", "log_dV_3d", "log_volume_weight_3d",
        "supersample_factor", "supersample_radius", "supersample_method",
    }
    if load_velocity:
        part_keys.update({
            "vrad_3d_fields", "rhat_x_3d", "rhat_y_3d", "rhat_z_3d",
        })
    try:
        for k, nsim in enumerate(field_indices):
            if k % size != rank:
                continue
            fprint(f"rank {rank}: warming field {k} (nsim={int(nsim)}).")
            partial = _load_volume_data_for_H0(
                field_name, field_kwargs, [int(nsim)], galaxy_bias, Om0,
                subcube_radius=subcube_radius,
                voxel_subsample_fraction=voxel_subsample_fraction,
                voxel_subsample_seed=voxel_subsample_seed,
                load_velocity=load_velocity,
                geometry=geometry,
                cache_dir=None,
                cache_enabled=False,
                return_cache_fields=True,
                supersample_factor=supersample_factor,
                supersample_radius=supersample_radius,
                field_smoothing_scale=field_smoothing_scale)
            field_arrays = {
                key: np.asarray(value)
                for key, value in partial.items()
                if key in part_keys
            }
            part_path = join(part_dir, f"field_{k:06d}_rank_{rank}.npz")
            _write_field_cache_mpi_part(part_path, k, field_arrays)
            local_parts.append({"field_order": k, "path": part_path})
            del partial, field_arrays
        manifest = {
            "rank": rank, "parts": local_parts, "count": len(local_parts),
            "error": None,
        }
    except Exception as exc:
        manifest = {
            "rank": rank, "parts": [], "count": 0, "error": repr(exc),
        }

    manifests = comm.gather(manifest, root=0)
    error = None
    cache_meta = None
    if rank == 0:
        try:
            failures = [
                f"rank {item['rank']}: {item['error']}"
                for item in manifests
                if item["error"] is not None
            ]
            if failures:
                raise RuntimeError("; ".join(failures))

            parts = []
            for item in manifests:
                parts.extend(item["parts"])
            parts.sort(key=lambda item: item["field_order"])
            if len(parts) != len(field_indices):
                raise RuntimeError(
                    f"MPI field-cache warmup produced {len(parts)} field(s), "
                    f"expected {len(field_indices)}.")

            first_order, first = _read_field_cache_mpi_part(parts[0]["path"])
            if first_order != parts[0]["field_order"]:
                raise RuntimeError(
                    "MPI field-cache warmup found inconsistent field order "
                    f"in `{parts[0]['path']}`.")

            density_shape = (
                (len(field_indices),) + first["rho_3d_fields"].shape[1:])
            cache_meta = {"density_shape": density_shape}

            for out_idx, part in enumerate(parts):
                if out_idx == 0:
                    field_order, arrays = first_order, first
                else:
                    field_order, arrays = _read_field_cache_mpi_part(
                        part["path"])
                if field_order != part["field_order"]:
                    raise RuntimeError(
                        "MPI field-cache warmup found inconsistent field "
                        f"order in `{part['path']}`.")
                for label in (
                        "r_3d", "supersample_factor",
                        "supersample_radius", "supersample_method"):
                    if not np.array_equal(arrays[label], first[label]):
                        raise RuntimeError(
                            "MPI field-cache warmup found inconsistent "
                            f"`{label}` arrays across fields.")
                if not np.isclose(arrays["log_dV_3d"], first["log_dV_3d"]):
                    raise RuntimeError(
                        "MPI field-cache warmup found inconsistent "
                        "`log_dV_3d` values across fields.")
                has_weight = "log_volume_weight_3d" in arrays
                first_has_weight = "log_volume_weight_3d" in first
                if has_weight != first_has_weight:
                    raise RuntimeError(
                        "MPI field-cache warmup found inconsistent "
                        "`log_volume_weight_3d` presence across fields.")
                if has_weight and not np.array_equal(
                        arrays["log_volume_weight_3d"],
                        first["log_volume_weight_3d"]):
                    raise RuntimeError(
                        "MPI field-cache warmup found inconsistent "
                        "`log_volume_weight_3d` arrays across fields.")
                if load_velocity:
                    for label in ("rhat_x_3d", "rhat_y_3d", "rhat_z_3d"):
                        if not np.array_equal(arrays[label], first[label]):
                            raise RuntimeError(
                                "MPI field-cache warmup found inconsistent "
                                f"`{label}` arrays across fields.")
                density_cache_arrays = {
                    "rho_3d_fields": arrays["rho_3d_fields"],
                    "r_3d": arrays["r_3d"],
                    "log_dV_3d": arrays["log_dV_3d"],
                    "supersample_factor": arrays["supersample_factor"],
                    "supersample_radius": arrays["supersample_radius"],
                    "supersample_method": arrays["supersample_method"],
                }
                if "log_volume_weight_3d" in arrays:
                    density_cache_arrays["log_volume_weight_3d"] = (
                        arrays["log_volume_weight_3d"])
                _write_field_cache(
                    density_cache_path[out_idx],
                    f"H0 3D volume density field "
                    f"{int(field_indices[out_idx])}",
                    density_cache_arrays)
                if load_velocity:
                    velocity_cache_arrays = {
                        "vrad_3d_fields": arrays["vrad_3d_fields"],
                        "r_3d": arrays["r_3d"],
                        "log_dV_3d": arrays["log_dV_3d"],
                        "supersample_factor": arrays["supersample_factor"],
                        "supersample_radius": arrays["supersample_radius"],
                        "supersample_method": arrays["supersample_method"],
                    }
                    for label in ("rhat_x_3d", "rhat_y_3d", "rhat_z_3d"):
                        velocity_cache_arrays[label] = arrays[label]
                    _write_field_cache(
                        velocity_cache_path[out_idx],
                        f"H0 3D volume velocity field "
                        f"{int(field_indices[out_idx])}",
                        velocity_cache_arrays)
                if out_idx != 0:
                    del arrays
        except Exception as exc:
            error = repr(exc)
        finally:
            shutil.rmtree(part_dir, ignore_errors=True)

    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"MPI field-cache warmup failed: {error}")
    cache_meta = comm.bcast(cache_meta, root=0)

    if os.environ.get("CANDEL_FIELD_CACHE_MPI", "0") == "1":
        return {
            "density_3d_fields": _ArrayShapeOnly(cache_meta["density_shape"]),
            "density_3d_mode": mode,
            "volume_density_batch_size": 1,
        }

    cache_root = os.path.dirname(os.path.dirname(density_cache_path[0]))
    expected_supersampling = _h0_volume_supersampling_cache_arrays(
        supersample_factor, supersample_radius)
    density_cached = _read_h0_volume_cache_superset(
        cache_root, {
            "kind": "volume_field_data",
            "product": "h0_volume",
            "loader_name": field_name,
            "field_indices": _jsonable(np.asarray(field_indices)),
            "subcube_radius": subcube_radius,
            "max_radius": max_radius,
            "geometry": geometry,
            "load_velocity": False,
            **field_smoothing_cache_payload(field_smoothing_scale),
            **_h0_volume_cache_supersampling_payload(
                supersample_factor, supersample_radius),
        }, "H0 3D volume density", density_required, field_indices,
        expected_r_3d=max_radius,
        expected_supersampling=expected_supersampling)
    velocity_cached = None
    if load_velocity:
        velocity_cached = _read_h0_volume_cache_superset(
            cache_root, {
                "kind": "volume_field_data",
                "product": "h0_volume",
                "loader_name": field_name,
                "field_indices": _jsonable(np.asarray(field_indices)),
                "subcube_radius": subcube_radius,
                "max_radius": max_radius,
                "geometry": geometry,
                "load_velocity": True,
                **field_smoothing_cache_payload(field_smoothing_scale),
                **_h0_volume_cache_supersampling_payload(
                    supersample_factor, supersample_radius),
            }, "H0 3D volume velocity", velocity_required, field_indices,
            expected_r_3d=max_radius,
            expected_supersampling=expected_supersampling)
    if density_cached is None or (load_velocity and velocity_cached is None):
        raise RuntimeError(
            "failed to read MPI-warmed H0 3D volume cache.")
    cached = density_cached
    if load_velocity:
        cached = {**density_cached, **velocity_cached}
    return _cached_h0_volume_result(
        cached, source_meta, mode, Om0, load_velocity,
        voxel_subsample_fraction, voxel_subsample_seed)


def _load_volume_data_for_H0(
        field_name, field_kwargs, field_indices, galaxy_bias, Om0,
        subcube_radius=None, voxel_subsample_fraction=1.0,
        voxel_subsample_seed=42, load_velocity=False, geometry="sphere",
        cache_dir=None, cache_enabled=True, return_cache_fields=False,
        supersample_factor=1, supersample_radius=0.0,
        supersample_target_dx=None, field_smoothing_scale=None):
    """Load 3D voxel data for H0 selection integrals.

    Returns a dict to be merged into an H0-model data dict. The density field
    is stored as ``density_3d_fields`` with shape ``(n_fields, n_voxels)`` for
    spherical geometry or when voxel subsampling is enabled; unsubsampled
    cubic geometry keeps ``(n_fields, nx, ny, nz)``.
    Velocity projections and line-of-sight unit vectors are included only when
    ``load_velocity`` is True.
    """
    if geometry not in ("sphere", "cube"):
        raise ValueError(
            f"`selection_integral_geometry` must be 'sphere' or 'cube', "
            f"got {geometry!r}.")
    voxel_subsample_fraction = _validate_voxel_subsample_fraction(
        voxel_subsample_fraction, "model.density_3d_subsample_fraction")
    voxel_subsample_seed = _validate_voxel_subsample_seed(
        voxel_subsample_seed, "model.density_3d_subsample_seed")
    supersample_factor, supersample_radius, supersample_target_dx = \
        _validate_h0_volume_supersampling(
            supersample_factor, supersample_radius, supersample_target_dx)
    field_smoothing_scale = validate_field_smoothing_scale(
        field_smoothing_scale)
    mode = _volume_density_mode(galaxy_bias)
    rho_fields = []
    vrad_fields = [] if load_velocity else None
    log_r_3d = None
    coord_frame = None
    obs_sub_ref = None
    disp_ref = None
    r_sub_ref = None
    dx_ref = None
    shape_ref = None
    voxel_mask_ref = None
    quad_ref = None
    log_volume_weight_3d = None
    loaders = []
    source_meta = []

    for nsim in field_indices:
        kwargs = dict(field_kwargs)
        kwargs["nsim"] = int(nsim)
        loader = name2field_loader(field_name)(**kwargs)
        loaders.append(loader)
        source_meta.append(_field_source_metadata(loader))

    native_dx = None
    target_dx_requested = supersample_target_dx is not None
    if target_dx_requested:
        native_dx = _field_loader_native_dx(loaders[0])
        supersample_factor = _h0_volume_resolved_supersample_factor(
            native_dx, supersample_factor, supersample_target_dx)
    if field_smoothing_scale is not None:
        native_dx = native_dx or _field_loader_native_dx(loaders[0])
        if field_smoothing_scale <= native_dx:
            raise ValueError(
                f"`model.field_3d_smoothing_scale` must exceed the "
                f"native field voxel size {native_dx:g} Mpc/h; got "
                f"{field_smoothing_scale:g} Mpc/h.")
    use_supersampling = (
        supersample_factor > 1 and supersample_radius > 0.0)

    Om0_model = Om0
    Om0 = _reconstruction_omega_m(
        field_name, field_kwargs, fallback=Om0_model)
    if not np.isclose(Om0, Om0_model):
        fprint(
            f"using reconstruction Om0={Om0:g} for `{field_name}` "
            f"instead of model Om0={Om0_model:g}.")

    if cache_enabled:
        expected_max_r_3d = _expected_h0_volume_max_radius_from_loader(
            loaders[0], geometry, subcube_radius, supersample_factor,
            supersample_radius)
        base_cache_payload = {
            "kind": "volume_field_data",
            "product": "h0_volume",
            "loader_name": field_name,
            "field_indices": _jsonable(np.asarray(field_indices)),
            "subcube_radius": subcube_radius,
            "max_radius": expected_max_r_3d,
            "geometry": geometry,
            "sources": source_meta,
        }
        base_cache_payload.update(_h0_volume_cache_sampling_payload(
            voxel_subsample_fraction, voxel_subsample_seed))
        base_cache_payload.update(_h0_volume_cache_supersampling_payload(
            supersample_factor, supersample_radius))
        supersampling_cache_arrays = _h0_volume_supersampling_cache_arrays(
            supersample_factor, supersample_radius)
        density_cache_payload = {
            **base_cache_payload,
            **field_smoothing_cache_payload(field_smoothing_scale),
            "load_velocity": False,
        }
        velocity_cache_payload = {
            **base_cache_payload,
            **field_smoothing_cache_payload(field_smoothing_scale),
            "load_velocity": True,
        }
        density_cache_paths = [
            _field_cache_path(
                cache_dir, _VOLUME_FIELD_CACHE_PREFIX,
                {**density_cache_payload, "field_indices": [int(nsim)],
                 "sources": [source_meta[i]]})
            for i, nsim in enumerate(field_indices)
        ]
        velocity_cache_paths = [
            _field_cache_path(
                cache_dir, _VOLUME_FIELD_CACHE_PREFIX,
                {**velocity_cache_payload, "field_indices": [int(nsim)],
                 "sources": [source_meta[i]]})
            for i, nsim in enumerate(field_indices)
        ]
        supersampling_required = [
            "supersample_factor", "supersample_radius", "supersample_method"]
        density_required = [
            "rho_3d_fields", "r_3d", "log_dV_3d", *supersampling_required]
        if ((geometry == "sphere" and subcube_radius is not None)
                or use_supersampling):
            density_required.append("log_volume_weight_3d")
        velocity_required = [
            "vrad_3d_fields", "r_3d", "log_dV_3d",
            "rhat_x_3d", "rhat_y_3d", "rhat_z_3d",
            *supersampling_required]
        raw_read_allowed = field_allows_raw_product_reads(field_name)
        density_cached = _read_h0_volume_cache_superset(
            cache_dir, density_cache_payload, "H0 3D volume density",
            density_required, field_indices, expected_r_3d=expected_max_r_3d,
            expected_supersampling=supersampling_cache_arrays)
        if load_velocity:
            velocity_cached = _read_h0_volume_cache_superset(
                cache_dir, velocity_cache_payload,
                "H0 3D volume velocity", velocity_required,
                field_indices, expected_r_3d=expected_max_r_3d,
                expected_supersampling=supersampling_cache_arrays)
            cached = None
            if density_cached is not None and velocity_cached is not None:
                cached = {**density_cached, **velocity_cached}
        else:
            cached = density_cached
        if cached is not None:
            return _cached_h0_volume_result(
                cached, source_meta, mode, Om0, load_velocity,
                voxel_subsample_fraction, voxel_subsample_seed)
        if not raw_read_allowed and not _field_cache_warmup_active():
            density_attempt = density_cache_paths[:5]
            velocity_attempt = None
            if load_velocity:
                velocity_attempt = velocity_cache_paths[:5]
            raise _h0_volume_missing_cache_error(
                "H0 3D volume data", density_attempt, velocity_attempt)
        fprint("H0 3D volume cache:")
        fprint(f"  files: {len(field_indices)} per-field file(s)")
        fprint(f"  directory: `{cache_dir}`")
        fprint("  status: missing/stale; building from raw fields")
        mpi_comm = _field_cache_mpi_comm()
        if mpi_comm is not None and len(field_indices) > 1:
            return _load_volume_data_for_H0_mpi(
                mpi_comm, field_name, field_kwargs, field_indices,
                galaxy_bias, Om0, subcube_radius,
                voxel_subsample_fraction, voxel_subsample_seed,
                load_velocity, geometry, density_cache_paths,
                velocity_cache_paths,
                source_meta, mode, density_required, velocity_required,
                supersample_factor, supersample_radius,
                field_smoothing_scale=field_smoothing_scale,
                max_radius=expected_max_r_3d)
    else:
        density_cache_paths = None
        velocity_cache_paths = None

    if use_supersampling and target_dx_requested:
        fprint(
            "  H0 volume supersampling target: "
            f"native_dx={native_dx:g} Mpc/h, "
            f"target_dx={supersample_target_dx:g} Mpc/h, "
            f"resolved_dx={native_dx / supersample_factor:g} Mpc/h.")

    for k, (nsim, loader) in enumerate(zip(field_indices, loaders)):
        frame = getattr(loader, "coordinate_frame", "icrs")
        if coord_frame is None:
            coord_frame = frame
        elif frame != coord_frame:
            raise ValueError(
                "All 3D volume fields must use the same coordinate frame; "
                f"got {coord_frame!r} and {frame!r}.")

        rho = loader.load_density()
        norm = _density_unit_normalization(field_name)
        if norm is not None:
            divisor, label, detail = norm
            if k == 0:
                fprint(f"  normalizing {label} 3D density ({detail}).")
            rho /= divisor

        obs = np.asarray(loader.observer_pos, dtype=np.float32)
        dx = float(loader.boxsize) / rho.shape[0]
        if field_smoothing_scale is not None:
            if k == 0:
                fprint(
                    "  applying real-space Gaussian smoothing to the 3D "
                    "density and velocity fields with scale "
                    f"{field_smoothing_scale:g} "
                    "Mpc/h.")
            rho = apply_gaussian_smoothing(
                rho.astype(np.float32, copy=False),
                field_smoothing_scale, loader.boxsize, make_copy=True)

        if k == 0 and geometry == "sphere" and subcube_radius is not None:
            _warn_coarse_sphere_radius(
                subcube_radius, dx, "model.selection_integral_grid_radius")

        extract_radius = subcube_radius
        if (geometry == "sphere" and subcube_radius is not None):
            extract_radius = subcube_radius + 0.5 * np.sqrt(3.0) * dx

        if extract_radius is not None:
            rho_sub, obs_sub, slices = _extract_subcube(
                rho, obs, dx, extract_radius)
        else:
            rho_sub, obs_sub, slices = rho, obs, None
        del rho

        if log_r_3d is None:
            dx_ref = dx
            shape_ref = rho_sub.shape
            log_r_grid, log_dV = _volume_density_geometry(
                rho_sub.shape, obs_sub, dx)
            obs_sub_ref = obs_sub
            nsub = rho_sub.shape
            ax = [(np.arange(nsub[i], dtype=np.float32) + 0.5) * dx
                  for i in range(3)]
            disp_ref = [ax[0][:, None, None] - obs_sub[0],
                        ax[1][None, :, None] - obs_sub[1],
                        ax[2][None, None, :] - obs_sub[2]]
            r_sub_ref = np.sqrt(
                disp_ref[0]**2 + disp_ref[1]**2 + disp_ref[2]**2)
            r_sub_ref = np.maximum(r_sub_ref, 0.25 * dx)
            quad_ref = _h0_volume_quadrature_geometry(
                log_r_grid, disp_ref, r_sub_ref, dx, geometry,
                subcube_radius, supersample_factor, supersample_radius,
                store_rhat=load_velocity)
            voxel_mask_ref = quad_ref["voxel_mask"]
            log_r_3d = quad_ref["log_r_3d"]
            log_volume_weight_3d = quad_ref["log_volume_weight_3d"]
            if use_supersampling:
                n_base = int(np.sum(voxel_mask_ref))
                n_super = len(quad_ref["sup_flat"])
                n_out = len(log_r_3d)
                if supersample_target_dx is None:
                    target_msg = ""
                else:
                    target_msg = (
                        f", target_dx={supersample_target_dx:g} Mpc/h, "
                        f"native_dx={native_dx:g} Mpc/h, "
                        f"resolved_dx={dx / supersample_factor:g} Mpc/h")
                fprint(
                    "  H0 volume supersampling: "
                    f"{n_super:,}/{n_base:,} parent voxels intersect "
                    f"r < {supersample_radius:g} Mpc/h; "
                    "trilinear interpolation, "
                    f"{n_out:,} quadrature points total"
                    f"{target_msg}.")
        else:
            if rho_sub.shape != shape_ref:
                raise ValueError(
                    "All 3D volume fields must have the same sub-cube shape; "
                    f"got {shape_ref} and {rho_sub.shape}.")
            if not np.isclose(dx, dx_ref):
                raise ValueError(
                    "All 3D volume fields must have the same voxel size; "
                    f"got {dx_ref} and {dx} Mpc/h.")
            if not np.allclose(obs_sub, obs_sub_ref):
                raise ValueError(
                    "All 3D volume fields must have the same observer "
                    "position after sub-cube extraction.")

        rho_out = _h0_volume_apply_quadrature(rho_sub, quad_ref)
        rho_fields.append(rho_out)

        if load_velocity:
            v_rad_out = np.zeros(np.asarray(log_r_3d).shape,
                                 dtype=np.float32)
            if hasattr(loader, "load_velocity_component"):
                try:
                    for comp in range(3):
                        v_full = loader.load_velocity_component(comp)
                        if field_smoothing_scale is not None:
                            v_full = apply_gaussian_smoothing(
                                v_full.astype(np.float32, copy=False),
                                field_smoothing_scale, loader.boxsize,
                                make_copy=True)
                        if slices is not None:
                            v_comp = v_full[slices[0], slices[1], slices[2]]
                        else:
                            v_comp = v_full
                        del v_full
                        v_comp = _h0_volume_apply_quadrature(
                            v_comp, quad_ref)
                        v_rad_out += v_comp * quad_ref["rhat_fields"][comp]
                        del v_comp
                finally:
                    if hasattr(loader, "clear_velocity_cache"):
                        loader.clear_velocity_cache()
            else:
                v_full = loader.load_velocity()
                for comp in range(3):
                    v_comp = v_full[comp]
                    if field_smoothing_scale is not None:
                        v_comp = apply_gaussian_smoothing(
                            v_comp.astype(np.float32, copy=False),
                            field_smoothing_scale, loader.boxsize,
                            make_copy=True)
                    if slices is not None:
                        v_comp = v_comp[slices[0], slices[1], slices[2]]
                    v_comp = _h0_volume_apply_quadrature(v_comp, quad_ref)
                    v_rad_out += v_comp * quad_ref["rhat_fields"][comp]
                    del v_comp
                del v_full
            vrad_fields.append(v_rad_out)

        if not np.any(~voxel_mask_ref):
            msg = (f"  field {k} (nsim={int(nsim)}): cube {rho_sub.shape}, "
                   f"dx={dx:.4f} Mpc/h")
        else:
            msg = (f"  field {k} (nsim={int(nsim)}): sub-cube "
                   f"{rho_sub.shape}, {len(log_r_3d)} "
                   f"weighted volume points, dx={dx:.4f} Mpc/h")
        fprint(msg + ".")

    r_3d = jnp.asarray(np.exp(np.asarray(log_r_3d)).astype(np.float32))

    rhat_fields = {}
    if load_velocity:
        for i, label in enumerate(("rhat_x_3d", "rhat_y_3d", "rhat_z_3d")):
            rhat_fields[label] = quad_ref["rhat_fields"][i]

    if len(rho_fields) == 1:
        rho_fields = rho_fields[0][None, ...]
    else:
        rho_fields = np.stack(rho_fields)
    if load_velocity:
        if len(vrad_fields) == 1:
            vrad_fields = vrad_fields[0][None, ...]
        else:
            vrad_fields = np.stack(vrad_fields)
    else:
        vrad_fields = None

    if return_cache_fields:
        out = {
            "rho_3d_fields": rho_fields,
            "r_3d": r_3d,
            "log_dV_3d": log_dV,
            **_h0_volume_supersampling_cache_arrays(
                supersample_factor, supersample_radius),
        }
        if log_volume_weight_3d is not None:
            out["log_volume_weight_3d"] = log_volume_weight_3d
        if load_velocity:
            out["vrad_3d_fields"] = vrad_fields
            out.update(rhat_fields)
        return out

    runtime_arrays = {
        "rho_3d_fields": rho_fields,
        "r_3d": r_3d,
        "log_dV_3d": log_dV,
    }
    if log_volume_weight_3d is not None:
        runtime_arrays["log_volume_weight_3d"] = log_volume_weight_3d
    if load_velocity:
        runtime_arrays["vrad_3d_fields"] = vrad_fields
        runtime_arrays.update(rhat_fields)
    runtime_arrays, actual_fraction = _subsample_h0_volume_arrays(
        runtime_arrays, voxel_subsample_fraction, voxel_subsample_seed)
    if not np.isclose(actual_fraction, 1.0):
        fprint(
            f"  applying random voxel subsample f={actual_fraction:.6g} "
            f"({np.size(runtime_arrays['r_3d']):,} kept).")

    runtime_rhat_fields = None
    if load_velocity:
        runtime_rhat_fields = {
            label: runtime_arrays[label]
            for label in ("rhat_x_3d", "rhat_y_3d", "rhat_z_3d")
        }
    result = _h0_volume_runtime_result(
        runtime_arrays["rho_3d_fields"], runtime_arrays["r_3d"],
        runtime_arrays["log_dV_3d"], source_meta, mode, Om0, load_velocity,
        vrad_fields=runtime_arrays.get("vrad_3d_fields"),
        log_volume_weight_3d=runtime_arrays.get("log_volume_weight_3d"),
        rhat_fields=runtime_rhat_fields)
    if cache_enabled:
        for i, nsim in enumerate(field_indices):
            density_cache_arrays = {
                "rho_3d_fields": rho_fields[i:i + 1],
                "r_3d": r_3d,
                "log_dV_3d": log_dV,
                **supersampling_cache_arrays,
            }
            if log_volume_weight_3d is not None:
                density_cache_arrays["log_volume_weight_3d"] = (
                    log_volume_weight_3d)
            _write_field_cache(
                density_cache_paths[i],
                f"H0 3D volume density field {int(nsim)}",
                density_cache_arrays)
            if load_velocity:
                velocity_cache_arrays = {
                    "vrad_3d_fields": vrad_fields[i:i + 1],
                    "r_3d": r_3d,
                    "log_dV_3d": log_dV,
                    **supersampling_cache_arrays}
                velocity_cache_arrays.update(rhat_fields)
                _write_field_cache(
                    velocity_cache_paths[i],
                    f"H0 3D volume velocity field {int(nsim)}",
                    velocity_cache_arrays)

    return result


def _load_h0_volume_data_from_config(config, los_data_path, reconstruction,
                                     label, velocity_selections,
                                     field_indices=None):
    """Load shared 3D volume data for H0 normalizers."""
    which_sel = get_nested(config, "model/which_selection", None)
    which_run = get_nested(config, "model/which_run", None)
    needs_no_selection_volume = (
        which_sel is None
        and which_run in ("CH0", "EDD_TRGB", "EDD_TRGB_grouped"))
    if which_sel is None and not needs_no_selection_volume:
        return None
    use_recon = get_nested(config, "model/use_reconstruction", False)
    if not use_recon:
        return None
    if los_data_path is None or reconstruction is None:
        raise ValueError(
            f"{label} 3D volume normalizer requires host LOS data.")

    grid_radius = get_nested(
        config, "model/selection_integral_grid_radius", None)
    if grid_radius is None:
        raise ValueError(
            "3D selection integrals require explicit "
            "`model.selection_integral_grid_radius` in Mpc/h.")

    Om0 = get_nested(config, "model/Om", get_nested(config, "model/Om0", 0.3))
    galaxy_bias = get_nested(config, "model/which_bias", "linear")
    legacy_downsample = get_nested(
        config, "model/density_3d_downsample", None)
    if legacy_downsample not in (None, 1):
        raise ValueError(
            "`model.density_3d_downsample` has been replaced by "
            "`model.density_3d_subsample_fraction`; use a fraction in "
            f"(0, 1], got legacy downsample {legacy_downsample!r}.")
    voxel_subsample_fraction = _validate_voxel_subsample_fraction(
        get_nested(config, "model/density_3d_subsample_fraction", 1.0),
        "model.density_3d_subsample_fraction")
    voxel_subsample_seed = _validate_voxel_subsample_seed(
        get_nested(config, "model/density_3d_subsample_seed", 42),
        "model.density_3d_subsample_seed")
    supersample_factor, supersample_radius, supersample_target_dx = (
        _h0_volume_supersampling_from_config(config))
    field_smoothing_scale = field_smoothing_scale_from_config(config)
    geometry = get_nested(
        config, "model/selection_integral_geometry", "sphere")
    if geometry not in ("sphere", "cube"):
        raise ValueError(
            "`model.selection_integral_geometry` must be 'sphere' or 'cube'.")
    load_vel = which_sel in velocity_selections
    recon_main = get_nested(config, "io/reconstruction_main", {})
    field_kwargs = recon_main.get(reconstruction, {})
    if not field_kwargs:
        raise ValueError(
            f"No `io.reconstruction_main.{reconstruction}` configuration found "
            f"for {label} 3D volume normalizer.")
    cache_enabled = _field_cache_enabled_from_config(config)
    cache_dir = (
        _field_cache_dir_from_config(config)
        if cache_enabled else None)

    if field_indices is None:
        paths = los_data_path if isinstance(
            los_data_path, (list, tuple)) else [los_data_path]
        indices = []
        for path in paths:
            with File(path, "r") as f:
                if "field_indices" in f:
                    indices.extend(f["field_indices"][:])
                else:
                    indices.extend(np.arange(f["los_density"].shape[0]))
        field_indices = np.asarray(indices, dtype=np.int32)
    else:
        field_indices = np.asarray(field_indices, dtype=np.int32)

    fprint(f"loading {len(field_indices)} 3D density cube(s) for {label} "
           f"volume normalizer (geometry={geometry}, "
           f"radius={grid_radius} Mpc/h, "
           f"voxel_subsample_fraction={voxel_subsample_fraction:g}, "
           f"supersample_radius={supersample_radius:g} Mpc/h, "
           f"supersample_target_dx={supersample_target_dx}, "
           f"field_smoothing_scale={field_smoothing_scale}, "
           f"velocity={load_vel}).")
    if supersample_target_dx is not None and supersample_radius > 0.0:
        fprint(
            f"{label} 3D volume supersampling enabled: "
            f"target_dx={supersample_target_dx:g} Mpc/h for parent voxels "
            f"intersecting r < {supersample_radius:g} Mpc/h; "
            "trilinear interpolation.")
    elif supersample_factor > 1 and supersample_radius > 0.0:
        fprint(
            f"{label} 3D volume supersampling enabled for parent voxels "
            f"intersecting r < {supersample_radius:g} Mpc/h; "
            "trilinear interpolation.")
    else:
        fprint(f"{label} 3D volume supersampling disabled.")
    if cache_enabled:
        fprint(f"field cache enabled: `{cache_dir}`.")
    else:
        fprint("field cache disabled.")
    return _load_volume_data_for_H0(
        reconstruction, field_kwargs, field_indices,
        galaxy_bias, Om0,
        subcube_radius=grid_radius,
        voxel_subsample_fraction=voxel_subsample_fraction,
        voxel_subsample_seed=voxel_subsample_seed,
        load_velocity=load_vel,
        geometry=geometry,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
        supersample_factor=supersample_factor,
        supersample_radius=supersample_radius,
        supersample_target_dx=supersample_target_dx,
        field_smoothing_scale=field_smoothing_scale)


def _load_volume_density_3d(loader_name, loader_kwargs, downsample=1,
                            nsim=None, subcube_radius=None,
                            pad_subcube_boundary=False, cache_dir=None,
                            cache_enabled=True,
                            return_coordinate_frame=False,
                            field_smoothing_scale=None):
    """Load a 3D density cube and return NumPy arrays.

    `rho_3d` is dimensionless (1 + δ) on a regular Cartesian grid of side
    `loader.boxsize` (Mpc/h). `observer_pos` is the observer's position in
    box coordinates (Mpc/h). `dx` is the voxel side length (Mpc/h). If
    `subcube_radius` is set, the returned cube is cropped to that half-side
    around the observer in Mpc/h. When ``pad_subcube_boundary`` is True, the
    extraction radius is expanded by half a voxel diagonal so later spherical
    boundary weighting has all needed cells.

    `downsample` (int ≥ 1) keeps every Nth voxel along each axis (point
    subsampling); the voxel side `dx` is rescaled by N. Returns
    ``(rho_3d, observer_pos, dx)`` unless ``return_coordinate_frame`` is true,
    in which case the loader coordinate frame is appended.
    """
    if not isinstance(downsample, int) or downsample < 1:
        raise ValueError(
            f"`density_3d_downsample` must be a positive int, got {downsample!r}.")  # noqa
    field_smoothing_scale = validate_field_smoothing_scale(
        field_smoothing_scale)

    loader_kwargs = dict(loader_kwargs)
    if nsim is not None:
        loader_kwargs.setdefault("nsim", nsim)

    loader_cls = name2field_loader(loader_name)
    loader = loader_cls(**loader_kwargs)
    if cache_enabled:
        cache_payload = {
            "kind": "volume_field_data",
            "product": "pv_density_cube",
            "loader_name": loader_name,
            "loader_kwargs": _jsonable(loader_kwargs),
            "downsample": int(downsample),
            "nsim": None if nsim is None else int(nsim),
            "subcube_radius": subcube_radius,
            "pad_subcube_boundary": bool(pad_subcube_boundary),
            **field_smoothing_cache_payload(field_smoothing_scale),
            "source": _field_source_metadata(loader),
        }
        cache_path = _field_cache_path(
            cache_dir, _VOLUME_FIELD_CACHE_PREFIX, cache_payload)
        cached = _read_field_cache(
            cache_path, "PV 3D density", ["rho", "observer_pos", "dx"])
        if cached is not None:
            out = (
                cached["rho"].astype(np.float32, copy=False),
                cached["observer_pos"].astype(np.float32, copy=False),
                float(cached["dx"]))
            if return_coordinate_frame:
                return out + (getattr(loader, "coordinate_frame", "icrs"),)
            return out
        if (not field_allows_raw_product_reads(loader_name)
                and not _field_cache_warmup_active()):
            raise _h0_volume_missing_cache_error(
                "PV 3D density", cache_path)
        fprint("PV 3D density cache:")
        fprint(f"  path: `{cache_path}`")
        fprint("  status: missing/stale; building from raw field")
    else:
        cache_path = None

    rho = loader.load_density()
    norm = _density_unit_normalization(loader_name)
    if norm is not None:
        divisor, label, detail = norm
        fprint(f"  normalizing the {label} 3D density ({detail}).")
        rho = rho / divisor
    if rho.ndim != 3 or rho.shape[0] != rho.shape[1] or rho.shape[0] != rho.shape[2]:  # noqa
        raise ValueError(
            f"Volume density cube must be cubic 3D, got shape {rho.shape}.")

    dx = float(loader.boxsize) / rho.shape[0]
    if field_smoothing_scale is not None:
        if field_smoothing_scale <= dx:
            raise ValueError(
                f"`model.field_3d_smoothing_scale` must exceed the "
                f"native field voxel size {dx:g} Mpc/h; got "
                f"{field_smoothing_scale:g} Mpc/h.")
        fprint(
            "  applying real-space Gaussian smoothing to the PV 3D density "
            f"field with scale {field_smoothing_scale:g} Mpc/h.")
        rho = apply_gaussian_smoothing(
            rho.astype(np.float32, copy=False),
            field_smoothing_scale, loader.boxsize, make_copy=True)
    if downsample > 1:
        rho = rho[::downsample, ::downsample, ::downsample]
        dx *= downsample
        fprint(
            f"  downsampled by factor {downsample} -> shape {rho.shape}, "
            f"dx = {dx:.4f} Mpc/h.")

    obs = np.asarray(loader.observer_pos, dtype=np.float32)
    if subcube_radius is not None:
        extract_radius = subcube_radius
        if pad_subcube_boundary:
            extract_radius += 0.5 * np.sqrt(3.0) * dx
        rho, obs, _ = _extract_subcube(rho, obs, dx, extract_radius)

    rho = rho.astype(np.float32)
    _write_field_cache(
        cache_path, "PV 3D density",
        {"rho": rho, "observer_pos": obs, "dx": np.asarray(dx)})
    if return_coordinate_frame:
        return rho, obs, dx, getattr(loader, "coordinate_frame", "icrs")
    return rho, obs, dx


def _cached_pv_volume_density_result(cached):
    """Normalize a grouped PV density cache dictionary."""
    out = {
        "rho_fields": np.asarray(cached["rho_fields"], dtype=np.float32),
        "log_r_3d": _h0_log_radius_from_r(cached["r_3d"]),
        "log_dV_3d": float(np.asarray(cached["log_dV_3d"]).reshape(-1)[0]),
    }
    if "coordinate_frame" in cached:
        out["coordinate_frame"] = str(
            np.asarray(cached["coordinate_frame"]).item())
    else:
        out["coordinate_frame"] = "icrs"
    for key in (
            "log_volume_weight_3d",
            "rhat_x_3d", "rhat_y_3d", "rhat_z_3d"):
        if key in cached:
            out[key] = np.asarray(cached[key], dtype=np.float32)
    return out


def _prepare_pv_volume_density_arrays(
        fields, geometry="cube", radius=None, store_rhat_3d=False,
        voxel_subsample_fraction=1.0, voxel_subsample_seed=42):
    """Build the compact PV volume-normalizer arrays for caching."""
    if geometry not in ("cube", "sphere"):
        raise ValueError(
            f"`density_3d_geometry` must be 'cube' or 'sphere', "
            f"got {geometry!r}.")
    if geometry == "sphere" and radius is None:
        raise ValueError(
            "`pv_model.density_3d_radius` is required when "
            "`density_3d_geometry = 'sphere'`.")
    voxel_subsample_fraction = _validate_voxel_subsample_fraction(
        voxel_subsample_fraction, "pv_model.density_3d_subsample_fraction")
    voxel_subsample_seed = _validate_voxel_subsample_seed(
        voxel_subsample_seed, "pv_model.density_3d_subsample_seed")

    def _unpack_field(field):
        if len(field) == 3:
            rho_3d, obs_pos, dx = field
            coord_frame = "icrs"
        elif len(field) == 4:
            rho_3d, obs_pos, dx, coord_frame = field
        else:
            raise ValueError(
                "3D density fields must be "
                "(rho_3d, observer_pos, dx[, coordinate_frame]).")
        return rho_3d, obs_pos, dx, coord_frame

    field_iter = iter(fields)
    try:
        rho0, obs0, dx0, coord_frame0 = _unpack_field(next(field_iter))
    except StopIteration:
        raise ValueError("At least one 3D density field is required.")

    log_r_grid, log_dV = _volume_density_geometry(rho0.shape, obs0, dx0)
    rhat_fields = None
    if geometry == "sphere" or store_rhat_3d:
        if geometry == "sphere":
            _warn_coarse_sphere_radius(
                radius, dx0, "pv_model.density_3d_radius")
        nsub = rho0.shape
        ax = [(np.arange(nsub[i], dtype=np.float32) + 0.5) * dx0
              for i in range(3)]
        disp = [ax[0][:, None, None] - obs0[0],
                ax[1][None, :, None] - obs0[1],
                ax[2][None, None, :] - obs0[2]]
        r_sub = np.sqrt(disp[0]**2 + disp[1]**2 + disp[2]**2)
        r_sub = np.maximum(r_sub, 0.25 * dx0)
    if geometry == "sphere":
        voxel_weight = _sphere_voxel_weights(disp, radius, dx0)
        voxel_mask = voxel_weight > 0.0
        r_3d = np.exp(np.asarray(log_r_grid)[voxel_mask])
        log_volume_weight_3d = np.log(
            voxel_weight[voxel_mask]).astype(np.float32)
    else:
        voxel_mask = None
        r_3d = np.exp(np.asarray(log_r_grid))
        log_volume_weight_3d = None
    if store_rhat_3d:
        rhat_fields = {}
        for i, label in enumerate(
                ("rhat_x_3d", "rhat_y_3d", "rhat_z_3d")):
            rhat = (disp[i] / r_sub).astype(np.float32)
            if voxel_mask is not None:
                rhat = rhat[voxel_mask]
            rhat_fields[label] = rhat

    voxel_subsample_idx, actual_fraction = _choose_voxel_subsample_indices(
        int(np.size(r_3d)), voxel_subsample_fraction,
        voxel_subsample_seed)
    if voxel_subsample_idx is not None:
        r_3d = r_3d.reshape(-1)[voxel_subsample_idx]
        if log_volume_weight_3d is not None:
            log_volume_weight_3d = log_volume_weight_3d.reshape(-1)[
                voxel_subsample_idx]
        if rhat_fields is not None:
            for label in rhat_fields:
                rhat_fields[label] = rhat_fields[label].reshape(-1)[
                    voxel_subsample_idx]
        log_dV = log_dV - float(np.log(actual_fraction))
        fprint(
            f"  applying PV 3D normalizer voxel subsample "
            f"f={actual_fraction:.6g} ({len(voxel_subsample_idx):,} "
            "kept).")

    rho_fields = []
    fields_iter = chain(((rho0, obs0, dx0, coord_frame0),),
                        (_unpack_field(f) for f in field_iter))
    for rho_3d, obs_pos, dx, coord_frame in fields_iter:
        if rho_3d.shape != rho0.shape:
            raise ValueError(
                "All 3D density fields must have the same shape; got "
                f"{rho_3d.shape} and {rho0.shape}.")
        if not np.allclose(obs_pos, obs0) or not np.isclose(dx, dx0):
            raise ValueError(
                "All 3D density fields must share observer position and "
                "voxel size to reuse the volume geometry.")
        if coord_frame != coord_frame0:
            raise ValueError(
                "All 3D density fields must share coordinate frame; got "
                f"{coord_frame0!r} and {coord_frame!r}.")
        if voxel_mask is not None:
            rho_3d = rho_3d[voxel_mask]
        if voxel_subsample_idx is not None:
            rho_3d = rho_3d.reshape(-1)[voxel_subsample_idx]
        rho_fields.append(rho_3d.astype(np.float32))

    out = {
        "rho_fields": np.stack(rho_fields),
        "r_3d": r_3d.astype(np.float32),
        "log_dV_3d": np.asarray(log_dV),
        "observer_pos": np.asarray(obs0, dtype=np.float32),
        "dx": np.asarray(dx0),
        "coordinate_frame": np.asarray(coord_frame0),
    }
    if log_volume_weight_3d is not None:
        out["log_volume_weight_3d"] = log_volume_weight_3d
    if rhat_fields is not None:
        out.update(rhat_fields)
    return out


def _pv_mpi_placeholder(cache_meta):
    """Small stand-in returned by non-root PV cache-warmup ranks."""
    return {
        "rho_fields": np.ones((0, 0), dtype=np.float32),
        "r_3d": np.ones(0, dtype=np.float32),
        "log_dV_3d": 0.0,
        "coordinate_frame": "icrs",
        "mpi_density_shape": cache_meta["density_shape"],
    }


def _load_one_pv_volume_density_field(
        loader_name, loader_kwargs, k, nsim, downsample, subcube_radius,
        pad_subcube_boundary, field_smoothing_scale=None):
    loaded = _load_volume_density_3d(
        loader_name, loader_kwargs, downsample=downsample,
        nsim=int(nsim), subcube_radius=subcube_radius,
        pad_subcube_boundary=pad_subcube_boundary,
        cache_dir=None, cache_enabled=False,
        return_coordinate_frame=True,
        field_smoothing_scale=field_smoothing_scale)
    rho, obs, dx, coord_frame = loaded
    fprint(
        f"  field {k} (nsim={int(nsim)}): cube shape {rho.shape}, "
        f"dx = {dx:.4f} Mpc/h, "
        f"observer at {obs.tolist()} Mpc/h.")
    return rho, obs, dx, coord_frame


def _load_volume_density_3d_fields_mpi(
        comm, loader_name, loader_kwargs, field_indices, downsample,
        subcube_radius, pad_subcube_boundary, cache_path, geometry, radius,
        store_rhat_3d, voxel_subsample_fraction, voxel_subsample_seed,
        required, field_smoothing_scale=None, max_radius=None):
    """Build one grouped PV volume-density cache split over MPI ranks."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        fprint(f"MPI field-cache warmup: splitting {len(field_indices)} "
               f"PV field(s) over {size} rank(s).")
        cache_dir = os.path.dirname(cache_path[0])
        os.makedirs(cache_dir, exist_ok=True)
        part_dir = tempfile.mkdtemp(
            prefix=".tmp_pv_volume_mpi_",
            dir=cache_dir)
    else:
        part_dir = None
    part_dir = comm.bcast(part_dir, root=0)

    local_parts = []
    try:
        for k, nsim in enumerate(field_indices):
            if k % size != rank:
                continue
            fprint(f"rank {rank}: warming PV field {k} "
                   f"(nsim={int(nsim)}).")
            field = _load_one_pv_volume_density_field(
                loader_name, loader_kwargs, k, nsim, downsample,
                subcube_radius, pad_subcube_boundary,
                field_smoothing_scale=field_smoothing_scale)
            prepared = _prepare_pv_volume_density_arrays(
                [field], geometry=geometry, radius=radius,
                store_rhat_3d=store_rhat_3d,
                voxel_subsample_fraction=voxel_subsample_fraction,
                voxel_subsample_seed=voxel_subsample_seed)
            part_path = join(part_dir, f"field_{k:06d}_rank_{rank}.npz")
            _write_field_cache_mpi_part(part_path, k, prepared)
            local_parts.append({"field_order": k, "path": part_path})
            del field, prepared
        manifest = {
            "rank": rank, "parts": local_parts, "count": len(local_parts),
            "error": None,
        }
    except Exception as exc:
        manifest = {
            "rank": rank, "parts": [], "count": 0, "error": repr(exc),
        }

    manifests = comm.gather(manifest, root=0)
    error = None
    cache_meta = None
    if rank == 0:
        try:
            failures = [
                f"rank {item['rank']}: {item['error']}"
                for item in manifests
                if item["error"] is not None
            ]
            if failures:
                raise RuntimeError("; ".join(failures))

            parts = []
            for item in manifests:
                parts.extend(item["parts"])
            parts.sort(key=lambda item: item["field_order"])
            if len(parts) != len(field_indices):
                raise RuntimeError(
                    f"MPI PV field-cache warmup produced {len(parts)} "
                    f"field(s), expected {len(field_indices)}.")

            first_order, first = _read_field_cache_mpi_part(parts[0]["path"])
            if first_order != parts[0]["field_order"]:
                raise RuntimeError(
                    "MPI PV field-cache warmup found inconsistent field "
                    f"order in `{parts[0]['path']}`.")

            density_shape = (
                (len(field_indices),) + first["rho_fields"].shape[1:])
            cache_meta = {"density_shape": density_shape}

            invariant_keys = [
                "r_3d", "log_volume_weight_3d",
                "rhat_x_3d", "rhat_y_3d", "rhat_z_3d",
                "observer_pos", "coordinate_frame",
            ]
            for out_idx, part in enumerate(parts):
                if out_idx == 0:
                    field_order, arrays = first_order, first
                else:
                    field_order, arrays = _read_field_cache_mpi_part(
                        part["path"])
                if field_order != part["field_order"]:
                    raise RuntimeError(
                        "MPI PV field-cache warmup found inconsistent field "
                        f"order in `{part['path']}`.")
                for label in invariant_keys:
                    if (label in arrays) != (label in first):
                        raise RuntimeError(
                            "MPI PV field-cache warmup found inconsistent "
                            f"`{label}` presence across fields.")
                    if label in arrays and not np.array_equal(
                            arrays[label], first[label]):
                        raise RuntimeError(
                            "MPI PV field-cache warmup found inconsistent "
                            f"`{label}` arrays across fields.")
                if not np.isclose(arrays["log_dV_3d"],
                                  first["log_dV_3d"]):
                    raise RuntimeError(
                        "MPI PV field-cache warmup found inconsistent "
                        "`log_dV_3d` values across fields.")
                if not np.isclose(arrays["dx"], first["dx"]):
                    raise RuntimeError(
                        "MPI PV field-cache warmup found inconsistent "
                        "`dx` values across fields.")
                cache_arrays = {
                    key: arrays[key]
                    for key in required
                    if key in arrays
                }
                if "coordinate_frame" in arrays:
                    cache_arrays["coordinate_frame"] = (
                        arrays["coordinate_frame"])
                _write_field_cache(
                    cache_path[out_idx],
                    f"PV 3D density field {int(field_indices[out_idx])}",
                    cache_arrays)
                if out_idx != 0:
                    del arrays
        except Exception as exc:
            error = repr(exc)
        finally:
            shutil.rmtree(part_dir, ignore_errors=True)

    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"MPI PV field-cache warmup failed: {error}")
    cache_meta = comm.bcast(cache_meta, root=0)

    if os.environ.get("CANDEL_FIELD_CACHE_MPI", "0") == "1":
        return _pv_mpi_placeholder(cache_meta)

    cache_root = os.path.dirname(os.path.dirname(cache_path[0]))
    cached = _read_pv_volume_cache_superset(
        cache_root, {
            "kind": "volume_field_data",
            "product": "pv_volume_density",
            "loader_name": loader_name,
            "loader_kwargs": _jsonable(loader_kwargs),
            "field_indices": _jsonable(field_indices),
            "downsample": int(downsample),
            "subcube_radius": subcube_radius,
            "max_radius": max_radius,
            "pad_subcube_boundary": bool(pad_subcube_boundary),
            "voxel_subsample_fraction": float(voxel_subsample_fraction),
            "voxel_subsample_seed": int(voxel_subsample_seed),
            "store_rhat_3d": bool(store_rhat_3d),
            **field_smoothing_cache_payload(field_smoothing_scale),
        }, "PV 3D density", required, field_indices,
        expected_r_3d=max_radius)
    if cached is None:
        raise RuntimeError("failed to read MPI-warmed PV 3D density cache.")
    return _cached_pv_volume_density_result(cached)


def _load_volume_density_3d_fields(
        loader_name, loader_kwargs, field_indices, downsample=1,
        subcube_radius=None, pad_subcube_boundary=False, cache_dir=None,
        cache_enabled=True, geometry="cube", radius=None,
        store_rhat_3d=False, voxel_subsample_fraction=1.0,
        voxel_subsample_seed=42, field_smoothing_scale=None):
    """Load and cache one grouped PV 3D density product.

    The cached ``rho_fields`` are already downsampled, cropped, masked to the
    requested geometry, and randomly voxel-subsampled according to the cache
    key. Runtime bias transforms remain outside the cache.
    """
    field_indices = np.asarray(field_indices, dtype=np.int32)
    if field_indices.size == 0:
        raise ValueError("At least one PV 3D density field is required.")
    field_smoothing_scale = validate_field_smoothing_scale(
        field_smoothing_scale)

    required = ["rho_fields", "r_3d", "log_dV_3d", "observer_pos", "dx"]
    if geometry == "sphere":
        required.append("log_volume_weight_3d")
    if store_rhat_3d:
        required.extend(("rhat_x_3d", "rhat_y_3d", "rhat_z_3d"))

    if cache_enabled:
        source_meta = []
        first_loader = None
        for i, nsim in enumerate(field_indices):
            kwargs = dict(loader_kwargs)
            kwargs["nsim"] = int(nsim)
            loader = name2field_loader(loader_name)(**kwargs)
            if i == 0:
                first_loader = loader
            source_meta.append(_field_source_metadata(loader))
        expected_max_r_3d = _expected_pv_volume_max_radius_from_loader(
            first_loader, downsample, subcube_radius,
            pad_subcube_boundary, geometry, radius,
            voxel_subsample_fraction, voxel_subsample_seed)

        cache_payload = {
            "kind": "volume_field_data",
            "product": "pv_volume_density",
            "loader_name": loader_name,
            "loader_kwargs": _jsonable(loader_kwargs),
            "field_indices": _jsonable(field_indices),
            "downsample": int(downsample),
            "subcube_radius": subcube_radius,
            "max_radius": expected_max_r_3d,
            "pad_subcube_boundary": bool(pad_subcube_boundary),
            "voxel_subsample_fraction": float(voxel_subsample_fraction),
            "voxel_subsample_seed": int(voxel_subsample_seed),
            "store_rhat_3d": bool(store_rhat_3d),
            **field_smoothing_cache_payload(field_smoothing_scale),
            "sources": source_meta,
        }
        cache_paths = [
            _field_cache_path(
                cache_dir, _VOLUME_FIELD_CACHE_PREFIX,
                {**cache_payload, "field_indices": [int(nsim)],
                 "sources": [source_meta[i]]})
            for i, nsim in enumerate(field_indices)
        ]
        cached = _read_pv_volume_cache_superset(
            cache_dir, cache_payload, "PV 3D density", required,
            field_indices, expected_r_3d=expected_max_r_3d)
        if cached is not None:
            return _cached_pv_volume_density_result(cached)
        if (not field_allows_raw_product_reads(loader_name)
                and not _field_cache_warmup_active()):
            raise _h0_volume_missing_cache_error(
                "PV 3D volume density", cache_paths[:5])
        fprint("PV 3D density cache:")
        fprint(f"  files: {len(field_indices)} per-field file(s)")
        fprint(f"  directory: `{cache_dir}`")
        fprint("  status: missing/stale; building from raw fields")
        mpi_comm = _field_cache_mpi_comm()
        if mpi_comm is not None and len(field_indices) > 1:
            return _load_volume_density_3d_fields_mpi(
                mpi_comm, loader_name, loader_kwargs, field_indices,
                downsample, subcube_radius, pad_subcube_boundary, cache_paths,
                geometry, radius, store_rhat_3d, voxel_subsample_fraction,
                voxel_subsample_seed, required,
                field_smoothing_scale=field_smoothing_scale,
                max_radius=expected_max_r_3d)
    else:
        cache_paths = None

    fields = []
    obs_ref = None
    dx_ref = None
    coord_ref = None
    for k, nsim in enumerate(field_indices):
        loaded = _load_volume_density_3d(
            loader_name, loader_kwargs, downsample=downsample,
            nsim=int(nsim), subcube_radius=subcube_radius,
            pad_subcube_boundary=pad_subcube_boundary,
            cache_dir=cache_dir, cache_enabled=False,
            return_coordinate_frame=True,
            field_smoothing_scale=field_smoothing_scale)
        rho, obs, dx, coord_frame = loaded
        if obs_ref is None:
            obs_ref = obs
            dx_ref = dx
            coord_ref = coord_frame
        else:
            if rho.shape != fields[0][0].shape:
                raise ValueError(
                    "All PV 3D density fields must have the same shape; "
                    f"got {fields[0][0].shape} and {rho.shape}.")
            if not np.allclose(obs, obs_ref) or not np.isclose(dx, dx_ref):
                raise ValueError(
                    "All PV 3D density fields must share observer position "
                    "and voxel size.")
            if coord_frame != coord_ref:
                raise ValueError(
                    "All PV 3D density fields must share coordinate frame; "
                    f"got {coord_ref!r} and {coord_frame!r}.")
        fields.append((rho, obs, dx, coord_frame))
        fprint(
            f"  field {k} (nsim={int(nsim)}): cube shape {rho.shape}, "
            f"dx = {dx:.4f} Mpc/h, "
            f"observer at {obs.tolist()} Mpc/h.")

    prepared = _prepare_pv_volume_density_arrays(
        fields, geometry=geometry, radius=radius,
        store_rhat_3d=store_rhat_3d,
        voxel_subsample_fraction=voxel_subsample_fraction,
        voxel_subsample_seed=voxel_subsample_seed)
    if cache_enabled:
        for i, nsim in enumerate(field_indices):
            field_cache = dict(prepared)
            field_cache["rho_fields"] = prepared["rho_fields"][i:i + 1]
            _write_field_cache(
                cache_paths[i], f"PV 3D density field {int(nsim)}",
                field_cache)
    return _cached_pv_volume_density_result(prepared)


def _volume_density_geometry(shape, observer_pos, dx):
    """Return log voxel radius and log voxel volume for one grid geometry."""
    x = (np.arange(shape[0], dtype=np.float32) + 0.5) * dx - observer_pos[0]
    y = (np.arange(shape[1], dtype=np.float32) + 0.5) * dx - observer_pos[1]
    z = (np.arange(shape[2], dtype=np.float32) + 0.5) * dx - observer_pos[2]
    r_3d = np.sqrt(x[:, None, None]**2 + y[None, :, None]**2
                   + z[None, None, :]**2)

    return (
        jnp.asarray(np.log(np.maximum(r_3d, 0.25 * dx)).astype(np.float32)),
        float(3.0 * np.log(dx)),
    )


def _volume_density_mode(galaxy_bias):
    """Minimal density representation needed by the 3D bias normalizer."""
    if galaxy_bias in ("powerlaw", "double_powerlaw", "manticore_stdp"):
        return "log_rho"
    return "delta"
