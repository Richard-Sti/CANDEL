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
import hashlib
import json
import os
import tempfile
from itertools import chain
from os.path import abspath, exists, isabs, join

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from h5py import File
from jax import core as jcore
from jax import numpy as jnp
from jax.nn import one_hot
from scipy import linalg
from scipy.linalg import cholesky

from ..cosmo.cosmography import Redshift2Distance
from ..field.loader import name2field_loader
from ..model.integration import simpson_log_weights
from ..model.interp import LOSInterpolator
from ..util import (SPEED_OF_LIGHT, fprint, fsection, get_nested,
                    get_root_data, load_config, radec_to_cartesian,
                    radec_to_galactic)
from .dust import read_dustmap

# Hard cap on the per-axis voxel count — keeps the volume sum cheap and forces
# the user to coarsen larger native grids.
_VOLUME_DENSITY_NGRID_MAX = 257
_FIELD_CACHE_VERSION = 1
_SPHERE_RADIUS_DX_WARN_MIN = 15.0

###############################################################################
#                            Helper functions                                 #
###############################################################################


def _jsonable(value):
    """Convert loader/config values to stable JSON-compatible data."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _field_source_metadata(loader):
    """Metadata that invalidates cached field products when inputs change."""
    state = {}
    paths = []

    def _collect_paths(value):
        if isinstance(value, str):
            if exists(value):
                paths.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect_paths(item)
        elif isinstance(value, dict):
            for item in value.values():
                _collect_paths(item)

    for key, value in vars(loader).items():
        state[key] = _jsonable(value)
        _collect_paths(value)

    state["boxsize"] = _jsonable(getattr(loader, "boxsize", None))
    state["coordinate_frame"] = getattr(loader, "coordinate_frame", None)
    state["observer_pos"] = _jsonable(np.asarray(loader.observer_pos))

    source_stats = []
    for path in sorted(set(paths)):
        try:
            st = os.stat(path)
        except OSError:
            continue
        stat = {
            "path": abspath(path),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "is_dir": os.path.isdir(path),
        }
        source_stats.append(stat)
        if stat["is_dir"]:
            try:
                entries = sorted(os.listdir(path))
            except OSError:
                entries = []
            for entry in entries:
                entry_path = join(path, entry)
                try:
                    entry_st = os.stat(entry_path)
                except OSError:
                    continue
                source_stats.append({
                    "path": abspath(entry_path),
                    "size": entry_st.st_size,
                    "mtime_ns": entry_st.st_mtime_ns,
                    "is_dir": os.path.isdir(entry_path),
                })

    return {
        "loader_class": type(loader).__name__,
        "state": state,
        "sources": source_stats,
    }


def _field_cache_enabled_from_config(config=None, model_config=None):
    """Return whether field-derived arrays should be disk-cached."""
    if model_config is not None and "field_cache_enabled" in model_config:
        return bool(model_config["field_cache_enabled"])
    if model_config is not None and "density_3d_cache_enabled" in model_config:
        return bool(model_config["density_3d_cache_enabled"])
    if config is None:
        return False

    for key in (
            "model/field_cache_enabled",
            "model/density_3d_cache_enabled",
            "pv_model/field_cache_enabled",
            "pv_model/density_3d_cache_enabled",
            "io/field_cache_enabled"):
        value = get_nested(config, key, None)
        if value is not None:
            return bool(value)
    return True


def _field_cache_dir_from_config(config=None, model_config=None):
    """Resolve the directory for cached reduced field products."""
    cache_dir = None
    if model_config is not None:
        cache_dir = model_config.get(
            "field_cache_dir",
            model_config.get("density_3d_cache_dir", None))
    if cache_dir is None and config is not None:
        for key in (
                "model/field_cache_dir",
                "model/density_3d_cache_dir",
                "pv_model/field_cache_dir",
                "pv_model/density_3d_cache_dir",
                "io/field_cache_dir"):
            cache_dir = get_nested(config, key, None)
            if cache_dir is not None:
                break
    if cache_dir is None:
        if config is None:
            return None
        cache_dir = join(get_root_data(config), "field_cache")
    elif config is not None and not isabs(cache_dir):
        cache_dir = join(config.get("root_main", get_root_data(config)),
                         cache_dir)
    return abspath(cache_dir)


def _field_cache_path(cache_dir, prefix, payload):
    """Return the cache path for a stable JSON payload."""
    if cache_dir is None:
        return None
    payload = _jsonable({"version": _FIELD_CACHE_VERSION, **payload})
    key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return join(cache_dir, prefix, f"{digest}.npz")


def _read_field_cache(cache_path, label, required_keys):
    """Load an `.npz` cache file if present and complete."""
    if cache_path is None or not exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as f:
            missing = [key for key in required_keys if key not in f.files]
            if missing:
                fprint(f"ignoring incomplete {label} cache `{cache_path}` "
                       f"(missing {missing}).")
                return None
            out = {key: f[key] for key in f.files}
    except Exception as exc:
        fprint(f"ignoring unreadable {label} cache `{cache_path}`: {exc}")
        return None
    fprint(f"loaded {label} cache from `{cache_path}`.")
    return out


def _write_field_cache(cache_path, label, arrays):
    """Atomically write an `.npz` cache file."""
    if cache_path is None:
        return
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_", suffix=".npz", dir=cache_dir)
    os.close(fd)
    try:
        np.savez(
            tmp_path, **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        fprint(f"could not write {label} cache `{cache_path}`: {exc}")
        return
    fprint(f"wrote {label} cache to `{cache_path}`.")


def _zcmb_blat_mask(zcmb, RA, dec, zcmb_min=None, zcmb_max=None, b_min=None):
    """Build a boolean mask for redshift and galactic latitude cuts."""
    mask = np.ones(len(zcmb), dtype=bool)
    if zcmb_min is not None:
        mask &= zcmb > zcmb_min
    if zcmb_max is not None:
        mask &= zcmb < zcmb_max
    if b_min is not None:
        b = radec_to_galactic(RA, dec)[1]
        mask &= np.abs(b) > b_min
    return mask


def _density_unit_normalization(source):
    """Return the raw-density divisor needed to get dimensionless 1 + delta."""
    source_str = str(source)
    source_lower = source_str.lower()
    source_upper = source_str.upper()

    if "manticore" in source_lower:
        return 0.306 * 275.4, "Manticore", "Om = 0.306"
    if source_upper == "CB1" or "_CB1" in source_upper:
        return 0.307 * 275.4, "CB1", "Om = 0.307"
    if source_upper == "CB2" or "_CB2" in source_upper:
        return 0.3111 * 275.4, "CB2", "Om = 0.3111"
    if source_upper == "HAMLET_V1" or "HAMLET_V1" in source_upper:
        return 0.3 * 275.4, "HAMLET_V1", "Om = 0.3"
    return None


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
    """Warn when the spherical boundary is poorly resolved by the voxel grid."""
    radius_over_dx = float(radius) / float(dx)
    if radius_over_dx < _SPHERE_RADIUS_DX_WARN_MIN:
        fprint(
            f"warning: `{label}` spans only {radius_over_dx:.1f} voxels; "
            "sphere boundary weights are approximate, so consider increasing "
            "the radius or decreasing `density_3d_downsample`.")


def _precompute_cosmo_3d(log_r_3d, Om0):
    """Precompute h=1 distance modulus and cosmological redshift on 3D grid.

    At runtime: mu(r, h) = mu_at_h1 - 5*log10(h), z_cosmo is h-independent.
    Both depend only on r in Mpc/h, which is fixed by the grid geometry.
    """
    from ..cosmo.cosmography import Distance2Distmod, Distance2Redshift

    r_flat = jnp.exp(jnp.asarray(log_r_3d).ravel())
    mu_flat = Distance2Distmod(Om0=Om0)(r_flat, h=1.0)
    z_flat = Distance2Redshift(Om0=Om0)(r_flat, h=1.0)

    shape = log_r_3d.shape
    return mu_flat.reshape(shape), z_flat.reshape(shape)


def _load_volume_data_for_H0(
        field_name, field_kwargs, field_indices, galaxy_bias, Om0,
        subcube_radius=None, downsample=1, load_velocity=False,
        geometry="sphere", cache_dir=None, cache_enabled=True):
    """Load 3D voxel data for H0 selection integrals.

    Returns a dict to be merged into an H0-model data dict.
    """
    if geometry not in ("sphere", "cube"):
        raise ValueError(
            f"`selection_integral_geometry` must be 'sphere' or 'cube', "
            f"got {geometry!r}.")
    mode = _volume_density_mode(galaxy_bias)
    density_fields = []
    vrad_fields = [] if load_velocity else None
    log_r_3d = None
    coord_frame = None
    obs_sub_ref = None
    disp_ref = None
    r_sub_ref = None
    dx_ref = None
    shape_ref = None
    voxel_mask_ref = None
    log_volume_weight_3d = None
    loaders = []
    source_meta = []

    for nsim in field_indices:
        kwargs = dict(field_kwargs)
        kwargs["nsim"] = int(nsim)
        loader = name2field_loader(field_name)(**kwargs)
        loaders.append(loader)
        source_meta.append(_field_source_metadata(loader))

    if cache_enabled:
        cache_payload = {
            "kind": "h0_volume_data",
            "field_name": field_name,
            "field_kwargs": _jsonable(field_kwargs),
            "field_indices": _jsonable(np.asarray(field_indices)),
            "galaxy_bias": galaxy_bias,
            "density_mode": mode,
            "Om0": float(Om0),
            "subcube_radius": subcube_radius,
            "downsample": int(downsample),
            "load_velocity": bool(load_velocity),
            "geometry": geometry,
            "sources": source_meta,
        }
        cache_path = _field_cache_path(
            cache_dir, "h0_volume_data", cache_payload)
        required = [
            "density_3d_fields", "log_r_3d", "log_dV_3d",
            "mu_at_h1_3d", "zcosmo_3d"]
        if geometry == "sphere" and subcube_radius is not None:
            required.append("log_volume_weight_3d")
        if load_velocity:
            required.extend([
                "vrad_3d_fields", "rhat_x_3d", "rhat_y_3d", "rhat_z_3d"])
        cached = _read_field_cache(
            cache_path, "H0 3D volume data", required)
        if cached is not None:
            coord_frame = source_meta[0]["state"]["coordinate_frame"]
            result = {
                "density_3d_fields": jnp.asarray(
                    cached["density_3d_fields"]),
                "log_r_3d": jnp.asarray(cached["log_r_3d"]),
                "log_dV_3d": float(cached["log_dV_3d"]),
                "mu_at_h1_3d": jnp.asarray(cached["mu_at_h1_3d"]),
                "zcosmo_3d": jnp.asarray(cached["zcosmo_3d"]),
                "density_3d_mode": mode,
                "volume_density_batch_size": 1,
                "coordinate_frame_3d": coord_frame,
            }
            if "log_volume_weight_3d" in cached:
                result["log_volume_weight_3d"] = jnp.asarray(
                    cached["log_volume_weight_3d"])
            if load_velocity:
                result["vrad_3d_fields"] = jnp.asarray(
                    cached["vrad_3d_fields"])
                for label in ("rhat_x_3d", "rhat_y_3d", "rhat_z_3d"):
                    result[label] = jnp.asarray(cached[label])
            return result
        fprint("H0 3D volume data cache miss; loading reconstruction "
               f"fields and will cache at `{cache_path}`.")
    else:
        cache_path = None

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

        if downsample > 1:
            rho = rho[::downsample, ::downsample, ::downsample]
            dx *= downsample

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

        if any(n > _VOLUME_DENSITY_NGRID_MAX for n in rho_sub.shape):
            raise ValueError(
                f"Volume density grid {rho_sub.shape} exceeds the per-axis "
                f"cap of {_VOLUME_DENSITY_NGRID_MAX}; set "
                "`selection_integral_grid_radius` or increase "
                "`density_3d_downsample`.")

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
            if geometry == "sphere" and subcube_radius is not None:
                voxel_weight = _sphere_voxel_weights(
                    disp_ref, subcube_radius, dx)
                voxel_mask_ref = voxel_weight > 0.0
                log_r_3d = log_r_grid[voxel_mask_ref]
                log_volume_weight_3d = jnp.asarray(
                    np.log(voxel_weight[voxel_mask_ref]).astype(np.float32))
            else:
                voxel_mask_ref = None
                log_r_3d = log_r_grid
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

        if mode == "log_rho":
            density = np.log(rho_sub).astype(np.float32)
        else:
            density = (rho_sub - 1.0).astype(np.float32)
        if voxel_mask_ref is None:
            density_fields.append(density)
        else:
            density_fields.append(density[voxel_mask_ref])

        if load_velocity:
            v_rad = np.zeros(rho_sub.shape, dtype=np.float32)
            if hasattr(loader, "load_velocity_component"):
                for comp in range(3):
                    v_full = loader.load_velocity_component(comp)
                    if downsample > 1:
                        v_full = v_full[::downsample, ::downsample,
                                        ::downsample]
                    if slices is not None:
                        v_comp = v_full[slices[0], slices[1], slices[2]]
                    else:
                        v_comp = v_full
                    del v_full
                    v_rad += v_comp * disp_ref[comp] / r_sub_ref
                    del v_comp
            else:
                v_full = loader.load_velocity()
                for comp in range(3):
                    v_comp = v_full[comp]
                    if downsample > 1:
                        v_comp = v_comp[::downsample, ::downsample,
                                        ::downsample]
                    if slices is not None:
                        v_comp = v_comp[slices[0], slices[1], slices[2]]
                    v_rad += v_comp * disp_ref[comp] / r_sub_ref
                del v_full
            if voxel_mask_ref is None:
                vrad_fields.append(v_rad)
            else:
                vrad_fields.append(v_rad[voxel_mask_ref])

        if voxel_mask_ref is None:
            fprint(f"  field {k} (nsim={int(nsim)}): cube {rho_sub.shape}, "
                   f"dx={dx:.4f} Mpc/h.")
        else:
            fprint(f"  field {k} (nsim={int(nsim)}): sub-cube "
                   f"{rho_sub.shape}, {int(np.sum(voxel_mask_ref))} "
                   f"weighted spherical voxels, dx={dx:.4f} Mpc/h.")

    mu_at_h1_3d, zcosmo_3d = _precompute_cosmo_3d(log_r_3d, Om0)

    result = {
        "density_3d_fields": jnp.asarray(np.stack(density_fields)),
        "log_r_3d": log_r_3d,
        "log_dV_3d": log_dV,
        "mu_at_h1_3d": mu_at_h1_3d,
        "zcosmo_3d": zcosmo_3d,
        "density_3d_mode": mode,
        "volume_density_batch_size": 1,
        "coordinate_frame_3d": coord_frame,
    }
    if log_volume_weight_3d is not None:
        result["log_volume_weight_3d"] = log_volume_weight_3d

    if load_velocity:
        result["vrad_3d_fields"] = jnp.asarray(np.stack(vrad_fields))
        for i, label in enumerate(("rhat_x_3d", "rhat_y_3d", "rhat_z_3d")):
            rhat = (disp_ref[i] / r_sub_ref).astype(np.float32)
            if voxel_mask_ref is not None:
                rhat = rhat[voxel_mask_ref]
            result[label] = jnp.asarray(rhat)

    if cache_enabled:
        cache_arrays = {
            key: value for key, value in result.items()
            if key not in (
                "density_3d_mode", "volume_density_batch_size",
                "coordinate_frame_3d")
        }
        _write_field_cache(cache_path, "H0 3D volume data", cache_arrays)

    return result


def _load_h0_volume_data_from_config(config, los_data_path, which_los,
                                     label, velocity_selections):
    """Load the shared 3D selection-integral data for H0 models."""
    which_sel = get_nested(config, "model/which_selection", None)
    if which_sel is None:
        return None
    if not get_nested(config, "model/use_reconstruction", False):
        return None
    if los_data_path is None or which_los is None:
        raise ValueError(
            f"{label} selection integral requires host LOS data.")

    grid_radius = get_nested(
        config, "model/selection_integral_grid_radius", None)
    if grid_radius is None:
        raise ValueError(
            "3D selection integrals require explicit "
            "`model.selection_integral_grid_radius` in Mpc/h.")

    Om0 = get_nested(config, "model/Om", get_nested(config, "model/Om0", 0.3))
    galaxy_bias = get_nested(config, "model/which_bias", "linear")
    downsample_3d = get_nested(config, "model/density_3d_downsample", 1)
    if not isinstance(downsample_3d, int) or downsample_3d < 1:
        raise ValueError(
            "`model.density_3d_downsample` must be a positive int, "
            f"got {downsample_3d!r}.")
    geometry = get_nested(
        config, "model/selection_integral_geometry", "sphere")
    if geometry not in ("sphere", "cube"):
        raise ValueError(
            "`model.selection_integral_geometry` must be 'sphere' or 'cube'.")
    load_vel = which_sel in velocity_selections
    recon_main = get_nested(config, "io/reconstruction_main", {})
    field_kwargs = recon_main.get(which_los, {})
    if not field_kwargs:
        raise ValueError(
            f"No `io.reconstruction_main.{which_los}` configuration found "
            f"for {label} 3D selection integral.")
    cache_enabled = _field_cache_enabled_from_config(config)
    cache_dir = (
        _field_cache_dir_from_config(config)
        if cache_enabled else None)

    with File(los_data_path, "r") as f:
        if "field_indices" in f:
            field_indices = f["field_indices"][:]
        else:
            field_indices = np.arange(f["los_density"].shape[0])

    fprint(f"loading {len(field_indices)} 3D density cube(s) for {label} "
           f"selection integral (geometry={geometry}, "
           f"radius={grid_radius} Mpc/h, "
           f"downsample={downsample_3d}, velocity={load_vel}).")
    if cache_enabled:
        fprint(f"field cache enabled: `{cache_dir}`.")
    else:
        fprint("field cache disabled.")
    return _load_volume_data_for_H0(
        which_los, field_kwargs, field_indices,
        galaxy_bias, Om0,
        subcube_radius=grid_radius,
        downsample=downsample_3d,
        load_velocity=load_vel,
        geometry=geometry,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled)


def _load_volume_density_3d(loader_name, loader_kwargs, downsample=1,
                            nsim=None, subcube_radius=None,
                            pad_subcube_boundary=False, cache_dir=None,
                            cache_enabled=True):
    """Load a 3D density cube and return (rho_3d, observer_pos, dx) as NumPy.

    `rho_3d` is dimensionless (1 + δ) on a regular Cartesian grid of side
    `loader.boxsize` (Mpc/h). `observer_pos` is the observer's position in
    box coordinates (Mpc/h). `dx` is the voxel side length (Mpc/h). If
    `subcube_radius` is set, the returned cube is cropped to that half-side
    around the observer in Mpc/h.

    `downsample` (int ≥ 1) keeps every Nth voxel along each axis (point
    subsampling); the voxel side `dx` is rescaled by N. The voxel cap is
    enforced after downsampling so a finely-sampled native cube can be
    coarsened to fit.
    """
    if not isinstance(downsample, int) or downsample < 1:
        raise ValueError(
            f"`density_3d_downsample` must be a positive int, got {downsample!r}.")  # noqa

    loader_kwargs = dict(loader_kwargs)
    if nsim is not None:
        loader_kwargs.setdefault("nsim", nsim)

    loader_cls = name2field_loader(loader_name)
    loader = loader_cls(**loader_kwargs)
    if cache_enabled:
        cache_payload = {
            "kind": "pv_volume_density_3d",
            "loader_name": loader_name,
            "loader_kwargs": _jsonable(loader_kwargs),
            "downsample": int(downsample),
            "nsim": None if nsim is None else int(nsim),
            "subcube_radius": subcube_radius,
            "pad_subcube_boundary": bool(pad_subcube_boundary),
            "source": _field_source_metadata(loader),
        }
        cache_path = _field_cache_path(
            cache_dir, "pv_volume_density_3d", cache_payload)
        cached = _read_field_cache(
            cache_path, "PV 3D density", ["rho", "observer_pos", "dx"])
        if cached is not None:
            return (
                cached["rho"].astype(np.float32, copy=False),
                cached["observer_pos"].astype(np.float32, copy=False),
                float(cached["dx"]))
        fprint("PV 3D density cache miss; loading reconstruction field "
               f"and will cache at `{cache_path}`.")
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

    if any(n > _VOLUME_DENSITY_NGRID_MAX for n in rho.shape):
        raise ValueError(
            f"Volume density grid {rho.shape} exceeds the per-axis cap of "
            f"{_VOLUME_DENSITY_NGRID_MAX}; increase `density_3d_downsample`.")
    rho = rho.astype(np.float32)
    _write_field_cache(
        cache_path, "PV 3D density",
        {"rho": rho, "observer_pos": obs, "dx": np.asarray(dx)})
    return rho, obs, dx


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
    if galaxy_bias in ("powerlaw", "double_powerlaw"):
        return "log_rho"
    return "delta"


def _filter_data(data, mask, los_data_path=None):
    """Apply boolean mask to data arrays, report counts, and load LOS."""
    n_total = len(mask)
    n_kept = int(np.sum(mask))
    fprint(f"removed {n_total - n_kept} objects, thus {n_kept} remain.")
    for k in data:
        if isinstance(data[k], np.ndarray):
            data[k] = data[k][mask]
    if los_data_path:
        data = load_los(los_data_path, data, mask=mask)
    return data


def _compute_r_grid(r_limits, dr, data, Om=0.3):
    """Compute the radial grid for Malmquist bias integration."""
    if isinstance(r_limits, str) and r_limits.startswith("auto"):
        if "_" in r_limits:
            h_auto = float(r_limits.split("_")[1])
        else:
            h_auto = 1.0

        if "czcmb" in data:
            cz_obs = data["czcmb"]
        elif "zcmb" in data:
            cz_obs = data["zcmb"] * SPEED_OF_LIGHT
        else:
            raise KeyError("Data must contain 'czcmb' or 'zcmb'.")

        cz_obs_lim = [float(np.min(cz_obs)), float(np.max(cz_obs))]
        cz_obs_lim[0] = max(cz_obs_lim[0], 50.0)
        redshift2distance = Redshift2Distance(Om0=Om)
        r_from_cz = redshift2distance(
            np.array(cz_obs_lim), h=h_auto, is_velocity=True)
        r_min_raw = float(r_from_cz[0])
        r_max_raw = float(r_from_cz[1])
        buffer_low = max(r_min_raw * 0.25, 15.0)
        buffer_high = max(r_max_raw * 0.25, 15.0)
        rmin = max(r_min_raw - buffer_low, 0.01)
        rmax = r_max_raw + buffer_high
        fprint(f"auto r_limits_malmquist (h={h_auto}): [{rmin:.1f}, "
               f"{rmax:.1f}] Mpc "
               f"(buffer: -{buffer_low:.1f}, +{buffer_high:.1f} Mpc)")
    else:
        rmin, rmax = r_limits
        fprint(f"setting the LOS radial grid from {rmin} to {rmax} Mpc.")

    num_points = int(round((rmax - rmin) / dr)) + 1
    # Simpson's rule requires an odd number of points.
    if num_points % 2 == 0:
        num_points += 1
    dr_eff = (rmax - rmin) / (num_points - 1) if num_points > 1 else 0.0
    fprint(f"r-grid: n_r={num_points}, dr={dr_eff:.2f} Mpc")

    return np.linspace(rmin, rmax, num_points)


def effective_rank_entropy(C):
    """
    Compute the entropy-based effective rank (Shannon effective rank) of C.

    https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf
    """
    w = np.linalg.eigvalsh(C)
    # Remove negative eigenvalues (numerical artefacts)
    w = w[w > 0]
    p = w / np.sum(w)
    p_nonzero = p[p > 0]
    return np.exp(-np.sum(p_nonzero * np.log(p_nonzero)))


def precompute_pixel_projection(rhat_data, nside, sigma_deg=None):
    """
    Precompute the pixel projection matrix for a given set of LOS vectors.
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    rhat_pix = np.stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)], axis=1)

    d = rhat_data @ rhat_pix.T  # radial projection factors

    if sigma_deg is None:
        # One-hot on nearest pixel (max dot == min angle)
        p_max = np.argmax(d, axis=1)
        w = one_hot(p_max, rhat_pix.shape[0], dtype=rhat_data.dtype)
    else:
        raise NotImplementedError("Gaussian smoothing is not implemented")

    return w * d


###############################################################################
#                             Data frames                                     #
###############################################################################


def load_PV_dataframes(config_path):
    """Loads PV dataframes from the given configuration file."""
    config = load_config(config_path)

    if config["pv_model"]["kind"].startswith("precomputed_los_"):
        los_reconstruction = config["pv_model"]["kind"].replace("precomputed_los_", "")  # noqa
    else:
        los_reconstruction = None

    config_io = config["io"]
    config_pv_model = config["pv_model"]
    names = config_io.pop("catalogue_name")
    if isinstance(names, str):
        names = [names]

    dfs = []
    fsection("Data")
    fprint(f"loading {len(names)} PV dataframes: {names}")
    multi = len(names) > 1
    for name in names:
        if multi:
            fprint(f"--- {name} ---")
        is_mock = name.startswith("CF4_mock")
        if is_mock:
            kwargs = config_io["CF4_mock"].copy()
        else:
            kwargs = config_io[name].copy()

        try_pop_los = is_mock and los_reconstruction is None
        if los_reconstruction is not None and not is_mock:
            kwargs["los_data_path"] = kwargs.pop("los_file").replace(
                "<X>", los_reconstruction)
            fprint(
                f"loading existing LOS data from {kwargs['los_data_path']}.")

        recon_kwargs = None
        if los_reconstruction is not None:
            recon_main = config_io.get("reconstruction_main", {})
            recon_kwargs = recon_main.get(los_reconstruction, None)
        field_cache_enabled = _field_cache_enabled_from_config(
            config, config_pv_model)
        field_cache_dir = (
            _field_cache_dir_from_config(config, config_pv_model)
            if field_cache_enabled else None)
        if los_reconstruction is not None:
            if field_cache_enabled:
                fprint(f"field cache enabled: `{field_cache_dir}`.")
            else:
                fprint("field cache disabled.")

        df = PVDataFrame.from_config_dict(
            kwargs, name, try_pop_los=try_pop_los,
            config_pv_model=config_pv_model,
            reconstruction_kwargs=recon_kwargs,
            reconstruction_name=los_reconstruction,
            field_cache_dir=field_cache_dir,
            field_cache_enabled=field_cache_enabled)
        dfs.append(df)

    if len(dfs) == 1:
        return dfs[0]

    return dfs


class PVDataFrame:
    """Lightweight container for PV data."""
    add_eta_truncation = False

    def __init__(self, data, los_radial_decay_scale=5):
        # Convert numeric arrays to JAX, skip string arrays
        self.data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.str_):
                continue
            self.data[k] = jnp.asarray(v)
        self.name = None

        if "los_velocity" in self.data:
            self.has_precomputed_los = True
            self.num_fields = self.data["los_delta"].shape[0]
            fprint(f"marginalising over {self.num_fields} field realisations.")

            kwargs = {"r0_decay_scale": los_radial_decay_scale}
            self.f_los_delta = LOSInterpolator(
                self.data["los_r"], self.data["los_delta"], **kwargs)
            self.f_los_log_density = LOSInterpolator(
                self.data["los_r"], jnp.log(self.data["los_density"]),
                **kwargs)
            self.f_los_velocity = LOSInterpolator(
                self.data["los_r"], self.data["los_velocity"], **kwargs)

            self.data["los_delta_r_grid"] = self.f_los_delta.interp_many_steps_per_galaxy(self.data["r_grid"])              # noqa
            self.data["los_velocity_r_grid"] = self.f_los_velocity.interp_many_steps_per_galaxy(self.data["r_grid"])        # noqa
            self.data["los_log_density_r_grid"] = self.f_los_log_density.interp_many_steps_per_galaxy(self.data["r_grid"])  # noqa
        else:
            self.num_fields = 1
            self.has_precomputed_los = False

        # Pre-compute Simpson log weights for the radial grid.
        if "r_grid" in self.data:
            self._simpson_log_w = simpson_log_weights(self.data["r_grid"])
            # Reused every step in the LOS integrand (`(r/R)^q` and Jacobian).
            self.data["log_r_grid"] = jnp.log(self.data["r_grid"])
            self.data["log_jac_los"] = 2.0 * self.data["log_r_grid"]
        else:
            self._simpson_log_w = None

        self.has_calibrators = bool(self.num_calibrators > 0)
        self._cache = {}
        self.has_volume_density_3d = False

    def attach_volume_density_3d(self, rho_3d, observer_pos, dx,
                                 galaxy_bias="linear", geometry="cube",
                                 radius=None):
        """Attach 3D density voxels for the volume-normalized empirical prior.

        Stores the minimal density representation needed by `galaxy_bias`,
        plus `log_r_3d` (log voxel distance from the observer; floored at
        `0.25 dx` so the central voxel is finite) and `log_dV = 3 log(dx)`.

        `log_r_3d` is precomputed so the per-step `(r/R)^q` is evaluated as
        `exp(q · (log_r_3d − log R))`, avoiding ~ngrid^3 `log` ops per leapfrog
        step. The `0.25 dx` floor at the central voxel only affects a single
        cell whose `(r/R)^q` is O((dx/R)^q) ≈ 0 anyway.
        """
        self.attach_volume_density_3d_fields(
            [(rho_3d, observer_pos, dx)], galaxy_bias=galaxy_bias,
            geometry=geometry, radius=radius)

    def attach_volume_density_3d_fields(self, fields, batch_size=1,
                                        galaxy_bias="linear",
                                        geometry="cube", radius=None):
        """Attach one 3D density field per field realisation.

        The model maps over the leading field axis with an explicit batch size,
        avoiding full-field vectorization of intermediates during the
        normalizer calculation. `geometry="sphere"` stores only voxels with
        non-zero fractional volume inside `radius` (Mpc/h) as flattened 1D
        arrays; `geometry="cube"` stores the full cube/sub-cube.
        """
        if geometry not in ("cube", "sphere"):
            raise ValueError(
                f"`density_3d_geometry` must be 'cube' or 'sphere', "
                f"got {geometry!r}.")
        if geometry == "sphere" and radius is None:
            raise ValueError(
                "`pv_model.density_3d_radius` is required when "
                "`density_3d_geometry = 'sphere'`.")
        field_iter = iter(fields)
        try:
            first = next(field_iter)
        except StopIteration:
            raise ValueError("At least one 3D density field is required.")

        mode = _volume_density_mode(galaxy_bias)
        rho0, obs0, dx0 = first
        log_r_grid, log_dV = _volume_density_geometry(rho0.shape, obs0, dx0)
        if geometry == "sphere":
            _warn_coarse_sphere_radius(
                radius, dx0, "pv_model.density_3d_radius")
            nsub = rho0.shape
            ax = [(np.arange(nsub[i], dtype=np.float32) + 0.5) * dx0
                  for i in range(3)]
            disp = [ax[0][:, None, None] - obs0[0],
                    ax[1][None, :, None] - obs0[1],
                    ax[2][None, None, :] - obs0[2]]
            voxel_weight = _sphere_voxel_weights(disp, radius, dx0)
            voxel_mask = voxel_weight > 0.0
            log_r_3d = log_r_grid[voxel_mask]
            log_volume_weight_3d = jnp.asarray(
                np.log(voxel_weight[voxel_mask]).astype(np.float32))
        else:
            voxel_mask = None
            log_r_3d = log_r_grid
            log_volume_weight_3d = None
        density_fields = []
        for rho_3d, obs_pos, dx in chain((first,), field_iter):
            if rho_3d.shape != rho0.shape:
                raise ValueError(
                    "All 3D density fields must have the same shape; got "
                    f"{rho_3d.shape} and {rho0.shape}.")
            if not np.allclose(obs_pos, obs0) or not np.isclose(dx, dx0):
                raise ValueError(
                    "All 3D density fields must share observer position and "
                    "voxel size to reuse the volume geometry.")
            if mode == "log_rho":
                density = np.log(rho_3d).astype(np.float32)
            else:
                density = (rho_3d - 1.0).astype(np.float32)
            if voxel_mask is not None:
                density = density[voxel_mask]
            density_fields.append(density)

        self.data["density_3d_fields"] = jnp.asarray(np.stack(density_fields))
        self.data["log_r_3d"] = log_r_3d
        if log_volume_weight_3d is not None:
            self.data["log_volume_weight_3d"] = log_volume_weight_3d
        self.log_dV_3d = log_dV
        self.density_3d_mode = mode
        self.volume_density_batch_size = int(batch_size)
        self.density_3d_geometry = geometry
        self.density_3d_radius = radius
        self.has_volume_density_3d = True

    @classmethod
    def from_config_dict(cls, config, name, try_pop_los, config_pv_model,
                         reconstruction_kwargs=None, reconstruction_name=None,
                         field_cache_dir=None, field_cache_enabled=True):
        root = config.pop("root")
        nsamples_subsample = config.pop("nsamples_subsample", None)
        seed_subsample = config.pop("seed_subsample", 42)
        sample_dust = False

        smooth_target = config_pv_model.get("smooth_target", None)
        if smooth_target is not None:
            config["los_data_path"] = config["los_data_path"].replace(
                ".hdf5", f"_smooth_to_{smooth_target}.hdf5")

        if "CF4_mock" in name:
            index = name.split("_")[-1]
            data = load_CF4_mock(root, index)
        elif "CF4_" in name:
            data = load_CF4_data(root, **config)

            dust_model = config.get("dust_model", None)
            if dust_model is not None:
                fprint(f"using `{dust_model}` for the dust model.")
                sample_dust = True
        elif name in _CATALOGUE_LOADERS:
            data = _CATALOGUE_LOADERS[name](root, **config)
        else:
            raise ValueError(f"Unknown catalogue name: {name}")

        if try_pop_los:
            for key in list(data.keys()):
                if key.startswith("los_"):
                    fprint(f"removing `{key}` from data.")
                    data.pop(key, None)

        r_limits = config_pv_model["r_limits_malmquist"]
        dr = config_pv_model["dr_malmquist"]
        Om = config.get("model", {}).get("Om", 0.3)
        data["r_grid"] = _compute_r_grid(r_limits, dr, data, Om)

        los_decay_scale = config_pv_model.get("los_decay_scale", 5.0)
        fprint(f"setting los_decay_scale to {los_decay_scale}")

        if "los_density" in data:
            data["los_log_density"] = np.log(data["los_density"])
            data["los_delta"] = data["los_density"] - 1

        if nsamples_subsample is not None:
            if name == "PantheonPlusLane":
                raise ValueError(
                    "Subsampling for Pantheon+ Lane is not supported because "
                    "of the complicated covariance matrix.")

            frame = cls(data, los_decay_scale)
            frame = frame.subsample(
                nsamples_subsample, los_decay_scale, seed=seed_subsample)
        else:
            frame = cls(data, los_decay_scale)

        frame.sample_dust = sample_dust

        # Precompute Vext_per_pix data
        nside = config_pv_model.get("Vext_per_pix_nside", None)
        if nside is not None:
            fprint(f"precomputing Vext_per_pix data for nside = {nside}.")
            frame.C_pix = precompute_pixel_projection(frame["rhat"], nside)

        # Hyperparameters for the TFR linewidth modelling
        if "eta_min" in config or "eta_max" in config:
            if config["add_eta_selection"]:
                frame.add_eta_truncation = True
                assert len(frame["e_eta"]) == len(frame)
            else:
                frame.add_eta_truncation = False
                fprint(f"disabling eta truncation for `{name}`.")

        if "eta_min" in config:
            frame.eta_min = config["eta_min"]
            if np.any(frame["eta"] < frame.eta_min):
                raise ValueError(
                    f"eta_min = {frame.eta_min} is smaller than the minimum "
                    f"eta value of {np.min(frame['eta'])}.")
        else:
            frame.eta_min = None

        if "eta_max" in config:
            frame.eta_max = config["eta_max"]
            if np.any(frame["eta"] > frame.eta_max):
                raise ValueError(
                    f"eta_max = {frame.eta_max} is larger than the maximum "
                    f"eta value of {np.max(frame['eta'])}.")
        else:
            frame.eta_max = None

        frame.with_lane_covmat = name == "PantheonPlusLane"
        frame.name = name

        if (
                config_pv_model.get("which_distance_prior", "empirical")
                == "empirical"):
            if reconstruction_kwargs is None or reconstruction_name is None:
                raise ValueError(
                    "The volume-normalized empirical distance prior requires "
                    "a precomputed reconstruction; set "
                    "`pv_model.kind = precomputed_los_<X>` and provide "
                    "`io.reconstruction_main.<X>` paths.")
            if not frame.has_precomputed_los:
                raise ValueError(
                    "The volume-normalized empirical distance prior requires "
                    "precomputed LOS data.")
            downsample = int(config_pv_model.get(
                "density_3d_downsample", 1))
            batch_size = int(config_pv_model.get(
                "density_3d_normalizer_batch_size", 1))
            if batch_size < 1:
                raise ValueError(
                    "`density_3d_normalizer_batch_size` must be positive, "
                    f"got {batch_size}.")
            galaxy_bias = config_pv_model.get("galaxy_bias", "unity")
            geometry = config_pv_model.get("density_3d_geometry", "cube")
            radius = config_pv_model.get("density_3d_radius", None)
            if geometry not in ("cube", "sphere"):
                raise ValueError(
                    "`pv_model.density_3d_geometry` must be 'cube' or "
                    f"'sphere', got {geometry!r}.")
            if geometry == "sphere" and radius is None:
                raise ValueError(
                    "`pv_model.density_3d_radius` is required when "
                    "`density_3d_geometry = 'sphere'`.")
            fprint(
                f"loading {frame.num_fields} volume density cube(s) via "
                f"{reconstruction_name} "
                f"loader (downsample={downsample}, "
                f"geometry={geometry}, radius={radius} Mpc/h, "
                f"normalizer_batch_size={batch_size}, "
                f"density_mode="
                f"{_volume_density_mode(galaxy_bias)}).")
            field_indices = np.asarray(
                data.get("los_field_indices", np.arange(frame.num_fields)),
                dtype=np.int32)
            if len(field_indices) != frame.num_fields:
                raise ValueError(
                    "Number of LOS field indices does not match field "
                    f"realisations: {len(field_indices)} != "
                    f"{frame.num_fields}.")

            def _iter_fields_3d():
                for k, nsim in enumerate(field_indices):
                    rho_3d, obs_pos, dx = _load_volume_density_3d(
                        reconstruction_name, reconstruction_kwargs,
                        downsample=downsample, nsim=int(nsim),
                        subcube_radius=radius,
                        pad_subcube_boundary=(geometry == "sphere"),
                        cache_dir=field_cache_dir,
                        cache_enabled=field_cache_enabled)
                    fprint(
                        f"  field {k} (nsim={int(nsim)}): "
                        f"cube shape {rho_3d.shape}, "
                        f"dx = {dx:.4f} Mpc/h, "
                        f"observer at {obs_pos.tolist()} Mpc/h.")
                    yield rho_3d, obs_pos, dx

            frame.attach_volume_density_3d_fields(
                _iter_fields_3d(), batch_size=batch_size,
                galaxy_bias=galaxy_bias,
                geometry=geometry, radius=radius)

        return frame

    def subsample(self, nsamples, los_radial_decay_scale, seed=42):
        """
        Returns a new frame with randomly selected `nsamples`. Keeps all
        calibrators in the sample (if present), and updates associated
        calibration fields accordingly.
        """
        fprint(f"subsampling from {len(self)} to {nsamples} galaxies.")

        gen = np.random.default_rng(seed)
        ndata = len(self)

        if nsamples > ndata:
            raise ValueError(f"`n_samples = {nsamples}` must be less than the "
                             f"number of data points of {ndata}.")

        main_mask = np.zeros(ndata, dtype=bool)
        if self.num_calibrators > 0:
            main_mask[self.data["is_calibrator"]] = True

        indx_choice = np.where(~main_mask)[0]
        indx_choice = gen.choice(
            indx_choice, nsamples - int(self.num_calibrators), replace=False)
        main_mask[indx_choice] = True

        keys_skip = [
            "is_calibrator", "mu_cal", "C_mu_cal", "std_mu_cal", "los_r",
            "mag_covmat",
            "los_density", "los_delta", "los_velocity", "los_log_density",
            "r_grid", "los_delta_r_grid", "los_velocity_r_grid",
            "los_log_density_r_grid", "log_r_grid", "log_jac_los",
            "los_field_indices", "density_3d_fields", "log_r_3d",
            "log_volume_weight_3d"]

        subsampled = {key: self[key][main_mask]
                      for key in self.keys() if key not in keys_skip}

        for key in keys_skip:
            if key in self.data:
                if key == "los_field_indices":
                    subsampled[key] = self.data[key]
                elif key.startswith("los_") and key != "los_r":
                    subsampled[key] = self[key][:, main_mask, ...]
                elif key == "is_calibrator":
                    subsampled[key] = self[key][main_mask]
                elif key == "mag_covmat":
                    subsampled[key] = self.data[key][main_mask][:, main_mask]
                else:
                    subsampled[key] = self.data[key]

        out = PVDataFrame(subsampled, los_radial_decay_scale)
        out.sample_dust = getattr(self, "sample_dust", False)
        out.name = self.name
        if self.has_volume_density_3d:
            out.has_volume_density_3d = True
            out.log_dV_3d = self.log_dV_3d
            out.density_3d_mode = self.density_3d_mode
            out.volume_density_batch_size = self.volume_density_batch_size
            out.density_3d_geometry = getattr(
                self, "density_3d_geometry", "cube")
            out.density_3d_radius = getattr(
                self, "density_3d_radius", None)
        return out

    def __getitem__(self, key):
        if key in self._cache:
            return jnp.asarray(self._cache[key])

        stat_funcs = {
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max
            }

        if key.startswith("e2_") and key.replace("e2_", "e_") in self.data:
            val = self.data[key.replace("e2_", "e_")]**2
        elif key == "theta":
            val = 0.5 * np.pi - np.deg2rad(self.data["dec"])
        elif key == "phi":
            val = np.deg2rad(self.data["RA"])
        elif key == "C_pix":
            val = self.C_pix
        elif key == "czcmb":
            val = self.data["zcmb"] * SPEED_OF_LIGHT
        elif key == "rhat":
            val = radec_to_cartesian(self.data["RA"], self.data["dec"])
            val /= np.linalg.norm(val, axis=1)[:, None]
        elif "_" in key:
            stat, field = key.split("_", 1)
            if stat in stat_funcs and field in self.data:
                val = stat_funcs[stat](self.data[field])
            else:
                return self.data[key]  # Fallback
        else:
            return self.data[key]

        # If val is a tracer (or contains one), skip caching.
        is_tracer = isinstance(val, jcore.Tracer)
        if not is_tracer:
            try:
                val_np = np.asarray(val)
                self._cache[key] = val_np
                return jnp.asarray(val_np)
            except Exception:
                # Conversion failed (likely because it's a tracer inside
                # a pytree)
                pass

        # Traced value path: do NOT mutate cache; just return it.
        return val

    def keys(self):
        return list(self.data.keys()) + list(self._cache.keys())

    @property
    def num_calibrators(self):
        if "mu_cal" in self.data:
            num_cal = jnp.sum(self.data["is_calibrator"])
        else:
            num_cal = 0

        return num_cal

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __repr__(self):
        n = len(self)
        num_cal = self.num_calibrators

        if num_cal > 0:
            return f"<PVDataFrame: {n} galaxies | {num_cal} calibrators>"
        else:
            return f"<PVDataFrame: {n} galaxies>"


###############################################################################
#                            Specific loaders                                 #
###############################################################################


def load_los(los_data_path, data, mask=None, verbose=True):
    with File(los_data_path, 'r') as f:
        if mask is None:
            data["los_density"] = f['los_density'][...].astype(np.float32)
            data["los_velocity"] = f['los_velocity'][...].astype(np.float32)
            data["los_r"] = f['r'][...]
            data["los_RA"] = f["RA"][...]
            data["los_dec"] = f["dec"][...]
        else:
            dens = f['los_density'][...][:, mask, ...]
            data["los_density"] = dens.astype(np.float32)
            vel = f['los_velocity'][...][:, mask, ...]
            data["los_velocity"] = vel.astype(np.float32)
            data["los_r"] = f['r'][...]
            # Random LOS always use mask=None, so this 1D index is safe.
            data["los_RA"] = f["RA"][...][mask]
            data["los_dec"] = f["dec"][...][mask]

        assert np.all(data["los_density"] > 0)
        assert np.all(np.isfinite(data["los_velocity"]))
        los_field_indices = np.arange(data["los_density"].shape[0])

        norm = _density_unit_normalization(los_data_path)
        if norm is not None:
            divisor, label, detail = norm
            fprint(f"normalizing the {label} LOS density ({detail})",
                   verbose=verbose)
            data["los_density"] /= divisor

        if "_CB1" in los_data_path:
            if len(data["los_density"]) == 100:
                fprint("downsampling the CB1 LOS density from 100 to 20",
                       verbose=verbose)
                data["los_density"] = data["los_density"][::5]
                data["los_velocity"] = data["los_velocity"][::5]
                los_field_indices = los_field_indices[::5]
        elif "_CF4.hdf5" in los_data_path and len(data["los_density"]) == 100:
            fprint("downsampling the CF4 LOS density from 100 to 20",
                   verbose=verbose)
            data["los_density"] = data["los_density"][::5]
            data["los_velocity"] = data["los_velocity"][::5]
            los_field_indices = los_field_indices[::5]

        data["los_field_indices"] = los_field_indices.astype(np.int32)

    return data


def load_SH0ES_calibration(calibration_path, pgc_CF4):
    """
    Load SH0ES distance modulus samples and match to CF4 galaxies by PGC ID.
    """
    with File(calibration_path, 'r') as f:
        mu_samples = f["distmod_samples"][...]
        pgc_SH0ES = f["pgc"][...]

    i_CF4 = []
    i_SH0ES = []

    for i, pgc_i in enumerate(pgc_CF4):
        if pgc_i in pgc_SH0ES:
            match = np.where(pgc_SH0ES == pgc_i)[0]
            assert len(match) == 1
            i_CF4.append(i)
            i_SH0ES.append(match[0])

    i_CF4 = np.array(i_CF4)
    i_SH0ES = np.array(i_SH0ES)

    is_calibrator = np.zeros(len(pgc_CF4), dtype=bool)
    is_calibrator[i_CF4] = True

    mu_cal = np.mean(mu_samples[:, i_SH0ES], axis=0)
    C_mu_cal = np.cov(mu_samples[:, i_SH0ES], rowvar=False)

    return is_calibrator, mu_cal, C_mu_cal


def load_CF4_data(root, which_band, best_mag_quality=True, eta_min=-0.3,
                  zcmb_min=None, zcmb_max=None, b_min=7.5,
                  remove_outliers=True, calibration=None, los_data_path=None,
                  return_all=False, dust_model=None, exclude_W1=False,
                  **kwargs):
    """
    Load CF4 TFR data and apply optional filters and dust correction removal.
    """
    with File(join(root, "CF4_TFR.hdf5"), 'r') as f:
        grp = f["cf4"]
        zcmb = grp["Vcmb"][...] / SPEED_OF_LIGHT
        RA = grp["RA"][...] * 15  # deg
        DEC = grp["DE"][...]
        mag = grp[which_band][...]
        mag_quality = grp["Qw"][...] if which_band == "w1" else grp["Qs"][...]
        eta = grp["lgWmxi"][...] - 2.5
        e_eta = grp["elgWi"][...]
        pgc = grp["pgc"][...]

        if dust_model is not None:
            if which_band not in ["w1", "w2"]:
                raise ValueError(
                    f"Band `{which_band}` is not supported for dust "
                    f"correction removal. Only `w1` and `w2` are supported.")

            Ab_default = grp[f"A_{which_band}"][...]
            fprint(f"switching the dust model to `{dust_model}`.")

            mag += Ab_default
            if dust_model == "default":
                ebv = Ab_default / (0.186 if which_band == "w1" else 0.123)
            else:
                ebv = read_dustmap(RA, DEC, dust_model)

            if not np.all(np.isfinite(ebv)):
                raise ValueError(
                    f"Non-finite E(B-V) values for dust map `{dust_model}`.")
        else:
            ebv = np.full_like(mag, np.nan)

    fprint(f"initially loaded {len(pgc)} galaxies from CF4 TFR data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        e_mag=np.full_like(mag, 0.05),
        eta=eta,
        e_eta=e_eta,
        ebv=ebv,
    )

    if return_all:
        return data

    mask = eta > eta_min
    if best_mag_quality:
        mask &= mag_quality == 5
    else:
        mask &= mag > 5

    mask &= _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)

    if remove_outliers:
        outliers = np.concatenate([
            np.genfromtxt(join(root, f"CF4_{b}_outliers.csv"),
                          delimiter=",", names=True)
            for b in ("W1", "i")
        ])
        mask &= ~np.isin(pgc, outliers["PGC"])

    if which_band == "i" and exclude_W1:
        with File(join(root, "CF4_TFR.hdf5"), 'r') as f:
            w1_quality = f["cf4"]["Qw"][...]
            w1_mag = f["cf4"]["w1"][...]
        fprint("excluding galaxies with W1 quality 5 or W1 mag < 5.")
        exclude = (w1_quality == 5) | (w1_mag > 5)
        mask &= ~exclude

    _filter_data(data, mask, los_data_path)
    pgc = pgc[mask]

    if calibration == "SH0ES":
        is_cal, mu, C_mu = load_SH0ES_calibration(
            join(root, "CF4_SH0ES_calibration.hdf5"), pgc)
        fprint(f"out of {len(pgc)} galaxies, {np.sum(is_cal)} are SH0ES "
               "calibrators.")
        data.update({
            "is_calibrator": is_cal,
            "mu_cal": mu,
            "C_mu_cal": C_mu,
            "std_mu_cal": np.sqrt(np.diag(C_mu)),
        })
    elif calibration:
        raise ValueError("Unknown calibration type.")

    return data


def load_CF4_mock(root, index):
    fname = join(root, f"mock_{index}.hdf5")
    with File(fname, 'r') as f:
        grp = f["mock"]
        data = {key: grp[key][...] for key in grp.keys()}
    return data


def load_2MTF(root, eta_min=-0.1, eta_max=0.2, zcmb_min=None, zcmb_max=None,
              b_min=7.5, los_data_path=None, return_all=False, **kwargs):
    """
    Load the 2MTF data from the given root directory.
    """
    with File(join(root, "PV_compilation.hdf5"), 'r') as f:
        grp = f["2MTF"]

        zcmb = grp["z_CMB"][...]
        RA = grp["RA"][...]
        DEC = grp["DEC"][...]
        mag = grp["mag"][...]
        eta = grp["eta"][...]

        e_eta = grp["e_eta"][...]
        e_mag = grp["e_mag"][...]

    fprint(f"initially loaded {len(zcmb)} galaxies from 2MTF data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        e_mag=e_mag,
        eta=eta,
        e_eta=e_eta,
    )

    if return_all:
        return data

    mask = (eta > eta_min) & (eta < eta_max)
    mask &= _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path)


def load_SFI(root, eta_min=-0.1, zcmb_min=None, zcmb_max=None,
             b_min=7.5, los_data_path=None, return_all=False, **kwargs):
    """
    Load the SFI++ data from the given root directory.
    """
    with File(join(root, "PV_compilation.hdf5"), 'r') as f:
        grp = f["SFI_gals"]

        zcmb = grp["z_CMB"][...]
        RA = grp["RA"][...]
        DEC = grp["DEC"][...]
        mag = grp["mag"][...]
        eta = grp["eta"][...]

        e_eta = grp["e_eta"][...]
        e_mag = grp["e_mag"][...]

    fprint(f"initially loaded {len(zcmb)} galaxies from SFI++ data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        e_mag=e_mag,
        eta=eta,
        e_eta=e_eta,
    )

    if return_all:
        return data

    mask = eta > eta_min
    mask &= _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path)


def _load_LOSS_Foundation(which, root, zcmb_min=None, zcmb_max=None,
                          b_min=7.5, los_data_path=None, return_all=False,
                          **kwargs):
    """
    Load the LOSS or Foundation SNe data from the given root directory.
    """
    with File(join(root, "PV_compilation.hdf5"), 'r') as f:
        grp = f[which]

        zcmb = grp["z_CMB"][...]
        RA = grp["RA"][...]
        DEC = grp["DEC"][...]
        mag = grp["mB"][...]
        c = grp["c"][...]
        x1 = grp["x1"][...]

        e_mag = grp["e_mB"][...]
        e_c = grp["e_c"][...]
        e_x1 = grp["e_x1"][...]

    fprint(f"initially loaded {len(zcmb)} galaxies from LOSS/Foundation data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        c=c,
        x1=x1,
        e_mag=e_mag,
        e_c=e_c,
        e_x1=e_x1
    )

    if return_all:
        return data

    mask = _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path)


def load_LOSS(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
              los_data_path=None, return_all=False, **kwargs):
    return _load_LOSS_Foundation(
        "LOSS", root, zcmb_min=zcmb_min, zcmb_max=zcmb_max,
        b_min=b_min, los_data_path=los_data_path, return_all=return_all,
        **kwargs)


def load_Foundation(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                    los_data_path=None, return_all=False, **kwargs):
    return _load_LOSS_Foundation(
        "Foundation", root, zcmb_min=zcmb_min, zcmb_max=zcmb_max,
        b_min=b_min, los_data_path=los_data_path, return_all=return_all,
        **kwargs)


def load_PantheonPlus_Lane(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                           los_data_path=None, return_all=False, **kwargs):
    if zcmb_max is not None and zcmb_max > 0.075:
        raise ValueError(f"`zcmb_max` of {zcmb_max} is too high for the "
                         "LOWZ sample which goes only up to 0.075.")
    fname = join(root, "full_ps1_input_LOWZ.csv")
    x = np.genfromtxt(fname, delimiter=",", names=True, dtype=None,
                      encoding=None)

    fprint(f"initially loaded {len(x)} galaxies from Pantheon+Lane data.")

    data = dict(
        zcmb=x["zCMB"],
        RA=x["RA"],
        dec=x["DEC"],
        mag=x["mB"],
        x1=x["x1"],
        c=x["c"],
    )

    if return_all:
        return data

    C = np.loadtxt(join(root, "PP_cov_new_LOWZ.txt"))

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    _filter_data(data, mask)

    C_idx = (3 * np.where(mask)[0][:, None] + np.arange(3)).ravel()
    data["mag_covmat"] = C[C_idx][:, C_idx]

    if los_data_path is not None:
        data = load_los(los_data_path, data, mask=mask)

    return data


def load_PantheonPlus(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                      los_data_path=None, return_all=False,
                      removed_PV_from_covmat=True, **kwargs):
    """
    Load the Pantheon+ data from the given root directory, the covariance
    is expected to have peculiar velocity contribution removed.
    """
    if removed_PV_from_covmat:
        arr_fname = "Pantheon+SH0ES_zsel.dat"
        covmat_fname = "Pantheon+SH0ES_zsel_STAT+SYS_noPV.cov"
    else:
        arr_fname = "Pantheon+SH0ES.dat"
        covmat_fname = "Pantheon+SH0ES_STAT+SYS.cov"

    arr = np.genfromtxt(
        join(root, arr_fname), names=True, dtype=None, encoding=None)

    fprint(f"initially loaded {len(arr)} galaxies from Pantheon+ data.")

    data = {
        "zcmb": arr["zCMB"],
        "e_zcmb": arr["zCMBERR"],
        "RA": arr["RA"],
        "dec": arr["DEC"],
        "mag": arr["m_b_corr"],
    }

    if return_all:
        return data

    covmat = np.loadtxt(join(root, covmat_fname), delimiter=",")
    size = int(covmat[0])
    C = np.reshape(covmat[1:], (size, size))

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    _filter_data(data, mask)

    C = C[mask][:, mask]
    data["mag_covmat"] = C
    data["e_mag"] = np.sqrt(np.diag(C))  # Do not use in the inference!

    if los_data_path is not None:
        data = load_los(los_data_path, data, mask=mask)

    return data


def load_SH0ES(root):
    """
    Load the SH0ES data which can be used to sample distances.

    NOTE: Set the zero-width prior to a delta prior so it is not sampled.
    """
    lstsq_results_path = join(root, 'lstsq_results.txt')
    Y_fits_path = join(root, 'ally_shoes_ceph_topantheonwt6.0_112221.fits')
    L_fits_path = join(root, 'alll_shoes_ceph_topantheonwt6.0_112221.fits')
    C_fits_path = join(root, 'allc_shoes_ceph_topantheonwt6.0_112221.fits')

    Y = fits.open(Y_fits_path)[0].data
    L = fits.open(L_fits_path)[0].data
    C = fits.open(C_fits_path)[0].data

    C_inv_cho = linalg.cho_solve(linalg.cho_factor(C), np.identity(C.shape[0]))
    q_lstsq, sigma_lstsq = np.loadtxt(lstsq_results_path, unpack=True)
    mu_list = q_lstsq
    width_list = sigma_lstsq * 10

    ks = np.where(width_list == 0)[0]
    if len(ks) > 0:
        fprint("warning: zero width found in the priors. Setting it to 1e-5.")
        fprint(f"indices of zero width: {ks}")

    if len(ks) != 1:
        raise ValueError("At most one zero width is allowed.")

    k = ks[0]
    fprint(f"found zero-width prior at index {k}. Setting it to 0.")
    width_list[k] = 1e-5
    fixed_idx = k
    fixed_value = 0.

    mu_list = jnp.asarray(mu_list)
    width_list = jnp.asarray(width_list)
    theta_min, theta_max = mu_list - width_list / 2, mu_list + width_list / 2

    data = {
        "Y": Y,
        "L": L,
        "C_inv_cho": C_inv_cho,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "fixed_idx": fixed_idx,
        "fixed_value": fixed_value,
        "C": C,
        }

    for key in data:
        if not key.startswith("fixed_"):
            data[key] = jnp.asarray(data[key], dtype=jnp.float32)

    return data


def load_SH0ES_separated(root, cepheid_host_cz_cmb_max=None,
                         los_data_path=None, rand_los_data_path=None,
                         volume_data=None):
    """
    Load the separated SH0ES data, separating the Cepheid and supernovae and
    covariance matrices.

    Structure of the covariance matrix indices:
    ------------------------------------------
    - Indices < 2150: Cepheid hosts without geometric anchors.
    - Index 2150: Start of NGC 4258 Cepheid hosts.
    - Index 2593: Start of M31 Cepheid hosts.
    - Index 2648: Start of LMC Cepheid hosts.

    - Index 3207: Uncertainty on HST zeropoint (sigma_HST).
    - Index 3208: Uncertainty on Gaia zeropoint (sigma_Gaia).
    - Index 3209: Prior on metallicity coefficient Z_W.
    - Index 3210: Unused term (likely placeholder).
    - Index 3211: Ground-based photometry systematic uncertainty (sigma_grnd).
    - Index 3212: Prior on P–L relation slope b_W.
    - Index 3213: Constraint on NGC 4258 anchor offset (delta_mu_N4258).
    - Index 3214: Constraint on LMC anchor offset (delta_mu_LMC).
    """

    # Unpack the SH0ES data.
    data = load_SH0ES(root)
    Y = np.array(data['Y'], copy=True)
    C = np.array(data['C'], copy=True)
    L = np.array(data['L'].T, copy=True)

    # Cepheid data and covariance matrix.
    OH = L[:, -4][:3130]
    logP = L[:, -6][:3130]
    mag_cepheid = Y[:3130]
    C_Cepheid = C[:3130, :3130]

    # Undo the removal of a slope of -3.285
    mag_cepheid += -3.285 * logP

    # This will organise the host distances as
    # `[Host with Cepheids but no geometric anchors, NGC4258, LMC, M31]`. There
    # are 37 of the former, so in total there are 40 distances to be inferred.
    L_dist = np.hstack([L[:, :37], L[:, [37, 39, 40]]])
    L_Cepheid_host_dist = L_dist[:3130]

    # N4258 and LMC anchors.
    mu_N4258_anchor = 29.398
    e_mu_N4258_anchor = 0.032

    mu_LMC_anchor = 18.477
    e_mu_LMC_anchor = 0.0263

    # Undo the anchor offsets.
    mag_cepheid[2150:2593] += mu_N4258_anchor
    mag_cepheid[2648:] += mu_LMC_anchor

    C_SN_Cepheid = C[3130:3207, 3130:3207]
    Y_SN_Cepheid = Y[3130:3207]

    # HST and Gaia zero-points
    M_HST = Y[3207]
    e_M_HST = C[3207, 3207]**0.5

    M_Gaia = Y[3208]
    e_M_Gaia = C[3208, 3208]**0.5

    # Systematic uncertainties btw ground-based and HST photometry.
    sigma_grnd = C[3211, 3211]**0.5
    # Indices of Cepheids needing ground-based dZP correction (column 45 of
    # the original L matrix is Delta_zp).
    idx_dZP = np.where(L[:3130, 45] == 1)[0]

    q_names = np.asanyarray(
        ['mu_M101', 'mu_M1337', 'mu_N0691', 'mu_N1015', 'mu_N0105',
         'mu_N1309', 'mu_N1365', 'mu_N1448', 'mu_N1559', 'mu_N2442',
         'mu_N2525', 'mu_N2608', 'mu_N3021', 'mu_N3147', 'mu_N3254',
         'mu_N3370', 'mu_N3447', 'mu_N3583', 'mu_N3972', 'mu_N3982',
         'mu_N4038', 'mu_N4424', 'mu_N4536', 'mu_N4639', 'mu_N4680',
         'mu_N5468', 'mu_N5584', 'mu_N5643', 'mu_N5728', 'mu_N5861',
         'mu_N5917', 'mu_N7250', 'mu_N7329', 'mu_N7541', 'mu_N7678',
         'mu_N0976', 'mu_U9391', 'Delta_mu_N4258', 'M_H1_W',
         'Delta_mu_LMC', 'mu_M31', 'b_W', 'MB0', 'Z_W', 'undefined',
         'Delta_zp', 'log10_H0'])

    L_SN_Cepheid_dist = L_dist[3130:3207]

    num_hosts = L_Cepheid_host_dist.shape[1] - 3
    num_cepheids = len(mag_cepheid)
    host_names = q_names[np.char.startswith(
        q_names.astype(str), "mu_")]
    host_names = host_names[:num_hosts]

    # Cepheid host redshifts and the PV covariance matrix.
    data_cepheid_host_redshift = np.load(
        join(root, "processed", "Cepheid_anchors_redshifts.npy"),
        allow_pickle=True)
    PV_covmat_cepheid_host = np.load(
        join(root, "processed", "PV_covmat_cepheid_hosts_fiducial.npy"),
        allow_pickle=True)

    def _load_los_or_none(path, keys):
        if path is None:
            return {k: None for k in keys}
        d = load_los(path, {})
        return {k: d[k] for k in keys}

    host_los_keys = ["los_density", "los_velocity", "los_r"]
    host_los = _load_los_or_none(los_data_path, host_los_keys)

    rand_los_keys = host_los_keys + ["los_RA", "los_dec"]
    rand_los = _load_los_or_none(rand_los_data_path, rand_los_keys)

    # Keep the brightest (lowest magnitude) SN per Cepheid host galaxy
    n_hosts = L_SN_Cepheid_dist.shape[1]
    best_mag = np.full(n_hosts, np.inf)
    best_idx = np.full(n_hosts, -1, dtype=int)

    for i, y in enumerate(Y_SN_Cepheid):
        # Assuming one-hot host assignment per SN
        j = np.where(L_SN_Cepheid_dist[i] == 1)[0][0]
        if y < best_mag[j]:    # use '>' if working in flux (higher = brighter)
            best_mag[j] = y
            best_idx[j] = i

    valid = best_idx >= 0
    unique_ks = best_idx[valid]

    mag_SN_unique_Cepheid_host = Y_SN_Cepheid[unique_ks]
    C_SN_unique_Cepheid_host = C_SN_Cepheid[np.ix_(unique_ks, unique_ks)]
    L_SN_unique_Cepheid_host_dist = L_SN_Cepheid_dist[unique_ks]

    data = {
        # Individual Cepheid data, covariance matrix and host association.
        "mag_cepheid": mag_cepheid,
        "logP": logP,
        "OH": OH,
        "C_Cepheid": C_Cepheid,
        "L_Cepheid": cholesky(C_Cepheid, lower=True),
        "L_Cepheid_host_dist": L_Cepheid_host_dist,
        "Cepheids_only": False,
        "num_cepheids": num_cepheids,
        "num_hosts": num_hosts,
        # Unique SNe in Cepheid host galaxies.
        "mag_SN_unique_Cepheid_host": mag_SN_unique_Cepheid_host,
        "C_SN_unique_Cepheid_host": C_SN_unique_Cepheid_host,
        "mean_std_mag_SN_unique_Cepheid_host": np.mean(np.sqrt(np.diag(C_SN_unique_Cepheid_host))),  # noqa
        "L_SN_unique_Cepheid_host": cholesky(C_SN_unique_Cepheid_host,
                                             lower=True),
        "L_SN_unique_Cepheid_host_dist": L_SN_unique_Cepheid_host_dist,
        # External constraints/priors.
        "mu_N4258_anchor": mu_N4258_anchor,
        "e_mu_N4258_anchor": e_mu_N4258_anchor,
        "mu_LMC_anchor": mu_LMC_anchor,
        "e_mu_LMC_anchor": e_mu_LMC_anchor,
        "M_HST": M_HST,
        "e_M_HST": e_M_HST,
        "M_Gaia": M_Gaia,
        "e_M_Gaia": e_M_Gaia,
        "sigma_grnd": sigma_grnd,
        "idx_dZP": idx_dZP,
        # Cepheid host galaxy information.
        "q_names": q_names,
        "host_names": host_names,
        "czcmb_cepheid_host": data_cepheid_host_redshift["zCMB"] * SPEED_OF_LIGHT,  # noqa
        "e_czcmb_cepheid_host": data_cepheid_host_redshift["zCMBERR"],
        "RA_host": data_cepheid_host_redshift["RA"],
        "dec_host": data_cepheid_host_redshift["DEC"],
        "PV_covmat_cepheid_host": PV_covmat_cepheid_host,
        "host_los_density": host_los["los_density"],
        "host_los_velocity": host_los["los_velocity"],
        "host_los_r": host_los["los_r"],
        # Random LOS for modelling selection
        "has_rand_los": rand_los_data_path is not None,
        "num_rand_los": rand_los["los_density"].shape[1] if rand_los["los_density"] is not None else 1,  # noqa
        "rand_los_density": rand_los["los_density"],
        "rand_los_velocity": rand_los["los_velocity"],
        "rand_los_r": rand_los["los_r"],
        "rand_los_RA": rand_los["los_RA"],
        "rand_los_dec": rand_los["los_dec"]
        }

    if cepheid_host_cz_cmb_max is not None:
        if cepheid_host_cz_cmb_max < 1000:
            raise ValueError(
                f"`cz_cmb_max` must be larger than 1000 km/s, got "
                f"{cepheid_host_cz_cmb_max} km/s. Otherwise could eliminate "
                "some geometric anchors.")

        # Switch this flag so that these runs cannot be done jointly with SNe
        # since some shapes might not be correct.
        data["Cepheids_only"] = True

        cz_host = data["czcmb_cepheid_host"]
        cz_host_all = np.hstack([data["czcmb_cepheid_host"], [667, 327, -582]])
        cz_cepheid = data["L_Cepheid_host_dist"] @ cz_host_all
        cz_unique_SN_Cepheid_host = data["L_SN_unique_Cepheid_host_dist"] @ cz_host_all  # noqa

        mask_host = cz_host < cepheid_host_cz_cmb_max
        mask_host_all = cz_host_all < cepheid_host_cz_cmb_max
        mask_cepheid = cz_cepheid < cepheid_host_cz_cmb_max
        mask_cz_unique_SN_Cepheid_host = (
            cz_unique_SN_Cepheid_host < cepheid_host_cz_cmb_max)

        fprint(f"Masking Cepheids with cz_cmb > {cepheid_host_cz_cmb_max} "
               f"km/s: Keeping {np.sum(mask_host)} out of {len(mask_host)}.")

        data["OH"] = data["OH"][mask_cepheid]
        data["logP"] = data["logP"][mask_cepheid]
        data["mag_cepheid"] = data["mag_cepheid"][mask_cepheid]

        # Remap idx_dZP: keep only indices that survive the mask, then
        # convert to new positions in the masked array.
        old_to_new = np.full(len(mask_cepheid), -1, dtype=int)
        old_to_new[mask_cepheid] = np.arange(mask_cepheid.sum())
        data["idx_dZP"] = old_to_new[data["idx_dZP"]]
        data["idx_dZP"] = data["idx_dZP"][data["idx_dZP"] >= 0]
        data["C_Cepheid"] = data["C_Cepheid"][mask_cepheid][:, mask_cepheid]
        data["L_Cepheid"] = cholesky(data["C_Cepheid"], lower=True)

        data["L_Cepheid_host_dist"] = data["L_Cepheid_host_dist"][mask_cepheid][:, mask_host_all]  # noqa
        data["czcmb_cepheid_host"] = data["czcmb_cepheid_host"][mask_host]
        data["e_czcmb_cepheid_host"] = data["e_czcmb_cepheid_host"][mask_host]
        data["RA_host"] = data["RA_host"][mask_host]
        data["dec_host"] = data["dec_host"][mask_host]
        data["PV_covmat_cepheid_host"] = data["PV_covmat_cepheid_host"][mask_host][:, mask_host]  # noqa

        data["L_SN_unique_Cepheid_host_dist"] = data["L_SN_unique_Cepheid_host_dist"][mask_cz_unique_SN_Cepheid_host][:, mask_host_all]  # noqa
        data["mag_SN_unique_Cepheid_host"] = data["mag_SN_unique_Cepheid_host"][mask_cz_unique_SN_Cepheid_host]  # noqa
        data["C_SN_unique_Cepheid_host"] = data["C_SN_unique_Cepheid_host"][mask_cz_unique_SN_Cepheid_host][:, mask_cz_unique_SN_Cepheid_host]  # noqa
        data["L_SN_unique_Cepheid_host"] = cholesky(data["C_SN_unique_Cepheid_host"], lower=True)  # noqa

        data["num_hosts"] = np.sum(mask_host)
        data["num_cepheids"] = np.sum(mask_cepheid)
        data["host_names"] = data["host_names"][mask_host]

        data["mask_host"] = mask_host

    data["Neff_C_SN_unique_Cepheid_host"] = effective_rank_entropy(data["C_SN_unique_Cepheid_host"]) # noqa
    data["Neff_PV_covmat_cepheid_host"] = effective_rank_entropy(data["PV_covmat_cepheid_host"])     # noqa
    data["Neff_C_Cepheid"] = effective_rank_entropy(data["C_Cepheid"])

    if volume_data is not None:
        data.update(volume_data)
        data["has_volume_density_3d"] = True
    else:
        data["has_volume_density_3d"] = False

    return data


def load_SH0ES_from_config(config_path):
    config = load_config(config_path, replace_los_prior=False)
    use_recon = get_nested(config, "model/use_reconstruction", False)
    config["io"]["load_host_los"] = use_recon
    config["io"]["load_rand_los"] = False
    d = config["io"]["SH0ES"]
    root = d["root"]
    cepheid_host_cz_cmb_max = d.get("cepheid_host_cz_cmb_max", None)
    which_host_los = d.get("which_host_los", None)
    if which_host_los is not None:
        if config["io"]["load_host_los"]:
            los_data_path = config["io"]["PV_main"]["SH0ES"]["los_file"].replace(  # noqa
                "<X>", which_host_los)
        else:
            los_data_path = None

        if config["io"]["load_rand_los"]:
            rand_los_data_path = config["io"]["los_file_random"].replace(
                "<X>", which_host_los)
        else:
            rand_los_data_path = None

    else:
        los_data_path = None
        rand_los_data_path = None

    volume_data = _load_h0_volume_data_from_config(
        config, los_data_path, which_host_los, "SH0ES",
        velocity_selections=("redshift", "SN_magnitude_redshift"))

    return load_SH0ES_separated(
        root, cepheid_host_cz_cmb_max,
        los_data_path=los_data_path, rand_los_data_path=rand_los_data_path,
        volume_data=volume_data)


def load_CCHP_from_config(config_path, ra_dec_only=False):
    """
    Load the processed CCHP TRGB catalogue from a TSV file.

    Returns data in EDD-TRGB-compatible format (host-level arrays with
    ``mag_obs``, ``e_mag_obs``, ``czcmb``, ``e_czcmb``, ``RA_host``,
    ``dec_host``) plus SN-level arrays (``m_Bprime``, ``e_m_Bprime``,
    ``sn_group_index``) for SN magnitude selection.

    Expects the TSV to contain at least the columns:
    SN, Galaxy, cz_cmb, e_czcmb, mu_TRGB_CCHP, sigma_TRGB_CCHP,
    m_Bprime_CSP, sigma_Bprime_CSP.
    """
    config = load_config(config_path, replace_los_prior=False)
    use_recon = get_nested(config, "model/use_reconstruction", False)
    config["io"]["load_host_los"] = use_recon
    config["io"]["load_rand_los"] = False
    path = config["io"]["CCHP"]["path"]
    redshift_source = get_nested(
        config, "io/CCHP_redshift_source/kind", "cz_cmb")
    if not isabs(path):
        path = join(get_root_data(config), path)

    data_tbl = np.genfromtxt(
        path,
        delimiter="\t",
        names=True,
        dtype=None,
        encoding="utf-8",
        missing_values=["-1", "nan", "NaN"],
        filling_values=np.nan,
    )

    # Check here about the wavelength!
    mag_trgb = data_tbl["mu_TRGB_CCHP"] - 4.049
    e_mag_trgb = data_tbl["sigma_TRGB_CCHP"]

    # Fixed anchor values (LMC and NGC 4258) for convenience.
    # LMC (Pietrzynski et al. 2019): https://arxiv.org/abs/1903.08096
    mu_LMC_anchor = 18.477
    e_mu_LMC_anchor = 0.026
    # Hoyt+2021 TRGB calibration: https://arxiv.org/abs/2106.13337
    mag_LMC_TRGB = 14.456
    e_mag_LMC_TRGB = 0.018

    # NGC 4258 distance (Reid et al. 2019)
    mu_N4258_anchor = 29.398
    e_mu_N4258_anchor = 0.032
    # Jang & Lee 2020 TRGB calibration: https://arxiv.org/abs/2008.04181
    # This is at F814W
    mag_N4258_TRGB = 25.347
    e_mag_N4258_TRGB = 0.0443

    ra = data_tbl["ra_deg"]
    dec = data_tbl["dec_deg"]

    source = redshift_source.lower()
    fprint(f"Using CCHP redshift source: {source}", verbose=True)
    if source == "cz_cmb":
        cz_cmb = data_tbl["cz_cmb"]
    elif source == "cz_cmb_ned":
        cz_cmb = data_tbl["cz_cmb_NED"]
    else:
        raise ValueError(
            "Unknown `io/CCHP_redshift_source/kind`: "
            f"{redshift_source}. Use 'cz_cmb' or 'cz_cmb_NED'.")

    e_czcmb = data_tbl["e_czcmb"]
    m_Bprime = data_tbl["m_Bprime_CSP"]
    e_m_Bprime = data_tbl["sigma_Bprime_CSP"]
    galaxies = np.asarray(data_tbl["Galaxy"])

    if ra_dec_only:
        return {
            "RA": ra,
            "DEC": dec,
            "cz_cmb": cz_cmb,
            "e_czcmb": e_czcmb,
        }

    # Mask entries with non-finite SN photometry
    sn_mask = np.isfinite(m_Bprime) & np.isfinite(e_m_Bprime)
    n_masked = int(np.sum(~sn_mask))
    if n_masked > 0:
        fprint(f"CCHP: masking {n_masked}/{len(sn_mask)} entries "
               "without finite SN photometry.")
    mag_trgb = mag_trgb[sn_mask]
    e_mag_trgb = e_mag_trgb[sn_mask]
    cz_cmb = cz_cmb[sn_mask]
    e_czcmb = e_czcmb[sn_mask]
    m_Bprime = m_Bprime[sn_mask]
    e_m_Bprime = e_m_Bprime[sn_mask]
    ra = ra[sn_mask]
    dec = dec[sn_mask]
    galaxies = galaxies[sn_mask]

    # Group by Galaxy (multiple SNe per host)
    galaxies_unique, inverse = np.unique(galaxies, return_inverse=True)
    n_hosts = len(galaxies_unique)
    first_idx = np.array([np.where(inverse == i)[0][0]
                          for i in range(n_hosts)])
    fprint(f"CCHP: {len(galaxies)} SNe across {n_hosts} unique hosts.")

    # Anchor calibration from config (with CCHP defaults)
    anchors = get_nested(config, "model/anchors", {})

    # Build output dict with EDD-compatible host-level keys
    data = {
        # Host-level arrays (one per unique host)
        "mag_obs": mag_trgb[first_idx],
        "e_mag_obs": e_mag_trgb[first_idx],
        "czcmb": cz_cmb[first_idx],
        "e_czcmb": e_czcmb[first_idx],
        "RA_host": ra[first_idx],
        "dec_host": dec[first_idx],
        "e_mag_median": float(np.median(e_mag_trgb[first_idx])),
        # SN-level arrays (one per SN)
        "m_Bprime": m_Bprime,
        "e_m_Bprime": e_m_Bprime,
        "e_m_Bprime_median": float(np.median(e_m_Bprime)),
        "sn_group_index": inverse.astype(np.int32),
        # Anchors
        "mu_LMC_anchor": anchors.get("mu_LMC", mu_LMC_anchor),
        "e_mu_LMC_anchor": anchors.get("e_mu_LMC", e_mu_LMC_anchor),
        "mag_LMC_TRGB": anchors.get("mag_LMC_TRGB", mag_LMC_TRGB),
        "e_mag_LMC_TRGB": anchors.get("e_mag_LMC_TRGB", e_mag_LMC_TRGB),
        "mu_N4258_anchor": anchors.get("mu_N4258", mu_N4258_anchor),
        "e_mu_N4258_anchor": anchors.get("e_mu_N4258", e_mu_N4258_anchor),
        "mag_N4258_TRGB": anchors.get("mag_N4258_TRGB", mag_N4258_TRGB),
        "e_mag_N4258_TRGB": anchors.get(
            "e_mag_N4258_TRGB", e_mag_N4258_TRGB),
    }

    # Load LOS data (host and/or random)
    los_data_path = None
    rand_los_data_path = None

    which_host_los = get_nested(
        config, "io/which_host_los",
        get_nested(config, "io/CCHP/which_host_los", None))
    if get_nested(config, "io/load_host_los", False):
        los_file = get_nested(config, "io/CCHP/los_file", None)
        if los_file is not None and which_host_los is not None:
            los_data_path = los_file.replace("<X>", which_host_los)
        else:
            los_data_path = los_file

    if get_nested(config, "io/load_rand_los", False):
        rand_file = get_nested(config, "io/los_file_random", None)
        if rand_file is not None and which_host_los is not None:
            rand_los_data_path = rand_file.replace("<X>", which_host_los)
        else:
            rand_los_data_path = rand_file

    if los_data_path is not None:
        host_los = load_los(los_data_path, {}, mask=None)
        # LOS file has one entry per row in the original TSV (25 entries).
        # Select valid SN entries, then extract host-level via first_idx.
        los_density = host_los["los_density"][:, sn_mask]
        los_velocity = host_los["los_velocity"][:, sn_mask]
        data["host_los_density"] = los_density[:, first_idx]
        data["host_los_velocity"] = los_velocity[:, first_idx]
        data["host_los_r"] = host_los["los_r"]

    if rand_los_data_path is not None:
        rand_los = load_los(rand_los_data_path, {}, mask=None, verbose=False)
        data["rand_los_density"] = rand_los["los_density"]
        data["rand_los_velocity"] = rand_los["los_velocity"]
        data["rand_los_r"] = rand_los["los_r"]
        data["rand_los_RA"] = rand_los.get("los_RA", None)
        data["rand_los_dec"] = rand_los.get("los_dec", None)
        data["has_rand_los"] = True
        data["num_rand_los"] = data["rand_los_density"].shape[1]
    else:
        data["has_rand_los"] = False

    volume_data = _load_h0_volume_data_from_config(
        config, los_data_path, which_host_los, "CCHP",
        velocity_selections=("redshift",))
    if volume_data is not None:
        data.update(volume_data)
        data["has_volume_density_3d"] = True
    else:
        data["has_volume_density_3d"] = False

    return data


def match_cchp_to_csp(cchp_data, csp_data):
    """
    Match CCHP TRGB hosts to CSP SNe by SN name.

    Handles naming convention differences: CCHP uses '2011fe' while CSP uses
    'SN2011fe'.

    Returns
    -------
    cchp_idx : ndarray
        Indices into cchp_data for matched SNe.
    csp_idx : ndarray
        Indices into csp_data for matched SNe.
    """
    cchp_names = cchp_data["SN"]
    csp_names = csp_data["sn"]

    # CSP names have "SN" prefix, strip it for matching
    csp_name_to_idx = {}
    for i, name in enumerate(csp_names):
        key = name[2:] if name.startswith("SN") else name
        csp_name_to_idx[key] = i

    cchp_idx, csp_idx = [], []
    for i, name in enumerate(cchp_names):
        if name in csp_name_to_idx:
            cchp_idx.append(i)
            csp_idx.append(csp_name_to_idx[name])

    cchp_idx = np.array(cchp_idx, dtype=int)
    csp_idx = np.array(csp_idx, dtype=int)

    fprint(f"matched {len(cchp_idx)}/{len(cchp_names)} CCHP SNe to CSP.")

    # Print unmatched CCHP SNe
    matched_set = set(cchp_idx)
    unmatched = [cchp_names[i] for i in range(len(cchp_names))
                 if i not in matched_set]
    if unmatched:
        fprint(f"unmatched CCHP SNe: {unmatched}")

    return cchp_idx, csp_idx


def load_CSP_from_config(config_path):
    """
    Load CSP SNe data from config, wrapped in PVDataFrame for inference.

    Uses config keys:
    - io.CSP.root: path to CSP data directory
    - io.CSP.which_sample: sample to select ("CSPI", "CSPII", or "LSQ")
    - model.r_limits_malmquist: radial grid limits for Malmquist bias
    - model.num_points_malmquist: number of grid points
    """
    config = load_config(config_path, replace_los_prior=False)

    csp_root = get_nested(config, "io/CSP/root", None)
    if csp_root is None:
        raise ValueError("CSP root not specified in config [io.CSP.root]")
    if not isabs(csp_root):
        csp_root = join(get_root_data(config), csp_root)

    # Get optional CSP loading parameters
    which_sample = get_nested(config, "io/CSP/which_sample", None)
    sample_str = which_sample if which_sample else "all"
    fprint(f"loading CSP sample: {sample_str}")

    data = load_CSP(csp_root, which_sample=which_sample)

    # Add radial grid for selection integral
    r_limits = config["model"]["r_limits_malmquist"]
    num_points = config["model"]["num_points_malmquist"]
    Om = get_nested(config, "model/Om", 0.3)
    data["r_grid"] = _compute_r_grid(r_limits, num_points, data, Om)

    fprint(f"loaded {len(data['sn'])} CSP SNe (sample: {sample_str}).")
    # Return raw dict; JointTRGBCSPModel wraps in PVDataFrame after matching
    return data


###############################################################################
#                                 SDSS FP                                     #
###############################################################################

def arcsec_to_radian(arcsec):
    return (arcsec * u.arcsec).to(u.radian).value


def load_SDSS_FP(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                 los_data_path=None, return_all=False, **kwargs):
    """Load the SDSS FP data from the given root directory."""
    fname = join(root, "SDSS_PV_public.dat")
    d_input = np.genfromtxt(fname, names=True, )

    rdev = d_input["deVRad_r"]
    e_rdev = d_input["deVRadErr_r"]
    boa = d_input["deVAB_r"]
    e_boa = d_input["deVABErr_r"]

    fprint(f"initially loaded {len(d_input)} galaxies from SDSS FP data.")

    theta_eff = arcsec_to_radian(rdev * np.sqrt(boa))
    e_theta_eff = theta_eff * np.sqrt(
        (e_rdev / rdev)**2 + (0.5 * e_boa / boa)**2)

    data = {
        "RA": d_input["RA"],
        "dec": d_input["Dec"],
        "zcmb": d_input["zcmb_group"],
        "theta_eff": theta_eff,
        "e_theta_eff": e_theta_eff,
        "log_theta_eff": np.log10(theta_eff),
        "e_log_theta_eff": e_theta_eff / (theta_eff * np.log(10)),
        "logI": d_input["i"],
        "e_logI": d_input["ei"],
        "logs": d_input["s"],
        "e_logs": d_input["es"],
        }

    if return_all:
        return data

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path)


def load_6dF_FP(root, which_band=None, zcmb_min=None, zcmb_max=None, b_min=7.5,
                los_data_path=None, return_all=False, **kwargs):
    """Load the 6dF FP data from the given root directory."""
    d = np.genfromtxt(join(root, "6dF_FP.dat"))

    RA = d[:, 2] * 360 / 24
    dec = d[:, 3]
    czcmb = d[:, 4]

    data = {
        "RA": RA,
        "dec": dec,
        "zcmb": czcmb / SPEED_OF_LIGHT,
    }

    fprint(f"initially loaded {len(d)} galaxies from 6dF FP data.")

    if return_all:
        return data
    elif which_band is None:
        raise ValueError("which_band must be one of 'J', 'H', 'K'.")

    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    dA_zcmb = cosmo.angular_diameter_distance(czcmb / SPEED_OF_LIGHT).value  # noqa

    if which_band == "J":
        logRe = d[:, 5]
        e_logRe = d[:, 6]

        logIe = d[:, 11]
        e_logIe = d[:, 12]
    elif which_band == "H":
        logRe = d[:, 7]
        e_logRe = d[:, 8]

        logIe = d[:, 13]
        e_logIe = d[:, 14]
    elif which_band == "K":
        logRe = d[:, 9]
        e_logRe = d[:, 10]

        logIe = d[:, 15]
        e_logIe = d[:, 16]
    else:
        raise ValueError(f"which_band must be one of 'J', 'H', 'K', got "
                         f"{which_band}.")
    logVd = d[:, 17]
    e_logVd = d[:, 18]

    log_theta_eff = logRe - np.log10(dA_zcmb * 1e3)
    e_log_theta_eff = e_logRe

    data.update({
        "logI": logIe,
        "e_logI": e_logIe,
        "logs": logVd,
        "e_logs": e_logVd,
        "log_theta_eff": log_theta_eff,
        "e_log_theta_eff": e_log_theta_eff,
    })

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path)


def load_generic(filepath, los_data_path=None, **kwargs):
    """
    Load generic catalog data from a .txt file with column names.

    Expected columns: RA, dec, Vcmb (in CMB frame).
    """
    d = np.genfromtxt(filepath, names=True)

    data = {
        "RA": d["RA"],
        "dec": d["dec"],
        "zcmb": d["Vcmb"] / SPEED_OF_LIGHT,
    }

    fprint(f"loaded {len(data['RA'])} galaxies from {filepath}.")

    if los_data_path is not None:
        data = load_los(los_data_path, data,)

    return data


def load_CSP(root, zcmb_min=None, zcmb_max=None, b_min=None, quality_min=None,
             st_min=None, st_max=None, t0_min=None, t0_max=None,
             phys_only=False, exclude_phys=True, which_sample=None,
             los_data_path=None, return_all=False, remove_duplicates=True,
             **kwargs):
    """
    Load CSP (Carnegie Supernova Project) SNe Ia data.

    Merges photometry from B_all_noj21.csv with coordinates from
    cspallcal_sncoords.csv, csp_sncoords.csv, and missing_coords_simbad.csv.

    Parameters
    ----------
    which_sample : str, optional
        Sample to select: "CSPI", "CSPII", or "LSQ".
    exclude_phys : bool
        If True, exclude physics sample (phys=0 only).
    """
    # Load main photometry file
    fname = join(root, "B_all_noj21.csv")
    d = np.genfromtxt(fname, names=True, dtype=None, encoding="utf-8")
    fprint(f"initially loaded {len(d)} SNe from CSP data.")

    # Remove duplicate SNe (same name, different calibration type)
    if remove_duplicates and not return_all:
        _, unique_idx = np.unique(d["sn"], return_index=True)
        unique_idx = np.sort(unique_idx)
        n_duplicates = len(d) - len(unique_idx)
        fprint(f"removed {n_duplicates} duplicate SNe (keeping first).")
        d = d[unique_idx]

    # Load coordinates from multiple sources
    coords_dict = {}
    coord_files = [
        "cspallcal_sncoords.csv",
        "csp_sncoords.csv",
        "missing_coords_simbad.csv",
    ]
    for fname in coord_files:
        fpath = join(root, fname)
        try:
            coords = np.genfromtxt(fpath, names=True, delimiter=",",
                                   dtype=None, encoding="utf-8")
            for row in coords:
                sn = row["sn"]
                if sn not in coords_dict:
                    ra, dec = row["snra"], row["sndec"]
                    if np.isfinite(ra) and np.isfinite(dec):
                        coords_dict[sn] = (ra, dec)
        except FileNotFoundError:
            pass

    # Match coordinates to main catalog
    RA = np.full(len(d), np.nan)
    dec_ = np.full(len(d), np.nan)

    for i, sn in enumerate(d["sn"]):
        if sn in coords_dict:
            RA[i], dec_[i] = coords_dict[sn]

    n_with_coords = np.sum(np.isfinite(RA))
    fprint(f"matched {n_with_coords}/{len(d)} SNe with coordinates.")

    # Build 3x3 covariance matrix for (peak_mag_B, st, BV)
    n = len(d)
    cov = np.zeros((n, 3, 3))
    cov[:, 0, 0] = d["eMmax"]**2       # var(peak_mag_B)
    cov[:, 1, 1] = d["est"]**2         # var(st)
    cov[:, 2, 2] = d["eBV"]**2         # var(BV)
    cov[:, 0, 1] = cov[:, 1, 0] = d["covMs"]      # cov(peak_mag_B, st)
    cov[:, 0, 2] = cov[:, 2, 0] = d["covBV_M"]    # cov(peak_mag_B, BV)
    # cov(st, BV) = 0 by assumption

    # Fix non-positive definite matrices by adding minimal diagonal
    for i in range(n):
        min_eig = np.linalg.eigvalsh(cov[i]).min()
        if min_eig <= 0:
            cov[i] += np.eye(3) * (abs(min_eig) + 1e-10)
            fprint(f"regularized non-PD covariance for {d['sn'][i]} "
                   f"(zcmb={d['zcmb'][i]:.4f}).")

    # Observation vector: (n_sn, 3) for (peak_mag_B, st, BV)
    obs_vec = np.stack([d["Mmax"], d["st"], d["BV"]], axis=-1)

    # Compute median measurement errors and correlations for selection integral
    sigma_m = np.sqrt(cov[:, 0, 0])
    sigma_s = np.sqrt(cov[:, 1, 1])
    sigma_BV = np.sqrt(cov[:, 2, 2])

    # Correlations from covariance
    rho_ms = cov[:, 0, 1] / (sigma_m * sigma_s)
    rho_mBV = cov[:, 0, 2] / (sigma_m * sigma_BV)
    rho_sBV = cov[:, 1, 2] / (sigma_s * sigma_BV)

    # Convert quality from string to float, empty strings become NaN
    quality = np.array([
        float(q) if q not in ('', '""') else np.nan for q in d["quality"]])

    # Redshift in km/s and default error (100 km/s)
    czcmb = d["zcmb"] * SPEED_OF_LIGHT
    e_czcmb = np.full(len(d), 100.0)

    data = {
        "sn": d["sn"],
        "zcmb": d["zcmb"],
        "czcmb": czcmb,
        "e_czcmb": e_czcmb,
        "zhel": d["zhel"],
        "peak_mag_B": d["Mmax"],
        "st": d["st"],
        "BV": d["BV"],
        "obs_vec": obs_vec,
        "cov": cov,
        "t0": d["t0"],
        "quality": quality,
        "phys": d["phys"],
        "sample": d["sample"],
        "RA": RA,
        "dec": dec_,
        "log_stellar_mass": d["m"],
        "log_stellar_mass_lower": d["ml"],
        "log_stellar_mass_upper": d["mu"],
        # Median values for selection integral
        "median_sigma_m": np.median(sigma_m),
        "median_sigma_s": np.median(sigma_s),
        "median_sigma_BV": np.median(sigma_BV),
        "median_rho_ms": np.median(rho_ms),
        "median_rho_mBV": np.median(rho_mBV),
        "median_rho_sBV": np.median(rho_sBV),
    }

    if return_all:
        return data

    # Filter out SNe without valid coordinates
    has_coords = np.isfinite(RA) & np.isfinite(dec_)
    n_no_coords = np.sum(~has_coords)
    if n_no_coords > 0:
        fprint(f"removing {n_no_coords} SNe without valid coordinates.")

    mask = has_coords
    mask &= _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)

    if quality_min is not None:
        mask &= data["quality"] >= quality_min
    if phys_only:
        mask &= data["phys"] == "1"
    if exclude_phys:
        mask &= data["phys"] == "0"
    if st_min is not None:
        mask &= data["st"] >= st_min
    if st_max is not None:
        mask &= data["st"] <= st_max
    if t0_min is not None:
        mask &= data["t0"] >= t0_min
    if t0_max is not None:
        mask &= data["t0"] <= t0_max
    if which_sample is not None:
        fprint(f"selecting CSP sample: {which_sample}")
        if which_sample == "LSQ":
            mask &= np.char.startswith(data["sn"], "LSQ")
        elif which_sample in ("CSPI", "CSPII"):
            mask &= data["sample"] == which_sample
        else:
            raise ValueError(f"Unknown sample: {which_sample}. "
                             "Must be 'CSPI', 'CSPII', or 'LSQ'.")

    return _filter_data(data, mask, los_data_path)


def _parse_edd_trgb_txt(fpath):
    """Parse an EDD TRGB text file (5 header lines, comma-delimited).

    Returns rows, header, and whether the file is the grouped format
    (extra Vcmb column at index 1).
    """
    with open(fpath) as f:
        lines = f.readlines()
    header = [c.strip() for c in lines[1].strip().split(",")]
    ncol = len(header)
    rows = []
    for line in lines[5:]:
        row = [c.strip().strip('"') for c in line.strip().split(",")]
        if len(row) == ncol:
            rows.append(row)

    has_group_vcmb = (header[1] == "Vcmb")
    return rows, header, has_group_vcmb


def _edd_col_float(rows, idx):
    """Extract a float column, returning NaN for empty/missing cells."""
    out = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        try:
            out[i] = float(row[idx])
        except (ValueError, IndexError):
            pass
    return out


def _edd_col_str(rows, idx):
    return np.array([row[idx].strip() for row in rows])


def _load_edd_trgb_core(fpath, label, zcmb_min=None, zcmb_max=None,
                        b_min=None, los_data_path=None, return_all=False,
                        return_mask=False, e_czcmb_default=20.0):
    """Shared loader for ungrouped and grouped EDD TRGB files.

    The grouped file has an extra CF4 group Vcmb at column 1 (detected
    automatically), stored as ``czcmb_group`` in km/s.
    """
    rows, header, has_group_vcmb = _parse_edd_trgb_txt(fpath)
    n_orig = len(rows)
    fprint(f"initially loaded {n_orig} galaxies from {label} data.")

    off = 1 if has_group_vcmb else 0

    RA = _edd_col_float(rows, 7 + off)        # RAJ
    dec = _edd_col_float(rows, 8 + off)        # DeJ
    czcmb = _edd_col_float(rows, 20 + off)     # individual Vcmb
    T814 = _edd_col_float(rows, 45 + off)
    T8_lo = _edd_col_float(rows, 46 + off)
    T8_hi = _edd_col_float(rows, 47 + off)
    A_814 = _edd_col_float(rows, 62 + off)
    names = _edd_col_str(rows, 35 + off)

    zcmb_arr = czcmb / SPEED_OF_LIGHT

    data = dict(
        RA=RA,
        dec=dec,
        zcmb=zcmb_arr,
        e_zcmb=np.full(n_orig, e_czcmb_default / SPEED_OF_LIGHT),
        mag=T814 - A_814,
        e_mag=(T8_hi - T8_lo) / 2,
    )

    if has_group_vcmb:
        data["czcmb_group"] = _edd_col_float(rows, 1)

    if return_all:
        return data

    keep = np.ones(n_orig, dtype=bool)

    # Drop anchor and satellite galaxies (treated separately in the model).
    drop = np.isin(names, ["LMC", "SMC", "NGC4258", "NGC4258-DF6"])
    if np.any(drop):
        fprint(f"dropping {np.sum(drop)} anchor/satellite galaxies: "
               f"{', '.join(names[drop])}")
    keep &= ~drop

    # Drop galaxies with missing TRGB magnitudes.
    bad_mag = keep & ~np.isfinite(data["mag"])
    if np.any(bad_mag):
        fprint(f"dropping {np.sum(bad_mag)} galaxies with missing TRGB "
               f"magnitudes.")
    keep &= ~bad_mag

    # Drop galaxies with fill-value Vcmb (9999 = no measured velocity).
    bad_vcmb = keep & (np.abs(czcmb) >= 9999)
    if np.any(bad_vcmb):
        fprint(f"dropping {np.sum(bad_vcmb)} galaxies with fill-value Vcmb.")
    keep &= ~bad_vcmb

    # Apply zcmb / galactic latitude cuts on the kept subset.
    sub_mask = _zcmb_blat_mask(
        zcmb_arr[keep], RA[keep], dec[keep], zcmb_min, zcmb_max, b_min)
    keep[np.where(keep)[0][~sub_mask]] = False

    for k in data:
        if isinstance(data[k], np.ndarray):
            data[k] = data[k][keep]
    n_kept = int(np.sum(keep))
    fprint(f"removed {n_orig - n_kept} objects, thus {n_kept} remain.")

    if los_data_path:
        data = load_los(los_data_path, data, mask=keep)

    if return_mask:
        return data, keep
    return data


def load_EDD_TRGB(root, **kwargs):
    """Load ungrouped EDD TRGB data (``EDD_TRGB.txt``)."""
    return _load_edd_trgb_core(
        join(root, "EDD_TRGB.txt"), "EDD TRGB", **kwargs)


def load_EDD_TRGB_grouped(root, **kwargs):
    """Load grouped EDD TRGB data (``EDD_TRGB_grouped.txt``).

    Includes ``czcmb_group`` from the CF4 group catalogue.
    """
    return _load_edd_trgb_core(
        join(root, "EDD_TRGB_grouped.txt"), "EDD TRGB grouped", **kwargs)


def _load_EDD_TRGB_from_config_common(config_path, config_key, loader):
    """Shared from_config logic for both EDD TRGB variants."""
    config = load_config(config_path, replace_los_prior=False)
    use_recon = get_nested(config, "model/use_reconstruction", False)
    config["io"]["load_host_los"] = use_recon
    config["io"]["load_rand_los"] = False
    d = config["io"]["PV_main"][config_key]
    root = d["root"]

    zcmb_min = get_nested(config, f"io/PV_main/{config_key}/zcmb_min", None)
    zcmb_max = get_nested(config, f"io/PV_main/{config_key}/zcmb_max", None)
    b_min = get_nested(config, f"io/PV_main/{config_key}/b_min", None)

    data, mask = loader(root, zcmb_min=zcmb_min, zcmb_max=zcmb_max,
                        b_min=b_min, return_mask=True)

    data["RA_host"] = data.pop("RA")
    data["dec_host"] = data.pop("dec")
    data["mag_obs"] = data.pop("mag")
    data["e_mag_obs"] = data.pop("e_mag")
    # For grouped data, use the group Vcmb instead of individual.
    if "czcmb_group" in data:
        data["czcmb"] = data.pop("czcmb_group")
        data.pop("zcmb")
    else:
        data["czcmb"] = data.pop("zcmb") * SPEED_OF_LIGHT
    data["e_czcmb"] = data.pop("e_zcmb") * SPEED_OF_LIGHT
    data["e_mag_median"] = float(np.median(data["e_mag_obs"]))

    which_los = get_nested(
        config, "io/which_host_los",
        get_nested(config, f"io/PV_main/{config_key}/which_host_los", None))

    def _resolve_los_path(path):
        if path is not None and which_los is not None:
            return path.replace("<X>", which_los)
        return path

    los_data_path = None
    rand_los_data_path = None
    if get_nested(config, "io/load_host_los", False):
        los_data_path = _resolve_los_path(d.get("los_file", None))
    if get_nested(config, "io/load_rand_los", False):
        rand_los_data_path = _resolve_los_path(
            get_nested(config, "io/los_file_random", None))

    fprint(f"reconstruction: {which_los or 'none'}")
    if los_data_path is not None:
        fprint(f"  host LOS path: {los_data_path}")
        host_los = load_los(los_data_path, {}, mask=mask)
        data["host_los_density"] = host_los["los_density"]
        data["host_los_velocity"] = host_los["los_velocity"]
        data["host_los_r"] = host_los["los_r"]
        fprint(f"  host LOS shape: {host_los['los_density'].shape}")

    if rand_los_data_path is not None:
        fprint(f"  random LOS path: {rand_los_data_path}")
        rand_los = load_los(rand_los_data_path, {}, mask=None, verbose=False)
        data["rand_los_density"] = rand_los["los_density"]
        data["rand_los_velocity"] = rand_los["los_velocity"]
        data["rand_los_r"] = rand_los["los_r"]
        data["rand_los_RA"] = rand_los.get("los_RA", None)
        data["rand_los_dec"] = rand_los.get("los_dec", None)
        data["has_rand_los"] = True
        data["num_rand_los"] = data["rand_los_density"].shape[1]
        fprint(f"  random LOS shape: {rand_los['los_density'].shape}"
               f" ({data['num_rand_los']} LOS)")
    else:
        data["has_rand_los"] = False

    volume_data = _load_h0_volume_data_from_config(
        config, los_data_path, which_los, config_key,
        velocity_selections=("redshift",))

    if volume_data is not None:
        data.update(volume_data)
        data["has_volume_density_3d"] = True
    else:
        data["has_volume_density_3d"] = False

    anchors = get_nested(config, "model/anchors", {})
    data["mu_LMC_anchor"] = anchors.get("mu_LMC", 18.477)
    data["e_mu_LMC_anchor"] = anchors.get("e_mu_LMC", 0.026)
    data["mag_LMC_TRGB"] = anchors.get("mag_LMC_TRGB", 14.456)
    data["e_mag_LMC_TRGB"] = anchors.get("e_mag_LMC_TRGB", 0.018)
    data["mu_N4258_anchor"] = anchors.get("mu_N4258", 29.398)
    data["e_mu_N4258_anchor"] = anchors.get("e_mu_N4258", 0.032)
    data["mag_N4258_TRGB"] = anchors.get("mag_N4258_TRGB", 25.347)
    data["e_mag_N4258_TRGB"] = anchors.get("e_mag_N4258_TRGB", 0.0443)

    return data


def load_EDD_TRGB_from_config(config_path):
    """Load ungrouped EDD TRGB data from config."""
    return _load_EDD_TRGB_from_config_common(
        config_path, "EDD_TRGB", load_EDD_TRGB)


def load_EDD_TRGB_grouped_from_config(config_path):
    """Load grouped EDD TRGB data from config."""
    return _load_EDD_TRGB_from_config_common(
        config_path, "EDD_TRGB_grouped", load_EDD_TRGB_grouped)


###############################################################################
#                            EDD 2MTF (K-band TFR)                            #
###############################################################################


def load_EDD_2MTF(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                  eta_min=None, eta_max=None,
                  los_data_path=None, return_all=False,
                  return_mask=False, **kwargs):
    """Load 2MTF data from the EDD text file.

    The file format is pipe-delimited with 5 header lines.
    Returns K-band apparent magnitudes (Ktc) and linewidths
    (eta = log10(Wc) - 2.5).
    """
    fpath = join(root, "EDD_2MTF.txt")
    lines = open(fpath).readlines()
    header = lines[1].strip().split("|")
    rows = []
    for line in lines[5:]:
        rows.append(line.strip().split("|"))

    def col(name):
        return np.array([float(r[header.index(name)]) for r in rows])

    RA = col("RA")
    dec = col("Dec")
    Ktc = col("Ktc")
    eKtc = col("eKtc")
    Wc = col("Wc")
    eWc = col("eWc")
    czcmb = col("czcmb")

    eta = np.log10(Wc) - 2.5
    e_eta = eWc / (Wc * np.log(10))
    zcmb = czcmb / SPEED_OF_LIGHT

    fprint(f"initially loaded {len(RA)} galaxies from EDD 2MTF data.")

    data = dict(
        RA=RA,
        dec=dec,
        zcmb=zcmb,
        mag=Ktc,
        e_mag=eKtc,
        eta=eta,
        e_eta=e_eta,
    )

    if return_all:
        return data

    mask = np.ones(len(RA), dtype=bool)
    mask &= _zcmb_blat_mask(zcmb, RA, dec, zcmb_min, zcmb_max, b_min)
    if eta_min is not None:
        mask &= eta > eta_min
    if eta_max is not None:
        mask &= eta < eta_max

    for k in data:
        if isinstance(data[k], np.ndarray):
            data[k] = data[k][mask]

    n_kept = int(np.sum(mask))
    fprint(f"removed {len(RA) - n_kept} objects, thus {n_kept} remain.")

    if los_data_path:
        data = load_los(los_data_path, data, mask=mask)

    if return_mask:
        return data, mask
    return data


def load_EDD_2MTF_from_config(config_path):
    """Load EDD 2MTF data with LOS from config."""
    config = load_config(config_path, replace_los_prior=False)
    use_recon = get_nested(config, "model/use_reconstruction", False)
    config["io"]["load_host_los"] = use_recon
    config["io"]["load_rand_los"] = use_recon
    d = config["io"]["PV_main"]["EDD_2MTF"]
    root = d["root"]

    zcmb_min = d.get("zcmb_min", None)
    zcmb_max = d.get("zcmb_max", None)
    b_min = d.get("b_min", 7.5)
    eta_min = d.get("eta_min", None)
    eta_max = d.get("eta_max", None)

    data, mask = load_EDD_2MTF(
        root, zcmb_min=zcmb_min, zcmb_max=zcmb_max, b_min=b_min,
        eta_min=eta_min, eta_max=eta_max,
        return_mask=True)

    # Rename to match model expectations
    data["RA_host"] = data.pop("RA")
    data["dec_host"] = data.pop("dec")
    data["czcmb"] = data.pop("zcmb") * SPEED_OF_LIGHT
    data["e_czcmb"] = np.full(len(data["czcmb"]), 10.0)  # ~10 km/s

    # Median errors for selection function
    data["e_mag_median"] = float(np.median(data["e_mag"]))
    data["e_eta_median"] = float(np.median(data["e_eta"]))

    # LOS data
    which_host_los = d.get("which_host_los", None)
    los_data_path = None
    rand_los_data_path = None

    if get_nested(config, "io/load_host_los", False):
        los_file = d.get("los_file", None)
        if los_file is not None and which_host_los is not None:
            los_data_path = los_file.replace("<X>", which_host_los)
        else:
            los_data_path = los_file

    if get_nested(config, "io/load_rand_los", False):
        rand_file = get_nested(config, "io/los_file_random", None)
        if rand_file is not None and which_host_los is not None:
            rand_los_data_path = rand_file.replace("<X>", which_host_los)
        else:
            rand_los_data_path = rand_file

    if los_data_path is not None:
        host_los = load_los(los_data_path, {}, mask=mask)
        data["host_los_density"] = host_los["los_density"]
        data["host_los_velocity"] = host_los["los_velocity"]
        data["host_los_r"] = host_los["los_r"]

    if rand_los_data_path is not None:
        rand_los = load_los(rand_los_data_path, {}, mask=None, verbose=False)
        data["rand_los_density"] = rand_los["los_density"]
        data["rand_los_velocity"] = rand_los["los_velocity"]
        data["rand_los_r"] = rand_los["los_r"]
        data["rand_los_RA"] = rand_los.get("los_RA", None)
        data["rand_los_dec"] = rand_los.get("los_dec", None)
        data["has_rand_los"] = True
        data["num_rand_los"] = data["rand_los_density"].shape[1]
    else:
        data["has_rand_los"] = False

    return data


###############################################################################
#                          Catalogue registry                                 #
###############################################################################


_CATALOGUE_LOADERS = {
    "2MTF": load_2MTF,
    "SFI": load_SFI,
    "SDSS_FP": load_SDSS_FP,
    "6dF_FP": load_6dF_FP,
    "LOSS": load_LOSS,
    "Foundation": load_Foundation,
    "PantheonPlus": load_PantheonPlus,
    "PantheonPlusLane": load_PantheonPlus_Lane,
    "CSP": load_CSP,
    "EDD_TRGB": load_EDD_TRGB,
    "EDD_TRGB_grouped": load_EDD_TRGB_grouped,
    "EDD_2MTF": load_EDD_2MTF,
}
