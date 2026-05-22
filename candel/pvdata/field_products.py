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
"""Shared helpers for reconstruction-derived preprocessing products."""
import atexit
import itertools
import math
import os
import shutil
import tempfile
from os.path import join, splitext

import numpy as np

from ..field import field_allows_raw_product_reads, field_mas_directory
from ..util import fprint, get_nested
from .angular_scatter import (angular_position_scatter_from_config,
                              scatter_data_coordinates)
from .field_cache import (_field_cache_dir_from_config,
                          _field_cache_enabled_from_config, _jsonable,
                          _los_field_cache_path)

_TEMP_LOS_DIRS = set()
_TEMP_LOS_COUNTER = itertools.count()


def _slug(value):
    text = str(value)
    out = "".join(ch if ch.isalnum() or ch in "_.+-" else "-"
                  for ch in text)
    return out.strip("-") or "value"


def _temporary_los_dir(config):
    parent = get_nested(config, "io/temporary_los_dir", None)
    if parent is None:
        path = tempfile.mkdtemp(prefix="candel_scattered_los_")
    else:
        parent = os.path.abspath(parent)
        os.makedirs(parent, exist_ok=True)
        path = tempfile.mkdtemp(prefix="candel_scattered_los_", dir=parent)
    _TEMP_LOS_DIRS.add(path)
    return path


def cleanup_temporary_los_files():
    """Remove temporary LOS directories created for scattered positions."""
    for path in list(_TEMP_LOS_DIRS):
        shutil.rmtree(path, ignore_errors=True)
        _TEMP_LOS_DIRS.discard(path)


atexit.register(cleanup_temporary_los_files)


def validate_field_smoothing_scale(
        value, label="model.field_3d_smoothing_scale"):
    """Return a non-zero Gaussian field smoothing scale, or ``None``."""
    if value is None:
        return None
    scale = float(value)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError(
            f"`{label}` must be a finite non-negative number in Mpc/h, "
            f"got {value!r}.")
    if scale == 0:
        return None
    return scale


def field_smoothing_scale_from_config(config):
    """Read the optional 3D density-field smoothing scale from a config."""
    return validate_field_smoothing_scale(
        get_nested(config, "model/field_3d_smoothing_scale", None))


def velocity_field_smoothing_scale_from_config(config):
    """Read the optional 3D velocity-field smoothing scale from a config."""
    return validate_field_smoothing_scale(
        get_nested(config, "model/velocity_3d_smoothing_scale", None),
        label="model.velocity_3d_smoothing_scale")


def field_smoothing_cache_payload(field_smoothing_scale):
    """Return the cache-key payload for optional density-field smoothing."""
    scale = validate_field_smoothing_scale(field_smoothing_scale)
    if scale is None:
        return {}
    return {"density_field_smoothing_scale": scale}


def velocity_field_smoothing_cache_payload(velocity_field_smoothing_scale):
    """Return the cache-key payload for optional velocity-field smoothing."""
    scale = validate_field_smoothing_scale(
        velocity_field_smoothing_scale,
        label="model.velocity_3d_smoothing_scale")
    if scale is None:
        return {}
    return {"velocity_field_smoothing_scale": scale}


def field_smoothing_tag(field_smoothing_scale):
    """Return a stable filename tag for non-zero density-field smoothing."""
    scale = validate_field_smoothing_scale(field_smoothing_scale)
    if scale is None:
        return None
    return f"density_smooth_R{scale:g}"


def velocity_field_smoothing_tag(velocity_field_smoothing_scale):
    """Return a stable filename tag for non-zero velocity-field smoothing."""
    scale = validate_field_smoothing_scale(
        velocity_field_smoothing_scale,
        label="model.velocity_3d_smoothing_scale")
    if scale is None:
        return None
    return f"velocity_smooth_R{scale:g}"


def field_smoothed_los_path(path, field_smoothing_scale,
                            velocity_field_smoothing_scale=None):
    """Append the field-smoothing suffix to a LOS file path if needed."""
    if path is None:
        return path
    tags = [
        tag for tag in (
            field_smoothing_tag(field_smoothing_scale),
            velocity_field_smoothing_tag(velocity_field_smoothing_scale),
        )
        if tag is not None
    ]
    if not tags:
        return path
    root, ext = splitext(path)
    return f"{root}_{'_'.join(tags)}{ext}"


def reconstruction_mas_from_config(config, reconstruction):
    """Return the configured MAS label for reconstructions that need one."""
    if config is None or reconstruction != "ManticoreLocalCOLA":
        return None
    recon_cfg = get_nested(
        config, f"io/reconstruction_main/{reconstruction}", {}) or {}
    return field_mas_directory(recon_cfg.get("which_MAS", "CIC"))


def reconstruction_los_label(config, reconstruction):
    """Return the reconstruction label used in LOS filenames."""
    mas = reconstruction_mas_from_config(config, reconstruction)
    if mas is None:
        return reconstruction
    return f"{reconstruction}_{mas}"


def resolve_los_data_path(path, reconstruction=None, field_smoothing_scale=None,
                          config=None, velocity_field_smoothing_scale=None):
    """Resolve ``<X>`` and optional field-smoothing LOS filename suffix."""
    if path is None:
        return None
    if reconstruction is not None:
        path = path.replace("<X>", reconstruction_los_label(
            config, reconstruction))
    return field_smoothed_los_path(
        path, field_smoothing_scale,
        velocity_field_smoothing_scale=velocity_field_smoothing_scale)


def los_radial_grid_payload(config, catalogue, reconstruction):
    """Return the radial interpolation grid encoded in LOS cache filenames."""
    if config is None:
        return None
    if "random_" in str(catalogue):
        rand_cfg = get_nested(config, "io/reconstruction_rand_los", {}) or {}
        rmin = rand_cfg.get("rmin", 0.1)
        rmax = rand_cfg.get("rmax", 251)
        dr = rand_cfg.get("dr", 1.0)
        recon_cfg = rand_cfg.get(reconstruction, {}) or {}
        rmin = recon_cfg.get("rmin", rmin)
        rmax = recon_cfg.get("rmax", rmax)
        dr = recon_cfg.get("dr", dr)
        num_steps = round((rmax - rmin) / dr) + 1
    else:
        grid_cfg = get_nested(config, "io/reconstruction_main", None)
        if grid_cfg is None:
            return None
        try:
            rmin = grid_cfg["rmin"]
            rmax = grid_cfg["rmax"]
            num_steps = grid_cfg["num_steps"]
        except KeyError:
            return None
    return {
        "rmin": float(rmin),
        "rmax": float(rmax),
        "num_steps": int(num_steps),
    }


def los_radial_grid_payload_from_array(r):
    """Return filename metadata for a concrete LOS radial grid array."""
    if len(r) == 0:
        return None
    return {
        "rmin": float(r[0]),
        "rmax": float(r[-1]),
        "num_steps": int(len(r)),
    }


def _format_los_radial_grid(radial_grid):
    """Return a compact human-readable LOS radial-grid summary."""
    return "r=[{:.6g}, {:.6g}] Mpc/h, n={}".format(
        float(radial_grid["rmin"]), float(radial_grid["rmax"]),
        int(radial_grid["num_steps"]))


def _format_cache_paths(paths):
    """Return a short display string for one or many cache paths."""
    if len(paths) == 1:
        return f"`{paths[0]}`"
    return f"{len(paths)} per-field files; first `{paths[0]}`"


def _normalise_field_indices(field_indices):
    """Return field indices as a JSON-stable list, preserving ``None``."""
    if field_indices is None:
        return None
    if isinstance(field_indices, str):
        return [int(field_indices)]
    try:
        return [int(x) for x in field_indices]
    except TypeError:
        return [int(field_indices)]


def _los_coordinate_keys(data):
    """Return the RA/dec key names in a loaded catalogue dictionary."""
    if "RA" in data:
        ra_key = "RA"
    elif "RA_host" in data:
        ra_key = "RA_host"
    else:
        raise KeyError(
            "Cannot build LOS cache from catalogue data without an `RA` "
            "or `RA_host` array.")

    if "dec" in data:
        dec_key = "dec"
    elif "DEC" in data:
        dec_key = "DEC"
    elif "dec_host" in data:
        dec_key = "dec_host"
    else:
        raise KeyError(
            "Cannot build LOS cache from catalogue data without a `dec`, "
            "`DEC`, or `dec_host` array.")
    return ra_key, dec_key


def _los_coordinates_from_data(data):
    """Return RA/dec arrays from a loaded catalogue dictionary."""
    ra_key, dec_key = _los_coordinate_keys(data)
    RA = data[ra_key]
    dec = data[dec_key]

    RA = np.asarray(RA)
    dec = np.asarray(dec)
    if len(RA) != len(dec):
        raise ValueError(
            f"Catalogue coordinate arrays have inconsistent lengths: "
            f"RA has {len(RA)}, dec has {len(dec)}.")
    return RA, dec


class LOSCacheRequest:
    """Deferred per-field LOS cache build from a loaded catalogue."""

    def __init__(self, catalogue, reconstruction, config, cache_paths,
                 cache_valid, field_indices, r, field_smoothing_scale=None,
                 config_path=None, filepath=None, temporary=False,
                 angular_position_scatter=None,
                 velocity_field_smoothing_scale=None):
        self.catalogue = catalogue
        self.reconstruction = reconstruction
        self.config = config
        self.cache_paths = list(cache_paths)
        self.cache_valid = list(cache_valid)
        self.field_indices = list(field_indices)
        self.r = np.asarray(r)
        self.field_smoothing_scale = field_smoothing_scale
        self.velocity_field_smoothing_scale = velocity_field_smoothing_scale
        self.config_path = config_path
        self.filepath = filepath
        self.temporary = bool(temporary)
        self.angular_position_scatter = angular_position_scatter
        self._resolved = None

    def __bool__(self):
        return True

    def __str__(self):
        return self.display_path

    @property
    def display_path(self):
        return _format_cache_paths(self.cache_paths).replace("`", "")

    @property
    def resolved_path(self):
        if self._resolved is not None:
            return self._resolved
        if len(self.cache_paths) == 1:
            return self.cache_paths[0]
        return self.cache_paths

    @property
    def requires_filtered_coordinates(self):
        return self.angular_position_scatter is not None

    def ensure_from_data(self, data, verbose=True):
        """Build missing cache files using the catalogue coordinates in data."""
        if self._resolved is not None:
            return self._resolved
        if self.angular_position_scatter is not None:
            keys = _los_coordinate_keys(data)
            scatter_data_coordinates(
                data, self.angular_position_scatter, keys=keys,
                label=f"{self.catalogue}/{self.reconstruction}")
        RA, dec = _los_coordinates_from_data(data)
        return self.ensure_from_coordinates(
            RA, dec, verbose=verbose, apply_scatter=False)

    def ensure_from_coordinates(self, RA, dec, verbose=True,
                                apply_scatter=True):
        """Build missing cache files using explicit sky coordinates."""
        if self._resolved is not None:
            return self._resolved

        from scripts.preprocess import field_input_los

        RA = np.asarray(RA)
        dec = np.asarray(dec)
        if len(RA) != len(dec):
            raise ValueError(
                f"`RA` and `dec` must have the same length, got "
                f"{len(RA)} and {len(dec)}.")

        if self.angular_position_scatter is not None and apply_scatter:
            coords = {"RA": RA, "dec": dec}
            scatter_data_coordinates(
                coords, self.angular_position_scatter,
                label=f"{self.catalogue}/{self.reconstruction}")
            RA, dec = coords["RA"], coords["dec"]
        metadata = {}
        if self.temporary:
            metadata["temporary_los"] = True
        if self.angular_position_scatter is not None:
            metadata.update({
                "angular_position_scatter_deg": float(
                    self.angular_position_scatter["sigma_deg"]),
                "angular_position_scatter_seed": int(
                    self.angular_position_scatter["seed"]),
            })

        for nsim, cache_path in zip(self.field_indices, self.cache_paths):
            if (not self.temporary
                    and field_input_los.los_file_matches_grid(
                        cache_path, self.r, verbose=False)):
                continue
            field_input_los.compute_los_file_from_coordinates(
                self.catalogue, self.reconstruction, self.config, RA, dec,
                filepath=self.filepath,
                field_smoothing_scale=self.field_smoothing_scale,
                velocity_field_smoothing_scale=(
                    self.velocity_field_smoothing_scale),
                output_path=cache_path, field_indices=[nsim],
                overwrite=True, r=self.r, verbose=verbose,
                metadata=metadata)

        self._resolved = self.resolved_path
        return self._resolved


def los_field_cache_path(config, catalogue, reconstruction, los_template,
                         field_smoothing_scale=None, field_indices=None,
                         radial_grid=None,
                         velocity_field_smoothing_scale=None):
    """Return the canonical field-cache path for a LOS HDF5 product."""
    if (config is None or reconstruction is None or los_template is None
            or not _field_cache_enabled_from_config(config)):
        return None
    field_indices = _normalise_field_indices(field_indices)
    if field_indices is None:
        return None
    if len(field_indices) != 1:
        raise ValueError(
            "LOS cache files are stored per realisation; use "
            "`los_field_cache_paths` for multiple fields.")
    if radial_grid is None:
        radial_grid = los_radial_grid_payload(config, catalogue, reconstruction)
    payload = {
        "kind": "los",
        "catalogue": catalogue,
        "reconstruction": reconstruction,
        "los_template": resolve_los_data_path(
            los_template, reconstruction, field_smoothing_scale,
            config=config,
            velocity_field_smoothing_scale=velocity_field_smoothing_scale),
        "field_indices": _jsonable(field_indices),
        **field_smoothing_cache_payload(field_smoothing_scale),
        **velocity_field_smoothing_cache_payload(
            velocity_field_smoothing_scale),
    }
    mas = reconstruction_mas_from_config(config, reconstruction)
    if mas is not None:
        payload["which_MAS"] = mas
    if radial_grid is not None:
        payload["radial_grid"] = _jsonable(radial_grid)
    return _los_field_cache_path(_field_cache_dir_from_config(config),
                                 payload)


def los_field_cache_paths(config, catalogue, reconstruction, los_template,
                          field_smoothing_scale=None, field_indices=None,
                          radial_grid=None,
                          velocity_field_smoothing_scale=None):
    """Return one canonical LOS cache path per requested field."""
    field_indices = _normalise_field_indices(field_indices)
    if field_indices is None:
        return None
    paths = [
        los_field_cache_path(
            config, catalogue, reconstruction, los_template,
            field_smoothing_scale=field_smoothing_scale,
            field_indices=[nsim], radial_grid=radial_grid,
            velocity_field_smoothing_scale=velocity_field_smoothing_scale)
        for nsim in field_indices
    ]
    if any(path is None for path in paths):
        return None
    return paths


def _temporary_los_paths(config, catalogue, reconstruction, field_indices):
    root = _temporary_los_dir(config)
    tag = f"{os.getpid()}_{next(_TEMP_LOS_COUNTER)}"
    paths = []
    for nsim in field_indices:
        fname = (
            f"los_{_slug(catalogue)}_{_slug(reconstruction)}_"
            f"field-{int(nsim)}_{tag}.hdf5")
        paths.append(join(root, fname))
    return paths


def resolve_or_build_los_data_path(
        config, catalogue, reconstruction, los_template,
        field_smoothing_scale=None, config_path=None, filepath=None,
        field_indices=None, verbose=True, velocity_field_smoothing_scale=None):
    """Resolve a LOS path, deferring raw-readable field cache builds."""
    legacy_path = resolve_los_data_path(
        los_template, reconstruction, field_smoothing_scale, config=config,
        velocity_field_smoothing_scale=velocity_field_smoothing_scale)
    scatter = angular_position_scatter_from_config(config)
    if scatter is not None:
        if reconstruction is None:
            return legacy_path
        if not field_allows_raw_product_reads(reconstruction):
            raise ValueError(
                "Angular position scatter requires on-the-fly LOS "
                f"interpolation, but reconstruction `{reconstruction}` "
                "does not allow raw product reads.")

        from scripts.preprocess import field_input_los
        selected = field_input_los._selected_field_indices(
            config, reconstruction, field_indices)
        r = field_input_los.radial_los_grid(
            config, catalogue, reconstruction, verbose=False)
        radial_grid = los_radial_grid_payload_from_array(r)
        cache_paths = _temporary_los_paths(
            config, catalogue, reconstruction, selected)
        fprint(
            f"temporary scattered LOS: {catalogue}/{reconstruction}",
            verbose=verbose)
        fprint(f"  grid: {_format_los_radial_grid(radial_grid)}",
               verbose=verbose)
        fprint(f"  path: {_format_cache_paths(cache_paths)}",
               verbose=verbose)
        fprint(
            f"  scatter: sigma={scatter['sigma_deg']:g} deg, "
            f"seed={scatter['seed']}; files will be cleaned up after run.",
            verbose=verbose)
        return LOSCacheRequest(
            catalogue, reconstruction, config, cache_paths,
            [False] * len(cache_paths), selected, r,
            field_smoothing_scale=field_smoothing_scale,
            velocity_field_smoothing_scale=velocity_field_smoothing_scale,
            config_path=config_path, filepath=filepath,
            temporary=True, angular_position_scatter=scatter)

    if (config is None or reconstruction is None or los_template is None
            or not _field_cache_enabled_from_config(config)):
        return legacy_path

    from scripts.preprocess import field_input_los
    try:
        selected = field_input_los._selected_field_indices(
            config, reconstruction, field_indices)
    except Exception:
        if field_allows_raw_product_reads(reconstruction):
            raise
        return legacy_path
    r = field_input_los.radial_los_grid(
        config, catalogue, reconstruction, verbose=False)
    radial_grid = los_radial_grid_payload_from_array(r)
    cache_paths = los_field_cache_paths(
        config, catalogue, reconstruction, los_template,
        field_smoothing_scale=field_smoothing_scale,
        velocity_field_smoothing_scale=velocity_field_smoothing_scale,
        field_indices=selected, radial_grid=radial_grid)
    if cache_paths is None:
        return legacy_path
    fprint(f"LOS cache: {catalogue}/{reconstruction}", verbose=verbose)
    fprint(f"  grid: {_format_los_radial_grid(radial_grid)}",
           verbose=verbose)
    fprint(f"  path: {_format_cache_paths(cache_paths)}", verbose=verbose)
    cache_valid = [
        field_input_los.los_file_matches_grid(path, r, verbose=verbose)
        for path in cache_paths
    ]
    if all(cache_valid):
        fprint("  status: cached", verbose=verbose)
        return cache_paths[0] if len(cache_paths) == 1 else cache_paths
    if not field_allows_raw_product_reads(reconstruction):
        missing = [
            path for path, valid in zip(cache_paths, cache_valid)
            if not valid]
        fprint("  status: missing/stale; raw loading disabled",
               verbose=verbose)
        raise FileNotFoundError(
            "Missing or stale per-field LOS cache file(s) for "
            f"`{catalogue}`/`{reconstruction}`: {missing[:5]}. "
            "Run scripts/preprocess/prepare_field_inputs.py --products los "
            "for this task set before inference.")

    fprint("  status: missing/stale; will build after catalogue load",
           verbose=verbose)
    return LOSCacheRequest(
        catalogue, reconstruction, config, cache_paths, cache_valid,
        selected, r, field_smoothing_scale=field_smoothing_scale,
        velocity_field_smoothing_scale=velocity_field_smoothing_scale,
        config_path=config_path, filepath=filepath)
