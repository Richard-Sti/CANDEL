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
import math
from os.path import splitext

import numpy as np

from ..field import field_allows_raw_product_reads
from ..util import fprint, get_nested
from .field_cache import (_field_cache_dir_from_config,
                          _field_cache_enabled_from_config, _jsonable,
                          _los_field_cache_path)


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
    """Read the optional 3D field smoothing scale from a config."""
    return validate_field_smoothing_scale(
        get_nested(config, "model/field_3d_smoothing_scale", None))


def field_smoothing_cache_payload(field_smoothing_scale):
    """Return the cache-key payload for optional field smoothing."""
    scale = validate_field_smoothing_scale(field_smoothing_scale)
    if scale is None:
        return {}
    return {"field_smoothing_scale": scale}


def field_smoothing_tag(field_smoothing_scale):
    """Return a stable filename tag for a non-zero field smoothing scale."""
    scale = validate_field_smoothing_scale(field_smoothing_scale)
    if scale is None:
        return None
    return f"field_smooth_R{scale:g}"


def field_smoothed_los_path(path, field_smoothing_scale):
    """Append the field-smoothing suffix to a LOS file path if needed."""
    tag = field_smoothing_tag(field_smoothing_scale)
    if path is None or tag is None:
        return path
    root, ext = splitext(path)
    return f"{root}_{tag}{ext}"


def resolve_los_data_path(path, reconstruction=None, field_smoothing_scale=None):
    """Resolve ``<X>`` and optional field-smoothing LOS filename suffix."""
    if path is None:
        return None
    if reconstruction is not None:
        path = path.replace("<X>", reconstruction)
    return field_smoothed_los_path(path, field_smoothing_scale)


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


def _los_coordinates_from_data(data):
    """Return RA/dec arrays from a loaded catalogue dictionary."""
    if "RA" in data:
        RA = data["RA"]
    elif "RA_host" in data:
        RA = data["RA_host"]
    else:
        raise KeyError(
            "Cannot build LOS cache from catalogue data without an `RA` "
            "or `RA_host` array.")

    if "dec" in data:
        dec = data["dec"]
    elif "DEC" in data:
        dec = data["DEC"]
    elif "dec_host" in data:
        dec = data["dec_host"]
    else:
        raise KeyError(
            "Cannot build LOS cache from catalogue data without a `dec`, "
            "`DEC`, or `dec_host` array.")

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
                 config_path=None, filepath=None):
        self.catalogue = catalogue
        self.reconstruction = reconstruction
        self.config = config
        self.cache_paths = list(cache_paths)
        self.cache_valid = list(cache_valid)
        self.field_indices = list(field_indices)
        self.r = np.asarray(r)
        self.field_smoothing_scale = field_smoothing_scale
        self.config_path = config_path
        self.filepath = filepath
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

    def ensure_from_data(self, data, verbose=True):
        """Build missing cache files using the catalogue coordinates in data."""
        RA, dec = _los_coordinates_from_data(data)
        return self.ensure_from_coordinates(RA, dec, verbose=verbose)

    def ensure_from_coordinates(self, RA, dec, verbose=True):
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

        for nsim, cache_path in zip(self.field_indices, self.cache_paths):
            if field_input_los.los_file_matches_grid(
                    cache_path, self.r, verbose=False):
                continue
            field_input_los.compute_los_file_from_coordinates(
                self.catalogue, self.reconstruction, self.config, RA, dec,
                filepath=self.filepath,
                field_smoothing_scale=self.field_smoothing_scale,
                output_path=cache_path, field_indices=[nsim],
                overwrite=True, r=self.r, verbose=verbose)

        self._resolved = self.resolved_path
        return self._resolved


def los_field_cache_path(config, catalogue, reconstruction, los_template,
                         field_smoothing_scale=None, field_indices=None,
                         radial_grid=None):
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
            los_template, reconstruction, field_smoothing_scale),
        "field_indices": _jsonable(field_indices),
        **field_smoothing_cache_payload(field_smoothing_scale),
    }
    if radial_grid is not None:
        payload["radial_grid"] = _jsonable(radial_grid)
    return _los_field_cache_path(_field_cache_dir_from_config(config),
                                 payload)


def los_field_cache_paths(config, catalogue, reconstruction, los_template,
                          field_smoothing_scale=None, field_indices=None,
                          radial_grid=None):
    """Return one canonical LOS cache path per requested field."""
    field_indices = _normalise_field_indices(field_indices)
    if field_indices is None:
        return None
    paths = [
        los_field_cache_path(
            config, catalogue, reconstruction, los_template,
            field_smoothing_scale=field_smoothing_scale,
            field_indices=[nsim], radial_grid=radial_grid)
        for nsim in field_indices
    ]
    if any(path is None for path in paths):
        return None
    return paths


def resolve_or_build_los_data_path(
        config, catalogue, reconstruction, los_template,
        field_smoothing_scale=None, config_path=None, filepath=None,
        field_indices=None, verbose=True):
    """Resolve a LOS path, deferring raw-readable field cache builds."""
    legacy_path = resolve_los_data_path(
        los_template, reconstruction, field_smoothing_scale)
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
        config_path=config_path, filepath=filepath)
