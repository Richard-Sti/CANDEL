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
import re
import tempfile
from os.path import abspath, exists, isabs, join

import numpy as np

from ..util import fprint, get_nested, get_root_data

_FIELD_CACHE_VERSION = 1


class _ArrayShapeOnly:
    """Tiny stand-in used by MPI cache warmup summaries."""
    def __init__(self, shape):
        self.shape = tuple(shape)


def _field_cache_mpi_comm():
    """Return an MPI communicator for explicit field-cache warmup jobs."""
    if os.environ.get("CANDEL_FIELD_CACHE_MPI", "0") != "1":
        return None
    try:
        from mpi4py import MPI
    except ImportError:
        fprint("CANDEL_FIELD_CACHE_MPI=1 but mpi4py is unavailable; "
               "warming field cache serially.")
        return None
    comm = MPI.COMM_WORLD
    return comm if comm.Get_size() > 1 else None


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
    """Metadata describing field inputs for cache payloads."""
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
        cache_dir = join(get_root_data(config), cache_dir)
    return abspath(cache_dir)


def _field_cache_slug(value, max_len=80):
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "-", text).strip("-")
    if not text:
        return "none"
    return text[:max_len].strip("-") or "none"


def _field_cache_float_tag(value):
    if value is None:
        return "none"
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace(".", "p")


def _field_cache_indices_tag(indices):
    values = [int(x) for x in indices]
    if not values:
        return "none"
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = value
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return "_".join(ranges)


def _parse_field_cache_indices_tag(tag):
    """Parse a compact field-index tag.

    This reverses `_field_cache_indices_tag`.
    """
    if tag == "none":
        return []
    values = []
    for part in tag.split("_"):
        if "-" in part:
            start, stop = part.split("-", 1)
            values.extend(range(int(start), int(stop) + 1))
        else:
            values.append(int(part))
    return values


def _field_cache_payload_digest(payload, length=24):
    """Stable digest for the complete field-cache payload."""
    payload = _jsonable({"version": _FIELD_CACHE_VERSION, **payload})
    key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:length]


def _h0_volume_cache_filename(payload):
    """Readable, cluster-portable filename for H0 3D volume caches."""
    parts = [
        f"v{_FIELD_CACHE_VERSION}",
        _field_cache_slug(payload["field_name"], max_len=70),
        f"fields-{_field_cache_indices_tag(payload['field_indices'])}",
        _field_cache_slug(payload["geometry"], max_len=20),
        f"r-{_field_cache_float_tag(payload['subcube_radius'])}",
        f"ds-{int(payload['downsample'])}",
        "vel" if payload["load_velocity"] else "density",
    ]
    return "__".join(parts) + ".npz"


def _pv_volume_density_cache_filename(payload):
    """Readable, cluster-portable filename for PV 3D density caches."""
    geometry = "sphere" if payload["pad_subcube_boundary"] else "cube"
    field_indices = payload.get("field_indices", None)
    if field_indices is None:
        nsim = payload.get("nsim", None)
        field_tag = "none" if nsim is None else _field_cache_indices_tag(
            [nsim])
    else:
        field_tag = _field_cache_indices_tag(field_indices)
    subsample_fraction = payload.get("voxel_subsample_fraction", 1.0)
    subsample_tag = "sub-{}-seed-{}".format(
        _field_cache_float_tag(subsample_fraction),
        int(payload.get("voxel_subsample_seed", 42)))
    rhat_tag = "rhat" if payload.get("store_rhat_3d", False) else "norhat"
    parts = [
        f"v{_FIELD_CACHE_VERSION}",
        _field_cache_slug(payload["loader_name"], max_len=70),
        f"fields-{field_tag}",
        geometry,
        f"r-{_field_cache_float_tag(payload['subcube_radius'])}",
        f"ds-{int(payload['downsample'])}",
        subsample_tag,
        rhat_tag,
        "density",
    ]
    return "__".join(parts) + ".npz"


def _field_cache_path(cache_dir, prefix, payload):
    """Return the cache path for a field-cache payload."""
    if cache_dir is None:
        return None
    if prefix == "h0_volume_data" and payload.get("kind") == "h0_volume_data":
        return join(cache_dir, prefix, _h0_volume_cache_filename(payload))
    if (prefix == "pv_volume_density_3d"
            and payload.get("kind") == "pv_volume_density_3d"):
        return join(
            cache_dir, prefix, _pv_volume_density_cache_filename(payload))
    digest = _field_cache_payload_digest(payload)
    return join(cache_dir, prefix, f"{digest}.npz")


def _read_field_cache(cache_path, label, required_keys):
    """Load an `.npz` cache file if present and complete."""
    if cache_path is None:
        return None
    fprint(f"checking {label} cache at `{cache_path}`.")
    if not exists(cache_path):
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


def _slice_h0_volume_cache_fields(cached, cached_indices, requested_indices):
    """Slice field-axis arrays in a cached H0 volume product."""
    requested_indices = [int(x) for x in requested_indices]
    cached_indices = [int(x) for x in cached_indices]
    rows = [cached_indices.index(x) for x in requested_indices]
    out = dict(cached)
    for key in ("rho_3d_fields", "vrad_3d_fields"):
        if key in out:
            out[key] = np.asarray(out[key])[rows]
    return out


def _read_h0_volume_cache_superset(cache_dir, payload, label, required_keys,
                                   requested_indices):
    """Load a cached H0 volume product whose field set contains the request."""
    if cache_dir is None:
        return None
    cache_subdir = join(cache_dir, "h0_volume_data")
    if not exists(cache_subdir):
        return None

    requested = [int(x) for x in requested_indices]
    field_slug = _field_cache_slug(payload["field_name"], max_len=70)
    geometry = _field_cache_slug(payload["geometry"], max_len=20)
    radius_tag = _field_cache_float_tag(payload["subcube_radius"])
    downsample_tag = f"ds-{int(payload['downsample'])}"
    kind_tag = "vel" if payload["load_velocity"] else "density"

    candidates = []
    for fname in os.listdir(cache_subdir):
        parts = fname[:-4].split("__") if fname.endswith(".npz") else []
        if len(parts) != 7:
            continue
        version, cached_field, fields_part, cached_geometry, cached_radius, \
            cached_downsample, cached_kind = parts
        if version != f"v{_FIELD_CACHE_VERSION}":
            continue
        if cached_field != field_slug or cached_geometry != geometry:
            continue
        if cached_radius != f"r-{radius_tag}":
            continue
        if cached_downsample != downsample_tag or cached_kind != kind_tag:
            continue
        if not fields_part.startswith("fields-"):
            continue
        cached_indices = _parse_field_cache_indices_tag(
            fields_part.removeprefix("fields-"))
        if all(index in cached_indices for index in requested):
            candidates.append((len(cached_indices), cached_indices,
                               join(cache_subdir, fname)))

    for _, cached_indices, path in sorted(candidates):
        cached = _read_field_cache(path, f"{label} superset", required_keys)
        if cached is not None:
            fprint(f"using {label} cache `{path}` sliced to field indices "
                   f"{requested}.")
            return _slice_h0_volume_cache_fields(
                cached, cached_indices, requested)
    return None


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


def _write_field_cache_mpi_part(path, field_order, field_arrays):
    """Write one warmed field-cache item to a temporary MPI part file."""
    arrays = {
        "field_order": np.asarray(field_order, dtype=np.int64),
    }
    arrays.update({key: np.asarray(value)
                   for key, value in field_arrays.items()})
    np.savez(path, **arrays)


def _read_field_cache_mpi_part(path):
    """Read one warmed field-cache item temporary file."""
    with np.load(path, allow_pickle=False) as f:
        field_order = int(f["field_order"])
        arrays = {key: f[key] for key in f.files if key != "field_order"}
    return field_order, arrays
