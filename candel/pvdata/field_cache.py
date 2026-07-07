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

from ..util import file_last_edited, fprint, get_nested, get_root_data

_LOS_FIELD_CACHE_PREFIX = "los"
_VOLUME_FIELD_CACHE_PREFIX = "volume_field_data"


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


def _field_cache_single_index(payload):
    """Return the single field index required by per-field cache files."""
    field_indices = payload.get("field_indices", None)
    if field_indices is None:
        nsim = payload.get("nsim", None)
        if nsim is None:
            raise ValueError("Per-field cache payload requires `nsim` or "
                             "one `field_indices` value.")
        return int(nsim)
    values = [int(x) for x in field_indices]
    if len(values) != 1:
        raise ValueError(
            "Field cache files are stored per realisation; payload "
            f"contains {values}.")
    return values[0]


def _field_cache_payload_digest(payload, length=24):
    """Stable digest for the complete field-cache payload."""
    payload = _jsonable(payload)
    key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:length]


def _volume_field_product_tag(product):
    """Readable filename tag for a prepared volume-field product."""
    return _field_cache_slug(str(product).replace("_", "-"), max_len=40)


def _los_field_cache_filename(payload):
    """Readable filename for cached LOS products."""
    field_indices = payload.get("field_indices", None)
    if field_indices is None:
        field_tag = "all"
    else:
        field_tag = _field_cache_indices_tag(field_indices)
    parts = [
        "los",
        _field_cache_slug(payload["catalogue"], max_len=50),
        f"field-{field_tag}",
    ]
    radial_grid = payload.get("radial_grid", None)
    if radial_grid is not None:
        parts.append(_radial_grid_cache_tag(radial_grid))
    which_MAS = payload.get("which_MAS", None)
    if which_MAS is not None:
        parts.append(f"mas-{_field_cache_slug(which_MAS, max_len=20)}")
    parts.extend(_field_smoothing_cache_tags(payload))
    return "__".join(parts) + ".hdf5"


def _los_field_cache_path(cache_dir, payload):
    """Return the canonical field-cache path for one LOS HDF5 product."""
    if cache_dir is None:
        return None
    return join(
        cache_dir,
        _field_cache_slug(payload["reconstruction"], max_len=70),
        _LOS_FIELD_CACHE_PREFIX,
        _los_field_cache_filename(payload),
    )


def _volume_field_source_name(payload):
    """Return the reconstruction/loader name for a volume product."""
    for key in ("loader_name", "field_name", "reconstruction"):
        if key in payload:
            return payload[key]
    raise KeyError("volume field cache payload must define a loader name")


def _volume_field_cache_subdir(payload):
    """Directory name for volume-field caches from one reconstruction."""
    return _field_cache_slug(_volume_field_source_name(payload), max_len=70)


def _volume_field_cache_filename(payload):
    """Readable, cluster-portable filename for prepared volume-field caches."""
    product = payload["product"]
    if product == "h0_volume":
        return _h0_volume_product_cache_filename(payload)
    product_tag = _volume_field_product_tag(product)
    if product == "pv_volume_density":
        return _pv_volume_density_product_cache_filename(payload, product_tag)
    if product == "pv_density_cube":
        return _pv_density_cube_product_cache_filename(payload, product_tag)
    raise ValueError(f"Unknown volume field cache product {product!r}.")


def _h0_volume_product_cache_filename(payload):
    """Readable filename for H0 selection-volume cache products."""
    supersample = payload.get("supersample", None)
    geometry = _field_cache_slug(payload["geometry"], max_len=20)
    nsim = _field_cache_single_index(payload)
    parts = [
        f"cache_{geometry}",
        f"field-{nsim}",
        f"r-{_field_cache_float_tag(payload['subcube_radius'])}",
        f"ds-{int(payload['downsample'])}",
    ]
    max_radius = payload.get("max_radius", None)
    if max_radius is not None:
        parts.append(f"rmax-{_field_cache_float_tag(max_radius)}")
    if supersample is not None:
        parts.append(_h0_volume_supersample_tag(supersample))
    parts.extend(_field_smoothing_cache_tags(payload))
    parts.append("vel" if payload["load_velocity"] else "density")
    return "__".join(parts) + ".npz"


def _h0_volume_supersample_tag(supersample):
    """Readable filename tag for H0 supersampling settings."""
    tag = "ss-f{}-r{}".format(
        int(supersample["factor"]),
        _field_cache_float_tag(supersample["radius"]))
    method = supersample.get("method", None)
    if method is not None:
        tag = f"{tag}-{_field_cache_slug(method, max_len=20)}"
    return tag


def _field_smoothing_cache_tag(field_smoothing_scale):
    """Readable filename tag for legacy field smoothing."""
    return "field-smooth-R{}".format(
        _field_cache_float_tag(field_smoothing_scale))


def _density_field_smoothing_cache_tag(field_smoothing_scale):
    """Readable filename tag for density-field smoothing."""
    return "density-smooth-R{}".format(
        _field_cache_float_tag(field_smoothing_scale))


def _velocity_field_smoothing_cache_tag(field_smoothing_scale):
    """Readable filename tag for velocity-field smoothing."""
    return "velocity-smooth-R{}".format(
        _field_cache_float_tag(field_smoothing_scale))


def _field_smoothing_cache_tags(payload):
    """Readable filename tags for density/velocity smoothing payloads."""
    tags = []
    field_smoothing = payload.get("field_smoothing_scale", None)
    if field_smoothing is not None:
        tags.append(_field_smoothing_cache_tag(field_smoothing))
    density_smoothing = payload.get("density_field_smoothing_scale", None)
    if density_smoothing is not None:
        tags.append(_density_field_smoothing_cache_tag(density_smoothing))
    velocity_smoothing = payload.get("velocity_field_smoothing_scale", None)
    if velocity_smoothing is not None:
        tags.append(_velocity_field_smoothing_cache_tag(velocity_smoothing))
    return tags


def _radial_grid_cache_tag(radial_grid):
    """Readable filename tag for a 1D radial interpolation grid."""
    return "r-{}-{}-n{}".format(
        _field_cache_float_tag(radial_grid["rmin"]),
        _field_cache_float_tag(radial_grid["rmax"]),
        int(radial_grid["num_steps"]))


def _pv_volume_density_product_cache_filename(payload, product_tag):
    """Readable filename for grouped PV volume-density cache products."""
    geometry = "sphere" if payload["pad_subcube_boundary"] else "cube"
    nsim = _field_cache_single_index(payload)
    subsample_fraction = payload.get("voxel_subsample_fraction", 1.0)
    subsample_tag = "sub-{}-seed-{}".format(
        _field_cache_float_tag(subsample_fraction),
        int(payload.get("voxel_subsample_seed", 42)))
    rhat_tag = "rhat" if payload.get("store_rhat_3d", False) else "norhat"
    parts = [
        product_tag,
        f"field-{nsim}",
        geometry,
        f"r-{_field_cache_float_tag(payload['subcube_radius'])}",
        f"ds-{int(payload['downsample'])}",
    ]
    max_radius = payload.get("max_radius", None)
    if max_radius is not None:
        parts.append(f"rmax-{_field_cache_float_tag(max_radius)}")
    parts.extend([subsample_tag, rhat_tag])
    parts.extend(_field_smoothing_cache_tags(payload))
    parts.append("density")
    return "__".join(parts) + ".npz"


def _pv_density_cube_product_cache_filename(payload, product_tag):
    """Readable filename for one prepared PV density cube cache product."""
    geometry = "sphere" if payload["pad_subcube_boundary"] else "cube"
    nsim = payload.get("nsim", None)
    field_tag = "none" if nsim is None else str(int(nsim))
    parts = [
        product_tag,
        f"field-{field_tag}",
        geometry,
        f"r-{_field_cache_float_tag(payload['subcube_radius'])}",
        f"ds-{int(payload['downsample'])}",
    ]
    max_radius = payload.get("max_radius", None)
    if max_radius is not None:
        parts.append(f"rmax-{_field_cache_float_tag(max_radius)}")
    parts.extend(_field_smoothing_cache_tags(payload))
    parts.append("density")
    return "__".join(parts) + ".npz"


def _field_cache_path(cache_dir, prefix, payload):
    """Return the cache path for a field-cache payload."""
    if cache_dir is None:
        return None
    if (prefix == _VOLUME_FIELD_CACHE_PREFIX
            and payload.get("kind") == "volume_field_data"):
        return join(cache_dir, _volume_field_cache_subdir(payload),
                    _volume_field_cache_filename(payload))
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
    edited = file_last_edited(cache_path)
    if edited is None:
        fprint(f"loaded {label} cache from `{cache_path}`.")
    else:
        fprint(f"loaded {label} cache from `{cache_path}` "
               f"(last edited {edited}).")
    return out


def _field_cache_payload_for_index(payload, nsim, source_index=None):
    """Return a per-field payload for one requested realisation."""
    out = dict(payload)
    out["field_indices"] = [int(nsim)]
    if source_index is not None and isinstance(out.get("sources"), list):
        sources = out["sources"]
        if len(sources) > source_index:
            out["sources"] = [sources[source_index]]
    return out


def _combine_per_field_caches(caches, field_keys):
    """Stack per-field cache files into the in-memory grouped schema."""
    out = dict(caches[0])
    for key in field_keys:
        if key in out:
            out[key] = np.concatenate(
                [np.asarray(cached[key]) for cached in caches], axis=0)
    return out


def _cache_max_radius_matches(cached, expected_r_3d, label, cache_path):
    """Return whether a cached volume has the requested maximum radius."""
    if expected_r_3d is None or "r_3d" not in cached:
        return True
    got = float(np.max(np.asarray(cached["r_3d"])))
    expected = float(np.max(np.asarray(expected_r_3d)))
    if np.isclose(got, expected):
        return True
    fprint(f"ignoring stale {label} cache `{cache_path}`: max(`r_3d`) "
           "does not match the requested volume radius.")
    return False


def _cache_supersampling_matches(cached, expected, label, cache_path):
    """Return whether cached H0 supersampling metadata matches."""
    if expected is None:
        return True
    factor = int(np.asarray(cached["supersample_factor"]).reshape(-1)[0])
    radius = float(np.asarray(cached["supersample_radius"]).reshape(-1)[0])
    method = str(np.asarray(cached["supersample_method"]).item())
    expected_factor = int(np.asarray(
        expected["supersample_factor"]).reshape(-1)[0])
    expected_radius = float(np.asarray(
        expected["supersample_radius"]).reshape(-1)[0])
    expected_method = str(np.asarray(expected["supersample_method"]).item())
    if (factor == expected_factor and np.isclose(radius, expected_radius)
            and method == expected_method):
        return True
    fprint(f"ignoring stale {label} cache `{cache_path}`: supersampling "
           "metadata does not match the requested settings.")
    return False


def _read_h0_volume_cache_superset(
        cache_dir, payload, label, required_keys, requested_indices,
        expected_r_3d=None, expected_supersampling=None):
    """Load and stack per-field cached H0 volume products."""
    if cache_dir is None:
        return None

    requested = [int(x) for x in requested_indices]
    caches = []
    for i, nsim in enumerate(requested):
        field_payload = _field_cache_payload_for_index(payload, nsim, i)
        path = _field_cache_path(cache_dir, _VOLUME_FIELD_CACHE_PREFIX,
                                 field_payload)
        cached = _read_field_cache(
            path, f"{label} field {nsim}", required_keys)
        if cached is None:
            return None
        if not _cache_max_radius_matches(cached, expected_r_3d, label, path):
            return None
        if not _cache_supersampling_matches(
                cached, expected_supersampling, label, path):
            return None
        caches.append(cached)
    fprint(f"using {label} per-field caches for indices {requested}.")
    return _combine_per_field_caches(
        caches, ("rho_3d_fields", "vrad_3d_fields"))


def _read_pv_volume_cache_superset(cache_dir, payload, label, required_keys,
                                   requested_indices,
                                   expected_r_3d=None):
    """Load and stack per-field cached PV volume-density products."""
    if cache_dir is None:
        return None

    requested = [int(x) for x in requested_indices]
    caches = []
    for i, nsim in enumerate(requested):
        field_payload = _field_cache_payload_for_index(payload, nsim, i)
        path = _field_cache_path(cache_dir, _VOLUME_FIELD_CACHE_PREFIX,
                                 field_payload)
        cached = _read_field_cache(
            path, f"{label} field {nsim}", required_keys)
        if cached is None:
            return None
        if not _cache_max_radius_matches(cached, expected_r_3d, label, path):
            return None
        caches.append(cached)
    fprint(f"using {label} per-field caches for indices {requested}.")
    return _combine_per_field_caches(caches, ("rho_fields",))


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
