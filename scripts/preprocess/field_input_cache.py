"""
Helpers for warming field-derived cache files without running inference.

Loading the model-ready data is enough to trigger cache writers in
``candel.pvdata``. These helpers reuse the same config loaders as production
runs, so generated cache keys match inference.
"""
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

# This is a CPU cache-prep script; set before importing candel/JAX.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_USER = os.environ.get("USER") or "user"
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),
                                 f"candel_mpl_{_USER}"))
os.environ.setdefault(
    "NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(),
                                    f"candel_numba_{_USER}"))

import tomli_w  # noqa: E402

import candel  # noqa: E402
from candel import get_nested  # noqa: E402
import numpy as np  # noqa: E402
from h5py import File  # noqa: E402

from candel.field.loader import name2field_loader  # noqa: E402
from candel.pvdata import catalogues as catalogues_mod  # noqa: E402
from candel.pvdata import field_cache as field_cache_mod  # noqa: E402
from candel.pvdata import field_products as field_products_mod  # noqa: E402
from candel.pvdata import frame as frame_mod  # noqa: E402
from candel.pvdata import los as los_mod  # noqa: E402
from candel.pvdata import volume_density as volume_density_mod  # noqa: E402
from candel.pvdata.field_cache import (  # noqa: E402
    _field_cache_dir_from_config,
)
from candel.util import SPEED_OF_LIGHT  # noqa: E402

pvdata_mod = SimpleNamespace(
    File=File,
    SPEED_OF_LIGHT=SPEED_OF_LIGHT,
    _field_cache_enabled_from_config=(
        field_cache_mod._field_cache_enabled_from_config),
    _VOLUME_FIELD_CACHE_PREFIX=field_cache_mod._VOLUME_FIELD_CACHE_PREFIX,
    _field_cache_path=field_cache_mod._field_cache_path,
    _field_source_metadata=field_cache_mod._field_source_metadata,
    _h0_volume_cache_sampling_payload=(
        volume_density_mod._h0_volume_cache_sampling_payload),
    _h0_volume_cache_supersampling_payload=(
        volume_density_mod._h0_volume_cache_supersampling_payload),
    _h0_volume_supersampling_from_config=(
        volume_density_mod._h0_volume_supersampling_from_config),
    _h0_volume_resolved_supersample_factor=(
        volume_density_mod._h0_volume_resolved_supersample_factor),
    _h0_volume_supersampling_cache_arrays=(
        volume_density_mod._h0_volume_supersampling_cache_arrays),
    field_smoothing_cache_payload=(
        field_products_mod.field_smoothing_cache_payload),
    field_smoothing_scale_from_config=(
        field_products_mod.field_smoothing_scale_from_config),
    velocity_field_smoothing_cache_payload=(
        field_products_mod.velocity_field_smoothing_cache_payload),
    velocity_field_smoothing_scale_from_config=(
        field_products_mod.velocity_field_smoothing_scale_from_config),
    resolve_los_data_path=field_products_mod.resolve_los_data_path,
    _field_loader_native_dx=volume_density_mod._field_loader_native_dx,
    _jsonable=field_cache_mod._jsonable,
    name2field_loader=name2field_loader,
    np=np,
)


ROOT = Path(__file__).resolve().parents[2]
_RANK = 0
_SIZE = 1
_COMM = None
_SH0ES_NUM_HOSTS_CACHE = {}


def _init_mpi_info():
    """Populate rank metadata when running under the scheduler's MPI env."""
    global _RANK, _SIZE, _COMM
    if os.environ.get("CANDEL_FIELD_CACHE_MPI", "0") != "1":
        return
    try:
        from mpi4py import MPI
    except ImportError:
        return
    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()


def _log(message, *, all_ranks=False):
    """Print warmer-level progress without duplicating every rank."""
    if all_ranks or _RANK == 0:
        candel.fprint(message, flush=True)


def _cache_fprint(*args, verbose=True, **kwargs):
    """Filter noisy reader logs; keep only cache-specific progress."""
    if not verbose:
        return
    text = " ".join(str(arg) for arg in args)
    if text.startswith("rank "):
        return
    keep = (
        "cache miss" in text.lower()
        or ("loaded " in text and " cache from " in text)
        or ("wrote " in text and " cache to " in text)
        or text.startswith("ignoring incomplete")
        or text.startswith("ignoring unreadable")
        or text.startswith("could not write")
        or text.startswith("MPI field-cache warmup")
    )
    if not keep:
        return
    if _RANK == 0:
        candel.fprint(*args, **kwargs)


def _install_quiet_reader_logs():
    """Silence catalogue-reader chatter while preserving cache messages."""
    for module in (
            catalogues_mod, field_cache_mod, frame_mod, los_mod,
            volume_density_mod):
        module.fprint = _cache_fprint


def _resolve_cli_path(path):
    """Resolve a path provided on the command line relative to the CWD."""
    return path if path.is_absolute() else Path.cwd() / path


def _parse_id_spec(spec):
    if not spec:
        return None
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return ids


def _read_task_file(path, task_spec=None):
    wanted = _parse_id_spec(task_spec)
    configs = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            idx_s, cfg = line.split(None, 1)
            idx = int(idx_s)
            if wanted is not None and idx not in wanted:
                continue
            cfg_path = Path(cfg)
            if not cfg_path.is_absolute():
                cfg_path = ROOT / cfg_path
            configs.append(cfg_path)
    return configs


def _looks_like_task_file(path):
    """Heuristic for generated tasks_*.txt files."""
    if path.name.startswith("tasks_") and path.suffix == ".txt":
        return True
    try:
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                idx_s, _ = line.split(None, 1)
                int(idx_s)
                return True
    except (OSError, ValueError):
        return False
    return False


def _short_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _table(rows, columns):
    widths = [
        max(len(str(title)), *(len(str(row.get(key, ""))) for row in rows))
        for key, title in columns
    ]
    header = "  ".join(
        str(title).ljust(width) for (_, title), width in zip(columns, widths))
    sep = "  ".join("-" * width for width in widths)
    lines = [header, sep]
    for row in rows:
        lines.append("  ".join(
            str(row.get(key, "")).ljust(width)
            for (key, _), width in zip(columns, widths)))
    return "\n".join(lines)


def _json_key(value):
    return json.dumps(
        pvdata_mod._jsonable(value), sort_keys=True, separators=(",", ":"))


def _h0_los_config(config):
    which_run = get_nested(config, "model/which_run", None)
    if which_run == "CH0":
        return (
            get_nested(config, "io/SH0ES/reconstruction", None),
            get_nested(config, "io/PV_main/SH0ES/los_file", None),
        )
    if which_run in ("CCHP", "CCHP_CSP"):
        return (
            get_nested(config, "io/CCHP/reconstruction", None),
            get_nested(config, "io/CCHP/los_file", None),
        )
    if which_run in ("EDD_TRGB", "EDD_TRGB_grouped"):
        return (
            get_nested(config, f"io/PV_main/{which_run}/reconstruction", None),
            get_nested(config, f"io/PV_main/{which_run}/los_file", None),
        )
    return None, None


def _h0_los_catalogue(config):
    """Return the LOS-prep catalogue name for the configured H0 run."""
    which_run = get_nested(config, "model/which_run", None)
    if which_run == "CH0":
        return "SH0ES"
    if which_run in ("CCHP", "CCHP_CSP", "EDD_TRGB",
                     "EDD_TRGB_grouped"):
        return which_run.replace("_CSP", "")
    return None


def _resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _as_path_list(path_or_paths):
    if isinstance(path_or_paths, (list, tuple)):
        return list(path_or_paths)
    return [path_or_paths]


def _all_paths_exist(path_or_paths):
    return all(_resolve_repo_path(path).exists()
               for path in _as_path_list(path_or_paths))


def _configured_or_available_field_indices(config, reconstruction):
    configured = get_nested(config, "io/field_indices", None)
    if configured is not None:
        configured = pvdata_mod.np.asarray(
            configured, dtype=pvdata_mod.np.int32)
        if configured.ndim == 0:
            configured = configured[None]
        return configured
    try:
        from scripts.preprocess import field_input_los as prep_los_mod
        return prep_los_mod.reconstruction_field_indices(config, reconstruction)
    except Exception:
        return None


def _sh0es_num_hosts(config):
    root = get_nested(config, "io/SH0ES/root", None)
    if root is None:
        return None
    cz_max = get_nested(config, "io/SH0ES/cepheid_host_cz_cmb_max", None)
    key = (root, cz_max)
    if key in _SH0ES_NUM_HOSTS_CACHE:
        return _SH0ES_NUM_HOSTS_CACHE[key]

    redshift_path = _resolve_repo_path(root) / "processed" / (
        "Cepheid_anchors_redshifts.npy")
    try:
        redshifts = pvdata_mod.np.load(redshift_path)
        czcmb = redshifts["zCMB"] * pvdata_mod.SPEED_OF_LIGHT
        if cz_max is not None:
            n_hosts = int(pvdata_mod.np.sum(czcmb < cz_max))
        else:
            n_hosts = int(len(czcmb))
    except Exception:
        n_hosts = None

    _SH0ES_NUM_HOSTS_CACHE[key] = n_hosts
    return n_hosts


def _h0_velocity_key(config):
    which_run = get_nested(config, "model/which_run", None)
    which_sel = get_nested(config, "model/which_selection", None)
    if (which_sel in ("redshift", "SN_magnitude_redshift")
            and which_run not in ("EDD_TRGB", "EDD_TRGB_grouped")):
        return "velocity"
    if which_run == "CH0" and which_sel == "SN_magnitude_or_redshift_Nmag":
        n_mag = get_nested(config, "model/num_hosts_selection_mag", None)
        n_hosts = _sh0es_num_hosts(config)
        if type(n_mag) is int and n_hosts is not None:
            return "velocity" if n_mag < n_hosts else "density"
        return f"mixed_Nmag={n_mag}"
    return "density"


def _field_indices_label(field_indices):
    values = sorted({int(x) for x in field_indices})
    if not values:
        return "-"

    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))

    labels = [
        str(lo) if lo == hi else f"{lo}-{hi}"
        for lo, hi in ranges[:3]
    ]
    if len(ranges) > 3:
        labels.append(f"+{len(ranges) - 3}")
    return ",".join(labels)


def _number_label(value):
    if value is None:
        return "-"
    return f"{float(value):g}"


def _radius_label(geometry, radius):
    if radius is None:
        return str(geometry)
    return f"{geometry} r<{_number_label(radius)}"


def _selection_label(selection):
    if selection in (None, "", "none", "-"):
        return "-"
    selection = str(selection)
    aliases = {
        "SN_magnitude_redshift": "mag+z",
        "SN_magnitude_or_redshift_Nmag": "mag|z_N",
    }
    return aliases.get(selection, selection)


def _cache_group_key(config):
    which_run = get_nested(config, "model/which_run", None)
    if which_run not in ("CH0", "CCHP", "CCHP_CSP", "EDD_TRGB",
                         "EDD_TRGB_grouped"):
        entries, _ = _pv_volume_cache_entries(config)
        if not entries:
            return None
        return _json_key({
            "kind": "volume_field_data_set",
            "product": "pv_volume_density",
            "cache_dir": _field_cache_dir_from_config(config),
            "entries": [entry["payload"] for entry in entries],
        })
    if _variant_action(config) != "check/cache":
        return None

    reconstruction, los_file = _h0_los_config(config)
    field_kwargs = get_nested(
        config, f"io/reconstruction_main/{reconstruction}", None)
    if reconstruction is None or field_kwargs is None:
        return None
    los_data_path = _resolve_los_path(
        los_file, reconstruction, config, _h0_los_catalogue(config))
    field_indices = _h0_field_indices_for_plan(config, reconstruction,
                                               los_data_path)
    if field_indices is None:
        return None

    sampling = pvdata_mod._h0_volume_cache_sampling_payload(
        get_nested(config, "model/density_3d_subsample_fraction", 1.0),
        get_nested(config, "model/density_3d_subsample_seed", 42))
    supersampling = _h0_supersampling_payload(
        config, reconstruction, field_kwargs, field_indices)
    field_smoothing = pvdata_mod.field_smoothing_cache_payload(
        pvdata_mod.field_smoothing_scale_from_config(config))
    velocity_smoothing = pvdata_mod.velocity_field_smoothing_cache_payload(
        pvdata_mod.velocity_field_smoothing_scale_from_config(config))
    return _json_key({
        "kind": "volume_field_data",
        "product": "h0_volume",
        "cache_dir": _field_cache_dir_from_config(config),
        "reconstruction": reconstruction,
        "los_file": los_file,
        "selection_integral_geometry": get_nested(
            config, "model/selection_integral_geometry", "sphere"),
        "selection_integral_grid_radius": get_nested(
            config, "model/selection_integral_grid_radius", None),
        "sampling": sampling,
        "supersampling": supersampling,
        "field_smoothing": field_smoothing,
        "velocity_smoothing": velocity_smoothing,
        "velocity": _h0_velocity_key(config),
    })


def _pv_volume_cache_entries(config):
    kind = get_nested(config, "pv_model/kind", "")
    if not isinstance(kind, str) or not kind.startswith("precomputed_los_"):
        return [], "no precomputed LOS"
    if get_nested(config, "pv_model/which_distance_prior",
                  "empirical") != "empirical":
        return [], "no empirical prior"
    if not pvdata_mod._field_cache_enabled_from_config(
            config, get_nested(config, "pv_model", {})):
        return [], "cache off"

    reconstruction = kind.replace("precomputed_los_", "")
    recon_kwargs = get_nested(
        config, f"io/reconstruction_main/{reconstruction}", None)
    if recon_kwargs is None:
        return [], f"missing reconstruction: {reconstruction}"

    names = get_nested(config, "io/catalogue_name", [])
    if isinstance(names, str):
        names = [names]
    cache_dir = _field_cache_dir_from_config(
        config, get_nested(config, "pv_model", {}))
    downsample = int(get_nested(config, "pv_model/density_3d_downsample", 1))
    geometry = get_nested(config, "pv_model/density_3d_geometry", "cube")
    radius = get_nested(config, "pv_model/density_3d_radius", None)
    pad_boundary = geometry == "sphere"
    voxel_subsample_fraction = get_nested(
        config, "pv_model/density_3d_subsample_fraction", 1.0)
    voxel_subsample_seed = get_nested(
        config, "pv_model/density_3d_subsample_seed", 42)
    store_rhat_3d = bool(get_nested(config, "pv_model/use_Mmiss", False))
    field_smoothing = pvdata_mod.field_smoothing_cache_payload(
        pvdata_mod.field_smoothing_scale_from_config(config))

    entries = {}
    geometry_tag = "sphere" if pad_boundary else "cube"
    for name in names:
        io_name = "CF4_mock" if str(name).startswith("CF4_mock") else name
        io_section = get_nested(config, f"io/{io_name}", None)
        if io_section is None or "los_file" not in io_section:
            return [], f"missing LOS config: {name}"
        los_path = _resolve_los_path(
            io_section["los_file"], reconstruction, config, name)
        if not _all_paths_exist(los_path):
            missing = [
                _short_path(_resolve_repo_path(path))
                for path in _as_path_list(los_path)
                if not _resolve_repo_path(path).exists()
            ]
            return [], f"missing LOS: {missing[:3]}"
        field_indices = _read_h0_field_indices(los_path)

        source_meta = []
        max_radii = []
        for nsim in field_indices:
            loader_kwargs = dict(recon_kwargs)
            loader_kwargs["nsim"] = int(nsim)
            loader = pvdata_mod.name2field_loader(
                reconstruction)(**loader_kwargs)
            source_meta.append(pvdata_mod._field_source_metadata(loader))
            max_radii.append(
                volume_density_mod._expected_pv_volume_max_radius_from_loader(
                    loader, downsample, radius, pad_boundary, geometry_tag,
                    radius, voxel_subsample_fraction, voxel_subsample_seed))
        for i, nsim in enumerate(field_indices):
            payload = {
                "kind": "volume_field_data",
                "product": "pv_volume_density",
                "loader_name": reconstruction,
                "loader_kwargs": pvdata_mod._jsonable(recon_kwargs),
                "field_indices": [int(nsim)],
                "downsample": downsample,
                "subcube_radius": radius,
                "max_radius": max_radii[i],
                "pad_subcube_boundary": bool(pad_boundary),
                "voxel_subsample_fraction": float(voxel_subsample_fraction),
                "voxel_subsample_seed": int(voxel_subsample_seed),
                "store_rhat_3d": store_rhat_3d,
                **field_smoothing,
                "sources": [source_meta[i]],
            }
            cache_path = Path(pvdata_mod._field_cache_path(
                cache_dir, pvdata_mod._VOLUME_FIELD_CACHE_PREFIX, payload))
            entries[str(cache_path)] = {
                "payload": payload,
                "path": cache_path,
            }

    return list(entries.values()), ""


def _h0_field_indices_for_plan(config, reconstruction, los_data_path):
    if los_data_path is not None and _all_paths_exist(los_data_path):
        return _read_h0_field_indices(los_data_path)
    return _configured_or_available_field_indices(config, reconstruction)


def _variant_action(config):
    """Describe what the warmer will ask the normal data loader to do."""
    which_run = get_nested(config, "model/which_run", None)
    h0_runs = ("CH0", "CCHP", "CCHP_CSP", "EDD_TRGB", "EDD_TRGB_grouped")
    if which_run not in h0_runs:
        return "check/cache"

    which_sel = get_nested(config, "model/which_selection", None)
    if not get_nested(config, "model/use_reconstruction", False):
        return "skip recon"
    needs_no_selection_volume = (
        which_sel in (None, "none")
        and which_run in ("CH0", "EDD_TRGB", "EDD_TRGB_grouped"))
    if which_sel in (None, "none") and not needs_no_selection_volume:
        return "skip 3D"
    if not pvdata_mod._field_cache_enabled_from_config(config):
        return "cache off"
    return "check/cache"


def _h0_supersampling_description(config):
    """Human-readable H0 volume supersampling settings."""
    which_run = get_nested(config, "model/which_run", None)
    h0_runs = ("CH0", "CCHP", "CCHP_CSP", "EDD_TRGB",
               "EDD_TRGB_grouped")
    if which_run not in h0_runs:
        return ""
    _, radius, target_dx = pvdata_mod._h0_volume_supersampling_from_config(
        config)
    if radius <= 0.0 or target_dx is None:
        return ""
    return (
        f"trilinear r<{radius:g} Mpc/h, "
        f"target_dx={target_dx:g} Mpc/h")


def _h0_supersampling_label(config):
    factor, radius, target_dx = pvdata_mod._h0_volume_supersampling_from_config(
        config)
    if radius <= 0.0:
        return "-"
    if target_dx is not None:
        return f"r<{radius:g},dx={target_dx:g}"
    if factor <= 1:
        return "-"
    return f"r<{radius:g},x{int(factor)}"


def _field_smoothing_description(config):
    """Human-readable field smoothing settings."""
    scale = pvdata_mod.field_smoothing_scale_from_config(config)
    velocity_scale = pvdata_mod.velocity_field_smoothing_scale_from_config(
        config)
    if scale is None and velocity_scale is None:
        return ""
    parts = []
    if scale is not None:
        parts.append(f"density R={scale:g} Mpc/h")
    if velocity_scale is not None:
        parts.append(f"velocity R={velocity_scale:g} Mpc/h")
    return "Gaussian " + ", ".join(parts)


def _field_smoothing_label(config):
    scale = pvdata_mod.field_smoothing_scale_from_config(config)
    velocity_scale = pvdata_mod.velocity_field_smoothing_scale_from_config(
        config)
    if scale is None and velocity_scale is None:
        return "-"
    if velocity_scale is None:
        return f"rhoR={scale:g}"
    if scale is None:
        return f"vR={velocity_scale:g}"
    return f"rhoR={scale:g},vR={velocity_scale:g}"


def _resolve_los_path(path, reconstruction, config=None, catalogue=None):
    field_smoothing_scale = (
        None if config is None else
        pvdata_mod.field_smoothing_scale_from_config(config))
    velocity_field_smoothing_scale = (
        None if config is None else
        pvdata_mod.velocity_field_smoothing_scale_from_config(config))
    if catalogue is not None:
        field_indices = _configured_or_available_field_indices(
            config, reconstruction)
        cache_path = field_products_mod.los_field_cache_paths(
            config, catalogue, reconstruction, path,
            field_smoothing_scale=field_smoothing_scale,
            velocity_field_smoothing_scale=velocity_field_smoothing_scale,
            field_indices=field_indices)
        if cache_path is not None:
            return cache_path
    return pvdata_mod.resolve_los_data_path(
        path, reconstruction, field_smoothing_scale, config=config,
        velocity_field_smoothing_scale=velocity_field_smoothing_scale)


def _h0_supersampling_payload(config, reconstruction, field_kwargs,
                              field_indices):
    factor, radius, target_dx = (
        pvdata_mod._h0_volume_supersampling_from_config(config))
    if target_dx is not None and radius > 0.0:
        kwargs = dict(field_kwargs)
        kwargs["nsim"] = int(field_indices[0])
        loader = pvdata_mod.name2field_loader(reconstruction)(**kwargs)
        dx = pvdata_mod._field_loader_native_dx(loader)
        factor = pvdata_mod._h0_volume_resolved_supersample_factor(
            dx, factor, target_dx)
    return pvdata_mod._h0_volume_cache_supersampling_payload(factor, radius)


def _cache_grid_status(f, expected_r_3d=None):
    """Return a stale-cache note if the stored volume radius differs."""
    if expected_r_3d is not None:
        got = float(pvdata_mod.np.max(f["r_3d"][...]))
        expected = float(pvdata_mod.np.max(pvdata_mod.np.asarray(
            expected_r_3d)))
        if not pvdata_mod.np.isclose(got, expected):
            return "stale", "max r_3d differs"
    return None


def _cache_supersampling_status(f, expected):
    """Return a stale-cache note if supersampling metadata differs."""
    factor = int(pvdata_mod.np.asarray(
        f["supersample_factor"]).reshape(-1)[0])
    radius = float(pvdata_mod.np.asarray(
        f["supersample_radius"]).reshape(-1)[0])
    method = str(pvdata_mod.np.asarray(f["supersample_method"]).item())
    expected_factor = int(pvdata_mod.np.asarray(
        expected["supersample_factor"]).reshape(-1)[0])
    expected_radius = float(pvdata_mod.np.asarray(
        expected["supersample_radius"]).reshape(-1)[0])
    expected_method = str(pvdata_mod.np.asarray(
        expected["supersample_method"]).item())
    if (factor == expected_factor and pvdata_mod.np.isclose(
            radius, expected_radius) and method == expected_method):
        return None
    return "stale", "supersampling metadata differs"


def _read_h0_field_indices(los_data_path):
    out = []
    for path in _as_path_list(los_data_path):
        los_read_path = _resolve_repo_path(path)
        with pvdata_mod.File(los_read_path, "r") as f:
            if "field_indices" in f:
                out.extend(f["field_indices"][:].astype(pvdata_mod.np.int32))
            else:
                out.extend(pvdata_mod.np.arange(
                    f["los_density"].shape[0], dtype=pvdata_mod.np.int32))
    return pvdata_mod.np.asarray(out, dtype=pvdata_mod.np.int32)


def _h0_cache_file_status(config):
    if _variant_action(config) != "check/cache":
        return None, None

    reconstruction, los_file = _h0_los_config(config)
    los_data_path = _resolve_los_path(
        los_file, reconstruction, config, _h0_los_catalogue(config))
    if reconstruction is None or los_data_path is None:
        return None, "no LOS"
    if not _all_paths_exist(los_data_path):
        return "missing", "needs LOS"

    field_indices = _read_h0_field_indices(los_data_path)

    field_kwargs = get_nested(
        config, f"io/reconstruction_main/{reconstruction}", None)
    if field_kwargs is None:
        return None, f"missing reconstruction: {reconstruction}"

    source_meta = []
    first_loader = None
    for i, nsim in enumerate(field_indices):
        kwargs = dict(field_kwargs)
        kwargs["nsim"] = int(nsim)
        loader = pvdata_mod.name2field_loader(reconstruction)(**kwargs)
        if i == 0:
            first_loader = loader
        source_meta.append(pvdata_mod._field_source_metadata(loader))

    geometry = get_nested(
        config, "model/selection_integral_geometry", "sphere")
    grid_radius = get_nested(
        config, "model/selection_integral_grid_radius", None)
    subsample_fraction = get_nested(
        config, "model/density_3d_subsample_fraction", 1.0)
    subsample_seed = get_nested(
        config, "model/density_3d_subsample_seed", 42)
    load_velocity = _h0_velocity_key(config) == "velocity"
    base_payload = {
        "kind": "volume_field_data",
        "product": "h0_volume",
        "loader_name": reconstruction,
        "field_indices": pvdata_mod._jsonable(
            pvdata_mod.np.asarray(field_indices)),
        "subcube_radius": grid_radius,
        "geometry": geometry,
        "sources": source_meta,
    }
    base_payload.update(pvdata_mod._h0_volume_cache_sampling_payload(
        subsample_fraction, subsample_seed))
    supersampling = _h0_supersampling_payload(
        config, reconstruction, field_kwargs, field_indices)
    base_payload.update(supersampling)
    supersample = supersampling.get("supersample", None)
    if supersample is None:
        supersample_factor = 1
        supersample_radius = 0.0
    else:
        supersample_factor = int(supersample["factor"])
        supersample_radius = float(supersample["radius"])
    expected_max_r_3d = (
        volume_density_mod._expected_h0_volume_max_radius_from_loader(
            first_loader, geometry, grid_radius, supersample_factor,
            supersample_radius))
    base_payload["max_radius"] = expected_max_r_3d
    expected_supersampling = pvdata_mod._h0_volume_supersampling_cache_arrays(
        supersample_factor, supersample_radius)
    field_smoothing = pvdata_mod.field_smoothing_cache_payload(
        pvdata_mod.field_smoothing_scale_from_config(config))
    velocity_smoothing = pvdata_mod.velocity_field_smoothing_cache_payload(
        pvdata_mod.velocity_field_smoothing_scale_from_config(config))
    density_payload = {
        **base_payload,
        **field_smoothing,
        "load_velocity": False,
    }
    velocity_payload = {**base_payload, **velocity_smoothing,
                        "load_velocity": True}
    cache_dir = _field_cache_dir_from_config(config)
    density_cache_paths = [
        Path(pvdata_mod._field_cache_path(
            cache_dir, pvdata_mod._VOLUME_FIELD_CACHE_PREFIX,
            {**density_payload, "field_indices": [int(nsim)],
             "sources": [source_meta[i]]}))
        for i, nsim in enumerate(field_indices)
    ]
    velocity_cache_paths = [
        Path(pvdata_mod._field_cache_path(
            cache_dir, pvdata_mod._VOLUME_FIELD_CACHE_PREFIX,
            {**velocity_payload, "field_indices": [int(nsim)],
             "sources": [source_meta[i]]}))
        for i, nsim in enumerate(field_indices)
    ]
    supersampling_required = [
        "supersample_factor", "supersample_radius", "supersample_method"]
    density_required = [
        "rho_3d_fields", "r_3d", "log_dV_3d", *supersampling_required]
    if ((geometry == "sphere" and grid_radius is not None)
            or supersampling):
        density_required.append("log_volume_weight_3d")
    velocity_required = [
        "vrad_3d_fields", "r_3d", "log_dV_3d",
        "rhat_x_3d", "rhat_y_3d", "rhat_z_3d",
        *supersampling_required]

    missing_files = 0
    for density_cache_path in density_cache_paths:
        if not density_cache_path.exists():
            missing_files += 1
            continue
        try:
            with pvdata_mod.np.load(
                    density_cache_path, allow_pickle=False) as f:
                missing = [
                    key for key in density_required if key not in f.files]
                if not missing:
                    stale = _cache_grid_status(f, expected_max_r_3d)
                    if stale is not None:
                        return stale
                    stale = _cache_supersampling_status(
                        f, expected_supersampling)
                    if stale is not None:
                        return stale
        except Exception as exc:
            return "unreadable", str(exc)
        if missing:
            return "incomplete", f"missing {missing}"
    if missing_files:
        return "missing", f"needs {missing_files} density cache file(s)"

    if load_velocity:
        missing_files = 0
        for velocity_cache_path in velocity_cache_paths:
            if not velocity_cache_path.exists():
                missing_files += 1
                continue
            try:
                with pvdata_mod.np.load(
                        velocity_cache_path, allow_pickle=False) as f:
                    missing = [
                        key for key in velocity_required if key not in f.files]
                    if not missing:
                        stale = _cache_grid_status(f, expected_max_r_3d)
                        if stale is not None:
                            return stale
                        stale = _cache_supersampling_status(
                            f, expected_supersampling)
                        if stale is not None:
                            return stale
            except Exception as exc:
                return "unreadable", str(exc)
            if missing:
                return "incomplete", f"missing {missing}"
        if missing_files:
            return "missing", f"needs {missing_files} velocity cache file(s)"
    if load_velocity:
        return "cached", "rho+vel"
    return "cached", "rho"


def _pv_cache_file_status(config):
    entries, note = _pv_volume_cache_entries(config)
    if note:
        return None, note
    if not entries:
        return None, "no PV 3D cache"

    missing = 0
    for entry in entries:
        path = entry["path"]
        payload = entry["payload"]
        geometry = "sphere" if payload["pad_subcube_boundary"] else "cube"
        store_rhat_3d = bool(payload.get("store_rhat_3d", False))
        if not path.exists():
            missing += 1
            continue
        try:
            loader_kwargs = dict(payload["loader_kwargs"])
            loader_kwargs["nsim"] = int(payload["field_indices"][0])
            loader = pvdata_mod.name2field_loader(
                payload["loader_name"])(**loader_kwargs)
            expected_max_r_3d = (
                volume_density_mod._expected_pv_volume_max_radius_from_loader(
                    loader, int(payload["downsample"]),
                    payload["subcube_radius"],
                    bool(payload["pad_subcube_boundary"]), geometry,
                    payload["subcube_radius"],
                    float(payload.get("voxel_subsample_fraction", 1.0)),
                    int(payload.get("voxel_subsample_seed", 42))))
            with pvdata_mod.np.load(path, allow_pickle=False) as f:
                missing_keys = [
                    key for key in (
                        "rho_fields", "r_3d", "log_dV_3d",
                        "observer_pos", "dx")
                    if key not in f.files]
                if geometry == "sphere" and (
                        "log_volume_weight_3d" not in f.files):
                    missing_keys.append("log_volume_weight_3d")
                if store_rhat_3d:
                    missing_keys.extend(
                        key for key in (
                            "rhat_x_3d", "rhat_y_3d", "rhat_z_3d")
                        if key not in f.files)
                if not missing_keys:
                    stale = _cache_grid_status(f, expected_max_r_3d)
                    if stale is not None:
                        return stale
        except Exception as exc:
            return "unreadable", str(exc)
        if missing_keys:
            return "incomplete", f"missing {missing_keys}"

    if missing:
        return "missing", f"missing {missing}/{len(entries)}"
    return "cached", f"{len(entries)} file(s)"


def _cache_product_description(config):
    which_run = get_nested(config, "model/which_run", None)
    h0_runs = ("CH0", "CCHP", "CCHP_CSP", "EDD_TRGB", "EDD_TRGB_grouped")
    if which_run in h0_runs:
        reconstruction, los_file = _h0_los_config(config)
        los_data_path = (
            None if reconstruction is None or los_file is None
            else _resolve_los_path(
                los_file, reconstruction, config, _h0_los_catalogue(config)))
        field_indices = _h0_field_indices_for_plan(
            config, reconstruction, los_data_path)
        geometry = get_nested(
            config, "model/selection_integral_geometry", "sphere")
        grid_radius = get_nested(
            config, "model/selection_integral_grid_radius", None)
        kind = "rho+vel" if _h0_velocity_key(config) == "velocity" else "rho"
        return {
            "model": reconstruction or "-",
            "fields": (
                "-" if field_indices is None
                else _field_indices_label(field_indices)),
            "radius": _radius_label(geometry, grid_radius),
            "kind": kind,
            "ss": _h0_supersampling_label(config),
            "smooth": _field_smoothing_label(config),
        }

    entries, _ = _pv_volume_cache_entries(config)
    kind = get_nested(config, "pv_model/kind", "")
    model = (
        kind.replace("precomputed_los_", "")
        if isinstance(kind, str) and kind.startswith("precomputed_los_")
        else str(kind or "-"))
    if not entries:
        return {
            "model": model,
            "fields": "-",
            "radius": "-",
            "kind": "rho",
            "ss": "-",
            "smooth": _field_smoothing_label(config),
        }

    payload = entries[0]["payload"]
    geometry = "sphere" if payload["pad_subcube_boundary"] else "cube"
    field_kind = "rho+rhat" if payload.get("store_rhat_3d", False) else "rho"
    return {
        "model": model,
        "fields": _field_indices_label(payload["field_indices"]),
        "radius": _radius_label(geometry, payload["subcube_radius"]),
        "kind": field_kind,
        "ss": "-",
        "smooth": _field_smoothing_label(config),
    }


def _variant_info(config_path, selection=None):
    config = candel.load_config(config_path, replace_los_prior=False)
    if selection is not None:
        config.setdefault("model", {})["which_selection"] = selection
    which_run = get_nested(config, "model/which_run", None)
    which_selection = get_nested(config, "model/which_selection", None)
    if which_run is None:
        which_run = get_nested(config, "io/catalogue_name", "?")
        if isinstance(which_run, list):
            which_run = "+".join(which_run)
    which_selection = _selection_label(which_selection)
    cache_dir = _field_cache_dir_from_config(config)
    action = _variant_action(config)
    product = _cache_product_description(config)
    return {
        "config": _short_path(config_path),
        "run": which_run,
        "selection": which_selection,
        "action": action,
        "cache_dir": cache_dir,
        "cache_group": _cache_group_key(config),
        "supersampling": _h0_supersampling_description(config),
        "smoothing": _field_smoothing_description(config),
        **product,
        "note": "",
    }


def _set_cache_status(info, config_path, selection):
    config = candel.load_config(config_path, replace_los_prior=False)
    if selection is not None:
        config.setdefault("model", {})["which_selection"] = selection
    if get_nested(config, "model/which_run", None) in (
            "CH0", "CCHP", "CCHP_CSP", "EDD_TRGB", "EDD_TRGB_grouped"):
        status, note = _h0_cache_file_status(config)
    else:
        status, note = _pv_cache_file_status(config)
    if status == "cached":
        info["action"] = "cached"
        info["note"] = note or "exists"
    elif status in ("incomplete", "unreadable", "stale"):
        info["note"] = note if status == "stale" else status
    elif status == "missing":
        info["note"] = note
    elif note and status is None:
        info["note"] = note


def _set_uncacheable_status(info, config_path, selection):
    """Mark check/cache variants with no cache key as non-runnable."""
    if info["cache_group"] is not None or info["action"] != "check/cache":
        return
    _set_cache_status(info, config_path, selection)
    if info["action"] == "check/cache":
        info["action"] = "skip cache"


def _plan_variants(configs, selections):
    variants = []
    seen_groups = {}
    for config_path in configs:
        if selections:
            for selection in selections:
                info = _variant_info(config_path, selection)
                group = info["cache_group"]
                if group is None:
                    _set_uncacheable_status(info, config_path, selection)
                elif group in seen_groups:
                    first = seen_groups[group]
                    info["duplicate_of"] = first["idx"]
                    info["action"] = f"duplicate #{first['idx']}"
                elif group is not None:
                    _set_cache_status(info, config_path, selection)
                    seen_groups[group] = {
                        "idx": len(variants) + 1,
                        "action": info["action"],
                    }
                variants.append({
                    "config_path": config_path,
                    "selection_override": selection,
                    **info,
                })
        else:
            info = _variant_info(config_path)
            group = info["cache_group"]
            if group is None:
                _set_uncacheable_status(info, config_path, None)
            elif group in seen_groups:
                first = seen_groups[group]
                info["duplicate_of"] = first["idx"]
                info["action"] = f"duplicate #{first['idx']}"
            elif group is not None:
                _set_cache_status(info, config_path, None)
                seen_groups[group] = {
                    "idx": len(variants) + 1,
                    "action": info["action"],
                }
            variants.append({
                "config_path": config_path,
                "selection_override": None,
                **info,
            })
    _annotate_cache_item_ids(variants)
    return variants


def _annotate_cache_item_ids(variants):
    counts = {}
    for variant in variants:
        group = variant.get("cache_group")
        if group is not None:
            counts[group] = counts.get(group, 0) + 1

    item_id = 1
    for variant in variants:
        if variant.get("cache_group") is None or "duplicate_of" in variant:
            continue
        variant["item_id"] = item_id
        variant["configs"] = counts[variant["cache_group"]]
        item_id += 1


def _print_plan(variants):
    if _RANK != 0:
        return
    size_display = int(os.environ.get("CANDEL_FIELD_CACHE_PLAN_SIZE", _SIZE))
    _log("")
    _log("Step 2/2: unique 3D volume-cache products")
    rows = []
    for variant in variants:
        if "item_id" not in variant:
            continue
        rows.append({
            "item": variant["item_id"],
            "run": variant["run"],
            "model": variant["model"],
            "selection": variant["selection"],
            "fields": variant["fields"],
            "radius": variant["radius"],
            "kind": variant["kind"],
            "ss": variant["ss"],
            "smooth": variant["smooth"],
            "status": (
                "missing" if variant["action"] == "check/cache"
                else variant["action"]),
            "configs": variant["configs"],
        })
    if rows:
        for line in _table(
                rows,
                [("item", "item"), ("run", "run"),
                 ("model", "model"), ("selection", "selection"),
                 ("fields", "fields"), ("radius", "radius"),
                 ("kind", "kind"), ("ss", "ss"),
                 ("smooth", "smooth"), ("status", "status"),
                 ("configs", "configs")]
        ).splitlines():
            _log(line)
    else:
        _log("  no cacheable field products inferred.")

    queued_items = [
        variant for variant in variants
        if "item_id" in variant and variant["action"] == "check/cache"
    ]
    queued = len(queued_items)
    cached = sum(
        "item_id" in variant and variant["action"] == "cached"
        for variant in variants)
    duplicates = sum("duplicate_of" in row for row in variants)
    skipped = sum(str(row["action"]).startswith("skip") for row in variants)
    other = len(variants) - len(rows) - duplicates - skipped
    skipped_by_action = {}
    for variant in variants:
        action = str(variant["action"])
        if action.startswith("skip"):
            skipped_by_action[action] = skipped_by_action.get(action, 0) + 1
    skipped_summary = ", ".join(
        f"{action}={count}" for action, count in sorted(
            skipped_by_action.items()))
    if not skipped_summary:
        skipped_summary = "none"
    _log(f"summary: missing={queued}, cached={cached}, "
         f"duplicates={duplicates}, skipped={skipped}, other={other}, "
         f"MPI ranks={size_display}.")
    _log(f"skipped breakdown: {skipped_summary}.")
    if queued:
        ids = [str(variant["item_id"]) for variant in queued_items]
        _log("missing cache item IDs: " + ", ".join(ids))
        _log("submit a subset with `--cache-items 1,3-5`.")


def _runnable_variants(variants, item_ids=None):
    if item_ids is not None:
        valid = {
            variant["item_id"] for variant in variants
            if "item_id" in variant
        }
        unknown = sorted(set(item_ids) - valid)
        if unknown:
            raise ValueError(
                f"Unknown cache item id(s): {unknown}.")
    return [
        variant for variant in variants
        if variant["action"] == "check/cache"
        and (item_ids is None or variant.get("item_id") in item_ids)
    ]


def _write_selection_override(config_path, selection):
    config = candel.load_config(config_path, replace_los_prior=False)
    config.setdefault("model", {})["which_selection"] = selection
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", prefix="field_input_cache_", suffix=".toml",
        delete=False)
    with tmp:
        tomli_w.dump(config, tmp)
    return Path(tmp.name)


def _summarise_loaded(obj):
    if isinstance(obj, list):
        n = len(obj)
        flags = [getattr(frame, "has_volume_density_3d", False)
                 for frame in obj]
        return f"{n} PV dataframe(s), volume_density={flags}"
    if isinstance(obj, dict):
        has_volume = bool(obj.get("has_volume_density_3d", False))
        shape = None
        if "density_3d_fields" in obj:
            shape = tuple(obj["density_3d_fields"].shape)
        return f"H0 data, volume_density={has_volume}, shape={shape}"
    has_volume = getattr(obj, "has_volume_density_3d", False)
    return f"PV dataframe, volume_density={has_volume}"


def _load_for_cache(config_path):
    config = candel.load_config(config_path, replace_los_prior=False)
    which_run = get_nested(config, "model/which_run", None)
    cache_dir = _field_cache_dir_from_config(config)
    _log(f"field cache directory: `{cache_dir}`.")

    run_config_path = config_path
    tmp_config_path = None
    h0_runs = ("CH0", "CCHP", "CCHP_CSP", "EDD_TRGB", "EDD_TRGB_grouped")
    if which_run in h0_runs and "field_indices" in config.get("io", {}):
        config["io"].pop("field_indices", None)
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", prefix="field_input_cache_", suffix=".toml",
            delete=False)
        with tmp:
            tomli_w.dump(config, tmp)
        tmp_config_path = Path(tmp.name)
        run_config_path = tmp_config_path

    old_warmup = os.environ.get("CANDEL_FIELD_CACHE_WARMUP", None)
    os.environ["CANDEL_FIELD_CACHE_WARMUP"] = "1"
    try:
        if which_run == "CH0":
            loaded = candel.pvdata.load_SH0ES_from_config(run_config_path)
        elif which_run in ("CCHP", "CCHP_CSP"):
            loaded = candel.pvdata.load_CCHP_from_config(run_config_path)
        elif which_run == "EDD_TRGB":
            loaded = candel.pvdata.load_EDD_TRGB_from_config(run_config_path)
        elif which_run == "EDD_TRGB_grouped":
            loaded = candel.pvdata.load_EDD_TRGB_grouped_from_config(
                run_config_path)
        else:
            loaded = candel.pvdata.load_PV_dataframes(run_config_path)
    finally:
        if old_warmup is None:
            os.environ.pop("CANDEL_FIELD_CACHE_WARMUP", None)
        else:
            os.environ["CANDEL_FIELD_CACHE_WARMUP"] = old_warmup
        if tmp_config_path is not None:
            try:
                tmp_config_path.unlink()
            except OSError:
                pass

    summary = _summarise_loaded(loaded)
    _log(summary)
    return summary
