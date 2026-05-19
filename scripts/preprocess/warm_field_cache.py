#!/usr/bin/env python
"""
Warm field-derived cache files without running inference.

Loading the model-ready data is enough to trigger the cache writers in
``candel.pvdata``.  This script is intentionally thin: it reuses the same
config loaders as production runs, so the generated cache keys match inference.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from textwrap import dedent

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
from candel.pvdata import frame as frame_mod  # noqa: E402
from candel.pvdata import los as los_mod  # noqa: E402
from candel.pvdata import volume_density as volume_density_mod  # noqa: E402
from candel.pvdata.field_cache import _field_cache_dir_from_config  # noqa: E402
from candel.util import SPEED_OF_LIGHT  # noqa: E402

pvdata_mod = SimpleNamespace(
    File=File,
    SPEED_OF_LIGHT=SPEED_OF_LIGHT,
    _field_cache_enabled_from_config=(
        field_cache_mod._field_cache_enabled_from_config),
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


def _bcast(value):
    if _COMM is None:
        return value
    return _COMM.bcast(value, root=0)


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


def _split_values(values):
    out = []
    for value in values or []:
        out.extend(v.strip() for v in value.split(",") if v.strip())
    return out


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
            get_nested(config, "io/SH0ES/which_host_los", None),
            get_nested(config, "io/PV_main/SH0ES/los_file", None),
        )
    if which_run in ("CCHP", "CCHP_CSP"):
        return (
            get_nested(config, "io/which_host_los",
                       get_nested(config, "io/CCHP/which_host_los", None)),
            get_nested(config, "io/CCHP/los_file", None),
        )
    if which_run in ("EDD_TRGB", "EDD_TRGB_grouped"):
        return (
            get_nested(config, "io/which_host_los",
                       get_nested(
                           config,
                           f"io/PV_main/{which_run}/which_host_los", None)),
            get_nested(config, f"io/PV_main/{which_run}/los_file", None),
        )
    return None, None


def _resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


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
    if which_sel in ("redshift", "SN_magnitude_redshift"):
        return "velocity"
    if which_run == "CH0" and which_sel == "SN_magnitude_or_redshift_Nmag":
        n_mag = get_nested(config, "model/num_hosts_selection_mag", None)
        n_hosts = _sh0es_num_hosts(config)
        if type(n_mag) is int and n_hosts is not None:
            return "velocity" if n_mag < n_hosts else "density"
        return f"mixed_Nmag={n_mag}"
    return "density"


def _cache_group_key(config):
    which_run = get_nested(config, "model/which_run", None)
    if which_run not in ("CH0", "CCHP", "CCHP_CSP", "EDD_TRGB",
                         "EDD_TRGB_grouped"):
        entries, _ = _pv_volume_cache_entries(config)
        if not entries:
            return None
        return _json_key({
            "kind": "pv_volume_density_3d_set",
            "cache_dir": _field_cache_dir_from_config(config),
            "entries": [entry["payload"] for entry in entries],
        })
    if _variant_action(config) != "check/cache":
        return None

    which_los, los_file = _h0_los_config(config)
    field_kwargs = get_nested(
        config, f"io/reconstruction_main/{which_los}", None)
    if which_los is None or field_kwargs is None:
        return None
    los_data_path = _resolve_los_path(los_file, which_los)
    if los_data_path is None or not _resolve_repo_path(los_data_path).exists():
        return None
    field_indices = _read_h0_field_indices(los_data_path)

    sampling = pvdata_mod._h0_volume_cache_sampling_payload(
        get_nested(config, "model/density_3d_subsample_fraction", 1.0),
        get_nested(config, "model/density_3d_subsample_seed", 42))
    supersampling = _h0_supersampling_payload(
        config, which_los, field_kwargs, field_indices)
    return _json_key({
        "kind": "h0_volume_data",
        "cache_dir": _field_cache_dir_from_config(config),
        "which_los": which_los,
        "los_file": los_file,
        "selection_integral_geometry": get_nested(
            config, "model/selection_integral_geometry", "sphere"),
        "selection_integral_grid_radius": get_nested(
            config, "model/selection_integral_grid_radius", None),
        "sampling": sampling,
        "supersampling": supersampling,
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

    entries = {}
    for name in names:
        io_name = "CF4_mock" if str(name).startswith("CF4_mock") else name
        io_section = get_nested(config, f"io/{io_name}", None)
        if io_section is None or "los_file" not in io_section:
            return [], f"missing LOS config: {name}"
        los_path = _resolve_los_path(io_section["los_file"], reconstruction)
        los_read_path = _resolve_repo_path(los_path)
        if not los_read_path.exists():
            return [], f"missing LOS: {_short_path(los_read_path)}"
        with pvdata_mod.File(los_read_path, "r") as f:
            if "field_indices" in f:
                field_indices = f["field_indices"][:]
            else:
                field_indices = pvdata_mod.np.arange(f["los_density"].shape[0])

        source_meta = []
        for nsim in field_indices:
            loader_kwargs = dict(recon_kwargs)
            loader_kwargs["nsim"] = int(nsim)
            loader = pvdata_mod.name2field_loader(
                reconstruction)(**loader_kwargs)
            source_meta.append(pvdata_mod._field_source_metadata(loader))
        payload = {
            "kind": "pv_volume_density_3d",
            "loader_name": reconstruction,
            "loader_kwargs": pvdata_mod._jsonable(recon_kwargs),
            "field_indices": pvdata_mod._jsonable(
                pvdata_mod.np.asarray(field_indices)),
            "downsample": downsample,
            "subcube_radius": radius,
            "pad_subcube_boundary": bool(pad_boundary),
            "voxel_subsample_fraction": float(voxel_subsample_fraction),
            "voxel_subsample_seed": int(voxel_subsample_seed),
            "store_rhat_3d": store_rhat_3d,
            "sources": source_meta,
        }
        cache_path = Path(pvdata_mod._field_cache_path(
            cache_dir, "pv_volume_density_3d", payload))
        entries[str(cache_path)] = {
            "payload": payload,
            "path": cache_path,
        }

    return list(entries.values()), ""


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


def _resolve_los_path(path, which_los):
    if path is not None and which_los is not None:
        path = path.replace("<X>", which_los)
    return path


def _h0_supersampling_payload(config, which_los, field_kwargs,
                              field_indices):
    factor, radius, target_dx = (
        pvdata_mod._h0_volume_supersampling_from_config(config))
    if target_dx is not None and radius > 0.0:
        kwargs = dict(field_kwargs)
        kwargs["nsim"] = int(field_indices[0])
        loader = pvdata_mod.name2field_loader(which_los)(**kwargs)
        dx = pvdata_mod._field_loader_native_dx(loader)
        factor = pvdata_mod._h0_volume_resolved_supersample_factor(
            dx, factor, target_dx)
    return pvdata_mod._h0_volume_cache_supersampling_payload(factor, radius)


def _read_h0_field_indices(los_data_path):
    los_read_path = _resolve_repo_path(los_data_path)
    with pvdata_mod.File(los_read_path, "r") as f:
        if "field_indices" in f:
            return f["field_indices"][:]
        return pvdata_mod.np.arange(f["los_density"].shape[0])


def _h0_cache_file_status(config):
    if _variant_action(config) != "check/cache":
        return None, None

    which_los, los_file = _h0_los_config(config)
    los_data_path = _resolve_los_path(los_file, which_los)
    if which_los is None or los_data_path is None:
        return None, "no LOS"
    los_read_path = _resolve_repo_path(los_data_path)
    if not los_read_path.exists():
        return None, f"missing LOS: {_short_path(los_read_path)}"

    field_indices = _read_h0_field_indices(los_data_path)

    field_kwargs = get_nested(
        config, f"io/reconstruction_main/{which_los}", None)
    if field_kwargs is None:
        return None, f"missing reconstruction: {which_los}"

    source_meta = []
    for nsim in field_indices:
        kwargs = dict(field_kwargs)
        kwargs["nsim"] = int(nsim)
        loader = pvdata_mod.name2field_loader(which_los)(**kwargs)
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
        "kind": "h0_volume_data",
        "field_name": which_los,
        "field_indices": pvdata_mod._jsonable(
            pvdata_mod.np.asarray(field_indices)),
        "subcube_radius": grid_radius,
        "geometry": geometry,
        "sources": source_meta,
    }
    base_payload.update(pvdata_mod._h0_volume_cache_sampling_payload(
        subsample_fraction, subsample_seed))
    supersampling = _h0_supersampling_payload(
        config, which_los, field_kwargs, field_indices)
    base_payload.update(supersampling)
    density_payload = {**base_payload, "load_velocity": False}
    velocity_payload = {**base_payload, "load_velocity": True}
    cache_dir = _field_cache_dir_from_config(config)
    density_cache_path = Path(pvdata_mod._field_cache_path(
        cache_dir, "h0_volume_data", density_payload))
    velocity_cache_path = Path(pvdata_mod._field_cache_path(
        cache_dir, "h0_volume_data", velocity_payload))
    density_required = ["rho_3d_fields", "r_3d", "log_dV_3d"]
    if ((geometry == "sphere" and grid_radius is not None)
            or supersampling):
        density_required.append("log_volume_weight_3d")
    velocity_required = [
        "vrad_3d_fields", "rhat_x_3d", "rhat_y_3d", "rhat_z_3d"]

    if not density_cache_path.exists():
        return "missing", _short_path(density_cache_path)
    try:
        with pvdata_mod.np.load(density_cache_path, allow_pickle=False) as f:
            missing = [key for key in density_required if key not in f.files]
    except Exception as exc:
        return "unreadable", str(exc)
    if missing:
        return "incomplete", f"missing {missing}"

    if load_velocity:
        if not velocity_cache_path.exists():
            return "missing", _short_path(velocity_cache_path)
        try:
            with pvdata_mod.np.load(
                    velocity_cache_path, allow_pickle=False) as f:
                missing = [
                    key for key in velocity_required if key not in f.files]
        except Exception as exc:
            return "unreadable", str(exc)
        if missing:
            return "incomplete", f"missing {missing}"
    if load_velocity:
        return "cached", f"{_short_path(density_cache_path)} + velocity"
    return "cached", _short_path(density_cache_path)


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
            with pvdata_mod.np.load(path, allow_pickle=False) as f:
                missing_keys = [
                    key for key in (
                        "rho_fields", "log_r_3d", "log_dV_3d",
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
        except Exception as exc:
            return "unreadable", str(exc)
        if missing_keys:
            return "incomplete", f"missing {missing_keys}"

    if missing:
        return "missing", f"missing {missing}/{len(entries)}"
    return "cached", f"{len(entries)} file(s)"


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
    if which_selection is None:
        which_selection = get_nested(config, "pv_model/kind", "-")
    cache_dir = _field_cache_dir_from_config(config)
    action = _variant_action(config)
    return {
        "config": _short_path(config_path),
        "run": which_run,
        "selection": which_selection,
        "action": action,
        "cache_dir": cache_dir,
        "cache_group": _cache_group_key(config),
        "supersampling": _h0_supersampling_description(config),
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
    elif status in ("incomplete", "unreadable"):
        info["note"] = status
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
                    if first["action"] == "cached":
                        info["action"] = "cached"
                        info["note"] = "exists"
                    else:
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
                if first["action"] == "cached":
                    info["action"] = "cached"
                    info["note"] = "exists"
                else:
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
    return variants


def _read_requested_variants(args):
    configs = []
    inputs = [_resolve_cli_path(p) for p in args.inputs]
    task_file = args.task_file
    if task_file is None and inputs and _looks_like_task_file(inputs[0]):
        task_file = inputs.pop(0)

    if task_file is not None:
        task_file = _resolve_cli_path(task_file)
        configs.extend(_read_task_file(task_file, args.tasks))
    configs.extend(inputs)

    if not configs:
        raise ValueError("provide a task file or at least one config")
    for config_path in configs:
        if not config_path.exists():
            raise FileNotFoundError(config_path)

    return _plan_variants(configs, _split_values(args.selection))


def _print_plan(variants):
    if _RANK != 0:
        return
    size_display = int(os.environ.get("CANDEL_FIELD_CACHE_PLAN_SIZE", _SIZE))
    _log("")
    _log("Field-cache warmup plan")
    rows = []
    for i, variant in enumerate(variants, 1):
        rows.append({
            "#": i,
            "run": variant["run"],
            "selection": variant["selection"],
            "action": variant["action"],
            "supersampling": variant["supersampling"],
            "note": variant["note"],
            "config": variant["config"],
        })
    for line in _table(
            rows,
            [("#", "#"), ("run", "run"), ("selection", "selection"),
             ("action", "action"), ("supersampling", "supersampling"),
             ("note", "note"), ("config", "config")]
    ).splitlines():
        _log(line)
    queued = sum(row["action"] == "check/cache" for row in variants)
    duplicates = sum("duplicate_of" in row for row in variants)
    cached = sum(row["action"] == "cached" for row in variants)
    skipped = sum(str(row["action"]).startswith("skip") for row in variants)
    _log(f"summary: {queued} unique check/cache, {cached} cached, "
         f"{duplicates} duplicate(s), {skipped} skip(s), "
         f"{size_display} MPI rank(s).")


def _runnable_variants(variants):
    return [variant for variant in variants
            if variant["action"] == "check/cache"]


def _write_selection_override(config_path, selection):
    config = candel.load_config(config_path, replace_los_prior=False)
    config.setdefault("model", {})["which_selection"] = selection
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", prefix="warm_field_cache_", suffix=".toml", delete=False)
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
    if (which_run in ("CH0", "CCHP", "CCHP_CSP", "EDD_TRGB",
                     "EDD_TRGB_grouped")
            and "field_indices" in config.get("io", {})):
        config["io"].pop("field_indices", None)
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", prefix="warm_field_cache_", suffix=".toml",
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


def main():
    _init_mpi_info()
    _install_quiet_reader_logs()

    parser = argparse.ArgumentParser(
        description="Generate field cache files by loading CANDEL configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            examples:
              warm_field_cache.py scripts/runs/tasks_CH0_main.txt
              warm_field_cache.py scripts/runs/tasks_CH0_main.txt --tasks 12-23
              warm_field_cache.py scripts/runs/configs/config_CH0.toml \
--selection SN_magnitude,redshift
            """))
    parser.add_argument(
        "inputs", nargs="*", type=Path,
        help=("Task file as first argument, or config TOML files to warm."))
    parser.add_argument(
        "--task-file", type=Path,
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--tasks", default=None,
        help="Comma-separated task ids/ranges from the task file, e.g. 3,5-8.")
    parser.add_argument(
        "--selection", action="append", default=[],
        help=("Override model.which_selection for H0 configs. May be repeated "
              "or comma-separated, e.g. --selection SN_magnitude,redshift."))
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Print the cache warmup plan and exit without loading data.")
    args = parser.parse_args()

    if _RANK == 0:
        try:
            variants = _read_requested_variants(args)
            error = None
        except Exception as exc:
            variants = []
            error = str(exc)
    else:
        variants = []
        error = None
    error, variants = _bcast((error, variants))
    if error is not None:
        parser.error(error)

    _print_plan(variants)
    variants = _runnable_variants(variants)

    if args.plan_only:
        if not variants:
            raise SystemExit(3)
        return

    if not variants:
        _log("")
        _log("Nothing to warm.")
        return

    for i, variant in enumerate(variants, 1):
        config_path = variant["config_path"]
        selection = variant["selection_override"]
        run_config = config_path
        label = str(config_path)
        if selection is not None:
            label = f"{label} [which_selection={selection}]"
            run_config = _write_selection_override(config_path, selection)

        _log("")
        _log(f"[{i}/{len(variants)}] warming {variant['run']} "
             f"selection={variant['selection']} from `{label}`.")
        if variant["supersampling"]:
            _log(f"  supersampling: {variant['supersampling']}")
        try:
            summary = _load_for_cache(str(run_config))
            _log(f"[{i}/{len(variants)}] done: {summary}")
        finally:
            if selection is not None:
                run_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
