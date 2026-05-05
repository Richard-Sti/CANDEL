#!/usr/bin/env python
"""
Warm field-derived cache files without running inference.

Loading the model-ready data is enough to trigger the cache writers in
``candel.pvdata.data``.  This script is intentionally thin: it reuses the same
config loaders as production runs, so the generated cache keys match inference.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
from textwrap import dedent

# This is a CPU cache-prep script; set before importing candel/JAX.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tomli_w  # noqa: E402

import candel  # noqa: E402
from candel import get_nested  # noqa: E402
import candel.pvdata.data as pvdata_mod  # noqa: E402
from candel.pvdata.data import _field_cache_dir_from_config  # noqa: E402


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
    pvdata_mod.fprint = _cache_fprint


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
        return None
    if _variant_action(config) != "check/cache":
        return None

    which_los, los_file = _h0_los_config(config)
    field_kwargs = get_nested(
        config, f"io/reconstruction_main/{which_los}", None)
    if which_los is None or field_kwargs is None:
        return None

    return _json_key({
        "kind": "h0_volume_data",
        "cache_dir": _field_cache_dir_from_config(config),
        "which_run": which_run,
        "which_los": which_los,
        "los_file": los_file,
        "field_kwargs": field_kwargs,
        "which_bias": get_nested(config, "model/which_bias", "linear"),
        "Om": get_nested(config, "model/Om",
                         get_nested(config, "model/Om0", 0.3)),
        "selection_integral_geometry": get_nested(
            config, "model/selection_integral_geometry", "sphere"),
        "selection_integral_grid_radius": get_nested(
            config, "model/selection_integral_grid_radius", None),
        "density_3d_downsample": get_nested(
            config, "model/density_3d_downsample", 1),
        "velocity": _h0_velocity_key(config),
    })


def _variant_action(config):
    """Describe what the warmer will ask the normal data loader to do."""
    which_run = get_nested(config, "model/which_run", None)
    h0_runs = ("CH0", "CCHP", "CCHP_CSP", "EDD_TRGB", "EDD_TRGB_grouped")
    if which_run not in h0_runs:
        return "check/cache"

    which_sel = get_nested(config, "model/which_selection", None)
    if which_sel in (None, "none"):
        return "skip 3D"
    if not get_nested(config, "model/use_reconstruction", False):
        return "skip recon"
    if not pvdata_mod._field_cache_enabled_from_config(config):
        return "cache off"
    return "check/cache"


def _resolve_los_path(path, which_los):
    if path is not None and which_los is not None:
        path = path.replace("<X>", which_los)
    return path


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

    with pvdata_mod.File(los_read_path, "r") as f:
        if "field_indices" in f:
            field_indices = f["field_indices"][:]
        else:
            field_indices = pvdata_mod.np.arange(f["los_density"].shape[0])

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

    galaxy_bias = get_nested(config, "model/which_bias", "linear")
    geometry = get_nested(
        config, "model/selection_integral_geometry", "sphere")
    grid_radius = get_nested(
        config, "model/selection_integral_grid_radius", None)
    downsample = get_nested(config, "model/density_3d_downsample", 1)
    load_velocity = _h0_velocity_key(config) == "velocity"
    payload = {
        "kind": "h0_volume_data",
        "field_name": which_los,
        "field_kwargs": field_kwargs,
        "field_indices": pvdata_mod._jsonable(
            pvdata_mod.np.asarray(field_indices)),
        "galaxy_bias": galaxy_bias,
        "density_mode": pvdata_mod._volume_density_mode(galaxy_bias),
        "Om0": float(get_nested(config, "model/Om",
                                get_nested(config, "model/Om0", 0.3))),
        "subcube_radius": grid_radius,
        "downsample": int(downsample),
        "load_velocity": bool(load_velocity),
        "geometry": geometry,
        "sources": source_meta,
    }
    cache_path = pvdata_mod._field_cache_path(
        _field_cache_dir_from_config(config), "h0_volume_data", payload)
    required = [
        "density_3d_fields", "log_r_3d", "log_dV_3d",
        "mu_at_h1_3d", "zcosmo_3d"]
    if geometry == "sphere" and grid_radius is not None:
        required.append("log_volume_weight_3d")
    if load_velocity:
        required.extend([
            "vrad_3d_fields", "rhat_x_3d", "rhat_y_3d", "rhat_z_3d"])

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return "missing", _short_path(cache_path)
    try:
        with pvdata_mod.np.load(cache_path, allow_pickle=False) as f:
            missing = [key for key in required if key not in f.files]
    except Exception as exc:
        return "unreadable", str(exc)
    if missing:
        return "incomplete", f"missing {missing}"
    return "cached", _short_path(cache_path)


def _variant_info(config_path, selection=None):
    config = candel.load_config(config_path, replace_los_prior=False)
    if selection is not None:
        config.setdefault("model", {})["which_selection"] = selection
    which_run = get_nested(config, "model/which_run", "?")
    which_selection = get_nested(config, "model/which_selection", "-")
    cache_dir = _field_cache_dir_from_config(config)
    action = _variant_action(config)
    return {
        "config": _short_path(config_path),
        "run": which_run,
        "selection": which_selection,
        "action": action,
        "cache_dir": cache_dir,
        "cache_group": _cache_group_key(config),
        "note": "",
    }


def _set_cache_status(info, config_path, selection):
    config = candel.load_config(config_path, replace_los_prior=False)
    if selection is not None:
        config.setdefault("model", {})["which_selection"] = selection
    status, note = _h0_cache_file_status(config)
    if status == "cached":
        info["action"] = "cached"
        info["note"] = "exists"
    elif status in ("incomplete", "unreadable"):
        info["note"] = status
    elif note and status is None:
        info["note"] = note


def _plan_variants(configs, selections):
    variants = []
    seen_groups = {}
    for config_path in configs:
        if selections:
            for selection in selections:
                info = _variant_info(config_path, selection)
                group = info["cache_group"]
                if group in seen_groups:
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
            if group in seen_groups:
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
    _log("")
    _log("Field-cache warmup plan")
    rows = []
    for i, variant in enumerate(variants, 1):
        rows.append({
            "#": i,
            "run": variant["run"],
            "selection": variant["selection"],
            "action": variant["action"],
            "note": variant["note"],
            "config": variant["config"],
        })
    for line in _table(
            rows,
            [("#", "#"), ("run", "run"), ("selection", "selection"),
             ("action", "action"), ("note", "note"), ("config", "config")]
    ).splitlines():
        _log(line)
    queued = sum(row["action"] == "check/cache" for row in variants)
    duplicates = sum("duplicate_of" in row for row in variants)
    cached = sum(row["action"] == "cached" for row in variants)
    skipped = sum(str(row["action"]).startswith("skip") for row in variants)
    _log(f"summary: {queued} unique check/cache, {cached} cached, "
         f"{duplicates} duplicate(s), {skipped} skip 3D, "
         f"{_SIZE} MPI rank(s).")


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

    if which_run == "CH0":
        loaded = candel.pvdata.load_SH0ES_from_config(config_path)
    elif which_run in ("CCHP", "CCHP_CSP"):
        loaded = candel.pvdata.load_CCHP_from_config(config_path)
    elif which_run == "EDD_TRGB":
        loaded = candel.pvdata.load_EDD_TRGB_from_config(config_path)
    elif which_run == "EDD_TRGB_grouped":
        loaded = candel.pvdata.load_EDD_TRGB_grouped_from_config(config_path)
    else:
        loaded = candel.pvdata.load_PV_dataframes(config_path)

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
              warm_field_cache.py scripts/runs/configs/config_shoes.toml --selection SN_magnitude,redshift
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
        try:
            summary = _load_for_cache(str(run_config))
            _log(f"[{i}/{len(variants)}] done: {summary}")
        finally:
            if selection is not None:
                run_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
