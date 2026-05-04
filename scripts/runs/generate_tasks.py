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
Generate configuration files and task lists for batch parameter inference runs.

This script automates the setup of parameter inference experiments for CANDEL
by generating `.toml` configuration files based on a central config template
and a set of parameter overrides. It also writes out a task list for downstream
execution (e.g., with SLURM or local scripting).

Main capabilities:
------------------
- Expand a grid of override parameters (single values or lists)
- Apply overrides to nested config keys using slash-delimited paths (e.g.
  "model/priors/beta")
- Automatically generate descriptive output tags based on key config flags
- Create output directories and write `.toml` configs to disk
- Log all generated tasks in `tasks_<index>.txt` for batch submission

Special handling for joint catalogues:
--------------------------------------
If *both* `inference/model` and `io/catalogue_name` are provided as lists of
equal length, they are treated as **paired inputs**, representing joint
likelihood models with separate data vectors and submodels per catalogue.

For example:
    "inference/model": ["TFRModel", "PantheonPlusModel"]
    "io/catalogue_name": ["CF4_W1", "Pantheon"]

This setup will yield a single configuration where the models and catalogues
are interpreted jointly (e.g., by `JointPVModel`). All other override
parameters (e.g. priors, flags) will be expanded via Cartesian product
*independently* of the model/catalogue pair.

Note:
    If `inference/model` and `io/catalogue_name` are both lists but not of
    equal length, the script will raise an error to prevent unintended
    mismatches.

Usage:
------
1. Add or edit a named sweep in ``task_specs.py``.
2. Run the script:
       $ python generate_tasks.py build 0
   or, for backward compatibility:
       $ python generate_tasks.py 0
3. Use the generated `tasks_0.txt` with `submit.sh` to run the tasks. Run
   ``python generate_tasks.py list`` to see registered task indices.

Typical output:
- One `.toml` file per combination of overrides
- A summary task list with config paths for tracking and reproducibility
- Descriptive filenames/tags that capture salient flags (e.g. catalogue, MNR,
  selection mode, Nmag split for mixed selections, Vext settings)

This script is meant to streamline robust, reproducible inference workflows in
CANDEL.
"""
import re
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from os import makedirs
from os.path import join
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from task_specs import TASK_SPECS


RUN_DIR = Path(__file__).resolve().parent
CANDEL_ROOT = RUN_DIR.parent.parent
TASK_INDEX_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
VERBOSE = True

fprint = None
get_nested = None
load_config = None
replace_prior_with_delta = None
tomli_w = None


def load_toml_writer():
    """Import TOML writing support only for commands that need it."""
    global tomli_w
    if tomli_w is None:
        import tomli_w as _tomli_w
        tomli_w = _tomli_w


def load_candel_helpers():
    """Import heavier CANDEL helpers only for build/dry-run commands."""
    global fprint, get_nested, load_config, replace_prior_with_delta
    if load_config is None:
        from candel import (  # noqa
            fprint as _fprint,
            get_nested as _get_nested,
            load_config as _load_config,
            replace_prior_with_delta as _replace_prior_with_delta,
        )
        fprint = _fprint
        get_nested = _get_nested
        load_config = _load_config
        replace_prior_with_delta = _replace_prior_with_delta


def log(message):
    """Emit generator diagnostics when verbose output is enabled."""
    if VERBOSE and fprint is not None:
        fprint(message)


# Keys that must come from local_config.toml at job runtime, not baked into
# generated configs. Baking them in defeats the portability mechanism in
# load_config, which injects local_config.toml only for keys not already set.
_MACHINE_KEYS = {
    "root_main", "root_data", "root_results",
    "python_exec", "machine", "modules", "modules_gpu",
    "use_frozen", "gpu_ld_library_path", "watcher_dir",
}


def load_local_config():
    """Load machine-specific settings from local_config.toml at project root.

    List/dict values (e.g. ``gpu_ld_library_path``) are dropped: downstream
    ``expand_override_grid`` treats every list as a Cartesian product
    dimension, but these entries are runtime environment, not overrides.
    Machine path keys are also excluded so they are never baked into generated
    configs — they are injected at job runtime from the executing machine's
    local_config.toml.
    """
    local_config_path = CANDEL_ROOT / "local_config.toml"
    if local_config_path.exists():
        with open(local_config_path, 'rb') as f:
            cfg = tomllib.load(f)
        return {k: v for k, v in cfg.items()
                if not isinstance(v, (list, dict))
                and k not in _MACHINE_KEYS}
    return {}


def overwrite_config(config, key, value):
    """Return a new config dict with a nested key overwritten."""
    new_config = deepcopy(config)
    keys = key.split("/")
    d = new_config
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]

    log(f"overwriting config['{'/'.join(keys)}'] = {value}")
    d[keys[-1]] = value
    return new_config


def overwrite_subtree(config, key_path, subtree):
    """
    Overwrite a nested subtree (dict) at a slash-separated key path.
    """
    new_config = deepcopy(config)
    keys = key_path.split("/")
    d = new_config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = subtree
    log(f"overwriting subtree config['{'/'.join(keys)}'] = {subtree}")
    return new_config


def _is_active(value):
    """Check if value is active (not None or 'none')."""
    if value is None:
        return False
    return str(value).lower() != "none"


def _is_delta_prior(prior):
    """Check if prior is a delta distribution."""
    return isinstance(prior, dict) and prior.get("dist") == "delta"


def generate_dynamic_tag(config, base_tag="default"):
    """Generate a descriptive tag string based on selected config values."""
    parts = []
    which_run = get_nested(config, "model/which_run", None)

    if which_run in ("CH0", "CCHP", "CCHP_CSP", "EDD_TRGB",
                      "EDD_TRGB_grouped"):
        model_name = which_run
        catalogue = which_run
        parts.append(which_run)
    else:
        model_name = get_nested(config, "inference/model", None)
        catalogue = get_nested(config, "io/catalogue_name", None)
        if catalogue:
            if isinstance(catalogue, list):
                parts.append(",".join(catalogue))
            else:
                parts.append(str(catalogue))

    if get_nested(config, "pv_model/kind", "").startswith("precomputed_los"):
        parts.append(get_nested(config, "pv_model/galaxy_bias", ""))

    smooth_target = get_nested(config, "pv_model/smooth_target", None)
    if _is_active(smooth_target):
        parts.append(f"smooth{smooth_target}")

    if model_name and "TFR" in model_name:
        if not get_nested(config, "model/marginalize_eta", True):
            parts.append("eta_sampled")

    # Zeropoint dipole if it's not a delta distribution
    zeropoint_dip_prior = get_nested(
        config, "model/priors/zeropoint_dipole", None)
    if isinstance(zeropoint_dip_prior, dict):
        if zeropoint_dip_prior.get("dist") == "vector_components_uniform":
            parts.append("zeropoint_dipole_UnifComponents")
        elif not _is_delta_prior(zeropoint_dip_prior):
            parts.append("zeropoint_dipole")

    if _is_delta_prior(get_nested(config, "model/priors/Vext", None)):
        parts.append("noVext")

    _mono = get_nested(config, "model/which_Vext_monopole", "none")
    if _mono == "none" and get_nested(
            config, "model/use_Vext_monopole", False):
        _mono = "constant"
    if _mono == "constant":
        parts.append("Vmono")
    elif _mono == "sigmoid":
        parts.append("Vmono_sigmoid")

    if get_nested(config, "model/use_Vext_quadrupole", False):
        parts.append("Vquad")

    if get_nested(config, "model/use_Vext_octupole", False):
        parts.append("Voct")

    which_Vext = get_nested(config, "pv_model/which_Vext", None)
    if which_Vext is not None and which_Vext != "constant":
        parts.append(f"Vext_{which_Vext}")

    which_dist_prior = get_nested(
        config, "pv_model/which_distance_prior", "empirical")
    if which_dist_prior != "empirical":
        parts.append(f"rprior-{which_dist_prior}")

    # Only include beta/b1 info if using precomputed LOS (reconstruction)
    pv_kind = get_nested(config, "pv_model/kind", "")
    if pv_kind.startswith("precomputed_los"):
        beta_prior = get_nested(config, "model/priors/beta", None)
        if _is_delta_prior(beta_prior) and beta_prior.get("value") != 0.:
            parts.append(f"beta_{beta_prior.get('value')}")

        b1_prior = get_nested(config, "model/priors/b1", None)
        if _is_delta_prior(b1_prior):
            parts.append(f"b1_{b1_prior.get('value')}")

    dust_model = get_nested(config, f"io/{catalogue}/dust_model", None)
    if _is_active(dust_model):
        parts.append(f"dust-{dust_model}")

    # Run-specific tags
    if which_run == "CH0":
        which_sel = get_nested(config, "model/which_selection", None)
        if _is_active(which_sel):
            parts.append(f"sel-{which_sel}")
            if which_sel == "SN_magnitude_or_redshift_Nmag":
                parts.append(
                    f"Nmag{get_nested(config, 'model/num_hosts_selection_mag', None)}")  # noqa

        if get_nested(config, "model/use_uniform_mu_host_priors", False):
            parts.append("uniform_mu_host")

        r_prior = get_nested(config, "model/which_distance_prior", "volume")
        if r_prior != "volume":
            parts.append(r_prior)

        if not get_nested(config, "model/use_Cepheid_host_redshift", True):
            parts.append("no_Cepheid_redshift")

        use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        use_pv_covmat = get_nested(
            config, "model/use_fiducial_Cepheid_host_PV_covariance", False)
        Vext_prior = get_nested(config, "model/priors/Vext", None)
        if not use_reconstruction and not use_pv_covmat \
                and not _is_delta_prior(Vext_prior):
            parts.append("Vext")

        if use_reconstruction:
            which_los = get_nested(config, "io/SH0ES/which_host_los", None)
            parts.append(which_los)
            beta_prior = get_nested(config, "model/priors/beta", None)
            if isinstance(which_los, str) and "manticore" in which_los.lower():
                if not _is_delta_prior(beta_prior):
                    parts.append("beta_free")
            if get_nested(config, "model/use_density_dependent_sigma_v", False):  # noqa
                parts.append("sigv_rho")

        if use_pv_covmat:
            parts.append("PV_covmat")

        if get_nested(config, "model/use_PV_covmat_scaling", False):
            parts.append("PV_covmat_scaling")

        if get_nested(config, "model/weight_selection_by_covmat_Neff", False):
            parts.append("weight_by_Neff")

    elif which_run in ("CCHP", "CCHP_CSP"):
        which_sel = get_nested(config, "model/which_selection", None)
        if _is_active(which_sel):
            parts.append(f"sel-{which_sel}")
        if get_nested(config, "model/infer_sel", False):
            parts.append("infer_sel")
        if get_nested(config, "model/use_reconstruction", False):
            parts.append(get_nested(config, "io/which_host_los", None))
        redshift_kind = get_nested(
            config, "io/CCHP_redshift_source/kind", "cz_cmb")
        if redshift_kind != "cz_cmb":
            parts.append(redshift_kind)

    elif which_run in ("EDD_TRGB", "EDD_TRGB_grouped"):
        which_sel = get_nested(config, "model/which_selection", None)
        if _is_active(which_sel):
            parts.append(f"sel-{which_sel}")
        if get_nested(config, "model/use_reconstruction", False):
            parts.append(get_nested(
                config, "io/which_host_los",
                get_nested(config,
                           f"io/PV_main/{which_run}/which_host_los", None)))

    shared = get_nested(config, "inference/shared_params", None)
    if _is_active(shared):
        if isinstance(shared, list):
            shared_str = "+".join(shared)
        else:
            shared_str = str(shared).replace(",", "+")
        parts.append(f"shared-{shared_str}")

    if base_tag != "default":
        parts.append(base_tag)

    return "_".join(p for p in parts if p)


def expand_override_grid(overrides):
    """Expand override grid into a list of flat key-value combinations."""
    model_key = "inference/model"
    cat_key = "io/catalogue_name"

    has_model_catalogue_lists = (
        model_key in overrides
        and cat_key in overrides
        and isinstance(overrides[model_key], list)
        and isinstance(overrides[cat_key], list)
    )

    if has_model_catalogue_lists:
        if len(overrides[model_key]) != len(overrides[cat_key]):
            raise ValueError(
                f"`{model_key}` and `{cat_key}` must have the same length "
                "when both are lists.")

    is_joint_model = has_model_catalogue_lists

    if is_joint_model:
        # Extract grouped keys
        grouped_models = overrides[model_key]
        grouped_cats = overrides[cat_key]

        # Collect remaining keys
        other_keys = {
            k: v if isinstance(v, list) else [v]
            for k, v in overrides.items()
            if k not in (model_key, cat_key)
        }

        # Cartesian expansion over non-grouped keys
        if not other_keys:
            return [{
                model_key: grouped_models,
                cat_key: grouped_cats
            }]

        keys = list(other_keys.keys())
        value_lists = list(other_keys.values())
        combos = product(*value_lists)

        results = []
        for combo in combos:
            entry = {
                model_key: list(grouped_models),
                cat_key: list(grouped_cats),
            }
            entry.update(dict(zip(keys, combo)))
            results.append(entry)
        return results

    # Fallback: standard Cartesian product for everything
    if not overrides:
        return [{}]

    keys = list(overrides.keys())
    value_lists = [
        v if isinstance(v, list) else [v] for v in overrides.values()]
    return [dict(zip(keys, combo)) for combo in product(*value_lists)]


@dataclass(frozen=True)
class SweepSpec:
    """Validated task-list sweep definition."""
    name: str
    description: str
    config_path: Path
    tag: str
    common: dict
    datasets: list
    expected_tasks: int | None = None


def get_sweep_spec(tasks_index):
    """Return a validated sweep spec for a named task index."""
    if not TASK_INDEX_RE.fullmatch(tasks_index):
        raise ValueError(
            f"Invalid tasks_index '{tasks_index}'. Use only letters, numbers, "
            "underscore, dash, and dot.")
    if tasks_index not in TASK_SPECS:
        available = ", ".join(sorted(TASK_SPECS))
        raise ValueError(
            f"Unknown tasks_index '{tasks_index}'. Available specs: {available}")

    raw = TASK_SPECS[tasks_index]
    config_path = (RUN_DIR / raw["config_path"]).resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Base config for '{tasks_index}' does not exist: {config_path}")

    datasets = raw.get("datasets") or [{}]
    return SweepSpec(
        name=tasks_index,
        description=raw.get("description", ""),
        config_path=config_path,
        tag=raw.get("tag", "default"),
        common=raw.get("common", {}),
        datasets=datasets,
        expected_tasks=raw.get("expected_tasks"),
    )


def build_override_combinations(spec, local_cfg):
    """Build all override combinations for a sweep spec."""
    all_override_combinations = []
    for dataset in spec.datasets:
        overrides = {**local_cfg, **spec.common, **dataset}
        all_override_combinations.extend(expand_override_grid(overrides))

    if spec.expected_tasks is not None:
        n_tasks = len(all_override_combinations)
        if n_tasks != spec.expected_tasks:
            raise ValueError(
                f"Spec '{spec.name}' produced {n_tasks} tasks, expected "
                f"{spec.expected_tasks}.")

    return all_override_combinations


def apply_pv_kind_rules(config, key, value):
    """Apply compatibility rules implied by the selected PV model kind."""
    if key != "pv_model/kind" or value.startswith("precomputed_los"):
        return config

    # No reconstruction: force unity galaxy bias and delta reconstruction priors.
    config = overwrite_config(config, "pv_model/galaxy_bias", "unity")
    config = overwrite_config(config, "model/use_reconstruction", False)
    for prior_name, prior_value in (
            ("alpha", 1.), ("beta", 0.), ("b1", 0.), ("b2", 0.),
            ("b3", 0.), ("delta_b1", 0.)):
        config = replace_prior_with_delta(config, prior_name, prior_value)
    return config


def apply_overrides(config, override_set):
    """Apply one flat override dictionary to a loaded base config."""
    local_config = deepcopy(config)
    for key, value in override_set.items():
        local_config = apply_pv_kind_rules(local_config, key, value)

        if isinstance(value, dict):
            local_config = overwrite_subtree(local_config, key, value)
        else:
            local_config = overwrite_config(local_config, key, value)

    return local_config


def apply_los_runtime_rules(config, override_set):
    """Apply LOS-field rules that should be enforced for generated configs."""
    los_keys = [
        "io/PV_main/EDD_TRGB/which_host_los",
        "io/PV_main/EDD_TRGB_grouped/which_host_los",
        "io/PV_main/EDD_2MTF/which_host_los",
        "io/SH0ES/which_host_los",
        "io/which_host_los",
    ]
    for key in los_keys:
        los = get_nested(config, key, None)
        if not (isinstance(los, str) and "manticore" in los.lower()):
            continue

        if get_nested(config, "model/run_ppc", False):
            log("forcing run_ppc=False for Manticore field.")
            config = overwrite_config(config, "model/run_ppc", False)

        beta_prior = get_nested(config, "model/priors/beta", None)
        if "model/priors/beta" not in override_set:
            if not _is_delta_prior(beta_prior):
                log("defaulting beta=delta(1) for Manticore.")
                config = overwrite_subtree(
                    config, "model/priors/beta",
                    {"dist": "delta", "value": 1.0})
        break

    return config


def validate_generated_config(config):
    """Validate generated config values that commonly fail late at runtime."""
    which_run = get_nested(config, "model/which_run", None)
    valid_runs = (None, "CH0", "CCHP", "CCHP_CSP", "EDD_TRGB",
                  "EDD_TRGB_grouped")
    if which_run not in valid_runs:
        raise ValueError(
            f"Invalid which_run='{which_run}'. Must be one of {valid_runs}.")


def finalize_output_path(config, tag):
    """Set io/fname_output and return the generated config filename stem."""
    dynamic_tag = generate_dynamic_tag(config, base_tag=tag)
    kind = get_nested(config, "pv_model/kind", None)
    stem = f"{kind}_{dynamic_tag}" if kind else dynamic_tag

    # io/fname_output is relative; load_config resolves it against root_results
    # from local_config.toml at job runtime.
    fname_out = join(config["io"]["root_output"], f"{stem}.hdf5")
    config = overwrite_config(config, "io/fname_output", fname_out)
    return config, stem, fname_out


def drop_machine_keys(config):
    """Remove machine-local keys before writing a portable generated config."""
    return {k: v for k, v in config.items() if k not in _MACHINE_KEYS}


def prepare_generated_tasks(spec, base_config, override_combinations):
    """Prepare generated config objects and task-list rows without writing."""
    generated = []
    seen_stems = set()
    seen_outputs = set()
    seen_output_filenames = set()

    for idx, override_set in enumerate(override_combinations):
        local_config = apply_overrides(base_config, override_set)
        local_config = apply_los_runtime_rules(local_config, override_set)
        validate_generated_config(local_config)
        local_config, stem, fname_out = finalize_output_path(
            local_config, spec.tag)

        if stem in seen_stems:
            raise ValueError(f"Duplicate generated TOML stem: {stem}")
        if fname_out in seen_outputs:
            raise ValueError(f"Duplicate io/fname_output: {fname_out}")
        fname_out_name = Path(fname_out).name
        if fname_out_name in seen_output_filenames:
            raise ValueError(
                f"Duplicate io/fname_output filename: {fname_out_name}")
        seen_stems.add(stem)
        seen_outputs.add(fname_out)
        seen_output_filenames.add(fname_out_name)

        generated.append((idx, stem, local_config))

    return generated


def write_generated_tasks(tasks_index, spec, generated, clean=False):
    """Write generated TOML configs and the matching tasks_<index>.txt file."""
    load_toml_writer()
    gen_dir = RUN_DIR / "generated_configs" / tasks_index
    makedirs(gen_dir, exist_ok=True)

    task_file = RUN_DIR / f"tasks_{tasks_index}.txt"
    task_file_tmp = task_file.with_name(f".{task_file.name}.tmp")
    expected_configs = set()

    for _, stem, local_config in generated:
        toml_out = gen_dir / f"{stem}.toml"
        toml_tmp = toml_out.with_name(f".{toml_out.name}.tmp")
        expected_configs.add(toml_out)
        log(f"writing the configuration file to `{toml_out}`")

        with open(toml_tmp, "wb") as f:
            tomli_w.dump(drop_machine_keys(local_config), f)
        toml_tmp.replace(toml_out)

    with open(task_file_tmp, "w") as task_fh:
        for idx, stem, _ in generated:
            toml_out = gen_dir / f"{stem}.toml"
            rel_path = toml_out.relative_to(CANDEL_ROOT)
            line = f"{idx} {rel_path}\n"
            task_fh.write(line)

    if clean:
        for old_config in gen_dir.glob("*.toml"):
            if old_config not in expected_configs:
                old_config.unlink()

    task_file_tmp.replace(task_file)
    log(f"wrote task list to `{task_file}`")


def list_specs():
    """Print available task specs for agentic discovery."""
    names = sorted(TASK_SPECS)
    if not names:
        print("No task specs registered.")
        return

    name_width = max(len("task_index"), *(len(name) for name in names))
    task_width = max(
        len("tasks"),
        *(len(str(TASK_SPECS[name].get("expected_tasks", "?")))
          for name in names),
    )
    config_width = max(
        len("config"),
        *(len(TASK_SPECS[name].get("config_path", "")) for name in names),
    )

    print(f"Registered task specs ({len(names)})")
    print()
    print(
        f"{'task_index':<{name_width}}  "
        f"{'tasks':>{task_width}}  "
        f"{'config':<{config_width}}  "
        "description"
    )
    print(
        f"{'-' * name_width}  "
        f"{'-' * task_width}  "
        f"{'-' * config_width}  "
        f"{'-' * 11}"
    )
    for name in names:
        spec = TASK_SPECS[name]
        description = spec.get("description", "")
        expected_tasks = spec.get("expected_tasks", "?")
        config_path = spec.get("config_path", "")
        print(
            f"{name:<{name_width}}  "
            f"{expected_tasks:>{task_width}}  "
            f"{config_path:<{config_width}}  "
            f"{description}"
        )

    print()
    print("Inspect:  python generate_tasks.py show <task_index>")
    print("Dry-run:  python generate_tasks.py build <task_index> --dry-run")


def show_spec(tasks_index):
    """Print one task spec as TOML-like data."""
    load_toml_writer()
    spec = get_sweep_spec(tasks_index)
    raw = deepcopy(TASK_SPECS[tasks_index])
    raw["config_path"] = str(spec.config_path)
    print(tomli_w.dumps({tasks_index: raw}))


def parse_args():
    parser = ArgumentParser(
        description=(
            "Generate CANDEL batch task lists and TOML configs from registered "
            "task specs."
        ),
        epilog=(
            "Examples:\n"
            "  python generate_tasks.py\n"
            "  python generate_tasks.py list\n"
            "  python generate_tasks.py show S8_FP_student_t\n"
            "  python generate_tasks.py build S8_FP_student_t --dry-run\n"
            "  python generate_tasks.py build S8_FP_student_t --clean\n"
            "  python generate_tasks.py S8_FP_student_t\n\n"
            "Task specs live in scripts/runs/task_specs.py. Use --dry-run "
            "before writing configs for a new or edited spec."
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command_or_task", nargs="?",
        help=("Task index to build, or one of: list, show, build. "
              "With no arguments, lists registered specs."))
    parser.add_argument(
        "task_index", nargs="?",
        help="Task index for the show/build commands.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Prepare tasks and print task rows without writing files.")
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove existing generated TOML files for this task index first.")
    return parser.parse_args()


def resolve_command(args):
    command = args.command_or_task
    if command is None:
        return "list", None
    if command == "list":
        return "list", None
    if command == "show":
        return "show", args.task_index
    if command == "build":
        return "build", args.task_index
    return "build", command


def main():
    global VERBOSE
    args = parse_args()
    command, tasks_index = resolve_command(args)

    if command == "list":
        list_specs()
        return

    if not tasks_index:
        raise ValueError(f"`{command}` requires a task index.")

    if command == "show":
        show_spec(tasks_index)
        return

    local_cfg = load_local_config()
    spec = get_sweep_spec(tasks_index)
    override_combinations = build_override_combinations(spec, local_cfg)
    load_candel_helpers()
    VERBOSE = not args.dry_run
    base_config = load_config(
        spec.config_path, replace_none=False, replace_los_prior=False,
        fill_paths=False)
    generated = prepare_generated_tasks(spec, base_config, override_combinations)

    if args.dry_run:
        for idx, stem, _ in generated:
            rel_path = (
                RUN_DIR / "generated_configs" / tasks_index / f"{stem}.toml"
            ).relative_to(CANDEL_ROOT)
            print(f"{idx} {rel_path}")
        print(f"# dry_run: {len(generated)} tasks")
        return

    write_generated_tasks(
        tasks_index, spec, generated, clean=args.clean)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))
