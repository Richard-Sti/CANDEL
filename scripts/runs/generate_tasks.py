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
1. Edit the `common` / `individual_datasets` structures in the
   ``if __name__ == "__main__":`` block at the bottom of this script to
   specify your sweep.
2. Run the script:
       $ python generate_tasks.py 0
3. Use the generated `tasks_0.txt` with a SLURM script or manual loop to run
   the tasks.

Typical output:
- One `.toml` file per combination of overrides
- A summary task list with config paths for tracking and reproducibility
- Descriptive filenames/tags that capture salient flags (e.g. catalogue, MNR,
  selection mode, Nmag split for mixed selections, Vext settings)

This script is meant to streamline robust, reproducible inference workflows in
CANDEL.
"""
import hashlib
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from os import makedirs
from os.path import join
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import tomli_w

from candel import fprint, get_nested, load_config, replace_prior_with_delta  # noqa


# Keys that must come from local_config.toml at job runtime, not baked into
# generated configs. Baking them in defeats the portability mechanism in
# load_config, which injects local_config.toml only for keys not already set.
_MACHINE_KEYS = {
    "root_main", "root_data", "root_results",
    "python_exec", "machine", "modules", "modules_gpu",
    "use_frozen", "gpu_ld_library_path",
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
    project_root = Path(__file__).resolve().parent.parent.parent
    local_config_path = project_root / "local_config.toml"
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

    fprint(f"overwriting config['{'/'.join(keys)}'] = {value}")
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
    fprint(f"overwriting subtree config['{'/'.join(keys)}'] = {subtree}")
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

        if get_nested(config, "model/use_uniform_mu_host_priors", False):
            parts.append("uniform_mu_host")

        r_prior = get_nested(config, "model/which_distance_prior", "volume")
        if r_prior != "volume":
            parts.append(r_prior)

        if not get_nested(config, "model/use_Cepheid_host_redshift", True):
            parts.append("no_Cepheid_redshift")

        if get_nested(config, "model/use_reconstruction", False):
            parts.append(get_nested(config, "io/SH0ES/which_host_los", None))
            if get_nested(config, "model/use_density_dependent_sigma_v", False):  # noqa
                parts.append("sigv_rho")

        if get_nested(config, "model/use_fiducial_Cepheid_host_PV_covariance",
                      False):
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


def write_provenance_footer(fh, tasks_index, n_tasks, body_sha256,
                            local_cfg, files):
    """Append a `#`-prefixed footer embedding the verbatim contents of
    ``files`` plus the filtered ``local_config.toml`` dict that fed the
    override grid, so the task list is fully self-describing. Submission
    scripts skip any line beginning with `#`, so the footer is inert at
    runtime.
    """
    bar = "# " + "=" * 60
    fh.write("#\n")
    fh.write(bar + "\n")
    fh.write("# == GENERATOR PROVENANCE " + "=" * 36 + "\n")
    fh.write(bar + "\n")
    fh.write(f"# generated_utc: {datetime.now(timezone.utc).isoformat()}\n")
    fh.write(f"# tasks_index:   {tasks_index}\n")
    fh.write(f"# n_tasks:       {n_tasks}\n")
    fh.write(f"# body_sha256:   {body_sha256}\n")
    fh.write("#\n")
    fh.write("# Verbatim source of the generator and base config template at\n")
    fh.write("# the time this file was produced. Strip the leading '# ' from\n")
    fh.write("# each line to recover the originals.\n")
    fh.write(bar + "\n")

    fh.write("#\n")
    fh.write("# --- BEGIN local_config (filtered: machine keys excluded) ---\n")
    if local_cfg:
        for k, v in local_cfg.items():
            fh.write(f"# {k} = {v!r}\n")
    else:
        fh.write("# <empty>\n")
    fh.write("# --- END local_config ---\n")

    for label, path in files:
        fh.write("#\n")
        fh.write(f"# --- BEGIN FILE: {label} ---\n")
        try:
            with open(path, "r") as src:
                for line in src:
                    fh.write("# " + line.rstrip("\n") + "\n")
        except OSError as e:
            fh.write(f"# <ERROR reading {path}: {e}>\n")
        fh.write(f"# --- END FILE: {label} ---\n")


def expand_override_grid(overrides):
    """Expand override grid into a list of flat key-value combinations."""
    model_key = "inference/model"
    cat_key = "io/catalogue_name"

    is_joint_model = (
        model_key in overrides
        and cat_key in overrides
        and isinstance(overrides[model_key], list)
        and isinstance(overrides[cat_key], list)
        and len(overrides[model_key]) == len(overrides[cat_key])
    )

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "tasks_index", type=str, nargs="?", default="0",
        help="Arbitrary tag/index for this task list.")
    args = parser.parse_args()

    tag = "default"
    tasks_index = args.tasks_index

    _local_cfg = load_local_config()

    # config_path is set inside each branch and used after.
    config_path = None
    if tasks_index in ("ch0", "ch0_mag"):
        # CH0: SH0ES Cepheid H0, Carrick2015 field.
        # config_shoes.toml already contains all required settings:
        #   which_run = "CH0", use_reconstruction = true,
        #   which_host_los = "Carrick2015",
        #   selection_integral_grid_radius = 100.0 Mpc/h
        #   ch0          -> which_selection = "redshift"  (from config)
        #   ch0_mag      -> which_selection = "SN_magnitude"
        config_path = "./configs/config_shoes.toml"
        overrides = {
            **{k: v for k, v in _local_cfg.items()},
            "io/root_output": "results/CH0",
        }
        if tasks_index == "ch0_mag":
            overrides["model/which_selection"] = "SN_magnitude"
        all_override_combinations = expand_override_grid(overrides)

    else:
        config_path = "./configs/config.toml"
        # --- S8 from PVs: linear bias, b1 fixed on a grid ---
        # 21 values: 0.5, 0.6, ..., 2.5
        b1_values = [round(0.5 + 0.1 * i, 1) for i in range(21)]
        b1_priors = [{"dist": "delta", "value": v} for v in b1_values]

        common = {
            **{k: v for k, v in _local_cfg.items()},
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "linear",
            "pv_model/density_3d_downsample": 1,
            "model/priors/beta": {"dist": "uniform", "low": 0.0, "high": 2.0},
            "model/priors/b1": b1_priors,
            "inference/num_chains": 1,
            "inference/num_warmup": 2000,
            "inference/num_samples": 10000,
            "io/root_output": "results/S8",
        }

        datasets = [
            {"inference/model": "TFRModel", "io/catalogue_name": "CF4_W1"},
            {"inference/model": "FPModel",  "io/catalogue_name": "SDSS_FP"},
        ]

        all_override_combinations = []
        for dataset in datasets:
            all_override_combinations.extend(
                expand_override_grid({**common, **dataset}))

    config = load_config(
        config_path, replace_none=False, replace_los_prior=False,
        fill_paths=False)

    candel_root = Path(__file__).resolve().parent.parent.parent
    gen_dir = candel_root / "scripts" / "runs" / "generated_configs" / tasks_index
    makedirs(gen_dir, exist_ok=True)

    task_file = f"tasks_{tasks_index}.txt"
    body_hash = hashlib.sha256()
    n_written = 0

    with open(task_file, "w") as task_fh:
        for idx, override_set in enumerate(all_override_combinations):
            local_config = deepcopy(config)

            for key, value in override_set.items():
                # Special handling for kind: transform before writing
                if key == "pv_model/kind":
                    if not value.startswith("precomputed_los"):
                        # No reconstruction: force unity galaxy bias
                        local_config = overwrite_config(
                            local_config, "pv_model/galaxy_bias", "unity")
                        local_config = overwrite_config(
                            local_config, "model/use_reconstruction", False)
                        local_config = replace_prior_with_delta(
                            local_config, "alpha", 1.)
                        local_config = replace_prior_with_delta(
                            local_config, "beta", 0.)
                        local_config = replace_prior_with_delta(
                            local_config, "b1", 0.)
                        local_config = replace_prior_with_delta(
                            local_config, "b2", 0.)
                        local_config = replace_prior_with_delta(
                            local_config, "b3", 0.)
                        local_config = replace_prior_with_delta(
                            local_config, "delta_b1", 0.)

                if isinstance(value, dict):
                    local_config = overwrite_subtree(local_config, key, value)
                else:
                    local_config = overwrite_config(local_config, key, value)

            # Force PPC off for Manticore (too expensive with many fields)
            _los_keys = [
                "io/PV_main/EDD_TRGB/which_host_los",
                "io/PV_main/EDD_TRGB_grouped/which_host_los",
                "io/PV_main/EDD_2MTF/which_host_los",
                "io/SH0ES/which_host_los",
                "io/which_host_los",
            ]
            for _k in _los_keys:
                _los = get_nested(local_config, _k, None)
                if isinstance(_los, str) and "manticore" in _los.lower():
                    if get_nested(local_config, "model/run_ppc", False):
                        fprint("forcing run_ppc=False for Manticore field.")
                        local_config = overwrite_config(
                            local_config, "model/run_ppc", False)
                    # Default beta to delta(1) for Manticore unless already
                    # explicitly set to a non-default prior in manual_overrides.
                    beta_prior = get_nested(
                        local_config, "model/priors/beta", None)
                    if "model/priors/beta" not in override_set:
                        if not _is_delta_prior(beta_prior):
                            fprint("defaulting beta=delta(1) for Manticore.")
                            local_config = overwrite_subtree(
                                local_config, "model/priors/beta",
                                {"dist": "delta", "value": 1.0})
                    break

            # Validate which_run
            which_run = get_nested(local_config, "model/which_run", None)
            valid_runs = (None, "CH0", "CCHP", "CCHP_CSP", "EDD_TRGB",
                          "EDD_TRGB_grouped")
            if which_run not in valid_runs:
                raise ValueError(
                    f"Invalid which_run='{which_run}'. "
                    f"Must be one of {valid_runs}.")

            dynamic_tag = generate_dynamic_tag(local_config, base_tag=tag)

            kind = get_nested(local_config, "pv_model/kind", None)
            stem = f"{kind}_{dynamic_tag}" if kind else dynamic_tag
            # io/fname_output is relative; load_config resolves it against
            # root_results from local_config.toml at job runtime.
            fname_out = join(local_config["io"]["root_output"],
                             f"{stem}.hdf5")
            local_config = overwrite_config(
                local_config, "io/fname_output", fname_out)

            toml_out = gen_dir / f"{stem}.toml"
            fprint(f"writing the configuration file to `{toml_out}`")
            # Drop machine-specific keys that load_config injected from
            # local_config.toml — they must be resolved at job runtime on
            # the executing machine, not baked in here.
            to_dump = {
                k: v for k, v in local_config.items()
                if k not in _MACHINE_KEYS
            }
            with open(toml_out, "wb") as f:
                tomli_w.dump(to_dump, f)

            rel_path = toml_out.relative_to(candel_root)
            line = f"{idx} {rel_path}\n"
            task_fh.write(line)
            body_hash.update(line.encode())
            n_written += 1

        write_provenance_footer(
            task_fh,
            tasks_index=tasks_index,
            n_tasks=n_written,
            body_sha256=body_hash.hexdigest(),
            local_cfg=_local_cfg,
            files=[
                ("scripts/runs/generate_tasks.py", Path(__file__).resolve()),
                ("scripts/runs/configs/config.toml",
                 Path(config_path).resolve()),
            ],
        )

    fprint(f"wrote task list to `{task_file}`")
