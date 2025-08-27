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
1. Edit the `manual_overrides` dictionary near the bottom of the script to
   specify your sweep.
2. Run the script:
       $ python generate_tasks.py 0
3. Use the generated `tasks_0.txt` with a SLURM script or manual loop to run
   the tasks.

Typical output:
- One `.toml` file per combination of overrides
- A summary task list with config paths for tracking and reproducibility

This script is meant to streamline robust, reproducible inference workflows in
CANDEL.
"""
from argparse import ArgumentParser
from copy import deepcopy
from itertools import product
from os import makedirs
from os.path import exists, join, splitext

import tomli_w

from candel import fprint, load_config, replace_prior_with_delta, get_nested


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


def generate_dynamic_tag(config, base_tag="default"):
    """Generate a descriptive tag string based on selected config values."""
    parts = []
    is_CH0 = get_nested(config, "model/is_CH0", False)

    if is_CH0:
        model_name = "CH0"
        catalogue = "CH0"
        parts.append("CH0")
    else:
        model_name = get_nested(config, "inference/model", None)
        catalogue = get_nested(config, "io/catalogue_name", None)
        if catalogue:
            if isinstance(catalogue, list):
                parts.append(",".join(catalogue))
            else:
                parts.append(str(catalogue))

        use_mnr = get_nested(config, "model/use_MNR", False)
        parts.append("MNR" if use_mnr else "noMNR")

    # Clusters scaling relation choice
    if get_nested(config, "inference/model", None) == "ClustersModel":
        parts.append(get_nested(config, "io/Clusters/which_relation", None))

    # Fixed beta value from delta prior
    if get_nested(config, "pv_model/kind/", "").startswith("precomputed_los"):
        beta_prior = get_nested(config, "model/priors/beta", {})
        if isinstance(beta_prior, dict) and beta_prior.get("dist") == "delta":
            val = beta_prior.get("value")
            if val is not None:
                parts.append(f"beta{val}")

    if "TFR" in model_name and use_mnr and not get_nested(config, "model/marginalize_eta", True):  # noqa
        parts.append("eta_sampled")

    # Zeropoint dipole if it's not a delta distribution
    zeropoint_dip_prior = get_nested(config, "model/priors/zeropoint_dipole", None)  # noqa
    if isinstance(zeropoint_dip_prior, dict) and zeropoint_dip_prior.get("dist") != "delta":  # noqa
        dist_name = zeropoint_dip_prior.get("dist")
        if dist_name == "vector_components_uniform":
            parts.append("zeropoint_dipole_UnifComponents")
        else:
            parts.append("zeropoint_dipole")

    # If Vext is a delta distribution (not sampled)
    Vext_prior = get_nested(config, "model/priors/Vext", {})
    if isinstance(Vext_prior, dict) and Vext_prior.get("dist") == "delta":
        parts.append("noVext")

    # Flag if sampling the dust prior
    dust_model = get_nested(config, f"io/{catalogue}/dust_model", None)
    if dust_model is not None and dust_model.lower() != "none":
        parts.append(f"dust-{dust_model}")

    if is_CH0:
        # Which selection
        which_sel = get_nested(config, "model/which_selection", None)
        if which_sel is not None and which_sel != "none":
            parts.append(f"sel-{which_sel}")

        if get_nested(config, "model/use_uniform_mu_host_priors", False):
            parts.append("uniform_mu_host")

        r_prior = get_nested(config, "model/which_distance_prior", "volume")
        if r_prior != "volume":
            parts.append(r_prior)

        if not get_nested(config, "model/use_Cepheid_host_redshift", True):
            parts.append("no_Cepheid_redshift")

        use_reconstruction = get_nested(config, "model/use_reconstruction", False)  # noqa
        if use_reconstruction:
            parts.append(get_nested(config, "io/SH0ES/which_host_los", None))

        if get_nested(config, "model/use_fiducial_Cepheid_host_PV_covariance", False):  # noqa
            parts.append("PV_covmat")

        if get_nested(config, "model/use_PV_covmat_scaling", False):
            parts.append("PV_covmat_scaling")

        if get_nested(config, "model/weight_selection_by_covmat_Neff", False):
            parts.append("weight_by_Neff")

    if base_tag != "default":
        parts.append(base_tag)

    print(parts)

    return "_".join(p for p in parts if p is not None)


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
        "tasks_index", type=int, nargs="?", default=0,
        help="Index of the task to run (default: 0)")
    args = parser.parse_args()

    config_path = "./config_shoes.toml"
    config = load_config(
        config_path, replace_none=False, replace_los_prior=False)

    tag = "default"
    tasks_index = args.tasks_index

    # Multiple override options → this creates a job per combination
    # # --- TFR/SN/FP/Cluster flow model over-rides ---
    # manual_overrides = {
    #     "inference/num_samples": 1000,
    #     "inference/num_chains": 1,
    #     # "pv_model/kind": "precomputed_los_Carrick2015",
    #     "pv_model/kind": "Vext",
    #     # "io/catalogue_name": [f"CF4_mock_{n}" for n in range(70)],
    #     "io/catalogue_name": "Clusters",
    #     # "inference/shared_params": "none",
    #     "inference/model": "ClustersModel",
    #     "io/root_output": "results_test/",
    #     "io/Clusters/which_relation": "LTY",
    #     "model/use_MNR": False,
    #     # "model/marginalize_eta": True,
    #     # "io/CF4_i/exclude_W1": True,
    #     # "io/CF4_W1/best_mag_quality": False,
    #     # "io/CF4_W1/zcmb_min": 0.01,
    #     # "io/CF4_W1/dust_model": ["none", "default", "SFD", "CSFD", "Planck2016"],  # noqa
    #     # "io/Clusters/which_relation": ["LT", "LTY"],
    #     # "model/priors/beta": [
    #     #     {"dist": "normal", "loc": 0.43, "scale": 0.1},
    #     #     {"dist": "delta", "value": 1.0},
    #     # ],
    #     # "model/priors/zeropoint_dipole": [
    #     #     {"dist": "delta", "value": [0.0, 0.0, 0.0]},
    #     #     # {"dist": "vector_uniform_fixed", "low": 0.0, "high": 0.3},
    #     #     # {"dist": "vector_components_uniform", "low": -0.3, "high": 0.3},  # noqa
    #     # ],
    # }

    # --- CH0 overrides ---
    manual_overrides = {
        "io/root_output": "results/CH0",
        "model/which_selection": ["none", "redshift", "SN_magnitude", "SN_magnitude_redshift", "empirical"],  # noqa
        "model/use_reconstruction": False,
        "model/use_fiducial_Cepheid_host_PV_covariance": False,
        "model/use_PV_covmat_scaling": False,
        "model/weight_selection_by_covmat_Neff": False,  # Only for redshift sel!  # noqa
        # "io/SH0ES/which_host_los": "Carrick2015",
        # "model/priors/Vext": [
        #     {"dist": "vector_uniform_fixed", "low": 0.0, "high": 2500},
        #     {"dist": "delta", "value": [0., 0., 0.]},
        # ],
        # "model/priors/beta": [
        #     {"dist": "normal", "loc": 0.43, "scale": 0.02},
        #     {"dist": "normal", "loc": 1.0, "scale": 0.5},
        #     {"dist": "delta", "value": 1.0},
        # ],
    }

    # manticore_2MPP_MULTIBIN_N256_DES_V2

    task_file = f"tasks_{tasks_index}.txt"
    log_dir = f"logs_{tasks_index}"

    override_combinations = expand_override_grid(manual_overrides)

    with open(task_file, "w") as task_fh:
        for idx, override_set in enumerate(override_combinations):
            local_config = deepcopy(config)

            for key, value in override_set.items():
                # Special handling for kind: transform before writing
                if key == "pv_model/kind":
                    if "Vext" in value:
                        config = replace_prior_with_delta(config, "alpha", 1.)
                        config = replace_prior_with_delta(config, "beta", 0.)
                        config = replace_prior_with_delta(config, "b1", 0.)
                    else:
                        value = f"precomputed_los_{value}"
                        fprint(f"transformed kind override to: {value}")

                if isinstance(value, dict):
                    local_config = overwrite_subtree(local_config, key, value)
                else:
                    local_config = overwrite_config(local_config, key, value)

            # Check that the output directory exists
            fdir_out = join(
                local_config["root_main"], local_config["io"]["root_output"])
            if not exists(fdir_out):
                fprint(f"creating output directory `{fdir_out}`")
                makedirs(fdir_out, exist_ok=True)

            dynamic_tag = generate_dynamic_tag(local_config, base_tag=tag)

            kind = get_nested(local_config, "pv_model/kind", None)
            if kind is None:
                fname_out = join(
                    local_config["io"]["root_output"], f"{dynamic_tag}.hdf5")
            else:
                fname_out = join(
                    local_config["io"]["root_output"],
                    f"{kind}_{dynamic_tag}.hdf5")

            local_config = overwrite_config(
                local_config, "io/fname_output", fname_out)

            toml_out = join(
                local_config["root_main"],
                splitext(fname_out)[0] + ".toml"
            )
            fprint(f"writing the configuration file to `{toml_out}`")
            with open(toml_out, "wb") as f:
                tomli_w.dump(local_config, f)

            task_fh.write(f"{idx} {toml_out}\n")

    fprint(f"wrote task list to `{task_file}`")
