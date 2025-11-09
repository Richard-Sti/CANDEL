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
Prepare a director structure for new runs, including copying and overwriting
the default configuration file.
"""

from argparse import ArgumentParser
from copy import deepcopy
from itertools import product
from os import makedirs
from os.path import exists, join, splitext

import tomli_w

from candel import fprint, load_config, replace_prior_with_delta


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


def get_nested(config, key_path, default=None):
    """Recursively access a nested value using a slash-separated key."""
    keys = key_path.split("/")
    current = config
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return default
        current = current[k]
    return current


def generate_dynamic_tag(config, base_tag="default"):
    """Generate a descriptive tag string based on selected config values."""
    parts = []

    if base_tag != "default":
        parts.append(base_tag)

    catalogue_value = get_nested(config, "io/catalogue_name", None)
    if isinstance(catalogue_value, list):
        catalogue_names = catalogue_value
        if base_tag == "default":
            parts.append("joint")
    elif isinstance(catalogue_value, str) and catalogue_value:
        catalogue_names = [catalogue_value]
        if base_tag == "default":
            parts.append(catalogue_value)
    else:
        catalogue_names = ["Clusters"]
        if base_tag == "default":
            parts.append("Clusters")

    # MNR flag
    use_mnr = get_nested(config, "pv_model/use_MNR", False)
    parts.append("MNR" if use_mnr else "noMNR")

    # Clusters scaling relation choice
    relations = []
    for name in catalogue_names:
        rel = get_nested(config, f"io/{name}/which_relation", None)
        if rel and rel not in relations:
            relations.append(rel)
    parts.extend(relations)

    # Vext configuration - only add non-default cases
    which_vext = get_nested(config, "pv_model/which_Vext", "constant")
    if which_vext == "per_pix":
        parts.append("pixVext")
    elif which_vext == "radial":
        parts.append("radVext")
    else:
        # Check for separate Vext_quad component first
        Vext_quad_prior = get_nested(config, "model/priors/Vext_quad", {})
        if isinstance(Vext_quad_prior, dict) and Vext_quad_prior.get("dist") != "delta":
            parts.append("quadVext")  # Quadrupole implicitly includes dipole
        else:
            # Check regular Vext prior
            Vext_prior = get_nested(config, "model/priors/Vext", {})
            if isinstance(Vext_prior, dict):
                vext_dist = Vext_prior.get("dist", "")
                if vext_dist == "vector_uniform_fixed":
                    parts.append("dipVext")
                elif vext_dist == "quadrupole_uniform_fixed":
                    parts.append("quadVext")  # Main Vext is quadrupole
                # delta case is default, don't add anything

    # Zeropoint A configuration - check per_pix first
    which_A = get_nested(config, "pv_model/which_A", "constant")
    if which_A == "per_pix":
        parts.append("pixA")
    else:
        # Check for quadrupole zeropoint first (implies dipole too)
        quad_prior = get_nested(config, "model/priors/zeropoint_quad", {})
        if isinstance(quad_prior, dict) and quad_prior.get("dist") != "delta":
            parts.append("quadA")
        else:
            # Check dipole zeropoint
            dip_prior = get_nested(config, "model/priors/zeropoint_dipole", {})
            if isinstance(dip_prior, dict):
                dip_dist = dip_prior.get("dist", "")
                if dip_dist == "vector_uniform_fixed":
                    parts.append("dipA")
                # delta case is default, don't add anything

    # Flag if sampling the dust prior
    dust_flags = []
    for name in catalogue_names:
        dust_model = get_nested(config, f"io/{name}/dust_model", None)
        if isinstance(dust_model, str) and dust_model.lower() != "none":
            label = f"dust-{dust_model}"
            if label not in dust_flags:
                dust_flags.append(label)
    parts.extend(dust_flags)

    # if remove_noY is true then label tag with hasY:
    has_y = False
    for name in catalogue_names:
        if get_nested(config, f"io/{name}/remove_noY", False):
            has_y = True
            break
    if has_y:
        parts.append("hasY")

    return "_".join(parts)


def expand_override_grid(overrides):
    """
    Convert a dictionary with lists of override values into a list of flat
    key-value combinations.
    """
    keys, values = zip(*[
        (k, v if isinstance(v, list) else [v])
        for k, v in overrides.items()
    ])
    return [dict(zip(keys, combo)) for combo in product(*values)]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "tasks_index", type=int, nargs="?", default=0,
        help="Index of the task to run (default: 0)")
    args = parser.parse_args()

    config_path = "scripts/cluster_runs/config_clusters.toml"
    config = load_config(
        config_path, replace_none=False, replace_los_prior=False)

    tasks_index = args.tasks_index

    task_file = f"tasks_{tasks_index}.txt"
    log_dir = f"logs_{tasks_index}"

    base = {
        "pv_model/kind": ["Vext", "precomputed_los_Carrick2015", "precomputed_los_manticore"],
        "pv_model/which_Vext": ["constant"],
        "io/root_output": "results/final",
        "model/priors/Vext": [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]}],
        "model/priors/zeropoint_dipole": [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]}],
    }

    dipole_settings = deepcopy(base)

    # Dipoles and permutations
    dipole_settings["model/priors/Vext"] = [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]}, 
            {"dist": "vector_uniform_fixed", "low": 0.0, "high": 2000.0},
        ]
    dipole_settings["model/priors/zeropoint_dipole"] = [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]},
            {"dist": "vector_uniform_fixed", "low": 0.0, "high": 0.1},
        ]

    dipole_combinations = expand_override_grid(dipole_settings)
    
    # Per-pixel Vext
    pixelVext_settings = deepcopy(base)
    pixelVext_settings["pv_model/which_Vext"] = ["per_pix"]

    pixelVext_combinations = expand_override_grid(pixelVext_settings)

    # Per-pixel A
    pixelA_settings = deepcopy(base)
    pixelA_settings["pv_model/which_A"] = ["per_pix"]

    pixelA_combinations = expand_override_grid(pixelA_settings)

    # Dipole and quadrupole Vext
    quadVext_settings = deepcopy(base)

    quadVext_settings["model/priors/Vext"] = [
        {"dist": "vector_uniform_fixed", "low": 0.0, "high": 2000.0},
    ]
    quadVext_settings["model/priors/Vext_quad"] = [
        {"dist": "quadrupole_uniform_fixed", "low": 0.0, "high": 2000.0},
    ]

    quadVext_combinations = expand_override_grid(quadVext_settings)

    # Dipole and quadrupole zeropoint
    quad_zeropoint_settings = deepcopy(base)
    quad_zeropoint_settings["model/priors/zeropoint_dipole"] = [
        {"dist": "vector_uniform_fixed", "low": 0.0, "high": 0.3},
    ]
    quad_zeropoint_settings["model/priors/zeropoint_quad"] = [
        {"dist": "quadrupole_uniform_fixed", "low": 0.0, "high": 0.3},
    ]

    quad_zeropoint_combinations = expand_override_grid(quad_zeropoint_settings)
    
    # Radial Vext
    radialVext_settings = deepcopy(base)
    radialVext_settings["pv_model/which_Vext"] = ["radial"]

    radialVext_combinations = expand_override_grid(radialVext_settings)
    
    override_combinations = (
        dipole_combinations
        + quadVext_combinations
        + quad_zeropoint_combinations
        + pixelA_combinations
        + pixelVext_combinations
        + radialVext_combinations
    )

    base_clusters_section = deepcopy(config["io"].get("Clusters", {}))
    if not base_clusters_section:
        raise ValueError("Base config must define [io.Clusters] for template settings.")

    def build_cluster_section(**updates):
        section = deepcopy(base_clusters_section)
        section.setdefault("remove_noY", False)
        section.setdefault("only_missing_Y", False)
        section.update(updates)
        return section

    scenarios = [
        {
            "label": "LT",
            "overrides": {
                "inference/model": "ClustersModel",
                "inference/shared_params": "none",
                "io/catalogue_name": "Clusters",
                "io/Clusters": build_cluster_section(
                    which_relation="LT",
                    finite_logY=False,
                    remove_noY=False,
                    only_missing_Y=False,
                ),
            },
        },
        {
            "label": "YT",
            "overrides": {
                "inference/model": "ClustersModel",
                "inference/shared_params": "none",
                "io/catalogue_name": "Clusters",
                "io/Clusters": build_cluster_section(
                    which_relation="YT",
                    finite_logY=True,
                    remove_noY=True,
                    only_missing_Y=False,
                ),
            },
        },
        {
            "label": "LTYT",
            "overrides": {
                "inference/model": "ClustersModel",
                "inference/shared_params": "none",
                "io/catalogue_name": "Clusters",
                "io/Clusters": build_cluster_section(
                    which_relation="LTYT",
                    finite_logY=True,
                    remove_noY=True,
                    only_missing_Y=False,
                ),
            },
        },
        {
            "label": "Joint",
            "overrides": {
                "inference/model": ["ClustersModel", "ClustersModel"],
                "io/catalogue_name": ["Clusters_hasY", "Clusters_LTtail"],
                "io/Clusters_hasY": build_cluster_section(
                    which_relation="LTYT",
                    finite_logY=True,
                    remove_noY=True,
                    only_missing_Y=False,
                ),
                "io/Clusters_LTtail": build_cluster_section(
                    which_relation="LT",
                    finite_logY=False,
                    remove_noY=False,
                    only_missing_Y=True,
                ),
            },
            "shared_params_base": [
                "sigma_v",
                "zeropoint_dipole",
                "zeropoint_quad",
                "beta",
            ],
            "share_flow": True,
        },
    ]

    def flow_shared_params(which_vext):
        mapping = {
            "constant": ["Vext", "Vext_quad"],
            "radial": ["Vext_rad"],
            "radial_magnitude": ["Vext_radmag"],
        }
        return mapping.get(which_vext, [])

    print(f"Total override combinations per scenario: {len(override_combinations)}")

    task_counter = 0
    with open(task_file, "w") as task_fh:
        for scenario in scenarios:
            scenario_overrides = scenario["overrides"]
            scenario_label = scenario["label"]

            for override_set in override_combinations:
                local_config = deepcopy(config)

                for key, value in scenario_overrides.items():
                    if isinstance(value, dict):
                        local_config = overwrite_subtree(local_config, key, value)
                    else:
                        local_config = overwrite_config(local_config, key, value)

                for key, value in override_set.items():
                    if isinstance(value, dict):
                        local_config = overwrite_subtree(local_config, key, value)
                    else:
                        local_config = overwrite_config(local_config, key, value)

                shared_base = scenario.get("shared_params_base", None)
                if shared_base:
                    shared_list = list(shared_base)
                    if scenario.get("share_flow", False):
                        which_vext = get_nested(
                            local_config, "pv_model/which_Vext", "constant")
                        shared_list.extend(flow_shared_params(which_vext))
                    shared_list = list(dict.fromkeys(shared_list))
                    shared_str = ",".join(shared_list)
                    local_config = overwrite_config(
                        local_config, "inference/shared_params", shared_str)

                fdir_out = join(
                    local_config["root_main"], local_config["io"]["root_output"])
                if not exists(fdir_out):
                    fprint(f"creating output directory `{fdir_out}`")
                    makedirs(fdir_out, exist_ok=True)

                kind = get_nested(local_config, "pv_model/kind", "unknown")
                if kind.startswith("precomputed_los_"):
                    if "manticore" in kind.lower():
                        beta_prior = {"dist": "normal", "loc": 1.0, "scale": 0.02}
                        local_config = overwrite_subtree(
                            local_config, "model/priors/beta", beta_prior)
                        fprint("set beta prior to Normal(1.0, 0.02) for manticore reconstruction")

                        local_config = overwrite_config(
                            local_config, "pv_model/galaxy_bias", "powerlaw")
                        fprint("set galaxy_bias to 'powerlaw' for manticore reconstruction")

                    elif "carrick" in kind.lower():
                        beta_prior = {"dist": "normal", "loc": 0.43, "scale": 0.02}
                        local_config = overwrite_subtree(
                            local_config, "model/priors/beta", beta_prior)
                        fprint("set beta prior to Normal(0.43, 0.02) for Carrick2015 reconstruction")

                        local_config = overwrite_config(
                            local_config, "pv_model/galaxy_bias", "linear")
                        fprint("set galaxy_bias to 'linear' for Carrick2015 reconstruction")

                dynamic_tag = generate_dynamic_tag(local_config, base_tag=scenario_label)
                kind_for_filename = kind.replace("precomputed_los_", "")

                fname_out = join(
                    local_config["io"]["root_output"],
                    f"{kind_for_filename}_{dynamic_tag}.hdf5"
                )
                local_config = overwrite_config(
                    local_config, "io/fname_output", fname_out)

                toml_out = join(
                    local_config["root_main"],
                    splitext(fname_out)[0] + ".toml"
                )
                fprint(f"writing the configuration file to `{toml_out}`")
                with open(toml_out, "wb") as f:
                    tomli_w.dump(local_config, f)

                task_fh.write(f"{task_counter} {toml_out}\n")
                task_counter += 1

    fprint(f"wrote task list to `{task_file}`")
