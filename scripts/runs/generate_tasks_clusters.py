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

    # Catalogue name
    catalogue = get_nested(config, "io/catalogue_name", None)
    if catalogue:
        parts.append(catalogue)

    # MNR flag
    use_mnr = get_nested(config, "pv_model/use_MNR", False)
    parts.append("MNR" if use_mnr else "noMNR")

    # Clusters scaling relation choice
    scaling_relation = get_nested(config, "io/Clusters/which_relation", None)
    if scaling_relation:
        parts.append(scaling_relation)

    # Extract reconstruction name from pv_model/kind
    kind = get_nested(config, "pv_model/kind", "")
    if kind.startswith("precomputed_los"):
        beta_prior = get_nested(config, "model/priors/beta", {})
        if isinstance(beta_prior, dict) and beta_prior.get("dist") == "delta":
            val = beta_prior.get("value")
            if val is not None:
                parts.append(f"beta{val}")

    # Vext configuration - only add non-default cases
    which_vext = get_nested(config, "pv_model/which_Vext", "constant")
    if which_vext == "per_pix":
        parts.append("pixVext")
    else:
        # Check for quadrupole Vext first (implies dipole too)
        Vext_quad_prior = get_nested(config, "model/priors/Vext_quad", {})
        if isinstance(Vext_quad_prior, dict) and Vext_quad_prior.get("dist") != "delta":
            parts.append("quadVext")
        else:
            # Check regular Vext prior
            Vext_prior = get_nested(config, "model/priors/Vext", {})
            if isinstance(Vext_prior, dict):
                vext_dist = Vext_prior.get("dist", "")
                if vext_dist == "vector_uniform_fixed":
                    parts.append("dipVext")
                elif vext_dist == "quadrupole_uniform_fixed":
                    parts.append("quadVext")
                # delta case is default, don't add anything

    # Zeropoint configuration - only add non-default cases
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
    dust_model = get_nested(config, f"io/{catalogue}/dust_model", None)
    if dust_model is not None and dust_model.lower() != "none":
        parts.append(f"dust-{dust_model}")

    # if remove_noY is true then label tag with hasY:
    remove_noY = get_nested(config, f"io/{catalogue}/remove_noY", False)
    if remove_noY:
        parts.append("hasY")

    return "_".join(parts) if base_tag == "default" else base_tag


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

    config_path = "config_clusters.toml"
    config = load_config(
        config_path, replace_none=False, replace_los_prior=False)

    tag = "default"
    tasks_index = args.tasks_index

    task_file = f"tasks_LT_vs_LTY_{tasks_index}.txt"
    log_dir = f"logs_LT_vs_LTY_{tasks_index}"

    # Dipoles for LT and LTY with Manticore
    base = {
        "pv_model/kind": ["precomputed_los_manticore_2MPP_MULTIBIN_N256_DES_V2"],
        #"pv_model/kind": ["precomputed_los_Carrick2015"],
        "pv_model/galaxy_bias": ["powerlaw"],
        "pv_model/which_Vext": ["constant"],
        "io/catalogue_name": "Clusters",
        "io/root_output": "results/Clusters",
        "pv_model/use_MNR": False,
        "io/Clusters/which_relation": ["LTYT"],
        "io/Clusters/remove_noY": [True],
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
            {"dist": "vector_uniform_fixed", "low": 0.0, "high": 0.3},
        ]

    dipole_combinations = expand_override_grid(dipole_settings)
    
    # Per-pixel Vext
    pixel_settings = deepcopy(base)
    pixel_settings["pv_model/which_Vext"] = ["per_pix"]

    pixel_combinations = expand_override_grid(pixel_settings)

    # Dipole and quadrupole Vext
    quadVext_settings = deepcopy(base)

    quadVext_settings["pv_model/priors/Vext"] = [
        {"dist": "vector_uniform_fixed", "low": 0.0, "high": 2000.0},
    ]
    quadVext_settings["pv_model/priors/Vext_quad"] = [
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
    
    # Combine both lists
    override_combinations = dipole_combinations + pixel_combinations + \
                            quadVext_combinations  + quad_zeropoint_combinations

    print(f"Total combinations: {len(override_combinations)}")

    with open(task_file, "w") as task_fh:
        for idx, override_set in enumerate(override_combinations):
            local_config = deepcopy(config)

            for key, value in override_set.items():
                # Special handling for kind: transform before writing
                # if key == "pv_model/kind":
                #     if "Vext" in value:
                #         config = replace_prior_with_delta(config, "alpha", 1.)
                #         config = replace_prior_with_delta(config, "beta", 0.)
                #     else:
                #         value = f"precomputed_los_{value}"
                #         fprint(f"transformed kind override to: {value}")

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

            kind = get_nested(local_config, "pv_model/kind", "unknown")

            fname_out = join(
                local_config["io"]["root_output"],
                f"{kind}_{dynamic_tag}.hdf5"
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

            task_fh.write(f"{idx} {toml_out}\n")

    fprint(f"wrote task list to `{task_file}`")
