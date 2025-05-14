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

from copy import deepcopy
from itertools import product
from os.path import join, splitext

import tomli_w
from candel import fprint, load_config


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
    if get_nested(config, "inference/model", None) == "Clusters_DistMarg":
        parts.append(get_nested(config, "io/Clusters/which_relation", None))

    # Fixed beta value from delta prior
    if get_nested(config, "pv_model/kind/", "").startswith("precomputed_los"):
        beta_prior = get_nested(config, "model/priors/beta", {})
        if isinstance(beta_prior, dict) and beta_prior.get("dist") == "delta":
            val = beta_prior.get("value")
            if val is not None:
                parts.append(f"beta{val}")

    # aTFRdipole if it's not a delta distribution
    aTFRdip_prior = get_nested(config, "model/priors/TFR_zeropoint_dipole", {})
    if isinstance(aTFRdip_prior, dict) and aTFRdip_prior.get("dist") != "delta":  # noqa
        parts.append("aTFRdipole")

    # If Vext is a delta distribution (not sampled)
    Vext_prior = get_nested(config, "model/priors/Vext", {})
    if isinstance(Vext_prior, dict) and Vext_prior.get("dist") == "delta":
        parts.append("noVext")

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
    config_path = "./config.toml"
    config = load_config(config_path, replace_none=False)

    tag = "default"
    tasks_index = 0

    # Multiple override options → this creates a job per combination
    manual_overrides = {
        "pv_model/kind": "constant",
        "io/catalogue_name": "Clusters",
        "io/root_output": "results",
        "pv_model/use_MNR": False,
        "io/Clusters/which_relation": ["LT", "LTY"],
        # "model/priors/beta": [
        #     {"dist": "normal", "loc": 0.43, "scale": 0.1},
        #     {"dist": "delta", "value": 1.0},
        # ],
        # "model/priors/TFR_zeropoint_dipole": [
        #     {"dist": "delta", "value": [0.0, 0.0, 0.0]},
        #     {"dist": "vector_uniform", "low": 0.0, "high": 1.0},
        # ],
        # "model/priors/Vext": [
        #     {"dist": "delta", "value": [0.0, 0.0, 0.0]},
        #     {"dist": "vector_uniform_fixed", "low": 0.0, "high": 2500.0},
        # ],
    }

    task_file = f"tasks_{tasks_index}.txt"
    log_dir = f"logs_{tasks_index}"

    override_combinations = expand_override_grid(manual_overrides)

    with open(task_file, "w") as task_fh:
        for idx, override_set in enumerate(override_combinations):
            local_config = deepcopy(config)

            for key, value in override_set.items():
                # Special handling for kind: transform before writing
                if key == "pv_model/kind" and value != "constant":
                    value = f"precomputed_los_{value}"
                    fprint(f"transformed kind override to: {value}")

                if isinstance(value, dict):
                    local_config = overwrite_subtree(local_config, key, value)
                else:
                    local_config = overwrite_config(local_config, key, value)

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
