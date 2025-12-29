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
Generate a single joint-run configuration for Carrick2015 reconstruction.

This creates a joint run using LTYT likelihood for Clusters_hasY and LT
likelihood for Clusters_LTtail, and writes a tasks file with the TOML path.
"""
from argparse import ArgumentParser
from copy import deepcopy
from os import makedirs
from os.path import exists, join, splitext

import numpy as np
import tomli_w

from candel import fprint, load_config
from candel.pvdata.data import load_clusters


def overwrite_config(config, key, value):
    """Return a new config dict with a nested key overwritten."""
    new_config = deepcopy(config)
    keys = key.split("/")
    d = new_config
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value
    return new_config


def overwrite_subtree(config, key_path, subtree):
    """Overwrite a nested subtree (dict) at a slash-separated key path."""
    new_config = deepcopy(config)
    keys = key_path.split("/")
    d = new_config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = subtree
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


def replace_none_with_string(obj, replacement="none"):
    """Replace None values with a string for TOML serialization."""
    if isinstance(obj, dict):
        return {k: replace_none_with_string(v, replacement) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_none_with_string(v, replacement) for v in obj]
    if obj is None:
        return replacement
    return obj


def build_cluster_section(base, **updates):
    section = deepcopy(base)
    section.setdefault("remove_noY", False)
    section.setdefault("only_missing_Y", False)
    section.setdefault("finite_logY", False)
    section.setdefault("convert_to_CMB_frame", True)
    section.update(updates)
    return section


def main():
    parser = ArgumentParser(
        description="Generate joint Carrick2015 run config and tasks file."
    )
    parser.add_argument(
        "--config",
        default="scripts/cluster_runs/config_clusters.toml",
        help="Base config TOML path.",
    )
    parser.add_argument(
        "--output-root",
        default="results/joint",
        help="Output directory (relative to root_main).",
    )
    parser.add_argument(
        "--task-file",
        default="tasks_joint.txt",
        help="Tasks file to write (relative to CWD).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    shared_params = [
        "A_LT",
        "B_LT",
        "A_YT",
        "B_YT",
        "Vext",
        "sigma_v",
        "sigma_LT",
        "sigma_YT",
        "rho12",
        "beta",
        "b1",
        "R_dist_emp",
        "p_dist_emp",
        "n_dist_emp",
    ]

    config = overwrite_config(
        config, "inference/model", ["ClustersModel", "ClustersModel"]
    )
    config = overwrite_config(
        config, "inference/shared_params", ",".join(shared_params)
    )
    config = overwrite_config(
        config, "pv_model/which_Vext", "constant"
    )
    config = overwrite_config(
        config,
        "io/catalogue_name",
        ["Clusters_hasY", "Clusters_LTtail"],
    )
    config = overwrite_config(config, "io/root_output", args.output_root)

    base_clusters = get_nested(config, "io/Clusters", {})
    if not base_clusters:
        raise ValueError("Base config must define [io.Clusters].")

    base_clusters_full = deepcopy(base_clusters)
    base_clusters_full["finite_logY"] = False
    base_clusters_full["remove_noY"] = False
    base_clusters_full["only_missing_Y"] = False
    root = base_clusters_full.pop("root", None)
    if root is None:
        raise ValueError("Base Clusters config must include `root`.")
    full_data = load_clusters(root, subtract_logT_mean=False, **base_clusters_full)
    logT_mean = float(np.mean(full_data["logT"]))
    fprint(f"using shared logT_mean={logT_mean:.6g} for joint Clusters data.")

    config = overwrite_subtree(
        config,
        "io/Clusters_hasY",
        build_cluster_section(
            base_clusters,
            which_relation="LTYT",
            finite_logY=True,
            remove_noY=True,
            only_missing_Y=False,
            logT_mean=logT_mean,
        ),
    )
    config = overwrite_subtree(
        config,
        "io/Clusters_LTtail",
        build_cluster_section(
            base_clusters,
            which_relation="LT",
            finite_logY=False,
            remove_noY=False,
            only_missing_Y=True,
            logT_mean=logT_mean,
        ),
    )

    root_main = config.get("root_main", ".")
    fdir_out = join(root_main, args.output_root)
    if not exists(fdir_out):
        fprint(f"creating output directory `{fdir_out}`")
        makedirs(fdir_out, exist_ok=True)

    runs = [
        ("Carrick2015", "precomputed_los_Carrick2015"),
        ("Vext", "Vext"),
    ]

    task_lines = []
    for idx, (label, kind) in enumerate(runs):
        run_config = overwrite_config(config, "pv_model/kind", kind)
        fname_out = join(args.output_root, f"{label}_Joint_LTYT_LT.hdf5")
        run_config = overwrite_config(run_config, "io/fname_output", fname_out)

        toml_out = join(root_main, splitext(fname_out)[0] + ".toml")
        fprint(f"writing the configuration file to `{toml_out}`")
        run_config = replace_none_with_string(run_config, "none")
        with open(toml_out, "wb") as f:
            tomli_w.dump(run_config, f)
        task_lines.append(f"{idx} {toml_out}\n")

    with open(args.task_file, "w") as f:
        f.writelines(task_lines)
    fprint(f"wrote task list to `{args.task_file}`")


if __name__ == "__main__":
    main()
