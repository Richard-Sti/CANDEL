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
Lightweight utility functions that don't require JAX or heavy dependencies.

Use this module for scripts that only need basic config loading and printing,
to avoid the startup overhead of JAX initialization.
"""

try:
    # Python 3.11+
    import tomllib  # noqa
except ModuleNotFoundError:
    # Backport for <=3.10
    import tomli as tomllib

from datetime import datetime
from os.path import abspath, isabs, join


def fprint(*args, verbose=True, **kwargs):
    """Prints a message with a timestamp prepended."""
    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S%f")[:-6]
        print(f"{timestamp}", *args, **kwargs)


def convert_none_strings(d):
    """
    Convert all string values in a dictionary to None if they are equal to
    "none" (case insensitive). This is useful for parsing TOML files where
    "none" is used to represent None values.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            convert_none_strings(v)
        elif isinstance(v, str) and v.strip().lower() == "none":
            d[k] = None
    return d


def replace_prior_with_delta(config, param, value, verbose=True):
    """Replace the prior of `param` with a delta distribution at `value`."""
    if param not in config.get("model", {}).get("priors", {}):
        return config

    fprint(f"replacing prior of `{param}` with a delta function.",
           verbose=verbose)
    priors = config.setdefault("model", {}).setdefault("priors", {})
    priors.pop(param, None)
    priors[param] = {
        "dist": "delta",
        "value": value
        }
    return config


def convert_to_absolute_paths(config):
    """Recursively convert relative paths in config to absolute paths."""
    root = config["root_main"]
    root_data = config.get("root_data", root)

    path_keys_root = {
        "fname_output",
    }
    path_keys_data = {
        "root",
        "los_file",
        "los_file_random",
        "path_density",
        "path_velocity",
    }

    def _recurse(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _recurse(v)
            elif isinstance(v, str):
                if k in path_keys_root and not isabs(v):
                    d[k] = abspath(join(root, v))
                elif k in path_keys_data and not isabs(v):
                    d[k] = abspath(join(root_data, v))

    _recurse(config)
    return config


def load_config(config_path, replace_none=True, fill_paths=True,
                replace_los_prior=True):
    """
    Load a TOML configuration file and convert "none" strings to None.
    """
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    # Convert "none" strings to None
    if replace_none:
        config = convert_none_strings(config)

    # Assign delta priors if not using an underlying reconstruction.
    kind = config.get("pv_model", {}).get("kind", "")
    if replace_los_prior and not kind.startswith("precomputed_los"):
        config = replace_prior_with_delta(config, "alpha", 1.)
        config = replace_prior_with_delta(config, "beta", 0.)
        config = replace_prior_with_delta(config, "b1", 0.)
        config = replace_prior_with_delta(config, "delta_b1", 0.)

    # Convert relative paths to absolute paths
    if fill_paths:
        config = convert_to_absolute_paths(config)

    shared_params = config["inference"].get("shared_params", None)
    if shared_params and str(shared_params).lower() != "none":
        config["inference"]["shared_params"] = shared_params.split(",")

    return config
