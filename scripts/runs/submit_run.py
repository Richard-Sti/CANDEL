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

import subprocess
from copy import deepcopy
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


if __name__ == "__main__":
    config_path = "./config.toml"
    config = load_config(config_path, replace_none=False)
    to_submit = True

    kind = "Carrick2015"
    catalogues = "CF4_W1"
    tag = None

    # Process the catalogue/simulation names
    if isinstance(catalogues, str):
        catalogues = [catalogues]

    if kind not in ["constant"]:
        kind_original = kind
        kind = f"precomputed_los_{kind}"
        fprint(f"updated kind from `{kind_original}` to `{kind}`")

    if tag is None:
        tag = "default"

    for catalogue in catalogues:
        local_config = overwrite_config(
            config, "io/catalogue_name", catalogue)

        fname_out = join(
            local_config["io"]["root_output"],
            f"{kind}_{catalogue}_{tag}.hdf5")
        local_config = overwrite_config(
            local_config, "io/fname_output", fname_out)

        # Dump the configuration to a file
        toml_out = join(
            local_config["root_main"], splitext(fname_out)[0] + ".toml")
        fprint(f"writing the configuration file to `{toml_out}`")
        with open(toml_out, "wb") as f:
            tomli_w.dump(local_config, f)

        if to_submit:
            cmd = ["./_submit_run.sh", toml_out]
            fprint(f"Submitting job with: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                fprint(f"Submission failed with return code {e.returncode}")
                fprint(f"Command: {' '.join(e.cmd)}")
                raise
