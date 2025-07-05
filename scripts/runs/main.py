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
A short script to run inference on a PV model. It loads the model and data
from the configuration file and runs the inference.

This script is expected to be run either from the command line or from a shell
submission script.
"""

import sys
import time
from argparse import ArgumentParser
from os.path import exists

import candel
from candel import fprint, get_nested


def insert_comment_at_top(path: str, label: str):
    """Insert a comment line with a timestamp at the top of a file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    comment = f"# Job {label} at: {timestamp}\n"

    with open(path, "r") as f:
        original = f.readlines()

    with open(path, "w") as f:
        f.write(comment)
        f.writelines(original)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run inference on a PV model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    args = parser.parse_args()

    insert_comment_at_top(args.config, "started")

    data = candel.pvdata.load_PV_dataframes(args.config)
    config = candel.load_config(args.config)

    fname_out = get_nested(config, "io/fname_output")
    skip_if_exists = get_nested(config, "inference/skip_if_exists", False)
    if skip_if_exists and exists(fname_out):
        fprint(f"Output file `{fname_out}` already exists. "
               f"Skipping inference.")
        insert_comment_at_top(args.config, "skipped")
        sys.exit(0)

    model_name = config["inference"]["model"]
    data_name = config["io"]["catalogue_name"]

    fprint(f"Loading model `{model_name}` from `{args.config}` for "
           f"data `{data_name}`")

    shared_param = get_nested(config, "inference/shared_params", None)
    model = candel.model.name2model(model_name, shared_param, args.config)

    if isinstance(data, list):
        if not isinstance(model, candel.model.JointPVModel):
            raise TypeError(
                "You provided multiple datasets, but the selected model "
                f"`{model.__class__.__name__}` is not JointPVModel.")

        if len(data) != len(model.submodels):
            raise ValueError(
                f"Number of datasets ({len(data)}) does not match number "
                f"of submodels ({len(model.submodels)}) in the joint model.")

    model_kwargs = {"data": data, }
    candel.run_pv_inference(model, model_kwargs)

    insert_comment_at_top(args.config, "finished")
