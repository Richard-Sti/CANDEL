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

import time
from argparse import ArgumentParser

import candel
from candel import fprint


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

    model_name = config["inference"]["model"]
    data_name = config["io"]["catalogue_name"]

    fprint(f"Loading model `{model_name}` from `{args.config}` for "
           f"data `{data_name}`")

    model = candel.model.name2model(model_name)(args.config)
    model_kwargs = {"data": data, }
    candel.run_pv_inference(model, model_kwargs)

    insert_comment_at_top(args.config, "finished")
