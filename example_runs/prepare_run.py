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


import os
import sys
from datetime import datetime
from copy import deepcopy

template_path = "config.toml"
template = toml.load(template_path)

# Example: change the data band and output path
def prepare_config(data_band="w1", tag="001"):
    config = deepcopy(template)
    
    # Modify output path
    output_path = f"results/samples_band_{data_band}_run_{tag}.hdf5"
    config["io"]["fname_output"] = output_path
    
    # Add band override (your `load_CF4_data` might read this from config)
    config["data"] = {"band": data_band}

    # Save new config
    config_name = f"runs/run_{tag}.toml"
    os.makedirs("runs", exist_ok=True)
    with open(config_name, "w") as f:
        toml.dump(config, f)

    return config_name


if __name__ == "__main__":
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = prepare_config(data_band=sys.argv[1], tag=tag)

    # Optionally submit (SLURM/parallel call/etc.)
    os.system(f"python main.py {config_path}")




def process_config(config):
    def _walk(d, path=()):
        for k, v in d.items():
            full_path = path + (k,)
            if isinstance(v, dict):
                _walk(v, full_path)
            elif isinstance(v, str) and v.strip().lower() == "none":
                d[k] = None

    _walk(config)
    return config