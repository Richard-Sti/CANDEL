# Copyright (C) 2026 Richard Stiskalek
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
"""Run DE MAP optimization for a maser disk galaxy and save results to TOML.

Usage:
    python run_de_map.py <galaxy_name> [--seed 42]
"""
import os
import sys

import tomli

with open(os.path.join(os.path.dirname(__file__), "../../local_config.toml"), "rb") as f:
    _lcfg = tomli.load(f)
ld = os.environ.get("LD_LIBRARY_PATH", "")
needed = [p for p in _lcfg.get("gpu_ld_library_path", []) if p not in ld]
if needed:
    os.environ["LD_LIBRARY_PATH"] = ":".join(needed) + (f":{ld}" if ld else "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import tempfile
import time

import jax
import numpy as np
import tomli
import tomli_w

from candel.inference.optimise import find_MAP
from candel.model.model_H0_maser import MaserDiskModel
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import fprint, fsection

_devs = jax.devices()
_dev_names = ", ".join(d.device_kind for d in _devs)
print(f"JAX platform: {jax.default_backend()}, devices: {_devs} ({_dev_names})",
      flush=True)

# ---- Load master config ----
with open("scripts/megamaser/config_maser.toml", "rb") as f:
    master_cfg = tomli.load(f)

# ---- Parse args ----
parser = argparse.ArgumentParser()
parser.add_argument("galaxy", type=str)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

galaxy = args.galaxy
seed = args.seed

# ---- Validate galaxy ----
galaxies = master_cfg["model"]["galaxies"]
if galaxy not in galaxies:
    print(f"Unknown galaxy '{galaxy}'. "
          f"Available: {list(galaxies.keys())}", flush=True)
    sys.exit(1)

gcfg = galaxies[galaxy]
v_sys_obs = gcfg["v_sys_obs"]

# ---- Load data ----
fsection(f"Loading {galaxy} data")
_mcfg = master_cfg["model"]
data = load_megamaser_spots("data/Megamaser", galaxy, v_sys_obs=v_sys_obs,
                           clump_galaxies=_mcfg.get("clump_galaxies"))

if "D_lo" in gcfg and "D_hi" in gcfg:
    data["D_lo"] = float(gcfg["D_lo"])
    data["D_hi"] = float(gcfg["D_hi"])

# ---- Build model ----
config = {
    "inference": master_cfg["inference"],
    "model": master_cfg["model"],
    "io": master_cfg["io"],
    "optimise": master_cfg.get("optimise", {}),
}

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()
model = MaserDiskModel(tmp.name, data)
os.unlink(tmp.name)

# ---- Run DE MAP ----
fsection(f"DE MAP optimization ({galaxy}, {data['n_spots']} spots)")
t0 = time.time()
init_params = find_MAP(model, model_kwargs={}, seed=seed)
dt = time.time() - t0

# ---- Print results ----
fsection(f"MAP results ({galaxy}, {dt:.0f}s)")
for k, v in sorted(init_params.items()):
    v = np.asarray(v)
    if v.ndim == 0:
        fprint(f"  {k:20s} = {float(v):12.4f}")
    else:
        fprint(f"  {k:20s} = [{len(v)} values]")

# ---- Save to config_maser.toml under [model.galaxies.<galaxy>.init] ----
cfg_path = os.path.join(os.path.dirname(__file__), "config_maser.toml")
with open(cfg_path, "rb") as f:
    cfg_data = tomli.load(f)

init_section = {}
for k, v in init_params.items():
    v = np.asarray(v)
    if v.ndim == 0:
        init_section[k] = round(float(v), 4)
    else:
        init_section[k] = [round(float(x), 4) for x in v]
cfg_data["model"]["galaxies"][galaxy]["init"] = init_section

with open(cfg_path, "wb") as f:
    tomli_w.dump(cfg_data, f)

fprint(f"Saved MAP init to {cfg_path} under [model.galaxies.{galaxy}.init]")
