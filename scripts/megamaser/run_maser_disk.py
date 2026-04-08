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
"""Generic maser disk inference script (NUTS or NSS nested sampling).

Usage:
    python run_maser_disk.py <galaxy_name> [--sampler nuts|nss] [options]

All settings (priors, sampler config) are read from config_maser.toml.
CLI args override config values where provided.
"""
import os, sys, tomli
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
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tomli
import tomli_w
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median, init_to_value

from candel.inference.inference import print_clean_summary
from candel.inference.nested import print_nested_summary, run_nss
from candel.model.model_H0_maser import MaserDiskModel
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import fprint, fsection, plot_corner

print(f"JAX platform: {jax.default_backend()}, devices: {jax.devices()}", flush=True)

# ---- Load master config ----
with open("scripts/megamaser/config_maser.toml", "rb") as f:
    master_cfg = tomli.load(f)

inf_cfg = master_cfg["inference"]

# ---- Parse args (CLI overrides config) ----
parser = argparse.ArgumentParser()
parser.add_argument("galaxy", type=str)
parser.add_argument("--sampler", type=str, default=None,
                    choices=["nuts", "nss"])
parser.add_argument("--phi-prior", action="store_true")
parser.add_argument("--seed", type=int, default=None)
# NUTS
parser.add_argument("--num-warmup", type=int, default=None)
parser.add_argument("--num-samples", type=int, default=None)
# NSS
parser.add_argument("--n-live", type=int, default=None)
parser.add_argument("--num-mcmc-steps", type=int, default=None)
parser.add_argument("--num-delete", type=int, default=None)
parser.add_argument("--termination", type=float, default=None)
args = parser.parse_args()

galaxy = args.galaxy
sampler = args.sampler or inf_cfg.get("sampler", "nss")
seed = args.seed or inf_cfg.get("seed", 42)

# Distance-prior tag for output filenames
_D_TAG = {
    "data_estimate_uniform": "Dflat",
    "data_estimate_volume":  "Dvol",
    "uniform":               "Dflat",
}
_d_prior_dist = master_cfg["model"]["priors"]["D"].get("dist", "unknown")
dist_tag = _D_TAG.get(_d_prior_dist, _d_prior_dist.replace("_", ""))

# ---- Validate galaxy ----
galaxies = master_cfg["model"]["galaxies"]
if galaxy not in galaxies:
    print(f"Unknown galaxy '{galaxy}'. "
          f"Available: {list(galaxies.keys())}", flush=True)
    sys.exit(1)

v_sys_obs = galaxies[galaxy]["v_sys_obs"]

# ---- Load data ----
fsection(f"Loading {galaxy} data")
data = load_megamaser_spots("data/Megamaser", galaxy, v_sys_obs=v_sys_obs)

# ---- Build model config from master config ----
use_phi_prior = args.phi_prior or master_cfg["model"].get("phi_prior", False)

dense_mass_blocks = inf_cfg.get("dense_mass_blocks", [
    ["D_c", "eta", "dv_sys"],
    ["i0", "di_dr", "Omega0", "dOmega_dr"],
    ["x0", "y0"],
])

config = {
    "inference": {
        "num_warmup": args.num_warmup or inf_cfg.get("num_warmup", 1000),
        "num_samples": args.num_samples or inf_cfg.get("num_samples", 1000),
        "num_chains": inf_cfg.get("num_chains", 1),
        "chain_method": inf_cfg.get("chain_method", "sequential"),
        "seed": seed,
        "dense_mass_blocks": dense_mass_blocks,
        "init_maxiter": inf_cfg.get("init_maxiter", 0),
        "init_method": inf_cfg.get("init_method", "lbfgs"),
        "max_tree_depth": inf_cfg.get("max_tree_depth", 10),
    },
    "model": master_cfg["model"],
    "io": master_cfg["io"],
}
# Ensure phi_prior flag is set
config["model"]["phi_prior"] = use_phi_prior

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()
model = MaserDiskModel(tmp.name, data)
os.unlink(tmp.name)

# ---- Run sampler ----
n_spots = data["n_spots"]
phi_mode = "phi prior" if use_phi_prior else "no phi prior"

if sampler == "nuts":
    num_warmup = args.num_warmup or inf_cfg.get("num_warmup", 1000)
    num_samples = args.num_samples or inf_cfg.get("num_samples", 1000)

    fsection(f"Running NUTS ({galaxy}, {n_spots} spots, {phi_mode})")
    init_method = inf_cfg.get("init_method", "median")
    if init_method == "sobol_adam":
        from candel.inference.optimise import find_MAP
        init_params = find_MAP(model, model_kwargs={}, seed=seed)
        init_strategy = init_to_value(values=init_params)
    else:
        init_strategy = init_to_median(num_samples=20)
    t0 = time.time()
    kernel = NUTS(model, max_tree_depth=inf_cfg.get("max_tree_depth", 10),
                  target_accept_prob=0.8,
                  init_strategy=init_strategy)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=1, progress_bar=True)
    mcmc.run(random.PRNGKey(seed))
    dt = time.time() - t0

    samples = mcmc.get_samples()
    n_div = int(mcmc.get_extra_fields()['diverging'].sum())
    print(f"\nWall time: {dt:.0f}s, Divergences: {n_div}", flush=True)
    suffix = f"nuts_{dist_tag}"
    meta = None

elif sampler == "nss":
    n_live = args.n_live or inf_cfg.get("n_live", 5000)
    num_mcmc_steps = args.num_mcmc_steps or inf_cfg.get("num_mcmc_steps", 0)
    num_delete = args.num_delete or inf_cfg.get("num_delete", 250)
    termination = args.termination or inf_cfg.get("termination", -3)

    if num_mcmc_steps == 0:
        num_mcmc_steps = None  # run_nss will use ndim

    fsection(f"Running NSS ({galaxy}, {n_spots} spots, {phi_mode})")
    fprint(f"n_live={n_live}, mcmc_steps={num_mcmc_steps}, "
           f"num_delete={num_delete}")
    t0 = time.time()
    samples = run_nss(
        model, model_kwargs={},
        n_live=n_live, num_mcmc_steps=num_mcmc_steps,
        num_delete=num_delete,
        termination=termination, seed=seed,
    )
    dt = time.time() - t0

    meta = samples.pop("__nested__")
    print(f"\nWall time: {dt:.0f}s", flush=True)
    print(f"log Z = {meta['log_Z']:.2f} +/- {meta['log_Z_err']:.2f}",
          flush=True)
    print(f"n_eff = {meta['n_eff']}", flush=True)
    suffix = f"nss_{dist_tag}"

# ---- Print results ----
fsection("Results")
param_keys = ['D_c', 'eta', 'log_MBH', 'i0', 'di_dr', 'Omega0', 'dOmega_dr',
              'x0', 'y0', 'dv_sys',
              'sigma_x_floor', 'sigma_y_floor',
              'sigma_v_sys', 'sigma_v_hv',
              'sigma_a_floor']
if use_phi_prior:
    param_keys += ['phi_mu_red', 'phi_sigma_red',
                   'phi_mu_blue', 'phi_sigma_blue',
                   'phi_mu_sys', 'phi_sigma_sys']
for k in param_keys:
    if k in samples:
        s = np.asarray(samples[k])
        print(f"  {k:20s} = {s.mean():10.3f} +/- {s.std():8.3f}", flush=True)

D_c = np.asarray(samples['D_c'])
z_cosmo = D_c * 73.0 / 299792.458
D_A = D_c / (1 + z_cosmo)
M_BH = 10**np.asarray(samples['log_MBH'])
print(f"\n  D_A = {D_A.mean():.1f} +/- {D_A.std():.1f} Mpc", flush=True)
print(f"  M_BH = {M_BH.mean():.2e} +/- {M_BH.std():.2e} M_sun", flush=True)

# Summary table
fsection("Summary")
if sampler == "nuts":
    print_clean_summary(samples)
else:
    print_nested_summary(samples, meta=meta)

# Spot classification plot
outdir = master_cfg["io"].get("root_output", "results/Maser")
os.makedirs(outdir, exist_ok=True)

_cls = np.where(~data["is_highvel"], "systemic",
                np.where(data["is_blue"], "blue HV", "red HV"))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for cls, col in [("systemic", "forestgreen"), ("blue HV", "royalblue"),
                 ("red HV", "tomato")]:
    m = _cls == cls
    axes[0].scatter(data["x"][m], data["y"][m], c=col, s=12, alpha=0.7,
                    label=f"{cls} ({m.sum()})")
    axes[1].scatter(data["x"][m],
                    data["velocity"][m] - data["v_sys_obs"],
                    c=col, s=12, alpha=0.7)
axes[0].set_xlabel(r"$\Delta x$ (mas)")
axes[0].set_ylabel(r"$\Delta y$ (mas)")
axes[0].legend(fontsize=8)
axes[0].invert_xaxis()
axes[1].set_xlabel(r"$\Delta x$ (mas)")
axes[1].set_ylabel(r"$v - v_\mathrm{sys}$ (km s$^{-1}$)")
fig.tight_layout()
fname_spots = os.path.join(outdir, f"{galaxy}_{suffix}_spots.png")
fig.savefig(fname_spots, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Spot classification plot saved to {fname_spots}", flush=True)

# Corner plots (smoothed and unsmoothed)
fname_corner = os.path.join(outdir, f"{galaxy}_{suffix}_corner.png")
plot_corner(samples, show_fig=False, filename=fname_corner)
fname_corner_raw = os.path.join(outdir, f"{galaxy}_{suffix}_corner_raw.png")
plot_corner(samples, show_fig=False, filename=fname_corner_raw, smooth=False)

outpath = os.path.join(outdir, f"{galaxy}_{suffix}.npz")
save_dict = {k: np.asarray(samples[k]) for k in samples if k != 'r_ang'}
save_dict.update(D_A=D_A, M_BH=M_BH)
if meta is not None:
    save_dict.update(log_Z=meta['log_Z'], log_Z_err=meta['log_Z_err'])
np.savez(outpath, **save_dict)
print(f"Saved to {outpath}", flush=True)
