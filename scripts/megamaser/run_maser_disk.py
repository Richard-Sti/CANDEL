"""Generic maser disk inference script.

Usage:
    python run_maser_disk.py <galaxy_name> [--no-phi-prior] [--seed N]

Galaxy name and v_sys_obs are read from config_maser.toml.
"""
import os, sys, tomli
with open(os.path.join(os.path.dirname(__file__), "../../local_config.toml"), "rb") as f:
    _lcfg = tomli.load(f)
ld = os.environ.get("LD_LIBRARY_PATH", "")
needed = [p for p in _lcfg.get("gpu_ld_library_path", []) if p not in ld]
if needed:
    os.environ["LD_LIBRARY_PATH"] = ":".join(needed) + (f":{ld}" if ld else "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")
import argparse
import tempfile
import numpy as np
import jax.numpy as jnp
import tomli
import tomli_w
import time
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
from jax import random

from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.model.model_H0_maser import MaserDiskModel
from candel.util import fprint, fsection
import jax
print(f"JAX platform: {jax.default_backend()}, devices: {jax.devices()}", flush=True)

# ---- Parse args ----
parser = argparse.ArgumentParser()
parser.add_argument("galaxy", type=str)
parser.add_argument("--no-phi-prior", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-warmup", type=int, default=1000)
parser.add_argument("--num-samples", type=int, default=1000)
args = parser.parse_args()

galaxy = args.galaxy

# ---- Load master config to get v_sys_obs ----
with open("scripts/megamaser/config_maser.toml", "rb") as f:
    master_cfg = tomli.load(f)

galaxies = master_cfg["model"]["galaxies"]
if galaxy not in galaxies:
    print(f"Unknown galaxy '{galaxy}'. "
          f"Available: {list(galaxies.keys())}", flush=True)
    sys.exit(1)

v_sys_obs = galaxies[galaxy]["v_sys_obs"]

# ---- Load data ----
fsection(f"Loading {galaxy} data")
data = load_megamaser_spots("data/Megamaser", galaxy, v_sys_obs=v_sys_obs)

# ---- Build config ----
use_phi_prior = not args.no_phi_prior

dense_mass_blocks = [
    ["D_c", "log_MBH", "dv_sys"],
    ["i0", "di_dr"],
    ["Omega0", "dOmega_dr"],
    ["x0", "y0"],
]
if use_phi_prior:
    dense_mass_blocks += [
        ["phi_mu_red", "phi_sigma_red"],
        ["phi_mu_blue", "phi_sigma_blue"],
        ["phi_mu_sys", "phi_sigma_sys"],
    ]
dense_mass_blocks.append(["sigma_a_floor_sys", "sigma_a_floor_hv"])

priors = {
    "H0": {"dist": "delta", "value": 73.0},
    "sigma_pec": {"dist": "delta", "value": 250.0},
    "D": {"dist": "data_estimate_uniform", "half_width": 30.0},
    "log_MBH": {"dist": "data_estimate_truncated_normal",
                "scale": 0.5, "low": 6.0, "high": 9.0},
    "R_phys": {"dist": "uniform", "low": 0.01, "high": 3.0},
    "x0": {"dist": "uniform", "low": -500.0, "high": 500.0},
    "y0": {"dist": "uniform", "low": -500.0, "high": 500.0},
    "i0": {"dist": "uniform", "low": 60.0, "high": 110.0},
    "Omega0": {"dist": "uniform", "low": 0.0, "high": 360.0},
    "dOmega_dr": {"dist": "uniform", "low": -30.0, "high": 30.0},
    "di_dr": {"dist": "uniform", "low": -30.0, "high": 30.0},
    "dv_sys": {"dist": "uniform", "low": -100.0, "high": 100.0},
    "sigma_x_floor": {"dist": "truncated_normal",
                      "mean": 10.0, "scale": 10.0,
                      "low": 0.0, "high": 100.0},
    "sigma_y_floor": {"dist": "truncated_normal",
                      "mean": 10.0, "scale": 10.0,
                      "low": 0.0, "high": 100.0},
    "sigma_v_sys": {"dist": "truncated_normal",
                    "mean": 2.0, "scale": 2.0,
                    "low": 0.0, "high": 20.0},
    "sigma_v_hv": {"dist": "truncated_normal",
                   "mean": 2.0, "scale": 2.0,
                   "low": 0.0, "high": 20.0},
    "sigma_a_floor_sys": {"dist": "truncated_normal",
                          "mean": 0.1, "scale": 0.2,
                          "low": 0.0, "high": 1.0},
    "sigma_a_floor_hv": {"dist": "truncated_normal",
                         "mean": 0.1, "scale": 0.2,
                         "low": 0.0, "high": 1.0},
}

if use_phi_prior:
    priors.update({
        "phi_mu_red": {"dist": "uniform", "low": 0.0, "high": 90.0},
        "phi_sigma_red": {"dist": "uniform", "low": 1.0, "high": 90.0},
        "phi_mu_blue": {"dist": "uniform", "low": 0.0, "high": 90.0},
        "phi_sigma_blue": {"dist": "uniform", "low": 1.0, "high": 90.0},
        "phi_mu_sys": {"dist": "uniform", "low": -45.0, "high": 45.0},
        "phi_sigma_sys": {"dist": "uniform", "low": 1.0, "high": 90.0},
    })

config = {
    "inference": {
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": 1,
        "chain_method": "sequential",
        "seed": args.seed,
        "dense_mass_blocks": dense_mass_blocks,
        "init_maxiter": 0,
        "max_tree_depth": 10,
    },
    "model": {
        "which_run": "maser_disk",
        "Om": 0.315,
        "use_selection": False,
        "marginalise_r": True,
        "phi_prior": use_phi_prior,
        "priors": priors,
    },
    "io": {"fname_output": f"results/Maser/{galaxy}.hdf5"},
}

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()
model = MaserDiskModel(tmp.name, data)
os.unlink(tmp.name)

# ---- Run MCMC ----
n_spots = data["n_spots"]
mode = "phi prior" if use_phi_prior else "no phi prior"
fsection(f"Running NUTS ({galaxy}, {n_spots} spots, {mode})")
t0 = time.time()
kernel = NUTS(model, max_tree_depth=10, target_accept_prob=0.8,
              init_strategy=init_to_median(num_samples=20))
mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples,
            num_chains=1, progress_bar=True)
mcmc.run(random.PRNGKey(args.seed))
dt = time.time() - t0
mcmc.print_summary(exclude_deterministic=True)

samples = mcmc.get_samples()
n_div = int(mcmc.get_extra_fields()['diverging'].sum())
print(f"\nWall time: {dt:.0f}s, Divergences: {n_div}", flush=True)

# ---- Print results ----
fsection("Results")
param_keys = ['D_c', 'log_MBH', 'i0', 'di_dr', 'Omega0', 'dOmega_dr',
              'x0', 'y0', 'dv_sys',
              'sigma_x_floor', 'sigma_y_floor',
              'sigma_v_sys', 'sigma_v_hv',
              'sigma_a_floor_sys', 'sigma_a_floor_hv']
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

outpath = f"results/Maser/{galaxy}_mode2.npz"
np.savez(outpath,
         **{k: np.asarray(samples[k]) for k in samples if k != 'r_ang'},
         D_A=D_A, M_BH=M_BH)
print(f"Saved to {outpath}", flush=True)
