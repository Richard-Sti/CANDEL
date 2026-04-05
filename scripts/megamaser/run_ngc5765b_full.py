"""Full fit: NGC5765b with all 192 spots, di/dr included."""
import sys
sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import tempfile
import numpy as np
import jax.numpy as jnp
import tomli_w
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from jax import random

from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.model.model_H0_maser import MaserDiskModel
from candel.util import fprint, fsection

# ---- Load data ----
fsection("Loading NGC5765b data")
data = load_megamaser_spots("data/Megamaser", "NGC5765b", v_sys_obs=8327.6)

# ---- Config ----
config = {
    "inference": {
        "num_warmup": 500,
        "num_samples": 1000,
        "num_chains": 1,
        "chain_method": "sequential",
        "seed": 42,
        "dense_mass_blocks": [
            ["D_c", "log_MBH", "dv_sys"],
            ["i0", "di_dr"],
            ["Omega0", "dOmega_dr"],
            ["x0", "y0"],
        ],
        "init_maxiter": 0,
        "max_tree_depth": 10,
    },
    "model": {
        "which_run": "maser_disk",
        "Om": 0.315,
        "use_selection": False,
        "fit_di_dr": True,
        "marginalise_r": False,
        "priors": {
            "H0": {"dist": "delta", "value": 73.0},
            "sigma_pec": {"dist": "delta", "value": 250.0},
            "D": {"dist": "uniform", "low": 50.0, "high": 200.0},
            "log_MBH": {"dist": "uniform", "low": 6.5, "high": 9.0},
            "R_phys": {"dist": "uniform", "low": 0.01, "high": 3.0},
            "x0": {"dist": "uniform", "low": -500.0, "high": 500.0},
            "y0": {"dist": "uniform", "low": -500.0, "high": 500.0},
            "i0": {"dist": "uniform", "low": 60.0, "high": 110.0},
            "Omega0": {"dist": "uniform", "low": 100.0, "high": 200.0},
            "dOmega_dr": {"dist": "uniform", "low": -30.0, "high": 30.0},
            "di_dr": {"dist": "uniform", "low": -30.0, "high": 30.0},
            "dv_sys": {"dist": "normal", "loc": 0.0, "scale": 500.0},
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
            "sigma_a_floor": {"dist": "uniform",
                              "low": 0.0, "high": 5.0},
        },
    },
    "io": {
        "fname_output": "results/Maser/NGC5765b_full.hdf5",
    },
}

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()

model = MaserDiskModel(tmp.name, data)

# Init near published values (Pesce parameterization)
z_est = 8327.6 / 299792.458
D_c_init = 126.3 * (1 + z_est)

# Rough r_ang init from impact parameter
x0_init, y0_init = -0.044, -0.100
dx = data["x"] - x0_init
dy = data["y"] - y0_init
impact = np.sqrt(dx**2 + dy**2)
r_init = np.clip(impact / 0.5, model._r_ang_lo + 0.01,
                 model._r_ang_hi - 0.01)

init_values = {
    'D_c': jnp.array(D_c_init),
    'log_MBH': jnp.array(np.log10(4.55e7)),
    'x0': jnp.array(-44.0),
    'y0': jnp.array(-100.0),
    'i0': jnp.array(73.0),
    'di_dr': jnp.array(12.7),
    'Omega0': jnp.array(149.2),
    'dOmega_dr': jnp.array(-2.7),
    'dv_sys': jnp.array(5.0),
    'sigma_x_floor': jnp.array(12.0),
    'sigma_y_floor': jnp.array(2.0),
    'sigma_v_sys': jnp.array(1.8),
    'sigma_v_hv': jnp.array(3.6),
    'sigma_a_floor': jnp.array(0.08),
    'r_ang': jnp.array(r_init),
}

# Run
fsection("Running NUTS (full 192 spots)")
kernel = NUTS(model, max_tree_depth=10, target_accept_prob=0.8,
              init_strategy=init_to_value(values=init_values))
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000,
            num_chains=1, progress_bar=True)
mcmc.run(random.PRNGKey(42))
mcmc.print_summary(exclude_deterministic=True)

samples = mcmc.get_samples()
n_div = int(mcmc.get_extra_fields()['diverging'].sum())
print(f"\nDivergences: {n_div}")

# Report
fsection("Results")
for k in ['D_c', 'log_MBH', 'i0', 'di_dr', 'Omega0', 'dOmega_dr',
          'x0', 'y0', 'dv_sys',
          'sigma_x_floor', 'sigma_y_floor',
          'sigma_v_sys', 'sigma_v_hv', 'sigma_a_floor']:
    if k in samples:
        s = np.asarray(samples[k])
        print(f"  {k:20s} = {s.mean():10.3f} +/- {s.std():8.3f}  "
              f"[{np.percentile(s, 16):.3f}, {np.percentile(s, 84):.3f}]")

if 'D_c' in samples:
    D_c = np.asarray(samples['D_c'])
    z_cosmo = D_c * 73.0 / 299792.458
    D_A = D_c / (1 + z_cosmo)
    M_BH = 10**np.asarray(samples['log_MBH'])
    H0_implied = 299792.458 * z_cosmo.mean() / D_A.mean()
    print(f"\n  D_A                  = {D_A.mean():10.3f} +/- {D_A.std():8.3f}")
    print(f"  M_BH                 = {M_BH.mean():.2e} +/- {M_BH.std():.2e}")
    print(f"  H0 (implied)         = {H0_implied:.1f}")

# Save samples
save_dict = {}
for k in samples:
    if k == 'r_ang':
        continue  # don't save per-spot
    save_dict[k] = np.asarray(samples[k])
# Add derived quantities
save_dict['D_A'] = np.asarray(D_A)
save_dict['M_BH'] = np.asarray(M_BH)
np.savez("results/Maser/NGC5765b_full_samples.npz", **save_dict)
print("Saved to results/Maser/NGC5765b_full_samples.npz")

print("\n  === References ===")
print("  Gao+2016: D_A=126.3±11.6, M_BH=4.55e7, i0=94.5, di/dr=-10.6")
print("  Pesce+2020: D_A=112.2+5.4/-5.1 (revised)")

os.unlink(tmp.name)
