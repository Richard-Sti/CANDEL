"""Quick test: fit NGC5765b with marginalised r and proper priors."""
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

# ---- Config: marginalise_r eliminates per-spot params ----
config = {
    "inference": {
        "num_warmup": 500,
        "num_samples": 500,
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
        "marginalise_r": True,
        "priors": {
            "H0": {"dist": "delta", "value": 73.0},
            "sigma_pec": {"dist": "delta", "value": 250.0},
            "D": {"dist": "uniform", "low": 50.0, "high": 200.0},
            "log_MBH": {"dist": "uniform", "low": 6.5, "high": 9.0},
            "R_phys": {"dist": "uniform", "low": 0.01, "high": 3.0},
            "x0": {"dist": "uniform", "low": -500.0, "high": 500.0},
            "y0": {"dist": "uniform", "low": -500.0, "high": 500.0},
            "i0": {"dist": "uniform", "low": 70.0, "high": 110.0},
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
        "fname_output": "results/Maser/test_ngc5765b_margr.hdf5",
    },
}

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()

# ---- Build model ----
model = MaserDiskModel(tmp.name, data)

# ---- Init near Gao+2016 values ----
z_est = 8327.6 / 299792.458
D_c_est = 126.3 * (1 + z_est)

init_values = {
    'D_c': jnp.array(D_c_est),
    'log_MBH': jnp.array(np.log10(4.55e7)),
    'x0': jnp.array(-44.0),     # uas
    'y0': jnp.array(-100.0),    # uas
    'i0': jnp.array(94.5),
    'di_dr': jnp.array(-10.6),
    'Omega0': jnp.array(146.7),
    'dOmega_dr': jnp.array(-3.46),
    'dv_sys': jnp.array(0.0),
    'sigma_x_floor': jnp.array(3.0),
    'sigma_y_floor': jnp.array(3.0),
    'sigma_v_sys': jnp.array(1.0),
    'sigma_v_hv': jnp.array(1.5),
    'sigma_a_floor': jnp.array(0.04),
}

# ---- Run NUTS ----
fsection("Running NUTS (marginalised r)")
kernel = NUTS(model, max_tree_depth=10, target_accept_prob=0.8,
              init_strategy=init_to_value(values=init_values))
mcmc = MCMC(kernel, num_warmup=500, num_samples=500,
            num_chains=1, progress_bar=True)
mcmc.run(random.PRNGKey(42))
mcmc.print_summary(exclude_deterministic=True)

samples = mcmc.get_samples()
n_div = int(mcmc.get_extra_fields()['diverging'].sum())
print(f"\nDivergences: {n_div}")

# ---- Report ----
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
    print(f"\n  D_A (approx)         = {D_A.mean():10.3f} +/- {D_A.std():8.3f}")

print("\n  Gao+2016 reference:")
print(f"  D_A = 126.3 +/- 11.6 Mpc")
print(f"  M_BH = 4.55e7 +/- 0.40e7 Msun (log10 = 7.658)")
print(f"  i0 = 94.5 +/- 0.25 deg")
print(f"  Omega0 = 146.7 +/- 0.125 deg")
print(f"  di/dr = -10.6 +/- 1.13 deg/mas")

os.unlink(tmp.name)
