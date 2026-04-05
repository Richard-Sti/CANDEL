"""Profile Mode 2 (marginalise_r) forward+gradient to find bottlenecks."""
import sys
sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import time
import tempfile
import numpy as np
import jax.numpy as jnp
import tomli_w
from jax import random

from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.model.model_H0_maser import MaserDiskModel
from candel.util import fprint, fsection
from numpyro.infer.util import potential_energy

# ---- Load data ----
data = load_megamaser_spots("data/Megamaser", "NGC5765b", v_sys_obs=8327.6)

config = {
    "inference": {"num_warmup": 10, "num_samples": 10, "num_chains": 1,
                  "chain_method": "sequential", "seed": 42,
                  "init_maxiter": 0, "max_tree_depth": 5},
    "model": {
        "which_run": "maser_disk", "Om": 0.315,
        "use_selection": False, "fit_di_dr": True,
        "marginalise_r": True,
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
    "io": {"fname_output": "/dev/null"},
}

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()

model = MaserDiskModel(tmp.name, data)

# Test point
params = {
    'D_c': jnp.array(129.6),
    'log_MBH': jnp.array(7.66),
    'x0': jnp.array(-44.0),
    'y0': jnp.array(-93.0),
    'i0': jnp.array(73.0),
    'di_dr': jnp.array(12.7),
    'Omega0': jnp.array(149.2),
    'dOmega_dr': jnp.array(-2.7),
    'dv_sys': jnp.array(5.0),
    'sigma_x_floor': jnp.array(12.0),
    'sigma_y_floor': jnp.array(3.0),
    'sigma_v_sys': jnp.array(1.8),
    'sigma_v_hv': jnp.array(3.6),
    'sigma_a_floor': jnp.array(0.08),
}

# ---- Compile and time forward ----
fsection("Profiling Mode 2")
print(f"Spots: {model.n_spots} ({model._n_sys} sys, "
      f"{model._n_red} red, {model._n_blue} blue)")
print(f"r_ang grid: {len(model._r_ang_grid)} points")
print(f"phi grids: HV half={len(model._sin_phi1_red)}, "
      f"sys={len(model._sin_phi_sys)}")

n_r = len(model._r_ang_grid)
n_phi_hv = len(model._sin_phi1_red)
n_phi_sys = len(model._sin_phi_sys)
print(f"\nGrid sizes per spot type:")
print(f"  Systemic: {model._n_sys} spots × {n_r} r × {n_phi_sys} phi "
      f"= {model._n_sys * n_r * n_phi_sys:,} evals")
print(f"  Red HV:   {model._n_red} spots × {n_r} r × {n_phi_hv} phi "
      f"= {model._n_red * n_r * n_phi_hv:,} evals")
print(f"  Blue HV:  {model._n_blue} spots × {n_r} r × {n_phi_hv} phi "
      f"= {model._n_blue * n_r * n_phi_hv:,} evals")
total = (model._n_sys * n_r * n_phi_sys
         + (model._n_red + model._n_blue) * n_r * n_phi_hv)
print(f"  TOTAL: {total:,} forward model evaluations per likelihood call")

# Compile
pe_fn = jax.jit(lambda p: potential_energy(model, model_args=(), model_kwargs={},
                                            params=p))
print("\nCompiling forward...")
t0 = time.time()
pe = pe_fn(params)
jax.block_until_ready(pe)
print(f"  JIT compile: {time.time()-t0:.1f}s, PE = {float(pe):.1f}")

# Time forward
N = 20
t0 = time.time()
for _ in range(N):
    pe = pe_fn(params)
    jax.block_until_ready(pe)
dt_fwd = (time.time() - t0) / N
print(f"  Forward: {dt_fwd*1000:.1f} ms")

# Compile gradient
print("\nCompiling gradient...")
grad_fn = jax.jit(jax.grad(lambda p: potential_energy(
    model, model_args=(), model_kwargs={}, params=p)))
t0 = time.time()
g = grad_fn(params)
jax.block_until_ready(g)
print(f"  JIT compile: {time.time()-t0:.1f}s")

# Time gradient
t0 = time.time()
for _ in range(N):
    g = grad_fn(params)
    jax.block_until_ready(g)
dt_grad = (time.time() - t0) / N
print(f"  Gradient: {dt_grad*1000:.1f} ms")
print(f"  Grad/fwd ratio: {dt_grad/dt_fwd:.1f}x")

# ---- Now test with reduced grids ----
for n_r_test, n_phi_hv_test, n_phi_sys_test in [
        (251, 252, 503),  # current
        (101, 101, 201),  # reduced
        (51, 51, 101),    # aggressive
]:
    print(f"\n--- Grid: n_r={n_r_test}, n_phi_hv={n_phi_hv_test}, "
          f"n_phi_sys={n_phi_sys_test} ---")
    evals = (model._n_sys * n_r_test * n_phi_sys_test
             + (model._n_red + model._n_blue) * n_r_test * n_phi_hv_test)
    print(f"  Total evals: {evals:,} "
          f"({evals/total:.1%} of current)")

os.unlink(tmp.name)
