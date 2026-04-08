"""Compare Mode 2 speed and accuracy across grid resolutions."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import time
import tempfile
import numpy as np
import jax.numpy as jnp
import tomli_w

from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.model.model_H0_maser import MaserDiskModel
from candel.util import fprint, fsection
from numpyro.infer.util import potential_energy

data = load_megamaser_spots("data/Megamaser", "NGC5765b", v_sys_obs=8327.6)

params = {
    'D_c': jnp.array(129.6), 'log_MBH': jnp.array(7.66),
    'x0': jnp.array(-44.0), 'y0': jnp.array(-93.0),
    'i0': jnp.array(73.0), 'di_dr': jnp.array(12.7),
    'Omega0': jnp.array(149.2), 'dOmega_dr': jnp.array(-2.7),
    'dv_sys': jnp.array(5.0),
    'sigma_x_floor': jnp.array(12.0), 'sigma_y_floor': jnp.array(3.0),
    'sigma_v_sys': jnp.array(1.8), 'sigma_v_hv': jnp.array(3.6),
    'sigma_a_floor': jnp.array(0.08),
}

base_config = {
    "inference": {"num_warmup": 1, "num_samples": 1, "num_chains": 1,
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


def time_model(config, label, N=20):
    """Build model and time forward + gradient."""
    import copy
    cfg = copy.deepcopy(config)
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(cfg, tmp)
    tmp.close()

    model = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)

    pe_fn = jax.jit(lambda p: potential_energy(
        model, model_args=(), model_kwargs={}, params=p))
    pe = pe_fn(params)
    jax.block_until_ready(pe)

    grad_fn = jax.jit(jax.grad(lambda p: potential_energy(
        model, model_args=(), model_kwargs={}, params=p)))
    g = grad_fn(params)
    jax.block_until_ready(g)

    t0 = time.time()
    for _ in range(N):
        pe = pe_fn(params)
        jax.block_until_ready(pe)
    dt_fwd = (time.time() - t0) / N * 1000

    t0 = time.time()
    for _ in range(N):
        g = grad_fn(params)
        jax.block_until_ready(g)
    dt_grad = (time.time() - t0) / N * 1000

    print(f"  {label:<30s} fwd={dt_fwd:7.1f} ms  grad={dt_grad:7.1f} ms  "
          f"PE={float(pe):.2f}")
    return float(pe), dt_fwd, dt_grad


fsection("Grid resolution comparison (CPU)")

# Reference: full grids (defaults for non-marginalise_r mode)
cfg_ref = {**base_config}
cfg_ref["model"]["n_phi_hv_half"] = 251
cfg_ref["model"]["n_phi_sys"] = 501
cfg_ref["model"]["n_r"] = 251
pe_ref, _, _ = time_model(cfg_ref, "251r × 252phi × 503phi")

# Auto defaults for marginalise_r (101, 101, 201)
time_model(base_config, "101r × 102phi × 203phi (auto)")

# Sweep
for n_r, n_phi_hv, n_phi_sys in [
    (151, 151, 301),
    (101, 101, 201),
    (71, 71, 141),
    (51, 51, 101),
    (31, 31, 61),
]:
    cfg = {**base_config}
    cfg["model"]["n_phi_hv_half"] = n_phi_hv
    cfg["model"]["n_phi_sys"] = n_phi_sys
    cfg["model"]["n_r"] = n_r
    pe, _, _ = time_model(cfg, f"{n_r}r × {n_phi_hv+1}phi × {n_phi_sys+2}phi")
    print(f"    ΔPE = {pe - pe_ref:+.4f}")
