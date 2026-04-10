"""Benchmark Mode 2 maser model on GPU: forward + gradient timing."""
import os
import time

import jax
# Test both float32 and float64
import jax.numpy as jnp
import tempfile
import tomli_w

from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.model.model_H0_maser import MaserDiskModel
from numpyro.infer.util import potential_energy

print(f"JAX devices: {jax.devices()}")
print(f"JAX platform: {jax.default_backend()}")

data = load_megamaser_spots("data/Megamaser", "NGC5765b", v_sys_obs=8327.6)

config = {
    "inference": {"num_warmup": 1, "num_samples": 1, "num_chains": 1,
                  "chain_method": "sequential", "seed": 42,
                  "init_maxiter": 0, "max_tree_depth": 5},
    "model": {
        "which_run": "maser_disk", "Om": 0.315,
        "use_selection": False,
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
            "sigma_a_floor": {"dist": "truncated_normal",
                              "mean": 0.1, "scale": 0.2,
                              "low": 0.0, "high": 0.3},
        },
    },
    "io": {"fname_output": "/dev/null"},
}


def bench(label, use_f64):
    if use_f64:
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)

    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, tmp)
    tmp.close()
    model = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)

    dtype = jnp.float64 if use_f64 else jnp.float32
    params = {k: jnp.array(v, dtype=dtype) for k, v in {
        'D_c': 129.6, 'log_MBH': 7.66,
        'x0': -44.0, 'y0': -93.0,
        'i0': 73.0, 'di_dr': 12.7,
        'Omega0': 149.2, 'dOmega_dr': -2.7,
        'dv_sys': 5.0,
        'sigma_x_floor': 12.0, 'sigma_y_floor': 3.0,
        'sigma_v_sys': 1.8, 'sigma_v_hv': 3.6,
        'sigma_a_floor': 0.08,
    }.items()}

    pe_fn = jax.jit(lambda p: potential_energy(
        model, model_args=(), model_kwargs={}, params=p))
    grad_fn = jax.jit(jax.grad(lambda p: potential_energy(
        model, model_args=(), model_kwargs={}, params=p)))

    # Warmup / compile
    pe = pe_fn(params)
    jax.block_until_ready(pe)
    g = grad_fn(params)
    jax.block_until_ready(g)

    N = 100
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

    print(f"  {label:<20s}  fwd={dt_fwd:7.2f} ms  grad={dt_grad:7.2f} ms  "
          f"PE={float(pe):.2f}")
    return dt_fwd, dt_grad


print("\n=== GPU Benchmark: Mode 2 (marginalise_r) ===")
print(f"    192 spots, 251 r × 252/503 phi grid")
print()

fwd_32, grad_32 = bench("float32", use_f64=False)
fwd_64, grad_64 = bench("float64", use_f64=True)

print(f"\n  float64/float32 ratio:  fwd={fwd_64/fwd_32:.1f}x  "
      f"grad={grad_64/grad_32:.1f}x")
print(f"\n  At 100 leapfrog steps: {100 * grad_32 / 1000:.1f}s (f32), "
      f"{100 * grad_64 / 1000:.1f}s (f64)")
print(f"  At 1023 leapfrog steps: {1023 * grad_32 / 1000:.1f}s (f32), "
      f"{1023 * grad_64 / 1000:.1f}s (f64)")
