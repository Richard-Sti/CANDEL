"""NGC5765b fit on GPU with marginalised r — tests whether
marginalising over per-spot radii makes the posterior less sensitive
to the initial guess (no multimodality trap).

Runs two chains from different starting points:
  Chain 0: init near Pesce+2020 solution (i0=73, di_dr=12.7)
  Chain 1: init at prior median (i0=85, di_dr=0) — the "bad" start

If marginalisation fixes the multimodality, both chains should
converge to the same distance.
"""
import sys
sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")

import os
import time

import jax
# float32 is sufficient for maser disk model and ~32x faster on consumer GPUs

import tempfile
import numpy as np
import jax.numpy as jnp
import tomli_w
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value, init_to_median
from jax import random

from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.model.model_H0_maser import MaserDiskModel
from candel.util import fprint, fsection

print(f"JAX devices: {jax.devices()}")
print(f"JAX platform: {jax.default_backend()}")

# ---- Load data ----
fsection("Loading NGC5765b data")
data = load_megamaser_spots("data/Megamaser", "NGC5765b", v_sys_obs=8327.6)

# ---- Config: marginalise_r = True ----
config = {
    "inference": {
        "num_warmup": 1000,
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
    "io": {
        "fname_output": "results/Maser/NGC5765b_gpu_margr.hdf5",
    },
}

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()

model = MaserDiskModel(tmp.name, data)

# ---- Define two init strategies ----
# Init A: near Pesce+2020 (the "good" start)
z_est = 8327.6 / 299792.458
D_c_good = 129.0 * (1 + z_est)

init_good = {
    'D_c': jnp.array(D_c_good),
    'log_MBH': jnp.array(np.log10(4.78e7)),
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

# Init B: prior median (the "bad" start — what caused the trap before)
init_bad = {
    'D_c': jnp.array(125.0),
    'log_MBH': jnp.array(7.75),
    'x0': jnp.array(0.0),
    'y0': jnp.array(0.0),
    'i0': jnp.array(85.0),
    'di_dr': jnp.array(0.0),
    'Omega0': jnp.array(150.0),
    'dOmega_dr': jnp.array(0.0),
    'dv_sys': jnp.array(0.0),
    'sigma_x_floor': jnp.array(10.0),
    'sigma_y_floor': jnp.array(10.0),
    'sigma_v_sys': jnp.array(2.0),
    'sigma_v_hv': jnp.array(2.0),
    'sigma_a_floor': jnp.array(0.5),
}


def run_chain(label, init_values, seed):
    """Run a single chain and return samples + diagnostics."""
    fsection(f"Chain: {label}")
    kernel = NUTS(model, max_tree_depth=10, target_accept_prob=0.8,
                  init_strategy=init_to_value(values=init_values))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000,
                num_chains=1, progress_bar=True)

    t0 = time.time()
    mcmc.run(random.PRNGKey(seed))
    dt = time.time() - t0

    mcmc.print_summary(exclude_deterministic=True)
    samples = mcmc.get_samples()
    n_div = int(mcmc.get_extra_fields()['diverging'].sum())

    print(f"\n  Wall time: {dt:.1f}s ({dt/2000:.3f} s/sample)")
    print(f"  Divergences: {n_div}")

    # Key params
    D_c = np.asarray(samples['D_c'])
    z_cosmo = D_c * 73.0 / 299792.458
    D_A = D_c / (1 + z_cosmo)
    M_BH = 10**np.asarray(samples['log_MBH'])

    for k in ['D_c', 'log_MBH', 'i0', 'di_dr', 'Omega0', 'dOmega_dr',
              'x0', 'y0', 'dv_sys',
              'sigma_x_floor', 'sigma_y_floor',
              'sigma_v_sys', 'sigma_v_hv', 'sigma_a_floor']:
        if k in samples:
            s = np.asarray(samples[k])
            print(f"  {k:20s} = {s.mean():10.3f} +/- {s.std():8.3f}")

    print(f"\n  D_A                  = {D_A.mean():10.3f} +/- {D_A.std():8.3f}")
    print(f"  M_BH                 = {M_BH.mean():.2e} +/- {M_BH.std():.2e}")
    return samples, n_div, dt


# ---- Run both chains ----
samples_good, div_good, t_good = run_chain("good_init", init_good, seed=42)
samples_bad, div_bad, t_bad = run_chain("bad_init (prior median)", init_bad,
                                        seed=123)

# ---- Compare ----
fsection("Comparison: good init vs bad init")
print(f"{'Parameter':<20} {'Good init':>20} {'Bad init':>20} {'Agree?':>8}")
print("-" * 70)

for k in ['D_c', 'log_MBH', 'i0', 'di_dr', 'Omega0', 'dOmega_dr']:
    sg = np.asarray(samples_good[k])
    sb = np.asarray(samples_bad[k])
    diff = abs(sg.mean() - sb.mean())
    avg_std = 0.5 * (sg.std() + sb.std())
    agree = "YES" if diff < 2 * avg_std else "NO"
    print(f"{k:<20} {sg.mean():>9.2f}+/-{sg.std():.2f}"
          f" {sb.mean():>9.2f}+/-{sb.std():.2f} {agree:>8}")

# D_A comparison
D_A_good = np.asarray(samples_good['D_c']) / (
    1 + np.asarray(samples_good['D_c']) * 73.0 / 299792.458)
D_A_bad = np.asarray(samples_bad['D_c']) / (
    1 + np.asarray(samples_bad['D_c']) * 73.0 / 299792.458)
diff = abs(D_A_good.mean() - D_A_bad.mean())
avg_std = 0.5 * (D_A_good.std() + D_A_bad.std())
agree = "YES" if diff < 2 * avg_std else "NO"
print(f"{'D_A':<20} {D_A_good.mean():>9.2f}+/-{D_A_good.std():.2f}"
      f" {D_A_bad.mean():>9.2f}+/-{D_A_bad.std():.2f} {agree:>8}")

print(f"\nDivergences: good={div_good}, bad={div_bad}")
print(f"Wall time:   good={t_good:.0f}s, bad={t_bad:.0f}s")

print("\nReferences: Gao+2016 D_A=126.3, Pesce+2020 D_A=112.2")

# Save both
np.savez("results/Maser/NGC5765b_gpu_margr_good.npz",
         **{k: np.asarray(v) for k, v in samples_good.items()})
np.savez("results/Maser/NGC5765b_gpu_margr_bad.npz",
         **{k: np.asarray(v) for k, v in samples_bad.items()})
print("Saved results.")

os.unlink(tmp.name)
