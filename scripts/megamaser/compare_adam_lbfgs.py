"""Compare Sobol+Adam vs Sobol+L-BFGS-B for MAP finding.

Usage:
    python scripts/megamaser/compare_adam_lbfgs.py
"""
import os
import sys
import time

import tomli

with open(os.path.join(os.path.dirname(__file__),
                       "../../local_config.toml"), "rb") as f:
    _lcfg = tomli.load(f)
ld = os.environ.get("LD_LIBRARY_PATH", "")
needed = [p for p in _lcfg.get("gpu_ld_library_path", []) if p not in ld]
if needed:
    os.environ["LD_LIBRARY_PATH"] = ":".join(needed) + (f":{ld}" if ld else "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tomli
import tomli_w
from scipy.stats.qmc import Sobol
from tqdm import trange

from candel.inference.optimise import (
    _build_logp_flat,
    _get_bounds_from_trace,
    _print_points_table,
    _select_distinct,
)
from candel.model.model_H0_maser import MaserDiskModel
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import fprint, fsection, patch_tqdm

patch_tqdm()
print(f"JAX: {jax.default_backend()}, {jax.devices()}", flush=True)

# ---- Config ----
with open("scripts/megamaser/config_maser.toml", "rb") as f:
    master_cfg = tomli.load(f)

galaxy = "NGC5765b"
seed = 42
gcfg = master_cfg["model"]["galaxies"][galaxy]
data = load_megamaser_spots("data/Megamaser", galaxy,
                            v_sys_obs=gcfg["v_sys_obs"])
if "D_lo" in gcfg and "D_hi" in gcfg:
    data["D_lo"] = float(gcfg["D_lo"])
    data["D_hi"] = float(gcfg["D_hi"])

config = {
    "inference": {
        "num_warmup": 1, "num_samples": 1, "num_chains": 1,
        "chain_method": "sequential", "seed": seed,
        "init_maxiter": 0, "max_tree_depth": 5,
        "init_method": "sobol_adam",
    },
    "model": master_cfg["model"],
    "io": master_cfg["io"],
    "optimise": master_cfg.get("optimise", {}),
}
tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()
model = MaserDiskModel(tmp.name, data)
os.unlink(tmp.name)

# ---- Build flat log-density and bounds ----
names, sizes, lo_sobol, hi_sobol = _get_bounds_from_trace(
    model, (), {}, sobol_n_sigma=1, seed=seed)
_, _, lo_clip, hi_clip = _get_bounds_from_trace(
    model, (), {}, sobol_n_sigma=10, seed=seed)
D = len(lo_sobol)

logp_fn = _build_logp_flat(model, (), {}, names, sizes)
logp_batch = jax.jit(jax.vmap(logp_fn))

# ---- Sobol survey (shared) ----
fsection(f"Sobol survey ({D}D)")
N_sobol = 2**14
sampler = Sobol(d=D, scramble=True, seed=seed)
sobol_01 = sampler.random(N_sobol)
sobol_points = lo_sobol + sobol_01 * (hi_sobol - lo_sobol)

# Compile
_ = logp_batch(jnp.array(sobol_points[:4]))
jax.block_until_ready(_)

t0 = time.time()
logp_all = []
batch_size = 1024
for i in range(0, N_sobol, batch_size):
    vals = logp_batch(jnp.array(sobol_points[i:i + batch_size]))
    jax.block_until_ready(vals)
    logp_all.append(np.asarray(vals))
logp_all = np.concatenate(logp_all)
logp_all = np.where(np.isfinite(logp_all), logp_all, -np.inf)
fprint(f"Sobol done in {time.time() - t0:.1f}s, "
       f"best logP = {logp_all.max():.1f}")

# Select M=10 starts
M = 10
selected = _select_distinct(sobol_points, logp_all, M, 0.05)
x0 = sobol_points[selected]
fprint(f"Selected {len(selected)} starts")

lo_jax = jnp.array(lo_clip)
hi_jax = jnp.array(hi_clip)
eps = 1e-6 * (hi_jax - lo_jax)

# ========== Adam ==========
fsection("Adam (5000 steps, cosine LR)")
n_steps = 5000
lr, lr_end, n_restarts = 0.1, 0.005, 3
steps_per_cycle = n_steps // n_restarts
boundaries = [steps_per_cycle * i for i in range(1, n_restarts)]
schedules = [
    optax.cosine_decay_schedule(
        init_value=lr, decay_steps=steps_per_cycle,
        alpha=lr_end / lr)
    for _ in range(n_restarts)
]
schedule = optax.join_schedules(schedules, boundaries)
optimizer = optax.adam(schedule)


@jax.jit
def adam_step(x, opt_state):
    def _single(xi, osi):
        g = jax.grad(lambda z: -logp_fn(z))(xi)
        updates, new_osi = optimizer.update(g, osi)
        xi_new = optax.apply_updates(xi, updates)
        xi_new = jnp.clip(xi_new, lo_jax + eps, hi_jax - eps)
        return xi_new, new_osi
    return jax.vmap(_single)(x, opt_state)


x_adam = jnp.array(x0)
opt_state = jax.vmap(optimizer.init)(x_adam)

# Compile
x_adam, opt_state = adam_step(x_adam, opt_state)
jax.block_until_ready(x_adam)

t0 = time.time()
for step in trange(1, n_steps, desc="Adam"):
    x_adam, opt_state = adam_step(x_adam, opt_state)
jax.block_until_ready(x_adam)
dt_adam = time.time() - t0

logp_adam = np.asarray(logp_batch(x_adam))
fprint(f"Adam: {dt_adam:.1f}s, best logP = {logp_adam.max():.2f}")
fsection("Adam final points")
_print_points_table(np.asarray(x_adam), logp_adam, names, sizes)

# ========== L-BFGS-B (scipy, JAX autodiff gradients) ==========
fsection("L-BFGS-B (scipy, JAX gradients)")
from scipy.optimize import minimize as scipy_minimize

# JAX value-and-grad, compiled once
_val_and_grad = jax.jit(jax.value_and_grad(lambda x: -logp_fn(x)))

# Warm up JIT
_v, _g = _val_and_grad(jnp.array(x0[0]))
jax.block_until_ready((_v, _g))

# Barrier: add steep penalty near boundaries
barrier_scale = 1e4


def scipy_objective(x_np):
    x_jax = jnp.array(x_np)
    val, grad = _val_and_grad(x_jax)
    val, grad = float(val), np.asarray(grad, dtype=np.float64)

    # Log-barrier: penalise approaching bounds
    w = hi_sobol - lo_sobol
    dist_lo = (x_np - lo_sobol) / w
    dist_hi = (hi_sobol - x_np) / w
    # Clip to avoid log(0)
    dist_lo = np.clip(dist_lo, 1e-10, None)
    dist_hi = np.clip(dist_hi, 1e-10, None)
    barrier = -np.sum(np.log(dist_lo) + np.log(dist_hi))
    barrier_grad = -(1.0 / (dist_lo * w) - 1.0 / (dist_hi * w))

    return val + barrier / barrier_scale, grad + barrier_grad / barrier_scale


bounds_scipy = list(zip(lo_sobol + 1e-8 * (hi_sobol - lo_sobol),
                        hi_sobol - 1e-8 * (hi_sobol - lo_sobol)))

t0 = time.time()
x_scipy = []
logp_scipy = []
for i in range(M):
    res = scipy_minimize(
        scipy_objective, x0[i], method='L-BFGS-B', jac=True,
        bounds=bounds_scipy,
        options={'maxiter': 10000, 'maxfun': 50000,
                 'ftol': 1e-12, 'gtol': 1e-8})
    x_scipy.append(res.x)
    # Evaluate true logP (without barrier)
    lp = float(logp_fn(jnp.array(res.x)))
    logp_scipy.append(lp)
    fprint(f"  start {i}: logP={lp:.2f}, nit={res.nit}, "
           f"success={res.success}")
dt_scipy = time.time() - t0

x_scipy = np.array(x_scipy)
logp_scipy = np.array(logp_scipy)
fprint(f"\nScipy L-BFGS-B: {dt_scipy:.1f}s, "
       f"best logP = {logp_scipy.max():.2f}")
fsection("Scipy L-BFGS-B final points")
_print_points_table(x_scipy, logp_scipy, names, sizes)

# ========== Summary ==========
fsection("Comparison")
fprint(f"{'Method':<20} {'Time':>8} {'Best logP':>12}")
fprint(f"{'-'*20} {'-'*8} {'-'*12}")
fprint(f"{'Adam':<20} {dt_adam:>7.1f}s {logp_adam.max():>12.2f}")
fprint(f"{'Scipy L-BFGS-B':<20} {dt_scipy:>7.1f}s {logp_scipy.max():>12.2f}")
