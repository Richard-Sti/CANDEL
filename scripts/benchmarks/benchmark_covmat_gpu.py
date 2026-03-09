"""
GPU benchmark for the fast PV covariance tensor decomposition.

Times assembly, Cholesky, and full forward+gradient for N = 50, 500, 1000
on GPU using JAX JIT.

Usage:
    python benchmark_covmat_gpu.py
"""
import json
import os
import subprocess
import sys
import time

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["PYTHONUNBUFFERED"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from candel.cosmo.pv_covariance import compute_dD_dtau
from candel.cosmo.pv_covmat_fast import (
    assemble_pv_covariance_jax,
    precompute_psi_functions,
    pv_covariance_log_likelihood,
)
from candel.util import radec_to_cartesian


def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {"name": parts[0], "mem_used_mb": int(parts[1]),
                    "mem_total_mb": int(parts[2]), "util_pct": int(parts[3])}
    except Exception:
        pass
    return None


def generate_galaxies(n_gal, seed=42):
    rng = np.random.default_rng(seed)
    RA = rng.uniform(0, 360, n_gal)
    cos_dec = rng.uniform(-1, 1, n_gal)
    dec = np.degrees(np.arcsin(cos_dec))
    r = 100 * rng.uniform(0, 1, n_gal)**(1. / 3)
    return r, RA, dec


def benchmark_size(N, s_jax, pp_jax, pperp_jax, sv3, n_warmup=5, n_rep=50):
    """Benchmark assembly, Cholesky, and gradient for a given N."""
    r, RA, dec = generate_galaxies(N)
    rhat = radec_to_cartesian(RA, dec)

    r_jax = jnp.array(r, dtype=jnp.float32)
    rhat_jax = jnp.array(rhat, dtype=jnp.float32)

    rng = np.random.default_rng(123)
    v_obs = jnp.array(rng.normal(0, 200, N), dtype=jnp.float32)
    v_pred = jnp.array(rng.normal(0, 100, N), dtype=jnp.float32)
    beta = 0.4
    sigma_v = 150.0

    # --- Assembly only ---
    @jax.jit
    def asm(r_):
        return assemble_pv_covariance_jax(
            r_, rhat_jax, s_jax, pp_jax, pperp_jax, sv3)

    # JIT compile
    t0 = time.perf_counter()
    C = asm(r_jax); C.block_until_ready()
    jit_asm_s = time.perf_counter() - t0

    for _ in range(n_warmup):
        C = asm(r_jax); C.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        C = asm(r_jax); C.block_until_ready()
    dt_asm = (time.perf_counter() - t0) / n_rep * 1000

    # --- Assembly + Cholesky ---
    @jax.jit
    def asm_chol(r_):
        C = assemble_pv_covariance_jax(
            r_, rhat_jax, s_jax, pp_jax, pperp_jax, sv3)
        C_total = beta**2 * C + sigma_v**2 * jnp.eye(N)
        L = jnp.linalg.cholesky(C_total)
        return C, L

    t0 = time.perf_counter()
    C, L = asm_chol(r_jax); L.block_until_ready()
    jit_chol_s = time.perf_counter() - t0

    for _ in range(n_warmup):
        C, L = asm_chol(r_jax); L.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        C, L = asm_chol(r_jax); L.block_until_ready()
    dt_chol = (time.perf_counter() - t0) / n_rep * 1000

    # --- Full log-likelihood ---
    @jax.jit
    def log_like(r_):
        C = assemble_pv_covariance_jax(
            r_, rhat_jax, s_jax, pp_jax, pperp_jax, sv3)
        return pv_covariance_log_likelihood(v_obs, v_pred, C, beta, sigma_v)

    t0 = time.perf_counter()
    ll = log_like(r_jax); ll.block_until_ready()
    jit_ll_s = time.perf_counter() - t0

    for _ in range(n_warmup):
        ll = log_like(r_jax); ll.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        ll = log_like(r_jax); ll.block_until_ready()
    dt_ll = (time.perf_counter() - t0) / n_rep * 1000

    # --- Gradient of log-likelihood ---
    grad_fn = jax.jit(jax.grad(log_like))
    t0 = time.perf_counter()
    g = grad_fn(r_jax); g.block_until_ready()
    jit_grad_s = time.perf_counter() - t0

    for _ in range(n_warmup):
        g = grad_fn(r_jax); g.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        g = grad_fn(r_jax); g.block_until_ready()
    dt_grad = (time.perf_counter() - t0) / n_rep * 1000

    return {
        "N": N,
        "assembly_ms": round(dt_asm, 3),
        "asm_chol_ms": round(dt_chol, 3),
        "cholesky_ms": round(dt_chol - dt_asm, 3),
        "log_like_ms": round(dt_ll, 3),
        "gradient_ms": round(dt_grad, 3),
        "jit_asm_s": round(jit_asm_s, 2),
        "jit_chol_s": round(jit_chol_s, 2),
        "jit_ll_s": round(jit_ll_s, 2),
        "jit_grad_s": round(jit_grad_s, 2),
    }


def main():
    gpu = get_gpu_info()
    print(f"Platform: {jax.default_backend()}")
    if gpu:
        print(f"GPU: {gpu['name']} ({gpu['mem_total_mb']} MB)")
    print()

    # Synthetic P(k)
    k = np.logspace(-4, np.log10(20), 5000)
    Pk = 2e4 * (k / 0.05)**0.96 / (1 + (k / 0.05)**1.5)**2
    dDdtau = compute_dD_dtau()

    print("Precomputing psi functions...")
    t0 = time.perf_counter()
    s_grid, psi_par, psi_perp, sv3 = precompute_psi_functions(
        k, Pk, dDdtau, s_max=300, n_s=5000)
    print(f"  Done in {time.perf_counter() - t0:.2f} s")
    print(f"  sigma_v = {np.sqrt(3 * sv3):.1f} km/s\n")

    s_jax = jnp.array(s_grid, dtype=jnp.float32)
    pp_jax = jnp.array(psi_par, dtype=jnp.float32)
    pperp_jax = jnp.array(psi_perp, dtype=jnp.float32)

    sizes = [50, 500, 1000]
    results = []

    for N in sizes:
        print(f"--- N = {N} ---")
        if gpu:
            g = get_gpu_info()
            print(f"  GPU mem before: {g['mem_used_mb']}/{g['mem_total_mb']} MB")

        r = benchmark_size(N, s_jax, pp_jax, pperp_jax, float(sv3))
        results.append(r)

        print(f"  Assembly:    {r['assembly_ms']:.3f} ms "
              f"(JIT: {r['jit_asm_s']:.2f} s)")
        print(f"  Cholesky:    {r['cholesky_ms']:.3f} ms "
              f"(JIT: {r['jit_chol_s']:.2f} s)")
        print(f"  Log-like:    {r['log_like_ms']:.3f} ms "
              f"(JIT: {r['jit_ll_s']:.2f} s)")
        print(f"  Gradient:    {r['gradient_ms']:.3f} ms "
              f"(JIT: {r['jit_grad_s']:.2f} s)")

        if gpu:
            g = get_gpu_info()
            print(f"  GPU mem after: {g['mem_used_mb']}/{g['mem_total_mb']} MB")
        print()

    # Summary table
    print(f"{'='*70}")
    print(f"SUMMARY (GPU: {gpu['name'] if gpu else 'unknown'})")
    print(f"{'='*70}")
    print(f"{'N':>6} {'assembly':>12} {'cholesky':>12} {'log-like':>12} "
          f"{'gradient':>12}")
    print(f"{'-'*54}")
    for r in results:
        print(f"{r['N']:>6} {r['assembly_ms']:>10.3f} ms "
              f"{r['cholesky_ms']:>10.3f} ms "
              f"{r['log_like_ms']:>10.3f} ms "
              f"{r['gradient_ms']:>10.3f} ms")

    # Save
    outfile = os.path.join(
        os.path.dirname(__file__), "results", "covmat_gpu_benchmark.json")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump({"gpu": gpu, "results": results}, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
