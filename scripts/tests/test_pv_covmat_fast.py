"""
Tests for pv_covmat_fast vs the existing pv_covariance implementation.

Verifies:
  1. precompute_psi_functions produces sensible ψ_∥, ψ_⊥
  2. assemble_pv_covariance_numpy matches compute_covariance_matrix
  3. assemble_pv_covariance_jax matches the NumPy version
  4. JAX version is JIT-compilable
  5. JAX version is differentiable (gradient check)
  6. pv_covariance_log_likelihood is correct
  7. Timing comparison

Usage:
    python test_pv_covmat_fast.py
    python test_pv_covmat_fast.py --n_gal 50 --n_jobs 12
"""
import argparse
import os
import sys
import time

os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from candel.cosmo.pv_covariance import (  # noqa
    compute_covariance_matrix,
    compute_dD_dtau,
)

from candel.cosmo.pv_covariance import get_Pk_CAMB  # noqa

try:
    import camb  # noqa
    HAS_CAMB = True
except ImportError:
    HAS_CAMB = False
from candel.cosmo.pv_covmat_fast import (  # noqa
    assemble_pv_covariance_jax,
    assemble_pv_covariance_numpy,
    precompute_psi_functions,
    pv_covariance_log_likelihood,
)
from candel.util import radec_to_cartesian  # noqa


def generate_test_galaxies(n_gal, seed=42):
    """Generate random galaxy positions."""
    rng = np.random.default_rng(seed)
    RA = rng.uniform(0, 360, n_gal)
    dec = rng.uniform(-90, 90, n_gal)
    r = rng.uniform(20, 200, n_gal)
    return r, RA, dec


def test_psi_functions(k, Pk, dDdtau):
    """Test that ψ functions are well-behaved."""
    print("=" * 60)
    print("TEST 1: Precompute ψ functions")
    print("=" * 60)

    t0 = time.perf_counter()
    s_grid, psi_par, psi_perp, sigma_v_sq_3 = precompute_psi_functions(
        k, Pk, dDdtau, s_max=600, n_s=10000)
    dt = time.perf_counter() - t0

    print(f"  Precompute time: {dt:.3f} s")
    print(f"  σ²_v/3 = {sigma_v_sq_3:.2f} (km/s)²")
    print(f"  σ_v = {np.sqrt(3 * sigma_v_sq_3):.1f} km/s")
    print(f"  ψ_∥(0) = {psi_par[0]:.2f}, ψ_⊥(0) = {psi_perp[0]:.2f}")
    print(f"  ψ_∥(100) = {psi_par[np.searchsorted(s_grid, 100)]:.2f}")
    print(f"  ψ_∥(300) = {psi_par[np.searchsorted(s_grid, 300)]:.2f}")

    # Sanity checks
    assert np.isfinite(psi_par).all(), "ψ_∥ has non-finite values"
    assert np.isfinite(psi_perp).all(), "ψ_⊥ has non-finite values"
    assert np.abs(psi_par[0] - psi_perp[0]) < 1e-6 * np.abs(psi_par[0]), \
        f"ψ_∥(0) ≠ ψ_⊥(0): {psi_par[0]} vs {psi_perp[0]}"
    assert sigma_v_sq_3 > 0, "σ²_v/3 must be positive"
    # ψ should decay at large s
    assert np.abs(psi_par[-1]) < np.abs(psi_par[0]) * 0.01, \
        "ψ_∥ should decay at large s"

    print("  PASSED\n")
    return s_grid, psi_par, psi_perp, sigma_v_sq_3


def test_numpy_vs_old(r, RA, dec, k, Pk, dDdtau, s_grid, psi_par, psi_perp,
                      sigma_v_sq_3, n_jobs=1):
    """Test that the tensor decomposition matches the Bessel-sum code."""
    print("=" * 60)
    print("TEST 2: NumPy tensor decomposition vs Bessel-sum")
    print("=" * 60)

    n_gal = len(r)
    rhat = radec_to_cartesian(RA, dec)

    # Old implementation
    print(f"  Computing old Bessel-sum C ({n_gal}×{n_gal}, "
          f"n_jobs={n_jobs})...")
    t0 = time.perf_counter()
    C_old = compute_covariance_matrix(
        r, RA, dec, k, Pk, dDdtau, ell_max=2000, n_jobs=n_jobs)
    dt_old = time.perf_counter() - t0
    print(f"  Old time: {dt_old:.2f} s")

    # New implementation
    print("  Computing new tensor-decomposition C...")
    t0 = time.perf_counter()
    C_new = assemble_pv_covariance_numpy(
        r, rhat, s_grid, psi_par, psi_perp, sigma_v_sq_3)
    dt_new = time.perf_counter() - t0
    print(f"  New time: {dt_new:.4f} s")

    # Compare
    diag_old = np.diag(C_old)
    diag_new = np.diag(C_new)
    print(f"  Old diagonal range: [{diag_old.min():.2f}, {diag_old.max():.2f}]")
    print(f"  New diagonal (all): {diag_new[0]:.2f}")

    # Off-diagonal comparison
    mask = ~np.eye(n_gal, dtype=bool)
    off_old = C_old[mask]
    off_new = C_new[mask]

    abs_err = np.abs(off_old - off_new)
    rel_err = abs_err / (np.abs(off_old) + 1e-10)

    print(f"  Off-diagonal abs error: max={abs_err.max():.4f}, "
          f"mean={abs_err.mean():.4f}")
    print(f"  Off-diagonal rel error: max={rel_err.max():.4f}, "
          f"mean={rel_err.mean():.4f}")
    print(f"  Speedup: {dt_old / dt_new:.0f}x")

    # The diagonal is different by construction: old code computes it from
    # the Bessel sum while new code uses σ²_v/3. They should agree.
    diag_rel = np.abs(diag_old - diag_new) / np.abs(diag_old)
    print(f"  Diagonal rel error: max={diag_rel.max():.6f}, "
          f"mean={diag_rel.mean():.6f}")

    # Check symmetry
    assert np.allclose(C_new, C_new.T), "C_new is not symmetric"

    # Tolerance for agreement
    rtol = 5e-3
    passed = rel_err.max() < rtol and diag_rel.max() < rtol
    if passed:
        print(f"  PASSED (rtol={rtol})\n")
    else:
        print(f"  FAILED (rtol={rtol})\n")
        print(f"  Worst off-diag pair: "
              f"old={off_old[abs_err.argmax()]:.4f}, "
              f"new={off_new[abs_err.argmax()]:.4f}")

    return C_old, C_new, passed


def test_jax_vs_numpy(r, RA, dec, s_grid, psi_par, psi_perp, sigma_v_sq_3,
                      C_numpy):
    """Test JAX assembly matches NumPy assembly."""
    print("=" * 60)
    print("TEST 3: JAX assembly vs NumPy assembly")
    print("=" * 60)

    import jax.numpy as jnp

    rhat = radec_to_cartesian(RA, dec)

    r_jax = jnp.array(r)
    rhat_jax = jnp.array(rhat)
    s_jax = jnp.array(s_grid)
    pp_jax = jnp.array(psi_par)
    pperp_jax = jnp.array(psi_perp)

    t0 = time.perf_counter()
    C_jax = assemble_pv_covariance_jax(
        r_jax, rhat_jax, s_jax, pp_jax, pperp_jax, sigma_v_sq_3)
    C_jax_np = np.array(C_jax)
    dt = time.perf_counter() - t0
    print(f"  JAX assembly time (incl. tracing): {dt:.4f} s")

    mask = ~np.eye(len(r), dtype=bool)
    abs_err = np.abs(C_numpy[mask] - C_jax_np[mask])
    rel_err = abs_err / (np.abs(C_numpy[mask]) + 1e-10)
    print(f"  Off-diagonal abs error: max={abs_err.max():.6f}, "
          f"mean={abs_err.mean():.6f}")
    print(f"  Off-diagonal rel error: max={rel_err.max():.6f}, "
          f"mean={rel_err.mean():.6f}")

    rtol = 1e-3
    passed = rel_err.max() < rtol
    print(f"  {'PASSED' if passed else 'FAILED'} (rtol={rtol})\n")
    return C_jax, passed


def test_jit(r, RA, dec, s_grid, psi_par, psi_perp, sigma_v_sq_3):
    """Test JIT compilation and timing."""
    print("=" * 60)
    print("TEST 4: JIT compilation + timing")
    print("=" * 60)

    import jax
    import jax.numpy as jnp

    rhat = radec_to_cartesian(RA, dec)
    r_jax = jnp.array(r)
    rhat_jax = jnp.array(rhat)
    s_jax = jnp.array(s_grid)
    pp_jax = jnp.array(psi_par)
    pperp_jax = jnp.array(psi_perp)

    @jax.jit
    def assemble(r_):
        return assemble_pv_covariance_jax(
            r_, rhat_jax, s_jax, pp_jax, pperp_jax, sigma_v_sq_3)

    # JIT compile
    t0 = time.perf_counter()
    C = assemble(r_jax)
    C.block_until_ready()
    jit_time = time.perf_counter() - t0
    print(f"  JIT compile: {jit_time:.3f} s")

    # Warm up
    for _ in range(3):
        C = assemble(r_jax)
        C.block_until_ready()

    # Time
    n_rep = 50
    t0 = time.perf_counter()
    for _ in range(n_rep):
        C = assemble(r_jax)
        C.block_until_ready()
    dt = (time.perf_counter() - t0) / n_rep * 1000
    print(f"  Assembly time: {dt:.3f} ms (N={len(r)})")
    print(f"  PASSED\n")
    return True


def test_gradients(r, RA, dec, s_grid, psi_par, psi_perp, sigma_v_sq_3):
    """Test that JAX autodiff works through the assembly + log-likelihood."""
    print("=" * 60)
    print("TEST 5: Gradient check (autodiff vs finite differences)")
    print("=" * 60)

    import jax
    import jax.numpy as jnp

    rhat = radec_to_cartesian(RA, dec)
    N = len(r)

    r_jax = jnp.array(r, dtype=jnp.float64)
    rhat_jax = jnp.array(rhat, dtype=jnp.float64)
    s_jax = jnp.array(s_grid, dtype=jnp.float64)
    pp_jax = jnp.array(psi_par, dtype=jnp.float64)
    pperp_jax = jnp.array(psi_perp, dtype=jnp.float64)

    rng = np.random.default_rng(123)
    v_obs = jnp.array(rng.normal(0, 200, N), dtype=jnp.float64)
    v_pred = jnp.array(rng.normal(0, 100, N), dtype=jnp.float64)
    beta = 0.4
    sigma_v = 150.0

    def log_like_fn(r_):
        C = assemble_pv_covariance_jax(
            r_, rhat_jax, s_jax, pp_jax, pperp_jax,
            float(sigma_v_sq_3))
        return pv_covariance_log_likelihood(v_obs, v_pred, C, beta, sigma_v)

    # Autodiff gradient
    grad_fn = jax.grad(log_like_fn)
    grad_auto = np.array(grad_fn(r_jax))

    # Finite difference gradient (for a subset of parameters)
    n_check = min(5, N)
    eps = 1e-5
    grad_fd = np.zeros(N)
    for i in range(n_check):
        r_plus = np.array(r_jax).copy()
        r_minus = np.array(r_jax).copy()
        r_plus[i] += eps
        r_minus[i] -= eps
        f_plus = float(log_like_fn(jnp.array(r_plus, dtype=jnp.float64)))
        f_minus = float(log_like_fn(jnp.array(r_minus, dtype=jnp.float64)))
        grad_fd[i] = (f_plus - f_minus) / (2 * eps)

    print(f"  Checking {n_check} gradient components:")
    max_rel_err = 0
    for i in range(n_check):
        denom = max(np.abs(grad_fd[i]), np.abs(grad_auto[i]), 1e-10)
        rel = np.abs(grad_auto[i] - grad_fd[i]) / denom
        max_rel_err = max(max_rel_err, rel)
        status = "OK" if rel < 0.05 else "FAIL"
        print(f"    r[{i}]: autodiff={grad_auto[i]:.6e}, "
              f"fd={grad_fd[i]:.6e}, rel_err={rel:.2e} [{status}]")

    passed = max_rel_err < 0.05
    print(f"  {'PASSED' if passed else 'FAILED'} (max rel err = {max_rel_err:.2e})\n")

    return passed


def test_log_likelihood(r, RA, dec, s_grid, psi_par, psi_perp, sigma_v_sq_3):
    """Test log-likelihood against direct NumPy computation."""
    print("=" * 60)
    print("TEST 6: Log-likelihood correctness")
    print("=" * 60)

    import jax.numpy as jnp

    rhat = radec_to_cartesian(RA, dec)
    N = len(r)

    C_np = assemble_pv_covariance_numpy(
        r, rhat, s_grid, psi_par, psi_perp, sigma_v_sq_3)

    rng = np.random.default_rng(99)
    v_obs = rng.normal(0, 200, N)
    v_pred = rng.normal(0, 100, N)
    beta = 0.5
    sigma_v = 200.0

    # NumPy reference
    C_total_np = beta**2 * C_np + sigma_v**2 * np.eye(N)
    L_np = np.linalg.cholesky(C_total_np)
    residual = v_obs - v_pred
    alpha = np.linalg.solve(L_np, residual)
    log_det = 2 * np.sum(np.log(np.diag(L_np)))
    ll_ref = -0.5 * (N * np.log(2 * np.pi) + log_det + np.dot(alpha, alpha))

    # JAX version
    r_jax = jnp.array(r)
    rhat_jax = jnp.array(rhat)
    C_jax = assemble_pv_covariance_jax(
        r_jax, rhat_jax, jnp.array(s_grid), jnp.array(psi_par),
        jnp.array(psi_perp), sigma_v_sq_3)
    ll_jax = float(pv_covariance_log_likelihood(
        jnp.array(v_obs), jnp.array(v_pred), C_jax, beta, sigma_v))

    rel_err = np.abs(ll_ref - ll_jax) / np.abs(ll_ref)
    print(f"  NumPy log-likelihood: {ll_ref:.4f}")
    print(f"  JAX log-likelihood:   {ll_jax:.4f}")
    print(f"  Relative error: {rel_err:.2e}")

    passed = rel_err < 1e-4
    print(f"  {'PASSED' if passed else 'FAILED'}\n")
    return passed


def test_timing_scaling(k, Pk, dDdtau, s_grid, psi_par, psi_perp,
                        sigma_v_sq_3, sizes=[20, 50, 100, 200, 500]):
    """Benchmark assembly + Cholesky for different N."""
    print("=" * 60)
    print("TEST 7: Timing scaling")
    print("=" * 60)

    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(77)
    s_jax = jnp.array(s_grid)
    pp_jax = jnp.array(psi_par)
    pperp_jax = jnp.array(psi_perp)

    print(f"  {'N':>6} {'assembly':>12} {'cholesky':>12} {'total':>12}")
    print(f"  {'-'*48}")

    for N in sizes:
        RA = rng.uniform(0, 360, N)
        dec = rng.uniform(-90, 90, N)
        r = rng.uniform(20, 200, N)
        rhat = radec_to_cartesian(RA, dec)

        r_jax = jnp.array(r)
        rhat_jax = jnp.array(rhat)

        @jax.jit
        def step(r_):
            C = assemble_pv_covariance_jax(
                r_, rhat_jax, s_jax, pp_jax, pperp_jax, sigma_v_sq_3)
            C_total = 0.16 * C + 150**2 * jnp.eye(N)
            L = jnp.linalg.cholesky(C_total)
            return C, L

        # JIT
        C, L = step(r_jax)
        L.block_until_ready()

        # Warmup
        for _ in range(3):
            C, L = step(r_jax)
            L.block_until_ready()

        # Time
        n_rep = 20
        t0 = time.perf_counter()
        for _ in range(n_rep):
            C, L = step(r_jax)
            L.block_until_ready()
        dt_total = (time.perf_counter() - t0) / n_rep * 1000

        # Assembly only
        @jax.jit
        def asm_only(r_):
            return assemble_pv_covariance_jax(
                r_, rhat_jax, s_jax, pp_jax, pperp_jax, sigma_v_sq_3)

        C = asm_only(r_jax)
        C.block_until_ready()
        for _ in range(3):
            C = asm_only(r_jax)
            C.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(n_rep):
            C = asm_only(r_jax)
            C.block_until_ready()
        dt_asm = (time.perf_counter() - t0) / n_rep * 1000
        dt_chol = dt_total - dt_asm

        print(f"  {N:>6} {dt_asm:>10.3f} ms {dt_chol:>10.3f} ms "
              f"{dt_total:>10.3f} ms")

    print("  DONE\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gal", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--kmax", type=float, default=20.0)
    parser.add_argument("--skip_old", action="store_true",
                        help="Skip comparison with old Bessel-sum code")
    parser.add_argument("--skip_scaling", action="store_true",
                        help="Skip scaling benchmark")
    args = parser.parse_args()

    if HAS_CAMB:
        print("Loading P(k) from CAMB...")
        k, Pk = get_Pk_CAMB(kmax=args.kmax)
    else:
        print("CAMB not available, using synthetic Eisenstein-Hu-like P(k)...")
        k = np.logspace(-4, np.log10(args.kmax), 5000)
        # Simple power-law P(k) with suppression at high k
        Pk = 2e4 * (k / 0.05)**0.96 / (1 + (k / 0.05)**1.5)**2.0
    dDdtau = compute_dD_dtau()
    print(f"  k range: [{k.min():.4f}, {k.max():.2f}] h/Mpc, "
          f"n_k = {len(k)}")
    print(f"  dD/dτ = {dDdtau:.4f}\n")

    r, RA, dec = generate_test_galaxies(args.n_gal)

    # Test 1: ψ functions
    s_grid, psi_par, psi_perp, sv3 = test_psi_functions(k, Pk, dDdtau)

    # Test 2: NumPy tensor vs old Bessel-sum
    if not args.skip_old:
        C_old, C_new, passed2 = test_numpy_vs_old(
            r, RA, dec, k, Pk, dDdtau, s_grid, psi_par, psi_perp, sv3,
            n_jobs=args.n_jobs)
    else:
        C_new = assemble_pv_covariance_numpy(
            r, radec_to_cartesian(RA, dec), s_grid, psi_par, psi_perp, sv3)

    # Test 3: JAX vs NumPy
    C_jax, passed3 = test_jax_vs_numpy(
        r, RA, dec, s_grid, psi_par, psi_perp, sv3, C_new)

    # Test 4: JIT
    test_jit(r, RA, dec, s_grid, psi_par, psi_perp, sv3)

    # Test 5: Gradients
    test_gradients(r[:10], RA[:10], dec[:10], s_grid, psi_par, psi_perp, sv3)

    # Test 6: Log-likelihood
    test_log_likelihood(r, RA, dec, s_grid, psi_par, psi_perp, sv3)

    # Test 7: Scaling
    if not args.skip_scaling:
        test_timing_scaling(k, Pk, dDdtau, s_grid, psi_par, psi_perp, sv3)


if __name__ == "__main__":
    main()
