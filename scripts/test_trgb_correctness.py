"""Correctness tests for TRGB model: verify log-density and gradients
at reference parameter points remain unchanged after optimizations."""
import sys
sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density

import candel

CONFIG = "scripts/runs/config_EDD_TRGB.toml"
REF_FILE = "scripts/trgb_reference_values.npy"

PARAM_POINTS = [
    {
        "H0": 73.0, "M_TRGB": -4.05, "sigma_int": 0.12,
        "sigma_v": 250.0, "beta": 0.43, "b1": 1.2,
        "Vext_phi": 4.5, "Vext_cos_theta": 0.3, "Vext_mag": 150.0,
        "mu_LMC": 18.48, "mu_N4258": 29.40,
        "mag_lim_TRGB": 25.0, "mag_lim_TRGB_width": 0.75,
        "cz_lim_selection": 2500.0, "cz_lim_selection_width": 200.0,
    },
    {
        "H0": 68.0, "M_TRGB": -4.10, "sigma_int": 0.08,
        "sigma_v": 200.0, "beta": 0.45, "b1": 1.5,
        "Vext_phi": 3.0, "Vext_cos_theta": -0.2, "Vext_mag": 200.0,
        "mu_LMC": 18.50, "mu_N4258": 29.35,
        "mag_lim_TRGB": 24.5, "mag_lim_TRGB_width": 0.50,
        "cz_lim_selection": 3000.0, "cz_lim_selection_width": 150.0,
    },
    {
        "H0": 78.0, "M_TRGB": -3.95, "sigma_int": 0.15,
        "sigma_v": 300.0, "beta": 0.40, "b1": 1.0,
        "Vext_phi": 1.0, "Vext_cos_theta": 0.8, "Vext_mag": 100.0,
        "mu_LMC": 18.45, "mu_N4258": 29.45,
        "mag_lim_TRGB": 25.5, "mag_lim_TRGB_width": 1.0,
        "cz_lim_selection": 2000.0, "cz_lim_selection_width": 250.0,
    },
]

# Tolerances: float32 has ~7 decimal digits of precision
LD_ATOL = 0.05       # absolute tolerance on log-density
GRAD_RTOL = 1e-4     # relative tolerance on gradients


def make_params(params):
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in params.items()}


def evaluate_log_density(model, params):
    p = make_params(params)
    ld, _ = log_density(model, (), {}, p)
    return float(ld)


def evaluate_gradient(model, params):
    p = make_params(params)

    def f(p):
        return log_density(model, (), {}, p)[0]

    grads = jax.grad(f)(p)
    return {k: float(v) for k, v in grads.items()}


def test_log_density(model, ref_data):
    print("=== Testing log-density values ===")
    all_pass = True
    for i, params in enumerate(PARAM_POINTS):
        ld = evaluate_log_density(model, params)
        ref_ld = ref_data[f"point_{i}"]["log_density"]
        diff = abs(ld - ref_ld)
        ok = diff < LD_ATOL
        status = "PASS" if ok else "FAIL"
        print(f"  Point {i}: {ld:.6f} (ref {ref_ld:.6f}, "
              f"diff {diff:.2e}) [{status}]")
        if not ok:
            all_pass = False
    return all_pass


def test_gradients(model, ref_data):
    print("\n=== Testing gradient values ===")
    all_pass = True
    check_keys = ["H0", "M_TRGB", "sigma_int", "beta", "b1"]
    for i, params in enumerate(PARAM_POINTS):
        grads = evaluate_gradient(model, params)
        ref_grads = ref_data[f"point_{i}"]["grads"]
        print(f"  Point {i}:")
        for k in check_keys:
            g = grads.get(k)
            rg = ref_grads.get(k)
            if g is None or rg is None:
                continue
            if abs(rg) > 1e-6:
                rdiff = abs(g - rg) / abs(rg)
                ok = rdiff < GRAD_RTOL
                print(f"    d/d{k}: {g:.6f} (ref {rg:.6f}, "
                      f"rdiff {rdiff:.2e}) [{('PASS' if ok else 'FAIL')}]")
            else:
                adiff = abs(g - rg)
                ok = adiff < 1e-3
                print(f"    d/d{k}: {g:.6f} (ref {rg:.6f}, "
                      f"adiff {adiff:.2e}) [{('PASS' if ok else 'FAIL')}]")
            if not ok:
                all_pass = False
    return all_pass


def test_simpson_consistency():
    """Verify ln_simpson_precomputed matches ln_simpson on a test case."""
    from candel.model.simpson import (ln_simpson, ln_simpson_precomputed,
                                      simpson_log_weights)
    print("\n=== Testing Simpson consistency ===")

    rng = np.random.default_rng(42)
    x = jnp.linspace(0.01, 150, 301)
    log_w = simpson_log_weights(x)

    # 2D test case
    ln_y = jnp.asarray(rng.standard_normal((50, 301)))
    result_old = ln_simpson(ln_y, x[None, :], axis=-1)
    result_new = ln_simpson_precomputed(ln_y, log_w, axis=-1)

    diff = float(jnp.max(jnp.abs(result_old - result_new)))
    ok = diff < 1e-5
    print(f"  Max abs diff (2D): {diff:.2e} [{'PASS' if ok else 'FAIL'}]")

    # 3D test case
    ln_y_3d = jnp.asarray(rng.standard_normal((3, 50, 301)))
    result_old_3d = ln_simpson(ln_y_3d, x[None, None, :], axis=-1)
    result_new_3d = ln_simpson_precomputed(ln_y_3d, log_w, axis=-1)

    diff_3d = float(jnp.max(jnp.abs(result_old_3d - result_new_3d)))
    ok_3d = diff_3d < 1e-5
    print(f"  Max abs diff (3D): {diff_3d:.2e} [{'PASS' if ok_3d else 'FAIL'}]")

    return ok and ok_3d


if __name__ == "__main__":
    print("Loading model and reference data...")
    data = candel.pvdata.load_EDD_TRGB_from_config(CONFIG)
    model = candel.model.TRGBModel(CONFIG, data)
    ref_data = np.load(REF_FILE, allow_pickle=True).item()
    print(f"  {model.num_hosts} hosts, {model.num_fields} fields\n")

    results = []
    results.append(("Simpson consistency", test_simpson_consistency()))
    results.append(("Log-density", test_log_density(model, ref_data)))
    results.append(("Gradients", test_gradients(model, ref_data)))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        all_pass &= ok

    if all_pass:
        print("\nAll tests PASSED.")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
