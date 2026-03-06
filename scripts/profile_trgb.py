"""Profile and capture reference values for the TRGB model forward pass."""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density

sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")
import candel  # noqa

CONFIG = "scripts/runs/config_EDD_TRGB.toml"

# Reference parameter points (3 distinct points for regression testing)
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


def make_unconstrained(params):
    """Convert to dict of jnp arrays for log_density."""
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in params.items()}


def evaluate_log_density(model, params):
    """Evaluate log density at given parameter point."""
    p = make_unconstrained(params)
    ld, _ = log_density(model, (), {}, p)
    return ld


def evaluate_gradient(model, params):
    """Evaluate gradient of log density at given parameter point."""
    p = make_unconstrained(params)

    def f(p):
        return log_density(model, (), {}, p)[0]

    grad_fn = jax.grad(f)
    grads = grad_fn(p)
    return grads


def profile_forward_pass(model, params, n_reps=50):
    """Time the forward pass (log density evaluation)."""
    p = make_unconstrained(params)

    @jax.jit
    def f(p):
        return log_density(model, (), {}, p)[0]

    # Warmup (JIT compilation)
    _ = f(p).block_until_ready()

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = f(p).block_until_ready()
        times.append(time.perf_counter() - t0)

    return np.array(times)


def profile_gradient(model, params, n_reps=50):
    """Time the gradient computation."""
    p = make_unconstrained(params)

    @jax.jit
    def g(p):
        return jax.grad(lambda p: log_density(model, (), {}, p)[0])(p)

    # Warmup
    _ = jax.tree.map(lambda x: x.block_until_ready(), g(p))

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = jax.tree.map(lambda x: x.block_until_ready(), g(p))
        times.append(time.perf_counter() - t0)

    return np.array(times)


if __name__ == "__main__":
    print("Loading data and model...")
    data = candel.pvdata.load_EDD_TRGB_from_config(CONFIG)
    model = candel.model.TRGBModel(CONFIG, data)
    print(f"  Loaded {model.num_hosts} hosts, "
          f"{model.num_fields} fields, "
          f"{model.num_rand_los} random LOS")

    # Evaluate log densities at reference points
    print("\n=== Reference log densities ===")
    for i, params in enumerate(PARAM_POINTS):
        ld = evaluate_log_density(model, params)
        print(f"  Point {i}: log_density = {float(ld):.8f}")

    # Evaluate gradients at reference points
    print("\n=== Reference gradients (selected params) ===")
    show_keys = ["H0", "M_TRGB", "sigma_int", "beta", "b1"]
    for i, params in enumerate(PARAM_POINTS):
        grads = evaluate_gradient(model, params)
        print(f"  Point {i}:")
        for k in show_keys:
            if k in grads:
                print(f"    d/d{k} = {float(grads[k]):.8f}")

    # Profile forward pass
    print("\n=== Profiling forward pass ===")
    fwd_times = profile_forward_pass(model, PARAM_POINTS[0])
    print(f"  Forward pass: {fwd_times.mean()*1000:.3f} +/- "
          f"{fwd_times.std()*1000:.3f} ms (n={len(fwd_times)})")

    # Profile gradient
    print("\n=== Profiling gradient ===")
    grad_times = profile_gradient(model, PARAM_POINTS[0])
    print(f"  Gradient:     {grad_times.mean()*1000:.3f} +/- "
          f"{grad_times.std()*1000:.3f} ms (n={len(grad_times)})")

    # Save reference values for comparison
    ref_data = {}
    for i, params in enumerate(PARAM_POINTS):
        ld = float(evaluate_log_density(model, params))
        grads = evaluate_gradient(model, params)
        ref_data[f"point_{i}"] = {
            "log_density": ld,
            "grads": {k: float(grads[k]) for k in grads},
        }
    np.save("scripts/trgb_reference_values.npy", ref_data)
    print("\nSaved reference values to scripts/trgb_reference_values.npy")
