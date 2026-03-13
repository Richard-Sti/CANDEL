"""Test sigmoid monopole implementation: forward pass and gradient."""
import sys
sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import tomli_w
from numpyro import handlers
from numpyro.infer.util import log_density

from candel.model.pv_utils import sigmoid_monopole_radial
from candel.pvdata.data import load_EDD_TRGB_from_config
from candel.model.model_H0_TRGB import TRGBModel
from candel.util import load_config

CONFIG_PATH = "scripts/runs/config_EDD_TRGB.toml"


def make_tmp_config(overrides):
    config = load_config(CONFIG_PATH, replace_los_prior=False)
    for k, v in overrides.items():
        keys = k.split("/")
        d = config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    with tempfile.NamedTemporaryFile(
            mode='wb', suffix='.toml', dir='scripts/runs',
            delete=False) as f:
        tomli_w.dump(config, f)
        return f.name


def test_model(label, overrides, expect_params):
    print(f"\n{'=' * 60}")
    print(f"Test: {label}")
    print(f"{'=' * 60}")

    tmp = make_tmp_config(overrides)
    try:
        data = load_EDD_TRGB_from_config(tmp)
        model = TRGBModel(tmp, data)

        rng_key = jax.random.PRNGKey(42)
        with handlers.seed(rng_seed=rng_key):
            trace = handlers.trace(model).get_trace()

        params = {k: v["value"] for k, v in trace.items()
                  if v["type"] == "sample"
                  and not v.get("is_observed", False)
                  and "_ll" not in k}

        log_d, _ = log_density(model, (), {}, params)
        print(f"  log_density = {log_d:.4f}")
        assert jnp.isfinite(log_d), f"log_density not finite: {log_d}"
        print("  Forward pass: PASSED")

        found = [k for k in params if "Vext_mono" in k]
        print(f"  Monopole params found: {found}")
        for p in expect_params:
            assert p in found, f"Expected param '{p}' not found"
        for p in found:
            print(f"    {p} = {float(params[p]):.4f}")

        # Gradient
        def loss_fn(p):
            ld, _ = log_density(model, (), {}, p)
            return ld

        grads = jax.grad(loss_fn)(params)
        for p in expect_params:
            g = float(grads[p])
            print(f"    d(logp)/d({p}) = {g:.6f}")
            assert jnp.isfinite(grads[p]), f"Gradient not finite for {p}"
        print("  Gradient: PASSED")
    finally:
        os.unlink(tmp)


# Test 0: sigmoid_monopole_radial function
print("=" * 60)
print("Test 0: sigmoid_monopole_radial basic behavior")
print("=" * 60)

r = jnp.linspace(0.1, 50, 100)
V_left, r_t, k = 100.0, 20.0, 1.0 / 3.0

val_small_r = sigmoid_monopole_radial(V_left, r_t, k, jnp.array(0.1))
assert abs(val_small_r - V_left) < 1.0
print(f"  V(r=0.1) = {val_small_r:.3f} (expected ~{V_left})")

val_at_rt = sigmoid_monopole_radial(V_left, r_t, k, jnp.array(r_t))
assert abs(val_at_rt - V_left / 2) < 0.01
print(f"  V(r={r_t}) = {val_at_rt:.3f} (expected {V_left/2})")

val_large_r = sigmoid_monopole_radial(V_left, r_t, k, jnp.array(100.0))
assert abs(val_large_r) < 1.0
print(f"  V(r=100) = {val_large_r:.3f} (expected ~0)")

# Verify angle parameterization: k = tan(angle)
angle = 0.8  # ~46 degrees
k_from_angle = jnp.tan(angle)
V_angle = sigmoid_monopole_radial(V_left, r_t, k_from_angle, r)
print(f"  angle={angle:.2f} rad -> k={k_from_angle:.4f}")
print(f"  V(r=5) = {sigmoid_monopole_radial(V_left, r_t, k_from_angle, jnp.array(5.)):.3f}")
print("  PASSED")

# Test 1: no monopole
test_model(
    "no monopole",
    {"model/which_Vext_monopole": "none"},
    expect_params=[],
)

# Test 2: constant monopole
test_model(
    "constant monopole",
    {"model/which_Vext_monopole": "constant"},
    expect_params=["Vext_mono"],
)

# Test 3: sigmoid monopole
test_model(
    "sigmoid monopole",
    {"model/which_Vext_monopole": "sigmoid"},
    expect_params=["Vext_mono_left", "Vext_mono_rt", "Vext_mono_angle"],
)

# Test 4: backward compat — legacy bool
test_model(
    "legacy use_Vext_monopole=True -> constant",
    {"model/use_Vext_monopole": True},
    expect_params=["Vext_mono"],
)

print(f"\n{'=' * 60}")
print("ALL TESTS PASSED")
print("=" * 60)
