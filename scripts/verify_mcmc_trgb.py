"""Run a short MCMC chain to verify the optimized TRGB model works
correctly end-to-end with NUTS sampling."""
import sys
sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import set_platform
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

import candel

CONFIG = "scripts/runs/config_EDD_TRGB.toml"

# Fixed init point for reproducibility
INIT_PARAMS = {
    "H0": jnp.float32(73.0),
    "M_TRGB": jnp.float32(-4.05),
    "sigma_int": jnp.float32(0.12),
    "sigma_v": jnp.float32(250.0),
    "beta": jnp.float32(0.43),
    "b1": jnp.float32(1.2),
    "Vext_phi": jnp.float32(4.5),
    "Vext_cos_theta": jnp.float32(0.3),
    "Vext_mag": jnp.float32(150.0),
    "mu_LMC": jnp.float32(18.48),
    "mu_N4258": jnp.float32(29.40),
    "mag_lim_TRGB": jnp.float32(25.0),
    "mag_lim_TRGB_width": jnp.float32(0.75),
}

NUM_WARMUP = 50
NUM_SAMPLES = 100


if __name__ == "__main__":
    set_platform("cpu")
    print("Loading model...")
    data = candel.pvdata.load_EDD_TRGB_from_config(CONFIG)
    model = candel.model.TRGBModel(CONFIG, data)

    print(f"\nRunning NUTS: {NUM_WARMUP} warmup + {NUM_SAMPLES} samples")
    kernel = NUTS(model, init_strategy=init_to_value(values=INIT_PARAMS))
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=1, chain_method="sequential",
                progress_bar=True)
    mcmc.run(jax.random.key(42))

    samples = mcmc.get_samples()

    print("\n=== Posterior summary ===")
    show_keys = ["H0", "M_TRGB", "sigma_int", "sigma_v", "beta", "b1"]
    for k in show_keys:
        if k in samples:
            v = np.asarray(samples[k])
            print(f"  {k:20s}: mean={v.mean():.4f}  std={v.std():.4f}  "
                  f"[{v.min():.4f}, {v.max():.4f}]")

    # Check no NaN/Inf in samples
    any_bad = False
    for k, v in samples.items():
        arr = np.asarray(v)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"  WARNING: {k} contains NaN/Inf!")
            any_bad = True

    if any_bad:
        print("\nFAILED: Some samples contain NaN/Inf!")
        sys.exit(1)

    # Check that H0 is in a reasonable range
    h0 = np.asarray(samples["H0"])
    if h0.mean() < 50 or h0.mean() > 100:
        print(f"\nWARNING: H0 mean = {h0.mean():.2f} is outside "
              "reasonable range [50, 100]")

    # Check acceptance rate from diagnostics
    extra_fields = mcmc.get_extra_fields()
    if "accept_prob" in extra_fields:
        acc = float(np.mean(np.asarray(extra_fields["accept_prob"])))
        print(f"\n  Mean acceptance probability: {acc:.3f}")
        if acc < 0.4:
            print("  WARNING: Low acceptance rate")

    n_div = int(np.sum(np.asarray(extra_fields.get("diverging", [0]))))
    print(f"  Divergences: {n_div}")

    print("\nMCMC verification PASSED.")
