"""
Benchmark NumPyro chain_method="vectorized" vs "sequential" on GPU.

Synthetic TRGB-like model with representative array sizes:
  n_fields=30, n_gal=100, n_r=301

Usage:
    python benchmark_chains_gpu.py [--num_samples N] [--output FILE]
"""
import argparse
import json
import os
import time
import traceback

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


class PrecomputedLogLike(dist.Distribution):
    """Wraps a precomputed per-element log-likelihood as a distribution.

    obs=jnp.zeros(n) gives a 1-d tensor, which is safe under vmap
    (chain_method="vectorized"), unlike numpyro.factor which uses
    obs=jnp.zeros(()) and fails with vmap.
    """
    support = dist.constraints.real

    def __init__(self, log_like):
        self._log_like = log_like
        super().__init__(batch_shape=log_like.shape)

    def sample(self, key, sample_shape=()):
        return jnp.zeros(sample_shape + self.batch_shape)

    def log_prob(self, value):
        return self._log_like


def make_synthetic_data(n_fields=30, n_gal=100, n_r=301, seed=0):
    rng = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    return {
        "los_density": jax.random.normal(k1, (n_fields, n_gal, n_r)) * 0.1 + 1.0,
        "los_velocity": jax.random.normal(k2, (n_fields, n_gal, n_r)) * 100.0,
        "mu_obs": jax.random.normal(k3, (n_gal,)) * 0.2 + 32.0,
        "sigma_obs": jnp.full((n_gal,), 0.08),
        "r": jnp.linspace(0.01, 300.0, n_r),
    }


# ---------- Model variants for bisecting the vmap issue ----------

def trivial_model(data):
    """Just a Normal. Should always work with vectorized chains."""
    x = numpyro.sample("x", dist.Normal(0, 1))
    numpyro.sample("obs", dist.Normal(x, 1), obs=data["mu_obs"][0])


def multi_param_model(data):
    """Multiple parameters including a vector."""
    H0 = numpyro.sample("H0", dist.Uniform(60.0, 80.0))
    beta = numpyro.sample("beta", dist.Uniform(0.1, 1.0))
    Vext = numpyro.sample("Vext", dist.Normal(jnp.zeros(3), 200.0))
    y = H0 + beta + Vext.sum()
    numpyro.sample("obs", dist.Normal(y, 1), obs=data["mu_obs"][0])


def factor_model(data):
    """Per-galaxy likelihood with numpyro.factor (scalar)."""
    H0 = numpyro.sample("H0", dist.Uniform(60.0, 80.0))
    mu_pred = jnp.full_like(data["mu_obs"], 5.0 * jnp.log10(100.0 * H0 / 2.998e5) + 25.0)
    log_like = dist.Normal(mu_pred, 0.1).log_prob(data["mu_obs"]).sum()
    numpyro.factor("ll", log_like)


def array_likelihood_model(data):
    """Per-galaxy likelihood with PrecomputedLogLike."""
    H0 = numpyro.sample("H0", dist.Uniform(60.0, 80.0))
    mu_pred = jnp.full_like(data["mu_obs"], 5.0 * jnp.log10(100.0 * H0 / 2.998e5) + 25.0)
    log_like = dist.Normal(mu_pred, 0.1).log_prob(data["mu_obs"])
    numpyro.sample("ll", PrecomputedLogLike(log_like), obs=jnp.zeros(len(data["mu_obs"])))


def trgb_like_model(data):
    """Full simplified TRGB-like model."""
    n_fields, n_gal, n_r = data["los_density"].shape

    H0 = numpyro.sample("H0", dist.Uniform(60.0, 80.0))
    beta = numpyro.sample("beta", dist.Uniform(0.1, 1.0))
    sigma_v = numpyro.sample("sigma_v", dist.Uniform(50.0, 500.0))
    Vext = numpyro.sample("Vext", dist.Normal(jnp.zeros(3), 200.0 * jnp.ones(3)))

    c = 2.998e5
    r = data["r"]
    mu_r = 5.0 * jnp.log10(r * H0 / c) + 25.0

    mean_dens = data["los_density"].mean(axis=0)
    mean_vel = data["los_velocity"].mean(axis=0)
    vext_proj = Vext[0]
    vpec = beta * mean_vel + vext_proj
    delta_mu = -5.0 / jnp.log(10.0) * vpec / (c * r)

    mu_pred = mu_r[None, :] + delta_mu
    log_dens = jnp.log(jnp.clip(mean_dens, 1e-6))

    sigma_tot = jnp.sqrt(data["sigma_obs"][:, None]**2
                         + (5.0 / jnp.log(10.0) * sigma_v / (c * r))**2)
    log_like_r = (dist.Normal(mu_pred, sigma_tot)
                  .log_prob(data["mu_obs"][:, None])
                  + log_dens)
    log_like = jax.scipy.special.logsumexp(log_like_r, axis=-1)

    numpyro.factor("ll", log_like.sum())


# ---------- Benchmark runner ----------

def run_benchmark(model_fn, data, chain_method, num_chains,
                  num_warmup, num_samples):
    def _make_mcmc():
        return MCMC(
            NUTS(model_fn),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=False,
        )

    rng_key = jax.random.PRNGKey(42)

    # First run (includes JIT compilation + warmup)
    mcmc = _make_mcmc()
    t0 = time.perf_counter()
    mcmc.run(rng_key, data)
    t_total = time.perf_counter() - t0

    # Second run with fresh MCMC object: measures JIT-cached performance.
    # Re-creating avoids the double-vmap bug in NumPyro where calling
    # mcmc.run twice wraps _sample_fn in a second vmap layer.
    mcmc = _make_mcmc()
    t0 = time.perf_counter()
    mcmc.run(rng_key, data)
    t_sample = time.perf_counter() - t0

    n = num_chains * num_samples
    return {
        "chain_method": chain_method,
        "num_chains": num_chains,
        "t_total_s": t_total,
        "t_sample_s": t_sample,
        "ms_per_sample": t_sample / n * 1e3,
        "samples_per_s": n / t_sample,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_warmup", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON file to write results to")
    args = parser.parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"NumPyro version: {numpyro.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"num_warmup={args.num_warmup}, num_samples={args.num_samples}\n")

    data = make_synthetic_data()
    print(f"Data shapes: los_density={data['los_density'].shape}, "
          f"mu_obs={data['mu_obs'].shape}\n")

    # ---- Phase 1: Diagnose which model complexity breaks vectorized ----
    print("=" * 60)
    print("PHASE 1: Diagnosing vectorized chain support")
    print("=" * 60)
    diagnostic_models = [
        ("trivial", trivial_model),
        ("multi_param", multi_param_model),
        ("factor", factor_model),
        ("array_likelihood", array_likelihood_model),
        ("trgb_like", trgb_like_model),
    ]
    for name, model_fn in diagnostic_models:
        print(f"\n  Testing {name} model (vectorized x4)...", flush=True)
        try:
            kernel = NUTS(model_fn)
            mcmc = MCMC(kernel, num_warmup=10, num_samples=10,
                        num_chains=4, chain_method="vectorized",
                        progress_bar=False)
            mcmc.run(jax.random.PRNGKey(0), data)
            print(f"    {name}: OK")
        except Exception as e:
            print(f"    {name}: FAILED — {e}")
            traceback.print_exc()
            break  # no point continuing

    # ---- Phase 2: Full benchmark ----
    print("\n" + "=" * 60)
    print("PHASE 2: Full benchmark")
    print("=" * 60)

    configs = [
        ("sequential", 1),
        ("sequential", 4),
        ("vectorized", 4),
        ("vectorized", 8),
        ("vectorized", 16),
    ]

    results = []
    for chain_method, num_chains in configs:
        tag = f"{chain_method:>12s}  x{num_chains:2d} chains"
        print(f"\nRunning {tag} ...", flush=True)
        try:
            r = run_benchmark(trgb_like_model, data, chain_method, num_chains,
                              args.num_warmup, args.num_samples)
            print(f"  {tag}:  {r['ms_per_sample']:.2f} ms/sample  "
                  f"({r['samples_per_s']:.1f} samples/s)  "
                  f"[JIT+warmup {r['t_total_s']:.1f}s, "
                  f"pure {r['t_sample_s']:.1f}s]")
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print("\n--- Summary (sorted by ms/sample) ---")
    results.sort(key=lambda x: x["ms_per_sample"])
    for r in results:
        print(f"  {r['chain_method']:>12s} x{r['num_chains']:2d}: "
              f"{r['ms_per_sample']:.2f} ms/sample")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
