"""Test NSS nested sampling on a 20D correlated Gaussian.

Verifies evidence recovery and posterior mean/std in 20 dimensions.
Analytic log Z = 20/2 * log(2*pi) + 0.5*log(det(Sigma)) - 20*log(20)
(uniform prior on [-10, 10]^20).
"""
import tqdm as _tqdm
_OrigTqdm = _tqdm.tqdm


class _SlowTqdm(_OrigTqdm):
    def __init__(self, *a, **kw):
        kw.setdefault("mininterval", 5)
        super().__init__(*a, **kw)


_tqdm.tqdm = _SlowTqdm

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from candel.inference.nested import run_nss  # noqa: E402


def print_header():
    print("=" * 60)
    print("TEST: 20D correlated Gaussian — NSS nested sampling")
    print("=" * 60)
    devices = jax.devices()
    print(f"JAX backend:  {jax.default_backend()}")
    print(f"JAX devices:  {devices}")
    for d in devices:
        print(f"  {d.device_kind}: {d}")
    print(f"JAX version:  {jax.__version__}")
    print(f"NumPyro ver:  {numpyro.__version__}")
    print()


def make_precision_and_logZ(ndim, seed=0):
    """Build a random positive-definite precision matrix and analytic log Z."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((ndim, ndim)) * 0.3
    Sigma = A @ A.T + np.eye(ndim)
    Sigma_inv = np.linalg.inv(Sigma)
    sign, logdet_Sigma = np.linalg.slogdet(Sigma)

    prior_width = 20.0  # [-10, 10]
    log_Z = (0.5 * ndim * np.log(2 * np.pi)
             + 0.5 * logdet_Sigma
             - ndim * np.log(prior_width))
    return Sigma, Sigma_inv, logdet_Sigma, log_Z


def build_model(ndim, mu, Sigma_inv):
    """NumPyro model: 20D Gaussian likelihood with uniform priors."""
    mu_jax = jnp.array(mu)
    Sinv_jax = jnp.array(Sigma_inv)

    def model():
        xs = []
        for i in range(ndim):
            xs.append(numpyro.sample(f"x{i}", dist.Uniform(-10, 10)))
        x = jnp.stack(xs)
        diff = x - mu_jax
        ll = -0.5 * diff @ Sinv_jax @ diff
        numpyro.factor("ll", ll)

    return model


def main():
    print_header()

    ndim = 20
    n_live = 5000
    num_mcmc_steps = ndim  # p = d (Yallup+2025)
    num_delete = 500
    termination = -3

    mu = np.zeros(ndim)
    Sigma, Sigma_inv, logdet_Sigma, log_Z_true = make_precision_and_logZ(
        ndim, seed=0)

    print("Likelihood:   N(0, Sigma) with random Sigma")
    print(f"Prior:        Uniform(-10, 10)^{ndim}")
    print(f"ndim:         {ndim}")
    print(f"Sigma cond:   {np.linalg.cond(Sigma):.1f}")
    print(f"Analytic logZ: {log_Z_true:.4f}")
    print()
    print("NSS settings:")
    print(f"  n_live:          {n_live}")
    print(f"  num_mcmc_steps:  {num_mcmc_steps}")
    print(f"  num_delete:      {num_delete}")
    print(f"  termination:     {termination}  "
          f"(stop when log(Z_live/Z_dead) < {termination})")
    print()

    model = build_model(ndim, mu, Sigma_inv)

    samples = run_nss(
        model,
        n_live=n_live,
        num_mcmc_steps=num_mcmc_steps,
        num_delete=num_delete,
        termination=termination,
        seed=42,
    )
    meta = samples.pop("__nested__")

    log_Z_est = meta["log_Z"]
    log_Z_err = meta["log_Z_err"]
    diff = log_Z_est - log_Z_true
    print(f"\nEstimated log Z = {log_Z_est:.4f} +/- {log_Z_err:.4f}")
    print(f"True      log Z = {log_Z_true:.4f}")
    print(f"Difference      = {diff:.4f} ({abs(diff)/log_Z_err:.1f} sigma)")
    print(f"n_eff           = {meta['n_eff']}")

    # Check posterior means and stds
    stds_true = np.sqrt(np.diag(Sigma))
    print("\nPosterior summary (truth: mean=0, std=diag(Sigma)^0.5):")
    max_bias = 0.0
    for i in range(ndim):
        s = samples[f"x{i}"]
        m, sd = float(s.mean()), float(s.std())
        bias_sigma = abs(m - mu[i]) / stds_true[i]
        max_bias = max(max_bias, bias_sigma)
        print(f"  x{i:02d}: mean={m:+.3f} +/- {sd:.3f}  "
              f"(true std={stds_true[i]:.3f}, bias={bias_sigma:.1f}sigma)")

    # Verdict
    print(f"\nMax posterior mean bias: {max_bias:.1f}sigma")
    evidence_ok = abs(diff) < 3 * log_Z_err + 1.0
    posterior_ok = max_bias < 3.0
    print(f"Evidence recovery: {'PASS' if evidence_ok else 'FAIL'}")
    print(f"Posterior means:   {'PASS' if posterior_ok else 'FAIL'}")

    if not (evidence_ok and posterior_ok):
        raise SystemExit("FAILED: nested sampler validation")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
