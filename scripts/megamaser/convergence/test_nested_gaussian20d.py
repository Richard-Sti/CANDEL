"""Test NSS nested sampling on a 20D correlated Gaussian.

Verifies evidence recovery and posterior mean/std in 20 dimensions.
Analytic log Z = 20/2 * log(2*pi) + 0.5*log(det(Sigma)) - 20*log(20)
(uniform prior on [-10, 10]^20).
"""
from candel.util import patch_tqdm
patch_tqdm()

import argparse  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from candel.inference.nested import run_nss  # noqa: E402

DEFAULT_DEGENERATE_CONDITION = 1e6


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


def make_precision_and_logZ(ndim, seed=0, condition=None):
    """Build a random positive-definite precision matrix and analytic log Z."""
    rng = np.random.default_rng(seed)
    if condition is None:
        A = rng.standard_normal((ndim, ndim)) * 0.3
        Sigma = A @ A.T + np.eye(ndim)
        Sigma_inv = np.linalg.inv(Sigma)
        sign, logdet_Sigma = np.linalg.slogdet(Sigma)
        eigvals = np.linalg.eigvalsh(Sigma)
    else:
        if condition < 1:
            raise ValueError("`condition` must be at least 1.")
        Q, R = np.linalg.qr(rng.standard_normal((ndim, ndim)))
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs
        eigvals = np.geomspace(1.0 / condition, 1.0, ndim)
        Sigma = (Q * eigvals) @ Q.T
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma_inv = (Q * (1.0 / eigvals)) @ Q.T
        logdet_Sigma = np.sum(np.log(eigvals))
        sign = 1

    prior_width = 20.0  # [-10, 10]
    log_Z = (0.5 * ndim * np.log(2 * np.pi)
             + 0.5 * logdet_Sigma
             - ndim * np.log(prior_width))
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")
    return Sigma, Sigma_inv, logdet_Sigma, log_Z, eigvals


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


def run_case(label, model, ndim, mu, Sigma, log_Z_true, args,
             stale_retries, cov_jitter, print_posterior=True):
    print(f"\nCase: {label}")
    samples = run_nss(
        model,
        n_live=args.n_live,
        num_mcmc_steps=args.num_mcmc_steps,
        num_delete=args.num_delete,
        termination=args.termination,
        seed=args.seed,
        max_steps=args.max_steps,
        max_shrinkage=args.max_shrinkage,
        stale_retries=stale_retries,
        cov_jitter=cov_jitter,
    )
    meta = samples.pop("__nested__")

    log_Z_est = meta["log_Z"]
    log_Z_err = meta["log_Z_err"]
    diff = log_Z_est - log_Z_true
    sigma_diff = abs(diff) / log_Z_err
    print(f"\nEstimated log Z = {log_Z_est:.4f} +/- {log_Z_err:.4f}")
    print(f"True      log Z = {log_Z_true:.4f}")
    print(f"Difference      = {diff:.4f} ({sigma_diff:.1f} sigma)")
    print(f"n_eff           = {meta['n_eff']}")
    print(f"n_total         = {meta.get('n_total', np.nan)}")

    diagnostics = meta.get("diagnostics", {})
    if diagnostics:
        print("HRSS diagnostics:")
        print(f"  acceptance_rate:     "
              f"{diagnostics.get('acceptance_rate', np.nan):.3f}")
        print(f"  stale_fraction:      "
              f"{diagnostics.get('stale_fraction', np.nan):.3f}")
        print(f"  num_retries:         "
              f"{diagnostics.get('num_retries', 0)}")
        print(f"  num_slice_evals:     "
              f"{diagnostics.get('num_slice_evals', 0)}")
        print(f"  mean_slice_evals:    "
              f"{diagnostics.get('mean_slice_evals', np.nan):.2f}")
        print(f"  mean_shrink:         "
              f"{diagnostics.get('mean_shrink', np.nan):.2f}")
        print(f"  max_cov_jitter:      "
              f"{diagnostics.get('max_cov_jitter', 0.0):.3e}")
        print(f"  num_cov_regularized: "
              f"{diagnostics.get('num_cov_regularized', 0)}")

    stds_true = np.sqrt(np.diag(Sigma))
    max_bias = 0.0
    if print_posterior:
        print("\nPosterior summary "
              "(truth: mean=0, std=diag(Sigma)^0.5):")
    for i in range(ndim):
        s = samples[f"x{i}"]
        m, sd = float(s.mean()), float(s.std())
        bias_sigma = abs(m - mu[i]) / stds_true[i]
        max_bias = max(max_bias, bias_sigma)
        if print_posterior:
            print(f"  x{i:02d}: mean={m:+.3f} +/- {sd:.3f}  "
                  f"(true std={stds_true[i]:.3f}, "
                  f"bias={bias_sigma:.1f}sigma)")

    print(f"\nMax posterior mean bias: {max_bias:.1f}sigma")
    evidence_ok = abs(diff) < 3 * log_Z_err + 1.0
    posterior_ok = max_bias < 3.0
    print(f"Evidence recovery: {'PASS' if evidence_ok else 'FAIL'}")
    print(f"Posterior means:   {'PASS' if posterior_ok else 'FAIL'}")
    return {
        "label": label,
        "log_Z": log_Z_est,
        "log_Z_err": log_Z_err,
        "diff": diff,
        "sigma_diff": sigma_diff,
        "n_eff": meta["n_eff"],
        "diagnostics": diagnostics,
        "passed": evidence_ok and posterior_ok,
    }


def print_comparison(results):
    print("\nA/B comparison:")
    print(f"{'case':>12s} {'dlogZ':>10s} {'sigma':>8s} {'accept':>8s} "
          f"{'stale':>8s} {'evals':>8s} {'retries':>8s} {'cov_reg':>8s}")
    for result in results:
        diagnostics = result["diagnostics"]
        print(f"{result['label']:>12s} "
              f"{result['diff']:>+10.3f} "
              f"{result['sigma_diff']:>8.2f} "
              f"{diagnostics.get('acceptance_rate', np.nan):>8.3f} "
              f"{diagnostics.get('stale_fraction', np.nan):>8.3f} "
              f"{diagnostics.get('num_slice_evals', 0):>8d} "
              f"{diagnostics.get('num_retries', 0):>8d} "
              f"{diagnostics.get('num_cov_regularized', 0):>8d}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degenerate", action="store_true",
                        help=("Compatibility alias; the controlled degenerate "
                              "Gaussian is now the default."))
    parser.add_argument("--random-covariance", action="store_true",
                        help=("Use the original mild random covariance instead "
                              "of the controlled degenerate default."))
    parser.add_argument("--condition", type=float,
                        default=DEFAULT_DEGENERATE_CONDITION,
                        help=("Use a controlled covariance spectrum with this "
                              "condition number. Defaults to "
                              f"{DEFAULT_DEGENERATE_CONDITION:g}."))
    parser.add_argument("--compare-improvements", action="store_true",
                        help=("Run old HRSS settings followed by the improved "
                              "settings and print diagnostics side by side."))
    parser.add_argument("--n-live", type=int, default=5000)
    parser.add_argument("--num-mcmc-steps", type=int, default=None)
    parser.add_argument("--num-delete", type=int, default=500)
    parser.add_argument("--termination", type=float, default=-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matrix-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-shrinkage", type=int, default=100)
    parser.add_argument("--stale-retries", type=int, default=1)
    parser.add_argument("--cov-jitter", type=float, default=1e-6)
    parser.add_argument("--quiet-posterior", action="store_true",
                        help="Do not print per-coordinate posterior summaries.")
    args = parser.parse_args()
    if args.random_covariance:
        args.condition = None
    return args


def main():
    args = parse_args()
    print_header()

    ndim = 20
    if args.num_mcmc_steps is None:
        args.num_mcmc_steps = ndim  # p = d (Yallup+2026)

    mu = np.zeros(ndim)
    Sigma, Sigma_inv, logdet_Sigma, log_Z_true, eigvals = (
        make_precision_and_logZ(ndim, seed=args.matrix_seed,
                                condition=args.condition))

    if args.condition is None:
        print("Likelihood:   N(0, Sigma) with random Sigma")
    else:
        print("Likelihood:   N(0, Sigma) with controlled degenerate Sigma")
    print(f"Prior:        Uniform(-10, 10)^{ndim}")
    print(f"ndim:         {ndim}")
    print(f"Sigma cond:   {np.linalg.cond(Sigma):.1f}")
    print(f"Sigma eigs:   [{eigvals.min():.3e}, {eigvals.max():.3e}]")
    print(f"Analytic logZ: {log_Z_true:.4f}")
    print()
    print("NSS settings:")
    print(f"  n_live:          {args.n_live}")
    print(f"  num_mcmc_steps:  {args.num_mcmc_steps}")
    print(f"  num_delete:      {args.num_delete}")
    print(f"  termination:     {args.termination}  "
          f"(stop when log(Z_live/Z_dead) < {args.termination})")
    print(f"  max_steps:       {args.max_steps}")
    print(f"  max_shrinkage:   {args.max_shrinkage}")
    print()

    model = build_model(ndim, mu, Sigma_inv)

    if args.compare_improvements:
        results = [
            run_case("old", model, ndim, mu, Sigma, log_Z_true, args,
                     stale_retries=0, cov_jitter=0.0,
                     print_posterior=not args.quiet_posterior),
            run_case("improved", model, ndim, mu, Sigma, log_Z_true, args,
                     stale_retries=args.stale_retries,
                     cov_jitter=args.cov_jitter,
                     print_posterior=not args.quiet_posterior),
        ]
        print_comparison(results)
        if not results[-1]["passed"]:
            raise SystemExit("FAILED: improved nested sampler validation")
    else:
        result = run_case(
            "default", model, ndim, mu, Sigma, log_Z_true, args,
            stale_retries=args.stale_retries, cov_jitter=args.cov_jitter,
            print_posterior=not args.quiet_posterior)
        if not result["passed"]:
            raise SystemExit("FAILED: nested sampler validation")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
