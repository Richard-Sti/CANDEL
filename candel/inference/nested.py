# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Nested sampling interface for NumPyro models via NSS.

Decomposes a NumPyro model into separate log-prior and log-likelihood
callables, then runs the NSS nested slice sampler.
"""
from functools import partial
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.distributions import Delta, Unit
from numpyro.infer.initialization import init_to_median
from numpyro.infer.util import log_density

from ..util import fprint
from .optimise import _prior_bounds


def _get_param_info(model, model_args, model_kwargs, seed=42):
    """Extract free parameter names, sizes, bounds, and distributions."""
    substituted = handlers.substitute(
        handlers.seed(model, rng_seed=seed),
        substitute_fn=init_to_median(num_samples=15),
    )
    tr = handlers.trace(substituted).get_trace(*model_args, **model_kwargs)

    names, sizes, lo, hi, dists = [], [], [], [], {}
    for k, v in tr.items():
        if v["type"] != "sample" or v.get("is_observed", False):
            continue
        if v["fn"].__class__.__name__ == "Delta":
            continue

        lb, ub = _prior_bounds(v["fn"], sobol_n_sigma=None)
        if lb is None:
            continue

        size = int(np.prod(v["value"].shape))
        lb = np.atleast_1d(np.asarray(lb))
        ub = np.atleast_1d(np.asarray(ub))
        names.append(k)
        sizes.append(size)
        lo.append(np.broadcast_to(lb.ravel(), size))
        hi.append(np.broadcast_to(ub.ravel(), size))
        dists[k] = v["fn"]

    return names, sizes, np.concatenate(lo), np.concatenate(hi), dists


def _unflatten(x, names, sizes):
    """Convert flat parameter vector to named dict."""
    params = {}
    offset = 0
    for name, size in zip(names, sizes):
        params[name] = (x[offset:offset + size].reshape(())
                        if size == 1 else x[offset:offset + size])
        offset += size
    return params


def decompose_model(model, model_args=(), model_kwargs=None, seed=42):
    """Decompose a NumPyro model into log-prior and log-likelihood.

    In NumPyro, `sample(name, dist)` sites contribute to the prior, while
    `factor(name, value)` sites (which appear as observed Unit samples)
    contribute to the likelihood.

    Both returned functions enforce prior support: they return -inf outside
    the parameter bounds. This is critical because NumPyro's Uniform.log_prob
    does not return -inf outside its support when validate_args=False.

    Returns
    -------
    log_prior_fn, log_likelihood_fn, log_joint_fn : callables
    names : list of str
    sizes : list of int
    lo, hi : 1D arrays of parameter bounds
    dists : dict mapping param names to numpyro distributions
    """
    if model_kwargs is None:
        model_kwargs = {}

    names, sizes, lo, hi, dists = _get_param_info(
        model, model_args, model_kwargs, seed)

    def _get_trace(params):
        def sub_fn(msg):
            if msg["name"] in params:
                return params[msg["name"]]
        substituted = handlers.substitute(
            handlers.seed(model, rng_seed=seed), substitute_fn=sub_fn)
        return handlers.trace(substituted).get_trace(
            *model_args, **model_kwargs)

    lo_jax = jnp.array(lo)
    hi_jax = jnp.array(hi)

    def _in_support(x):
        return jnp.all((x >= lo_jax) & (x <= hi_jax))

    def log_prior_fn(x):
        params = _unflatten(x, names, sizes)
        tr = _get_trace(params)
        lp = 0.0
        for k, v in tr.items():
            if (v["type"] == "sample"
                    and not v.get("is_observed", False)
                    and not isinstance(v["fn"], (Delta, Unit))):
                lp = lp + jnp.sum(v["fn"].log_prob(v["value"]))
        # NumPyro Uniform.log_prob doesn't return -inf outside the support
        # when validate_args=False (the default). Enforce support explicitly.
        return jnp.where(_in_support(x), lp, -jnp.inf)

    def log_likelihood_fn(x):
        params = _unflatten(x, names, sizes)
        tr = _get_trace(params)
        ll = 0.0
        for k, v in tr.items():
            if (v["type"] == "sample"
                    and v.get("is_observed", False)
                    and isinstance(v["fn"], Unit)):
                ll = ll + jnp.sum(v["fn"].log_prob(v["value"]))
        # Enforce prior support: NSS slice sampler checks loglikelihood but
        # not log_prior for acceptance. Without this, the sampler explores
        # outside the prior support, biasing the evidence by ~O(d) nats.
        ll = jnp.where(_in_support(x), ll, -jnp.inf)
        # Clamp +inf from numerical overflow (e.g. phi prior weight
        # log_Z underflow). Must preserve -inf for out-of-support.
        max_ll = jnp.finfo(ll.dtype).max / 2
        return jnp.where(ll == jnp.inf, max_ll, ll)

    def log_joint_fn(x):
        params = _unflatten(x, names, sizes)
        ld, _ = log_density(model, model_args, model_kwargs, params)
        return ld

    return (log_prior_fn, log_likelihood_fn, log_joint_fn,
            names, sizes, lo, hi, dists)


def validate_decomposition(log_prior_fn, log_likelihood_fn, log_joint_fn,
                           lo, hi, n_test=10, seed=123, atol=1e-2, rtol=1e-5):
    """Check that log_prior + log_likelihood == log_joint at random points."""
    key = jax.random.PRNGKey(seed)
    ndim = len(lo)
    test_points = lo + jax.random.uniform(key, (n_test, ndim)) * (hi - lo)

    for i in range(n_test):
        x = test_points[i]
        lp = float(log_prior_fn(x))
        ll = float(log_likelihood_fn(x))
        lj = float(log_joint_fn(x))
        diff = abs(lp + ll - lj)
        scale = max(abs(lj), 1.0)
        if diff > atol + rtol * scale:
            raise ValueError(
                f"Decomposition failed at point {i}: "
                f"prior={lp:.4f} + ll={ll:.4f} = {lp+ll:.4f} != "
                f"joint={lj:.4f} (diff={diff:.6f}, "
                f"tol={atol + rtol * scale:.6f})")

    # Verify that log_prior and log_likelihood return -inf outside the prior
    # support. Without this, the slice sampler explores outside the prior,
    # causing an O(d) evidence bias (see CLAUDE.md "NSS evidence bias fix").
    key2 = jax.random.PRNGKey(seed + 999)
    for dim in range(ndim):
        for side, offset in [("above upper", hi[dim] + 1.0),
                             ("below lower", lo[dim] - 1.0)]:
            x_out = lo + jax.random.uniform(key2, (ndim,)) * (hi - lo)
            key2, _ = jax.random.split(key2)
            x_out = x_out.at[dim].set(offset)
            lp_out = float(log_prior_fn(x_out))
            ll_out = float(log_likelihood_fn(x_out))
            if np.isfinite(lp_out):
                raise ValueError(
                    f"log_prior_fn returned finite value ({lp_out:.4f}) "
                    f"{side} bound at dim {dim}. "
                    f"This causes O(d) evidence bias.")
            if np.isfinite(ll_out):
                raise ValueError(
                    f"log_likelihood_fn returned finite value ({ll_out:.4f}) "
                    f"{side} bound at dim {dim}. "
                    f"This causes O(d) evidence bias.")

    fprint(f"Decomposition validated at {n_test} points "
           f"(atol={atol}, rtol={rtol})")


def sample_prior(dists, names, sizes, n, seed=0):
    """Draw n samples from the prior distributions."""
    key = jax.random.PRNGKey(seed)
    ndim = sum(sizes)
    samples = jnp.empty((n, ndim))
    offset = 0
    for name, size in zip(names, sizes):
        key, subkey = jax.random.split(key)
        s = dists[name].sample(subkey, (n,))
        if s.ndim == 1:
            s = s[:, None]
        samples = samples.at[:, offset:offset + size].set(s)
        offset += size
    return samples



def run_nss(model, model_args=(), model_kwargs=None,
            n_live=500, num_mcmc_steps=50, num_delete=1,
            termination=-3, seed=42, validate=True):
    """Run NSS nested sampling on a NumPyro model.

    Recommended settings (Yallup+2025, arXiv:2601.23252):
      n_live=1000, num_mcmc_steps=ndim, num_delete=n_live//10.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    model_args, model_kwargs : tuple, dict
        Arguments passed to the model.
    n_live : int
        Number of live points.
    num_mcmc_steps : int
        Number of slice sampling steps per dead point (p=d recommended).
    num_delete : int
        Number of dead points per iteration (10% of n_live recommended).
    termination : float
        log(Z_live / Z_dead) threshold for termination.
    seed : int
        Random seed.
    validate : bool
        If True, validate the prior/likelihood decomposition first.

    Returns
    -------
    samples : dict
        Weighted posterior samples, resampled to equal weights.
        Same format as MCMC output: ``{name: array(n_eff,)}`` for
        scalar params or ``{name: array(n_eff, dim)}`` for vector params.
        Contains ``__nested__`` key with metadata (log_Z, n_eff, etc.).
    """
    try:
        import blackjax  # noqa: F401
        import tqdm
        from blackjax.ns.adaptive import AdaptiveNSState, init_integrator
        from blackjax.ns.base import init_state_strategy
        from blackjax.ns.utils import finalise, log_weights
        from blackjax.smc.tuning.from_particles import \
            particles_covariance_matrix
    except ImportError as e:
        raise ImportError(
            "Nested sampling requires 'nss' and the handley-lab blackjax "
            "fork. Install with:\n"
            "  pip install git+https://github.com/yallup/nss.git\n"
            "  pip install 'blackjax @ git+https://github.com/handley-lab/"
            "blackjax@nested_sampling' --no-deps"
        ) from e

    if model_kwargs is None:
        model_kwargs = {}

    # ---- Decompose model ----
    (log_prior_fn, log_likelihood_fn, log_joint_fn,
     names, sizes, lo, hi, dists) = decompose_model(
        model, model_args, model_kwargs, seed)

    if validate:
        validate_decomposition(
            log_prior_fn, log_likelihood_fn, log_joint_fn, lo, hi)

    ndim = sum(sizes)
    if num_mcmc_steps is None:
        num_mcmc_steps = ndim

    init_batch_size = num_delete
    fprint(f"NSS: ndim={ndim}, num_mcmc_steps={num_mcmc_steps}, "
           f"n_live={n_live}, num_delete={num_delete}")

    # ---- Draw initial live points from prior ----
    initial_samples = sample_prior(dists, names, sizes, n_live, seed)

    # ---- Batched initialisation (avoids GPU OOM) ----
    init_state_fn = partial(
        init_state_strategy,
        logprior_fn=log_prior_fn,
        loglikelihood_fn=log_likelihood_fn,
    )
    batched_fn = jax.jit(jax.vmap(init_state_fn))

    all_results = []
    batches = range(0, n_live, init_batch_size)
    for i in tqdm.tqdm(batches, desc="init live pts", unit=" batch"):
        batch = initial_samples[i:i + init_batch_size]
        result = jax.block_until_ready(batched_fn(batch))
        result = jax.tree.map(lambda x: np.asarray(x), result)
        all_results.append(result)

    particles = jax.tree.map(
        lambda *arrs: jnp.concatenate(arrs, axis=0), *all_results)
    integrator = init_integrator(particles)
    cov = jnp.atleast_2d(particles_covariance_matrix(particles.position))

    state = AdaptiveNSState(
        particles=particles,
        inner_kernel_params={"cov": cov},
        integrator=integrator,
    )

    # ---- Build NSS step function ----
    algo = blackjax.nss(
        logprior_fn=log_prior_fn,
        loglikelihood_fn=log_likelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_mcmc_steps,
    )

    @jax.jit
    def step_fn(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    # ---- Run NSS loop ----
    rng_key = jax.random.PRNGKey(seed + 1)

    # Warmup JIT
    (_, rng_key), _ = jax.block_until_ready(
        step_fn((state, rng_key), None))

    dead = []
    t0 = timer()
    n_dead = 0
    gap_prev = None
    gap_rate_ema = None
    ema_alpha = num_delete / n_live  # ~1 step per timescale → α≈0.1 for defaults
    with tqdm.tqdm(desc="NSS", unit=" pts") as pbar:
        while not (state.integrator.logZ_live
                   - state.integrator.logZ < termination):
            (state, rng_key), dead_info = step_fn(
                (state, rng_key), None)
            dead.append(dead_info)
            n_dead += num_delete

            logZ = float(state.integrator.logZ)
            gap = float(state.integrator.logZ_live) - logZ

            # Rolling EMA of gap-change rate (nats/pt, expected < 0)
            if gap_prev is not None:
                d_gap = (gap - gap_prev) / num_delete
                gap_rate_ema = (d_gap if gap_rate_ema is None
                                else ema_alpha * d_gap
                                + (1 - ema_alpha) * gap_rate_ema)
            gap_prev = gap

            # ETA: remaining pts ≈ (gap − termination) / |d_gap/pt|
            eta_str = "?"
            pts_per_sec = pbar.format_dict.get("rate") or 0
            if (gap_rate_ema is not None and gap_rate_ema < 0
                    and pts_per_sec > 0):
                remaining_pts = (gap - termination) / (-gap_rate_ema)
                eta_sec = remaining_pts / pts_per_sec
                if eta_sec < 3600:
                    eta_str = f"{eta_sec / 60:.0f}m"
                else:
                    eta_str = f"{eta_sec / 3600:.1f}h"

            pbar.set_postfix({"logZ": f"{logZ:.2f}",
                              "gap": f"{gap:.2f}",
                              "ETA": eta_str})
            pbar.update(num_delete)

            # Check for non-finite likelihood values
            ll_live = state.particles.loglikelihood
            ll_dead = dead_info.particles.loglikelihood
            if not jnp.all(jnp.isfinite(ll_live)):
                n_bad = int((~jnp.isfinite(ll_live)).sum())
                print(f"\n*** WARNING: {n_bad} non-finite live "
                      f"logL at {n_dead} dead ***", flush=True)
                bad_idx = jnp.where(~jnp.isfinite(ll_live))[0]
                for idx in bad_idx[:3]:
                    pos = state.particles.position[idx]
                    print(f"  idx={int(idx)}: "
                          f"logL={float(ll_live[idx]):.4f}"
                          f"  params={np.asarray(pos)}", flush=True)
                break
            if not jnp.all(jnp.isfinite(ll_dead)):
                n_bad = int((~jnp.isfinite(ll_dead)).sum())
                print(f"\n*** WARNING: {n_bad} non-finite dead "
                      f"logL at {n_dead} dead ***", flush=True)
                for j in range(min(n_bad, 3)):
                    idx = jnp.where(~jnp.isfinite(ll_dead))[0][j]
                    pos = dead_info.particles.position[idx]
                    print(f"  idx={int(idx)}: logL="
                          f"{float(ll_dead[idx]):.4f}"
                          f"  params={np.asarray(pos)}", flush=True)
                break
            if not jnp.isfinite(logZ):
                print(f"\n*** WARNING: logZ={logZ} at {n_dead} dead "
                      f"but all logL finite! ***", flush=True)
                print(f"  max live logL: {float(ll_live.max()):.2f}",
                      flush=True)
                print(f"  dead logL: {np.asarray(ll_dead)}", flush=True)
                break
    dt = timer() - t0

    # ---- Finalise and compute evidence ----
    final_state = finalise(state, dead)
    logw = log_weights(rng_key, final_state)
    logw_mean = logw.mean(axis=-1)

    # Evidence estimate (one per random compression realisation)
    minimum = jnp.nan_to_num(logw).min()
    logzs = jax.scipy.special.logsumexp(
        jnp.nan_to_num(logw, nan=minimum), axis=0)
    log_Z = float(logzs.mean())
    log_Z_err = float(logzs.std())

    # Resample to equal-weight samples
    positions = final_state.particles.position
    logw_mean = jnp.where(jnp.isfinite(logw_mean), logw_mean, -jnp.inf)
    logw_norm = logw_mean - jax.scipy.special.logsumexp(logw_mean)
    p = jnp.exp(logw_norm)
    p = jnp.where(jnp.isfinite(p), p, 0.0)
    p = p / p.sum()
    n_eff = int(jnp.exp(-jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0))))
    key2 = jax.random.PRNGKey(seed + 2)
    indices = jax.random.choice(
        key2, positions.shape[0], shape=(max(n_eff, 100),), p=p)
    resampled = positions[indices]

    # Unpack into named dict (same format as MCMC samples)
    samples = {}
    offset = 0
    for name, size in zip(names, sizes):
        s = resampled[:, offset:offset + size]
        samples[name] = s.squeeze(axis=-1) if size == 1 else s
        offset += size

    samples["__nested__"] = {
        "log_Z": log_Z,
        "log_Z_err": log_Z_err,
        "n_eff": n_eff,
        "time": dt,
        "names": names,
        "sizes": sizes,
    }

    return samples


def print_nested_summary(samples, meta=None):
    """Print a summary table for nested sampling posterior samples."""
    if meta is None:
        meta = samples.get("__nested__")

    if meta is not None:
        print(f"\nlog Z = {meta['log_Z']:.2f} +/- {meta['log_Z_err']:.2f}")
        print(f"n_eff = {meta['n_eff']}")

    header = (f"{'':>20s} {'mean':>10s} {'std':>10s} {'median':>10s} "
              f"{'5.0%':>10s} {'95.0%':>10s}")
    print(f"\n{header}")
    print("-" * len(header))

    for k, v in samples.items():
        if k == "__nested__":
            continue
        v = np.asarray(v)
        if v.ndim == 0:
            continue
        if v.ndim == 1:
            _print_param_row(k, v)
        elif v.ndim == 2:
            for i in range(v.shape[1]):
                _print_param_row(f"{k}[{i}]", v[:, i])


def _print_param_row(name, x):
    """Print one row of the summary table."""
    q5, q50, q95 = np.percentile(x, [5, 50, 95])
    print(f"{name:>20s} {x.mean():10.3f} {x.std():10.3f} "
          f"{q50:10.3f} {q5:10.3f} {q95:10.3f}")
