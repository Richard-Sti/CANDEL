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
Nested Slice Sampler (NSS) for NumPyro models — self-contained, no blackjax.

Decomposes a NumPyro model into separate log-prior and log-likelihood
callables, then runs a faithful reimplementation of the NSS algorithm from
Yallup+2026 (arXiv:2601.23252) using hit-and-run slice sampling.

Algorithm matches the handley-lab blackjax fork exactly:
  - Same stepping-out/shrinking slice kernel (Neal 2003)
  - Same Mahalanobis-normalised direction proposal
  - Same covariance update (empirical, ddof=0) from live points
  - Same evidence integrator (logX, logZ, logZ_live)
  - Same stochastic prior-volume compression for log_weights
"""
import os
from functools import partial
from timeit import default_timer as timer
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from numpyro import handlers
from numpyro.distributions import Delta, Unit
from numpyro.infer.initialization import init_to_median
from numpyro.infer.util import log_density

from ..util import fprint
from .optimise import _prior_bounds

# ── Model decomposition (unchanged) ────────────────────────────────────


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

    def _prior_and_likelihood(x):
        """Single trace → (log_prior, log_likelihood)."""
        params = _unflatten(x, names, sizes)
        tr = _get_trace(params)
        lp = 0.0
        ll = 0.0
        for k, v in tr.items():
            if v["type"] != "sample":
                continue
            if not v.get("is_observed", False):
                if not isinstance(v["fn"], (Delta, Unit)):
                    lp = lp + jnp.sum(v["fn"].log_prob(v["value"]))
            elif isinstance(v["fn"], Unit):
                ll = ll + jnp.sum(v["fn"].log_prob(v["value"]))

        in_supp = _in_support(x)
        lp = jnp.where(in_supp, lp, -jnp.inf)
        ll = jnp.where(in_supp, ll, -jnp.inf)
        ll = jnp.where(jnp.isnan(ll), -jnp.inf, ll)
        max_ll = jnp.finfo(ll.dtype).max / 2
        ll = jnp.where(ll == jnp.inf, max_ll, ll)
        return lp, ll

    def log_prior_fn(x):
        lp, _ = _prior_and_likelihood(x)
        return lp

    def log_likelihood_fn(x):
        _, ll = _prior_and_likelihood(x)
        return ll

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
    # causing an O(d) evidence bias.
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


# ── Self-contained Nested Slice Sampler ─────────────────────────────────────
#
# Faithful reimplementation of the handley-lab blackjax NSS fork.
# References:
#   Yallup+2026  arXiv:2601.23252   (NSS algorithm)
#   Neal (2003)  Ann. Stat. 31(3)   (slice sampling)
#   Skilling (2006)                 (nested sampling evidence integral)

class _Particle(NamedTuple):
    position: jax.Array       # (ndim,)
    logprior: jax.Array       # scalar — log prior density
    loglikelihood: jax.Array  # scalar
    logL_birth: jax.Array     # scalar — logL threshold at birth


class _Integrator(NamedTuple):
    logX: jax.Array       # log prior volume remaining
    logZ: jax.Array       # accumulated log evidence from dead points
    logZ_live: jax.Array  # live-point estimate of remaining log evidence


class _NSSState(NamedTuple):
    particles: _Particle    # leading axis (n_live,)
    integrator: _Integrator
    cov: jax.Array          # (ndim, ndim) empirical covariance


class _DeadInfo(NamedTuple):
    # leading axis: (deletion batch,) per step, (n_total,) after finalise
    particles: _Particle
    info: "_NSSInfo" = None


class _SliceInfo(NamedTuple):
    is_accepted: jax.Array
    num_steps: jax.Array
    num_shrink: jax.Array


class _SliceStats(NamedTuple):
    num_chains: jax.Array
    num_transitions: jax.Array
    num_accepted: jax.Array
    num_step_out: jax.Array
    num_shrink: jax.Array
    num_stale_chains: jax.Array
    num_retries: jax.Array


class _NSSInfo(NamedTuple):
    stats: _SliceStats
    max_cov_jitter: jax.Array
    max_cov_condition: jax.Array
    num_cov_regularized: jax.Array


def _logmeanexp(x):
    n = jnp.array(x.shape[0], dtype=x.dtype)
    return jax.scipy.special.logsumexp(x) - jnp.log(n)


def _log1mexp(x):
    """Numerically stable log(1 - exp(x)) for x < 0."""
    x = jnp.minimum(x, -jnp.finfo(x.dtype).eps)
    return jnp.where(x > -0.6931472,          # threshold ≈ log(2)
                     jnp.log(-jnp.expm1(x)),
                     jnp.log1p(-jnp.exp(x)))


def _broadcast_chain_mask(mask, x):
    shape = mask.shape + (1,) * (x.ndim - mask.ndim)
    return mask.reshape(shape)


def _select_chain_values(mask, new, old):
    return jnp.where(_broadcast_chain_mask(mask, new), new, old)


def _add_masked_slice_stats(stats, extra, mask):
    mask = mask.astype(jnp.int32)
    return _SliceStats(*(a + b * mask for a, b in zip(stats, extra)))


def _finalise_chain_stats(chain_stats, accepted_any, retry_counts):
    stats = jax.tree.map(lambda x: jnp.sum(x, axis=0), chain_stats)
    return stats._replace(
        num_chains=jnp.array(accepted_any.shape[0], dtype=jnp.int32),
        num_stale_chains=jnp.sum((~accepted_any).astype(jnp.int32)),
        num_retries=jnp.sum(retry_counts.astype(jnp.int32)),
    )


def _merge_retry_attempt(new_p, chain_stats, accepted_any, retry_counts,
                         retry_new_p, retry_stats, retry_accepted):
    stale = ~accepted_any
    new_p = jax.tree.map(
        lambda retry, current: _select_chain_values(stale, retry, current),
        retry_new_p, new_p)
    chain_stats = _add_masked_slice_stats(chain_stats, retry_stats, stale)
    retry_counts = retry_counts + stale.astype(jnp.int32)
    accepted_any = accepted_any | (stale & retry_accepted)
    return new_p, chain_stats, accepted_any, retry_counts


def _has_stale_chains(accepted_any):
    return bool(np.asarray(jax.device_get(jnp.any(~accepted_any))))


def _regularize_covariance(cov, relative_jitter):
    """Symmetrise and minimally jitter a live-point covariance matrix."""
    cov = 0.5 * (cov + cov.T)
    dtype = cov.dtype
    ndim = cov.shape[0]
    if relative_jitter is None:
        relative_jitter = 0.0
    relative_jitter = jnp.array(relative_jitter, dtype=dtype)

    eigvals = jnp.linalg.eigvalsh(cov)
    min_eig = jnp.min(eigvals)
    max_eig = jnp.max(eigvals)
    diag_scale = jnp.maximum(jnp.mean(jnp.diag(cov)),
                             jnp.array(1.0, dtype=dtype))
    floor = jnp.maximum(relative_jitter * diag_scale,
                        jnp.finfo(dtype).eps * diag_scale)
    jitter = jnp.where(relative_jitter > 0.0,
                       jnp.maximum(floor - min_eig, 0.0),
                       jnp.array(0.0, dtype=dtype))
    cov = cov + jitter * jnp.eye(ndim, dtype=dtype)
    min_reg = min_eig + jitter
    max_reg = max_eig + jitter
    condition = max_reg / jnp.maximum(min_reg, floor)
    regularized = jitter > 0.0
    return cov, jitter, condition, regularized


def _init_integrator(particles):
    """Initialise evidence integrator from initial live points."""
    dtype = particles.loglikelihood.dtype
    logX = jnp.array(0.0, dtype=dtype)
    logZ = jnp.array(-jnp.inf, dtype=dtype)
    logZ_live = _logmeanexp(particles.loglikelihood) + logX
    return _Integrator(logX, logZ, logZ_live)


def _update_integrator(integrator, live_particles, dead_particles):
    """Update evidence integrator after one NS step.

    Treats batch deletion as sequential: when the j-th dead point in the
    current deletion batch is removed, there are n_live-j live points.
    """
    ll_live = live_particles.loglikelihood
    ll_dead = dead_particles.loglikelihood
    n_live = ll_live.shape[0]
    n_dead = ll_dead.shape[0]
    dtype = ll_live.dtype

    num_live = jnp.arange(n_live, n_live - n_dead, -1, dtype=dtype)
    delta_logX = -1.0 / num_live
    logX = integrator.logX + jnp.cumsum(delta_logX)
    log_dX = logX + jnp.log(1.0 - jnp.exp(delta_logX))
    log_dZ = ll_dead + log_dX

    logZ = jnp.logaddexp(
        integrator.logZ, jax.scipy.special.logsumexp(log_dZ))
    logZ_live = _logmeanexp(ll_live) + logX[-1]
    return _Integrator(logX[-1], logZ, logZ_live)


def _make_particle(position, logprior_fn, loglikelihood_fn, logL_birth):
    """Evaluate a position and wrap it into a _Particle."""
    lp = logprior_fn(position)
    ll = loglikelihood_fn(position)
    birth = logL_birth * jnp.ones_like(ll)
    return _Particle(position, lp, ll, birth)


def _sample_direction(rng_key, position, cov):
    """Sample a Mahalanobis-normalised direction from N(0, cov), scaled by 2.

    Matches blackjax.mcmc.ss.sample_direction_from_covariance exactly.
    """
    ndim = position.shape[0]
    d = jax.random.multivariate_normal(
        rng_key, jnp.zeros(ndim, dtype=cov.dtype), cov)
    invcov = jnp.linalg.inv(cov)
    norm = jnp.sqrt(jnp.einsum("i,ij,j", d, invcov, d))
    return d / norm * 2.0


def _hrss_step(rng_key, particle, logprior_fn, loglikelihood_fn,
               logL_0, cov, max_steps=10, max_shrinkage=100):
    """One constrained hit-and-run slice sampling step (Neal 2003).

    Accepts a new position iff:
      logprior(x_new) >= logprior(x) + log(u)   [vertical slice on prior]
      loglikelihood(x_new) > logL_0              [NS likelihood contour]

    Parameters
    ----------
    rng_key    : jax.random.PRNGKey — random key for the HRSS proposal
    particle   : _Particle — current live point
    logprior_fn : callable — log-prior evaluator
    loglikelihood_fn : callable — log-likelihood evaluator
    logL_0     : scalar — current likelihood threshold
    cov        : (ndim, ndim) — covariance for direction proposal
    max_steps  : int — max stepping-out steps per side (default 10)
    max_shrinkage : int — max shrink iterations (default 100)

    Returns
    -------
    _Particle
        Accepted new particle, unchanged if no accepted sample is found.
    _SliceInfo
        Acceptance and stepping-out/shrink diagnostics for this HRSS step.
    """
    vs_key, hs_key, dir_key = jax.random.split(rng_key, 3)

    d = _sample_direction(dir_key, particle.position, cov)

    # Vertical slice: fix a log-prior level
    u = jax.random.uniform(vs_key)
    logslice = particle.logprior + jnp.log(u)

    def _eval(t):
        x = particle.position + t * d
        new_p = _make_particle(
            x, logprior_fn, loglikelihood_fn, logL_0)
        ok = ((new_p.logprior >= logslice)
              & (new_p.loglikelihood > logL_0))
        return new_p, ok

    # Stepping-out: find interval [left, right] that brackets the current pt
    rng_key, subkey = jax.random.split(hs_key)
    u2, v = jax.random.uniform(subkey, (2,))
    j = jnp.floor(max_steps * v).astype(jnp.int32)
    k = (max_steps - 1) - j

    def _step_body(carry):
        i, s, t, _ = carry
        t = t + s
        _, is_acc = _eval(t)
        return i - 1, s, t, is_acc

    def _step_cond(carry):
        i, _, _, is_acc = carry
        return is_acc & (i > 0)

    j, _, left, _ = jax.lax.while_loop(
        _step_cond, _step_body,
        (j + 1, jnp.array(-1.0), 1.0 - u2, True))
    k, _, right, _ = jax.lax.while_loop(
        _step_cond, _step_body,
        (k + 1, jnp.array(1.0), -u2, True))

    # Shrinking: uniform sample within [left, right], shrink on rejection
    def _shrink_body(carry):
        n, key, lo, hi, p, _ = carry
        key, subkey = jax.random.split(key)
        t = jax.random.uniform(subkey, minval=lo, maxval=hi)
        new_p, is_acc = _eval(t)
        lo = jnp.where(t < 0.0, t, lo)
        hi = jnp.where(t > 0.0, t, hi)
        return n + 1, key, lo, hi, new_p, is_acc

    def _shrink_cond(carry):
        n, _, _, _, _, is_acc = carry
        return ~is_acc & (n < max_shrinkage)

    n_shrink, _, _, _, new_p, is_acc = jax.lax.while_loop(
        _shrink_cond, _shrink_body,
        (0, rng_key, left, right, particle, False))

    # If shrinkage found nothing, stay put
    new_p = jax.tree.map(
        lambda new, old: jnp.where(is_acc, new, old), new_p, particle)
    info = _SliceInfo(
        is_accepted=is_acc,
        num_steps=max_steps + 1 - j - k,
        num_shrink=n_shrink,
    )
    return new_p, info


def _nss_attempt(rng_key, state, logprior_fn, loglikelihood_fn,
                 num_delete, num_mcmc_steps, max_steps=10,
                 max_shrinkage=100, cov_jitter=1e-6):
    """Propose one batch of NSS replacements without retry branching.

    Matches blackjax.nss exactly:
      - delete_fn: top_k(-logL, num_delete)
      - L*: max(dead loglikelihoods)
      - start pts: uniform over survivors with logL > L*
      - MCMC: num_mcmc_steps HRSS steps via lax.scan
      - cov update: jnp.cov(ddof=0, rowvar=False)
      - integrator: updated after replacement
    """
    particles = state.particles

    # Identify num_delete worst live points
    _, dead_idx = jax.lax.top_k(
        -particles.loglikelihood, num_delete)
    dead_p = jax.tree.map(lambda x: x[dead_idx], particles)
    logL_0 = dead_p.loglikelihood.max()

    # Select starting particles uniformly from survivors (logL > L*)
    choice_key, mcmc_key = jax.random.split(rng_key)
    w = (particles.loglikelihood > logL_0).astype(jnp.float32)
    w = jnp.where(w.sum() > 0.0, w, jnp.ones_like(w))
    start_idx = jax.random.choice(
        choice_key, particles.loglikelihood.shape[0],
        shape=(num_delete,), p=w / w.sum(), replace=True)
    start_p = jax.tree.map(lambda x: x[start_idx], particles)

    cov, cov_jitter_added, cov_condition, cov_regularized = (
        _regularize_covariance(state.cov, cov_jitter))

    # Run num_mcmc_steps HRSS steps for each replacement particle (take last)
    def one_chain(rng_key, p0):
        def body(p, k):
            new_p, info = _hrss_step(
                k, p, logprior_fn, loglikelihood_fn,
                logL_0, cov, max_steps, max_shrinkage)
            return new_p, info

        keys = jax.random.split(rng_key, num_mcmc_steps)
        final_p, infos = jax.lax.scan(body, p0, keys)
        accepted = infos.is_accepted.astype(jnp.int32)
        stats = _SliceStats(
            num_chains=jnp.array(0, dtype=jnp.int32),
            num_transitions=jnp.array(num_mcmc_steps, dtype=jnp.int32),
            num_accepted=jnp.sum(accepted),
            num_step_out=jnp.sum(infos.num_steps.astype(jnp.int32)),
            num_shrink=jnp.sum(infos.num_shrink.astype(jnp.int32)),
            num_stale_chains=jnp.array(0, dtype=jnp.int32),
            num_retries=jnp.array(0, dtype=jnp.int32),
        )
        return final_p, stats, jnp.any(infos.is_accepted)

    new_p, chain_stats, accepted_any = jax.vmap(one_chain)(
        jax.random.split(mcmc_key, num_delete), start_p)
    return (dead_idx, dead_p, new_p, chain_stats, accepted_any,
            cov_jitter_added, cov_condition,
            cov_regularized.astype(jnp.int32))


def _nss_finish(state, dead_idx, dead_p, new_p, chain_stats, accepted_any,
                retry_counts, cov_jitter_added, cov_condition,
                cov_regularized):
    """Finish one NSS iteration after retry selection is resolved."""
    particles = state.particles
    stats = _finalise_chain_stats(chain_stats, accepted_any, retry_counts)

    # Replace dead positions
    updated = jax.tree.map(
        lambda full, new: full.at[dead_idx].set(new), particles, new_p)

    # Update covariance and integrator
    new_cov = jnp.atleast_2d(
        jnp.cov(updated.position, ddof=0, rowvar=False))
    new_integrator = _update_integrator(
        state.integrator, updated, dead_p)

    info = _NSSInfo(
        stats=stats,
        max_cov_jitter=cov_jitter_added,
        max_cov_condition=cov_condition,
        num_cov_regularized=cov_regularized,
    )
    return _NSSState(updated, new_integrator, new_cov), _DeadInfo(dead_p, info)


def _make_nss_step(logprior_fn, loglikelihood_fn, num_delete,
                   num_mcmc_steps, max_steps=10, max_shrinkage=100,
                   stale_retries=1, cov_jitter=1e-6):
    """Build an NSS step; retry attempts reuse the same compiled kernel."""

    @jax.jit
    def _attempt(state, rng_key):
        return _nss_attempt(
            rng_key, state, logprior_fn, loglikelihood_fn,
            num_delete, num_mcmc_steps, max_steps=max_steps,
            max_shrinkage=max_shrinkage, cov_jitter=cov_jitter)

    @jax.jit
    def _finish(state, dead_idx, dead_p, new_p, chain_stats, accepted_any,
                retry_counts, cov_jitter_added, cov_condition,
                cov_regularized):
        return _nss_finish(
            state, dead_idx, dead_p, new_p, chain_stats, accepted_any,
            retry_counts, cov_jitter_added, cov_condition, cov_regularized)

    def _step(state, rng_key):
        rng_key, attempt_key = jax.random.split(rng_key)
        (dead_idx, dead_p, new_p, chain_stats, accepted_any,
         cov_jitter_added, cov_condition, cov_regularized) = _attempt(
             state, attempt_key)
        retry_counts = jnp.zeros(accepted_any.shape, dtype=jnp.int32)

        for _ in range(stale_retries):
            if not _has_stale_chains(accepted_any):
                break
            rng_key, retry_key = jax.random.split(rng_key)
            (_, _, retry_new_p, retry_stats, retry_accepted,
             _, _, _) = _attempt(state, retry_key)
            new_p, chain_stats, accepted_any, retry_counts = (
                _merge_retry_attempt(
                    new_p, chain_stats, accepted_any, retry_counts,
                    retry_new_p, retry_stats, retry_accepted))

        return _finish(
            state, dead_idx, dead_p, new_p, chain_stats, accepted_any,
            retry_counts, cov_jitter_added, cov_condition, cov_regularized)

    return _step


def _adjust_num_delete_for_devices(num_delete, n_devices, n_live):
    """Return the total deletion count for ``num_delete`` chains/device."""
    if n_devices <= 1:
        return num_delete
    adjusted = num_delete * n_devices
    if adjusted >= n_live:
        raise ValueError(
            "The per-device `num_delete` multiplied by the number of NSS "
            f"devices must be less than n_live; got {num_delete} * "
            f"{n_devices} = {adjusted} for n_live={n_live}.")
    return adjusted


def _make_nss_step_pmap(logprior_fn, loglikelihood_fn, total_num_delete,
                        num_mcmc_steps, devices, max_steps=10,
                        max_shrinkage=100, stale_retries=1,
                        cov_jitter=1e-6):
    """Build one NSS step with replacement chains sharded over devices."""
    n_devices = len(devices)
    if total_num_delete % n_devices != 0:
        raise ValueError(
            "multi-device NSS requires the total deletion count to be "
            "divisible by the number of devices.")
    chains_per_device = total_num_delete // n_devices

    @jax.jit
    def _prepare(state, rng_key):
        particles = state.particles

        _, dead_idx = jax.lax.top_k(
            -particles.loglikelihood, total_num_delete)
        dead_p = jax.tree.map(lambda x: x[dead_idx], particles)
        logL_0 = dead_p.loglikelihood.max()

        choice_key, mcmc_key = jax.random.split(rng_key)
        w = (particles.loglikelihood > logL_0).astype(jnp.float32)
        w = jnp.where(w.sum() > 0.0, w, jnp.ones_like(w))
        start_idx = jax.random.choice(
            choice_key, particles.loglikelihood.shape[0],
            shape=(total_num_delete,),
            p=w / w.sum(), replace=True)
        start_p = jax.tree.map(lambda x: x[start_idx], particles)
        mcmc_keys = jax.random.split(mcmc_key, n_devices)
        return dead_idx, dead_p, logL_0, start_p, mcmc_keys

    @partial(jax.pmap, in_axes=(0, 0, None, None), devices=devices)
    def _run_chains(device_key, p0_device, logL_0, cov):
        def one_chain(rng_key, p0):
            def body(p, k):
                new_p, info = _hrss_step(
                    k, p, logprior_fn, loglikelihood_fn,
                    logL_0, cov, max_steps, max_shrinkage)
                return new_p, info

            keys = jax.random.split(rng_key, num_mcmc_steps)
            final_p, infos = jax.lax.scan(body, p0, keys)
            accepted = infos.is_accepted.astype(jnp.int32)
            stats = _SliceStats(
                num_chains=jnp.array(0, dtype=jnp.int32),
                num_transitions=jnp.array(num_mcmc_steps, dtype=jnp.int32),
                num_accepted=jnp.sum(accepted),
                num_step_out=jnp.sum(infos.num_steps.astype(jnp.int32)),
                num_shrink=jnp.sum(infos.num_shrink.astype(jnp.int32)),
                num_stale_chains=jnp.array(0, dtype=jnp.int32),
                num_retries=jnp.array(0, dtype=jnp.int32),
            )
            return final_p, stats, jnp.any(infos.is_accepted)

        keys = jax.random.split(device_key, chains_per_device)
        return jax.vmap(one_chain)(keys, p0_device)

    @jax.jit
    def _finish(state, dead_idx, dead_p, new_p, chain_stats, accepted_any,
                retry_counts, cov_jitter_added, cov_condition,
                cov_regularized):
        particles = state.particles
        stats = _finalise_chain_stats(chain_stats, accepted_any, retry_counts)
        updated = jax.tree.map(
            lambda full, new: full.at[dead_idx].set(new), particles, new_p)
        new_cov = jnp.atleast_2d(
            jnp.cov(updated.position, ddof=0, rowvar=False))
        new_integrator = _update_integrator(
            state.integrator, updated, dead_p)
        info = _NSSInfo(
            stats=stats,
            max_cov_jitter=cov_jitter_added,
            max_cov_condition=cov_condition,
            num_cov_regularized=cov_regularized.astype(jnp.int32),
        )
        return _NSSState(updated, new_integrator, new_cov), _DeadInfo(dead_p, info)

    def _attempt(state, rng_key):
        dead_idx, dead_p, logL_0, start_p, mcmc_keys = _prepare(
            state, rng_key)
        start_p = jax.tree.map(
            lambda x: x.reshape(
                (n_devices, chains_per_device) + x.shape[1:]),
            start_p)
        cov, cov_jitter_added, cov_condition, cov_regularized = (
            _regularize_covariance(state.cov, cov_jitter))
        new_p, chain_stats, accepted_any = _run_chains(
            mcmc_keys, start_p, logL_0, cov)
        new_p = jax.tree.map(
            lambda x: x.reshape((total_num_delete,) + x.shape[2:]),
            new_p)
        chain_stats = jax.tree.map(
            lambda x: x.reshape((total_num_delete,) + x.shape[2:]),
            chain_stats)
        accepted_any = accepted_any.reshape((total_num_delete,))
        return (dead_idx, dead_p, new_p, chain_stats, accepted_any,
                cov_jitter_added, cov_condition,
                cov_regularized.astype(jnp.int32))

    def _step(state, rng_key):
        rng_key, attempt_key = jax.random.split(rng_key)
        (dead_idx, dead_p, new_p, chain_stats, accepted_any,
         cov_jitter_added, cov_condition, cov_regularized) = _attempt(
             state, attempt_key)
        retry_counts = jnp.zeros(accepted_any.shape, dtype=jnp.int32)

        for _ in range(stale_retries):
            if not _has_stale_chains(accepted_any):
                break
            rng_key, retry_key = jax.random.split(rng_key)
            (_, _, retry_new_p, retry_stats, retry_accepted,
             _, _, _) = _attempt(state, retry_key)
            new_p, chain_stats, accepted_any, retry_counts = (
                _merge_retry_attempt(
                    new_p, chain_stats, accepted_any, retry_counts,
                    retry_new_p, retry_stats, retry_accepted))

        return _finish(state, dead_idx, dead_p, new_p, chain_stats,
                       accepted_any, retry_counts, cov_jitter_added,
                       cov_condition, cov_regularized)

    return _step, chains_per_device


def _resolve_devices(devices_arg):
    """Return local devices for NSS chain sharding, or one device fallback."""
    devices = list(jax.local_devices())
    if not devices:
        return []

    if devices_arg is None:
        devices_arg = "auto"
    if isinstance(devices_arg, str):
        value = devices_arg.strip().lower()
        if value in ("auto", ""):
            non_cpu = [d for d in devices if d.platform != "cpu"]
            return non_cpu if len(non_cpu) > 1 else devices[:1]
        if value in ("1", "one", "single", "false", "off", "none", "no"):
            return devices[:1]
        try:
            n_requested = int(value)
        except ValueError as exc:
            raise ValueError(
                "`devices` must be 'auto', 1, or an integer.") from exc
    else:
        n_requested = int(devices_arg)

    if n_requested < 1:
        raise ValueError("`devices` must be at least 1.")
    if n_requested == 1:
        return devices[:1]
    n_available = len(devices)
    if n_available < n_requested:
        fprint(f"NSS: requested {n_requested} local devices but only "
               f"{n_available} visible; using {n_available}.")
    return devices[:min(n_requested, n_available)]


def _finalise(state, dead_list):
    """Concatenate all dead particles with the final live particles."""
    all_p = [d.particles for d in dead_list] + [state.particles]
    merged = jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0), *all_p)
    return _DeadInfo(merged)


def _compute_num_live(dead_info):
    """Effective n_live at each death event (birth/death counting).

    Handles batch deletions and the logL_birth=nan of initial live points
    identically to blackjax.ns.utils.compute_num_live.
    """
    ll_birth = dead_info.particles.logL_birth
    ll_death = dead_info.particles.loglikelihood
    n = ll_death.shape[0]

    birth_ev = jnp.column_stack([ll_birth, jnp.ones(n, dtype=jnp.int32)])
    death_ev = jnp.column_stack([ll_death, -jnp.ones(n, dtype=jnp.int32)])
    combined = jnp.concatenate([birth_ev, death_ev], axis=0)

    logL_col = combined[:, 0]
    n_col = combined[:, 1].astype(jnp.int32)
    not_nan = (~jnp.isnan(logL_col)).astype(jnp.int32)

    # lexsort: NaN births first, then by logL, then by event type
    sorted_idx = jnp.lexsort((n_col, logL_col, not_nan))
    sorted_n = n_col[sorted_idx]
    cumsum = jnp.maximum(jnp.cumsum(sorted_n), 0)
    death_mask = sorted_n == -1
    return cumsum[death_mask] + 1


def _log_weights(rng_key, dead_info, n_compress=100):
    """Log importance weights via stochastic prior-volume compression.

    Matches blackjax.ns.utils.log_weights exactly (beta=1).
    Returns array of shape (n_total, n_compress).
    """
    ll = dead_info.particles.loglikelihood
    sort_idx = jnp.argsort(ll)
    unsort_idx = jnp.empty_like(sort_idx).at[sort_idx].set(
        jnp.arange(len(sort_idx)))

    sorted_info = _DeadInfo(jax.tree.map(
        lambda x: x[sort_idx], dead_info.particles))
    num_live = _compute_num_live(sorted_info)

    rng_key, subkey = jax.random.split(rng_key)
    # Sample -Exp(1) directly to avoid log(0) in float32.
    r = -jax.random.exponential(
        subkey, (len(ll), n_compress))
    t = r / num_live[:, jnp.newaxis]
    logX = jnp.cumsum(t, axis=0)

    # log(dX_i) = log((X_{i-1} - X_{i+1}) / 2)
    logXp = jnp.concatenate(
        [jnp.zeros((1, n_compress)), logX[:-1]], axis=0)
    logXm = jnp.concatenate(
        [logX[1:], jnp.full((1, n_compress), -jnp.inf)],
        axis=0)
    # Float32 cumsum rounding can produce log_diff > 0.
    # _log1mexp(x > 0) returns NaN; clip to prevent this.
    log_diff = jnp.minimum(logXm - logXp, 0.0)
    logdX = _log1mexp(log_diff) + logXp - jnp.log(2.0)

    ll_s = sorted_info.particles.loglikelihood
    log_w = logdX + ll_s[:, jnp.newaxis]
    return log_w[unsort_idx]


def _summarise_nss_infos(dead_list):
    """Aggregate per-iteration NSS diagnostics from dead-point records."""
    infos = [d.info for d in dead_list if getattr(d, "info", None) is not None]
    if not infos:
        return {
            "num_transitions": 0,
            "num_chains": 0,
            "num_accepted": 0,
            "num_step_out": 0,
            "num_shrink": 0,
            "num_slice_evals": 0,
            "num_stale_chains": 0,
            "num_retries": 0,
            "acceptance_rate": np.nan,
            "stale_fraction": np.nan,
            "mean_step_out": np.nan,
            "mean_shrink": np.nan,
            "mean_slice_evals": np.nan,
            "max_cov_jitter": 0.0,
            "max_cov_condition": np.nan,
            "num_cov_regularized": 0,
        }

    def _int(field):
        return int(sum(np.asarray(getattr(info.stats, field)).item()
                       for info in infos))

    num_transitions = _int("num_transitions")
    num_chains = _int("num_chains")
    num_accepted = _int("num_accepted")
    num_step_out = _int("num_step_out")
    num_shrink = _int("num_shrink")
    num_slice_evals = num_step_out + num_shrink
    num_stale = _int("num_stale_chains")
    num_retries = _int("num_retries")
    max_cov_jitter = max(float(np.asarray(info.max_cov_jitter).item())
                         for info in infos)
    max_cov_condition = max(float(np.asarray(info.max_cov_condition).item())
                            for info in infos)
    num_cov_regularized = int(sum(
        np.asarray(info.num_cov_regularized).item() for info in infos))

    return {
        "num_transitions": num_transitions,
        "num_chains": num_chains,
        "num_accepted": num_accepted,
        "num_step_out": num_step_out,
        "num_shrink": num_shrink,
        "num_slice_evals": num_slice_evals,
        "num_stale_chains": num_stale,
        "num_retries": num_retries,
        "acceptance_rate": (num_accepted / num_transitions
                            if num_transitions else np.nan),
        "stale_fraction": (num_stale / num_chains if num_chains else np.nan),
        "mean_step_out": (num_step_out / num_transitions
                          if num_transitions else np.nan),
        "mean_shrink": (num_shrink / num_transitions
                        if num_transitions else np.nan),
        "mean_slice_evals": (num_slice_evals / num_transitions
                             if num_transitions else np.nan),
        "max_cov_jitter": max_cov_jitter,
        "max_cov_condition": max_cov_condition,
        "num_cov_regularized": num_cov_regularized,
    }


def _nss_info_from_summary(summary):
    stats = _SliceStats(
        jnp.array(summary.get("num_chains", 0), dtype=jnp.int32),
        jnp.array(summary.get("num_transitions", 0), dtype=jnp.int32),
        jnp.array(summary.get("num_accepted", 0), dtype=jnp.int32),
        jnp.array(summary.get("num_step_out", 0), dtype=jnp.int32),
        jnp.array(summary.get("num_shrink", 0), dtype=jnp.int32),
        jnp.array(summary.get("num_stale_chains", 0), dtype=jnp.int32),
        jnp.array(summary.get("num_retries", 0), dtype=jnp.int32),
    )
    return _NSSInfo(
        stats=stats,
        max_cov_jitter=jnp.array(summary.get("max_cov_jitter", 0.0)),
        max_cov_condition=jnp.array(summary.get("max_cov_condition", np.nan)),
        num_cov_regularized=jnp.array(
            summary.get("num_cov_regularized", 0), dtype=jnp.int32),
    )


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def _save_nss_checkpoint(path, state, dead, rng_key, n_dead):
    p = state.particles
    d_all = jax.tree.map(
        lambda *xs: np.asarray(jnp.concatenate(xs, axis=0)),
        *[d.particles for d in dead]) if dead else None
    data = dict(
        pos=np.asarray(p.position),
        logprior=np.asarray(p.logprior),
        loglikelihood=np.asarray(p.loglikelihood),
        logL_birth=np.asarray(p.logL_birth),
        logX=np.asarray(state.integrator.logX),
        logZ=np.asarray(state.integrator.logZ),
        logZ_live=np.asarray(state.integrator.logZ_live),
        cov=np.asarray(state.cov),
        rng_key=np.asarray(rng_key),
        n_dead=np.array(n_dead),
    )
    diagnostics = _summarise_nss_infos(dead)
    for key, value in diagnostics.items():
        if key in ("acceptance_rate", "stale_fraction",
                   "mean_step_out", "mean_shrink", "num_slice_evals",
                   "mean_slice_evals"):
            continue
        data[f"diag_{key}"] = np.asarray(value)
    if d_all is not None:
        data["dead_pos"] = np.asarray(d_all.position)
        data["dead_logprior"] = np.asarray(d_all.logprior)
        data["dead_loglikelihood"] = np.asarray(d_all.loglikelihood)
        data["dead_logL_birth"] = np.asarray(d_all.logL_birth)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **data)
    os.replace(tmp, path)


def _load_nss_checkpoint(path):
    d = np.load(path)
    particles = _Particle(
        position=jnp.array(d["pos"]),
        logprior=jnp.array(d["logprior"]),
        loglikelihood=jnp.array(d["loglikelihood"]),
        logL_birth=jnp.array(d["logL_birth"]),
    )
    integrator = _Integrator(
        logX=jnp.array(d["logX"]),
        logZ=jnp.array(d["logZ"]),
        logZ_live=jnp.array(d["logZ_live"]),
    )
    state = _NSSState(particles, integrator, jnp.array(d["cov"]))
    rng_key = jnp.array(d["rng_key"])
    n_dead = int(d["n_dead"])
    dead = []
    if "dead_pos" in d:
        diag_keys = (
            "num_chains", "num_transitions", "num_accepted", "num_step_out",
            "num_shrink", "num_stale_chains", "num_retries",
            "max_cov_jitter", "max_cov_condition", "num_cov_regularized",
        )
        diagnostics = {
            key: d[f"diag_{key}"].item()
            for key in diag_keys if f"diag_{key}" in d
        }
        info = (_nss_info_from_summary(diagnostics)
                if diagnostics else None)
        dead_p = _Particle(
            position=jnp.array(d["dead_pos"]),
            logprior=jnp.array(d["dead_logprior"]),
            loglikelihood=jnp.array(d["dead_loglikelihood"]),
            logL_birth=jnp.array(d["dead_logL_birth"]),
        )
        dead = [_DeadInfo(dead_p, info)]
    return state, dead, rng_key, n_dead


# ── Public API ───────────────────────────────────────────────────────────────

def run_nss(model, model_args=(), model_kwargs=None,
            n_live=500, num_mcmc_steps=50, num_delete=1,
            termination=-3, seed=42, validate=True,
            checkpoint_dir=None, checkpoint_path=None, resume_path=None,
            checkpoint_interval=900, devices="auto", max_steps=10,
            max_shrinkage=100, stale_retries=1, cov_jitter=1e-6):
    """Run the Nested Slice Sampler on a NumPyro model.

    Recommended settings (Yallup+2026, arXiv:2601.23252):
      num_mcmc_steps=ndim, total deletions per iteration near n_live//10.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    model_args, model_kwargs : tuple, dict
        Arguments passed to the model.
    n_live : int
        Number of live points.
    num_mcmc_steps : int
        Number of HRSS steps per dead point (p=d recommended).
    num_delete : int
        Deletion batch size on one device. In multi-device mode, this is the
        number of replacement chains run on each device; dead points are still
        selected globally, so the total deletion count is
        ``num_delete * n_devices``.
    termination : float
        Termination threshold for ``log(Z_live / Z_dead)``.
    seed : int
        Random seed.
    validate : bool
        If True, validate the prior/likelihood decomposition first.
    checkpoint_path : str or None
        Explicit checkpoint ``.npz`` path. Required to enable checkpointing.
    checkpoint_dir : str or None
        Deprecated; checkpointing now requires ``checkpoint_path``.
    resume_path : str or None
        Path to a checkpoint ``.npz`` to resume from. Resuming restores live
        points, dead points, the evidence integrator, RNG key, and dead count.
    checkpoint_interval : float
        Minimum time in seconds between periodic checkpoint writes.
    devices : {"auto", int}
        Number of visible local devices to use for replacement-chain
        parallelism. ``"auto"`` uses all non-CPU local devices only when more
        than one is visible; otherwise the original single-device path is used.
    max_steps : int
        Maximum stepping-out steps on each side of one HRSS transition.
    max_shrinkage : int
        Maximum shrinkage proposals in one HRSS transition.
    stale_retries : int
        Number of extra full replacement-chain attempts when an entire chain
        produces no accepted HRSS transition.
    cov_jitter : float
        Relative eigenvalue floor used to regularise the live-point covariance
        before drawing HRSS directions. Set to 0 to recover the old behaviour.

    Returns
    -------
    samples : dict
        Weighted posterior samples resampled to equal weights.
        Same format as MCMC: ``{name: array(n_eff,)}`` for scalars.
        Contains ``__nested__`` metadata with ``log_Z``, ``log_Z_err``,
        ``n_eff``, particle counts, run time, names, and sizes.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if max_steps < 1:
        raise ValueError("`max_steps` must be at least 1.")
    if max_shrinkage < 1:
        raise ValueError("`max_shrinkage` must be at least 1.")
    if stale_retries < 0:
        raise ValueError("`stale_retries` must be non-negative.")
    if cov_jitter < 0:
        raise ValueError("`cov_jitter` must be non-negative.")

    # Decompose model into prior + likelihood
    (log_prior_fn, log_likelihood_fn, log_joint_fn,
     names, sizes, lo, hi, dists) = decompose_model(
        model, model_args, model_kwargs, seed)

    if validate:
        validate_decomposition(
            log_prior_fn, log_likelihood_fn, log_joint_fn, lo, hi)

    ndim = sum(sizes)
    if num_mcmc_steps is None:
        num_mcmc_steps = ndim

    local_devices = _resolve_devices(devices)
    num_delete_per_device = num_delete
    total_num_delete = num_delete
    if len(local_devices) > 1:
        total_num_delete = _adjust_num_delete_for_devices(
            num_delete_per_device, len(local_devices), n_live)
        if total_num_delete != num_delete_per_device:
            fprint(
                f"NSS: using num_delete={num_delete_per_device}/device "
                f"across {len(local_devices)} local devices "
                f"({total_num_delete} total).")

    if len(local_devices) > 1:
        fprint(f"NSS: ndim={ndim}, num_mcmc_steps={num_mcmc_steps}, "
               f"n_live={n_live}, num_delete_total={total_num_delete}")
    else:
        fprint(f"NSS: ndim={ndim}, num_mcmc_steps={num_mcmc_steps}, "
               f"n_live={n_live}, num_delete={total_num_delete}")
    fprint(f"NSS: max_steps={max_steps}, max_shrinkage={max_shrinkage}, "
           f"stale_retries={stale_retries}, cov_jitter={cov_jitter:g}")

    use_pmap = len(local_devices) > 1 and total_num_delete > 1
    if use_pmap:
        step_fn, chains_per_device = _make_nss_step_pmap(
            log_prior_fn, log_likelihood_fn, total_num_delete,
            num_mcmc_steps, local_devices, max_steps=max_steps,
            max_shrinkage=max_shrinkage, stale_retries=stale_retries,
            cov_jitter=cov_jitter)
        fprint("NSS: replacement chains sharded over "
               f"{len(local_devices)} local devices "
               f"({chains_per_device}/device).")
    else:
        if len(local_devices) > 1 and total_num_delete <= 1:
            fprint("NSS: multiple devices visible but num_delete <= 1; "
                   "using single-device step.")

        step_fn = _make_nss_step(
            log_prior_fn, log_likelihood_fn, total_num_delete,
            num_mcmc_steps, max_steps=max_steps,
            max_shrinkage=max_shrinkage, stale_retries=stale_retries,
            cov_jitter=cov_jitter)

    if checkpoint_dir is not None and checkpoint_path is None:
        raise ValueError(
            "`checkpoint_path` must be set explicitly; `checkpoint_dir` "
            "does not imply a checkpoint filename.")
    _ckpt_path = checkpoint_path

    if resume_path is not None:
        state, dead, rng_key, n_dead = _load_nss_checkpoint(resume_path)
        fprint(f"Resumed from {resume_path} (n_dead={n_dead}, "
               f"logZ={float(state.integrator.logZ):.2f})")
        # Warmup JIT with restored state
        fprint("NSS: compiling/warming replacement step...", flush=True)
        rng_key, subkey = jax.random.split(rng_key)
        _warmup_state, _ = jax.block_until_ready(step_fn(state, subkey))
        del _warmup_state
        fprint("NSS: warmup complete; entering sampling loop", flush=True)
    else:
        # ---- Draw initial live points from prior ----
        initial_samples = sample_prior(dists, names, sizes, n_live, seed)

        # ---- Batched init (avoids GPU OOM) ----
        init_fn = partial(
            _make_particle, logprior_fn=log_prior_fn,
            loglikelihood_fn=log_likelihood_fn,
            logL_birth=jnp.nan)
        batched_fn = jax.jit(jax.vmap(init_fn))

        all_results = []
        for i in tqdm.tqdm(range(0, n_live, total_num_delete),
                           desc="init live pts", unit=" batch"):
            batch = initial_samples[i:i + total_num_delete]
            result = jax.block_until_ready(batched_fn(batch))
            result = jax.tree.map(lambda x: np.asarray(x), result)
            all_results.append(result)

        particles = jax.tree.map(
            lambda *xs: jnp.concatenate(xs, axis=0), *all_results)

        integrator = _init_integrator(particles)
        cov = jnp.atleast_2d(
            jnp.cov(particles.position, ddof=0, rowvar=False))
        state = _NSSState(particles, integrator, cov)
        rng_key = jax.random.PRNGKey(seed + 1)

        if _ckpt_path is not None:
            _save_nss_checkpoint(_ckpt_path, state, [], rng_key, 0)

        # ---- Warmup JIT ----
        fprint("NSS: compiling/warming replacement step...", flush=True)
        rng_key, subkey = jax.random.split(rng_key)
        _warmup_state, _ = jax.block_until_ready(step_fn(state, subkey))
        del _warmup_state
        fprint("NSS: warmup complete; entering sampling loop", flush=True)

        dead = []
        n_dead = 0
    _last_ckpt_time = timer()
    t0 = timer()

    with tqdm.tqdm(desc="NSS", unit=" pts", initial=n_dead) as pbar:
        while not (state.integrator.logZ_live
                   - state.integrator.logZ < termination):
            rng_key, subkey = jax.random.split(rng_key)
            state, dead_info = step_fn(state, subkey)
            dead.append(dead_info)
            n_dead += total_num_delete

            logZ = float(state.integrator.logZ)
            gap = float(state.integrator.logZ_live) - logZ

            pbar.set_postfix({"logZ": f"{logZ:.2f}",
                              "gap":  f"{gap:.2f}"})
            pbar.update(total_num_delete)

            if (_ckpt_path is not None
                    and timer() - _last_ckpt_time >= checkpoint_interval):
                _save_nss_checkpoint(
                    _ckpt_path, state, dead, rng_key, n_dead)
                _last_ckpt_time = timer()

            # Non-finite likelihood checks
            ll_live = state.particles.loglikelihood
            ll_dead = dead_info.particles.loglikelihood
            if not jnp.all(jnp.isfinite(ll_live)):
                n_bad = int((~jnp.isfinite(ll_live)).sum())
                print(f"\n*** WARNING: {n_bad} non-finite live "
                      f"logL at {n_dead} dead ***", flush=True)
                bad_idx = jnp.where(~jnp.isfinite(ll_live))[0]
                for idx in bad_idx[:3]:
                    pos = state.particles.position[idx]
                    print(f"  idx={int(idx)}: logL={float(ll_live[idx]):.4f}"
                          f"  params={np.asarray(pos)}", flush=True)
            if not jnp.all(jnp.isfinite(ll_dead)):
                n_bad = int((~jnp.isfinite(ll_dead)).sum())
                print(f"\n*** WARNING: {n_bad} non-finite dead "
                      f"logL at {n_dead} dead ***", flush=True)
                for j in range(min(n_bad, 3)):
                    idx = jnp.where(~jnp.isfinite(ll_dead))[0][j]
                    pos = dead_info.particles.position[idx]
                    print(f"  idx={int(idx)}: logL={float(ll_dead[idx]):.4f}"
                          f"  params={np.asarray(pos)}", flush=True)
            if not jnp.isfinite(logZ):
                print(f"\n*** WARNING: logZ={logZ} at {n_dead} dead "
                      f"but all logL finite! ***", flush=True)
                print(f"  max live logL: "
                      f"{float(ll_live.max()):.2f}",
                      flush=True)
                print(f"  dead logL: "
                      f"{np.asarray(ll_dead)}",
                      flush=True)
                break

    dt = timer() - t0

    # ---- Finalise and compute evidence ----
    final = _finalise(state, dead)
    diagnostics = _summarise_nss_infos(dead)
    logw = _log_weights(rng_key, final)
    # Average weights over stochastic compression realisations:
    # log(mean_j(w_ij)) = logsumexp(logw, -1) - log(n_compress)
    logw_mean = (jax.scipy.special.logsumexp(logw, axis=-1)
                 - jnp.log(logw.shape[-1]))

    minimum = jnp.nan_to_num(logw).min()
    logzs = jax.scipy.special.logsumexp(
        jnp.nan_to_num(logw, nan=minimum), axis=0)
    log_Z = float(logzs.mean())
    log_Z_err = float(logzs.std())

    # Resample to equal-weight posterior samples
    positions = final.particles.position
    # NaN-safe: replace NaN weights with -inf
    logw_mean_safe = jnp.where(
        jnp.isfinite(logw_mean), logw_mean, -jnp.inf)
    logw_norm = (logw_mean_safe
                 - jax.scipy.special.logsumexp(logw_mean_safe))
    logw_norm = jnp.where(
        jnp.isfinite(logw_norm), logw_norm, -jnp.inf)
    p = jnp.exp(logw_norm)
    p = jnp.where(jnp.isfinite(p), p, 0.0)
    p = p / p.sum()
    n_eff = int(jnp.exp(
        -jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0))))
    key2 = jax.random.PRNGKey(seed + 2)
    indices = jax.random.choice(
        key2, positions.shape[0],
        shape=(max(n_eff, 100),), p=p)
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
        "n_live": int(state.particles.position.shape[0]),
        "n_dead": int(n_dead),
        "n_total": int(positions.shape[0]),
        "num_delete_total": int(total_num_delete),
        "time": dt,
        "names": names,
        "sizes": sizes,
        "diagnostics": diagnostics,
        "final_logZ_live_gap": float(state.integrator.logZ_live
                                     - state.integrator.logZ),
        "max_steps": int(max_steps),
        "max_shrinkage": int(max_shrinkage),
        "stale_retries": int(stale_retries),
        "cov_jitter": float(cov_jitter),
    }

    return samples


def print_nested_summary(samples, meta=None):
    """Print a summary table for nested sampling posterior samples."""
    if meta is None:
        meta = samples.get("__nested__")

    if meta is not None:
        print(f"\nlog Z = {meta['log_Z']:.2f} +/- {meta['log_Z_err']:.2f}")
        print(f"n_eff = {meta['n_eff']}")
        if "n_total" in meta:
            print(f"n_total = {meta['n_total']} "
                  f"(dead={meta.get('n_dead', 'unknown')}, "
                  f"live={meta.get('n_live', 'unknown')})")
        diagnostics = meta.get("diagnostics", {})
        if diagnostics:
            print("HRSS diagnostics: "
                  f"accept={diagnostics.get('acceptance_rate', np.nan):.3f}, "
                  f"stale={diagnostics.get('stale_fraction', np.nan):.3f}, "
                  f"mean_evals="
                  f"{diagnostics.get('mean_slice_evals', np.nan):.2f}, "
                  f"mean_shrink={diagnostics.get('mean_shrink', np.nan):.2f}, "
                  f"cov_reg={diagnostics.get('num_cov_regularized', 0)}")

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
