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
Yallup+2025 (arXiv:2601.23252) using hit-and-run slice sampling.

Algorithm matches the handley-lab blackjax fork exactly:
  - Same stepping-out/shrinking slice kernel (Neal 2003)
  - Same Mahalanobis-normalised direction proposal
  - Same covariance update (empirical, ddof=0) from live points
  - Same evidence integrator (logX, logZ, logZ_live)
  - Same stochastic prior-volume compression for log_weights
"""
from functools import partial
from timeit import default_timer as timer
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.distributions import Delta, Unit
from numpyro.infer.initialization import init_to_median
from numpyro.infer.util import log_density

from ..util import fprint
from .optimise import _prior_bounds


# ── Model decomposition (unchanged) ─────────────────────────────────────────

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


# ── Self-contained Nested Slice Sampler ─────────────────────────────────────
#
# Faithful reimplementation of the handley-lab blackjax NSS fork.
# References:
#   Yallup+2025  arXiv:2601.23252   (NSS algorithm)
#   Neal (2003)  Ann. Stat. 31(3)   (slice sampling)
#   Skilling (2006)                 (nested sampling evidence integral)

class _Particle(NamedTuple):
    position:      jax.Array   # (ndim,)
    logprior:      jax.Array   # scalar — log prior density (= "logdensity" in blackjax)
    loglikelihood: jax.Array   # scalar
    logL_birth:    jax.Array   # scalar — logL threshold at birth (nan for initial pts)


class _Integrator(NamedTuple):
    logX:      jax.Array   # log prior volume remaining
    logZ:      jax.Array   # accumulated log evidence from dead points
    logZ_live: jax.Array   # live-point estimate of remaining log evidence


class _NSSState(NamedTuple):
    particles:  _Particle    # each field has leading axis (n_live,)
    integrator: _Integrator
    cov:        jax.Array    # (ndim, ndim) empirical live-point covariance


class _DeadInfo(NamedTuple):
    particles: _Particle     # leading axis: (num_delete,) per step, (n_total,) after finalise


def _logmeanexp(x):
    n = jnp.array(x.shape[0], dtype=x.dtype)
    return jax.scipy.special.logsumexp(x) - jnp.log(n)


def _log1mexp(x):
    """Numerically stable log(1 - exp(x)) for x < 0."""
    return jnp.where(x > -0.6931472,          # threshold ≈ log(2)
                     jnp.log(-jnp.expm1(x)),
                     jnp.log1p(-jnp.exp(x)))


def _init_integrator(particles):
    """Initialise evidence integrator from initial live points."""
    dtype = particles.loglikelihood.dtype
    logX = jnp.array(0.0, dtype=dtype)
    logZ = jnp.array(-jnp.inf, dtype=dtype)
    logZ_live = _logmeanexp(particles.loglikelihood) + logX
    return _Integrator(logX, logZ, logZ_live)


def _update_integrator(integrator, live_particles, dead_particles):
    """Update evidence integrator after one NS step.

    Treats batch deletion as sequential: when the j-th dead point
    (j=0..num_delete-1) is removed, there are n_live-j live points.
    """
    ll_live = live_particles.loglikelihood
    ll_dead = dead_particles.loglikelihood
    n_live  = ll_live.shape[0]
    n_dead  = ll_dead.shape[0]
    dtype   = ll_live.dtype

    num_live   = jnp.arange(n_live, n_live - n_dead, -1, dtype=dtype)
    delta_logX = -1.0 / num_live
    logX       = integrator.logX + jnp.cumsum(delta_logX)
    log_dX     = logX + jnp.log(1.0 - jnp.exp(delta_logX))
    log_dZ     = ll_dead + log_dX

    logZ      = jnp.logaddexp(integrator.logZ,
                              jax.scipy.special.logsumexp(log_dZ))
    logZ_live = _logmeanexp(ll_live) + logX[-1]
    return _Integrator(logX[-1], logZ, logZ_live)


def _make_particle(position, logprior_fn, loglikelihood_fn, logL_birth):
    """Evaluate a position and wrap it into a _Particle."""
    lp    = logprior_fn(position)
    ll    = loglikelihood_fn(position)
    birth = logL_birth * jnp.ones_like(ll)
    return _Particle(position, lp, ll, birth)


def _sample_direction(rng_key, position, cov):
    """Sample a Mahalanobis-normalised direction from N(0, cov), scaled by 2.

    Matches blackjax.mcmc.ss.sample_direction_from_covariance exactly.
    """
    ndim = position.shape[0]
    d    = jax.random.multivariate_normal(
        rng_key, mean=jnp.zeros(ndim, dtype=cov.dtype), cov=cov)
    invcov = jnp.linalg.inv(cov)
    norm   = jnp.sqrt(jnp.einsum("i,ij,j", d, invcov, d))
    return d / norm * 2.0


def _hrss_step(rng_key, particle, logprior_fn, loglikelihood_fn,
               logL_0, cov, max_steps=10, max_shrinkage=100):
    """One constrained hit-and-run slice sampling step (Neal 2003).

    Accepts a new position iff:
      logprior(x_new) >= logprior(x) + log(u)   [vertical slice on prior]
      loglikelihood(x_new) > logL_0              [NS likelihood contour]

    Parameters
    ----------
    rng_key
    particle   : _Particle — current live point
    logL_0     : scalar — current likelihood threshold
    cov        : (ndim, ndim) — covariance for direction proposal
    max_steps  : int — max stepping-out steps per side (default 10)
    max_shrinkage : int — max shrink iterations (default 100)

    Returns
    -------
    _Particle — accepted new particle (unchanged if no accepted sample found)
    """
    vs_key, hs_key, dir_key = jax.random.split(rng_key, 3)

    d = _sample_direction(dir_key, particle.position, cov)

    # Vertical slice: fix a log-prior level
    u        = jax.random.uniform(vs_key)
    logslice = particle.logprior + jnp.log(u)

    def _eval(t):
        x     = particle.position + t * d
        new_p = _make_particle(x, logprior_fn, loglikelihood_fn, logL_0)
        ok    = (new_p.logprior >= logslice) & (new_p.loglikelihood > logL_0)
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

    j, _, left,  _ = jax.lax.while_loop(
        _step_cond, _step_body, (j + 1, jnp.array(-1.0), 1.0 - u2, True))
    k, _, right, _ = jax.lax.while_loop(
        _step_cond, _step_body, (k + 1, jnp.array( 1.0),      -u2, True))

    # Shrinking: uniform sample within [left, right], shrink on rejection
    def _shrink_body(carry):
        n, key, lo, hi, p, _ = carry
        key, subkey = jax.random.split(key)
        t     = jax.random.uniform(subkey, minval=lo, maxval=hi)
        new_p, is_acc = _eval(t)
        lo = jnp.where(t < 0.0, t, lo)
        hi = jnp.where(t > 0.0, t, hi)
        return n + 1, key, lo, hi, new_p, is_acc

    def _shrink_cond(carry):
        n, _, _, _, _, is_acc = carry
        return ~is_acc & (n < max_shrinkage)

    _, _, _, _, new_p, is_acc = jax.lax.while_loop(
        _shrink_cond, _shrink_body,
        (0, rng_key, left, right, particle, False))

    # If shrinkage found nothing, stay put
    return jax.tree.map(
        lambda new, old: jnp.where(is_acc, new, old), new_p, particle)


def _nss_step(rng_key, state, logprior_fn, loglikelihood_fn,
              num_delete, num_mcmc_steps, max_steps=10, max_shrinkage=100):
    """One NSS iteration: remove num_delete worst live points, replace above L*.

    Matches blackjax.nss exactly:
      - delete_fn   : top_k(-logL, num_delete)
      - L*          : max(dead loglikelihoods)
      - start pts   : uniform over survivors with logL > L*
      - MCMC        : num_mcmc_steps HRSS steps via lax.scan, take last
      - cov update  : jnp.cov(positions, ddof=0, rowvar=False) from new live pts
      - integrator  : updated after replacement
    """
    particles = state.particles

    # Identify num_delete worst live points
    _, dead_idx = jax.lax.top_k(-particles.loglikelihood, num_delete)
    dead_p      = jax.tree.map(lambda x: x[dead_idx], particles)
    logL_0      = dead_p.loglikelihood.max()

    # Select starting particles uniformly from survivors (logL > L*)
    choice_key, mcmc_key = jax.random.split(rng_key)
    w = (particles.loglikelihood > logL_0).astype(jnp.float32)
    w = jnp.where(w.sum() > 0.0, w, jnp.ones_like(w))
    start_idx = jax.random.choice(
        choice_key, particles.loglikelihood.shape[0],
        shape=(num_delete,), p=w / w.sum(), replace=True)
    start_p = jax.tree.map(lambda x: x[start_idx], particles)

    # Run num_mcmc_steps HRSS steps for each replacement particle (take last)
    def one_chain(rng_key, p0):
        def body(p, k):
            return _hrss_step(k, p, logprior_fn, loglikelihood_fn,
                              logL_0, state.cov, max_steps, max_shrinkage), None
        keys     = jax.random.split(rng_key, num_mcmc_steps)
        final_p, _ = jax.lax.scan(body, p0, keys)
        return final_p

    new_p = jax.vmap(one_chain)(jax.random.split(mcmc_key, num_delete), start_p)

    # Replace dead positions
    updated = jax.tree.map(
        lambda full, new: full.at[dead_idx].set(new), particles, new_p)

    # Update covariance and integrator
    new_cov        = jnp.atleast_2d(jnp.cov(updated.position, ddof=0, rowvar=False))
    new_integrator = _update_integrator(state.integrator, updated, dead_p)

    return _NSSState(updated, new_integrator, new_cov), _DeadInfo(dead_p)


def _finalise(state, dead_list):
    """Concatenate all dead particles with the final live particles."""
    all_p  = [d.particles for d in dead_list] + [state.particles]
    merged = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *all_p)
    return _DeadInfo(merged)


def _compute_num_live(dead_info):
    """Effective n_live at each death event (birth/death counting).

    Handles batch deletions and the logL_birth=nan of initial live points
    identically to blackjax.ns.utils.compute_num_live.
    """
    ll_birth = dead_info.particles.logL_birth
    ll_death = dead_info.particles.loglikelihood
    n = ll_death.shape[0]

    birth_ev = jnp.column_stack([ll_birth,  jnp.ones(n, dtype=jnp.int32)])
    death_ev = jnp.column_stack([ll_death, -jnp.ones(n, dtype=jnp.int32)])
    combined = jnp.concatenate([birth_ev, death_ev], axis=0)

    logL_col = combined[:, 0]
    n_col    = combined[:, 1].astype(jnp.int32)
    not_nan  = (~jnp.isnan(logL_col)).astype(jnp.int32)

    # lexsort: NaN births come first (not_nan=0), then sort by logL, then by event type
    sorted_idx = jnp.lexsort((n_col, logL_col, not_nan))
    sorted_n   = n_col[sorted_idx]
    cumsum     = jnp.maximum(jnp.cumsum(sorted_n), 0)
    death_mask = sorted_n == -1
    return cumsum[death_mask] + 1


def _log_weights(rng_key, dead_info, n_compress=100):
    """Log importance weights via stochastic prior-volume compression.

    Matches blackjax.ns.utils.log_weights exactly (beta=1).
    Returns array of shape (n_total, n_compress).
    """
    ll       = dead_info.particles.loglikelihood
    sort_idx = jnp.argsort(ll)
    unsort_idx = jnp.empty_like(sort_idx).at[sort_idx].set(jnp.arange(len(sort_idx)))

    sorted_info = _DeadInfo(jax.tree.map(lambda x: x[sort_idx], dead_info.particles))
    num_live    = _compute_num_live(sorted_info)

    rng_key, subkey = jax.random.split(rng_key)
    # log(1-u) for u~Uniform(0,1) equals -Exp(1) in distribution.
    # Sample exponential directly to avoid log(0) when u rounds to 1.0 in float32.
    r    = -jax.random.exponential(subkey, (len(ll), n_compress))
    t    = r / num_live[:, jnp.newaxis]
    logX = jnp.cumsum(t, axis=0)

    # log(dX_i) = log((X_{i-1} - X_{i+1}) / 2)
    logXp  = jnp.concatenate([jnp.zeros((1, n_compress)), logX[:-1]], axis=0)
    logXm  = jnp.concatenate([logX[1:], jnp.full((1, n_compress), -jnp.inf)], axis=0)
    # Float32 cumsum rounding over O(10^5) steps can produce log_diff > 0.
    # _log1mexp(x > 0) returns NaN; clip to prevent this.
    log_diff = jnp.minimum(logXm - logXp, 0.0)
    logdX  = _log1mexp(log_diff) + logXp - jnp.log(2.0)

    ll_s   = sorted_info.particles.loglikelihood
    log_w  = logdX + ll_s[:, jnp.newaxis]
    return log_w[unsort_idx]


# ── Public API ───────────────────────────────────────────────────────────────

def run_nss(model, model_args=(), model_kwargs=None,
            n_live=500, num_mcmc_steps=50, num_delete=1,
            termination=-3, seed=42, validate=True):
    """Run the Nested Slice Sampler on a NumPyro model.

    Recommended settings (Yallup+2025, arXiv:2601.23252):
      num_mcmc_steps=ndim, num_delete=n_live//10.

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
        Weighted posterior samples resampled to equal weights.
        Same format as MCMC: ``{name: array(n_eff,)}`` for scalars.
        Contains ``__nested__`` key with metadata (log_Z, log_Z_err, n_eff).
    """
    import tqdm

    if model_kwargs is None:
        model_kwargs = {}

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

    fprint(f"NSS: ndim={ndim}, num_mcmc_steps={num_mcmc_steps}, "
           f"n_live={n_live}, num_delete={num_delete}")

    # ---- Draw initial live points from prior ----
    initial_samples = sample_prior(dists, names, sizes, n_live, seed)

    # ---- Batched initialisation (avoids GPU OOM, batch size = num_delete) ----
    init_fn    = partial(_make_particle, logprior_fn=log_prior_fn,
                         loglikelihood_fn=log_likelihood_fn,
                         logL_birth=jnp.nan)
    batched_fn = jax.jit(jax.vmap(init_fn))

    all_results = []
    for i in tqdm.tqdm(range(0, n_live, num_delete),
                       desc="init live pts", unit=" batch"):
        batch  = initial_samples[i:i + num_delete]
        result = jax.block_until_ready(batched_fn(batch))
        result = jax.tree.map(lambda x: np.asarray(x), result)
        all_results.append(result)

    particles = jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0), *all_results)

    integrator = _init_integrator(particles)
    cov        = jnp.atleast_2d(jnp.cov(particles.position, ddof=0, rowvar=False))
    state      = _NSSState(particles, integrator, cov)

    # ---- JIT the NSS step ----
    @jax.jit
    def step_fn(state, rng_key):
        return _nss_step(rng_key, state, log_prior_fn, log_likelihood_fn,
                         num_delete, num_mcmc_steps)

    # ---- Warmup JIT ----
    rng_key = jax.random.PRNGKey(seed + 1)
    rng_key, subkey = jax.random.split(rng_key)
    state, _ = jax.block_until_ready(step_fn(state, subkey))

    # ---- Main loop ----
    dead       = []
    t0         = timer()
    n_dead     = 0
    gap_prev   = None
    gap_rate_ema = None
    ema_alpha  = num_delete / n_live

    with tqdm.tqdm(desc="NSS", unit=" pts") as pbar:
        while not (state.integrator.logZ_live
                   - state.integrator.logZ < termination):
            rng_key, subkey = jax.random.split(rng_key)
            state, dead_info = step_fn(state, subkey)
            dead.append(dead_info)
            n_dead += num_delete

            logZ = float(state.integrator.logZ)
            gap  = float(state.integrator.logZ_live) - logZ

            if gap_prev is not None:
                d_gap        = (gap - gap_prev) / num_delete
                gap_rate_ema = (d_gap if gap_rate_ema is None
                                else ema_alpha * d_gap
                                + (1 - ema_alpha) * gap_rate_ema)
            gap_prev = gap

            # ETA
            eta_str    = "?"
            pts_per_sec = pbar.format_dict.get("rate") or 0
            if gap_rate_ema is not None and gap_rate_ema < 0 and pts_per_sec > 0:
                remaining = (gap - termination) / (-gap_rate_ema)
                eta_sec   = remaining / pts_per_sec
                eta_str   = (f"{eta_sec / 60:.0f}m" if eta_sec < 3600
                             else f"{eta_sec / 3600:.1f}h")

            pbar.set_postfix({"logZ": f"{logZ:.2f}",
                              "gap":  f"{gap:.2f}",
                              "ETA":  eta_str})
            pbar.update(num_delete)

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
                break
            if not jnp.all(jnp.isfinite(ll_dead)):
                n_bad = int((~jnp.isfinite(ll_dead)).sum())
                print(f"\n*** WARNING: {n_bad} non-finite dead "
                      f"logL at {n_dead} dead ***", flush=True)
                for j in range(min(n_bad, 3)):
                    idx = jnp.where(~jnp.isfinite(ll_dead))[0][j]
                    pos = dead_info.particles.position[idx]
                    print(f"  idx={int(idx)}: logL={float(ll_dead[idx]):.4f}"
                          f"  params={np.asarray(pos)}", flush=True)
                break
            if not jnp.isfinite(logZ):
                print(f"\n*** WARNING: logZ={logZ} at {n_dead} dead "
                      f"but all logL finite! ***", flush=True)
                print(f"  max live logL: {float(ll_live.max()):.2f}", flush=True)
                print(f"  dead logL: {np.asarray(ll_dead)}", flush=True)
                break

    dt = timer() - t0

    # ---- Finalise and compute evidence ----
    final = _finalise(state, dead)
    logw  = _log_weights(rng_key, final)
    # Average weights over stochastic compression realisations in linear space:
    # log(mean_j(w_ij)) = logsumexp(logw, axis=-1) - log(n_compress)
    logw_mean = jax.scipy.special.logsumexp(logw, axis=-1) - jnp.log(logw.shape[-1])

    minimum = jnp.nan_to_num(logw).min()
    logzs   = jax.scipy.special.logsumexp(
        jnp.nan_to_num(logw, nan=minimum), axis=0)
    log_Z     = float(logzs.mean())
    log_Z_err = float(logzs.std())

    # Resample to equal-weight posterior samples
    positions  = final.particles.position
    # NaN-safe: replace NaN weights with -inf (zero probability)
    logw_mean_safe = jnp.where(jnp.isfinite(logw_mean), logw_mean, -jnp.inf)
    logw_norm  = logw_mean_safe - jax.scipy.special.logsumexp(logw_mean_safe)
    logw_norm  = jnp.where(jnp.isfinite(logw_norm), logw_norm, -jnp.inf)
    p          = jnp.exp(logw_norm)
    p          = jnp.where(jnp.isfinite(p), p, 0.0)
    p          = p / p.sum()
    n_eff      = int(jnp.exp(-jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0))))
    key2       = jax.random.PRNGKey(seed + 2)
    indices    = jax.random.choice(
        key2, positions.shape[0], shape=(max(n_eff, 100),), p=p)
    resampled  = positions[indices]

    # Unpack into named dict (same format as MCMC samples)
    samples = {}
    offset  = 0
    for name, size in zip(names, sizes):
        s = resampled[:, offset:offset + size]
        samples[name] = s.squeeze(axis=-1) if size == 1 else s
        offset += size

    samples["__nested__"] = {
        "log_Z":     log_Z,
        "log_Z_err": log_Z_err,
        "n_eff":     n_eff,
        "time":      dt,
        "names":     names,
        "sizes":     sizes,
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
