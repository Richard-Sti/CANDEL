# Copyright (C) 2025 Richard Stiskalek
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
"""Nested sampling interface for NumPyro models via NSS.

Decomposes a NumPyro model into separate log-prior and log-likelihood
callables, then runs the NSS nested slice sampler.
"""
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.distributions import Delta, Unit
from numpyro.infer.initialization import init_to_median
from numpyro.infer.util import log_density

from .optimise import _prior_bounds


def _get_param_info(model, model_args, model_kwargs, seed=42):
    """Extract free parameter names, sizes, bounds, and distributions.

    Returns (names, sizes, lo, hi, dists) where dists is a dict mapping
    parameter names to their numpyro distribution objects.
    """
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
        """Check all parameters are within their prior support bounds."""
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
            # factor() sites: observed Unit, log-factor stored in log_prob
            if (v["type"] == "sample"
                    and v.get("is_observed", False)
                    and isinstance(v["fn"], Unit)):
                ll = ll + jnp.sum(v["fn"].log_prob(v["value"]))
        # Enforce prior support: NSS slice sampler checks loglikelihood but
        # not log_prior for acceptance. Without this, the sampler explores
        # outside the prior support, biasing the evidence by ~O(d) nats.
        return jnp.where(_in_support(x), ll, -jnp.inf)

    # Build joint for cross-validation
    def log_joint_fn(x):
        params = _unflatten(x, names, sizes)
        ld, _ = log_density(model, model_args, model_kwargs, params)
        return ld

    return log_prior_fn, log_likelihood_fn, log_joint_fn, names, sizes, lo, hi, dists


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
                f"joint={lj:.4f} (diff={diff:.6f}, tol={atol + rtol * scale:.6f})")

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

    print(f"Decomposition validated at {n_test} points (atol={atol}, rtol={rtol})")


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
        from nss.ns import run_nested_sampling
        from blackjax.ns.utils import log_weights
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

    (log_prior_fn, log_likelihood_fn, log_joint_fn,
     names, sizes, lo, hi, dists) = decompose_model(
        model, model_args, model_kwargs, seed)

    if validate:
        validate_decomposition(
            log_prior_fn, log_likelihood_fn, log_joint_fn, lo, hi)

    # Draw initial live points from prior
    initial_samples = sample_prior(dists, names, sizes, n_live, seed)

    key = jax.random.PRNGKey(seed + 1)
    final_state, res = run_nested_sampling(
        key, log_likelihood_fn, log_prior_fn,
        num_mcmc_steps=num_mcmc_steps,
        initial_samples=initial_samples,
        num_delete=num_delete,
        termination=termination,
    )
    print(res)

    # Extract weighted samples
    key2 = jax.random.PRNGKey(seed + 2)
    logw = log_weights(key2, final_state)
    logw_mean = logw.mean(axis=-1)

    # Positions from finalised state (dead + live)
    positions = final_state.particles.position

    # Filter out NaN weights (can happen with large num_delete)
    valid = jnp.isfinite(logw_mean)
    logw_mean = jnp.where(valid, logw_mean, -jnp.inf)

    # Resample to equal-weight samples (like MCMC output)
    logw_norm = logw_mean - jax.scipy.special.logsumexp(logw_mean)
    p = jnp.exp(logw_norm)
    p = jnp.where(jnp.isfinite(p), p, 0.0)
    p = p / p.sum()
    n_eff = int(jnp.exp(-jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0))))
    key3 = jax.random.PRNGKey(seed + 3)
    indices = jax.random.choice(
        key3, positions.shape[0], shape=(max(n_eff, 100),), p=p)
    resampled = positions[indices]

    # Unpack into named dict (same format as MCMC samples)
    samples = {}
    offset = 0
    for name, size in zip(names, sizes):
        s = resampled[:, offset:offset + size]
        samples[name] = s.squeeze(axis=-1) if size == 1 else s
        offset += size

    # Attach metadata as attributes (accessible but not breaking MCMC compat)
    samples["__nested__"] = {
        "log_Z": float(res.logZs.mean()),
        "log_Z_err": float(res.logZs.std()),
        "log_weights": logw_mean,
        "n_eff": n_eff,
        "results": res,
        "names": names,
        "sizes": sizes,
    }

    return samples
