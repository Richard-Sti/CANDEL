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
"""Sobol + Adam multi-start MAP optimizer for NumPyro models.

Strategy:
  1. Draw 2^N points from a scrambled Sobol sequence within prior bounds.
  2. Evaluate log-density at all points (vmapped, batched for memory).
  3. Select M best starts that are sufficiently distinct (L-inf distance).
  4. Run Adam from all M starts simultaneously (vmapped over starts)
     with cosine LR decay to find the MAP.
  5. Cluster final points to detect multiple modes.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpyro import handlers
from numpyro.infer.initialization import init_to_median
from numpyro.infer.util import log_density
from scipy.stats.qmc import Sobol
from tqdm import trange

from ..util import fprint, fsection

# -----------------------------------------------------------------------
# Bounds extraction
# -----------------------------------------------------------------------


def _prior_bounds(dist, sobol_n_sigma=5):
    """Derive finite optimizer bounds from a numpyro distribution.

    For Uniform priors, uses the support directly. For all other
    distributions (Normal, TruncatedNormal, HalfNormal, ...),
    uses mean +/- sobol_n_sigma * std clipped to any finite support bounds.
    This avoids wasting Sobol points in low-probability tails
    (e.g. TruncatedNormal(2, 5, 0, 100) -> [0, 27] not [0, 100]).

    Delta priors return (None, None) — caller should skip.
    """
    dist_name = dist.__class__.__name__
    if dist_name == "Delta":
        return None, None

    support = dist.support
    lb = getattr(support, "lower_bound", None)
    ub = getattr(support, "upper_bound", None)
    lb = float(lb) if lb is not None else -np.inf
    ub = float(ub) if ub is not None else np.inf

    # Uniform or no tightening requested: use support directly
    if dist_name == "Uniform" or sobol_n_sigma is None:
        return lb, ub

    # Try to tighten with mean +/- sobol_n_sigma * std.
    # Some distributions (e.g. TruncatedNormal) don't implement variance
    # but have a base_dist that does.
    try:
        mu = float(dist.mean)
        sigma = float(dist.variance ** 0.5)
    except (NotImplementedError, AttributeError, TypeError):
        base = getattr(dist, "base_dist", None)
        if base is not None:
            try:
                mu = float(base.mean)
                sigma = float(base.variance ** 0.5)
            except (NotImplementedError, AttributeError, TypeError):
                return lb, ub
        else:
            return lb, ub

    tight_lo = mu - sobol_n_sigma * sigma
    tight_hi = mu + sobol_n_sigma * sigma

    # Clip to support (if finite)
    lb = max(lb, tight_lo) if np.isfinite(lb) else tight_lo
    ub = min(ub, tight_hi) if np.isfinite(ub) else tight_hi
    return lb, ub


def _get_bounds_from_trace(model, model_args, model_kwargs, sobol_n_sigma=5,
                           seed=42):
    """Extract optimizer bounds from model priors.

    Uniform priors use their support. Others use mean +/- n_sigma * std
    clipped to support. Delta priors are skipped.
    """
    substituted = handlers.substitute(
        handlers.seed(model, rng_seed=seed),
        substitute_fn=init_to_median(num_samples=15),
    )
    tr = handlers.trace(substituted).get_trace(*model_args, **model_kwargs)

    names, lo, hi, sizes = [], [], [], []
    for k, v in tr.items():
        if v["type"] == "sample" and not v.get("is_observed", False):
            lb, ub = _prior_bounds(v["fn"], sobol_n_sigma=sobol_n_sigma)
            if lb is None:
                continue
            size = int(np.prod(v["value"].shape))
            lb = np.atleast_1d(np.asarray(lb))
            ub = np.atleast_1d(np.asarray(ub))
            names.append(k)
            lo.append(np.broadcast_to(lb.ravel(), size))
            hi.append(np.broadcast_to(ub.ravel(), size))
            sizes.append(size)

    return names, sizes, np.concatenate(lo), np.concatenate(hi)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _build_logp_flat(model, model_args, model_kwargs, names, sizes):
    """Build a flat-vector log-density function for the model."""
    def logp(x):
        params = {}
        offset = 0
        for name, size in zip(names, sizes):
            params[name] = x[offset:offset + size].reshape(()) \
                if size == 1 else x[offset:offset + size]
            offset += size
        ld, _ = log_density(model, model_args, model_kwargs, params)
        return ld
    return logp


def _select_distinct(points, logp_vals, M, min_dist_frac=0.01):
    """Select top-M points that are sufficiently distinct.

    Greedily picks the best point, then skips any candidate within
    min_dist_frac (L-inf normalised by range) of already selected points.
    Falls back to top-M if fewer than M distinct points found.
    """
    order = np.argsort(-logp_vals)
    selected = [order[0]]
    spread = np.max(points, axis=0) - np.min(points, axis=0)
    spread = np.where(spread > 0, spread, 1.0)

    for idx in order[1:]:
        if len(selected) >= M:
            break
        pt = points[idx]
        too_close = False
        for s in selected:
            if np.all(np.abs(pt - points[s]) / spread < min_dist_frac):
                too_close = True
                break
        if not too_close:
            selected.append(idx)

    if len(selected) < M:
        for idx in order:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= M:
                break

    return np.array(selected[:M])


def _cluster_modes(points, logp_vals, names, sizes, tol=0.02):
    """Cluster final optimizer points into modes.

    Two points belong to the same mode if their L-inf distance
    (normalised by parameter range) is < tol for all scalar params.

    Returns list of dicts, each with 'logp', 'count', 'best_idx',
    'params' (best point as dict), sorted by logP descending.
    """
    spread = np.max(points, axis=0) - np.min(points, axis=0)
    spread = np.where(spread > 0, spread, 1.0)

    order = np.argsort(-logp_vals)
    modes = []  # each: {'indices': [...], 'best_idx': int}

    for idx in order:
        matched = False
        for mode in modes:
            ref = points[mode["best_idx"]]
            if np.all(np.abs(points[idx] - ref) / spread < tol):
                mode["indices"].append(idx)
                matched = True
                break
        if not matched:
            modes.append({"indices": [idx], "best_idx": idx})

    result = []
    for mode in modes:
        best = mode["best_idx"]
        params = {}
        offset = 0
        for name, size in zip(names, sizes):
            val = points[best, offset:offset + size]
            params[name] = float(val) if size == 1 else val
            offset += size
        result.append({
            "logp": float(logp_vals[best]),
            "count": len(mode["indices"]),
            "best_idx": best,
            "params": params,
        })

    result.sort(key=lambda m: -m["logp"])
    return result


def _short_name(name, max_len=8):
    """Shorten parameter name for table display."""
    _ABBREV = {
        "sigma_x_floor": "sx_fl",
        "sigma_y_floor": "sy_fl",
        "sigma_v_sys": "sv_sys",
        "sigma_v_hv": "sv_hv",
        "sigma_a_floor": "sa_fl",
        "dOmega_dr": "dOm_dr",
        "Omega0": "Om0",
    }
    return _ABBREV.get(name, name)[:max_len]


def _auto_fmt(val):
    """Format a float with adaptive decimal places."""
    av = abs(val)
    if av == 0:
        return f"{val:>8.1f}"
    if av >= 100:
        return f"{val:>8.1f}"
    if av >= 1:
        return f"{val:>8.2f}"
    if av >= 0.01:
        return f"{val:>8.4f}"
    return f"{val:>8.1e}"


def _print_points_table(points, logp_vals, names, sizes,
                        max_params=20):
    """Print a table of M points with their logP values."""
    cols, col_offsets = [], []
    offset = 0
    for name, size in zip(names, sizes):
        if size == 1:
            cols.append(name)
            col_offsets.append(offset)
        offset += size
        if len(cols) >= max_params:
            break

    short = [_short_name(c) for c in cols]
    hdr = f"{'#':>3s} {'logP':>8s}"
    for s in short:
        hdr += f" {s:>8s}"
    fprint(hdr)
    fprint("-" * len(hdr))

    order = np.argsort(-logp_vals)
    for rank, idx in enumerate(order):
        row = f"{rank:3d} {logp_vals[idx]:8.1f}"
        for col_off in col_offsets:
            row += f" {_auto_fmt(points[idx, col_off])}"
        fprint(row)


def _print_modes(modes, names, sizes, max_params=20):
    """Print mode summary: one block per mode."""
    n_modes = len(modes)
    total = sum(m["count"] for m in modes)

    if n_modes == 1:
        fprint(f"unimodal: all {total} starts converged to the same mode")
    else:
        fprint(f"found {n_modes} modes from {total} starts")
        delta = modes[0]["logp"] - modes[1]["logp"]
        if delta < 5:
            fprint(f"WARNING: top 2 modes differ by only {delta:.1f} nats "
                   f"-- posterior may be multimodal")

    scalar_names = [n for n, s in zip(names, sizes) if s == 1][:max_params]

    for i, mode in enumerate(modes):
        fprint(f"\nmode {i}: logP = {mode['logp']:.2f} "
               f"({mode['count']}/{total} starts)")
        p = mode["params"]
        for name in scalar_names:
            if name in p and isinstance(p[name], (float, int)):
                fprint(f"  {_short_name(name, 15):15s}: "
                       f"{_auto_fmt(p[name])}")


# -----------------------------------------------------------------------
# Main optimizer
# -----------------------------------------------------------------------


def sobol_adam(model, model_args=(), model_kwargs=None,
               log2_N=14, M=10, n_steps=5000,
               lr=0.1, lr_end=0.005, n_restarts=3,
               sobol_n_sigma=1, sobol_batch=1024,
               min_dist_frac=0.01, seed=42, verbose=True):
    """Multi-start MAP optimizer: Sobol survey + parallel Adam.

    Uses cosine decay with warm restarts: the LR cycles from ``lr``
    down to ``lr_end`` a total of ``n_restarts`` times over
    ``n_steps``, allowing the optimizer to escape shallow local minima.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    model_args : tuple
        Positional arguments for the model.
    model_kwargs : dict or None
        Keyword arguments for the model.
    log2_N : int
        log2 of Sobol sample count (default 2^14 = 16384).
    M : int
        Number of Adam starts.
    n_steps : int
        Total Adam steps per start.
    lr : float
        Peak learning rate (at each restart).
    lr_end : float
        Minimum learning rate (at end of each cycle).
    n_restarts : int
        Number of cosine warm restart cycles.
    sobol_n_sigma : float
        For non-Uniform priors, Sobol bounds are mean +/- sobol_n_sigma * std
        (clipped to support). Use 1 for tight search, 5 for wide.
    sobol_batch : int
        Batch size for Sobol evaluation (memory control).
    min_dist_frac : float
        Minimum L-inf distance (as fraction of range) between starts.
    seed : int
        Random seed for Sobol scrambling.
    verbose : bool
        Print progress.

    Returns
    -------
    best_params : dict
        Best MAP parameter values.
    best_logp : float
        Log-density at the MAP.
    all_results : dict
        All M results: 'params' (M, D), 'logp' (M,), 'names', 'sizes',
        'modes' (list of mode dicts).
    """
    if model_kwargs is None:
        model_kwargs = {}

    # --- Extract bounds ---
    # Sobol bounds: tightened by sobol_n_sigma for non-Uniform priors
    names, sizes, lo_sobol, hi_sobol = _get_bounds_from_trace(
        model, model_args, model_kwargs,
        sobol_n_sigma=sobol_n_sigma, seed=seed)
    # Adam clip bounds: prior support, but use 10-sigma for unbounded
    _, _, lo_clip, hi_clip = _get_bounds_from_trace(
        model, model_args, model_kwargs, sobol_n_sigma=10, seed=seed)
    D = len(lo_sobol)
    N_sobol = 2**log2_N

    if verbose:
        fsection("Sobol + Adam MAP optimizer")
        fprint(f"{D}D, {N_sobol} Sobol points, {M} Adam starts "
               f"x {n_steps} steps")
        fprint(f"LR: {lr:.1e} -> {lr_end:.1e} "
               f"(cosine, {n_restarts} restarts)")
        if D <= 50:
            offset = 0
            for name, size in zip(names, sizes):
                s_lo, s_hi = lo_sobol[offset], hi_sobol[offset]
                c_lo, c_hi = lo_clip[offset], hi_clip[offset]
                line = f"  {name:20s}: Sobol [{s_lo:.4g}, {s_hi:.4g}]"
                if c_lo != s_lo or c_hi != s_hi:
                    line += f"  Adam [{c_lo:.4g}, {c_hi:.4g}]"
                fprint(line)
                offset += size

    # --- Build flat log-density ---
    logp_fn = _build_logp_flat(
        model, model_args, model_kwargs, names, sizes)
    logp_batch = jax.jit(jax.vmap(logp_fn))

    # --- Compile ---
    t0 = time.time()
    x_test = jnp.tile(
        jnp.array(0.5 * (lo_sobol + hi_sobol))[None, :], (4, 1))
    _ = logp_batch(x_test)
    jax.block_until_ready(_)
    if verbose:
        fprint(f"JIT compiled in {time.time() - t0:.1f}s")

    # --- Sobol survey ---
    sampler = Sobol(d=D, scramble=True, seed=seed)
    sobol_01 = sampler.random(N_sobol)
    sobol_points = lo_sobol + sobol_01 * (hi_sobol - lo_sobol)
    sobol_jax = jnp.array(sobol_points)

    t0 = time.time()
    logp_all = []
    n_batches = (N_sobol + sobol_batch - 1) // sobol_batch
    for i in trange(n_batches, desc="Sobol", disable=not verbose):
        start = i * sobol_batch
        end = min(start + sobol_batch, N_sobol)
        vals = logp_batch(sobol_jax[start:end])
        jax.block_until_ready(vals)
        logp_all.append(np.asarray(vals))
    logp_all = np.concatenate(logp_all)
    valid = np.isfinite(logp_all)
    logp_all = np.where(valid, logp_all, -np.inf)

    if verbose:
        fprint(f"Sobol done in {time.time() - t0:.1f}s "
               f"({valid.sum()}/{N_sobol} valid, "
               f"best logP = {logp_all[valid].max():.1f})")

    # --- Select M distinct starts ---
    selected = _select_distinct(sobol_points, logp_all, M, min_dist_frac)
    x0 = jnp.array(sobol_points[selected])

    if verbose:
        fsection("Starting points")
        _print_points_table(sobol_points[selected], logp_all[selected],
                            names, sizes)

    # --- Parallel Adam ---
    lo_jax = jnp.array(lo_clip)
    hi_jax = jnp.array(hi_clip)
    eps = 1e-6 * (hi_jax - lo_jax)

    # Cosine decay with warm restarts: n_restarts cycles over n_steps
    steps_per_cycle = n_steps // n_restarts
    boundaries = [steps_per_cycle * i for i in range(1, n_restarts)]
    schedules = [
        optax.cosine_decay_schedule(
            init_value=lr, decay_steps=steps_per_cycle,
            alpha=lr_end / lr)
        for _ in range(n_restarts)
    ]
    schedule = optax.join_schedules(schedules, boundaries)
    optimizer = optax.adam(schedule)

    @jax.jit
    def adam_step(x, opt_state):
        def _single(xi, osi):
            g = jax.grad(lambda z: -logp_fn(z))(xi)
            updates, new_osi = optimizer.update(g, osi)
            xi_new = optax.apply_updates(xi, updates)
            xi_new = jnp.clip(xi_new, lo_jax + eps, hi_jax - eps)
            return xi_new, new_osi
        return jax.vmap(_single)(x, opt_state)

    opt_state = jax.vmap(optimizer.init)(x0)

    # Compile
    t0 = time.time()
    x_cur, opt_state = adam_step(x0, opt_state)
    jax.block_until_ready(x_cur)
    if verbose:
        fprint(f"Adam JIT compiled in {time.time() - t0:.1f}s")

    # Run
    t0 = time.time()
    pbar = trange(1, n_steps, desc="Adam", disable=not verbose)
    best_logp_so_far = -np.inf
    for step in pbar:
        x_cur, opt_state = adam_step(x_cur, opt_state)
        if (step + 1) % 100 == 0:
            jax.block_until_ready(x_cur)
            lp_now = np.asarray(logp_batch(x_cur))
            best_logp_so_far = float(lp_now.max())
            cur_lr = float(schedule(step + 1))
            pbar.set_postfix(logP=f"{best_logp_so_far:.1f}",
                             lr=f"{cur_lr:.1e}")
    jax.block_until_ready(x_cur)
    x_final = np.asarray(x_cur)

    if verbose:
        fprint(f"Adam done in {time.time() - t0:.1f}s")

    # --- Evaluate final points and detect modes ---
    logp_final = np.asarray(logp_batch(jnp.array(x_final)))

    # Filter out points sitting at prior boundaries (Uniform priors).
    # For Uniform priors, lo_sobol == support lower bound. A point
    # within rtol of the boundary is likely a boundary artefact.
    rtol = 1e-3
    width = hi_sobol - lo_sobol
    at_lo = np.any(x_final <= lo_sobol + rtol * width, axis=1)
    at_hi = np.any(x_final >= hi_sobol - rtol * width, axis=1)
    at_boundary = at_lo | at_hi
    logp_filtered = np.where(at_boundary, -np.inf, logp_final)

    # Fall back to unfiltered if all points are at boundaries.
    if np.all(at_boundary):
        logp_filtered = logp_final
        if verbose:
            fprint("WARNING: all Adam points at prior boundaries, "
                   "using unfiltered logP")
    elif verbose and np.any(at_boundary):
        n_filt = int(at_boundary.sum())
        fprint(f"Filtered {n_filt}/{len(at_boundary)} boundary points")

    modes = _cluster_modes(x_final, logp_filtered, names, sizes)

    best_idx = np.argmax(logp_filtered)
    best_logp = float(logp_final[best_idx])
    best_params = modes[0]["params"]

    if verbose:
        fsection("Final points")
        _print_points_table(x_final, logp_final, names, sizes)
        fsection("Mode summary")
        _print_modes(modes, names, sizes)

    all_results = {
        "params": x_final,
        "logp": logp_final,
        "names": names,
        "sizes": sizes,
        "lo_sobol": lo_sobol,
        "hi_sobol": hi_sobol,
        "lo_clip": lo_clip,
        "hi_clip": hi_clip,
        "modes": modes,
    }

    return best_params, best_logp, all_results


# -----------------------------------------------------------------------
# Integration with run_H0_inference
# -----------------------------------------------------------------------


def find_MAP(model, model_kwargs=None, seed=42):
    """Find MAP estimate via Sobol + Adam.

    Drop-in replacement for ``find_initial_point``. Reads optimizer
    settings from ``model.config["optimise"]`` (optional).

    Config keys (all optional, under ``[optimise]``)::

        log2_N = 14
        M = 10
        n_steps = 5000
        lr = 0.1
        lr_end = 0.005
        n_restarts = 3
        sobol_n_sigma = 5
        sobol_batch = 1024
        min_dist_frac = 0.01

    Returns
    -------
    best_params : dict
        Constrained-space MAP parameter values, ready for
        ``init_to_value(values=best_params)``.
    """
    if model_kwargs is None:
        model_kwargs = {}

    opt_cfg = model.config.get("optimise", {})

    best_params, best_logp, results = sobol_adam(
        model,
        model_kwargs=model_kwargs,
        log2_N=opt_cfg.get("log2_N", 14),
        M=opt_cfg.get("M", 10),
        n_steps=opt_cfg.get("n_steps", 5000),
        lr=opt_cfg.get("lr", 0.1),
        lr_end=opt_cfg.get("lr_end", 0.005),
        n_restarts=opt_cfg.get("n_restarts", 3),
        sobol_n_sigma=opt_cfg.get("sobol_n_sigma", 1),
        sobol_batch=opt_cfg.get("sobol_batch", 128),
        min_dist_frac=opt_cfg.get("min_dist_frac", 0.01),
        seed=seed,
    )

    # Convert scalar floats to jnp arrays (matching find_initial_point)
    return {k: jnp.array(v) for k, v in best_params.items()}
