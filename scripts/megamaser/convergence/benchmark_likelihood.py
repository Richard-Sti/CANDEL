"""Benchmark maser likelihood evaluation (forward + gradient).

Times the functions that the production samplers call:
  Mode 1 (NUTS): potential_fn and value_and_grad(potential_fn)
  Mode 2 (NSS):  log_likelihood_fn (forward only)

Usage:
  python benchmark_likelihood.py                        # all galaxies
  python benchmark_likelihood.py --galaxies NGC4258     # single galaxy
  python benchmark_likelihood.py --n-repeats 200        # more samples
"""
import argparse
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np
import tomli

from candel.model.maser_convergence import build_model, resolve_grid_for_galaxy

CONFIG_PATH = "scripts/megamaser/config_maser.toml"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _prepare_init(model, gcfg):
    """Build init dict from config, removing non-sampled params."""
    init = gcfg["init"]
    params = {k: jnp.asarray(v) for k, v in init.items()}
    for k in ("H0", "sigma_pec", "r_ang"):
        params.pop(k, None)
    if not model.use_ecc:
        for k in ("e_x", "e_y", "ecc", "periapsis", "dperiapsis_dr"):
            params.pop(k, None)
    if not model.use_quadratic_warp:
        for k in ("d2i_dr2", "d2Omega_dr2"):
            params.pop(k, None)
    return params


def _estimate_r_ang(model, gcfg, init_params):
    """Data-driven per-spot r_ang estimate for Mode 1 init."""
    from candel.model.model_H0_maser import radius_from_los_velocity

    D_c = float(init_params["D_c"])
    D_A = D_c
    r_lo, r_hi = model.r_ang_range(D_A)
    r_lo, r_hi = float(r_lo), float(r_hi)

    eta = float(init_params["eta"])
    M_BH = 10.0 ** (eta + np.log10(D_A) - 7.0)
    sin_i = abs(np.sin(np.deg2rad(float(init_params["i0"]))))

    r_est = np.full(model.n_spots, np.sqrt(r_lo * r_hi))

    is_hv = np.asarray(model.is_highvel, dtype=bool)
    v_obs = np.asarray(model._all_v)
    dv_sys = float(init_params.get("dv_sys", jnp.array(0.0)))
    v_sys = float(gcfg["v_sys_obs"]) + dv_sys
    dv = np.maximum(np.abs(v_obs - v_sys), 1.0)
    r_vel = np.asarray(radius_from_los_velocity(dv, sin_i, D_A, M_BH))
    r_est[is_hv] = np.clip(r_vel[is_hv], r_lo * 1.01, r_hi * 0.99)

    x0 = float(init_params.get("x0", jnp.array(0.0)))
    y0 = float(init_params.get("y0", jnp.array(0.0)))
    x_obs = np.asarray(model._all_x)
    y_obs = np.asarray(model._all_y)
    r_pos = np.sqrt((x_obs - x0)**2 + (y_obs - y0)**2) / 1e3
    r_pos = np.maximum(r_pos, r_lo * 1.01)
    r_est[~is_hv] = np.clip(r_pos[~is_hv], r_lo * 1.01, r_hi * 0.99)

    return jnp.asarray(r_est)


def _time_fn(jit_fn, arg, n_warmup, n_repeats):
    """JIT-compile, warm up, then time n_repeats calls.

    Returns (jit_compile_time_s, times_ms).
    """
    t0 = time.perf_counter()
    jax.block_until_ready(jit_fn(arg))
    jit_time = time.perf_counter() - t0

    for _ in range(n_warmup):
        jax.block_until_ready(jit_fn(arg))

    times = np.empty(n_repeats)
    for i in range(n_repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(jit_fn(arg))
        times[i] = (time.perf_counter() - t0) * 1e3
    return jit_time, times


def _print_timing(label, jit_time, times):
    print(f"  JIT: {jit_time:.2f}s")
    print(f"  {label}: {np.median(times):.3f} ms  "
          f"(mean {np.mean(times):.3f} ± {np.std(times):.3f})",
          flush=True)


def _gpu_mem_MiB():
    """Current and peak GPU memory in MiB, or (nan, nan)."""
    try:
        stats = jax.devices()[0].memory_stats()
        if stats is None:
            return float('nan'), float('nan')
        return (stats["bytes_in_use"] / 2**20,
                stats["peak_bytes_in_use"] / 2**20)
    except Exception:
        return float('nan'), float('nan')


# -------------------------------------------------------------------
# Per-mode benchmarks
# -------------------------------------------------------------------

def _bench_mode1(model, galaxy, gcfg, master_cfg, grid, args):
    from numpyro.infer import init_to_value
    from numpyro.infer.util import initialize_model

    init_params = _prepare_init(model, gcfg)
    init_params["r_ang"] = _estimate_r_ang(model, gcfg, init_params)

    n_global = len([k for k in init_params if k != "r_ang"])
    print(f"Params: {n_global} global + r_ang[{model.n_spots}]", flush=True)

    print("Initializing model...", flush=True)
    model_info = initialize_model(
        jax.random.PRNGKey(42), model,
        init_strategy=init_to_value(values=init_params))
    z = model_info.param_info.z
    potential_fn = model_info.potential_fn

    ndim = sum(v.size for v in jax.tree.leaves(z))
    pe = float(potential_fn(z))
    print(f"Unconstrained dim: {ndim}")
    print(f"Init potential: {pe:.10f} (logp = {-pe:.10f})", flush=True)
    if not np.isfinite(pe):
        print("  WARNING: non-finite potential", flush=True)

    # Forward
    jit_fwd = jax.jit(potential_fn)
    print("\nForward (potential_fn)...", flush=True)
    jit_fwd_t, t_fwd = _time_fn(jit_fwd, z, args.n_warmup, args.n_repeats)
    _print_timing("Forward", jit_fwd_t, t_fwd)

    # Gradient
    jit_vg = jax.jit(jax.value_and_grad(potential_fn))
    print("\nGradient (value_and_grad)...", flush=True)
    jit_grad_t, t_grad = _time_fn(jit_vg, z, args.n_warmup, args.n_repeats)
    _print_timing("Grad", jit_grad_t, t_grad)

    mem_used, mem_peak = _gpu_mem_MiB()
    print(f"\nGPU memory: {mem_used:.0f} MiB in use, "
          f"{mem_peak:.0f} MiB peak", flush=True)

    return dict(galaxy=galaxy, mode="mode1", n_spots=model.n_spots,
                ndim=ndim, grid=grid, logp=-pe,
                jit_fwd=jit_fwd_t, jit_grad=jit_grad_t,
                fwd_ms=t_fwd, grad_ms=t_grad,
                mem_used=mem_used, mem_peak=mem_peak)


def _bench_mode2(model, galaxy, gcfg, grid, args):
    from candel.inference.nested import decompose_model

    print("Decomposing model...", flush=True)
    (_, log_likelihood_fn, _, names, sizes, lo, hi, _) = decompose_model(
        model, (), {}, 42)
    ndim = sum(sizes)
    print(f"Params: {ndim} ({', '.join(names)})", flush=True)

    # Construct flat initial vector from config init
    init = gcfg["init"]
    x_parts = []
    offset = 0
    for name, size in zip(names, sizes):
        if name in init:
            val = jnp.atleast_1d(jnp.asarray(init[name]))
            x_parts.append(val.ravel()[:size])
        else:
            mid = (jnp.asarray(lo[offset:offset + size])
                   + jnp.asarray(hi[offset:offset + size])) / 2
            x_parts.append(mid)
            print(f"  WARNING: '{name}' not in init, using midpoint",
                  flush=True)
        offset += size
    x0 = jnp.concatenate(x_parts)

    ll = float(log_likelihood_fn(x0))
    print(f"Init log-likelihood: {ll:.10f}", flush=True)
    if not np.isfinite(ll):
        print("  WARNING: non-finite log-likelihood", flush=True)

    # Forward only
    jit_ll = jax.jit(log_likelihood_fn)
    print("\nForward (log_likelihood_fn)...", flush=True)
    jit_fwd_t, t_fwd = _time_fn(jit_ll, x0, args.n_warmup, args.n_repeats)
    _print_timing("Forward", jit_fwd_t, t_fwd)

    mem_used, mem_peak = _gpu_mem_MiB()
    print(f"\nGPU memory: {mem_used:.0f} MiB in use, "
          f"{mem_peak:.0f} MiB peak", flush=True)

    return dict(galaxy=galaxy, mode="mode2", n_spots=model.n_spots,
                ndim=ndim, grid=grid, logp=ll,
                jit_fwd=jit_fwd_t, jit_grad=None,
                fwd_ms=t_fwd, grad_ms=None,
                mem_used=mem_used, mem_peak=mem_peak)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def benchmark_galaxy(galaxy, master_cfg, args):
    galaxies_cfg = master_cfg["model"]["galaxies"]
    gcfg = galaxies_cfg[galaxy]
    mode = (args.mode
            or gcfg.get("mode", master_cfg["model"].get("mode", "mode2")))

    print(f"\n{'=' * 60}")
    print(f"{galaxy}  (mode={mode}, {args.n_repeats} repeats)")
    print(f"{'=' * 60}", flush=True)

    grid = resolve_grid_for_galaxy(master_cfg, galaxy, mode)
    overrides = dict(mode=mode,
                     n_phi_hv_high=grid["n_hv_high"],
                     n_phi_hv_low=grid["n_hv_low"],
                     n_phi_sys=grid["n_sys"],
                     n_r_global=grid["n_r_global"])
    if args.spot_batch > 0:
        overrides["mode2_spot_batch"] = args.spot_batch
    model = build_model(galaxy, master_cfg, **overrides)

    print(f"Spots: {model.n_spots}")
    grid_str = f"phi=({grid['n_hv_high']}, {grid['n_hv_low']}, {grid['n_sys']})"
    if mode == "mode2":
        grid_str += f", r=({grid['n_r_local']}, {grid['n_r_global']})"
    print(f"Grid: {grid_str}", flush=True)

    if mode == "mode1":
        return _bench_mode1(model, galaxy, gcfg, master_cfg, grid, args)
    return _bench_mode2(model, galaxy, gcfg, grid, args)


def main():
    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)
    all_galaxies = list(master_cfg["model"]["galaxies"].keys())

    ap = argparse.ArgumentParser(
        description="Benchmark maser likelihood evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available galaxies: {', '.join(all_galaxies)}")
    ap.add_argument("--galaxies", nargs="+", default=None,
                    help="Galaxies to benchmark (default: all)")
    ap.add_argument("--n-repeats", type=int, default=100,
                    help="Timed iterations (default: 100)")
    ap.add_argument("--n-warmup", type=int, default=5,
                    help="Extra warm-up calls after JIT (default: 5)")
    ap.add_argument("--spot-batch", type=int, default=0,
                    help="Spot-axis chunk size for Mode 2 (0=off, default: 0)")
    ap.add_argument("--f64", action="store_true",
                    help="Use float64 (default: float32)")
    ap.add_argument("--mode", choices=["mode1", "mode2"], default=None,
                    help="Force mode (default: use config)")
    args = ap.parse_args()

    if args.f64:
        jax.config.update("jax_enable_x64", True)
    precision = "f64" if args.f64 else "f32"
    print(f"JAX backend: {jax.default_backend()}, {precision}", flush=True)

    galaxies = args.galaxies if args.galaxies is not None else all_galaxies
    print(f"Galaxies: {galaxies}")
    print(f"Repeats: {args.n_repeats}, warmup: {args.n_warmup}, "
          f"spot_batch: {args.spot_batch or 'off'}", flush=True)

    results = []
    for galaxy in galaxies:
        try:
            results.append(benchmark_galaxy(galaxy, master_cfg, args))
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(f"\n  SKIPPED: GPU OOM — run this galaxy alone or "
                      f"reduce --spot-batch", flush=True)
                results.append(dict(
                    galaxy=galaxy, mode="—", n_spots=0, ndim=0, grid={},
                    logp=float('nan'), jit_fwd=0, jit_grad=None,
                    fwd_ms=np.array([np.nan]), grad_ms=None,
                    mem_used=float('nan'), mem_peak=float('nan')))
            else:
                raise
        jax.clear_caches()
        gc.collect()

    # Summary table
    W = 120
    print(f"\n{'=' * W}")
    print(f"Summary ({precision})")
    print(f"{'=' * W}")
    hdr = (f"{'Galaxy':<14} {'Mode':<6} {'Spots':>5} {'Dim':>5} "
           f"{'logp':>16} {'Forward (ms)':>16} {'Grad (ms)':>16} "
           f"{'GPU (MiB)':>10} {'JIT fwd':>8} {'JIT grad':>9}")
    print(hdr)
    print("-" * W)
    for r in results:
        if np.isnan(r["fwd_ms"]).all():
            print(f"{r['galaxy']:<14} {'OOM':<6}")
            continue
        fwd = f"{np.median(r['fwd_ms']):.3f} ± {np.std(r['fwd_ms']):.3f}"
        if r["grad_ms"] is not None:
            grad = (f"{np.median(r['grad_ms']):.3f} ± "
                    f"{np.std(r['grad_ms']):.3f}")
            jit_g = f"{r['jit_grad']:.1f}s"
        else:
            grad = "—"
            jit_g = "—"
        mem = (f"{r['mem_peak']:.0f}" if np.isfinite(r["mem_peak"])
               else "?")
        logp = f"{r['logp']:.4f}" if np.isfinite(r.get("logp", np.nan)) else "?"
        print(f"{r['galaxy']:<14} {r['mode']:<6} {r['n_spots']:>5} "
              f"{r['ndim']:>5} {logp:>16} {fwd:>16} {grad:>16} "
              f"{mem:>10} {r['jit_fwd']:>7.1f}s {jit_g:>9}")
    print(f"{'=' * W}", flush=True)


if __name__ == "__main__":
    main()
