"""
Mode 2 (r + phi marginal) grid convergence test.

For each galaxy (Mode 2 union grid: per-spot sinh centred on the
posterior peak ∪ shared log-uniform global grid), build the model at
several (n_r_local, n_r_global, n_phi_hv_high, n_phi_hv_low, n_phi_sys)
settings and compare the production ll_disk against a float32
brute-force reference on a uniform (r-log, phi-full-2π) grid at the
config [init] block.

The reference is r-chunked so the per-batch intermediate fits on a
12 GB GPU.

Usage:
    python scripts/megamaser/convergence_grids.py [--galaxies ...]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import tomli

from candel import get_nested
from candel.model.maser_convergence import (
    bruteforce_ll_mode2, build_model, resolve_grid_for_galaxy)

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
ALL_GALAXIES = ["CGCG074-064", "NGC5765b", "NGC6264", "NGC6323", "UGC3789"]


def _phys_from_init(model, galaxies_cfg, galaxy):
    """Build (phys_args, phys_kw, diag) from the config init block by
    delegating to MaserDiskModel.phys_from_sample."""
    init = galaxies_cfg[galaxy]["init"]
    sample = {k: np.asarray(v) for k, v in init.items()}
    return model.phys_from_sample(sample)


# r-axis variations (paired n_r_local, n_r_global) tested while holding
# phi at the config default. Bracket the production grid (501, 301):
# one row below as a "production overkill?" check, one above as a
# convergence confirmation. Global-dominated configs and K_sigma > 10
# were tested previously and ruled out as inefficient — see git log.
R_EXTRAS = [
    (401, 201),
    (601, 301),
]

K_EXTRAS = []


def build_test_settings(default, K_def):
    """Build the Mode 2 sweep around the config default.

    Block A ("joint"): the config default itself.
    Block B ("r"):     vary (n_r_local, n_r_global), phi/K pinned at default.
    Block C ("K"):     vary K_sigma, r/phi pinned at default.

    The phi axis was previously swept here too; on the five MCP galaxies
    Δll proved insensitive to phi at production density (all rows bit-
    identical), and each row triggers a fresh XLA recompile because grid
    sizes are static shapes — so the phi block was dropped to save ~30 s
    of compile per row.
    """
    r_def = (default["n_r_local"], default["n_r_global"])
    p_def = (default["n_hv_high"], default["n_hv_low"], default["n_sys"])
    rows = [dict(tag="joint", n_r_local=r_def[0], n_r_global=r_def[1],
                 n_hv_high=p_def[0], n_hv_low=p_def[1], n_sys=p_def[2],
                 K_sigma=K_def)]
    r_set = sorted({*R_EXTRAS, r_def}, key=lambda x: (x[0], x[1]))
    for nl, ng in r_set:
        rows.append(dict(tag="r", n_r_local=nl, n_r_global=ng,
                         n_hv_high=p_def[0], n_hv_low=p_def[1],
                         n_sys=p_def[2], K_sigma=K_def))
    K_set = sorted({*K_EXTRAS, K_def})
    for K in K_set:
        if K == K_def:
            continue
        rows.append(dict(tag="K", n_r_local=r_def[0], n_r_global=r_def[1],
                         n_hv_high=p_def[0], n_hv_low=p_def[1],
                         n_sys=p_def[2], K_sigma=K))
    return rows


def _ll_test_array(m_t, phys_args, phys_kw, spot_batch):
    """Production-grid log-likelihood as a JAX scalar (no host transfer)."""
    groups = m_t._build_r_grids_mode2(
        phys_args[2], phys_args[3], phys_args[4], phys_args[16],
        phys_args[8], phys_args[15],
        phys_args=phys_args, phys_kw=phys_kw)
    ll = m_t._eval_phi_marginal(
        groups, phys_args, phys_kw, spot_batch=spot_batch)
    return jnp.sum(ll)


def _ll_test(m_t, phys_args, phys_kw, spot_batch):
    return float(_ll_test_array(m_t, phys_args, phys_kw, spot_batch))


def _time_production(m_prod, phys_args, phys_kw, spot_batch, attempts):
    """Time the production-grid likelihood under `jax.jit` — matches the
    numpyro NUTS path, which wraps the model density in jit.

    First call compiles + executes (counted as compile time); the next
    `attempts` calls hit the XLA cache and reflect true per-step cost.
    Without the jit, calls re-dispatch every primitive eagerly and the
    measured time is dominated by Python/XLA dispatch — irrelevant to
    NUTS wall-clock.
    """
    fn = jax.jit(
        lambda pa, pk: _ll_test_array(m_prod, pa, pk, spot_batch))

    t0 = time.perf_counter()
    out = fn(phys_args, phys_kw)
    out.block_until_ready()
    compile_time = time.perf_counter() - t0

    timings = []
    for _ in range(attempts):
        t0 = time.perf_counter()
        out = fn(phys_args, phys_kw)
        out.block_until_ready()
        timings.append(time.perf_counter() - t0)
    return compile_time, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxies", nargs="+", default=ALL_GALAXIES)
    parser.add_argument(
        "--spot-batch", type=int, default=16,
        help="Chunk size on the spot axis when evaluating the test-grid "
             "phi marginal. Lower if the largest (n_r_local+n_r_global, "
             "n_phi) combination OOMs on a small GPU (default: 16).")
    parser.add_argument(
        "--ref-spot-batch", type=int, default=None,
        help="Override [convergence.mode2_reference] spot_batch. Larger "
             "values cut JAX dispatch overhead in the brute-force "
             "reference at the cost of a (spot_batch · r_chunk · n_phi) "
             "× dtype live intermediate. Default: use config value.")
    parser.add_argument(
        "--timing-attempts", type=int, default=5,
        help="After the per-galaxy convergence sweep, time the production "
             "grid likelihood for this many post-compile attempts and "
             "report mean ± std / min / max. Set to 0 to skip (default: 5).")
    args = parser.parse_args()

    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_enable_x64", True)
    print(f"JAX platform: {jax.default_backend()}, precision: float64",
          flush=True)

    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)
    galaxies_cfg = master_cfg["model"]["galaxies"]
    ref_cfg = dict(master_cfg["convergence"]["mode2_reference"])
    if args.ref_spot_batch is not None:
        ref_cfg["spot_batch"] = int(args.ref_spot_batch)

    print("=" * 90)
    print("Mode 2 grid convergence")
    print(f"Reference (ll): {ref_cfg['n_r']} r (log-uniform) × "
          f"{ref_cfg['n_phi']} phi (uniform 2π), "
          f"r-chunk={ref_cfg['r_chunk']}, "
          f"spot_batch={ref_cfg['spot_batch']}, "
          f"dtype={ref_cfg['dtype']}")
    print(f"R_EXTRAS: {R_EXTRAS}")
    print(f"K_EXTRAS: {K_EXTRAS}")
    print("=" * 90)

    K_def = float(get_nested(master_cfg, "model/K_sigma", 10.0))

    summary = []
    for galaxy in args.galaxies:
        print(f"\n{'─' * 70}")
        print(f"Galaxy: {galaxy}")
        print(f"{'─' * 70}")
        # Per-galaxy production grid (per-galaxy block → generic [model]).
        default = resolve_grid_for_galaxy(master_cfg, galaxy, "mode2")
        test_settings = build_test_settings(default, K_def)
        print(f"  production grid: n_r_local={default['n_r_local']}, "
              f"n_r_global={default['n_r_global']}, "
              f"n_phi=({default['n_hv_high']}, {default['n_hv_low']}, "
              f"{default['n_sys']}), K_sigma={K_def}")
        print(f"  test settings: {len(test_settings)} combinations")
        model_ref = build_model(galaxy, master_cfg, mode="mode2")
        phys_args, phys_kw, diag = _phys_from_init(
            model_ref, galaxies_cfg, galaxy)
        print(f"  D_A={diag['D_A']:.2f} Mpc, n_spots={model_ref.n_spots}")

        print("  Computing brute-force reference ll...", flush=True)
        ll_ref = bruteforce_ll_mode2(model_ref, phys_args, phys_kw, ref_cfg)
        print(f"  Reference ll_disk = {ll_ref:.4f}")

        hdr = (f"  {'tag':>5} {'nr_loc':>7} {'nr_glb':>7} {'K':>5} "
               f"{'nhv_hi':>7} {'nhv_lo':>7} {'n_sys':>7}  "
               f"{'ll_disk':>12}  {'Δll':>9}")
        print(f"\n{hdr}")
        best_delta = float("inf")
        best_setting = None
        for setting in test_settings:
            m_t = build_model(galaxy, master_cfg, mode="mode2",
                              n_r_local=setting["n_r_local"],
                              n_r_global=setting["n_r_global"],
                              n_phi_hv_high=setting["n_hv_high"],
                              n_phi_hv_low=setting["n_hv_low"],
                              n_phi_sys=setting["n_sys"],
                              K_sigma=setting["K_sigma"])
            ll_total = _ll_test(m_t, phys_args, phys_kw, args.spot_batch)
            delta = ll_total - ll_ref

            row = dict(galaxy=galaxy, **setting,
                       ll=ll_total, delta=delta,
                       n_spots=model_ref.n_spots)
            line = (f"  {setting['tag']:>5} {setting['n_r_local']:>7d} "
                    f"{setting['n_r_global']:>7d} "
                    f"{setting['K_sigma']:>5.1f} "
                    f"{setting['n_hv_high']:>7d} "
                    f"{setting['n_hv_low']:>7d} {setting['n_sys']:>7d}  "
                    f"{ll_total:12.4f}  {delta:+9.4f}")
            print(line)
            summary.append(row)
            if abs(delta) < abs(best_delta):
                best_delta = delta
                best_setting = setting
        print(f"  Best Δll: {best_setting} Δ={best_delta:+.4f}")

        if args.timing_attempts > 0:
            print(f"  Production-grid timing (jit, n={args.timing_attempts}, "
                  f"spot_batch={args.spot_batch}):")
            for refine in (True, False):
                m_prod = build_model(galaxy, master_cfg, mode="mode2",
                                     refine_r_center=refine)
                compile_time, timings = _time_production(
                    m_prod, phys_args, phys_kw,
                    args.spot_batch, args.timing_attempts)
                t = np.asarray(timings)
                tag = "refine=on " if refine else "refine=off"
                print(f"    {tag}: compile {compile_time*1e3:.1f} ms, "
                      f"run {t.mean()*1e3:.2f} ± {t.std()*1e3:.2f} ms "
                      f"(min {t.min()*1e3:.2f}, max {t.max()*1e3:.2f})")

    # --------------------------------------------------------------
    # Summary table + reasonable-values picker.
    # --------------------------------------------------------------
    print(f"\n{'=' * 110}")
    print("SUMMARY — Δ ll_disk (cfg − brute-force ref, nats)")
    print(f"{'=' * 110}")
    header = (f"{'Galaxy':<14} {'tag':>5} {'nr_loc':>7} {'nr_glb':>7} "
              f"{'K':>5} {'nhv_hi':>7} {'nhv_lo':>7} {'n_sys':>7}  "
              f"{'ll_disk':>12}  {'Δll':>9}")
    print(header)
    print("-" * len(header))
    for r in summary:
        line = (f"{r['galaxy']:<14} {r['tag']:>5} {r['n_r_local']:>7d} "
                f"{r['n_r_global']:>7d} {r['K_sigma']:>5.1f} "
                f"{r['n_hv_high']:>7d} "
                f"{r['n_hv_low']:>7d} {r['n_sys']:>7d}  "
                f"{r['ll']:12.4f}  {r['delta']:+9.4f}")
        print(line)

    # Axis isolation: at the finest anchor, the joint Δ matches both
    # the r-sweep and phi-sweep finest rows. The residual at coarser
    # settings in each sweep is attributable to that axis alone.
    print("\nAxis isolation — Δll at coarsest setting of each single-axis "
          "sweep (phi-fixed for r-sweep, r-fixed for phi-sweep):")
    per_galaxy = {}
    for r in summary:
        per_galaxy.setdefault(r["galaxy"], []).append(r)
    for g, rows in per_galaxy.items():
        r_rows = [r for r in rows if r["tag"] == "r"]
        p_rows = [r for r in rows if r["tag"] == "phi"]
        if r_rows and p_rows:
            r_coarse = min(r_rows, key=lambda r: r["n_r_local"])
            p_coarse = min(p_rows, key=lambda r: r["n_hv_high"])
            print(f"  {g:<14} "
                  f"r-coarsest(nr_loc={r_coarse['n_r_local']}) "
                  f"Δ={r_coarse['delta']:+.3f}  |  "
                  f"phi-coarsest(nhv_hi={p_coarse['n_hv_high']}) "
                  f"Δ={p_coarse['delta']:+.3f}")

    print("\nPer-galaxy smallest setting with |Δll| ≤ 0.5 nats:")
    for g, rows in per_galaxy.items():
        ok = [r for r in rows if abs(r["delta"]) <= 0.5]
        if ok:
            # smallest = smallest n_r_local first, then n_hv_high
            best = min(ok, key=lambda r: (r["n_r_local"], r["n_hv_high"]))
            msg = (f"n_r_local={best['n_r_local']:d}, "
                   f"n_r_global={best['n_r_global']:d}, "
                   f"n_phi_hv_high={best['n_hv_high']:d}, "
                   f"n_phi_hv_low={best['n_hv_low']:d}, "
                   f"n_phi_sys={best['n_sys']:d}")
            print(f"  {g:<14} -> {msg}  (Δll={best['delta']:+.3f})")
        else:
            tight = min(rows, key=lambda r: abs(r["delta"]))
            print(f"  {g:<14} -> NONE within tolerance; "
                  f"tightest tried: "
                  f"n_r_local={tight['n_r_local']:d}, "
                  f"n_r_global={tight['n_r_global']:d}, "
                  f"n_phi_hv_high={tight['n_hv_high']:d}, "
                  f"n_phi_hv_low={tight['n_hv_low']:d}, "
                  f"n_phi_sys={tight['n_sys']:d}  "
                  f"(Δll={tight['delta']:+.3f})")


if __name__ == "__main__":
    main()
