"""Validate marginalisation methods against expensive reference integrals.

Compares:
1. Original grid (21x31, no mode 2) -- baseline
2. Grid + robust peak + mode 2 -- new default
3. Laplace + robust peak + mode 2 -- fast optional
4. Reference grid (51x71, robust peak, mode 1 only) -- gold standard

All evaluated on CGCG 074-064 (165 spots) with near-Pesce+2020 parameters.
"""
import sys
sys.path.insert(0, ".")

import time
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.model.maser_disk_marginalised import (
    build_grid_config, marginalise_spots, marginalise_spots_laplace,
    find_peak_rphi, find_peak_rphi_robust,
    _spot_log_likelihood_on_grid, _find_mode2,
    SCAN_NR, SCAN_NPHI,
)
from candel.model.maser_disk import warp_geometry
from candel.model.simpson import simpson_log_weights, ln_simpson_precomputed


def _phi_bounds(spot_type, n):
    phi_lo, phi_hi = np.empty(n), np.empty(n)
    for k in range(n):
        if spot_type[k] == "r":
            phi_lo[k], phi_hi[k] = 0, np.pi
        elif spot_type[k] == "b":
            phi_lo[k], phi_hi[k] = np.pi, 2 * np.pi
        else:
            phi_lo[k], phi_hi[k] = -np.pi / 2, np.pi / 2
    return jnp.array(phi_lo), jnp.array(phi_hi)


def run(galaxy="CGCG074-064"):
    data = load_megamaser_spots("data/Megamaser", galaxy=galaxy)
    n = data["n_spots"]
    phi_lo, phi_hi = _phi_bounds(data["spot_type"], n)

    x = jnp.array(data["x"]); sx = jnp.array(data["sigma_x"])
    y = jnp.array(data["y"]); sy = jnp.array(data["sigma_y"])
    v = jnp.array(data["velocity"]); a = jnp.array(data["a"])
    sa = jnp.array(data["sigma_a"])
    am = jnp.array(data["accel_measured"]); iss = jnp.array(data["is_systemic"])

    D = 87.6; M = 2.42e7; vs = 6925.; x0 = 0.005; y0 = -0.003
    i0 = jnp.deg2rad(85.); di = 0.0
    O0 = jnp.deg2rad(120.); dO = jnp.deg2rad(-5.)
    sxf = 0.01; syf = 0.01; svs = 5.; svh = 5.; saf = 0.5

    common = (x, sx, y, sy, v, a, sa, am, iss, phi_lo, phi_hi,
              x0, y0, D, M, vs, i0, di, O0, dO, sxf, syf, svs, svh, saf)

    # Grid configs
    gc = build_grid_config()  # 21x31
    dr_scan = jnp.linspace(-0.15, 0.15, SCAN_NR)
    dphi_scan = jnp.linspace(-0.50, 0.50, SCAN_NPHI)

    # Reference grid: 51x71 (expensive)
    gc_ref = build_grid_config(Nr=51, Nphi=71)

    print(f"=== {galaxy}: {n} spots ===\n")

    # ---- 1. Reference: 51x71 grid, robust peak, single mode ----
    print("Computing reference (51x71 grid, robust peak, mode 1)...")
    r_rob, phi_rob = find_peak_rphi_robust(
        x, y, v, a, am, iss, phi_lo, phi_hi,
        x0, y0, D, M, vs, i0, di, O0, dO,
        sx, sy, svs, svh, sa, saf, sxf, syf,
        dr_scan, dphi_scan, SCAN_NPHI)

    ref_per_spot = []
    for k in range(n):
        r_grid = jnp.clip(r_rob[k] + gc_ref["dr_offsets"], 0.01, 10.0)
        phi_grid = jnp.clip(r_rob[k] + gc_ref["dphi_offsets"],  # BUG: should be phi_rob
                            phi_lo[k], phi_hi[k])
        # Fix: center on phi_rob, not r_rob
        phi_grid = jnp.clip(phi_rob[k] + gc_ref["dphi_offsets"],
                            phi_lo[k], phi_hi[k])
        log_int = _spot_log_likelihood_on_grid(
            r_grid, phi_grid,
            x[k], sx[k], y[k], sy[k], v[k], a[k], sa[k], am[k], iss[k],
            x0, y0, D, M, vs, i0, di, O0, dO, sxf, syf, svs, svh, saf)
        ln_I = float(ln_simpson_precomputed(
            ln_simpson_precomputed(log_int, gc_ref["log_wphi"], axis=-1),
            gc_ref["log_wr"], axis=-1))
        ref_per_spot.append(ln_I)
    ref_per_spot = np.array(ref_per_spot)
    ref_total = ref_per_spot.sum()
    print(f"  Total ll: {ref_total:.4f}")

    # ---- 2. Original grid (21x31, old peak finder, no mode 2) ----
    print("\nMethod 1: Original grid (21x31, G-N peak, no mode 2)")
    fn1 = jax.jit(lambda: marginalise_spots(
        *common, gc["dr_offsets"], gc["dphi_offsets"],
        gc["log_wr"], gc["log_wphi"], bimodal=False))
    ll1, _, _ = fn1()
    t0 = time.perf_counter()
    for _ in range(30): out = fn1()
    jax.block_until_ready(out)
    t1 = (time.perf_counter() - t0) / 30 * 1e3
    print(f"  Total ll: {float(ll1):.4f}")
    print(f"  Diff vs ref: {float(ll1) - ref_total:.4f} ({(float(ll1) - ref_total)/n:.4f}/spot)")
    print(f"  Time: {t1:.2f} ms")

    # ---- 3. Grid + robust peak + mode 2 (NEW DEFAULT) ----
    print("\nMethod 2: Grid + robust peak + mode 2 (new default)")
    fn2 = jax.jit(lambda: marginalise_spots(
        *common, gc["dr_offsets"], gc["dphi_offsets"],
        gc["log_wr"], gc["log_wphi"],
        dr_scan, dphi_scan, SCAN_NPHI,
        bimodal=True))
    ll2, _, _ = fn2()
    t0 = time.perf_counter()
    for _ in range(30): out = fn2()
    jax.block_until_ready(out)
    t2 = (time.perf_counter() - t0) / 30 * 1e3
    print(f"  Total ll: {float(ll2):.4f}")
    print(f"  Diff vs ref: {float(ll2) - ref_total:.4f} ({(float(ll2) - ref_total)/n:.4f}/spot)")
    print(f"  Time: {t2:.2f} ms")

    # ---- 4. Laplace + robust peak + mode 2 (fast optional) ----
    print("\nMethod 3: Laplace + robust peak + mode 2 (fast optional)")
    fn3 = jax.jit(lambda: marginalise_spots_laplace(
        *common, dr_scan, dphi_scan, SCAN_NPHI))
    ll3, _, _ = fn3()
    t0 = time.perf_counter()
    for _ in range(30): out = fn3()
    jax.block_until_ready(out)
    t3 = (time.perf_counter() - t0) / 30 * 1e3
    print(f"  Total ll: {float(ll3):.4f}")
    print(f"  Diff vs ref: {float(ll3) - ref_total:.4f} ({(float(ll3) - ref_total)/n:.4f}/spot)")
    print(f"  Time: {t3:.2f} ms")

    # ---- Gradient comparison ----
    print("\n=== Gradient d(ll)/dD ===")

    def _grad(fn):
        def _ll(D_):
            args = list(common); args[13] = D_
            return fn(*args)[0]
        return float(jax.grad(_ll)(D))

    def _fn1(*args):
        return marginalise_spots(
            *args, gc["dr_offsets"], gc["dphi_offsets"],
            gc["log_wr"], gc["log_wphi"], bimodal=False)

    def _fn2(*args):
        return marginalise_spots(
            *args, gc["dr_offsets"], gc["dphi_offsets"],
            gc["log_wr"], gc["log_wphi"],
            dr_scan, dphi_scan, SCAN_NPHI, bimodal=True)

    def _fn3(*args):
        return marginalise_spots_laplace(
            *args, dr_scan, dphi_scan, SCAN_NPHI)

    g1 = _grad(_fn1)
    g2 = _grad(_fn2)
    g3 = _grad(_fn3)
    print(f"  Original grid:        {g1:.4f}")
    print(f"  Grid+robust+mode2:    {g2:.4f} (ratio vs orig: {g2/g1:.4f})")
    print(f"  Laplace+robust+mode2: {g3:.4f} (ratio vs orig: {g3/g1:.4f})")

    # ---- Summary ----
    print("\n=== Summary ===")
    print(f"{'Method':<30s} {'ll':>12s} {'diff/spot':>10s} {'grad ratio':>11s} {'time':>8s} {'speedup':>8s}")
    for label, ll, g, t in [
        ("Original grid", float(ll1), g1, t1),
        ("Grid+robust+mode2", float(ll2), g2, t2),
        ("Laplace+robust+mode2", float(ll3), g3, t3),
    ]:
        diff = (ll - ref_total) / n
        gr = g / g1
        spd = t1 / t
        print(f"{label:<30s} {ll:>12.2f} {diff:>+10.4f} {gr:>11.4f} {t:>7.2f}ms {spd:>7.2f}x")


if __name__ == "__main__":
    run()
