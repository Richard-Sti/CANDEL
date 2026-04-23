"""
Mode 2 (r + phi marginal) grid convergence test.

For each galaxy (Mode 2: adaptive sinh r-grid for HV and sys+accel;
log-uniform brute grid for sys-no-accel), build the model at several
(n_r_local, n_r_brute, n_phi_hv_high, n_phi_hv_low, n_phi_sys) settings
and compare ll_disk to a float32 brute-force reference on a uniform
(r-log, φ-full-2π) grid.

The reference runs per-spot with the r axis chunked so the per-spot
intermediate fits on a 12 GB GPU.

Usage:
    python scripts/megamaser/convergence_grids.py [--galaxies ...]
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import tomli

from candel.model.maser_convergence import (
    bruteforce_ll_mode2, build_model, get_default_grid,
)

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
ALL_GALAXIES = ["CGCG074-064", "NGC5765b", "NGC6264", "NGC6323", "UGC3789"]


def _phys_from_init(model, galaxies_cfg, galaxy):
    """Build (phys_args, phys_kw, diag) from the config init block by
    delegating to MaserDiskModel.phys_from_sample."""
    init = galaxies_cfg[galaxy]["init"]
    sample = {k: np.asarray(v) for k, v in init.items()}
    return model.phys_from_sample(sample)


# r-axis variations (paired n_r_local, n_r_brute) tested while holding
# phi at the config default.
R_EXTRAS = [
    (51,  201),
    (101, 501),
    (151, 501),
    (251, 1001),
    (401, 1001),
]

# phi-axis variations (n_hv_high, n_hv_low, n_sys) tested while holding
# r at the config default.
PHI_EXTRAS = [
    (1001,  301,  1501),
    (2001,  501,  2001),
    (3001,  501,  4501),
    (5001,  1001, 5001),
]


def build_test_settings(default):
    """Build the Mode 2 sweep around the config default.

    Block A ("joint"): the config default itself.
    Block B ("r"):     vary (n_r_local, n_r_brute), phi pinned at default.
    Block C ("phi"):   vary (n_hv_high, n_hv_low, n_sys), r pinned at default.

    The default appears in all three blocks (deduplicated within each).
    """
    r_def = (default["n_r_local"], default["n_r_brute"])
    p_def = (default["n_hv_high"], default["n_hv_low"], default["n_sys"])
    rows = [dict(tag="joint", n_r_local=r_def[0], n_r_brute=r_def[1],
                 n_hv_high=p_def[0], n_hv_low=p_def[1], n_sys=p_def[2])]
    r_set = sorted({*R_EXTRAS, r_def}, key=lambda x: (x[0], x[1]))
    for nl, nb in r_set:
        rows.append(dict(tag="r", n_r_local=nl, n_r_brute=nb,
                         n_hv_high=p_def[0], n_hv_low=p_def[1],
                         n_sys=p_def[2]))
    p_set = sorted({*PHI_EXTRAS, p_def}, key=lambda x: (x[0], x[2], x[1]))
    for nh, nl, ns in p_set:
        rows.append(dict(tag="phi", n_r_local=r_def[0], n_r_brute=r_def[1],
                         n_hv_high=nh, n_hv_low=nl, n_sys=ns))
    return rows



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxies", nargs="+", default=ALL_GALAXIES)
    parser.add_argument(
        "--spot-batch", type=int, default=16,
        help="Chunk size on the spot axis when evaluating the test-grid "
             "phi marginal. Lower if the largest (n_r_local, n_phi) "
             "combination OOMs on a small GPU (default: 16).")
    args = parser.parse_args()

    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_enable_x64", True)
    print(f"JAX platform: {jax.default_backend()}, precision: float64",
          flush=True)

    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)
    galaxies_cfg = master_cfg["model"]["galaxies"]
    ref_cfg = master_cfg["convergence"]["mode2_reference"]

    if "NGC4258" in args.galaxies:
        raise ValueError(
            "NGC4258 is Mode 1 only (forbid_marginalise_r = true).")

    default = get_default_grid(master_cfg)
    test_settings = build_test_settings(default)

    print("=" * 90)
    print("Mode 2 grid convergence")
    print(f"Reference: {ref_cfg['n_r']} r (log-uniform) × "
          f"{ref_cfg['n_phi']} phi (uniform 2π), "
          f"r-chunk={ref_cfg['r_chunk']}, dtype={ref_cfg['dtype']}")
    print(f"Config default: n_r_local={default['n_r_local']}, "
          f"n_r_brute={default['n_r_brute']}, "
          f"n_phi=({default['n_hv_high']}, {default['n_hv_low']}, "
          f"{default['n_sys']})")
    print(f"Test settings: {len(test_settings)} combinations")
    print("=" * 90)

    summary = []
    for galaxy in args.galaxies:
        print(f"\n{'─' * 70}")
        print(f"Galaxy: {galaxy}")
        print(f"{'─' * 70}")
        model_ref = build_model(galaxy, master_cfg, mode="mode2")
        phys_args, phys_kw, diag = _phys_from_init(
            model_ref, galaxies_cfg, galaxy)
        print(f"  D_A={diag['D_A']:.2f} Mpc, n_spots={model_ref.n_spots}")

        print("  Computing brute-force reference...", flush=True)
        ll_ref = bruteforce_ll_mode2(model_ref, phys_args, phys_kw, ref_cfg)
        print(f"  Reference ll_disk = {ll_ref:.4f}")

        print(f"\n  {'tag':>5} {'nr_loc':>7} {'nr_brt':>7} {'nhv_hi':>7} "
              f"{'nhv_lo':>7} {'n_sys':>7}  {'ll_disk':>12}  {'Δ':>9}")
        best_delta = float("inf")
        best_setting = None
        for setting in test_settings:
            m_t = build_model(galaxy, master_cfg, mode="mode2",
                              n_r_local=setting["n_r_local"],
                              n_r_brute=setting["n_r_brute"],
                              n_phi_hv_high=setting["n_hv_high"],
                              n_phi_hv_low=setting["n_hv_low"],
                              n_phi_sys=setting["n_sys"])
            i0 = phys_args[8]
            sigma_a_floor2 = phys_args[16]
            var_v_hv = phys_args[15]
            v_sys = phys_args[4]
            M_BH = phys_args[3]
            D_A = phys_args[2]
            groups = m_t._build_r_grids_mode2(
                D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
            ll = m_t._eval_phi_marginal(
                groups, phys_args, phys_kw, spot_batch=args.spot_batch)
            ll_total = float(jnp.sum(ll))
            delta = ll_total - ll_ref
            print(f"  {setting['tag']:>5} {setting['n_r_local']:>7d} "
                  f"{setting['n_r_brute']:>7d} "
                  f"{setting['n_hv_high']:>7d} {setting['n_hv_low']:>7d} "
                  f"{setting['n_sys']:>7d}  {ll_total:12.4f}  "
                  f"{delta:+9.4f}")
            summary.append(dict(
                galaxy=galaxy, **setting,
                ll=ll_total, delta=delta, n_spots=model_ref.n_spots))
            if abs(delta) < abs(best_delta):
                best_delta = delta
                best_setting = setting
        print(f"  Best: {best_setting} Δ={best_delta:+.4f}")

    # --------------------------------------------------------------
    # Summary table + reasonable-values picker.
    # --------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("SUMMARY — Δ ll_disk (cfg − brute-force ref), nats")
    print(f"{'=' * 90}")
    header = (f"{'Galaxy':<14} {'tag':>5} {'nr_loc':>7} {'nr_brt':>7} "
              f"{'nhv_hi':>7} {'nhv_lo':>7} {'n_sys':>7}  "
              f"{'ll_disk':>12}  {'Δ':>9}")
    print(header)
    print("-" * len(header))
    for r in summary:
        print(f"{r['galaxy']:<14} {r['tag']:>5} {r['n_r_local']:>7d} "
              f"{r['n_r_brute']:>7d} {r['n_hv_high']:>7d} "
              f"{r['n_hv_low']:>7d} {r['n_sys']:>7d}  {r['ll']:12.4f}  "
              f"{r['delta']:+9.4f}")

    # Axis isolation: at the finest anchor, the joint Δ matches both
    # the r-sweep and phi-sweep finest rows. The residual at coarser
    # settings in each sweep is attributable to that axis alone.
    print("\nAxis isolation — Δ at coarsest setting of each single-axis "
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
                  f"r-coarsest(nr_loc={r_coarse['n_r_local']}) Δ={r_coarse['delta']:+.3f}  |  "
                  f"phi-coarsest(nhv_hi={p_coarse['n_hv_high']}) Δ={p_coarse['delta']:+.3f}")

    print("\nPer-galaxy smallest setting with |Δ| ≤ 0.5 nats:")
    per_galaxy = {}
    for r in summary:
        per_galaxy.setdefault(r["galaxy"], []).append(r)
    for g, rows in per_galaxy.items():
        ok = [r for r in rows if abs(r["delta"]) <= 0.5]
        if ok:
            # smallest = smallest n_r_local first, then n_hv_high
            best = min(ok, key=lambda r: (r["n_r_local"], r["n_hv_high"]))
            print(f"  {g:<14} -> n_r_local={best['n_r_local']:d}, "
                  f"n_r_brute={best['n_r_brute']:d}, "
                  f"n_phi_hv_high={best['n_hv_high']:d}, "
                  f"n_phi_hv_low={best['n_hv_low']:d}, "
                  f"n_phi_sys={best['n_sys']:d}  "
                  f"(Δ={best['delta']:+.3f})")
        else:
            tight = min(rows, key=lambda r: abs(r["delta"]))
            print(f"  {g:<14} -> NONE within 0.5; tightest tried: "
                  f"n_r_local={tight['n_r_local']:d}, "
                  f"n_r_brute={tight['n_r_brute']:d}, "
                  f"n_phi_hv_high={tight['n_hv_high']:d}, "
                  f"n_phi_hv_low={tight['n_hv_low']:d}, "
                  f"n_phi_sys={tight['n_sys']:d}  "
                  f"(Δ={tight['delta']:+.3f})")


if __name__ == "__main__":
    main()
