"""Per-spot Mode 2 Δll diagnostic for MCP maser galaxies.

Two-stage approach per galaxy:
  1. Cheap moderate-resolution brute-force on ALL spots to find outliers.
  2. Full-resolution brute-force + 1D r posterior plots for the worst N.

Usage:
  python diagnose_mode2_delta.py                         # all Mode 2 galaxies
  python diagnose_mode2_delta.py --galaxies NGC4258      # single galaxy
  python diagnose_mode2_delta.py --galaxies NGC5765b UGC3789
"""
import argparse
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tomli
from jax.scipy.special import logsumexp

from candel.model.integration import trapz_log_weights
from candel.model.maser_convergence import build_model, resolve_grid_for_galaxy

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
OUT_DIR = "/mnt/users/rstiskalek/CANDEL/results/Megamaser/convergence"


def phys_from_init(model, galaxy, galaxies_cfg):
    init = galaxies_cfg[galaxy]["init"]
    sample = {k: np.asarray(v) for k, v in init.items()}
    return model.phys_from_sample(sample)


def per_spot_ll_production(model, phys_args, phys_kw, spot_batch):
    D_A, M_BH, v_sys = phys_args[2], phys_args[3], phys_args[4]
    i0, var_v_hv = phys_args[8], phys_args[15]
    sigma_a_floor2 = phys_args[16]
    groups = model._build_r_grids_mode2(
        D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv,
        phys_args=phys_args, phys_kw=phys_kw)
    ll = model._eval_phi_marginal(
        groups, phys_args, phys_kw, spot_batch=spot_batch)
    return np.asarray(ll)


def per_spot_bruteforce_all(model, phys_args, phys_kw,
                            n_r, n_phi, r_chunk_size, spot_batch):
    D_A = phys_args[2]
    r_min, r_max = model.r_ang_range(D_A)
    r_grid = jnp.exp(jnp.linspace(jnp.log(r_min), jnp.log(r_max), n_r))
    log_w_r = trapz_log_weights(r_grid)

    phi = jnp.linspace(0.0, 2 * jnp.pi, n_phi)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    log_w_phi = trapz_log_weights(phi)

    ll_out = np.zeros(model.n_spots)
    for type_key, idx in [("sys", model._idx_sys),
                          ("red", model._idx_red),
                          ("blue", model._idx_blue)]:
        n = int(idx.shape[0])
        if n == 0:
            continue
        has_any_accel = model._group_has_any_accel(type_key)
        for s in range(0, n, spot_batch):
            b_idx = idx[s:s + spot_batch]
            nb = int(b_idx.shape[0])
            partials = []
            for start in range(0, n_r, r_chunk_size):
                end = min(start + r_chunk_size, n_r)
                r_chunk = r_grid[start:end]
                lw_chunk = log_w_r[start:end]
                r_ang_2d = jnp.broadcast_to(
                    r_chunk[None, :], (nb, r_chunk.shape[0]))
                r_pre = model._r_precompute(
                    r_ang_2d, b_idx, *phys_args, **phys_kw,
                    has_any_accel=has_any_accel)
                nhc = model._phi_eval(r_pre, sin_phi, cos_phi)
                lnorm = r_pre["lnorm"] + r_pre["lnorm_a"]
                w2d = lw_chunk[None, :, None] + log_w_phi[None, None, :]
                chunk_ll = logsumexp(nhc + w2d, axis=(-2, -1))
                partials.append(lnorm + chunk_ll)
            ll_spots = logsumexp(
                jnp.stack(partials, axis=0), axis=0)
            ll_out[np.asarray(b_idx)] = np.asarray(ll_spots)
    return ll_out


def per_spot_bruteforce_subset(model, idx_subset, phys_args, phys_kw,
                               n_r, n_phi, r_chunk_size):
    D_A = phys_args[2]
    r_min, r_max = model.r_ang_range(D_A)
    r_grid = jnp.exp(jnp.linspace(jnp.log(r_min), jnp.log(r_max), n_r))
    log_w_r = trapz_log_weights(r_grid)

    phi = jnp.linspace(0.0, 2 * jnp.pi, n_phi)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    log_w_phi = trapz_log_weights(phi)

    has_accel = np.asarray(model._all_has_accel)
    ll_out = {}
    for gi in idx_subset:
        gi = int(gi)
        b_idx = jnp.array([gi])
        has_a = bool(has_accel[gi])
        partials = []
        for start in range(0, n_r, r_chunk_size):
            end = min(start + r_chunk_size, n_r)
            r_chunk = r_grid[start:end]
            lw_chunk = log_w_r[start:end]
            r_ang_2d = r_chunk[None, :]
            r_pre = model._r_precompute(
                r_ang_2d, b_idx, *phys_args, **phys_kw,
                has_any_accel=has_a)
            nhc = model._phi_eval(r_pre, sin_phi, cos_phi)
            lnorm = r_pre["lnorm"] + r_pre["lnorm_a"]
            w2d = lw_chunk[None, :, None] + log_w_phi[None, None, :]
            chunk_ll = logsumexp(nhc + w2d, axis=(-2, -1))
            partials.append(lnorm + chunk_ll)
        ll = float(logsumexp(jnp.stack(partials, axis=0), axis=0)[0])
        ll_out[gi] = ll
    return ll_out


def r_posterior_1spot(model, gi, r_grid, phys_args, phys_kw,
                      sin_phi, cos_phi, log_w_phi, r_batch):
    b_idx = jnp.array([gi])
    has_a = bool(model._all_has_accel[gi])
    n_r = int(r_grid.shape[0])

    @jax.jit
    def _one_chunk(r_chunk):
        r_ang_2d = r_chunk[None, :]
        r_pre = model._r_precompute(
            r_ang_2d, b_idx, *phys_args, **phys_kw,
            has_any_accel=has_a)
        nhc = model._phi_eval(r_pre, sin_phi, cos_phi)
        lnorm = r_pre["lnorm"] + r_pre["lnorm_a"]
        return lnorm[:, None] + logsumexp(nhc + log_w_phi, axis=-1)

    parts = []
    for s in range(0, n_r, r_batch):
        parts.append(_one_chunk(r_grid[s:s + r_batch]))
    return np.asarray(jnp.concatenate(parts, axis=-1)[0])


def run_galaxy(galaxy, master_cfg, args):
    """Run the two-stage diagnostic for one galaxy."""
    galaxies_cfg = master_cfg["model"]["galaxies"]
    print(f"\n{'=' * 60}", flush=True)
    print(f"Galaxy: {galaxy}", flush=True)
    print(f"{'=' * 60}", flush=True)

    grid = resolve_grid_for_galaxy(master_cfg, galaxy, "mode2")
    print(f"Production grid: n_r=({grid['n_r_local']}, {grid['n_r_global']}), "
          f"phi=({grid['n_hv_high']}, {grid['n_hv_low']}, {grid['n_sys']})",
          flush=True)

    model = build_model(galaxy, master_cfg, mode="mode2",
                        n_phi_hv_high=grid["n_hv_high"],
                        n_phi_hv_low=grid["n_hv_low"],
                        n_phi_sys=grid["n_sys"],
                        n_r_global=grid["n_r_global"])
    phys_args, phys_kw, diag = phys_from_init(model, galaxy, galaxies_cfg)
    print(f"D_A={diag['D_A']:.2f} Mpc, n_spots={model.n_spots}", flush=True)

    types = [("sys", model._idx_sys),
             ("red", model._idx_red),
             ("blue", model._idx_blue)]
    spot_types = np.full(model.n_spots, "", dtype=object)
    for name, idx in types:
        spot_types[np.asarray(idx)] = name
    has_accel = np.asarray(model._all_has_accel)

    # ── 1. Production ll ──
    print("Computing production ll...", flush=True)
    t0 = time.time()
    ll_prod = per_spot_ll_production(
        model, phys_args, phys_kw, args.spot_batch)
    print(f"  done ({time.time()-t0:.1f}s), total={ll_prod.sum():.4f}",
          flush=True)

    # ── 2. Screening ──
    print(f"Screening all spots "
          f"(n_r={args.n_r_screen}, n_phi={args.n_phi_screen})...",
          flush=True)
    t0 = time.time()
    ll_screen = per_spot_bruteforce_all(
        model, phys_args, phys_kw,
        args.n_r_screen, args.n_phi_screen,
        args.r_chunk_screen, args.spot_batch)
    dt = time.time() - t0
    delta_screen = ll_prod - ll_screen
    print(f"  done ({dt:.1f}s), total Dll={delta_screen.sum():.4f}",
          flush=True)

    for name, idx in types:
        idx_np = np.asarray(idx)
        d = delta_screen[idx_np]
        if len(idx_np) > 0:
            print(f"  {name:>4} ({len(idx_np):3d} spots): "
                  f"sum={d.sum():+.4f}, mean={d.mean():+.4f}, "
                  f"min={d.min():+.4f}, max={d.max():+.4f}", flush=True)

    # ── 3. Full-res on worst spots ──
    worst_idx = np.argsort(delta_screen)[:args.n_worst]
    print(f"\n{args.n_worst} worst spots from screening:", flush=True)
    for gi in worst_idx:
        print(f"  spot {gi:3d}  {spot_types[gi]:>4}  "
              f"accel={'Y' if has_accel[gi] else 'N'}  "
              f"Dll_screen={delta_screen[gi]:+.4f}", flush=True)

    print(f"\nFull brute-force on {args.n_worst} worst "
          f"(n_r={args.n_r_ref}, n_phi={args.n_phi_ref})...", flush=True)
    t0 = time.time()
    ll_ref_map = per_spot_bruteforce_subset(
        model, worst_idx, phys_args, phys_kw,
        args.n_r_ref, args.n_phi_ref, args.r_chunk_ref)
    print(f"  done ({time.time()-t0:.1f}s)", flush=True)

    deltas_full = {}
    print(f"\nFull-res Dll (prod - ref):", flush=True)
    for gi in worst_idx:
        gi = int(gi)
        d = float(ll_prod[gi]) - ll_ref_map[gi]
        deltas_full[gi] = d
        print(f"  spot {gi:3d}  {spot_types[gi]:>4}  "
              f"accel={'Y' if has_accel[gi] else 'N'}  "
              f"Dll={d:+.6f}  "
              f"prod={ll_prod[gi]:.4f}  ref={ll_ref_map[gi]:.4f}",
              flush=True)

    # ── 4. Dense r posteriors ──
    centres = model.get_mode2_centres(phys_args, phys_kw)
    r_min = float(centres["r_min"])
    r_max = float(centres["r_max"])

    rc_map, s_map = {}, {}
    for name, idx in types:
        c = centres.get(name)
        if c is None:
            continue
        idx_np = np.asarray(idx)
        rc_arr = np.asarray(c["r_c"])
        s_arr = np.asarray(c["s"])
        for i, gi in enumerate(idx_np):
            rc_map[int(gi)] = float(rc_arr[i])
            s_map[int(gi)] = float(s_arr[i])

    K_sigma = float(model._K_sigma)
    log_bin = (np.log(r_max) - np.log(r_min)) / (model._n_r_global - 1)
    zoom_half = 5 * log_bin

    n_phi_full = 20001
    phi_full = jnp.linspace(0.0, 2 * jnp.pi, n_phi_full)
    sin_phi_full = jnp.sin(phi_full)
    cos_phi_full = jnp.cos(phi_full)
    log_w_phi_full = trapz_log_weights(phi_full)

    print("\nComputing dense r posteriors...", flush=True)
    dense_data = {}
    for gi in worst_idx:
        gi = int(gi)
        tp = spot_types[gi]
        pc = model._phi_concat[tp]
        rc = rc_map.get(gi, np.sqrt(r_min * r_max))
        log_center = np.log(rc)
        log_lo_z = max(np.log(r_min), log_center - zoom_half)
        log_hi_z = min(np.log(r_max), log_center + zoom_half)
        r_local = jnp.exp(jnp.linspace(log_lo_z, log_hi_z,
                                        args.n_r_dense))
        lp_prod = r_posterior_1spot(
            model, gi, r_local, phys_args, phys_kw,
            pc["sin_phi"], pc["cos_phi"], pc["log_w_phi"],
            args.r_batch)
        lp_full = r_posterior_1spot(
            model, gi, r_local, phys_args, phys_kw,
            sin_phi_full, cos_phi_full, log_w_phi_full,
            args.r_batch)
        dense_data[gi] = dict(r=np.asarray(r_local), lp_prod=lp_prod,
                              lp_full=lp_full,
                              log_w=np.asarray(trapz_log_weights(r_local)))
        print(f"  spot {gi:3d} done", flush=True)

    # ── 5. Plot ──
    n_plot = len(worst_idx)
    ncols = min(4, n_plot)
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for k, gi in enumerate(worst_idx):
        gi = int(gi)
        ax = axes[k // ncols, k % ncols]
        dd = dense_data[gi]
        r_np = dd["r"]
        log_w_r = dd["log_w"]

        for lp, color, ls, label in [
                (dd["lp_prod"], "C0", "-", "prod phi"),
                (dd["lp_full"], "C1", "--", "full 2pi phi")]:
            log_Z = float(logsumexp(jnp.asarray(lp) + jnp.asarray(log_w_r)))
            p = np.exp(lp - log_Z)
            ax.plot(r_np, p, color=color, ls=ls, lw=1.2, label=label)

        if gi in rc_map:
            rc = rc_map[gi]
            s = s_map[gi]
            ax.axvline(rc, color="C3", ls=":", lw=1,
                       label=f"r_c={rc:.3f}, s={s:.4f}")
            ax.axvspan(np.exp(np.log(rc) - K_sigma * s),
                       np.exp(np.log(rc) + K_sigma * s),
                       color="C3", alpha=0.08)

        d_full = deltas_full.get(gi, delta_screen[gi])
        ax.set_title(
            f"#{gi} {spot_types[gi]} "
            f"{'acc' if has_accel[gi] else 'no-acc'}  "
            f"Dll={d_full:+.4f}",
            fontsize=9)
        ax.set_xlabel(r"$r_\mathrm{ang}$ [mas]", fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel(r"$p(r | d_i)$", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0, 0].legend(fontsize=7, loc="upper right")
    for k in range(n_plot, nrows * ncols):
        axes[k // ncols, k % ncols].set_visible(False)

    fig.suptitle(
        f"{galaxy} Mode 2: {n_plot} worst spots  "
        f"(screening total Dll={delta_screen.sum():.2f} nats over "
        f"{model.n_spots} spots)",
        fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{galaxy}_mode2_diagnose.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Per-spot Mode 2 Dll diagnostic for MCP galaxies.")
    ap.add_argument("--galaxies", nargs="+", default=None,
                    help="Galaxies to test (default: all with mode=mode2)")
    ap.add_argument("--spot-batch", type=int, default=4)
    ap.add_argument("--n-r-screen", type=int, default=5000)
    ap.add_argument("--n-phi-screen", type=int, default=5001)
    ap.add_argument("--r-chunk-screen", type=int, default=1000)
    ap.add_argument("--n-r-ref", type=int, default=50000)
    ap.add_argument("--n-phi-ref", type=int, default=50001)
    ap.add_argument("--r-chunk-ref", type=int, default=2000)
    ap.add_argument("--n-r-dense", type=int, default=2001)
    ap.add_argument("--r-batch", type=int, default=64)
    ap.add_argument("--n-worst", type=int, default=12)
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)
    print(f"JAX: {jax.default_backend()}, f64", flush=True)

    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)

    if args.galaxies is not None:
        galaxies = args.galaxies
    else:
        galaxies = [g for g, cfg in master_cfg["model"]["galaxies"].items()
                    if cfg.get("mode", master_cfg["model"].get("mode"))
                    == "mode2"]
        if not galaxies:
            galaxies = list(master_cfg["model"]["galaxies"].keys())

    print(f"Galaxies: {galaxies}", flush=True)
    for galaxy in galaxies:
        run_galaxy(galaxy, master_cfg, args)


if __name__ == "__main__":
    main()
