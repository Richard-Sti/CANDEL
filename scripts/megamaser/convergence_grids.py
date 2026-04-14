"""
Grid convergence benchmark for the megamaser disk model.

Reference: brute-force evaluation on a 20001 r x 20001 phi uniform grid,
processed in spot batches to fit in GPU memory.

Tests:
  1. Adaptive per-spot r-grid at default phi (varying n_r_local)
  2. Adaptive per-spot r-grid at varying phi resolution

Uses MAP parameter values from config_maser.toml for each galaxy.

Usage:
    python scripts/megamaser/convergence_grids.py [--galaxies NGC5765b UGC3789]
"""
import argparse
import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial
import tomli
import tomli_w

from candel.model.model_H0_maser import (
    MaserDiskModel, warp_geometry, PC_PER_MAS_MPC,
    _precompute_r_quantities, _observables_from_precomputed,
    LOG_2PI,
)
from candel.model.integration import trapz_log_weights
from candel.pvdata.megamaser_data import load_megamaser_spots

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
ALL_GALAXIES = ["NGC5765b", "UGC3789", "CGCG074-064", "NGC6264", "NGC6323"]

# Reference grid
N_R_REF = 20001
N_PHI_REF = 20001
# Batch size 1: x64 always on (for position residual precision),
# so arrays are float64 → 1 spot × 20001² × 8 bytes ≈ 3.2 GB.
BATCH_SIZE = 1


# -----------------------------------------------------------------------
# Brute-force reference evaluator (batched over spots)
# -----------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_r", "n_phi", "has_accel"))
def _bruteforce_batch(
        x_obs, y_obs, v_obs, var_x, var_y, var_v,
        a_obs, var_a, accel_w, has_accel,
        x0, y0, D_A, M_BH, v_sys,
        r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
        r_min, r_max, n_r, n_phi):
    """Brute-force logL for a batch of spots on uniform r × phi grid."""
    log_r = jnp.linspace(jnp.log(r_min), jnp.log(r_max), n_r)
    r_grid = jnp.exp(log_r)
    log_w_r = trapz_log_weights(r_grid)

    phi_grid = jnp.linspace(0.0, 2 * jnp.pi, n_phi)
    log_w_phi = trapz_log_weights(phi_grid)

    # Warp + precompute r-dependent quantities via model functions
    i_r, Om_r = warp_geometry(r_grid, r_ang_ref, i0, di_dr, Omega0, dOmega_dr)
    sin_i = jnp.sin(i_r)
    cos_i = jnp.cos(i_r)
    sin_O = jnp.sin(Om_r)
    cos_O = jnp.cos(Om_r)
    v_kep, gamma, z_g, a_mag, pA, pB, pC, pD = \
        _precompute_r_quantities(r_grid, D_A, M_BH, sin_i, cos_i, sin_O, cos_O)

    # Observables on (n_r, n_phi) grid
    sp = jnp.sin(phi_grid)[None, :]
    cp = jnp.cos(phi_grid)[None, :]
    X, Y, V, A = _observables_from_precomputed(
        sp, cp, x0, y0, v_sys,
        sin_i[:, None], r_grid[:, None],
        v_kep[:, None], gamma[:, None], z_g[:, None], a_mag[:, None],
        pA[:, None], pB[:, None], pC[:, None], pD[:, None])

    # Chi2: (n_batch, n_r, n_phi)
    chi2 = ((x_obs[:, None, None] - X[None]) ** 2
            * (1.0 / var_x)[:, None, None]
            + (y_obs[:, None, None] - Y[None]) ** 2
            * (1.0 / var_y)[:, None, None]
            + (v_obs[:, None, None] - V[None]) ** 2
            * (1.0 / var_v)[:, None, None])

    if has_accel:
        chi2 = chi2 + ((a_obs[:, None, None] - A[None]) ** 2
                       * (1.0 / var_a)[:, None, None]
                       * accel_w[:, None, None])

    lnorm = -0.5 * (3 * LOG_2PI + jnp.log(var_x) + jnp.log(var_y)
                     + jnp.log(var_v))
    if has_accel:
        lnorm = lnorm - 0.5 * (LOG_2PI + jnp.log(var_a)) * accel_w

    log_f = lnorm[:, None, None] - 0.5 * chi2
    log_w_2d = log_w_r[:, None] + log_w_phi[None, :]
    return logsumexp(log_f + log_w_2d[None], axis=(-2, -1))


# -----------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------

def build_model(galaxy, master_cfg, adaptive_r=True, n_r_local=201,
                K_sigma=5.0, G_phi_half=251, n_inner_sys=201,
                n_wing_sys=100):
    cfg = master_cfg.copy()
    cfg["model"] = master_cfg["model"].copy()
    cfg["model"]["adaptive_r"] = adaptive_r
    cfg["model"]["n_r_local"] = n_r_local
    cfg["model"]["K_sigma"] = K_sigma
    cfg["model"]["G_phi_half"] = G_phi_half
    cfg["model"]["n_inner_sys"] = n_inner_sys
    cfg["model"]["n_wing_sys"] = n_wing_sys
    gcfg = master_cfg["model"]["galaxies"][galaxy]
    data = load_megamaser_spots(
        master_cfg["io"]["maser_data"]["root"], galaxy=galaxy,
        v_sys_obs=gcfg["v_sys_obs"])
    for key in ("D_lo", "D_hi"):
        if key in gcfg:
            data[key] = float(gcfg[key])
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(cfg, tmp)
    tmp.close()
    model = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)
    return model


def extract_phys(galaxy, galaxies_cfg, model):
    init = galaxies_cfg[galaxy]["init"]
    H0 = float(init["H0"])
    D_c = float(init["D_c"])
    eta = float(init["eta"])
    h = H0 / 100.0
    z_cosmo = float(model.distance2redshift(
        jnp.atleast_1d(D_c), h=h).squeeze())
    D_A = D_c / (1.0 + z_cosmo)
    M_BH = 10.0**(eta + np.log10(D_A) - 7.0)
    x0 = float(init["x0"])
    y0 = float(init["y0"])
    v_sys = model.v_sys_obs + float(init.get("dv_sys", 0.0))
    return dict(
        D_A=D_A, M_BH=M_BH, x0=x0, y0=y0, v_sys=v_sys,
        i0=np.deg2rad(float(init["i0"])),
        di_dr=np.deg2rad(float(init["di_dr"])),
        Omega0=np.deg2rad(float(init["Omega0"])),
        dOmega_dr=np.deg2rad(float(init["dOmega_dr"])),
        sigma_x_floor2=float(init["sigma_x_floor"]) ** 2,
        sigma_y_floor2=float(init["sigma_y_floor"]) ** 2,
        var_v_sys=float(init["sigma_v_sys"]) ** 2,
        var_v_hv=float(init["sigma_v_hv"]) ** 2,
        sigma_a_floor2=float(init["sigma_a_floor"]) ** 2)


def _model_args(model, phys):
    return (
        phys["x0"], phys["y0"], phys["D_A"], phys["M_BH"], phys["v_sys"],
        model._r_ang_ref,
        jnp.asarray(phys["i0"]), jnp.asarray(phys["di_dr"]),
        jnp.asarray(phys["Omega0"]), jnp.asarray(phys["dOmega_dr"]),
        jnp.asarray(phys["sigma_x_floor2"]),
        jnp.asarray(phys["sigma_y_floor2"]),
        jnp.asarray(phys["var_v_sys"]), jnp.asarray(phys["var_v_hv"]),
        jnp.asarray(phys["sigma_a_floor2"]))


# -----------------------------------------------------------------------
# Evaluation functions
# -----------------------------------------------------------------------

def bruteforce_ll(model, phys, n_r=N_R_REF, n_phi=N_PHI_REF,
                  batch_size=BATCH_SIZE):
    """Reference logL via batched brute-force on uniform r × phi grid."""
    conv = phys["D_A"] * PC_PER_MAS_MPC
    r_min = model._R_phys_lo / conv
    r_max = model._R_phys_hi / conv

    var_x = model._all_sigma_x2 + phys["sigma_x_floor2"]
    var_y = model._all_sigma_y2 + phys["sigma_y_floor2"]
    sv_floor = jnp.where(model.is_highvel, phys["var_v_hv"], phys["var_v_sys"])
    var_v = model._all_sigma_v2 + sv_floor
    var_a = model._all_sigma_a2 + phys["sigma_a_floor2"]
    has_accel_all = np.asarray(model._all_has_accel)

    common = dict(
        x0=phys["x0"], y0=phys["y0"], D_A=phys["D_A"],
        M_BH=phys["M_BH"], v_sys=phys["v_sys"],
        r_ang_ref=model._r_ang_ref,
        i0=jnp.asarray(phys["i0"]),
        di_dr=jnp.asarray(phys["di_dr"]),
        Omega0=jnp.asarray(phys["Omega0"]),
        dOmega_dr=jnp.asarray(phys["dOmega_dr"]),
        r_min=r_min, r_max=r_max, n_r=n_r, n_phi=n_phi)

    total = 0.0
    for has_a in (True, False):
        mask = has_accel_all if has_a else ~has_accel_all
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        for start in range(0, len(idx), batch_size):
            b = idx[start:min(start + batch_size, len(idx))]
            ll_batch = _bruteforce_batch(
                model._all_x[b], model._all_y[b], model._all_v[b],
                var_x[b], var_y[b], var_v[b],
                model._all_a[b], var_a[b], model._all_accel_w[b], has_a,
                **common)
            total += float(jnp.sum(ll_batch))
    return total


def adaptive_ll(model, phys):
    """logL via adaptive per-spot r-grid + model phi grids."""
    ll = model._eval_adaptive_phi_r(*_model_args(model, phys))
    return float(jnp.sum(ll))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxies", nargs="+", default=ALL_GALAXIES)
    parser.add_argument("--float64", action="store_true",
                        help="Use float64 precision (default: float32)")
    args = parser.parse_args()

    jax.config.update("jax_platform_name", "gpu")
    # x64 needed: _phi_integrand uses float64 for position residuals
    jax.config.update("jax_enable_x64", True)
    dtype = "float64" if args.float64 else "float32 (mixed)"
    print(f"JAX platform: {jax.default_backend()}, dtype: {dtype}",
          flush=True)

    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)
    galaxies_cfg = master_cfg["model"]["galaxies"]

    if "NGC4258" in args.galaxies:
        raise ValueError(
            "NGC4258 is not supported by this script. Its position errors "
            "are too small for the 20001² reference grid.")

    batch_size = BATCH_SIZE

    print("=" * 70)
    print("Grid convergence benchmark")
    print(f"Reference: {N_R_REF} r (log-uniform) × {N_PHI_REF} phi (uniform)")
    print(f"Batch size: {batch_size} spots")
    print(f"Precision: {dtype}")
    print("=" * 70)

    n_fail = 0
    for galaxy in args.galaxies:
        print(f"\n{'─' * 60}")
        print(f"Galaxy: {galaxy}")
        print(f"{'─' * 60}")

        model0 = build_model(galaxy, master_cfg)
        phys = extract_phys(galaxy, galaxies_cfg, model0)
        print(f"  D_A={phys['D_A']:.1f} Mpc, n_spots={model0.n_spots}")

        # ---- Reference ----
        print("  Computing brute-force reference...", flush=True)
        ll_ref = bruteforce_ll(model0, phys, batch_size=batch_size)
        print(f"  Reference logL = {ll_ref:.4f}")

        # ---- Adaptive r at default phi ----
        print(f"\n  {'method':>30}  {'logL':>14}  {'Δ':>10}")
        ll_adaptive = {}
        for n_local in [51, 101, 151, 251]:
            model_a = build_model(galaxy, master_cfg,
                                  n_r_local=n_local, K_sigma=5.0)
            ll = adaptive_ll(model_a, phys)
            ll_adaptive[n_local] = ll
            print(f"  {'adaptive n=' + str(n_local):>30}  {ll:14.4f}  "
                  f"{ll - ll_ref:+10.4f}")

        # ---- Adaptive r at varying phi (Simpson HV, trapz sys) ----
        ll_phi = {}
        for G in [51, 101, 201, 401, 801]:
            n_sys = G
            n_wing = max(G // 4, 10)
            model_a = build_model(galaxy, master_cfg,
                                  n_r_local=201, K_sigma=5.0,
                                  G_phi_half=G, n_inner_sys=n_sys,
                                  n_wing_sys=n_wing)
            ll = adaptive_ll(model_a, phys)
            ll_phi[G] = ll
            print(f"  {'adaptive phi_hv=' + str(G):>30}  {ll:14.4f}  "
                  f"{ll - ll_ref:+10.4f}")

        # ---- Numerical consistency checks ----
        delta_r = abs(ll_adaptive[251] - ll_ref)
        delta_phi = abs(ll_phi[801] - ll_ref)
        delta_phi_conv = abs(ll_phi[801] - ll_phi[401])
        delta_r_conv = abs(ll_adaptive[251] - ll_adaptive[151])

        print(f"\n  Checks:")
        print(f"    |Δ(adaptive n=251)| = {delta_r:.4f}  (<2 nats: "
              f"{'PASS' if delta_r < 2 else 'FAIL'})")
        print(f"    |Δ(phi G=801)|     = {delta_phi:.4f}  (<2 nats: "
              f"{'PASS' if delta_phi < 2 else 'FAIL'})")
        print(f"    |phi 801-401|      = {delta_phi_conv:.4f}  (<0.1 nats: "
              f"{'PASS' if delta_phi_conv < 0.1 else 'FAIL'})")
        print(f"    |r 251-151|        = {delta_r_conv:.4f}  (<0.5 nats: "
              f"{'PASS' if delta_r_conv < 0.5 else 'FAIL'})")

        n_fail += sum([delta_r >= 2, delta_phi >= 2,
                       delta_phi_conv >= 0.1, delta_r_conv >= 0.5])

    if n_fail > 0:
        print(f"\n{n_fail} check(s) FAILED.", flush=True)
    else:
        print("\nAll checks passed.", flush=True)


if __name__ == "__main__":
    main()
