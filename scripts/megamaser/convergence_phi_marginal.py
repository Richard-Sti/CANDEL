"""
Phi-marginal convergence benchmark for the megamaser disk model (Mode 1).

At fixed global parameters and several per-spot r_ang values, tests that
uniform-grid phi integration converges. Uses brute-force trapezoidal
integration on [0, 2pi] at increasing resolution.

Usage:
    python scripts/megamaser/convergence_phi_marginal.py
    python scripts/megamaser/convergence_phi_marginal.py --galaxies NGC4258
"""
import argparse
import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import tomli
import tomli_w

from candel.model.model_H0_maser import (
    MaserDiskModel, C_v, C_a,
)
from candel.model.integration import trapz_log_weights
from candel.pvdata.megamaser_data import load_megamaser_spots

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
ALL_GALAXIES = ["NGC5765b", "UGC3789", "CGCG074-064", "NGC6264", "NGC6323",
                "NGC4258"]

N_PHI_REF = 200001
N_PHI_TEST = [101, 201, 501, 1001, 5001, 10001, 50001, 100001]
R_SCALES = [0.5, 1.0, 2.0]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def build_model(galaxy, master_cfg):
    cfg = master_cfg.copy()
    cfg["model"] = master_cfg["model"].copy()
    cfg["model"]["marginalise_r"] = False
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


def estimate_r_ang(model, phys):
    """Per-spot r_ang centering estimates from velocity/acceleration."""
    D_A, M_BH, v_sys = phys["D_A"], phys["M_BH"], phys["v_sys"]
    sin_i = np.abs(np.sin(phys["i0"]))
    r_min = float(model._r_ang_lo)
    r_max = float(model._r_ang_hi)
    eps = 1e-30

    dv = np.asarray(model._all_v) - v_sys
    r_vel = M_BH * (C_v * sin_i) ** 2 / (D_A * (dv ** 2 + eps))
    r_vel = np.clip(r_vel, r_min, r_max)

    a = np.asarray(model._all_a)
    r_acc = np.sqrt(C_a * M_BH * sin_i / (D_A ** 2 * (np.abs(a) + eps)))
    r_acc = np.clip(r_acc, r_min, r_max)

    sigma_a_total = np.sqrt(
        np.asarray(model._all_sigma_a) ** 2 + phys["sigma_a_floor2"])
    accel_snr = np.abs(a) / (sigma_a_total + eps)
    accel_good = np.asarray(model._all_has_accel) & (accel_snr >= 2.0)

    r_mid = np.exp(0.5 * (np.log(r_min) + np.log(r_max)))
    is_hv = np.asarray(model.is_highvel)
    r_est = np.where(is_hv, r_vel,
                     np.where(accel_good, r_acc, r_mid))
    return np.clip(r_est, r_min * 1.01, r_max * 0.99)


def eval_phi_ll(model, phys, r_ang, n_phi):
    """Sum of per-spot phi-marginal logL on uniform grid."""
    phi = jnp.linspace(0.0, 2 * jnp.pi, n_phi)
    log_w = trapz_log_weights(phi)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)

    ll = model._phi_integrand(
        r_ang, sin_phi, cos_phi, *_model_args(model, phys))
    return float(jnp.sum(logsumexp(ll + log_w[None, :], axis=-1)))


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
    # x64 for precise reference comparisons
    jax.config.update("jax_enable_x64", True)
    dtype = "float64" if args.float64 else "float64 (convergence test)"
    print(f"JAX platform: {jax.default_backend()}, dtype: {dtype}",
          flush=True)

    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)
    galaxies_cfg = master_cfg["model"]["galaxies"]

    print("=" * 70)
    print("Phi-marginal convergence (uniform brute-force grids)")
    print(f"Reference: {N_PHI_REF} points on [0, 2π]")
    print(f"Test grids: {N_PHI_TEST}")
    print(f"r_ang scales: {R_SCALES}")
    print(f"Precision: {dtype}")
    print("=" * 70)

    n_fail = 0
    for galaxy in args.galaxies:
        print(f"\n{'─' * 60}")
        print(f"Galaxy: {galaxy}")
        print(f"{'─' * 60}")

        model = build_model(galaxy, master_cfg)
        phys = extract_phys(galaxy, galaxies_cfg, model)
        r_est = estimate_r_ang(model, phys)
        print(f"  D_A={phys['D_A']:.1f} Mpc, n_spots={model.n_spots}")

        for scale in R_SCALES:
            r_ang = jnp.clip(jnp.asarray(r_est * scale),
                             model._r_ang_lo, model._r_ang_hi)
            print(f"\n  r_ang scale = {scale:.1f}x  "
                  f"(median = {float(jnp.median(r_ang)):.4f} mas)")

            print(f"  Reference: n_phi={N_PHI_REF}, "
                  f"test: {N_PHI_TEST}")
            ll_ref = eval_phi_ll(model, phys, r_ang, N_PHI_REF)
            print(f"  Reference logL = {ll_ref:.4f}")

            print(f"  {'n_phi':>12}  {'logL':>14}  {'Δ':>10}")
            ll_vals = {}
            for n_phi in N_PHI_TEST:
                ll = eval_phi_ll(model, phys, r_ang, n_phi)
                ll_vals[n_phi] = ll
                print(f"  {n_phi:>12}  {ll:14.4f}  {ll - ll_ref:+10.4f}")

            n_max = N_PHI_TEST[-1]
            n_prev = N_PHI_TEST[-2]
            d_fine = abs(ll_vals[n_max] - ll_ref)
            d_conv = abs(ll_vals[n_max] - ll_vals[n_prev])
            print(f"  Checks:")
            print(f"    |Δ(n={n_max})| = {d_fine:.4f}  (<2 nats: "
                  f"{'PASS' if d_fine < 2 else 'FAIL'})")
            print(f"    |{n_max}-{n_prev}| = {d_conv:.4f}  (<0.1 nats: "
                  f"{'PASS' if d_conv < 0.1 else 'FAIL'})")
            n_fail += (d_fine >= 2) + (d_conv >= 0.1)

    if n_fail > 0:
        print(f"\n{n_fail} check(s) FAILED.", flush=True)
    else:
        print("\nAll checks passed.", flush=True)


if __name__ == "__main__":
    main()
