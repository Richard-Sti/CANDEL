"""Convergence test for the phi (and r) numerical integrals.

Calls _eval_marginal_phi directly with fixed representative parameters —
no numpyro, no parameter transforms. Sweeps grid sizes and reports the
per-spot log-likelihood and its deviation from the reference (finest) grid.

Tests the two-zone systemic grid (dense arcsin inner + sparse linear wings)
and the arccos HV half-grid at various resolutions, using a uniform grid
as the ground-truth reference.

Usage:
    python scripts/megamaser/convergence_grids.py
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import candel.model.model_H0_maser as _maser_mod
from candel.model.model_H0_maser import (
    MaserDiskModel, _build_phi_half_grid_hv, _build_phi_grid_sys,
    _build_r_grid, warp_geometry,
)
from candel.model.integration import trapz_log_weights
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import fprint, fsection, patch_tqdm
patch_tqdm()

# ── Representative parameter values (NGC5765b posterior centre) ──────────────
D_A    = 121.0          # Mpc
M_BH   = 10**7.655      # M_sun
x0     = -45.7e-3       # mas  (sampled in uas → mas)
y0     = -98.2e-3       # mas
v_sys  = 8333.0         # km/s
i0     = jnp.deg2rad(84.95)
di_dr  = jnp.deg2rad(11.5)
Omega0 = jnp.deg2rad(146.85)
dOmega_dr = jnp.deg2rad(-3.52)
r_ang_ref  = 0.970      # mas  (pivot)
sigma_x_floor2 = (11.0e-3)**2
sigma_y_floor2 = (3.5e-3)**2
var_v_sys      = 2.3**2
var_v_hv       = 5.7**2
sigma_a_floor2 = 0.073**2

# ── Load NGC5765b spots ───────────────────────────────────────────────────────
data   = load_megamaser_spots("data/Megamaser", "NGC5765b", v_sys_obs=8327.6)
n_spots = data["n_spots"]
fprint(f"Loaded {n_spots} spots")

# ── Helper: build r_ang grid for Mode 2 ──────────────────────────────────────
R_MIN, R_MAX = 0.01, 3.0   # pc

PRIORS = {
    "H0":          {"dist": "delta", "value": 73.0},
    "sigma_pec":   {"dist": "delta", "value": 250.0},
    "D":           {"dist": "uniform", "low": 50.0, "high": 200.0},
    "log_MBH":     {"dist": "uniform", "low": 6.5,  "high": 9.0},
    "eta":         {"dist": "uniform", "low": 3.0,  "high": 8.0},
    "R_phys":      {"dist": "uniform", "low": R_MIN, "high": R_MAX},
    "x0":          {"dist": "normal",  "loc": 0.0, "scale": 150.0},
    "y0":          {"dist": "normal",  "loc": 0.0, "scale": 150.0},
    "i0":          {"dist": "uniform", "low": 60.0, "high": 110.0},
    "Omega0":      {"dist": "uniform", "low": 0.0,  "high": 360.0},
    "dOmega_dr":   {"dist": "uniform", "low": -20.0,"high": 20.0},
    "di_dr":       {"dist": "uniform", "low": -30.0,"high": 30.0},
    "dv_sys":      {"dist": "normal",  "loc": 0.0,  "scale": 25.0},
    "sigma_x_floor": {"dist": "truncated_normal", "mean": 10.0,
                      "scale": 5.0, "low": 0.0, "high": 100.0},
    "sigma_y_floor": {"dist": "truncated_normal", "mean": 10.0,
                      "scale": 5.0, "low": 0.0, "high": 100.0},
    "sigma_v_sys":   {"dist": "truncated_normal", "mean": 2.0,
                      "scale": 5.0, "low": 0.0, "high": 100.0},
    "sigma_v_hv":    {"dist": "truncated_normal", "mean": 2.0,
                      "scale": 5.0, "low": 0.0, "high": 100.0},
    "sigma_a_floor": {"dist": "truncated_normal", "mean": 0.1,
                      "scale": 0.2, "low": 0.0, "high": 0.3},
}


def _make_config(model_extras):
    import tempfile, tomli_w
    config = {
        "inference": {"num_warmup": 1, "num_samples": 1, "num_chains": 1,
                      "chain_method": "sequential", "seed": 42,
                      "init_maxiter": 0, "max_tree_depth": 5,
                      "init_method": "median"},
        "model": {
            "which_run": "maser_disk", "Om": 0.315,
            "use_selection": False, "marginalise_r": True,
            "priors": PRIORS,
            **model_extras,
        },
        "io": {"fname_output": "/dev/null"},
    }
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, tmp); tmp.close()
    m = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)
    return m


def _uniform_sys_grid(G=2001):
    """Uniform grid on [-pi/2, pi/2] for reference."""
    return np.linspace(-np.pi / 2, np.pi / 2, G)


def build_model(G_phi_half, n_inner_sys, inner_deg_sys, n_wing_sys, n_r,
                uniform_sys=None, uniform_hv=None):
    """Build model, optionally monkey-patching grids for uniform reference."""
    orig_sys = _maser_mod._build_phi_grid_sys
    orig_hv = _maser_mod._build_phi_half_grid_hv

    if uniform_sys is not None:
        _maser_mod._build_phi_grid_sys = lambda **kw: _uniform_sys_grid(uniform_sys)
    if uniform_hv is not None:
        _maser_mod._build_phi_half_grid_hv = lambda **kw: np.linspace(0, np.pi / 2, uniform_hv)

    m = _make_config({
        "G_phi_half": G_phi_half,
        "n_inner_sys": n_inner_sys, "inner_deg_sys": inner_deg_sys,
        "n_wing_sys": n_wing_sys, "n_r": n_r,
    })

    _maser_mod._build_phi_grid_sys = orig_sys
    _maser_mod._build_phi_half_grid_hv = orig_hv
    return m


# ── Spot type indices ────────────────────────────────────────────────────────
is_hv   = np.asarray(data["is_highvel"])
is_blue = np.asarray(data.get("is_blue", np.zeros(n_spots, bool)))
is_red  = is_hv & ~is_blue
accel   = np.asarray(data.get("accel_measured"))
idx_sys = np.where(~is_hv)[0]
idx_red = np.where(is_red)[0]
idx_blue = np.where(is_blue)[0]
fprint(f"Spot counts: {len(idx_sys)} sys, {len(idx_red)} red, {len(idx_blue)} blue")


def eval_integrals(m):
    from candel.model.integration import trapz_log_weights
    from candel.model.model_H0_maser import PC_PER_MAS_MPC
    r_ang_grid = m._R_phys_grid / (D_A * PC_PER_MAS_MPC)
    r_ang   = r_ang_grid[None, :].repeat(n_spots, axis=0)
    log_w_r = trapz_log_weights(r_ang_grid)
    ll = m._eval_marginal_phi(
        r_ang, x0, y0, D_A, M_BH, v_sys,
        m._r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor2, sigma_y_floor2,
        var_v_sys, var_v_hv, sigma_a_floor2,
        log_w_r=log_w_r,
    )
    return np.asarray(jax.block_until_ready(ll))


N_R = 501


def print_comparison(label, n_sys, n_hv, ll, ll_ref):
    """Print per-type summary: mean |Δ|, max |Δ|, and total Δ."""
    delta = ll - ll_ref
    row = f"  {label:<35}  {n_sys:>5}  {n_hv:>5}"
    for idx, name in [(idx_sys, "sys"), (idx_red, "red"), (idx_blue, "blue")]:
        d = delta[idx]
        row += f"  {np.mean(np.abs(d)):8.4f}  {np.max(np.abs(d)):8.4f}  {d.sum():+10.4f}"
    row += f"  {delta.sum():+12.4f}"
    print(row)


# ── 1. Build uniform reference ──────────────────────────────────────────────
fsection("Building uniform reference (sys=2001, hv=1001)")
m_ref = build_model(501, 31, 5.0, 10, N_R, uniform_sys=2001, uniform_hv=1001)
ll_ref = eval_integrals(m_ref)
fprint(f"Reference total logL = {ll_ref.sum():.4f}")


# ── 2. Two-zone systemic sweep (fix HV at uniform 1001) ─────────────────────
fsection("Systemic grid sweep (HV uniform=1001 fixed)")
hdr = (f"  {'Config':<35}  {'#sys':>5}  {'#hv':>5}"
       + "".join(f"  {'<|Δ|>_'+n:>8}  {'max|Δ|_'+n:>8}  {'Σ_'+n:>10}"
                 for n in ["sys", "red", "blue"])
       + f"  {'ΔlogL_tot':>12}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

# Reference line
print_comparison("uniform 2001 (ref)", 2001, 1001, ll_ref, ll_ref)

sys_configs = [
    # (n_inner, inner_deg, n_wing, label)
    (201, 5.0,  50, "two-zone 5° 301pts"),
    (101, 5.0,  30, "two-zone 5° 161pts"),
    (31,  5.0,  10, "two-zone 5° 51pts"),
    (31,  7.0,  10, "two-zone 7° 51pts"),
    (31,  10.0, 10, "two-zone 10° 51pts"),
    (31,  15.0, 10, "two-zone 15° 51pts"),
    (31,  20.0, 10, "two-zone 20° 51pts"),
    (51,  10.0, 15, "two-zone 10° 81pts"),
    (51,  15.0, 15, "two-zone 15° 81pts"),
    (51,  20.0, 15, "two-zone 20° 81pts"),
    (21,  10.0,  5, "two-zone 10° 31pts"),
    (21,  15.0,  5, "two-zone 15° 31pts"),
]

for n_inn, deg, n_w, label in sys_configs:
    m = build_model(501, n_inn, deg, n_w, N_R, uniform_hv=1001)
    ll = eval_integrals(m)
    n_sys = len(np.asarray(m._phi_sys))
    print_comparison(label, n_sys, 1001, ll, ll_ref)


# ── 3. HV grid sweep (fix systemic at uniform 2001) ─────────────────────────
fsection("HV grid sweep (sys uniform=2001 fixed)")
hdr = (f"  {'Config':<35}  {'#sys':>5}  {'#hv':>5}"
       + "".join(f"  {'<|Δ|>_'+n:>8}  {'max|Δ|_'+n:>8}  {'Σ_'+n:>10}"
                 for n in ["sys", "red", "blue"])
       + f"  {'ΔlogL_tot':>12}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

print_comparison("uniform 1001 (ref)", 2001, 1001, ll_ref, ll_ref)

for G_half in [501, 251, 101, 51, 31, 21]:
    m = build_model(G_half, 31, 5.0, 10, N_R, uniform_sys=2001)
    ll = eval_integrals(m)
    n_hv = len(np.asarray(m._phi_half))
    print_comparison(f"arccos G_half={G_half}", 2001, n_hv, ll, ll_ref)


# ── 4. Combined: proposed defaults vs reference ─────────────────────────────
fsection("Combined: proposed defaults vs uniform reference")
hdr = (f"  {'Config':<35}  {'#sys':>5}  {'#hv':>5}"
       + "".join(f"  {'<|Δ|>_'+n:>8}  {'max|Δ|_'+n:>8}  {'Σ_'+n:>10}"
                 for n in ["sys", "red", "blue"])
       + f"  {'ΔlogL_tot':>12}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

print_comparison("uniform 2001/1001 (ref)", 2001, 1001, ll_ref, ll_ref)

combined = [
    (51, 31, 5.0,  10, "proposed: 5° 51sys 52hv"),
    (51, 31, 10.0, 10, "proposed: 10° 51sys 52hv"),
    (51, 31, 15.0, 10, "proposed: 15° 51sys 52hv"),
    (51, 51, 10.0, 15, "larger: 10° 81sys 52hv"),
    (51, 51, 15.0, 15, "larger: 15° 81sys 52hv"),
]

for G_half, n_inn, deg, n_w, label in combined:
    m = build_model(G_half, n_inn, deg, n_w, N_R)
    ll = eval_integrals(m)
    n_sys = len(np.asarray(m._phi_sys))
    n_hv = len(np.asarray(m._phi_half))
    print_comparison(label, n_sys, n_hv, ll, ll_ref)
