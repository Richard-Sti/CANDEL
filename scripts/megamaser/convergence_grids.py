"""Convergence test for the phi (and r) numerical integrals.

Calls _eval_marginal_phi directly with fixed representative parameters —
no numpyro, no parameter transforms. Sweeps grid sizes and reports the
per-spot log-likelihood and its deviation from the reference (finest) grid.

Compares two HV phi grids:
  OLD: arcsin-spaced (uniform in sin(phi)) — dense near phi=0
  NEW: arccos-spaced (uniform in cos(phi)) — dense near phi=pi/2

Usage:
    python scripts/megamaser/convergence_grids.py
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import numpy as _np_old

from candel.model.model_H0_maser import (
    MaserDiskModel, _build_phi_half_grid_hv, _build_phi_grid_sys,
    _build_r_grid, warp_geometry,
)

# Old arcsin HV grid for comparison
def _build_phi_half_grid_hv_old(G_half=251, s_min=0.0001, s_max=0.999, n_patch=8):
    """Original arcsin-spaced HV half-grid (dense near phi=0)."""
    s = _np_old.linspace(s_min, s_max, G_half)
    phi = _np_old.arcsin(s)
    phi_cut = phi[-(n_patch + 1)]
    phi[-n_patch:] = _np_old.linspace(phi_cut, _np_old.pi / 2, n_patch + 2)[1:-1]
    phi = _np_old.append(phi, _np_old.pi / 2)
    return phi
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

def build_model_shell(G_phi_half, G_phi_sys, n_r, use_old_hv_grid=False):
    """Return a minimal MaserDiskModel with the requested grid sizes."""
    import candel.model.model_H0_maser as _maser_mod
    import tempfile, tomli_w

    # Temporarily swap the HV grid builder if testing the old spacing
    if use_old_hv_grid:
        _orig = _maser_mod._build_phi_half_grid_hv
        _maser_mod._build_phi_half_grid_hv = _build_phi_half_grid_hv_old
    config = {
        "inference": {"num_warmup": 1, "num_samples": 1, "num_chains": 1,
                      "chain_method": "sequential", "seed": 42,
                      "init_maxiter": 0, "max_tree_depth": 5,
                      "init_method": "median"},
        "model": {
            "which_run": "maser_disk", "Om": 0.315,
            "use_selection": False, "marginalise_r": True,
            "G_phi_half": G_phi_half, "G_phi_sys": G_phi_sys, "n_r": n_r,
            "priors": {
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
            },
        },
        "io": {"fname_output": "/dev/null"},
    }
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, tmp); tmp.close()
    m = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)

    if use_old_hv_grid:
        _maser_mod._build_phi_half_grid_hv = _orig

    return m


# ── One representative spot per type (with measured acceleration) ─────────────
is_hv   = np.asarray(data["is_highvel"])
is_blue = np.asarray(data.get("is_blue", np.zeros(n_spots, bool)))
is_red  = is_hv & ~is_blue
accel   = np.asarray(data.get("accel_measured"))
SPOT_INDICES = {
    "sys":  int(np.where(~is_hv & accel)[0][0]),
    "red":  int(np.where(is_red  & accel)[0][0]),
    "blue": int(np.where(is_blue & accel)[0][0]),
}
fprint("Representative spots: " +
       ", ".join(f"{k}=#{v}" for k, v in SPOT_INDICES.items()))


def eval_integrals(m):
    """Return per-spot log-likelihoods for all spots using model's grids.

    _eval_marginal_phi uses pre-split internal index arrays so we run all
    spots and extract the three representative ones afterwards.
    """
    r_ang   = m._r_ang_grid[None, :].repeat(n_spots, axis=0)  # (N, n_r)
    log_w_r = jnp.asarray(trapz_log_weights(np.asarray(m._r_ang_grid)))

    ll = m._eval_marginal_phi(
        r_ang, x0, y0, D_A, M_BH, v_sys,
        m._r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor2, sigma_y_floor2,
        var_v_sys, var_v_hv, sigma_a_floor2,
        log_w_r=log_w_r,
    )
    return np.asarray(jax.block_until_ready(ll))


# ── Grid sweep ────────────────────────────────────────────────────────────────
# (G_phi_half, G_phi_sys, n_r)
grids = [
    (1001, 2001, 1001),  # reference (finest)
    (501, 1001, 501),
    (251, 501,  251),
    (151, 301,  151),
    (101, 201,  101),
    (71,  141,  71),
    (51,  101,  51),
    (31,  61,   31),
]

def run_sweep(label, use_old_hv_grid):
    fsection(f"Grid convergence — {label} HV grid")
    hdr = f"  {'Grid (half/sys/r)':<24}" + "".join(
        f"  {'logL_'+n:>12}  {'ΔlogL_'+n:>10}" for n in SPOT_INDICES)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    ll_ref = None
    for G_half, G_sys, n_r in grids:
        m   = build_model_shell(G_half, G_sys, n_r, use_old_hv_grid=use_old_hv_grid)
        ll  = eval_integrals(m)
        lbl = f"{G_half}/{G_sys}/{n_r}"
        row = f"  {lbl:<24}"
        for spot_name, spot_idx in SPOT_INDICES.items():
            v = ll[spot_idx]
            if ll_ref is None:
                row += f"  {v:12.4f}  {'(ref)':>10}"
            else:
                row += f"  {v:12.4f}  {v - ll_ref[spot_idx]:+10.4f}"
        print(row)
        if ll_ref is None:
            ll_ref = ll

run_sweep("OLD arcsin", use_old_hv_grid=True)
run_sweep("NEW arccos", use_old_hv_grid=False)
