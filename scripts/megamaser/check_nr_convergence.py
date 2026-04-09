"""Quick n_r convergence check: evaluate total logL at MAP for different n_r."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import tempfile
import tomli_w

from candel.model.integration import trapz_log_weights
from candel.model.model_H0_maser import MaserDiskModel, PC_PER_MAS_MPC
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import fprint, fsection, patch_tqdm
patch_tqdm()

# MAP parameters from the gf=1 run (647522)
D_A = 121.0
M_BH = 10**7.655
x0 = -45.7e-3
y0 = -98.2e-3
v_sys = 8333.0
i0 = jnp.deg2rad(84.95)
di_dr = jnp.deg2rad(11.5)
Omega0 = jnp.deg2rad(146.85)
dOmega_dr = jnp.deg2rad(-3.52)
sigma_x_floor2 = (11.0e-3)**2
sigma_y_floor2 = (3.5e-3)**2
var_v_sys = 2.3**2
var_v_hv = 5.7**2
sigma_a_floor2 = 0.073**2

data = load_megamaser_spots("data/Megamaser", "NGC5765b", v_sys_obs=8327.6)
n_spots = data["n_spots"]

PRIORS = {
    "H0": {"dist": "delta", "value": 73.0},
    "sigma_pec": {"dist": "delta", "value": 250.0},
    "D": {"dist": "uniform", "low": 50.0, "high": 200.0},
    "log_MBH": {"dist": "uniform", "low": 6.5, "high": 9.0},
    "eta": {"dist": "uniform", "low": 3.0, "high": 8.0},
    "R_phys": {"dist": "uniform", "low": 0.01, "high": 2.0},
    "x0": {"dist": "normal", "loc": 0.0, "scale": 150.0},
    "y0": {"dist": "normal", "loc": 0.0, "scale": 150.0},
    "i0": {"dist": "uniform", "low": 60.0, "high": 110.0},
    "Omega0": {"dist": "uniform", "low": 0.0, "high": 360.0},
    "dOmega_dr": {"dist": "uniform", "low": -20.0, "high": 20.0},
    "di_dr": {"dist": "uniform", "low": -30.0, "high": 30.0},
    "dv_sys": {"dist": "normal", "loc": 0.0, "scale": 25.0},
    "sigma_x_floor": {"dist": "truncated_normal", "mean": 10.0,
                      "scale": 5.0, "low": 0.0, "high": 100.0},
    "sigma_y_floor": {"dist": "truncated_normal", "mean": 10.0,
                      "scale": 5.0, "low": 0.0, "high": 100.0},
    "sigma_v_sys": {"dist": "truncated_normal", "mean": 2.0,
                    "scale": 5.0, "low": 0.0, "high": 100.0},
    "sigma_v_hv": {"dist": "truncated_normal", "mean": 2.0,
                   "scale": 5.0, "low": 0.0, "high": 100.0},
    "sigma_a_floor": {"dist": "truncated_normal", "mean": 0.1,
                      "scale": 0.2, "low": 0.0, "high": 0.75},
}


def build_model(n_r):
    config = {
        "inference": {"num_warmup": 1, "num_samples": 1, "num_chains": 1,
                      "chain_method": "sequential", "seed": 42,
                      "init_maxiter": 0, "max_tree_depth": 5,
                      "init_method": "median"},
        "model": {
            "which_run": "maser_disk", "Om": 0.315,
            "use_selection": False, "marginalise_r": True,
            "priors": PRIORS,
            "G_phi_half": 202, "n_inner_sys": 202,
            "inner_deg_sys": 30.0, "n_wing_sys": 100,
            "n_r": n_r,
        },
        "io": {"fname_output": "/dev/null"},
    }
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, tmp)
    tmp.close()
    m = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)
    return m


def eval_logL(m, r_ang_grid=None):
    if r_ang_grid is None:
        r_ang_grid = jnp.asarray(m.build_r_ang_grid(D_A))
    r_ang = r_ang_grid[None, :].repeat(n_spots, axis=0)
    log_w_r = trapz_log_weights(r_ang_grid)
    ll = m._eval_marginal_phi(
        r_ang, x0, y0, D_A, M_BH, v_sys,
        m._r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
        sigma_x_floor2, sigma_y_floor2,
        var_v_sys, var_v_hv, sigma_a_floor2,
        log_w_r=log_w_r,
    )
    return float(jnp.sum(jax.block_until_ready(ll)))


import candel.model.model_H0_maser as _maser_mod

fsection("n_r convergence: sinh vs log-uniform vs 2-zone (NGC5765b)")

nr_values = [101, 151, 201, 251, 302, 351, 401, 502]


R_LO = 0.01  # pc
R_HI = 2.0   # pc
r_ang_lo = R_LO / (D_A * PC_PER_MAS_MPC)
r_ang_hi = R_HI / (D_A * PC_PER_MAS_MPC)

# Build one model (grid size doesn't matter, we override it)
m0 = build_model(502)


def run_sweep(label, grid_fn):
    """Run convergence sweep with externally built grids."""
    g_ref = jnp.asarray(grid_fn(r_ang_lo, r_ang_hi, 2001))
    ll_ref = eval_logL(m0, g_ref)
    fprint(f"\n{label}: ref n_r=2001, logL={ll_ref:.4f}")
    print(f"  {'n_r':>6}  {'logL':>12}  {'delta':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}")

    for n_r in nr_values:
        g = jnp.asarray(grid_fn(r_ang_lo, r_ang_hi, n_r))
        ll = eval_logL(m0, g)
        print(f"  {n_r:>6}  {ll:>12.4f}  {ll - ll_ref:>+10.4f}")


def _log_uniform_grid(r_min, r_max, n_r):
    return np.geomspace(r_min, r_max, n_r)


def _sinh_grid(r_min, r_max, n_r, scale=0.3):
    logr_lo, logr_hi = np.log(r_min), np.log(r_max)
    logr_c = 0.5 * (logr_lo + logr_hi)
    t_lo = np.arcsinh((logr_lo - logr_c) / scale)
    t_hi = np.arcsinh((logr_hi - logr_c) / scale)
    t = np.linspace(t_lo, t_hi, n_r)
    return np.exp(logr_c + np.sinh(t) * scale)


# Spot-aware grid: dense near observed spot radii
r_spots = np.sqrt(np.asarray(data['x'])**2 + np.asarray(data['y'])**2)
r_spots = r_spots[r_spots > 0.001]
log_r_spots = np.log(r_spots)
spot_logr_center = np.median(log_r_spots)
spot_logr_spread = np.std(log_r_spots)


def _spot_aware_grid(r_min, r_max, n_r):
    """Sinh grid centered on observed spot radii distribution."""
    logr_lo, logr_hi = np.log(r_min), np.log(r_max)
    logr_c = spot_logr_center
    scale = spot_logr_spread * 0.5
    t_lo = np.arcsinh((logr_lo - logr_c) / scale)
    t_hi = np.arcsinh((logr_hi - logr_c) / scale)
    t = np.linspace(t_lo, t_hi, n_r)
    return np.exp(logr_c + np.sinh(t) * scale)


spot_scale = max(0.5 * spot_logr_spread, 0.3)
fprint(f"spot logr: center={spot_logr_center:.3f}, "
       f"spread={spot_logr_spread:.3f}, scale={spot_scale:.3f}")
fprint(f"r_ang range: [{r_ang_lo:.4f}, {r_ang_hi:.4f}] mas")


def _spot_aware_conservative(r_min, r_max, n_r):
    """Spot-aware sinh with conservative floor on scale."""
    logr_lo, logr_hi = np.log(r_min), np.log(r_max)
    logr_c = spot_logr_center
    scale = spot_scale
    t_lo = np.arcsinh((logr_lo - logr_c) / scale)
    t_hi = np.arcsinh((logr_hi - logr_c) / scale)
    t = np.linspace(t_lo, t_hi, n_r)
    return np.exp(logr_c + np.sinh(t) * scale)


# Also test: what the model itself builds
def _model_grid(r_min, r_max, n_r):
    """Use the model's build_r_ang_grid (checks consistency)."""
    # Temporarily override n_r
    old_n_r = m0._n_r
    m0._n_r = n_r
    g = m0.build_r_ang_grid(D_A)
    m0._n_r = old_n_r
    return g


run_sweep("SINH (generic, scale=0.3)", _sinh_grid)
run_sweep("LOG-UNIFORM", _log_uniform_grid)
run_sweep("SPOT-AWARE (conservative)", _spot_aware_conservative)
run_sweep("MODEL build_r_ang_grid", _model_grid)
