"""Full fit v2: NGC5765b with better r_ang initialization."""
import sys
sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import tempfile
import numpy as np
import jax.numpy as jnp
import tomli_w
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from jax import random
from scipy.optimize import minimize

from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.model.model_H0_maser import (
    MaserDiskModel, C_v, C_a, C_g, SPEED_OF_LIGHT, warp_geometry)
from candel.util import fprint, fsection

# ---- Load data ----
fsection("Loading NGC5765b data")
data = load_megamaser_spots("data/Megamaser", "NGC5765b", v_sys_obs=8327.6)

# ---- Initialize r_ang and phi per spot using Gao+2016 parameters ----
# Use the geometry from the downsampled fit (Pesce parameterization)
D_A_init = 129.0
M_BH_init = 4.78e7
v_sys_init = 8327.6
x0_init = -0.044
y0_init = -0.093
i0_init = np.deg2rad(73.0)
di_dr_init = np.deg2rad(12.7)
Omega0_init = np.deg2rad(149.2)
dOmega_dr_init = np.deg2rad(-2.7)

def spot_chi2(params, x_obs, y_obs, v_obs, a_obs, ex, ey, ea_obs,
              x0, y0, D_A, M_BH, v_sys, i0, di_dr, Omega0, dOmega_dr):
    r, phi = params
    if r < 0.02 or r > 5.0:
        return 1e10

    i_r = float(i0 + di_dr * r)
    Om_r = float(Omega0 + dOmega_dr * r)

    sin_p, cos_p = np.sin(phi), np.cos(phi)
    sin_O, cos_O = np.sin(Om_r), np.cos(Om_r)
    sin_i, cos_i = np.sin(i_r), np.cos(i_r)

    X = x0 + r * (sin_p * sin_O - cos_p * cos_O * cos_i)
    Y = y0 + r * (sin_p * cos_O + cos_p * sin_O * cos_i)

    v_kep = C_v * np.sqrt(M_BH / (r * D_A))
    v_z = v_kep * sin_p * sin_i
    beta = v_kep / SPEED_OF_LIGHT
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    V = SPEED_OF_LIGHT * (gamma * (1.0 + v_z / SPEED_OF_LIGHT) *
        (1.0 / np.sqrt(1.0 - C_g * M_BH / (r * D_A))) *
        (1.0 + v_sys / SPEED_OF_LIGHT) - 1.0)

    a_mag = C_a * M_BH / (r**2 * D_A**2)
    A = a_mag * cos_p * sin_i

    chi2 = (x_obs - X)**2 / max(ex**2, 1e-8) + (y_obs - Y)**2 / max(ey**2, 1e-8)
    chi2 += (v_obs - V)**2 / 5.0**2
    if ea_obs < 1.0:
        chi2 += (a_obs - A)**2 / max(ea_obs**2, 1e-4)
    return float(chi2)

fprint("Optimizing per-spot (r, phi) initialization...")
n = data["n_spots"]
r_init = np.zeros(n)
phi_init = np.zeros(n)

for k in range(n):
    is_hv = data["is_highvel"][k]
    is_blue = data.get("is_blue", np.zeros(n, dtype=bool))[k]

    # Determine phi search range
    if not is_hv:
        phi_range = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 11)
    elif is_blue:
        phi_range = np.linspace(np.pi + 0.01, 2*np.pi - 0.01, 11)
    else:
        phi_range = np.linspace(0.01, np.pi - 0.01, 11)

    best_chi2 = 1e10
    best_r, best_phi = 0.5, 0.0

    for r_start in [0.2, 0.5, 0.9, 1.3, 1.8]:
        for phi_start in phi_range:
            try:
                res = minimize(spot_chi2, [r_start, phi_start],
                             args=(data["x"][k], data["y"][k],
                                   data["velocity"][k], data["a"][k],
                                   data["sigma_x"][k], data["sigma_y"][k],
                                   data["sigma_a"][k],
                                   x0_init, y0_init, D_A_init, M_BH_init,
                                   v_sys_init, i0_init, di_dr_init,
                                   Omega0_init, dOmega_dr_init),
                             method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-6})
                if res.fun < best_chi2:
                    best_chi2 = res.fun
                    best_r = res.x[0]
                    best_phi = res.x[1]
            except:
                pass

    r_init[k] = np.clip(best_r, 0.03, 4.5)

fprint(f"r_init range: [{r_init.min():.4f}, {r_init.max():.4f}] mas")
fprint(f"Median chi2 per spot: {np.median([spot_chi2([r_init[k], phi_init[k]], data['x'][k], data['y'][k], data['velocity'][k], data['a'][k], data['sigma_x'][k], data['sigma_y'][k], data['sigma_a'][k], x0_init, y0_init, D_A_init, M_BH_init, v_sys_init, i0_init, di_dr_init, Omega0_init, dOmega_dr_init) for k in range(n)]):.1f}")

# ---- Config ----
config = {
    "inference": {
        "num_warmup": 500,
        "num_samples": 1000,
        "num_chains": 1,
        "chain_method": "sequential",
        "seed": 42,
        "dense_mass_blocks": [
            ["D_c", "log_MBH", "dv_sys"],
            ["i0", "di_dr"],
            ["Omega0", "dOmega_dr"],
            ["x0", "y0"],
        ],
        "init_maxiter": 0,
        "max_tree_depth": 10,
    },
    "model": {
        "which_run": "maser_disk",
        "Om": 0.315,
        "use_selection": False,
        "fit_di_dr": True,
        "marginalise_r": False,
        "priors": {
            "H0": {"dist": "delta", "value": 73.0},
            "sigma_pec": {"dist": "delta", "value": 250.0},
            "D": {"dist": "uniform", "low": 50.0, "high": 200.0},
            "log_MBH": {"dist": "uniform", "low": 6.5, "high": 9.0},
            "R_phys": {"dist": "uniform", "low": 0.01, "high": 3.0},
            "x0": {"dist": "uniform", "low": -500.0, "high": 500.0},
            "y0": {"dist": "uniform", "low": -500.0, "high": 500.0},
            "i0": {"dist": "uniform", "low": 60.0, "high": 110.0},
            "Omega0": {"dist": "uniform", "low": 100.0, "high": 200.0},
            "dOmega_dr": {"dist": "uniform", "low": -30.0, "high": 30.0},
            "di_dr": {"dist": "uniform", "low": -30.0, "high": 30.0},
            "dv_sys": {"dist": "normal", "loc": 0.0, "scale": 500.0},
            "sigma_x_floor": {"dist": "truncated_normal",
                              "mean": 10.0, "scale": 10.0,
                              "low": 0.0, "high": 100.0},
            "sigma_y_floor": {"dist": "truncated_normal",
                              "mean": 10.0, "scale": 10.0,
                              "low": 0.0, "high": 100.0},
            "sigma_v_sys": {"dist": "truncated_normal",
                            "mean": 2.0, "scale": 2.0,
                            "low": 0.0, "high": 20.0},
            "sigma_v_hv": {"dist": "truncated_normal",
                           "mean": 2.0, "scale": 2.0,
                           "low": 0.0, "high": 20.0},
            "sigma_a_floor": {"dist": "uniform",
                              "low": 0.0, "high": 5.0},
        },
    },
    "io": {
        "fname_output": "results/Maser/NGC5765b_v2.hdf5",
    },
}

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()

model = MaserDiskModel(tmp.name, data)

# Clip r_init to model bounds
r_init = np.clip(r_init, model._r_ang_lo + 0.001, model._r_ang_hi - 0.001)

z_est = 8327.6 / 299792.458
D_c_init = 129.0 * (1 + z_est)

init_values = {
    'D_c': jnp.array(D_c_init),
    'log_MBH': jnp.array(np.log10(M_BH_init)),
    'x0': jnp.array(x0_init * 1e3),    # mas -> uas
    'y0': jnp.array(y0_init * 1e3),
    'i0': jnp.array(73.0),
    'di_dr': jnp.array(12.7),
    'Omega0': jnp.array(149.2),
    'dOmega_dr': jnp.array(-2.7),
    'dv_sys': jnp.array(5.0),
    'sigma_x_floor': jnp.array(12.0),
    'sigma_y_floor': jnp.array(2.0),
    'sigma_v_sys': jnp.array(1.8),
    'sigma_v_hv': jnp.array(3.6),
    'sigma_a_floor': jnp.array(0.08),
    'r_ang': jnp.array(r_init),
}

# Run
fsection("Running NUTS (full 192 spots, better init)")
kernel = NUTS(model, max_tree_depth=10, target_accept_prob=0.8,
              init_strategy=init_to_value(values=init_values))
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000,
            num_chains=1, progress_bar=True)
mcmc.run(random.PRNGKey(42))
mcmc.print_summary(exclude_deterministic=True)

samples = mcmc.get_samples()
n_div = int(mcmc.get_extra_fields()['diverging'].sum())
print(f"\nDivergences: {n_div}")

# Report
fsection("Results")
for k in ['D_c', 'log_MBH', 'i0', 'di_dr', 'Omega0', 'dOmega_dr',
          'x0', 'y0', 'dv_sys',
          'sigma_x_floor', 'sigma_y_floor',
          'sigma_v_sys', 'sigma_v_hv', 'sigma_a_floor']:
    if k in samples:
        s = np.asarray(samples[k])
        print(f"  {k:20s} = {s.mean():10.3f} +/- {s.std():8.3f}  "
              f"[{np.percentile(s, 16):.3f}, {np.percentile(s, 84):.3f}]")

if 'D_c' in samples:
    D_c = np.asarray(samples['D_c'])
    z_cosmo = D_c * 73.0 / 299792.458
    D_A = D_c / (1 + z_cosmo)
    M_BH = 10**np.asarray(samples['log_MBH'])
    H0_implied = 299792.458 * z_cosmo.mean() / D_A.mean()
    print(f"\n  D_A                  = {D_A.mean():10.3f} +/- {D_A.std():8.3f}")
    print(f"  M_BH                 = {M_BH.mean():.2e} +/- {M_BH.std():.2e}")
    print(f"  H0 (implied)         = {H0_implied:.1f}")

save_dict = {k: np.asarray(samples[k]) for k in samples if k != 'r_ang'}
save_dict['D_A'] = np.asarray(D_A)
save_dict['M_BH'] = np.asarray(M_BH)
np.savez("results/Maser/NGC5765b_v2_samples.npz", **save_dict)
print("Saved to results/Maser/NGC5765b_v2_samples.npz")

print("\n  Gao+2016: D_A=126.3±11.6, M_BH=4.55e7")
print("  Pesce+2020: D_A=112.2+5.4/-5.1")
print("  80-spot test: D_A=129.2±9.1")

os.unlink(tmp.name)
