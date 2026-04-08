# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Mock generator for maser disk (spot-level) forward model.

Reuses physics functions from candel.model.model_H0_maser.
"""
import numpy as np

from ..util import fprint, SPEED_OF_LIGHT
from ..cosmo.cosmography import Distance2Redshift
from ..model.model_H0_maser import (
    PC_PER_MAS_MPC, predict_position, predict_velocity_los,
    predict_acceleration_los, warp_geometry)


DEFAULT_TRUE_PARAMS = {
    "H0": 73.0,
    "sigma_pec": 250.0,
    "D_c": 90.0,            # Mpc (comoving distance)
    "M_BH": 2.42e7,         # solar masses
    "x0": 0.0013,           # mas
    "y0": 0.0075,           # mas
    "i0": 90.8,             # degrees
    "Omega0": 99.6,         # degrees
    "dOmega_dr": 2.0,       # degrees/mas
    "di_dr": 0.0,           # degrees/mas (no inclination warp)
    "sigma_x_floor": 0.002,       # mas
    "sigma_y_floor": 0.017,       # mas
    "sigma_v_sys": 4.8,           # km/s
    "sigma_v_hv": 4.3,            # km/s
    "sigma_a_floor": 0.43,         # km/s/yr
    "sigma_x": 0.05,             # mas (per-spot measurement)
    "sigma_y": 0.05,             # mas (per-spot measurement)
    "sigma_v": 1.0,              # km/s (per-spot velocity measurement)
    "sigma_a": 0.5,              # km/s/yr (per-spot measurement)
}


# -----------------------------------------------------------------------
# Spot generation helpers
# -----------------------------------------------------------------------

def _draw_spot_types(n_spots, rng):
    """Assign spot types matching CGCG 074-064 fractions."""
    probs = np.array([0.43, 0.30, 0.27])  # r, b, s
    types = rng.choice(["r", "b", "s"], size=n_spots, p=probs)
    return types


def _draw_radii_phys(n_spots, rng, R_min=0.01, R_max=1.5):
    """Draw physical orbital radii from uniform distribution in pc."""
    return rng.uniform(R_min, R_max, n_spots)


def _draw_phi(spot_types, rng):
    """Draw azimuthal angles from type-dependent uniform priors."""
    phi = np.empty(len(spot_types))
    for i, stype in enumerate(spot_types):
        if stype == "r":
            phi[i] = rng.uniform(0, np.pi)
        elif stype == "b":
            phi[i] = rng.uniform(np.pi, 2 * np.pi)
        else:  # systemic
            phi[i] = rng.uniform(-np.pi / 2, np.pi / 2)
    return phi


# -----------------------------------------------------------------------
# Main mock generator
# -----------------------------------------------------------------------


def gen_maser_disk_mock(seed, true_params=None, n_spots=50, Om0=0.315,
                        verbose=True):
    """Generate a mock maser disk dataset at the spot level.

    Samples true latent parameters (distance, disk geometry, spot positions),
    computes observables using the same physics as the forward model, adds
    measurement noise.

    Parameters
    ----------
    seed
        Random seed.
    true_params
        Override default true parameters (merged with DEFAULT_TRUE_PARAMS).
    n_spots
        Number of spots to generate.
    Om0
        Matter density parameter for the distance-redshift relation.
    verbose
        Print summary statistics.

    Returns
    -------
    data : dict
        Data dict matching what MaserDiskModel expects.
    true_params_expanded : dict
        All true values including per-spot r, phi.
    """
    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
    rng = np.random.default_rng(seed)

    H0 = tp["H0"]
    sigma_pec = tp["sigma_pec"]
    D_c = tp["D_c"]
    M_BH = tp["M_BH"]
    x0 = tp["x0"]
    y0 = tp["y0"]

    i0_rad = np.deg2rad(tp["i0"])
    di_dr_rad = np.deg2rad(tp["di_dr"])
    Omega0_rad = np.deg2rad(tp["Omega0"])
    dOmega_dr_rad = np.deg2rad(tp["dOmega_dr"])

    # ---- Cosmological redshift from D_c and H0 ----
    h = H0 / 100.0
    dc2z = Distance2Redshift(Om0=Om0)
    z_cosmo = float(dc2z(np.atleast_1d(D_c), h=h)[0])
    D_A = D_c / (1 + z_cosmo)

    # ---- Add peculiar velocity (exact relativistic composition) ----
    v_pec = rng.normal(0, sigma_pec)
    z_obs = (1 + z_cosmo) * (1 + v_pec / SPEED_OF_LIGHT) - 1
    v_sys_cmb = SPEED_OF_LIGHT * z_obs

    if verbose:
        fprint(f"D_c = {D_c:.1f} Mpc, D_A = {D_A:.1f} Mpc, "
               f"z_cosmo = {z_cosmo:.6f}")
        fprint(f"v_pec = {v_pec:.1f} km/s, v_sys_cmb = {v_sys_cmb:.1f} km/s")

    # ---- Generate maser spots ----
    spot_types = _draw_spot_types(n_spots, rng)
    R_phys_true = _draw_radii_phys(n_spots, rng)
    r_ang_true = R_phys_true / (D_A * PC_PER_MAS_MPC)
    phi_true = _draw_phi(spot_types, rng)

    # Warped geometry at each spot (using angular radius)
    r_ang_ref = float(np.median(r_ang_true))
    i_k, Omega_k = np.array(warp_geometry(
        r_ang_true, r_ang_ref, i0_rad, di_dr_rad,
        Omega0_rad, dOmega_dr_rad))

    # True observables (using angular radius and D_A)
    X_true, Y_true = np.array(predict_position(
        r_ang_true, phi_true, x0, y0, i_k, Omega_k))
    V_true = np.array(predict_velocity_los(
        r_ang_true, phi_true, D_A, M_BH, v_sys_cmb, i_k))
    A_true = np.array(predict_acceleration_los(
        r_ang_true, phi_true, D_A, M_BH, i_k))

    # ---- Measurement noise ----
    sigma_x_obs = np.full(n_spots, tp["sigma_x"])
    sigma_y_obs = np.full(n_spots, tp["sigma_y"])
    sigma_a_obs = np.full(n_spots, tp["sigma_a"])

    sigma_x_total = np.sqrt(tp["sigma_x"]**2 + tp["sigma_x_floor"]**2)
    sigma_y_total = np.sqrt(tp["sigma_y"]**2 + tp["sigma_y_floor"]**2)
    x_obs = X_true + rng.normal(0, sigma_x_total, n_spots)
    y_obs = Y_true + rng.normal(0, sigma_y_total, n_spots)

    sigma_v_obs = np.full(n_spots, tp["sigma_v"])
    is_hv = (spot_types == "b") | (spot_types == "r")
    sigma_v_intrinsic = np.where(is_hv, tp["sigma_v_hv"], tp["sigma_v_sys"])
    sigma_v_total = np.sqrt(sigma_v_obs**2 + sigma_v_intrinsic**2)
    v_obs = V_true + rng.normal(0, sigma_v_total)

    sigma_a_total = np.sqrt(tp["sigma_a"]**2 + tp["sigma_a_floor"]**2)
    a_obs = A_true + rng.normal(0, sigma_a_total, n_spots)

    is_systemic = spot_types == "s"
    is_highvel = is_hv
    is_blue = spot_types == "b"

    data = {
        "spot_type": spot_types,
        "velocity": v_obs,
        "x": x_obs,
        "sigma_x": sigma_x_obs,
        "y": y_obs,
        "sigma_v": sigma_v_obs,
        "sigma_y": sigma_y_obs,
        "a": a_obs,
        "sigma_a": sigma_a_obs,
        "is_systemic": is_systemic,
        "is_highvel": is_highvel,
        "is_blue": is_blue,
        "n_spots": n_spots,
        "galaxy_name": "MOCK",
        "v_sys_obs": v_sys_cmb,
    }

    true_params_expanded = {
        **tp,
        "D_A": D_A,
        "log_MBH": np.log10(M_BH),
        "v_sys": v_sys_cmb,
        "dv_sys": 0.0,
        "v_pec": v_pec,
        "z_cosmo": z_cosmo,
        "R_phys_true": R_phys_true,
        "r_ang_true": r_ang_true,
        "phi_true": phi_true,
        "X_true": X_true,
        "Y_true": Y_true,
        "V_true": V_true,
        "A_true": A_true,
    }

    if verbose:
        fprint(f"generated {n_spots} maser spots (seed={seed}).")
        fprint(f"  systemic: {is_systemic.sum()}, "
               f"high-vel: {is_highvel.sum()}")
        fprint(f"  v range: [{v_obs.min():.0f}, {v_obs.max():.0f}] km/s")
        fprint(f"  x range: [{x_obs.min():.4f}, {x_obs.max():.4f}] mas")
        fprint(f"  y range: [{y_obs.min():.4f}, {y_obs.max():.4f}] mas")

    return data, true_params_expanded
