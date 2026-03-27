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
"""Mock generator for maser disk (spot-level) forward model."""
from os.path import join, dirname

import numpy as np
from scipy.stats import norm

from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

from ..util import fprint, SPEED_OF_LIGHT

# Disk physics constants (must match candel/model/maser_disk.py)
C_v = 0.9420       # km/s: sqrt(GM_sun / (1 mas * 1 Mpc)) * 1e-3
C_a = 1.872e-4     # km/s/yr: GM_sun * yr / ((1 mas * 1 Mpc)^2 * 1e3)
C_g = 1.974e-11    # dimensionless: 2*GM_sun / (c^2 * 1 mas * 1 Mpc)


DEFAULT_TRUE_PARAMS = {
    "H0": 73.0,
    "sigma_pec": 250.0,
    "D": 87.6,              # Mpc (angular-diameter distance)
    "M_BH": 2.42e7,         # solar masses
    "v_sys": 6908.9,         # km/s (barycentric)
    "x0": 0.0013,           # mas
    "y0": 0.0075,           # mas
    "i0": 90.8,             # degrees
    "Omega0": 99.6,         # degrees
    "dOmega_dr": 4.7,       # degrees/mas
    "di_dr": 0.0,           # degrees/mas (no inclination warp)
    "sigma_x_floor": 0.002,       # mas
    "sigma_y_floor": 0.017,       # mas
    "sigma_v_sys": 4.8,           # km/s
    "sigma_v_hv": 4.3,            # km/s
    "sigma_a_floor": 0.43,        # km/s/yr
    "v_helio_to_cmb": 263.3,      # km/s
    "A_thr": 0.3,                 # km/s/yr (accel detection threshold)
    "sigma_det": 0.2,             # km/s/yr (accel detection width)
}

# VLBI beam parameters for CGCG 074-064
_BEAM_FWHM_X = 0.40   # mas (RA direction)
_BEAM_FWHM_Y = 0.95   # mas (Dec direction)


# -----------------------------------------------------------------------
# Numpy implementations of disk physics (same equations as maser_disk.py)
# -----------------------------------------------------------------------

def _warp_geometry(r, i0_rad, di_dr_rad, Omega0_rad, dOmega_dr_rad):
    i = i0_rad + di_dr_rad * r
    Omega = Omega0_rad + dOmega_dr_rad * r
    return i, Omega


def _predict_position(r, phi, x0, y0, i, Omega):
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_O = np.sin(Omega)
    cos_O = np.cos(Omega)
    cos_i = np.cos(i)
    X = x0 + r * (sin_phi * sin_O - cos_phi * cos_O * cos_i)
    Y = y0 + r * (sin_phi * cos_O + cos_phi * sin_O * cos_i)
    return X, Y


def _predict_velocity_los(r, phi, D, M_BH, v_sys, i):
    v_kep = C_v * np.sqrt(M_BH / (r * D))
    v_z = v_kep * np.sin(phi) * np.sin(i)
    v_total = v_kep
    beta = v_total / SPEED_OF_LIGHT
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    one_plus_z_D = gamma * (1.0 + v_z / SPEED_OF_LIGHT)
    one_plus_z_g = 1.0 / np.sqrt(1.0 - C_g * M_BH / (r * D))
    z_0 = v_sys / SPEED_OF_LIGHT
    V_obs = SPEED_OF_LIGHT * (one_plus_z_D * one_plus_z_g * (1.0 + z_0) - 1.0)
    return V_obs


def _predict_acceleration_los(r, phi, D, M_BH, i):
    a_mag = C_a * M_BH / (r**2 * D**2)
    A_z = a_mag * np.cos(phi) * np.sin(i)
    return A_z


# -----------------------------------------------------------------------
# Spot generation helpers
# -----------------------------------------------------------------------

def _draw_spot_types(n_spots, rng):
    """Assign spot types matching CGCG 074-064 fractions."""
    probs = np.array([0.43, 0.30, 0.27])  # r, b, s
    types = rng.choice(["r", "b", "s"], size=n_spots, p=probs)
    return types


def _draw_radii(n_spots, rng, r_min=0.2, r_max=1.4):
    """Draw orbital radii from uniform distribution in [r_min, r_max] mas."""
    return rng.uniform(r_min, r_max, n_spots)


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


def _draw_measurement_uncertainties(n_spots, rng, snr):
    """Realistic per-spot uncertainties from beam size and S/N."""
    sigma_x = 0.5 * _BEAM_FWHM_X / snr
    sigma_y = 0.5 * _BEAM_FWHM_Y / snr
    return sigma_x, sigma_y


def _draw_snr(n_spots, rng, snr_mu=2.0, snr_sigma=0.8, snr_min=1.0):
    """Draw S/N from a log-normal distribution."""
    ln_snr = rng.normal(snr_mu, snr_sigma, n_spots)
    snr = np.exp(ln_snr)
    return np.clip(snr, snr_min, None)


def _draw_accel_uncertainties(n_spots, rng, sigma_a_min=0.3, sigma_a_max=2.0):
    """Draw per-spot acceleration measurement uncertainties."""
    return rng.uniform(sigma_a_min, sigma_a_max, n_spots)


def _accel_measured_mask_physical(A_true, rng, A_thr, sigma_det):
    """Detection-probability mask based on true acceleration magnitude."""
    p_det = norm.cdf((np.abs(A_true) - A_thr) / sigma_det)
    return rng.random(len(A_true)) < p_det


# -----------------------------------------------------------------------
# Main mock generators
# -----------------------------------------------------------------------

def gen_maser_disk_mock(seed, true_params=None, n_spots=165,
                        D_range=(10.0, 200.0), verbose=True):
    """Generate a mock maser disk dataset at the spot level.

    Samples true latent parameters (distance, disk geometry, spot positions),
    computes observables using the same physics as the forward model, adds
    measurement noise, and applies S/N > 3 selection.

    Parameters
    ----------
    seed
        Random seed.
    true_params
        Override default true parameters (merged with DEFAULT_TRUE_PARAMS).
    n_spots
        Number of spots to generate before selection.
    D_range
        Allowed angular-diameter distance range in Mpc (only used if D is
        drawn from a prior rather than taken from true_params).
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
    D_true = tp["D"]
    M_BH = tp["M_BH"]
    x0 = tp["x0"]
    y0 = tp["y0"]
    i0_deg = tp["i0"]
    Omega0_deg = tp["Omega0"]
    dOmega_dr_deg = tp["dOmega_dr"]
    di_dr_deg = tp["di_dr"]
    v_helio_to_cmb = tp["v_helio_to_cmb"]

    # Convert angles to radians for physics
    i0_rad = np.deg2rad(i0_deg)
    di_dr_rad = np.deg2rad(di_dr_deg)
    Omega0_rad = np.deg2rad(Omega0_deg)
    dOmega_dr_rad = np.deg2rad(dOmega_dr_deg)

    # ---- Cosmological redshift from D_true and H0 (exact) ----
    cosmo = FlatLambdaCDM(H0=H0, Om0=0.315)
    z_cosmo = float(z_at_value(
        cosmo.angular_diameter_distance, D_true * u.Mpc,
        zmin=1e-6, zmax=1.0))
    cz_cosmo = SPEED_OF_LIGHT * z_cosmo

    # ---- Add peculiar velocity (exact relativistic composition) ----
    v_pec = rng.normal(0, sigma_pec)
    z_obs = (1 + z_cosmo) * (1 + v_pec / SPEED_OF_LIGHT) - 1
    v_sys_cmb = SPEED_OF_LIGHT * z_obs
    v_sys_bary = v_sys_cmb - v_helio_to_cmb

    if verbose:
        fprint(f"z_cosmo = {z_cosmo:.6f}, cz_cosmo = {cz_cosmo:.1f} km/s")
        fprint(f"v_pec = {v_pec:.1f} km/s, v_sys_bary = {v_sys_bary:.1f} km/s")

    # ---- Generate maser spots ----
    spot_types = _draw_spot_types(n_spots, rng)
    r_true = _draw_radii(n_spots, rng)
    phi_true = _draw_phi(spot_types, rng)

    # Warped geometry at each spot
    i_k, Omega_k = _warp_geometry(
        r_true, i0_rad, di_dr_rad, Omega0_rad, dOmega_dr_rad)

    # True observables
    X_true, Y_true = _predict_position(r_true, phi_true, x0, y0, i_k, Omega_k)
    V_true = _predict_velocity_los(
        r_true, phi_true, D_true, M_BH, v_sys_bary, i_k)
    A_true = _predict_acceleration_los(r_true, phi_true, D_true, M_BH, i_k)

    # ---- Measurement noise ----
    snr = _draw_snr(n_spots, rng)
    sigma_x_obs, sigma_y_obs = _draw_measurement_uncertainties(n_spots, rng,
                                                               snr)
    sigma_a_obs = _draw_accel_uncertainties(n_spots, rng)

    # Add noise consistent with the forward model's error structure
    sigma_x_total = np.sqrt(sigma_x_obs**2 + tp["sigma_x_floor"]**2)
    sigma_y_total = np.sqrt(sigma_y_obs**2 + tp["sigma_y_floor"]**2)
    x_obs = X_true + rng.normal(0, sigma_x_total)
    y_obs = Y_true + rng.normal(0, sigma_y_total)

    # Velocity: phi-dependent scatter interpolating between systemic and HV
    cos2_phi = np.cos(phi_true)**2
    sigma_v_per_spot = np.sqrt(
        tp["sigma_v_hv"]**2
        + (tp["sigma_v_sys"]**2 - tp["sigma_v_hv"]**2) * cos2_phi)
    v_obs = V_true + rng.normal(0, sigma_v_per_spot)

    # Accelerations: physical detection model + noise from formal + floor
    accel_measured = _accel_measured_mask_physical(
        A_true, rng, tp["A_thr"], tp["sigma_det"])
    sigma_a_total = np.sqrt(sigma_a_obs**2 + tp["sigma_a_floor"]**2)
    a_obs = np.where(accel_measured,
                     A_true + rng.normal(0, sigma_a_total),
                     0.0)

    # ---- S/N > 3 selection ----
    keep = snr >= 3.0
    # Galaxy-level rejection: need at least 10 spots
    if keep.sum() < 10:
        if verbose:
            fprint(f"WARNING: only {keep.sum()} spots survive S/N cut, "
                   f"galaxy rejected.")
        return None, tp

    # Apply selection
    spot_types = spot_types[keep]
    r_true = r_true[keep]
    phi_true = phi_true[keep]
    x_obs = x_obs[keep]
    y_obs = y_obs[keep]
    sigma_x_obs = sigma_x_obs[keep]
    sigma_y_obs = sigma_y_obs[keep]
    v_obs = v_obs[keep]
    a_obs = a_obs[keep]
    sigma_a_obs = sigma_a_obs[keep]
    accel_measured = accel_measured[keep]
    snr = snr[keep]
    X_true = X_true[keep]
    Y_true = Y_true[keep]
    V_true = V_true[keep]
    A_true = A_true[keep]

    n_kept = len(spot_types)
    is_systemic = spot_types == "s"
    is_highvel = (spot_types == "b") | (spot_types == "r")

    data = {
        "spot_type": spot_types,
        "velocity": v_obs,
        "x": x_obs,
        "sigma_x": sigma_x_obs,
        "y": y_obs,
        "sigma_y": sigma_y_obs,
        "a": a_obs,
        "sigma_a": sigma_a_obs,
        "accel_measured": accel_measured,
        "is_systemic": is_systemic,
        "is_highvel": is_highvel,
        "n_spots": n_kept,
        "galaxy_name": "MOCK",
    }

    true_params_expanded = {
        **tp,
        "v_sys": v_sys_bary,
        "v_pec": v_pec,
        "z_cosmo": z_cosmo,
        "r_true": r_true,
        "phi_true": phi_true,
        "X_true": X_true,
        "Y_true": Y_true,
        "V_true": V_true,
        "A_true": A_true,
    }

    if verbose:
        fprint(f"generated {n_kept}/{n_spots} maser spots after S/N >= 3 cut "
               f"(seed={seed}).")
        fprint(f"  systemic: {is_systemic.sum()}, "
               f"high-vel: {is_highvel.sum()}")
        fprint(f"  accel measured: {accel_measured.sum()}/{n_kept}")
        fprint(f"  v range: [{v_obs.min():.0f}, {v_obs.max():.0f}] km/s")
        fprint(f"  x range: [{x_obs.min():.4f}, {x_obs.max():.4f}] mas")
        fprint(f"  y range: [{y_obs.min():.4f}, {y_obs.max():.4f}] mas")

    return data, true_params_expanded


def gen_maser_mock_like_cgcg074(seed, true_params=None, verbose=True):
    """Generate a mock using real CGCG 074-064 spot uncertainties.

    Loads the actual MRT data to get per-spot measurement uncertainties
    (sigma_x, sigma_y, sigma_a) and spot types, then generates mock
    observables at similar orbital radii with those same noise levels.

    Parameters
    ----------
    seed
        Random seed.
    true_params
        Override default true parameters.
    verbose
        Print summary statistics.

    Returns
    -------
    data : dict
        Data dict matching what MaserDiskModel expects.
    true_params_expanded : dict
        All true values including per-spot r, phi.
    """
    from ..pvdata.megamaser_data import load_megamaser_spots

    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
    rng = np.random.default_rng(seed)

    # Load real data for uncertainties and spot structure
    data_root = join(dirname(dirname(dirname(__file__))), "data", "Megamaser")
    real = load_megamaser_spots(data_root, galaxy="CGCG074-064",
                               v_cmb_obs=7144.0)

    n_spots = real["n_spots"]
    spot_types = real["spot_type"].copy()
    sigma_x_real = real["sigma_x"].copy()
    sigma_y_real = real["sigma_y"].copy()
    sigma_a_real = real["sigma_a"].copy()
    accel_measured = real["accel_measured"].copy()

    H0 = tp["H0"]
    sigma_pec = tp["sigma_pec"]
    D_true = tp["D"]
    M_BH = tp["M_BH"]
    x0 = tp["x0"]
    y0 = tp["y0"]
    i0_deg = tp["i0"]
    Omega0_deg = tp["Omega0"]
    dOmega_dr_deg = tp["dOmega_dr"]
    di_dr_deg = tp["di_dr"]
    v_helio_to_cmb = tp["v_helio_to_cmb"]

    i0_rad = np.deg2rad(i0_deg)
    di_dr_rad = np.deg2rad(di_dr_deg)
    Omega0_rad = np.deg2rad(Omega0_deg)
    dOmega_dr_rad = np.deg2rad(dOmega_dr_deg)

    # Exact cosmological redshift
    cosmo = FlatLambdaCDM(H0=H0, Om0=0.315)
    z_cosmo = float(z_at_value(
        cosmo.angular_diameter_distance, D_true * u.Mpc,
        zmin=1e-6, zmax=1.0))
    cz_cosmo = SPEED_OF_LIGHT * z_cosmo

    # Peculiar velocity (exact relativistic composition)
    v_pec = rng.normal(0, sigma_pec)
    z_obs = (1 + z_cosmo) * (1 + v_pec / SPEED_OF_LIGHT) - 1
    v_sys_cmb = SPEED_OF_LIGHT * z_obs
    v_sys_bary = v_sys_cmb - v_helio_to_cmb

    if verbose:
        fprint(f"z_cosmo = {z_cosmo:.6f}, cz_cosmo = {cz_cosmo:.1f} km/s")
        fprint(f"v_pec = {v_pec:.1f} km/s, v_sys_bary = {v_sys_bary:.1f} km/s")

    # Draw per-spot latent variables (r, phi) with type-dependent priors
    r_true = _draw_radii(n_spots, rng)
    phi_true = _draw_phi(spot_types, rng)

    # Warped geometry
    i_k, Omega_k = _warp_geometry(
        r_true, i0_rad, di_dr_rad, Omega0_rad, dOmega_dr_rad)

    # True observables
    X_true, Y_true = _predict_position(r_true, phi_true, x0, y0, i_k, Omega_k)
    V_true = _predict_velocity_los(
        r_true, phi_true, D_true, M_BH, v_sys_bary, i_k)
    A_true = _predict_acceleration_los(r_true, phi_true, D_true, M_BH, i_k)

    # Add noise consistent with the forward model's error structure
    sigma_x_total = np.sqrt(sigma_x_real**2 + tp["sigma_x_floor"]**2)
    sigma_y_total = np.sqrt(sigma_y_real**2 + tp["sigma_y_floor"]**2)
    x_obs = X_true + rng.normal(0, sigma_x_total)
    y_obs = Y_true + rng.normal(0, sigma_y_total)

    # Velocity: phi-dependent scatter
    cos2_phi = np.cos(phi_true)**2
    sigma_v_per_spot = np.sqrt(
        tp["sigma_v_hv"]**2
        + (tp["sigma_v_sys"]**2 - tp["sigma_v_hv"]**2) * cos2_phi)
    v_obs = V_true + rng.normal(0, sigma_v_per_spot)

    # Accelerations: physical detection model
    accel_measured = _accel_measured_mask_physical(
        A_true, rng, tp["A_thr"], tp["sigma_det"])
    sigma_a_total = np.sqrt(sigma_a_real**2 + tp["sigma_a_floor"]**2)
    a_obs = np.where(accel_measured,
                     A_true + rng.normal(0, sigma_a_total),
                     0.0)

    is_systemic = spot_types == "s"
    is_highvel = (spot_types == "b") | (spot_types == "r")

    data = {
        "spot_type": spot_types,
        "velocity": v_obs,
        "x": x_obs,
        "sigma_x": sigma_x_real,
        "y": y_obs,
        "sigma_y": sigma_y_real,
        "a": a_obs,
        "sigma_a": sigma_a_real,
        "accel_measured": accel_measured,
        "is_systemic": is_systemic,
        "is_highvel": is_highvel,
        "n_spots": n_spots,
        "galaxy_name": "MOCK_CGCG074",
    }

    true_params_expanded = {
        **tp,
        "v_sys": v_sys_bary,
        "v_pec": v_pec,
        "z_cosmo": z_cosmo,
        "r_true": r_true,
        "phi_true": phi_true,
        "X_true": X_true,
        "Y_true": Y_true,
        "V_true": V_true,
        "A_true": A_true,
    }

    if verbose:
        fprint(f"generated {n_spots} mock spots matching CGCG 074-064 "
               f"structure (seed={seed}).")
        fprint(f"  systemic: {is_systemic.sum()}, "
               f"high-vel: {is_highvel.sum()}")
        fprint(f"  accel measured: {accel_measured.sum()}/{n_spots}")
        fprint(f"  v range: [{v_obs.min():.0f}, {v_obs.max():.0f}] km/s")

    return data, true_params_expanded


# -----------------------------------------------------------------------
# Default true parameters for each galaxy (multi-galaxy mock)
# -----------------------------------------------------------------------

# NGC 5765b (Gao+2016 / Pesce+2020)
_NGC5765B_TRUE_PARAMS = {
    "D": 112.2,
    "M_BH": 4.15e7,
    "v_sys": 8315.6,
    "x0": 0.0,
    "y0": 0.0,
    "i0": 72.4,
    "Omega0": 149.7,
    "dOmega_dr": -3.2,
    "di_dr": 0.0,
    "sigma_x_floor": 0.002,
    "sigma_y_floor": 0.017,
    "sigma_v_sys": 5.0,
    "sigma_v_hv": 5.0,
    "sigma_a_floor": 0.5,
    "v_helio_to_cmb": 210.1,
    "A_thr": 0.3,
    "sigma_det": 0.2,
}

# NGC 6264 (Kuo+2013 / Pesce+2020)
_NGC6264_TRUE_PARAMS = {
    "D": 149.0,
    "M_BH": 2.91e7,
    "v_sys": 10219.0,
    "x0": 0.0,
    "y0": 0.0,
    "i0": 91.3,
    "Omega0": 84.7,
    "dOmega_dr": 0.0,
    "di_dr": 0.0,
    "sigma_x_floor": 0.002,
    "sigma_y_floor": 0.017,
    "sigma_v_sys": 5.0,
    "sigma_v_hv": 5.0,
    "sigma_a_floor": 0.5,
    "v_helio_to_cmb": -3.3,
    "A_thr": 0.3,
    "sigma_det": 0.2,
}

# NGC 6323 (Kuo+2015 / Pesce+2020)
_NGC6323_TRUE_PARAMS = {
    "D": 106.0,
    "M_BH": 0.94e7,
    "v_sys": 7842.0,
    "x0": 0.0,
    "y0": 0.0,
    "i0": 91.5,
    "Omega0": 184.4,
    "dOmega_dr": 0.0,
    "di_dr": 0.0,
    "sigma_x_floor": 0.002,
    "sigma_y_floor": 0.017,
    "sigma_v_sys": 5.0,
    "sigma_v_hv": 5.0,
    "sigma_a_floor": 0.5,
    "v_helio_to_cmb": -33.5,
    "A_thr": 0.3,
    "sigma_det": 0.2,
}

# UGC 3789 (Reid+2013 / Pesce+2020)
_UGC3789_TRUE_PARAMS = {
    "D": 51.5,
    "M_BH": 1.09e7,
    "v_sys": 2900.0,
    "x0": 0.0,
    "y0": 0.0,
    "i0": 90.0,
    "Omega0": 140.0,
    "dOmega_dr": 0.0,
    "di_dr": 0.0,
    "sigma_x_floor": 0.002,
    "sigma_y_floor": 0.017,
    "sigma_v_sys": 5.0,
    "sigma_v_hv": 5.0,
    "sigma_a_floor": 0.2,
    "v_helio_to_cmb": 419.9,
    "A_thr": 0.3,
    "sigma_det": 0.2,
}

MULTI_GALAXY_DEFAULTS = {
    "CGCG074": DEFAULT_TRUE_PARAMS,
    "NGC5765b": _NGC5765B_TRUE_PARAMS,
    "NGC6264": _NGC6264_TRUE_PARAMS,
    "NGC6323": _NGC6323_TRUE_PARAMS,
    "UGC3789": _UGC3789_TRUE_PARAMS,
}


def gen_multi_galaxy_mock(seed, galaxy_names=None, H0=73.0, sigma_pec=250.0,
                          per_galaxy_overrides=None, n_spots=100,
                          verbose=True):
    """Generate mock maser data for multiple galaxies with shared H0.

    Parameters
    ----------
    seed
        Master random seed.
    galaxy_names
        List of galaxy keys (must be in MULTI_GALAXY_DEFAULTS).
        Default: all five galaxies.
    H0
        Shared true Hubble constant.
    sigma_pec
        Shared true peculiar velocity dispersion.
    per_galaxy_overrides
        Dict of {galaxy_name: {param: value}} to override defaults.
    n_spots
        Number of spots per galaxy (before S/N cut).
    verbose
        Print summary.

    Returns
    -------
    data_list : list of dict
        Per-galaxy data dicts for JointMaserModel.
    true_params : dict
        Shared + per-galaxy true parameters.
    """
    if galaxy_names is None:
        galaxy_names = list(MULTI_GALAXY_DEFAULTS.keys())

    overrides = per_galaxy_overrides or {}
    rng = np.random.default_rng(seed)
    cosmo = FlatLambdaCDM(H0=H0, Om0=0.315)

    data_list = []
    per_galaxy_truth = {}

    for gname in galaxy_names:
        if gname not in MULTI_GALAXY_DEFAULTS:
            raise ValueError(
                f"Unknown galaxy '{gname}'. "
                f"Available: {list(MULTI_GALAXY_DEFAULTS.keys())}")

        tp = {**MULTI_GALAXY_DEFAULTS[gname], **(overrides.get(gname, {}))}
        tp["H0"] = H0
        tp["sigma_pec"] = sigma_pec

        D_true = tp["D"]
        M_BH = tp["M_BH"]
        x0, y0 = tp["x0"], tp["y0"]
        i0_rad = np.deg2rad(tp["i0"])
        di_dr_rad = np.deg2rad(tp["di_dr"])
        Omega0_rad = np.deg2rad(tp["Omega0"])
        dOmega_dr_rad = np.deg2rad(tp["dOmega_dr"])
        v_helio_to_cmb = tp["v_helio_to_cmb"]

        # Exact cosmological redshift
        z_cosmo = float(z_at_value(
            cosmo.angular_diameter_distance, D_true * u.Mpc,
            zmin=1e-6, zmax=1.0))

        # Peculiar velocity
        v_pec = rng.normal(0, sigma_pec)
        z_obs = (1 + z_cosmo) * (1 + v_pec / SPEED_OF_LIGHT) - 1
        v_sys_cmb = SPEED_OF_LIGHT * z_obs
        v_sys_bary = v_sys_cmb - v_helio_to_cmb

        # Generate spots
        spot_types = _draw_spot_types(n_spots, rng)
        r_true = _draw_radii(n_spots, rng)
        phi_true = _draw_phi(spot_types, rng)

        i_k, Omega_k = _warp_geometry(
            r_true, i0_rad, di_dr_rad, Omega0_rad, dOmega_dr_rad)

        X_true, Y_true = _predict_position(
            r_true, phi_true, x0, y0, i_k, Omega_k)
        V_true = _predict_velocity_los(
            r_true, phi_true, D_true, M_BH, v_sys_bary, i_k)
        A_true = _predict_acceleration_los(
            r_true, phi_true, D_true, M_BH, i_k)

        # Measurement noise
        snr = _draw_snr(n_spots, rng)
        sigma_x_obs, sigma_y_obs = _draw_measurement_uncertainties(
            n_spots, rng, snr)
        sigma_a_obs = _draw_accel_uncertainties(n_spots, rng)

        sigma_x_total = np.sqrt(sigma_x_obs**2 + tp["sigma_x_floor"]**2)
        sigma_y_total = np.sqrt(sigma_y_obs**2 + tp["sigma_y_floor"]**2)
        x_obs = X_true + rng.normal(0, sigma_x_total)
        y_obs = Y_true + rng.normal(0, sigma_y_total)

        # Velocity: phi-dependent scatter
        cos2_phi = np.cos(phi_true)**2
        sigma_v_per_spot = np.sqrt(
            tp["sigma_v_hv"]**2
            + (tp["sigma_v_sys"]**2 - tp["sigma_v_hv"]**2) * cos2_phi)
        v_obs = V_true + rng.normal(0, sigma_v_per_spot)

        # Accelerations: physical detection model
        accel_measured = _accel_measured_mask_physical(
            A_true, rng, tp["A_thr"], tp["sigma_det"])
        sigma_a_total = np.sqrt(sigma_a_obs**2 + tp["sigma_a_floor"]**2)
        a_obs = np.where(accel_measured,
                         A_true + rng.normal(0, sigma_a_total), 0.0)

        # S/N cut
        keep = snr >= 3.0
        if keep.sum() < 10:
            if verbose:
                fprint(f"WARNING: {gname} only {keep.sum()} spots survive "
                       "S/N cut, skipping.")
            continue

        spot_types = spot_types[keep]
        is_systemic = spot_types == "s"
        is_highvel = (spot_types == "b") | (spot_types == "r")
        n_kept = int(keep.sum())

        data = {
            "spot_type": spot_types,
            "velocity": v_obs[keep],
            "x": x_obs[keep],
            "sigma_x": sigma_x_obs[keep],
            "y": y_obs[keep],
            "sigma_y": sigma_y_obs[keep],
            "a": a_obs[keep],
            "sigma_a": sigma_a_obs[keep],
            "accel_measured": accel_measured[keep],
            "is_systemic": is_systemic,
            "is_highvel": is_highvel,
            "n_spots": n_kept,
            "galaxy_name": f"MOCK_{gname}",
        }
        data_list.append(data)

        per_galaxy_truth[gname] = {
            **tp,
            "v_sys": v_sys_bary,
            "v_pec": v_pec,
            "z_cosmo": z_cosmo,
            "n_spots_kept": n_kept,
        }

        if verbose:
            fprint(f"{gname}: {n_kept}/{n_spots} spots, "
                   f"v_pec={v_pec:.1f} km/s, D={D_true:.1f} Mpc")

    true_params = {
        "H0": H0,
        "sigma_pec": sigma_pec,
        "per_galaxy": per_galaxy_truth,
    }

    if verbose:
        fprint(f"generated multi-galaxy mock with {len(data_list)} galaxies "
               f"(seed={seed}).")

    return data_list, true_params
