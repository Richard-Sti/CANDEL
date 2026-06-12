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
"""Mock generator for TRGB surveys."""
import numpy as np
from scipy.stats import norm

from ..cosmo.cosmography import Distance2Distmod, Distance2Redshift
from ..field import interpolate_los_density_velocity
from ..field.field_interp import build_regular_interpolator
from ..util import (SPEED_OF_LIGHT, galactic_to_radec_cartesian,
                    radec_to_cartesian)
from ._field_utils import (field_xyz_to_radec, galaxy_bias_log_weight,
                           galaxy_bias_params_from_values)

DEFAULT_TRUE_PARAMS = {
    "H0": 73.0,
    "M_TRGB": -4.05,
    "alpha_c": 0.2,
    "c_star": 1.23,
    "c_bar": 1.23,
    "w_c": 0.2,
    "sigma_int": 0.1,
    "sigma_v": 300.0,
    "Vext_mag": 150.0,
    "Vext_ell": 270.0,
    "Vext_b": 30.0,
    "beta": 0.43,
    "b1": 1.2,
    "b2": 0.0,
    "b3": 0.0,
    "alpha": 1.0,
    "delta_b1": 0.0,
    "alpha_low": 1.5,
    "alpha_high_frac": 0.5,
    "log_rho_t": 0.5,
    "log_rho_width": 0.5,
}

DEFAULT_COLOUR_STD = 0.2  # spread of dereddened F606W-F814W in mock
DEFAULT_COLOUR_ERR = 0.03

DEFAULT_ANCHORS = {
    "mu_LMC": 18.477,
    "e_mu_LMC": 0.026,
    "e_mag_LMC_TRGB": 0.018,
    "mu_N4258": 29.398,
    "e_mu_N4258": 0.032,
    "e_mag_N4258_TRGB": 0.0443,
}

SELECTION_TAIL_SIGMA = 5.0


def _apply_selection(mag_obs, mag_min, mag_lim, mag_lim_width, gen):
    """Return boolean selection mask."""
    n = len(mag_obs)
    if mag_lim is not None:
        p_sel = norm.cdf((mag_lim - mag_obs) / mag_lim_width)
        if mag_min is not None:
            p_sel -= norm.cdf((mag_min - mag_obs) / mag_lim_width)
        p_sel = np.clip(p_sel, 0.0, 1.0)
        return gen.random(n) < p_sel
    return np.ones(n, dtype=bool)


def _gen_field_path(nsamples, h, beta, rmin, rmax, e_mag, e_czcmb,
                    M_TRGB, alpha_c, c_star, colour_mean, colour_std,
                    e_colour_dered,
                    sigma_int, sigma_v, Vext,
                    mag_min, mag_lim, mag_lim_width,
                    which_bias, bias_params,
                    field_loader, r2mu, r2z, gen, verbose):
    """Sample TRGB hosts from a field, or from unit density if absent.

    With a 3D density field, galaxies are sampled using accept/reject with the
    configured model galaxy-bias law so that p(r, Ω) ∝ b[ρ(r,Ω)] * r².
    Without a field, the density is unity everywhere and velocities are zero.
    """
    has_field = field_loader is not None
    if has_field:
        # LOS grid for the model covers the full rmax range (in Mpc/h)
        r_grid = np.linspace(0.1, rmax * h, 301)

    # Sampling sphere: set from selection threshold, not the full rmax
    r_sample_Mpc = rmax
    if mag_lim is not None:
        M_sel = M_TRGB + alpha_c * (colour_mean - c_star)
        sigma_colour = abs(alpha_c) * colour_std
        mu_max = mag_lim - M_sel
        sigma_tot = np.sqrt(sigma_int**2 + e_mag**2
                            + mag_lim_width**2 + sigma_colour**2)
        mu_cutoff = mu_max + SELECTION_TAIL_SIGMA * sigma_tot
        r_sample_Mpc = min(10**((mu_cutoff - 25) / 5), rmax)
    r_sphere = r_sample_Mpc * h  # Mpc/h

    if verbose:
        label = "Field mock" if has_field else "Unit-density mock"
        print(f"{label}: 3D sampling "
              f"(r_sphere: {r_sphere:.1f} Mpc/h = "
              f"{r_sample_Mpc:.1f} Mpc)...")

    # --- Load fields and build 3D interpolators ---
    if has_field:
        eps = 1e-4
        density_raw = field_loader.load_density()
        density_log = np.log(density_raw + eps).astype(np.float32)
        f_density_3d = build_regular_interpolator(
            density_log, field_loader.boxsize,
            fill_value=np.float32(np.log(1 + eps)))

        log_weight_max = float(np.max(galaxy_bias_log_weight(
            density_raw, bias_params, which_bias)))
        del density_raw, density_log

        velocity_3d = field_loader.load_velocity()
        f_vel_3d = []
        for i in range(3):
            f_vel_3d.append(build_regular_interpolator(
                velocity_3d[i], field_loader.boxsize,
                fill_value=np.float32(0)))
        del velocity_3d

        if verbose:
            print(f"  galaxy bias = {which_bias}, "
                  f"max log weight = {log_weight_max:.3f}")

        obs = field_loader.observer_pos
        coord_frame = field_loader.coordinate_frame
    else:
        obs = None
        coord_frame = "icrs"
    rmin_h = rmin * h
    sigma_mag_tot = np.sqrt(sigma_int**2 + e_mag**2)
    sigma_cz_tot = np.sqrt(e_czcmb**2 + sigma_v**2)

    # --- Loop: sample, compute observables, select ---
    collected = {k: [] for k in [
        "RA", "dec", "r_h", "mag_obs", "cz_obs", "colour_dered"]}
    n_total_proposed = 0
    n_total_density_accepted = 0
    batch_size = 200000 if has_field else max(int(1.5 * nsamples), 100)

    while sum(len(v) for v in collected["RA"]) < nsamples:
        n_total_proposed += batch_size

        # Uniform in cube, cut to sphere
        xyz = gen.uniform(-r_sphere, r_sphere,
                          (batch_size, 3)).astype(np.float32)
        r_sq = np.sum(xyz**2, axis=1)
        in_shell = (r_sq < r_sphere**2) & (r_sq > rmin_h**2)
        xyz = xyz[in_shell]

        # Density accept/reject
        if has_field:
            rho_log = f_density_3d(xyz + obs[None, :])
            rho = np.exp(rho_log) - eps
            np.clip(rho, eps, None, out=rho)
            log_weight = galaxy_bias_log_weight(
                rho, bias_params, which_bias)
            p_accept = np.exp(np.minimum(log_weight - log_weight_max, 0.0))
            accept = gen.random(len(p_accept)) < p_accept
            xyz = xyz[accept]
        n_total_density_accepted += len(xyz)

        if len(xyz) == 0:
            continue

        r_h = np.linalg.norm(xyz, axis=1)

        # Convert to RA/dec
        RA, dec = field_xyz_to_radec(xyz, r_h, coord_frame)

        # Radial velocity at 3D positions
        rhat = xyz / r_h[:, None]
        Vpec_field = np.zeros(len(xyz), dtype=np.float32)
        if has_field:
            pos_box = (xyz + obs[None, :]).astype(np.float32)
            for i in range(3):
                Vpec_field += f_vel_3d[i](pos_box) * rhat[:, i]

        # Compute observables
        rhat_icrs = radec_to_cartesian(RA, dec)
        Vext_rad = rhat_icrs @ Vext
        Vpec = Vext_rad + beta * Vpec_field

        r_Mpc = r_h / h
        mu = np.asarray(r2mu(r_Mpc, h=h))
        z_cosmo = np.asarray(r2z(r_Mpc, h=h))

        colour_true = gen.normal(colour_mean, colour_std, len(r_Mpc))
        if e_colour_dered is None:
            colour_obs = colour_true
        else:
            colour_obs = gen.normal(colour_true, e_colour_dered)
        mag_obs = gen.normal(
            M_TRGB + alpha_c * (colour_true - c_star) + mu,
            sigma_mag_tot)
        cz_true = SPEED_OF_LIGHT * (
            (1 + z_cosmo) * (1 + Vpec / SPEED_OF_LIGHT) - 1)
        cz_obs = gen.normal(cz_true, sigma_cz_tot)

        # Apply selection
        sel = _apply_selection(mag_obs, mag_min, mag_lim, mag_lim_width, gen)

        collected["RA"].append(RA[sel])
        collected["dec"].append(dec[sel])
        collected["r_h"].append(r_h[sel])
        collected["mag_obs"].append(mag_obs[sel])
        collected["cz_obs"].append(cz_obs[sel])
        collected["colour_dered"].append(colour_obs[sel])

    if has_field:
        del f_density_3d, f_vel_3d

    # Trim to nsamples
    for k in collected:
        collected[k] = np.concatenate(collected[k])[:nsamples]

    n_selected = nsamples
    if verbose:
        print(f"  {n_total_proposed} proposed, "
              f"{n_total_density_accepted} density-accepted, "
              f"{n_selected} after selection")
        print(f"  max true distance retained: "
              f"{collected['r_h'].max() / h:.2f} Mpc "
              f"(allowed: {rmax:.2f} Mpc)")

    result = {
        "RA": collected["RA"],
        "dec": collected["dec"],
        "r_h": collected["r_h"],
        "mag_obs": collected["mag_obs"],
        "cz_obs": collected["cz_obs"],
        "colour_dered": collected["colour_dered"],
        "n_parent": n_total_density_accepted,
    }
    if has_field:
        # --- Interpolate full LOS for selected hosts ---
        if verbose:
            print(f"  interpolating LOS for {nsamples} hosts...")
        los_density, los_velocity = interpolate_los_density_velocity(
            field_loader, r_grid, collected["RA"], collected["dec"],
            verbose=verbose)
        # Host LOS data: (1, nsamples, n_r)
        result["host_los_density"] = los_density[None, ...]
        result["host_los_velocity"] = los_velocity[None, ...]
        result["host_los_r"] = r_grid
    return result


def gen_TRGB_mock(nsamples=480, Om=0.3, e_mag=0.05, e_czcmb=10.0,
                  rmin=0.5, rmax=40.0,
                  mag_min=22.1,
                  mag_lim=25.0, mag_lim_width=0.75,
                  which_bias="linear",
                  true_params=None, anchors=None,
                  colour_mean=None, colour_std=None,
                  e_colour_dered=DEFAULT_COLOUR_ERR,
                  noisy_anchors=True, field_loader=None,
                  density_3d_data=None, seed=42, verbose=True):
    """Generate a mock TRGB survey compatible with TRGBModel.

    When ``field_loader`` is None (default), distances are drawn from unit
    density on [rmin, rmax].  When a field loader is provided, distances are
    drawn from p(r) ~ b[rho(r)] * r^2 using the density field, and the field's
    radial peculiar velocity is included in the observed cz.

    Selection (optional):
      - mag_min, mag_lim: finite sigmoid window in observed TRGB magnitude

    Returns
    -------
    data : dict
        Data dict matching TRGBModel expectations.
    true_params : dict
        True parameter values used.
    n_parent : int
        Number accepted by the density sampler before observable selection.
    """
    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
    anch = {**DEFAULT_ANCHORS, **(anchors or {})}
    gen = np.random.default_rng(seed)
    bias_params = galaxy_bias_params_from_values(tp, which_bias, Om=Om)

    H0 = tp["H0"]
    M_TRGB = tp["M_TRGB"]
    alpha_c = tp["alpha_c"]
    c_star = tp["c_star"]
    sigma_int = tp["sigma_int"]
    sigma_v = tp["sigma_v"]

    h = H0 / 100
    beta = tp["beta"]
    r2mu = Distance2Distmod(Om0=Om)
    r2z = Distance2Redshift(Om0=Om)
    Vext = tp["Vext_mag"] * galactic_to_radec_cartesian(
        tp["Vext_ell"], tp["Vext_b"])

    cmean = tp.get("c_bar", c_star) if colour_mean is None else colour_mean
    cstd = tp.get("w_c", DEFAULT_COLOUR_STD) if colour_std is None \
        else colour_std

    collected = _gen_field_path(
        nsamples, h, beta, rmin, rmax, e_mag, e_czcmb,
        M_TRGB, alpha_c, c_star, cmean, cstd, e_colour_dered,
        sigma_int, sigma_v, Vext,
        mag_min, mag_lim, mag_lim_width,
        which_bias, bias_params,
        field_loader, r2mu, r2z, gen, verbose)
    n_parent = collected.pop("n_parent")

    # --- Anchor observations ---
    mu_LMC_true = anch["mu_LMC"]
    mu_N4258_true = anch["mu_N4258"]
    mag_LMC_true = M_TRGB + mu_LMC_true
    mag_N4258_true = M_TRGB + mu_N4258_true

    if noisy_anchors:
        mu_LMC_obs = float(gen.normal(mu_LMC_true, anch["e_mu_LMC"]))
        mu_N4258_obs = float(gen.normal(mu_N4258_true, anch["e_mu_N4258"]))
        e_mag_LMC = np.sqrt(anch["e_mag_LMC_TRGB"]**2 + sigma_int**2)
        e_mag_N4258 = np.sqrt(anch["e_mag_N4258_TRGB"]**2 + sigma_int**2)
        mag_LMC_obs = float(gen.normal(mag_LMC_true, e_mag_LMC))
        mag_N4258_obs = float(gen.normal(
            mag_N4258_true, e_mag_N4258))
    else:
        mu_LMC_obs = mu_LMC_true
        mu_N4258_obs = mu_N4258_true
        mag_LMC_obs = mag_LMC_true
        mag_N4258_obs = mag_N4258_true

    # --- Build data dict ---
    n_kept = len(collected["RA"])
    r_true = collected["r_h"] / h
    data = {
        "RA_host": collected["RA"],
        "dec_host": collected["dec"],
        "r_true": r_true,
        "mag_obs": collected["mag_obs"],
        "e_mag_obs": np.full(n_kept, e_mag),
        "czcmb": collected["cz_obs"],
        "e_czcmb": np.full(n_kept, e_czcmb),
        "e_mag_median": float(e_mag),
        "colour_dered": collected["colour_dered"],
        # Anchors
        "mu_LMC_anchor": mu_LMC_obs,
        "e_mu_LMC_anchor": anch["e_mu_LMC"],
        "mag_LMC_TRGB": mag_LMC_obs,
        "e_mag_LMC_TRGB": anch["e_mag_LMC_TRGB"],
        "mu_N4258_anchor": mu_N4258_obs,
        "e_mu_N4258_anchor": anch["e_mu_N4258"],
        "mag_N4258_TRGB": mag_N4258_obs,
        "e_mag_N4258_TRGB": anch["e_mag_N4258_TRGB"],
        "has_rand_los": False,
    }
    if e_colour_dered is not None:
        data["e_colour_dered"] = np.full(n_kept, e_colour_dered)

    # Add host LOS data for field-based mocks. Reconstruction integrals use
    # 3D density data, not random LOS.
    for k in ["host_los_density", "host_los_velocity", "host_los_r"]:
        if k in collected:
            data[k] = collected[k]
    if "host_los_r" in collected:
        data["has_rand_los"] = False
    if density_3d_data is not None:
        data.update(density_3d_data)
        data["has_volume_density_3d"] = True
    else:
        data["has_volume_density_3d"] = False

    # Store true Cartesian Vext for reference
    tp["Vext_x"], tp["Vext_y"], tp["Vext_z"] = Vext

    return data, tp, n_parent
