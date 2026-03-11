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
from ._field_utils import field_xyz_to_radec, smoothclip

DEFAULT_TRUE_PARAMS = {
    "H0": 73.0,
    "M_TRGB": -4.05,
    "sigma_int": 0.1,
    "sigma_v": 300.0,
    "Vext_mag": 150.0,
    "Vext_ell": 270.0,
    "Vext_b": 30.0,
    "beta": 0.43,
    "b1": 1.2,
}

DEFAULT_ANCHORS = {
    "mu_LMC": 18.477,
    "e_mu_LMC": 0.026,
    "e_mag_LMC_TRGB": 0.018,
    "mu_N4258": 29.398,
    "e_mu_N4258": 0.032,
    "e_mag_N4258_TRGB": 0.0443,
}


def _apply_selection(mag_obs, cz_obs, mag_lim, mag_lim_width,
                     cz_lim, cz_lim_width, gen):
    """Return boolean selection mask."""
    n = len(mag_obs)
    if mag_lim is not None:
        p_sel = norm.cdf((mag_lim - mag_obs) / mag_lim_width)
        return gen.random(n) < p_sel
    elif cz_lim is not None:
        if cz_lim_width:
            p_sel = norm.cdf((cz_lim - cz_obs) / cz_lim_width)
            return gen.random(n) < p_sel
        else:
            return cz_obs < cz_lim
    return np.ones(n, dtype=bool)


def _gen_homogeneous_path(nsamples, h, rmin, rmax, e_mag, e_czcmb,
                          M_TRGB, sigma_int, sigma_v, Vext,
                          mag_lim, mag_lim_width, cz_lim, cz_lim_width,
                          r2mu, r2z, gen, verbose):
    """Homogeneous (no field) distance sampling path."""
    # Tighten sampling sphere based on selection
    r_sample = rmax
    if mag_lim is not None:
        mu_max = mag_lim - M_TRGB
        sigma_tot = np.sqrt(sigma_int**2 + e_mag**2 + mag_lim_width**2)
        mu_cutoff = mu_max + 5 * sigma_tot
        r_sample = min(10**((mu_cutoff - 25) / 5), rmax)
    elif cz_lim is not None:
        r_sample = min(cz_lim / (h * 100) * 1.5, rmax)

    if verbose and r_sample < rmax:
        print(f"Homogeneous mock: tightened r_max from {rmax:.1f} to "
              f"{r_sample:.1f} Mpc based on selection")

    collected = {k: [] for k in ["RA", "dec", "r", "mag_obs", "cz_obs"]}
    n_accepted = 0
    n_parent = 0
    batch = max(int(1.5 * nsamples), 100)

    while n_accepted < nsamples:
        RA = gen.uniform(0, 360, batch)
        dec = np.rad2deg(np.arcsin(gen.uniform(-1, 1, batch)))
        rhat = radec_to_cartesian(RA, dec)

        u = gen.random(batch)
        r = (rmin**3 + u * (r_sample**3 - rmin**3))**(1 / 3)

        mu = np.asarray(r2mu(r, h=h))
        z_cosmo = np.asarray(r2z(r, h=h))
        Vext_rad = rhat @ Vext

        sigma_mag_tot = np.sqrt(sigma_int**2 + e_mag**2)
        mag_obs = gen.normal(M_TRGB + mu, sigma_mag_tot)

        cz_true = SPEED_OF_LIGHT * (
            (1 + z_cosmo) * (1 + Vext_rad / SPEED_OF_LIGHT) - 1)
        cz_obs = gen.normal(cz_true, np.sqrt(e_czcmb**2 + sigma_v**2))

        mask = _apply_selection(mag_obs, cz_obs, mag_lim, mag_lim_width,
                                cz_lim, cz_lim_width, gen)

        n_parent += batch
        n_accepted += mask.sum()

        for k, v in zip(
                ["RA", "dec", "r", "mag_obs", "cz_obs"],
                [RA, dec, r, mag_obs, cz_obs]):
            collected[k].append(v[mask])

    for k in collected:
        collected[k] = np.concatenate(collected[k])[:nsamples]

    if verbose:
        sel_frac = n_accepted / n_parent
        print(f"Generated {nsamples} TRGB hosts "
              f"(acceptance {sel_frac:.2f}, {n_parent} drawn).")
        print(f"  max true distance retained: "
              f"{collected['r'].max():.2f} Mpc "
              f"(allowed: {rmax:.2f} Mpc)")

    collected["n_parent"] = n_parent
    return collected


def _gen_field_path(nsamples, h, b1, beta, rmin, rmax, e_mag, e_czcmb,
                    M_TRGB, sigma_int, sigma_v, Vext,
                    mag_lim, mag_lim_width, cz_lim, cz_lim_width,
                    field_loader, num_rand_los, r2mu, r2z, gen, verbose):
    """Field-based (inhomogeneous Malmquist) distance sampling path.

    Galaxies are sampled from the 3D density field using accept/reject,
    ensuring both angular and radial positions follow the biased tracer
    distribution p(r, Ω) ∝ smoothclip(1 + b1*δ(r,Ω)) * r².
    """
    # LOS grid for the model covers the full rmax range (in Mpc/h)
    r_grid = np.linspace(0.1, rmax * h, 301)

    # Sampling sphere: set from selection threshold, not the full rmax
    r_sample_Mpc = rmax
    if mag_lim is not None:
        mu_max = mag_lim - M_TRGB
        sigma_tot = np.sqrt(sigma_int**2 + e_mag**2 + mag_lim_width**2)
        mu_cutoff = mu_max + 5 * sigma_tot
        r_sample_Mpc = min(10**((mu_cutoff - 25) / 5), rmax)
    elif cz_lim is not None:
        r_sample_Mpc = min(cz_lim / (h * 100) * 1.2, rmax)
    r_sphere = r_sample_Mpc * h  # Mpc/h

    if verbose:
        print(f"Field mock: 3D sampling "
              f"(r_sphere: {r_sphere:.1f} Mpc/h = "
              f"{r_sample_Mpc:.1f} Mpc, "
              f"r_grid: {r_grid[0]:.1f}–{r_grid[-1]:.1f} Mpc/h, "
              f"{len(r_grid)} points)...")

    # --- Load fields and build 3D interpolators ---
    eps = 1e-4
    density_raw = field_loader.load_density()
    density_log = np.log(density_raw + eps).astype(np.float32)
    f_density_3d = build_regular_interpolator(
        density_log, field_loader.boxsize,
        fill_value=np.float32(np.log(1 + eps)))

    delta_max = float(density_raw.max()) - 1
    max_weight = smoothclip(1 + b1 * delta_max)
    del density_raw, density_log

    velocity_3d = field_loader.load_velocity()
    f_vel_3d = []
    for i in range(3):
        f_vel_3d.append(build_regular_interpolator(
            velocity_3d[i], field_loader.boxsize,
            fill_value=np.float32(0)))
    del velocity_3d

    if verbose:
        print(f"  max delta = {delta_max:.1f}, "
              f"max weight = {max_weight:.1f}, "
              f"est. accept rate = {1 / max_weight:.4f}")

    obs = field_loader.observer_pos
    rmin_h = 0.1
    coord_frame = field_loader.coordinate_frame
    sigma_mag_tot = np.sqrt(sigma_int**2 + e_mag**2)
    sigma_cz_tot = np.sqrt(e_czcmb**2 + sigma_v**2)

    # --- Loop: sample, compute observables, select ---
    collected = {k: [] for k in [
        "RA", "dec", "r_h", "mag_obs", "cz_obs"]}
    n_total_proposed = 0
    n_total_density_accepted = 0
    batch_size = 200000

    while sum(len(v) for v in collected["RA"]) < nsamples:
        n_total_proposed += batch_size

        # Uniform in cube, cut to sphere
        xyz = gen.uniform(-r_sphere, r_sphere,
                          (batch_size, 3)).astype(np.float32)
        r_sq = np.sum(xyz**2, axis=1)
        in_shell = (r_sq < r_sphere**2) & (r_sq > rmin_h**2)
        xyz = xyz[in_shell]

        # Density accept/reject
        rho_log = f_density_3d(xyz + obs[None, :])
        rho = np.exp(rho_log) - eps
        np.clip(rho, eps, None, out=rho)
        weight = smoothclip(1 + b1 * (rho - 1))
        accept = gen.random(len(weight)) < (weight / max_weight)
        xyz = xyz[accept]
        n_total_density_accepted += len(xyz)

        if len(xyz) == 0:
            continue

        r_h = np.linalg.norm(xyz, axis=1)

        # Convert to RA/dec
        RA, dec = field_xyz_to_radec(xyz, r_h, coord_frame)

        # Radial velocity at 3D positions
        pos_box = (xyz + obs[None, :]).astype(np.float32)
        rhat = xyz / r_h[:, None]
        Vpec_field = np.zeros(len(xyz), dtype=np.float32)
        for i in range(3):
            Vpec_field += f_vel_3d[i](pos_box) * rhat[:, i]

        # Compute observables
        rhat_icrs = radec_to_cartesian(RA, dec)
        Vext_rad = rhat_icrs @ Vext
        Vpec = Vext_rad + beta * Vpec_field

        r_Mpc = r_h / h
        mu = np.asarray(r2mu(r_Mpc, h=h))
        z_cosmo = np.asarray(r2z(r_Mpc, h=h))

        mag_obs = gen.normal(M_TRGB + mu, sigma_mag_tot)
        cz_true = SPEED_OF_LIGHT * (
            (1 + z_cosmo) * (1 + Vpec / SPEED_OF_LIGHT) - 1)
        cz_obs = gen.normal(cz_true, sigma_cz_tot)

        # Apply selection
        sel = _apply_selection(mag_obs, cz_obs,
                               mag_lim, mag_lim_width,
                               cz_lim, cz_lim_width, gen)

        collected["RA"].append(RA[sel])
        collected["dec"].append(dec[sel])
        collected["r_h"].append(r_h[sel])
        collected["mag_obs"].append(mag_obs[sel])
        collected["cz_obs"].append(cz_obs[sel])

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

    # --- Interpolate full LOS for selected hosts ---
    if verbose:
        print(f"  interpolating LOS for {nsamples} hosts...")
    los_density, los_velocity = interpolate_los_density_velocity(
        field_loader, r_grid, collected["RA"], collected["dec"],
        verbose=verbose)

    # Random LOS for selection normalization
    RA_rand = gen.uniform(0, 360, num_rand_los)
    dec_rand = np.rad2deg(np.arcsin(gen.uniform(-1, 1, num_rand_los)))
    rand_density, rand_velocity = interpolate_los_density_velocity(
        field_loader, r_grid, RA_rand, dec_rand, verbose=False)

    result = {
        "RA": collected["RA"],
        "dec": collected["dec"],
        "r": collected["r_h"],
        "mag_obs": collected["mag_obs"],
        "cz_obs": collected["cz_obs"],
        "n_parent": n_total_density_accepted,
        # Host LOS data: (1, nsamples, n_r)
        "host_los_density": los_density[None, ...],
        "host_los_velocity": los_velocity[None, ...],
        "host_los_r": r_grid,
        # Random LOS data: (1, num_rand_los, n_r)
        "rand_los_density": rand_density[None, ...],
        "rand_los_velocity": rand_velocity[None, ...],
        "rand_los_r": r_grid,
        "rand_los_RA": RA_rand,
        "rand_los_dec": dec_rand,
    }
    return result


def gen_TRGB_mock(nsamples=480, Om=0.3, e_mag=0.05, e_czcmb=10.0,
                  rmin=0.5, rmax=40.0,
                  mag_lim=25.0, mag_lim_width=0.75,
                  cz_lim=None, cz_lim_width=None,
                  true_params=None, anchors=None,
                  noisy_anchors=True, field_loader=None,
                  num_rand_los=1000, seed=42, verbose=True):
    """Generate a mock TRGB survey compatible with TRGBModel.

    When ``field_loader`` is None (default), distances are drawn from
    p(r) ~ r^2 on [rmin, rmax] (homogeneous).  When a field loader is
    provided, distances are drawn from p(r) ~ (1 + b1*delta(r)) * r^2
    using the density field, and the field's radial peculiar velocity
    is included in the observed cz.

    Selection (optional):
      - mag_lim  : sigmoid cut p(sel) = Phi((mag_lim - m) / mag_lim_width)
      - cz_lim   : hard (cz_lim_width=None) or sigmoid cz cut

    Returns
    -------
    data : dict
        Data dict matching TRGBModel expectations.
    true_params : dict
        True parameter values used.
    n_parent : int
        Total number of objects drawn before selection.
    """
    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
    anch = {**DEFAULT_ANCHORS, **(anchors or {})}
    gen = np.random.default_rng(seed)

    H0 = tp["H0"]
    M_TRGB = tp["M_TRGB"]
    sigma_int = tp["sigma_int"]
    sigma_v = tp["sigma_v"]

    h = H0 / 100
    beta = tp["beta"]
    r2mu = Distance2Distmod(Om0=Om)
    r2z = Distance2Redshift(Om0=Om)
    Vext = tp["Vext_mag"] * galactic_to_radec_cartesian(
        tp["Vext_ell"], tp["Vext_b"])

    if field_loader is not None:
        collected = _gen_field_path(
            nsamples, h, tp["b1"], beta, rmin, rmax, e_mag, e_czcmb,
            M_TRGB, sigma_int, sigma_v, Vext,
            mag_lim, mag_lim_width, cz_lim, cz_lim_width,
            field_loader, num_rand_los, r2mu, r2z, gen, verbose)
    else:
        collected = _gen_homogeneous_path(
            nsamples, h, rmin, rmax, e_mag, e_czcmb,
            M_TRGB, sigma_int, sigma_v, Vext,
            mag_lim, mag_lim_width, cz_lim, cz_lim_width,
            r2mu, r2z, gen, verbose)
    n_parent = collected.pop("n_parent")

    # --- Anchor observations ---
    mu_LMC_true = anch["mu_LMC"]
    mu_N4258_true = anch["mu_N4258"]
    mag_LMC_true = M_TRGB + mu_LMC_true
    mag_N4258_true = M_TRGB + mu_N4258_true

    if noisy_anchors:
        mu_LMC_obs = float(gen.normal(mu_LMC_true, anch["e_mu_LMC"]))
        mu_N4258_obs = float(gen.normal(mu_N4258_true, anch["e_mu_N4258"]))
        mag_LMC_obs = float(gen.normal(mag_LMC_true, anch["e_mag_LMC_TRGB"]))
        mag_N4258_obs = float(gen.normal(
            mag_N4258_true, anch["e_mag_N4258_TRGB"]))
    else:
        mu_LMC_obs = mu_LMC_true
        mu_N4258_obs = mu_N4258_true
        mag_LMC_obs = mag_LMC_true
        mag_N4258_obs = mag_N4258_true

    # --- Build data dict ---
    n_kept = len(collected["RA"])
    data = {
        "RA_host": collected["RA"],
        "dec_host": collected["dec"],
        "mag_obs": collected["mag_obs"],
        "e_mag_obs": np.full(n_kept, e_mag),
        "czcmb": collected["cz_obs"],
        "e_czcmb": np.full(n_kept, e_czcmb),
        "e_mag_median": float(e_mag),
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

    # Add LOS data for field-based mocks
    for k in ["host_los_density", "host_los_velocity", "host_los_r",
              "rand_los_density", "rand_los_velocity", "rand_los_r",
              "rand_los_RA", "rand_los_dec"]:
        if k in collected:
            data[k] = collected[k]
    if "host_los_r" in collected:
        data["has_rand_los"] = True

    # Store true Cartesian Vext for reference
    tp["Vext_x"], tp["Vext_y"], tp["Vext_z"] = Vext

    return data, tp, n_parent
