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
"""Mock generator for 2MTF K-band TFR surveys."""
import numpy as np
from scipy.stats import norm

from ..cosmography import Distance2Distmod, Distance2Redshift
from ..util import SPEED_OF_LIGHT, radec_to_cartesian

DEFAULT_TRUE_PARAMS = {
    "H0": 73.0,
    "a_TFR": -21.0,
    "b_TFR": -8.0,
    "c_TFR": 0.0,
    "sigma_int": 0.4,
    "sigma_v": 150.0,
    "eta_mean": 0.0,
    "eta_std": 0.08,
    "Vext_x": 150.0,
    "Vext_y": -50.0,
    "Vext_z": 50.0,
    "beta": 0.43,
    "b1": 1.2,
}


def _get_absmag_TFR(eta, a, b, c=0.0):
    """TFR absolute magnitude."""
    return a + b * eta + np.where(eta > 0, c * eta**2, 0.0)


def _smoothclip(x, tau=0.1):
    """Smooth zero-clipping matching the model's smoothclip_nr."""
    return 0.5 * (x + np.sqrt(x**2 + tau**2))


def _apply_2MTF_selection(mag_obs, eta_obs, mag_lim, mag_lim_width,
                          eta_min_sel, eta_max_sel, gen):
    """Return boolean selection mask for 2MTF cuts."""
    n = len(mag_obs)
    sel = np.ones(n, dtype=bool)
    if mag_lim is not None:
        if mag_lim_width is not None:
            p_sel = norm.cdf((mag_lim - mag_obs) / mag_lim_width)
            sel &= gen.random(n) < p_sel
        else:
            sel &= mag_obs < mag_lim
    if eta_min_sel is not None:
        sel &= eta_obs > eta_min_sel
    if eta_max_sel is not None:
        sel &= eta_obs < eta_max_sel
    return sel


def _field_xyz_to_radec(pos_rel, r, coordinate_frame):
    """Convert field-frame Cartesian offsets to ICRS (RA, dec) in degrees."""
    from ..util import cartesian_to_radec, galactic_to_radec
    x, y, z = pos_rel[:, 0], pos_rel[:, 1], pos_rel[:, 2]
    if coordinate_frame == "icrs":
        return cartesian_to_radec(x, y, z)
    elif coordinate_frame == "galactic":
        ell = np.rad2deg(np.arctan2(y, x))
        b = np.rad2deg(np.arcsin(z / r))
        return galactic_to_radec(ell, b)
    elif coordinate_frame == "supergalactic":
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        sgl = np.rad2deg(np.arctan2(y, x))
        sgb = np.rad2deg(np.arcsin(z / r))
        c = SkyCoord(sgl=sgl * u.deg, sgb=sgb * u.deg,
                     frame='supergalactic')
        return c.icrs.ra.deg, c.icrs.dec.deg
    else:
        raise ValueError(
            f"Unknown coordinate frame: {coordinate_frame}")


def _gen_homogeneous_path(nsamples, h, rmin, rmax, e_mag, e_eta,
                          a_TFR, b_TFR, c_TFR, sigma_int, sigma_v,
                          eta_mean, eta_std, Vext,
                          mag_lim, mag_lim_width,
                          eta_min_sel, eta_max_sel,
                          r2mu, r2z, gen, verbose):
    """Homogeneous (no field) distance sampling path."""
    collected = {k: [] for k in ["RA", "dec", "r", "mag", "eta",
                                 "e_mag", "e_eta", "czcmb"]}
    n_accepted = 0
    n_parent = 0
    batch = max(int(3 * nsamples), 500)

    while n_accepted < nsamples:
        RA = gen.uniform(0, 360, batch)
        dec = np.rad2deg(np.arcsin(gen.uniform(-1, 1, batch)))
        rhat = radec_to_cartesian(RA, dec)

        u = gen.random(batch)
        r = (rmin**3 + u * (rmax**3 - rmin**3))**(1 / 3)

        eta_true = gen.normal(eta_mean, eta_std, batch)
        M_true = _get_absmag_TFR(eta_true, a_TFR, b_TFR, c_TFR)
        mu = np.asarray(r2mu(r, h=h))
        z_cosmo = np.asarray(r2z(r, h=h))

        sigma_mag_tot = np.sqrt(sigma_int**2 + e_mag**2)
        mag_obs = gen.normal(M_true + mu, sigma_mag_tot)
        eta_obs = gen.normal(eta_true, e_eta)

        Vext_rad = rhat @ Vext
        cz_true = SPEED_OF_LIGHT * (
            (1 + z_cosmo) * (1 + Vext_rad / SPEED_OF_LIGHT) - 1)
        cz_obs = gen.normal(cz_true, sigma_v)

        sel = _apply_2MTF_selection(mag_obs, eta_obs, mag_lim, mag_lim_width,
                                    eta_min_sel, eta_max_sel, gen)
        n_batch = int(sel.sum())
        n_parent += batch
        n_accepted += n_batch

        collected["RA"].append(RA[sel])
        collected["dec"].append(dec[sel])
        collected["r"].append(r[sel])
        collected["mag"].append(mag_obs[sel])
        collected["eta"].append(eta_obs[sel])
        collected["e_mag"].append(np.full(n_batch, e_mag))
        collected["e_eta"].append(np.full(n_batch, e_eta))
        collected["czcmb"].append(cz_obs[sel])

    for k in collected:
        collected[k] = np.concatenate(collected[k])[:nsamples]

    if verbose:
        sel_frac = n_accepted / n_parent
        print(f"Generated {nsamples} 2MTF hosts "
              f"(acceptance {sel_frac:.3f}, {n_parent} drawn).")

    collected["n_parent"] = n_parent
    return collected


def _gen_field_path(nsamples, h, b1, beta, rmin, rmax, e_mag, e_eta,
                    a_TFR, b_TFR, c_TFR, sigma_int, sigma_v,
                    eta_mean, eta_std, Vext,
                    mag_lim, mag_lim_width,
                    eta_min_sel, eta_max_sel,
                    field_loader, r2mu, r2z, gen, verbose):
    """Field-based (inhomogeneous Malmquist) distance sampling path."""
    from ..field import interpolate_los_density_velocity
    from ..field.field_interp import build_regular_interpolator

    # LOS grid in Mpc/h (field coordinates), matching model convention:
    # model queries at rh_grid = r_model * h (Mpc/h).
    r_los_grid = np.linspace(0.1, rmax * h, 301)

    # Sampling sphere
    r_sample_Mpc = rmax
    if mag_lim is not None:
        M_bright = a_TFR + b_TFR * 0.2  # bright end of TFR
        mu_max = mag_lim - M_bright
        sigma_tot = np.sqrt(sigma_int**2 + e_mag**2)
        mu_cutoff = mu_max + 5 * sigma_tot
        r_sample_Mpc = min(10**((mu_cutoff - 25) / 5), rmax)
    r_sphere = r_sample_Mpc * h

    if verbose:
        print(f"Field mock: 3D sampling "
              f"(r_sphere: {r_sphere:.1f} Mpc/h = "
              f"{r_sample_Mpc:.1f} Mpc, "
              f"r_los_grid: {r_los_grid[0]:.1f}-{r_los_grid[-1]:.1f} Mpc/h, "
              f"{len(r_los_grid)} points)...")

    # Load fields and build 3D interpolators
    eps = 1e-4
    density_raw = field_loader.load_density()
    density_log = np.log(density_raw + eps).astype(np.float32)
    f_density_3d = build_regular_interpolator(
        density_log, field_loader.boxsize,
        fill_value=np.float32(np.log(1 + eps)))

    delta_max = float(density_raw.max()) - 1
    max_weight = _smoothclip(1 + b1 * delta_max)
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

    # Sample, compute observables, select
    collected = {k: [] for k in [
        "RA", "dec", "r_h", "mag", "eta", "e_mag", "e_eta", "czcmb"]}
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
        weight = _smoothclip(1 + b1 * (rho - 1))
        accept = gen.random(len(weight)) < (weight / max_weight)
        xyz = xyz[accept]
        n_total_density_accepted += len(xyz)

        if len(xyz) == 0:
            continue

        r_h = np.linalg.norm(xyz, axis=1)
        RA, dec = _field_xyz_to_radec(xyz, r_h, coord_frame)

        # Radial velocity at 3D positions
        pos_box = (xyz + obs[None, :]).astype(np.float32)
        rhat_field = xyz / r_h[:, None]
        Vpec_field = np.zeros(len(xyz), dtype=np.float32)
        for i in range(3):
            Vpec_field += f_vel_3d[i](pos_box) * rhat_field[:, i]

        # Compute observables
        rhat_icrs = radec_to_cartesian(RA, dec)
        Vext_rad = rhat_icrs @ Vext
        Vpec = Vext_rad + beta * Vpec_field

        r_Mpc = r_h / h
        mu = np.asarray(r2mu(r_Mpc, h=h))
        z_cosmo = np.asarray(r2z(r_Mpc, h=h))

        n_gal = len(xyz)
        eta_true = gen.normal(eta_mean, eta_std, n_gal)
        M_true = _get_absmag_TFR(eta_true, a_TFR, b_TFR, c_TFR)
        mag_obs = gen.normal(M_true + mu, sigma_mag_tot)
        eta_obs = gen.normal(eta_true, e_eta)

        cz_true = SPEED_OF_LIGHT * (
            (1 + z_cosmo) * (1 + Vpec / SPEED_OF_LIGHT) - 1)
        cz_obs = gen.normal(cz_true, sigma_v)

        sel = _apply_2MTF_selection(mag_obs, eta_obs, mag_lim, mag_lim_width,
                                    eta_min_sel, eta_max_sel, gen)
        n_sel = int(sel.sum())

        collected["RA"].append(RA[sel])
        collected["dec"].append(dec[sel])
        collected["r_h"].append(r_h[sel])
        collected["mag"].append(mag_obs[sel])
        collected["eta"].append(eta_obs[sel])
        collected["e_mag"].append(np.full(n_sel, e_mag))
        collected["e_eta"].append(np.full(n_sel, e_eta))
        collected["czcmb"].append(cz_obs[sel])

    del f_density_3d, f_vel_3d

    for k in collected:
        collected[k] = np.concatenate(collected[k])[:nsamples]

    if verbose:
        print(f"  {n_total_proposed} proposed, "
              f"{n_total_density_accepted} density-accepted, "
              f"{nsamples} after selection")

    # Interpolate full LOS for selected hosts
    if verbose:
        print(f"  interpolating LOS for {nsamples} hosts...")
    los_density, los_velocity = interpolate_los_density_velocity(
        field_loader, r_los_grid, collected["RA"], collected["dec"],
        verbose=verbose)

    collected["n_parent"] = n_total_density_accepted
    # Host LOS data: (1, nsamples, n_r), r grid in Mpc/h
    collected["host_los_density"] = los_density[None, ...]
    collected["host_los_velocity"] = los_velocity[None, ...]
    collected["host_los_r"] = r_los_grid
    return collected


def gen_EDD_2MTF_mock(nsamples=500, Om=0.3, e_mag=0.04, e_eta=0.01,
                      rmin=0.5, rmax=150.0,
                      mag_lim=11.25, mag_lim_width=None,
                      eta_min_sel=None, eta_max_sel=None,
                      true_params=None, field_loader=None,
                      seed=42, verbose=True):
    """Generate a mock 2MTF survey compatible with EDD2MTFModel.

    When ``field_loader`` is None (default), distances are drawn from
    p(r) ~ r^2 on [rmin, rmax] (homogeneous).  When a field loader is
    provided, distances are drawn from p(r) ~ (1 + b1*delta(r)) * r^2
    using the density field, and the field's radial peculiar velocity
    is included in the observed cz.

    Selection:
      - mag_lim: apparent K-band magnitude limit
      - eta_min_sel/eta_max_sel: linewidth cuts on observed eta
    """
    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
    gen = np.random.default_rng(seed)

    H0 = tp["H0"]
    a_TFR = tp["a_TFR"]
    b_TFR = tp["b_TFR"]
    c_TFR = tp["c_TFR"]
    sigma_int = tp["sigma_int"]
    sigma_v = tp["sigma_v"]
    eta_mean = tp["eta_mean"]
    eta_std = tp["eta_std"]

    h = H0 / 100
    beta = tp["beta"]
    r2mu = Distance2Distmod(Om0=Om)
    r2z = Distance2Redshift(Om0=Om)
    Vext = np.array([tp["Vext_x"], tp["Vext_y"], tp["Vext_z"]])

    if field_loader is not None:
        collected = _gen_field_path(
            nsamples, h, tp["b1"], beta, rmin, rmax, e_mag, e_eta,
            a_TFR, b_TFR, c_TFR, sigma_int, sigma_v,
            eta_mean, eta_std, Vext,
            mag_lim, mag_lim_width, eta_min_sel, eta_max_sel,
            field_loader, r2mu, r2z, gen, verbose)
    else:
        collected = _gen_homogeneous_path(
            nsamples, h, rmin, rmax, e_mag, e_eta,
            a_TFR, b_TFR, c_TFR, sigma_int, sigma_v,
            eta_mean, eta_std, Vext,
            mag_lim, mag_lim_width, eta_min_sel, eta_max_sel,
            r2mu, r2z, gen, verbose)
    n_parent = collected.pop("n_parent")

    n_kept = len(collected["RA"])
    data = {
        "RA_host": collected["RA"],
        "dec_host": collected["dec"],
        "mag": collected["mag"],
        "e_mag": collected["e_mag"],
        "eta": collected["eta"],
        "e_eta": collected["e_eta"],
        "czcmb": collected["czcmb"],
        "e_czcmb": np.full(n_kept, sigma_v),
        "e_mag_median": e_mag,
        "e_eta_median": e_eta,
        "has_rand_los": False,
    }

    # Add LOS data for field-based mocks
    for k in ["host_los_density", "host_los_velocity", "host_los_r"]:
        if k in collected:
            data[k] = collected[k]

    return data, tp, n_parent
