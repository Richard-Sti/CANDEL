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
"""Shared utilities for mock generation and posterior predictive checks."""
import numpy as np

from ..field.field_interp import build_regular_interpolator
from ..model.pv_utils import validate_galaxy_bias
from ..util import (cartesian_to_radec, fprint, galactic_to_radec,
                    radec_to_cartesian)


def smoothclip(x, tau=0.1):
    """Smooth zero-clipping matching the model's smoothclip_nr."""
    return 0.5 * (x + np.sqrt(x**2 + tau**2))


def galaxy_bias_params_from_values(values, which_bias, Om=None, idx=None):
    """Return galaxy-bias parameters in ``lp_galaxy_bias`` order."""
    validate_galaxy_bias(which_bias)

    def get(name, default=None):
        if name in values:
            val = values[name]
        elif default is not None:
            val = default
        else:
            raise KeyError(name)
        if idx is None:
            return val
        arr = np.asarray(val)
        return arr[idx] if arr.ndim > 0 else val

    if which_bias == "uniform":
        return []
    if which_bias == "unity":
        return [1.0]
    if which_bias == "powerlaw":
        return [get("alpha")]
    if which_bias == "linear":
        return [get("b1")]
    if which_bias == "linear_from_beta":
        if "b1" in values:
            return [get("b1")]
        return [Om**0.55 / get("beta")]
    if which_bias == "linear_from_beta_stochastic":
        if "b1" in values:
            return [get("b1")]
        return [Om**0.55 / get("beta") + get("delta_b1")]
    if which_bias == "double_powerlaw":
        alpha_low = get("alpha_low")
        if "alpha_high" in values:
            alpha_high = get("alpha_high")
        else:
            alpha_high = alpha_low * get("alpha_high_frac")
        return [
            alpha_low, alpha_high, get("log_rho_t"), get("log_rho_width"),
        ]
    if which_bias == "quadratic":
        return [get("b1"), get("b2")]
    if which_bias == "cubic":
        return [get("b1"), get("b2"), get("b3")]
    raise ValueError(f"Invalid galaxy bias model '{which_bias}'.")


def galaxy_bias_log_weight(rho, bias_params, which_bias):
    """Evaluate model galaxy-bias log weights with NumPy model formulas."""
    rho = np.asarray(rho, dtype=np.float64)
    delta = rho - 1.0
    log_rho = np.log(np.clip(rho, 1e-6, None))
    params = [np.asarray(p) for p in bias_params]

    if which_bias in ("uniform", "unity"):
        return np.zeros_like(rho, dtype=np.float64)
    if which_bias == "powerlaw":
        return params[0] * log_rho
    if which_bias in (
            "linear", "linear_from_beta", "linear_from_beta_stochastic"):
        return np.log(smoothclip(1.0 + params[0] * delta))
    if which_bias == "double_powerlaw":
        alpha_low, alpha_high, log_rho_t, log_rho_width = params
        log_x = log_rho - log_rho_t
        z = log_x / log_rho_width
        return (
            alpha_low * log_x
            + ((alpha_high - alpha_low) * log_rho_width
               * np.logaddexp(0.0, z)))
    if which_bias == "quadratic":
        b1, b2 = params
        return np.log(smoothclip(1.0 + b1 * delta + b2 * delta**2))
    if which_bias == "cubic":
        b1, b2, b3 = params
        return np.log(
            smoothclip(1.0 + b1 * delta + b2 * delta**2 + b3 * delta**3))
    raise ValueError(f"Invalid galaxy bias model '{which_bias}'.")


def galaxy_bias_weight(rho, bias_params, which_bias):
    """Evaluate model galaxy-bias weights with overflow protection."""
    log_weight = galaxy_bias_log_weight(rho, bias_params, which_bias)
    return np.exp(np.clip(log_weight, -50.0, 50.0))


def field_xyz_to_radec(pos_rel, r, coordinate_frame):
    """Convert field-frame Cartesian offsets to ICRS (RA, dec) in degrees."""
    x, y, z = pos_rel[:, 0], pos_rel[:, 1], pos_rel[:, 2]
    if coordinate_frame == "icrs":
        _, ra, dec = cartesian_to_radec(x, y, z)
        return ra, dec
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
        raise ValueError(f"Unknown coordinate frame: {coordinate_frame}")


def compute_r_max_selection(mag_lim, M_abs, sigma_int, e_mag,
                            mag_lim_width=0.0, cz_lim=None, h=1.0,
                            r_max=150.0, colour_mean=None, c_star=None,
                            colour_std=None, alpha_c=0.2):
    """Tighten sampling sphere based on selection cuts.

    All arguments can be scalars or arrays; worst-case (most permissive)
    values are used.
    """
    if mag_lim is not None and not isinstance(mag_lim, str):
        ml = float(np.max(mag_lim))
        M_eff = M_abs
        if colour_mean is not None and c_star is not None:
            M_eff = M_abs + np.asarray(alpha_c) * (
                np.asarray(colour_mean) - np.asarray(c_star))
        M_min = float(np.min(M_eff))
        sint_max = float(np.max(sigma_int))
        e_max = float(np.max(e_mag))
        mw = float(np.max(mag_lim_width)) if mag_lim_width is not None else 0.0
        if isinstance(mw, str):
            mw = 0.0
        cstd = 0.0 if colour_std is None else float(
            np.max(np.abs(alpha_c) * np.asarray(colour_std)))

        sigma_tot = np.sqrt(sint_max**2 + e_max**2 + mw**2 + cstd**2)
        mu_cutoff = ml - M_min + 5 * sigma_tot
        return min(10**((mu_cutoff - 25) / 5), r_max)
    elif cz_lim is not None and not isinstance(cz_lim, str):
        cl = float(np.max(cz_lim))
        h_min = float(np.min(h))
        return min(cl / (h_min * 100) * 1.5, r_max)
    return r_max


def build_field_pool(field_loader, r_sphere, pool_size, gen,
                     rmin_h=0.1, density_divisor=None, verbose=True):
    """Pre-sample 3D positions and evaluate density/velocity in one batch.

    Returns dict with keys: r_h, rho, v_los, RA, dec, rhat_icrs, delta_max.
    """
    obs = field_loader.observer_pos
    coord_frame = field_loader.coordinate_frame

    eps = 1e-4
    fprint("loading density field for pool...", verbose=verbose)
    density_raw = field_loader.load_density()
    if density_divisor is not None:
        density_raw = density_raw / density_divisor
    density_log = np.log(density_raw + eps).astype(np.float32)
    f_density = build_regular_interpolator(
        density_log, field_loader.boxsize,
        fill_value=np.float32(np.log(1 + eps)))
    delta_max = float(density_raw.max()) - 1
    del density_raw, density_log

    fprint("loading velocity field for pool...", verbose=verbose)
    velocity_3d = field_loader.load_velocity()
    f_vel = []
    for i in range(3):
        f_vel.append(build_regular_interpolator(
            velocity_3d[i], field_loader.boxsize,
            fill_value=np.float32(0)))
    del velocity_3d

    # Sample positions uniformly in sphere
    n_cube = int(pool_size * 2.0)
    fprint(f"sampling {n_cube} candidate positions "
           f"(r_sphere={r_sphere:.1f} Mpc/h)...", verbose=verbose)
    xyz = gen.uniform(-r_sphere, r_sphere,
                      (n_cube, 3)).astype(np.float32)
    r_sq = np.sum(xyz**2, axis=1)
    mask = (r_sq < r_sphere**2) & (r_sq > rmin_h**2)
    xyz = xyz[mask]
    if len(xyz) > pool_size:
        xyz = xyz[:pool_size]

    r_h = np.linalg.norm(xyz, axis=1)

    # Evaluate density
    fprint(f"evaluating density at {len(xyz)} positions...", verbose=verbose)
    pos_box = (xyz + obs[None, :]).astype(np.float32)
    rho_log = f_density(pos_box)
    rho = np.exp(rho_log) - eps
    np.clip(rho, eps, None, out=rho)

    # Evaluate radial velocity
    fprint("evaluating velocity...", verbose=verbose)
    rhat = xyz / r_h[:, None]
    v_los = np.zeros(len(xyz), dtype=np.float32)
    for i in range(3):
        v_los += f_vel[i](pos_box) * rhat[:, i]

    del f_density, f_vel, pos_box

    # Convert to ICRS
    RA, dec = field_xyz_to_radec(xyz, r_h, coord_frame)
    rhat_icrs = radec_to_cartesian(RA, dec)

    fprint(f"pool ready ({len(xyz)} positions).", verbose=verbose)
    return {
        "r_h": r_h.astype(np.float64),
        "rho": rho.astype(np.float64),
        "v_los": v_los.astype(np.float64),
        "RA": RA.astype(np.float64),
        "dec": dec.astype(np.float64),
        "rhat_icrs": rhat_icrs.astype(np.float64),
        "delta_max": delta_max,
    }
