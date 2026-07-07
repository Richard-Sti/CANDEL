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

    if which_bias == "uniform":
        return np.zeros_like(rho, dtype=np.float64)
    if which_bias == "powerlaw":
        return params[0] * log_rho
    if which_bias in (
            "unity", "linear", "linear_from_beta",
            "linear_from_beta_stochastic"):
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


def _validated_smoothing_scale(scale, cellsize, label):
    """Return an optional smoothing scale validated against the voxel size."""
    if scale is None:
        return None
    scale = float(scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(f"`{label}` must be finite and non-negative.")
    if scale == 0.0:
        return None
    if scale <= cellsize:
        raise ValueError(
            f"`{label}` must exceed the field voxel size {cellsize:g} "
            f"Mpc/h; got {scale:g} Mpc/h.")
    return scale


def _density_max_within_radius(density, boxsize, observer_pos, radius):
    """Maximum grid-cell density within a sphere around the observer."""
    if radius is None:
        return float(np.max(density))

    ngrid = density.shape[0]
    cellsize = float(boxsize) / ngrid
    x = np.linspace(0.5 * cellsize, float(boxsize) - 0.5 * cellsize, ngrid)
    dx2 = (x - float(observer_pos[0]))**2
    dy2 = (x - float(observer_pos[1]))**2
    dz2 = (x - float(observer_pos[2]))**2
    yz2 = dy2[:, None] + dz2[None, :]
    r2 = float(radius)**2
    rho_max = -np.inf
    for i, x2 in enumerate(dx2):
        keep = yz2 <= r2 - x2
        if np.any(keep):
            rho_max = max(rho_max, float(np.max(density[i][keep])))
    if not np.isfinite(rho_max):
        raise ValueError("No density grid cells lie within the PPC radius.")
    return rho_max


def build_field_pool_evaluator(field_loader, density_divisor=None,
                               field_smoothing_scale=None,
                               velocity_field_smoothing_scale=None,
                               max_radius_h=None, verbose=True):
    """Load field products once and build interpolators for PPC pools."""
    obs = field_loader.observer_pos
    coord_frame = field_loader.coordinate_frame

    eps = 1e-4
    from ..field.field_interp import (apply_gaussian_smoothing,
                                      build_regular_interpolator)

    fprint("loading density field once for pool evaluator...",
           verbose=verbose)
    density_raw = field_loader.load_density()
    cellsize = float(field_loader.boxsize) / density_raw.shape[0]
    field_smoothing_scale = _validated_smoothing_scale(
        field_smoothing_scale, cellsize, "field_smoothing_scale")
    velocity_field_smoothing_scale = _validated_smoothing_scale(
        velocity_field_smoothing_scale, cellsize,
        "velocity_field_smoothing_scale")
    if density_divisor is not None:
        density_raw = density_raw / density_divisor
    if field_smoothing_scale is not None:
        fprint("applying Gaussian smoothing to density with scale "
               f"{field_smoothing_scale:.1f} Mpc/h.",
               verbose=verbose)
        density_raw = apply_gaussian_smoothing(
            density_raw.astype(np.float32, copy=False),
            field_smoothing_scale, field_loader.boxsize, make_copy=True)
    density_log = np.log(density_raw + eps).astype(np.float32)
    f_density = build_regular_interpolator(
        density_log, field_loader.boxsize,
        fill_value=np.float32(np.log(1 + eps)))
    delta_max = float(density_raw.max()) - 1
    delta_max_within_radius = (
        _density_max_within_radius(
            density_raw, field_loader.boxsize, obs, max_radius_h) - 1)
    del density_raw, density_log

    fprint("loading velocity field once for pool evaluator...",
           verbose=verbose)
    velocity_3d = field_loader.load_velocity()
    f_vel = []
    for i in range(3):
        v_comp = velocity_3d[i]
        if velocity_field_smoothing_scale is not None:
            if i == 0:
                fprint("applying Gaussian smoothing to velocity with scale "
                       f"{velocity_field_smoothing_scale:.1f} Mpc/h.",
                       verbose=verbose)
            v_comp = apply_gaussian_smoothing(
                v_comp.astype(np.float32, copy=False),
                velocity_field_smoothing_scale, field_loader.boxsize,
                make_copy=True)
        f_vel.append(build_regular_interpolator(
            v_comp, field_loader.boxsize,
            fill_value=np.float32(0)))
    del velocity_3d

    return {
        "observer_pos": obs,
        "coordinate_frame": coord_frame,
        "f_density": f_density,
        "f_vel": tuple(f_vel),
        "delta_max": delta_max,
        "delta_max_within_radius": delta_max_within_radius,
        "eps": eps,
    }


def _sample_galactic_masked_xyz(gen, r_sphere, pool_size, b_min, rmin_h):
    """Sample field-frame Galactic offsets directly outside |b| < b_min."""
    r_min3 = float(rmin_h)**3
    r_max3 = float(r_sphere)**3
    r_h = gen.uniform(r_min3, r_max3, int(pool_size))**(1.0 / 3.0)
    ell = gen.uniform(0.0, 2.0 * np.pi, int(pool_size))
    sin_b_min = np.sin(np.deg2rad(float(b_min)))
    abs_sin_b = gen.uniform(sin_b_min, 1.0, int(pool_size))
    sign = np.where(gen.integers(0, 2, int(pool_size)) == 0, -1.0, 1.0)
    sin_b = sign * abs_sin_b
    cos_b = np.sqrt(np.maximum(0.0, 1.0 - sin_b**2))
    return np.column_stack([
        r_h * cos_b * np.cos(ell),
        r_h * cos_b * np.sin(ell),
        r_h * sin_b,
    ]).astype(np.float32)


def build_field_pool(field_loader, r_sphere, pool_size, gen,
                     rmin_h=0.1, density_divisor=None,
                     field_smoothing_scale=None,
                     velocity_field_smoothing_scale=None,
                     field_evaluator=None, b_min=None, verbose=True):
    """Pre-sample 3D positions and evaluate density/velocity in one batch.

    Returns dict with keys: r_h, rho, v_los, RA, dec, rhat_icrs, delta_max.
    """
    if field_evaluator is None:
        field_evaluator = build_field_pool_evaluator(
            field_loader, density_divisor=density_divisor,
            field_smoothing_scale=field_smoothing_scale,
            velocity_field_smoothing_scale=velocity_field_smoothing_scale,
            verbose=verbose)

    obs = field_evaluator["observer_pos"]
    coord_frame = field_evaluator["coordinate_frame"]
    f_density = field_evaluator["f_density"]
    f_vel = field_evaluator["f_vel"]
    eps = field_evaluator["eps"]
    delta_max = field_evaluator["delta_max"]

    # Sample positions uniformly in sphere.
    if (b_min is not None and b_min > 0.0
            and coord_frame == "galactic"):
        fprint(f"sampling {pool_size} Galactic-masked candidate positions "
               f"(r_sphere={r_sphere:.1f} Mpc/h, |b| >= {b_min:g} deg)...",
               verbose=verbose)
        xyz = _sample_galactic_masked_xyz(
            gen, r_sphere, pool_size, b_min, rmin_h)
    else:
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
    fprint(f"evaluating density/velocity at {len(xyz)} positions...",
           verbose=verbose)
    pos_box = (xyz + obs[None, :]).astype(np.float32)
    rho_log = f_density(pos_box)
    rho = np.exp(rho_log) - eps
    np.clip(rho, eps, None, out=rho)

    # Evaluate radial velocity
    rhat = xyz / r_h[:, None]
    v_los = np.zeros(len(xyz), dtype=np.float32)
    for i in range(3):
        v_los += f_vel[i](pos_box) * rhat[:, i]

    del pos_box

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
