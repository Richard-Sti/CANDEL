# Copyright (C) 2025 Richard Stiskalek
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
"""Simple mock generator for the CF4 TFR sample."""
import numpy as np
from scipy.integrate import cumulative_simpson

from ..field import interpolate_los_density_velocity
from ..util import (SPEED_OF_LIGHT, fprint, galactic_to_radec,
                    galactic_to_radec_cartesian, radec_to_cartesian)


def sample_distance(r_grid, los_density, b1, R, p, n, gen):
    los_delta = los_density - 1
    pi_r = (1 + b1 * los_delta) * r_grid**p * np.exp(-(r_grid / R)**n)
    cdf_r = cumulative_simpson(pi_r, x=r_grid, initial=0)
    cdf_r /= cdf_r[-1]
    return np.interp(gen.uniform(), cdf_r, r_grid)


def gen_TFR_mock(nsamples, r_grid, Vext_mag, Vext_ell, Vext_b, sigma_v, beta,
                 a_TFR, b_TFR, c_TFR, sigma_int, zeropoint_dipole_mag,
                 zeropoint_dipole_ell, zeropoint_dipole_b, h, e_mag,
                 eta_prior_mean, eta_prior_std, e_eta, b_min, zcmb_max,
                 R, p, n, field_loader, r2distmod, r2z, Om=0.3, seed=42,
                 verbose=True):
    """
    Generate a mock TFR survey with distances sampled from an empirical
    distribution, without any further selection effects.
    """
    gen = np.random.default_rng(seed)

    # Sample the sky-coordinates of the sample.
    ell = gen.uniform(0, 360, size=nsamples)
    if b_min is None:
        b = np.arcsin(gen.uniform(-1, 1, size=nsamples))
    else:
        b = np.arcsin(gen.uniform(np.sin(np.deg2rad(b_min)), 1, size=nsamples))
        b[gen.random(nsamples) < 0.5] *= -1

    b = np.rad2deg(b)
    RA, dec = galactic_to_radec(ell, b)
    rhat = radec_to_cartesian(RA, dec)

    Vext = Vext_mag * galactic_to_radec_cartesian(Vext_ell, Vext_b)
    Vext_rad = np.sum(Vext[None, :] * rhat, axis=1)

    if field_loader is not None:
        los_density, los_velocity = interpolate_los_density_velocity(
            field_loader, r_grid, RA, dec, verbose=False)
    else:
        los_density = np.ones((nsamples, len(r_grid)))
        los_velocity = np.zeros_like(los_density)

    r = np.full(nsamples, np.nan)
    Vpec = np.full(nsamples, np.nan)
    if beta == 0:
        b1 = 0.
    else:
        b1 = Om**0.5 / beta
    for i in range(nsamples):
        Vpec[i] = Vext_rad[i]
        r[i] = sample_distance(r_grid, los_density[i], b1, R, p, n, gen)
        Vpec[i] += beta * np.interp(r[i], r_grid, los_velocity[i])

    eta = gen.normal(eta_prior_mean, eta_prior_std, size=nsamples)
    eta_obs = gen.normal(eta, e_eta, size=nsamples)

    M = a_TFR + b_TFR * eta + np.where(eta > 0, c_TFR * eta**2, 0)
    if zeropoint_dipole_mag is not None:
        dM = zeropoint_dipole_mag * galactic_to_radec_cartesian(
            zeropoint_dipole_ell, zeropoint_dipole_b)
        M += np.sum(dM[None, :] * rhat, axis=1)

    mag_obs = gen.normal(
        M + r2distmod(r, h=h), np.sqrt(sigma_int**2 + e_mag**2))
    zobs = gen.normal(
        (1 + r2z(r, h=h)) * (1 + Vpec / SPEED_OF_LIGHT) - 1,
        sigma_v / SPEED_OF_LIGHT)

    if los_density.ndim == 2:
        los_density = los_density[None, ...]
        los_velocity = los_velocity[None, ...]

    data = {
        "RA": RA,
        "dec": dec,
        "zcmb": zobs,
        "mag": mag_obs,
        "e_mag": np.ones_like(mag_obs) * e_mag,
        "eta": eta_obs,
        "e_eta": np.ones_like(eta_obs) * e_eta,
        "los_r": r_grid,
        "los_density": los_density,
        "los_velocity": los_velocity,
        }

    if zcmb_max is not None:
        mask = data["zcmb"] < zcmb_max
        fprint(f"Rejecting {np.sum(~mask)} samples with zcmb > {zcmb_max:.2f}",
               verbose=verbose)
        for key in data:
            if key in ["los_r"]:
                continue

            if key.startswith("los_"):
                data[key] = data[key][:, mask, ...]
            else:
                data[key] = data[key][mask]

    return data
