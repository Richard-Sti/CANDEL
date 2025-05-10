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
from scipy.stats import norm

from ..util import (SPEED_OF_LIGHT, galactic_to_radec,
                    galactic_to_radec_cartesian, radec_to_cartesian, fprint)
from ..field import interpolate_los_density_velocity


def reject_sample_distance(mu_TFR, sigma_TFR, h, distmod2dist,
                           log_grad_distmod2distance, r_h_density_grid,
                           log_los_density_grid, num_sigma=5, n_points=501,
                           seed=42):
    """
    Sample a distance modulus from the TFR calibration via rejection sampling.
    """
    gen = np.random.default_rng(seed)

    delta = num_sigma * sigma_TFR
    mu_grid = np.linspace(mu_TFR - delta, mu_TFR + delta, n_points)

    mu_min, mu_max = mu_grid[0], mu_grid[-1]

    r_grid = distmod2dist(mu_grid, h=h)
    log_jac = log_grad_distmod2distance(mu_grid, h=h)
    log_rho = np.interp(r_grid * h, r_h_density_grid, log_los_density_grid)

    prob = (+ 2 * np.log(r_grid)
            + log_jac
            + log_rho
            + norm(mu_TFR, sigma_TFR).logpdf(mu_grid)
            )
    prob -= np.max(prob)
    prob = np.exp(prob)

    while True:
        x = gen.uniform(mu_min, mu_max)
        p_i = np.interp(x, mu_grid, prob)

        if gen.uniform() < p_i:
            return x


def sample_magnitude(nsamples, mag_min, mag_max, gen):
    """
    Sample magnitude from a distribution that is proportional to 10^{0.6 mag}.
    """
    z = gen.uniform(0, 1, size=nsamples)

    ymin = 10**(3 * mag_min / 5)
    ymax = 10**(3 * mag_max / 5)

    return 5 / 3 * np.log10(z * (ymax - ymin) + ymin)


def gen_CF4_TFR_mock(nsamples, Vext_mag, Vext_ell, Vext_b, sigma_v, alpha,
                     beta, a_TFR, b_TFR, c_TFR, sigma_TFR, a_TFR_dipole_mag,
                     a_TFR_dipole_ell, a_TFR_dipole_b, h, mag, eta, mag_min,
                     mag_max, e_mag, eta_mean, eta_std, e_eta,
                     b_min, zcmb_max, r_h_max, distmod2dist, distmod2redshift,
                     log_grad_distmod2dist, field_loader, use_data_prior,
                     rmin_reconstruction, rmax_reconstruction,
                     num_steps_reconstruction, seed=42):
    """
    Generate a mock sample of galaxies with TFR distances that resembles
    the CF4 TFR sample.
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

    if use_data_prior:
        ks = np.random.choice(len(mag), size=nsamples, replace=True)
        mag_true = mag[ks]
        eta_true = eta[ks]
    else:
        mag_true = sample_magnitude(nsamples, mag_min, mag_max, gen)
        eta_true = gen.normal(eta_mean, eta_std, size=nsamples)

    mag_obs = gen.normal(mag_true, e_mag, size=nsamples)
    eta_obs = gen.normal(eta_true, e_eta, size=nsamples)

    r_los = np.linspace(
        rmin_reconstruction, rmax_reconstruction, num_steps_reconstruction)

    if field_loader is None:
        log_los_density_grid = np.ones(nsamples)
        los_velocity_grid = np.zeros(nsamples)
        los_density_precomp = np.ones((nsamples, len(r_los)))
        los_velocity_precomp = np.zeros((nsamples, len(r_los)))
    else:
        r_h_grid = np.arange(0, r_h_max + 0.1, 0.1)
        log_los_density_grid, los_velocity_grid = interpolate_los_density_velocity(  # noqa
            field_loader, r_h_grid, RA, dec)
        log_los_density_grid = alpha * np.log(log_los_density_grid)
        los_velocity_grid *= beta

        los_density_precomp, los_velocity_precomp = interpolate_los_density_velocity(  # noqa
            field_loader, r_los, RA, dec)

    if a_TFR_dipole_mag is not None:
        a_TFR_dipole = a_TFR_dipole_mag * galactic_to_radec_cartesian(
            a_TFR_dipole_ell, a_TFR_dipole_b)
        a_TFR += np.sum(a_TFR_dipole[None, :] * rhat, axis=1)

    mu_TFR = mag_true - (a_TFR + b_TFR * eta_true + np.where(
        eta_true > 0, c_TFR * eta_true**2, 0))
    # Reject sample the true distance from the TFR estimates.
    mu = np.zeros_like(mu_TFR)
    for i in range(len(mu_TFR)):
        mu[i] = reject_sample_distance(
            mu_TFR[i], sigma_TFR, h, distmod2dist,
            log_grad_distmod2dist, r_h_grid, log_los_density_grid[i],
            seed=seed + i)

    r_h = distmod2dist(mu,)
    los_velocity = np.full(nsamples, np.nan)
    for i in range(nsamples):
        # Interpolate the velocity field at the position of the galaxy
        los_velocity[i] = np.interp(r_h[i], r_h_grid, los_velocity_grid[i])

    zcosmo = distmod2redshift(mu, h=h)

    Vext = Vext_mag * galactic_to_radec_cartesian(Vext_ell, Vext_b)
    Vext_rad = np.sum(Vext[None, :] * rhat, axis=1)

    zpec = (los_velocity + Vext_rad) / SPEED_OF_LIGHT
    sigma_cz = sigma_v / SPEED_OF_LIGHT
    zcmb_true = (1 + zcosmo) * (1 + zpec) - 1
    zcmb = gen.normal(zcmb_true, sigma_cz, size=nsamples)

    data = {
        "RA": RA,
        "dec": dec,
        "zcmb": zcmb,
        "mag": mag_obs,
        "e_mag": np.ones_like(mag_obs) * e_mag,
        "eta": eta_obs,
        "e_eta": np.ones_like(eta_obs) * e_eta,
        "los_r": r_los,
        "los_density": los_density_precomp,
        "los_velocity": los_velocity_precomp,
        }

    if zcmb_max is not None:
        mask = zcmb < zcmb_max
        fprint(f"Rejecting {np.sum(~mask)} samples with zcmb > {zcmb_max:.2f}")
        for key in data:
            if key in ["los_r"]:
                continue
            data[key] = data[key][mask]

    return data
