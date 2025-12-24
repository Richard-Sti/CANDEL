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
from scipy.stats import norm

from ..field import interpolate_los_density_velocity
from ..util import (SPEED_OF_LIGHT, fprint, galactic_to_radec,
                    galactic_to_radec_cartesian, radec_to_cartesian)

C_LIGHT = 299792.458  # km/s
LN10 = np.log(10.0)


def sample_distance(r_grid, los_density, b1, R, p, n, gen):
    """Sample distance from p(r) ∝ (1 + b1*δ) r^p exp(-(r/R)^n)."""
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


def rk_draw(n, rmin, rmax, k, rng):
    """Draw distances with prior p(r) ∝ r^k on [rmin, rmax]."""
    assert k > -1, "k must be > -1 for normalizable prior"
    u = rng.random(n)
    return (rmin**(k+1) + u * (rmax**(k+1) - rmin**(k+1)))**(1.0 / (k+1))


def simulate_simple_catalog(n=1000, rmin=5.0, rmax=80.0, rmax_sel=None,
                            czmax_sel=None, czmax_sel_width=None,
                            H0_true=73.0, sigma_mu=0.4, sigma_vpec=300.0,
                            seed=12345, k=2, verbose=True):
    """
    Simulate a simple peculiar velocity catalog with TF-like distance modulus
    measurements and Gaussian velocity scatter.

    Generates true distances from a power-law prior p(r) ∝ r^k, adds Gaussian
    scatter to the distance modulus and radial velocities, and optionally
    applies selection cuts in distance modulus or redshift. Assumes M = 0,
    i.e. mu = 5 log10(r) + 25.

    Selection can be either:
        - Hard truncation: cz_obs < czmax_sel (when czmax_sel_width is None)
        - Sigmoid: p(select) = Φ((czmax_sel - cz_obs) / czmax_sel_width)

    Note: distances are labeled Mpc/h but the distance modulus formula assumes
    Mpc. The h-dependent offset is absorbed into the implicit M = 0.

    Parameters
    ----------
    n : int
        Number of objects to simulate.
    rmin, rmax : float
        Minimum and maximum true distances [Mpc/h] for the power-law prior.
    rmax_sel : float, optional
        If set, apply a selection cut mu_obs < 5 log10(rmax_sel) + 25.
        Objects are drawn from an extended range and rejected if above cut.
    czmax_sel : float, optional
        Transition point for cz selection [km/s]. For hard cut, this is the
        threshold. For sigmoid, this is where p(select) = 0.5.
    czmax_sel_width : float, optional
        Width of the sigmoid transition [km/s]. If None, use hard threshold.
        Smaller values give sharper transitions.
    H0_true : float
        True Hubble constant [km/s/Mpc] used to generate velocities.
    sigma_mu : float
        Gaussian scatter in distance modulus [mag].
    sigma_vpec : float
        Gaussian scatter in peculiar velocity [km/s].
    seed : int
        Random seed for reproducibility.
    k : float
        Power-law index for the distance prior. Must be > -1.
    verbose : bool
        If True, print iteration count when using selection cuts.

    Returns
    -------
    dict
        Dictionary with keys:
        - "cz": Observed radial velocities [km/s], shape (n,).
        - "mag": Observed distance moduli [mag], shape (n,).
        - "r_true": True distances [Mpc/h], shape (n,).
    """
    rng = np.random.default_rng(seed)

    if rmax_sel is not None:
        mu_max = 5.0 * np.log10(rmax_sel) + 25.0
        mu_max_sample = mu_max + 6 * sigma_mu
        rmax_sample = 10**((mu_max_sample - 25.0) / 5.0)

        r_true = []
        mu_obs = []

        batch_size = int(0.3 * n)

        i = 0
        nsampled = 0
        while nsampled < n:
            r_true_i = rk_draw(batch_size, rmin, rmax_sample, k, rng)
            mu_true_i = 5.0 * np.log10(r_true_i) + 25.0
            mu_obs_i = rng.normal(mu_true_i, sigma_mu, batch_size)
            mask = mu_obs_i < mu_max
            r_true.append(r_true_i[mask])
            mu_obs.append(mu_obs_i[mask])

            nsampled += mask.sum()
            i += 1

        r_true = np.concatenate(r_true)[:n]
        mu_obs = np.concatenate(mu_obs)[:n]
        if verbose:
            print(f"It took {i} iterations to get {len(r_true)} objects.")
        v_flow = H0_true * r_true
        v_obs = rng.normal(v_flow, sigma_vpec, n)  # observed Vcmb
    elif czmax_sel is not None:
        # Extend sampling range to account for scatter
        if czmax_sel_width is None:
            rmax_sample = czmax_sel / H0_true + 6 * sigma_vpec / H0_true
        else:
            # For sigmoid, extend further to capture tail
            rmax_sample = (czmax_sel + 6 * czmax_sel_width) / H0_true
            rmax_sample += 6 * sigma_vpec / H0_true

        r_true = []
        cz_obs = []

        batch_size = int(0.3 * n)

        i = 0
        nsampled = 0
        while nsampled < n:
            r_true_i = rk_draw(batch_size, rmin, rmax_sample, k, rng)
            v_flow_i = H0_true * r_true_i
            v_obs_i = rng.normal(v_flow_i, sigma_vpec, batch_size)

            if czmax_sel_width is None:
                # Hard threshold
                mask = v_obs_i < czmax_sel
            else:
                # Sigmoid selection: p(select) = Φ((czmax - cz) / width)
                p_select = norm.cdf((czmax_sel - v_obs_i) / czmax_sel_width)
                mask = rng.random(batch_size) < p_select

            r_true.append(r_true_i[mask])
            cz_obs.append(v_obs_i[mask])

            nsampled += mask.sum()
            i += 1

        r_true = np.concatenate(r_true)[:n]
        v_obs = np.concatenate(cz_obs)[:n]
        if verbose:
            print(f"It took {i} iterations to get {len(r_true)} objects.")
        mu_obs = rng.normal(5.0 * np.log10(r_true) + 25.0, sigma_mu)

    else:
        r_true = rk_draw(n, rmin, rmax, k, rng)
        mu_true = 5.0 * np.log10(r_true) + 25.0
        mu_obs = rng.normal(mu_true, sigma_mu, n)

        v_flow = H0_true * r_true
        v_obs = rng.normal(v_flow, sigma_vpec, n)  # observed Vcmb

    return {"cz": v_obs, "mag": mu_obs, "r_true": r_true}
