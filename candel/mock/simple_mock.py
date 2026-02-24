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
"""Simple peculiar velocity catalog mock generator."""
import numpy as np
from scipy.stats import norm


def rk_draw(n, rmin, rmax, k, rng):
    """Draw distances with prior p(r) ∝ r^k on [rmin, rmax]."""
    assert k > -1, "k must be > -1 for normalizable prior"
    u = rng.random(n)
    return (rmin**(k+1) + u * (rmax**(k+1) - rmin**(k+1)))**(1.0 / (k+1))


def _rejection_sample(n, rmin, rmax_sample, k, rng, gen_observables,
                      accept_fn, verbose=True):
    """Batch rejection-sample distances and observables.

    Parameters
    ----------
    gen_observables : callable(r_true, rng, batch_size) -> (obs, ...)
        Returns a tuple of arrays; the first is used for the acceptance test.
    accept_fn : callable(obs_first) -> bool mask
        Returns a boolean mask of accepted samples.
    """
    collected_r = []
    collected_obs = []
    batch_size = max(int(0.3 * n), 1)
    nsampled = 0
    niter = 0

    while nsampled < n:
        r_i = rk_draw(batch_size, rmin, rmax_sample, k, rng)
        obs_tuple = gen_observables(r_i, rng, batch_size)
        mask = accept_fn(obs_tuple[0])

        collected_r.append(r_i[mask])
        collected_obs.append(tuple(o[mask] for o in obs_tuple))
        nsampled += mask.sum()
        niter += 1

    r_out = np.concatenate(collected_r)[:n]
    obs_out = tuple(
        np.concatenate([c[j] for c in collected_obs])[:n]
        for j in range(len(collected_obs[0]))
    )
    if verbose:
        print(f"It took {niter} iterations to get {len(r_out)} objects.")
    return r_out, obs_out


def gen_simple_catalog(n=1000, rmin=5.0, rmax=80.0, rmax_sel=None,
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

        def gen_obs(r_i, rng, bs):
            mu_obs_i = rng.normal(5.0 * np.log10(r_i) + 25.0, sigma_mu, bs)
            return (mu_obs_i,)

        r_true, (mu_obs,) = _rejection_sample(
            n, rmin, rmax_sample, k, rng, gen_obs,
            lambda mu: mu < mu_max, verbose=verbose)
        v_obs = rng.normal(H0_true * r_true, sigma_vpec, n)

    elif czmax_sel is not None:
        if czmax_sel_width is None:
            rmax_sample = czmax_sel / H0_true + 6 * sigma_vpec / H0_true
        else:
            rmax_sample = (czmax_sel + 6 * czmax_sel_width) / H0_true
            rmax_sample += 6 * sigma_vpec / H0_true

        def gen_obs(r_i, rng, bs):
            v_obs_i = rng.normal(H0_true * r_i, sigma_vpec, bs)
            return (v_obs_i,)

        if czmax_sel_width is None:
            accept = lambda v: v < czmax_sel
        else:
            def accept(v):
                p_sel = norm.cdf((czmax_sel - v) / czmax_sel_width)
                return rng.random(len(v)) < p_sel

        r_true, (v_obs,) = _rejection_sample(
            n, rmin, rmax_sample, k, rng, gen_obs,
            accept, verbose=verbose)
        mu_obs = rng.normal(5.0 * np.log10(r_true) + 25.0, sigma_mu)

    else:
        r_true = rk_draw(n, rmin, rmax, k, rng)
        mu_obs = rng.normal(5.0 * np.log10(r_true) + 25.0, sigma_mu, n)
        v_obs = rng.normal(H0_true * r_true, sigma_vpec, n)

    return {"cz": v_obs, "mag": mu_obs, "r_true": r_true}
