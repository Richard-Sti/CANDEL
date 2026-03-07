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
"""Mock generator for combined TRGB + 2MTF surveys (two-group model).

Generates two independent galaxy samples:
  - TRGB group: TRGB-selected hosts, a fraction also has TFR observables.
  - TFR-only group: 2MTF hosts not in the TRGB sample, selected by
    TFR magnitude + linewidth cuts.
"""
import numpy as np
from scipy.stats import norm

from ...cosmography import Distance2Distmod, Distance2Redshift
from ...util import (SPEED_OF_LIGHT, galactic_to_radec_cartesian,
                     radec_to_cartesian)

DEFAULT_TRUE_PARAMS = {
    # Shared
    "H0": 73.0,
    "sigma_v": 300.0,
    "Vext_mag": 150.0,
    "Vext_ell": 270.0,
    "Vext_b": 30.0,
    "beta": 0.43,
    "b1": 1.2,
    # TRGB
    "M_TRGB": -4.05,
    "sigma_int_TRGB": 0.1,
    # TFR
    "a_TFR": -21.0,
    "b_TFR": -8.0,
    "c_TFR": 0.0,
    "sigma_int_TFR": 0.4,
    "eta_mean": 0.0,
    "eta_std": 0.08,
}

DEFAULT_ANCHORS = {
    "mu_LMC": 18.477,
    "e_mu_LMC": 0.026,
    "e_mag_LMC_TRGB": 0.018,
    "mu_N4258": 29.398,
    "e_mu_N4258": 0.032,
    "e_mag_N4258_TRGB": 0.0443,
}


def _get_absmag_TFR(eta, a, b, c=0.0):
    return a + b * eta + np.where(eta > 0, c * eta**2, 0.0)


def _apply_TRGB_selection(mag_obs_TRGB, mag_lim, mag_lim_width, gen):
    n = len(mag_obs_TRGB)
    if mag_lim is not None:
        p_sel = norm.cdf((mag_lim - mag_obs_TRGB) / mag_lim_width)
        return gen.random(n) < p_sel
    return np.ones(n, dtype=bool)


def _apply_TFR_selection(mag_obs_TFR, eta_obs, mag_lim, mag_lim_width,
                         eta_min, eta_max, gen):
    n = len(mag_obs_TFR)
    mask = np.ones(n, dtype=bool)
    if mag_lim is not None:
        if mag_lim_width is not None:
            p_sel = norm.cdf((mag_lim - mag_obs_TFR) / mag_lim_width)
            mask &= gen.random(n) < p_sel
        else:
            mask &= mag_obs_TFR < mag_lim
    if eta_min is not None:
        mask &= eta_obs >= eta_min
    if eta_max is not None:
        mask &= eta_obs <= eta_max
    return mask


def _gen_parent_galaxies(n, rmin, rmax, gen):
    """Draw parent galaxies uniform in volume."""
    RA = gen.uniform(0, 360, n)
    dec = np.rad2deg(np.arcsin(gen.uniform(-1, 1, n)))
    u = gen.random(n)
    r = (rmin**3 + u * (rmax**3 - rmin**3))**(1 / 3)
    return RA, dec, r


def _observe_galaxies(RA, dec, r, h, M_TRGB, sigma_int_TRGB,
                      a_TFR, b_TFR, c_TFR, sigma_int_TFR,
                      eta_mean, eta_std, sigma_v, Vext,
                      e_mag_TRGB, e_mag_TFR, e_eta, e_czcmb,
                      r2mu, r2z, gen, observe_trgb=True,
                      observe_tfr=True):
    """Generate observables for parent galaxies."""
    rhat = radec_to_cartesian(RA, dec)
    mu = np.asarray(r2mu(r, h=h))
    z_cosmo = np.asarray(r2z(r, h=h))
    Vext_rad = rhat @ Vext
    n = len(r)

    cz_true = SPEED_OF_LIGHT * (
        (1 + z_cosmo) * (1 + Vext_rad / SPEED_OF_LIGHT) - 1)
    cz_obs = gen.normal(cz_true, np.sqrt(e_czcmb**2 + sigma_v**2))

    out = {"RA": RA, "dec": dec, "r": r, "cz_obs": cz_obs}

    if observe_trgb:
        sigma_TRGB_tot = np.sqrt(sigma_int_TRGB**2 + e_mag_TRGB**2)
        out["mag_obs_TRGB"] = gen.normal(M_TRGB + mu, sigma_TRGB_tot)

    if observe_tfr:
        eta_true = gen.normal(eta_mean, eta_std, n)
        M_true_TFR = _get_absmag_TFR(eta_true, a_TFR, b_TFR, c_TFR)
        sigma_TFR_tot = np.sqrt(sigma_int_TFR**2 + e_mag_TFR**2)
        out["mag_obs_TFR"] = gen.normal(M_true_TFR + mu, sigma_TFR_tot)
        out["eta"] = gen.normal(eta_true, e_eta)

    return out


def gen_TRGB_2MTF_mock(n_trgb=300, n_tfr=2000, overlap_fraction=0.1,
                       Om=0.3,
                       e_mag_TRGB=0.05, e_mag_TFR=0.04,
                       e_eta=0.01, e_czcmb=10.0,
                       rmin_trgb=0.5, rmax_trgb=40.0,
                       rmin_tfr=0.5, rmax_tfr=200.0,
                       mag_lim_TRGB=25.0, mag_lim_TRGB_width=0.75,
                       mag_lim_TFR=11.25, mag_lim_TFR_width=None,
                       eta_min_sel=None, eta_max_sel=None,
                       true_params=None, anchors=None,
                       noisy_anchors=True, seed=42, verbose=True):
    """Generate a mock two-group TRGB + 2MTF survey.

    WARNING: the overlap between TRGB and TFR is assigned by randomly
    splitting the TRGB sample — a fixed fraction ``overlap_fraction``
    of TRGB hosts receive TFR observables drawn from the same parent
    distribution. No physical criterion (e.g. morphology, K-band
    brightness, or measurable linewidth) distinguishes overlap from
    TRGB-only hosts. This is a simplification; in real data the overlap
    is determined by additional selection effects not modeled here.

    Returns
    -------
    data_trgb : dict
        TRGB group data dict.
    data_tfr : dict
        TFR-only group data dict.
    true_params : dict
        True parameter values.
    n_parent : dict
        Parent population sizes for each group.
    """
    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
    anch = {**DEFAULT_ANCHORS, **(anchors or {})}
    gen = np.random.default_rng(seed)

    H0 = tp["H0"]
    h = H0 / 100
    M_TRGB = tp["M_TRGB"]
    sigma_int_TRGB = tp["sigma_int_TRGB"]
    a_TFR = tp["a_TFR"]
    b_TFR = tp["b_TFR"]
    c_TFR = tp["c_TFR"]
    sigma_int_TFR = tp["sigma_int_TFR"]
    eta_mean = tp["eta_mean"]
    eta_std = tp["eta_std"]
    sigma_v = tp["sigma_v"]
    Vext = tp["Vext_mag"] * galactic_to_radec_cartesian(
        tp["Vext_ell"], tp["Vext_b"])

    r2mu = Distance2Distmod(Om0=Om)
    r2z = Distance2Redshift(Om0=Om)

    obs_kwargs = dict(
        h=h, M_TRGB=M_TRGB, sigma_int_TRGB=sigma_int_TRGB,
        a_TFR=a_TFR, b_TFR=b_TFR, c_TFR=c_TFR,
        sigma_int_TFR=sigma_int_TFR,
        eta_mean=eta_mean, eta_std=eta_std, sigma_v=sigma_v, Vext=Vext,
        e_mag_TRGB=e_mag_TRGB, e_mag_TFR=e_mag_TFR,
        e_eta=e_eta, e_czcmb=e_czcmb,
        r2mu=r2mu, r2z=r2z,
    )

    # =================================================================
    #  TRGB group
    # =================================================================
    n_overlap = max(1, int(n_trgb * overlap_fraction))
    n_trgb_only = n_trgb - n_overlap

    # TRGB-only hosts (no TFR data)
    trgb_only = _gen_trgb_hosts(
        n_trgb_only, rmin_trgb, rmax_trgb,
        mag_lim_TRGB, mag_lim_TRGB_width,
        observe_tfr=False, verbose=verbose,
        label="TRGB-only", gen=gen, **obs_kwargs)

    # Overlap hosts (both TRGB and TFR)
    overlap = _gen_trgb_hosts(
        n_overlap, rmin_trgb, rmax_trgb,
        mag_lim_TRGB, mag_lim_TRGB_width,
        observe_tfr=True, verbose=verbose,
        label="overlap", gen=gen, **obs_kwargs)

    # Build TRGB data dict
    n_trgb_actual = len(trgb_only["RA"]) + len(overlap["RA"])
    n_overlap_actual = len(overlap["RA"])
    n_trgb_only_actual = len(trgb_only["RA"])

    data_trgb = _build_trgb_data(
        trgb_only, overlap, e_mag_TRGB, e_mag_TFR, e_eta, e_czcmb,
        anch, M_TRGB, gen, noisy_anchors)

    if verbose:
        print(f"TRGB group: {n_trgb_actual} hosts "
              f"({n_trgb_only_actual} TRGB-only, "
              f"{n_overlap_actual} overlap)")

    # =================================================================
    #  TFR-only group
    # =================================================================
    data_tfr_out, n_parent_tfr = _gen_tfr_only_hosts(
        n_tfr, rmin_tfr, rmax_tfr,
        mag_lim_TFR, mag_lim_TFR_width,
        eta_min_sel, eta_max_sel,
        verbose=verbose, gen=gen, **obs_kwargs)

    if verbose:
        print(f"TFR-only group: {len(data_tfr_out['mag'])} hosts")

    # True Cartesian Vext
    tp["Vext_x"], tp["Vext_y"], tp["Vext_z"] = Vext

    n_parent = {
        "trgb": trgb_only.get("n_parent", 0) + overlap.get("n_parent", 0),
        "tfr": n_parent_tfr,
    }

    return data_trgb, data_tfr_out, tp, n_parent


def _gen_trgb_hosts(nsamples, rmin, rmax, mag_lim, mag_lim_width,
                    observe_tfr, verbose, label, gen, **obs_kwargs):
    """Generate TRGB-selected hosts."""
    collected = {}
    keys = ["RA", "dec", "r", "mag_obs_TRGB", "cz_obs"]
    if observe_tfr:
        keys += ["mag_obs_TFR", "eta"]
    for k in keys:
        collected[k] = []

    n_accepted = 0
    n_parent = 0
    batch = max(int(1.5 * nsamples), 100)

    while n_accepted < nsamples:
        RA, dec, r = _gen_parent_galaxies(batch, rmin, rmax, gen)
        obs = _observe_galaxies(
            RA, dec, r, gen=gen, observe_trgb=True, observe_tfr=observe_tfr,
            **obs_kwargs)

        mask = _apply_TRGB_selection(
            obs["mag_obs_TRGB"], mag_lim, mag_lim_width, gen)

        n_parent += batch
        n_accepted += mask.sum()

        for k in keys:
            collected[k].append(obs[k][mask])

    for k in keys:
        collected[k] = np.concatenate(collected[k])[:nsamples]

    collected["n_parent"] = n_parent

    if verbose:
        print(f"  {label}: {nsamples} hosts "
              f"(acceptance {nsamples / n_parent:.3f})")

    return collected


def _gen_tfr_only_hosts(n_tfr, rmin, rmax, mag_lim, mag_lim_width,
                        eta_min, eta_max,
                        verbose, gen, **obs_kwargs):
    """Generate TFR-selected hosts (no TRGB)."""
    keys = ["RA", "dec", "r", "mag_obs_TFR", "eta", "cz_obs"]
    collected = {k: [] for k in keys}

    n_accepted = 0
    n_parent = 0
    batch = max(int(1.5 * n_tfr), 100)

    while n_accepted < n_tfr:
        RA, dec, r = _gen_parent_galaxies(batch, rmin, rmax, gen)
        obs = _observe_galaxies(
            RA, dec, r, gen=gen, observe_trgb=False, observe_tfr=True,
            **obs_kwargs)

        mask = _apply_TFR_selection(
            obs["mag_obs_TFR"], obs["eta"],
            mag_lim, mag_lim_width, eta_min, eta_max, gen)

        n_parent += batch
        n_accepted += mask.sum()

        for k in keys:
            collected[k].append(obs[k][mask])

    for k in keys:
        collected[k] = np.concatenate(collected[k])[:n_tfr]

    if verbose:
        print(f"  TFR-only: {n_tfr} hosts "
              f"(acceptance {n_tfr / n_parent:.3f})")

    n_kept = len(collected["RA"])
    e_mag_TFR = obs_kwargs["e_mag_TFR"]
    e_eta = obs_kwargs["e_eta"]
    e_czcmb = obs_kwargs["e_czcmb"]
    data_tfr = {
        "RA_host": collected["RA"],
        "dec_host": collected["dec"],
        "mag": collected["mag_obs_TFR"],
        "e_mag": np.full(n_kept, e_mag_TFR),
        "e_mag_median": float(e_mag_TFR),
        "eta": collected["eta"],
        "e_eta": np.full(n_kept, e_eta),
        "e_eta_median": float(e_eta),
        "czcmb": collected["cz_obs"],
        "e_czcmb": np.full(n_kept, e_czcmb),
    }

    return data_tfr, n_parent


def _build_trgb_data(trgb_only, overlap, e_mag_TRGB, e_mag_TFR,
                     e_eta, e_czcmb, anch, M_TRGB, gen, noisy_anchors):
    """Build the TRGB group data dict from TRGB-only + overlap hosts."""
    n1 = len(trgb_only["RA"])
    n2 = len(overlap["RA"])
    n_total = n1 + n2

    def _cat(key):
        a = trgb_only[key] if key in trgb_only else None
        b = overlap[key] if key in overlap else None
        if a is not None and b is not None:
            return np.concatenate([a, b])
        elif a is not None:
            return a
        else:
            return b

    # has_TFR mask: first n1 are TRGB-only, last n2 are overlap
    has_TFR = np.zeros(n_total, dtype=bool)
    has_TFR[n1:] = True

    # TFR arrays: padded with zeros for TRGB-only hosts
    mag_TFR = np.zeros(n_total)
    eta_arr = np.zeros(n_total)
    if n2 > 0:
        mag_TFR[n1:] = overlap["mag_obs_TFR"]
        eta_arr[n1:] = overlap["eta"]

    # Anchors
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

    data = {
        "RA_host": _cat("RA"),
        "dec_host": _cat("dec"),
        # TRGB observables
        "mag_obs": _cat("mag_obs_TRGB"),
        "e_mag_obs": np.full(n_total, e_mag_TRGB),
        "e_mag_obs_median": float(e_mag_TRGB),
        # TFR observables (overlap only; padded for TRGB-only)
        "mag_TFR": mag_TFR,
        "e_mag_TFR": np.full(n_total, e_mag_TFR),
        "eta_trgb": eta_arr,
        "e_eta_trgb": np.full(n_total, e_eta),
        "has_TFR": has_TFR,
        # Redshift
        "czcmb": _cat("cz_obs"),
        "e_czcmb": np.full(n_total, e_czcmb),
        # Anchors
        "mu_LMC_anchor": mu_LMC_obs,
        "e_mu_LMC_anchor": anch["e_mu_LMC"],
        "mag_LMC_TRGB": mag_LMC_obs,
        "e_mag_LMC_TRGB": anch["e_mag_LMC_TRGB"],
        "mu_N4258_anchor": mu_N4258_obs,
        "e_mu_N4258_anchor": anch["e_mu_N4258"],
        "mag_N4258_TRGB": mag_N4258_obs,
        "e_mag_N4258_TRGB": anch["e_mag_N4258_TRGB"],
    }

    return data
