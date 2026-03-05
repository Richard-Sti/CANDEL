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
"""Posterior predictive check for the EDD TRGB H0 model."""
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp, norm

from .util import (SPEED_OF_LIGHT, fprint, get_nested, load_config,
                   radec_to_cartesian)


###############################################################################
#                          Cosmography helpers                                #
###############################################################################


def _build_numpy_cosmography(Om0, r_max_Mpch=60, npoints=2000):
    """Build numpy-based distance-to-distmod and distance-to-redshift
    interpolators at H0=100."""
    cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
    z_grid = np.logspace(-8, np.log10(0.5), npoints)
    r_grid = cosmo.comoving_distance(z_grid).value  # Mpc/h
    mu_grid = cosmo.distmod(z_grid).value

    r2mu_100 = interp1d(r_grid, mu_grid, kind="cubic",
                        bounds_error=False, fill_value="extrapolate")
    r2z_100 = interp1d(r_grid, z_grid, kind="cubic",
                       bounds_error=False, fill_value="extrapolate")
    return r2mu_100, r2z_100


def _distmod(r, h, r2mu_100):
    """Distance modulus: mu(r, h) = mu_100(r*h) - 5*log10(h)."""
    return r2mu_100(r * h) - 5 * np.log10(h)


def _redshift(r, h, r2z_100):
    """Cosmological redshift: z(r, h) = z_100(r*h)."""
    return r2z_100(r * h)


###############################################################################
#                         LOS interpolation (numpy)                           #
###############################################################################


def _interp_los(r_query, los_r, los_values):
    """Vectorized linear interpolation on a uniform LOS grid.

    Parameters
    ----------
    r_query : (N,)
        Query distances.
    los_r : (n_steps,)
        Uniform radial grid.
    los_values : (N, n_steps)
        Per-galaxy LOS profiles (already indexed by galaxy).

    Returns
    -------
    (N,) interpolated values, clamped at boundaries.
    """
    r_min = los_r[0]
    dr = (los_r[-1] - los_r[0]) / (len(los_r) - 1)
    n_steps = len(los_r)

    idx_cont = np.clip((r_query - r_min) / dr, 0.0, n_steps - 1.0)
    idx_lo = np.floor(idx_cont).astype(np.intp).clip(0, n_steps - 2)
    t = idx_cont - idx_lo

    rows = np.arange(len(r_query))
    val_lo = los_values[rows, idx_lo]
    val_hi = los_values[rows, idx_lo + 1]
    return val_lo + t * (val_hi - val_lo)


###############################################################################
#                              Smooth clip                                    #
###############################################################################


def _smoothclip(x, tau=0.1):
    """Smooth zero-clipping matching the model's smoothclip_nr."""
    return 0.5 * (x + np.sqrt(x**2 + tau**2))


###############################################################################
#                         PPC generation                                      #
###############################################################################


def generate_trgb_ppc(samples, data, config, n_ppc=None, seed=42):
    """Generate posterior predictive samples for the EDD TRGB model.

    Parameters
    ----------
    samples : dict
        Posterior samples loaded from HDF5. Keys include H0, M_TRGB,
        sigma_int, sigma_v, Vext (Cartesian, shape (n_post, 3)),
        beta, b1, and optionally mag_lim_TRGB, mag_lim_TRGB_width, etc.
    data : dict
        Data dict from ``load_EDD_TRGB_from_config``.
    config : str or dict
        Path to config TOML or loaded config dict.
    n_ppc : int or None
        Number of PPC galaxies to generate. Default: n_post * n_hosts.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: mag_sim, cz_sim, mag_obs, cz_obs.
    """
    if isinstance(config, str):
        config = load_config(config, replace_los_prior=False)
    gen = np.random.default_rng(seed)

    # ---- Unpack posterior samples (flatten chains if needed) ----
    def _flat(x):
        if x.ndim > 1 and x.shape[-1] != 3:
            return x.reshape(-1)
        if x.ndim > 2:
            return x.reshape(-1, x.shape[-1])
        return x

    H0 = _flat(samples["H0"])
    M_TRGB = _flat(samples["M_TRGB"])
    sigma_int = _flat(samples["sigma_int"])
    sigma_v = _flat(samples["sigma_v"])
    Vext = _flat(samples["Vext"])   # (n_post, 3)
    beta = _flat(samples["beta"])
    b1 = _flat(samples["b1"])
    n_post = len(H0)

    # Selection parameters
    which_sel = get_nested(config, "model/which_selection", None)
    mag_lim_samples = _flat(samples["mag_lim_TRGB"]) \
        if "mag_lim_TRGB" in samples else None
    mag_width_samples = _flat(samples["mag_lim_TRGB_width"]) \
        if "mag_lim_TRGB_width" in samples else None
    cz_lim_samples = _flat(samples["cz_lim_selection"]) \
        if "cz_lim_selection" in samples else None
    cz_width_samples = _flat(samples["cz_lim_selection_width"]) \
        if "cz_lim_selection_width" in samples else None

    # Fixed selection thresholds from config (fallback)
    mag_lim_fixed = get_nested(config, "model/mag_lim_TRGB", None)
    mag_width_fixed = get_nested(config, "model/mag_lim_TRGB_width", None)
    cz_lim_fixed = get_nested(config, "model/cz_lim_selection", None)
    cz_width_fixed = get_nested(config, "model/cz_lim_selection_width", None)

    # ---- Data ----
    mag_obs = np.asarray(data["mag_obs"])
    cz_obs = np.asarray(data["czcmb"])
    e_mag_obs_all = np.asarray(data["e_mag_obs"])
    e_czcmb_all = np.asarray(data["e_czcmb"])
    n_hosts = len(mag_obs)

    if n_ppc is None:
        n_ppc = n_post * n_hosts

    # ---- Random LOS data ----
    use_reconstruction = get_nested(config, "model/use_reconstruction", False)
    has_rand_los = data.get("has_rand_los", False)

    if use_reconstruction and has_rand_los:
        # Shape: (n_fields, n_rand, n_steps)
        rand_density = np.asarray(data["rand_los_density"])
        rand_velocity = np.asarray(data["rand_los_velocity"])
        rand_los_r = np.asarray(data["rand_los_r"])
        rand_RA = np.asarray(data["rand_los_RA"])
        rand_dec = np.asarray(data["rand_los_dec"])
        rand_rhat = radec_to_cartesian(rand_RA, rand_dec)
        n_fields, n_rand, n_steps = rand_density.shape
    else:
        use_reconstruction = False
        n_rand = 0

    # ---- Cosmography (numpy, built once) ----
    Om = get_nested(config, "model/Om", 0.3)
    r2mu_100, r2z_100 = _build_numpy_cosmography(Om)

    # ---- Distance limits ----
    r_limits = get_nested(config, "model/r_limits_malmquist", [0.01, 150])
    r_min, r_max = r_limits[0], r_limits[1]

    # ---- Pre-compute max biased density for rejection sampling ----
    if use_reconstruction:
        b1_max = float(np.max(b1))
        delta_max = float(np.max(rand_density)) - 1
        rho_biased_max = _smoothclip(1 + b1_max * delta_max)
    else:
        rho_biased_max = 1.0

    # ---- Batch rejection sampling loop ----
    collected_mag = []
    collected_cz = []
    n_accepted = 0
    batch_size = max(int(2.0 * n_ppc), 1000)

    fprint(f"PPC: generating {n_ppc} galaxies "
           f"(n_post={n_post}, n_hosts={n_hosts})")

    while n_accepted < n_ppc:
        n_need = n_ppc - n_accepted
        batch = min(batch_size, max(n_need * 3, 1000))

        # 1. Draw posterior sample indices
        idx_post = gen.integers(0, n_post, batch)

        # 2. Draw distance from r^2 volume prior
        u = gen.random(batch)
        r = (r_min**3 + u * (r_max**3 - r_min**3))**(1.0 / 3)

        # Per-candidate posterior parameters
        h = H0[idx_post] / 100
        M = M_TRGB[idx_post]
        sint = sigma_int[idx_post]
        sv = sigma_v[idx_post]
        Vext_cand = Vext[idx_post]     # (batch, 3)
        bt = beta[idx_post]
        b1_cand = b1[idx_post]

        # 3. Galaxy bias rejection (if using reconstruction)
        if use_reconstruction:
            # Pick random LOS and random field realization
            idx_los = gen.integers(0, n_rand, batch)
            idx_field = gen.integers(0, n_fields, batch)

            # Interpolate density on selected LOS at r*h
            rh = r * h
            los_delta_batch = rand_density[idx_field, idx_los, :] - 1
            delta_at_r = _interp_los(rh, rand_los_r, los_delta_batch)

            weight = _smoothclip(1 + b1_cand * delta_at_r)
            accept_bias = gen.random(batch) < (weight / rho_biased_max)
        else:
            accept_bias = np.ones(batch, dtype=bool)
            idx_los = None
            idx_field = None

        if not np.any(accept_bias):
            continue

        # Apply bias mask
        r = r[accept_bias]
        h = h[accept_bias]
        M = M[accept_bias]
        sint = sint[accept_bias]
        sv = sv[accept_bias]
        Vext_cand = Vext_cand[accept_bias]
        bt = bt[accept_bias]
        n_batch = len(r)

        # 4. Compute observables
        # Distance modulus
        mu = _distmod(r, h, r2mu_100)

        # Measurement errors resampled from observed distribution
        e_mag = gen.choice(e_mag_obs_all, n_batch)
        e_cz = gen.choice(e_czcmb_all, n_batch)

        # Apparent magnitude
        sigma_mag = np.sqrt(e_mag**2 + sint**2)
        mag_sim = gen.normal(M + mu, sigma_mag)

        # Cosmological redshift and peculiar velocity
        z_cosmo = _redshift(r, h, r2z_100)

        if use_reconstruction:
            idx_los_acc = idx_los[accept_bias]
            idx_field_acc = idx_field[accept_bias]
            rh_acc = r * h

            los_vel_batch = rand_velocity[idx_field_acc, idx_los_acc, :]
            v_los = _interp_los(rh_acc, rand_los_r, los_vel_batch)
            rhat = rand_rhat[idx_los_acc]  # (n_batch, 3)

            Vext_rad = np.sum(Vext_cand * rhat, axis=1)
            Vpec = bt * v_los + Vext_rad
        else:
            # No reconstruction: random sky direction for Vext projection
            RA_rand = gen.uniform(0, 360, n_batch)
            dec_rand = np.rad2deg(np.arcsin(gen.uniform(-1, 1, n_batch)))
            rhat = radec_to_cartesian(RA_rand, dec_rand)
            Vext_rad = np.sum(Vext_cand * rhat, axis=1)
            Vpec = Vext_rad

        cz_pred = SPEED_OF_LIGHT * (
            (1 + z_cosmo) * (1 + Vpec / SPEED_OF_LIGHT) - 1)
        sigma_cz = np.sqrt(e_cz**2 + sv**2)
        cz_sim = gen.normal(cz_pred, sigma_cz)

        # 5. Selection cut
        if which_sel == "TRGB_magnitude":
            idx_post_acc = idx_post[accept_bias]
            if mag_lim_samples is not None:
                ml = mag_lim_samples[idx_post_acc]
            else:
                ml = mag_lim_fixed
            if mag_width_samples is not None:
                mw = mag_width_samples[idx_post_acc]
            else:
                mw = mag_width_fixed
            p_sel = norm.cdf((ml - mag_sim) / mw)
            accept_sel = gen.random(n_batch) < p_sel
        elif which_sel == "redshift":
            idx_post_acc = idx_post[accept_bias]
            if cz_lim_samples is not None:
                cl = cz_lim_samples[idx_post_acc]
            else:
                cl = cz_lim_fixed
            if cz_width_samples is not None:
                cw = cz_width_samples[idx_post_acc]
            else:
                cw = cz_width_fixed
            p_sel = norm.cdf((cl - cz_sim) / cw)
            accept_sel = gen.random(n_batch) < p_sel
        else:
            accept_sel = np.ones(n_batch, dtype=bool)

        collected_mag.append(mag_sim[accept_sel])
        collected_cz.append(cz_sim[accept_sel])
        n_accepted += int(np.sum(accept_sel))

    mag_sim = np.concatenate(collected_mag)[:n_ppc]
    cz_sim = np.concatenate(collected_cz)[:n_ppc]

    fprint(f"PPC: generated {n_ppc} simulated galaxies.")

    return {
        "mag_sim": mag_sim,
        "cz_sim": cz_sim,
        "mag_obs": mag_obs,
        "cz_obs": cz_obs,
    }


###############################################################################
#                            PPC plotting                                     #
###############################################################################


def plot_trgb_ppc(ppc, fname):
    """Plot 3-panel PPC comparison: mag histogram, cz histogram, scatter.

    Parameters
    ----------
    ppc : dict
        Output of ``generate_trgb_ppc``.
    fname : str
        Output filename for the figure.
    """
    import matplotlib.pyplot as plt

    mag_sim = ppc["mag_sim"]
    cz_sim = ppc["cz_sim"]
    mag_obs = ppc["mag_obs"]
    cz_obs = ppc["cz_obs"]

    ks_mag = ks_2samp(mag_obs, mag_sim)
    ks_cz = ks_2samp(cz_obs, cz_sim)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: magnitude histogram
    ax = axes[0]
    bins_mag = np.linspace(
        min(mag_obs.min(), mag_sim.min()) - 0.5,
        max(mag_obs.max(), mag_sim.max()) + 0.5,
        40)
    ax.hist(mag_sim, bins=bins_mag, density=True, alpha=0.5,
            color="C0", label="PPC")
    ax.hist(mag_obs, bins=bins_mag, density=True, histtype="step",
            color="k", linewidth=1.5, label="Observed")
    ax.set_xlabel(r"$m_{\rm TRGB}$ [mag]")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.text(0.05, 0.95, f"KS $p = {ks_mag.pvalue:.3f}$",
            transform=ax.transAxes, va="top", fontsize=8)

    # Panel 2: cz histogram
    ax = axes[1]
    bins_cz = np.linspace(
        min(cz_obs.min(), cz_sim.min()) - 200,
        max(cz_obs.max(), cz_sim.max()) + 200,
        40)
    ax.hist(cz_sim, bins=bins_cz, density=True, alpha=0.5,
            color="C0", label="PPC")
    ax.hist(cz_obs, bins=bins_cz, density=True, histtype="step",
            color="k", linewidth=1.5, label="Observed")
    ax.set_xlabel(r"$cz_{\rm CMB}$ [km/s]")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.text(0.05, 0.95, f"KS $p = {ks_cz.pvalue:.3f}$",
            transform=ax.transAxes, va="top", fontsize=8)

    # Panel 3: scatter plot
    ax = axes[2]
    ax.scatter(mag_sim, cz_sim, s=2, alpha=0.1, color="C0",
               label="PPC", rasterized=True)
    ax.scatter(mag_obs, cz_obs, s=15, color="k", zorder=5,
               label="Observed")
    ax.set_xlabel(r"$m_{\rm TRGB}$ [mag]")
    ax.set_ylabel(r"$cz_{\rm CMB}$ [km/s]")
    ax.legend(fontsize=8, markerscale=2)

    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    fprint(f"PPC plot saved to {fname}")
