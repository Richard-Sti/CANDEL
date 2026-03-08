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
from scipy.stats import ks_2samp, norm

from ..cosmography import Distance2Distmod, Distance2Redshift
from ..field import name2field_loader
from ..util import (SPEED_OF_LIGHT, fprint, get_nested, load_config,
                    radec_to_cartesian)
from ._field_utils import (build_field_pool, compute_r_max_selection,
                            smoothclip)


###############################################################################
#                         PPC generation                                      #
###############################################################################


def generate_trgb_ppc(samples, data, config, n_ppc=None, seed=42):
    """Generate posterior predictive samples for the EDD TRGB model.

    When a reconstruction field is available, galaxies are sampled from
    the full 3D density field (matching the mock generator). A large pool
    of 3D positions is pre-evaluated once, and the rejection-sampling loop
    operates on cached density/velocity values for efficiency.

    Parameters
    ----------
    samples : dict
        Posterior samples loaded from HDF5.
    data : dict
        Data dict from ``load_EDD_TRGB_from_config``.
    config : str or dict
        Path to config TOML or loaded config dict.
    n_ppc : int or None
        Number of PPC galaxies to generate.
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
    n_post = len(H0)
    if "Vext" in samples:
        Vext = _flat(samples["Vext"])
    else:
        Vext = np.zeros((n_post, 3))
    beta = _flat(samples.get("beta", np.zeros(n_post)))
    b1 = _flat(samples.get("b1", np.ones(n_post)))

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
        ppc_factor = get_nested(config, "model/ppc_factor", 10)
        n_ppc = ppc_factor * n_hosts

    # ---- Cosmography ----
    Om = get_nested(config, "model/Om", 0.3)
    r2mu = Distance2Distmod(Om0=Om)
    r2z = Distance2Redshift(Om0=Om)

    # ---- Distance limits ----
    r_limits = get_nested(config, "model/r_limits_malmquist", [0.01, 150])
    r_min, r_max = r_limits[0], r_limits[1]

    # ---- Effective sphere radius (selection-aware) ----
    r_max_eff = compute_r_max_selection(
        mag_lim=mag_lim_samples if mag_lim_samples is not None else mag_lim_fixed,  # noqa
        M_abs=M_TRGB, sigma_int=sigma_int, e_mag=e_mag_obs_all,
        mag_lim_width=mag_width_samples if mag_width_samples is not None else mag_width_fixed,  # noqa
        cz_lim=cz_lim_samples if cz_lim_samples is not None else cz_lim_fixed,
        h=H0 / 100, r_max=r_max)
    fprint(f"PPC: effective r_max = {r_max_eff:.1f} Mpc "
           f"(config r_max = {r_max:.1f} Mpc)")

    # ---- Check for reconstruction field ----
    use_reconstruction = get_nested(config, "model/use_reconstruction", False)

    if use_reconstruction:
        mag_sim, cz_sim = _ppc_field_path(
            gen, config, H0, M_TRGB, sigma_int, sigma_v, Vext, beta, b1,
            which_sel, mag_lim_samples, mag_lim_fixed,
            mag_width_samples, mag_width_fixed,
            cz_lim_samples, cz_lim_fixed, cz_width_samples, cz_width_fixed,
            e_mag_obs_all, e_czcmb_all,
            r_min, r_max_eff, r2mu, r2z, n_ppc, n_hosts)
    else:
        mag_sim, cz_sim = _ppc_homogeneous_path(
            gen, H0, M_TRGB, sigma_int, sigma_v, Vext, beta,
            which_sel, mag_lim_samples, mag_lim_fixed,
            mag_width_samples, mag_width_fixed,
            cz_lim_samples, cz_lim_fixed, cz_width_samples, cz_width_fixed,
            e_mag_obs_all, e_czcmb_all,
            r_min, r_max_eff, r2mu, r2z, n_ppc, n_hosts)

    return {
        "mag_sim": mag_sim,
        "cz_sim": cz_sim,
        "mag_obs": mag_obs,
        "cz_obs": cz_obs,
    }


###############################################################################
#                    Field-based PPC path (3D density)                        #
###############################################################################


def _ppc_field_path(gen, config, H0, M_TRGB, sigma_int, sigma_v, Vext,
                    beta, b1,
                    which_sel, mag_lim_samples, mag_lim_fixed,
                    mag_width_samples, mag_width_fixed,
                    cz_lim_samples, cz_lim_fixed,
                    cz_width_samples, cz_width_fixed,
                    e_mag_obs_all, e_czcmb_all,
                    r_min, r_max, r2mu, r2z, n_ppc, n_hosts):
    """PPC using 3D field sampling (matches mock generator)."""
    n_post = len(H0)
    h_max = float(np.max(H0)) / 100

    # Build field loader
    field_name = get_nested(
        config, "io/PV_main/EDD_TRGB/which_host_los", None)
    if field_name is None:
        raise ValueError(
            "use_reconstruction=True but no which_host_los specified")
    field_config = config["io"]["reconstruction_main"][field_name]
    loader_cls = name2field_loader(field_name)
    field_loader = loader_cls(**field_config)

    # Build position pool
    r_sphere = r_max * h_max
    pool_size = max(n_ppc * 100, 500_000)
    pool = build_field_pool(field_loader, r_sphere, pool_size, gen)

    r_h_pool = pool["r_h"]
    rho_pool = pool["rho"]
    v_los_pool = pool["v_los"]
    rhat_icrs_pool = pool["rhat_icrs"]
    n_pool = len(r_h_pool)

    b1_max = float(np.max(b1))
    rho_biased_max = smoothclip(1 + b1_max * pool["delta_max"])

    # Vectorized rejection-sampling loop
    collected_mag = []
    collected_cz = []
    n_accepted = 0
    batch_size = max(int(2.0 * n_ppc), 1000)

    fprint(f"PPC: generating {n_ppc} galaxies "
           f"(n_post={n_post}, n_hosts={n_hosts}, pool={n_pool})")

    while n_accepted < n_ppc:
        n_need = n_ppc - n_accepted
        batch = min(batch_size, max(n_need * 3, 1000))

        # Draw posterior sample and pool indices
        idx_post = gen.integers(0, n_post, batch)
        idx_pool = gen.integers(0, n_pool, batch)

        # Pool values
        r_h = r_h_pool[idx_pool]
        rho = rho_pool[idx_pool]
        v_los = v_los_pool[idx_pool]
        rhat = rhat_icrs_pool[idx_pool]

        # Posterior values
        h = H0[idx_post] / 100
        M = M_TRGB[idx_post]
        sint = sigma_int[idx_post]
        sv = sigma_v[idx_post]
        Vext_cand = Vext[idx_post]
        bt = beta[idx_post]
        b1_cand = b1[idx_post]

        # Distance cut: r_Mpc = r_h / h
        r_Mpc = r_h / h
        in_range = (r_Mpc >= r_min) & (r_Mpc <= r_max)

        # Density rejection
        weight = smoothclip(1 + b1_cand * (rho - 1))
        accept_bias = gen.random(batch) < (weight / rho_biased_max)

        accept = in_range & accept_bias
        if not np.any(accept):
            continue

        # Apply mask
        r_Mpc = r_Mpc[accept]
        h_acc = h[accept]
        M_acc = M[accept]
        sint_acc = sint[accept]
        sv_acc = sv[accept]
        Vext_acc = Vext_cand[accept]
        bt_acc = bt[accept]
        v_los_acc = v_los[accept]
        rhat_acc = rhat[accept]
        n_batch = len(r_Mpc)

        # Distance modulus
        mu = np.asarray(r2mu(r_Mpc, h=h_acc))

        # Measurement errors resampled from observed distribution
        e_mag = gen.choice(e_mag_obs_all, n_batch)
        e_cz = gen.choice(e_czcmb_all, n_batch)

        # Apparent magnitude
        sigma_mag = np.sqrt(e_mag**2 + sint_acc**2)
        mag_sim = gen.normal(M_acc + mu, sigma_mag)

        # Redshift with peculiar velocity from field
        z_cosmo = np.asarray(r2z(r_Mpc, h=h_acc))
        Vext_rad = np.sum(Vext_acc * rhat_acc, axis=1)
        Vpec = bt_acc * v_los_acc + Vext_rad

        cz_pred = SPEED_OF_LIGHT * (
            (1 + z_cosmo) * (1 + Vpec / SPEED_OF_LIGHT) - 1)
        sigma_cz = np.sqrt(e_cz**2 + sv_acc**2)
        cz_sim = gen.normal(cz_pred, sigma_cz)

        # Selection
        accept_sel = _apply_selection_ppc(
            mag_sim, cz_sim, which_sel, idx_post[accept],
            mag_lim_samples, mag_lim_fixed,
            mag_width_samples, mag_width_fixed,
            cz_lim_samples, cz_lim_fixed,
            cz_width_samples, cz_width_fixed, gen)

        collected_mag.append(mag_sim[accept_sel])
        collected_cz.append(cz_sim[accept_sel])
        n_accepted += int(np.sum(accept_sel))

    mag_sim = np.concatenate(collected_mag)[:n_ppc]
    cz_sim = np.concatenate(collected_cz)[:n_ppc]
    fprint(f"PPC: generated {n_ppc} simulated galaxies (field path).")
    return mag_sim, cz_sim


###############################################################################
#                    Homogeneous PPC path (no field)                          #
###############################################################################


def _ppc_homogeneous_path(gen, H0, M_TRGB, sigma_int, sigma_v, Vext, beta,
                          which_sel, mag_lim_samples, mag_lim_fixed,
                          mag_width_samples, mag_width_fixed,
                          cz_lim_samples, cz_lim_fixed,
                          cz_width_samples, cz_width_fixed,
                          e_mag_obs_all, e_czcmb_all,
                          r_min, r_max, r2mu, r2z, n_ppc, n_hosts):
    """PPC with homogeneous distance sampling (no reconstruction)."""
    n_post = len(H0)

    collected_mag = []
    collected_cz = []
    n_accepted = 0
    batch_size = max(int(2.0 * n_ppc), 1000)

    fprint(f"PPC: generating {n_ppc} galaxies "
           f"(n_post={n_post}, n_hosts={n_hosts}, homogeneous)")

    while n_accepted < n_ppc:
        n_need = n_ppc - n_accepted
        batch = min(batch_size, max(n_need * 3, 1000))

        # Draw posterior samples
        idx_post = gen.integers(0, n_post, batch)

        # Draw distance from r^2 volume prior
        u = gen.random(batch)
        r = (r_min**3 + u * (r_max**3 - r_min**3))**(1.0 / 3)

        h = H0[idx_post] / 100
        M = M_TRGB[idx_post]
        sint = sigma_int[idx_post]
        sv = sigma_v[idx_post]
        Vext_cand = Vext[idx_post]

        # Random sky direction
        RA_rand = gen.uniform(0, 360, batch)
        dec_rand = np.rad2deg(np.arcsin(gen.uniform(-1, 1, batch)))
        rhat = radec_to_cartesian(RA_rand, dec_rand)

        # Distance modulus
        mu = np.asarray(r2mu(r, h=h))

        # Measurement errors resampled from observed distribution
        e_mag = gen.choice(e_mag_obs_all, batch)
        e_cz = gen.choice(e_czcmb_all, batch)

        # Apparent magnitude
        sigma_mag = np.sqrt(e_mag**2 + sint**2)
        mag_sim = gen.normal(M + mu, sigma_mag)

        # Redshift (no field velocity, only Vext)
        z_cosmo = np.asarray(r2z(r, h=h))
        Vext_rad = np.sum(Vext_cand * rhat, axis=1)
        cz_pred = SPEED_OF_LIGHT * (
            (1 + z_cosmo) * (1 + Vext_rad / SPEED_OF_LIGHT) - 1)
        sigma_cz = np.sqrt(e_cz**2 + sv**2)
        cz_sim = gen.normal(cz_pred, sigma_cz)

        # Selection
        accept_sel = _apply_selection_ppc(
            mag_sim, cz_sim, which_sel, idx_post,
            mag_lim_samples, mag_lim_fixed,
            mag_width_samples, mag_width_fixed,
            cz_lim_samples, cz_lim_fixed,
            cz_width_samples, cz_width_fixed, gen)

        collected_mag.append(mag_sim[accept_sel])
        collected_cz.append(cz_sim[accept_sel])
        n_accepted += int(np.sum(accept_sel))

    mag_sim = np.concatenate(collected_mag)[:n_ppc]
    cz_sim = np.concatenate(collected_cz)[:n_ppc]
    fprint(f"PPC: generated {n_ppc} simulated galaxies (homogeneous).")
    return mag_sim, cz_sim


###############################################################################
#                        Selection helper                                     #
###############################################################################


def _apply_selection_ppc(mag_sim, cz_sim, which_sel, idx_post,
                         mag_lim_samples, mag_lim_fixed,
                         mag_width_samples, mag_width_fixed,
                         cz_lim_samples, cz_lim_fixed,
                         cz_width_samples, cz_width_fixed, gen):
    """Apply selection function, returning boolean mask."""
    n = len(mag_sim)
    if which_sel == "TRGB_magnitude":
        ml = mag_lim_samples[idx_post] \
            if mag_lim_samples is not None else mag_lim_fixed
        mw = mag_width_samples[idx_post] \
            if mag_width_samples is not None else mag_width_fixed
        p_sel = norm.cdf((ml - mag_sim) / mw)
        return gen.random(n) < p_sel
    elif which_sel == "redshift":
        cl = cz_lim_samples[idx_post] \
            if cz_lim_samples is not None else cz_lim_fixed
        cw = cz_width_samples[idx_post] \
            if cz_width_samples is not None else cz_width_fixed
        p_sel = norm.cdf((cl - cz_sim) / cw)
        return gen.random(n) < p_sel
    return np.ones(n, dtype=bool)


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
