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

from .field import name2field_loader
from .field.field_interp import build_regular_interpolator
from .util import (SPEED_OF_LIGHT, cartesian_to_radec, fprint,
                   galactic_to_radec, get_nested, load_config,
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
#                              Smooth clip                                    #
###############################################################################


def _smoothclip(x, tau=0.1):
    """Smooth zero-clipping matching the model's smoothclip_nr."""
    return 0.5 * (x + np.sqrt(x**2 + tau**2))


###############################################################################
#                      Coordinate conversion                                  #
###############################################################################


def _xyz_to_radec_icrs(xyz, r_h, coordinate_frame):
    """Convert field-frame Cartesian offsets to ICRS (RA, dec) in degrees."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if coordinate_frame == "icrs":
        return cartesian_to_radec(x, y, z)
    elif coordinate_frame == "galactic":
        ell = np.rad2deg(np.arctan2(y, x))
        b = np.rad2deg(np.arcsin(z / r_h))
        return galactic_to_radec(ell, b)
    elif coordinate_frame == "supergalactic":
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        sgl = np.rad2deg(np.arctan2(y, x))
        sgb = np.rad2deg(np.arcsin(z / r_h))
        c = SkyCoord(sgl=sgl * u.deg, sgb=sgb * u.deg,
                     frame='supergalactic')
        return c.icrs.ra.deg, c.icrs.dec.deg
    else:
        raise ValueError(f"Unknown coordinate frame: {coordinate_frame}")


###############################################################################
#                     Sampling sphere radius                                  #
###############################################################################


def _compute_r_max_eff(which_sel, mag_lim_samples, mag_lim_fixed,
                       mag_width_samples, mag_width_fixed,
                       cz_lim_samples, cz_lim_fixed,
                       M_TRGB, sigma_int, e_mag_obs, H0, r_max):
    """Compute effective maximum distance (Mpc) based on selection."""
    if which_sel == "TRGB_magnitude":
        ml = float(np.max(mag_lim_samples)) \
            if mag_lim_samples is not None else mag_lim_fixed
        if ml is None or isinstance(ml, str):
            return r_max

        M_min = float(np.min(M_TRGB))
        sint_max = float(np.max(sigma_int))
        e_mag_max = float(np.max(e_mag_obs))
        mw = float(np.max(mag_width_samples)) \
            if mag_width_samples is not None else mag_width_fixed
        if mw is None or isinstance(mw, str):
            mw = 0.0

        sigma_tot = np.sqrt(sint_max**2 + e_mag_max**2 + mw**2)
        mu_cutoff = ml - M_min + 5 * sigma_tot
        return min(10**((mu_cutoff - 25) / 5), r_max)

    elif which_sel == "redshift":
        cl = float(np.max(cz_lim_samples)) \
            if cz_lim_samples is not None else cz_lim_fixed
        if cl is None or isinstance(cl, str):
            return r_max
        h_min = float(np.min(H0)) / 100
        return min(cl / (h_min * 100) * 1.5, r_max)

    return r_max


###############################################################################
#                       3D field pool                                         #
###############################################################################


def _build_field_pool(field_loader, r_sphere, pool_size, gen):
    """Pre-sample 3D positions and evaluate density/velocity in one batch.

    Returns
    -------
    dict with r_h, rho, v_los, rhat_icrs, delta_max.
    """
    obs = field_loader.observer_pos
    coord_frame = field_loader.coordinate_frame

    eps = 1e-4
    fprint("PPC: loading density field...")
    density_raw = field_loader.load_density()
    density_log = np.log(density_raw + eps).astype(np.float32)
    f_density = build_regular_interpolator(
        density_log, field_loader.boxsize,
        fill_value=np.float32(np.log(1 + eps)))
    delta_max = float(density_raw.max()) - 1
    del density_raw, density_log

    fprint("PPC: loading velocity field...")
    velocity_3d = field_loader.load_velocity()
    f_vel = []
    for i in range(3):
        f_vel.append(build_regular_interpolator(
            velocity_3d[i], field_loader.boxsize,
            fill_value=np.float32(0)))
    del velocity_3d

    # Sample positions uniformly in sphere (overallocate for sphere/cube ratio)
    rmin_h = 0.1
    n_cube = int(pool_size * 2.0)
    fprint(f"PPC: sampling {n_cube} candidate positions "
           f"(r_sphere={r_sphere:.1f} Mpc/h)...")
    xyz = gen.uniform(-r_sphere, r_sphere,
                      (n_cube, 3)).astype(np.float32)
    r_sq = np.sum(xyz**2, axis=1)
    mask = (r_sq < r_sphere**2) & (r_sq > rmin_h**2)
    xyz = xyz[mask]
    if len(xyz) > pool_size:
        xyz = xyz[:pool_size]

    r_h = np.linalg.norm(xyz, axis=1)

    # Evaluate density at all pool positions (one batch)
    fprint(f"PPC: evaluating density at {len(xyz)} positions...")
    pos_box = (xyz + obs[None, :]).astype(np.float32)
    rho_log = f_density(pos_box)
    rho = np.exp(rho_log) - eps
    np.clip(rho, eps, None, out=rho)

    # Evaluate radial velocity at all pool positions (one batch)
    fprint("PPC: evaluating velocity...")
    rhat = xyz / r_h[:, None]
    v_los = np.zeros(len(xyz), dtype=np.float32)
    for i in range(3):
        v_los += f_vel[i](pos_box) * rhat[:, i]

    del f_density, f_vel, pos_box

    # Convert to ICRS direction vectors
    RA, dec = _xyz_to_radec_icrs(xyz, r_h, coord_frame)
    rhat_icrs = radec_to_cartesian(RA, dec)

    fprint(f"PPC: pool ready ({len(xyz)} positions)")
    return {
        "r_h": r_h.astype(np.float64),
        "rho": rho.astype(np.float64),
        "v_los": v_los.astype(np.float64),
        "rhat_icrs": rhat_icrs.astype(np.float64),
        "delta_max": delta_max,
    }


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

    # ---- Cosmography (numpy, built once) ----
    Om = get_nested(config, "model/Om", 0.3)
    r2mu_100, r2z_100 = _build_numpy_cosmography(Om)

    # ---- Distance limits ----
    r_limits = get_nested(config, "model/r_limits_malmquist", [0.01, 150])
    r_min, r_max = r_limits[0], r_limits[1]

    # ---- Effective sphere radius (selection-aware) ----
    r_max_eff = _compute_r_max_eff(
        which_sel, mag_lim_samples, mag_lim_fixed,
        mag_width_samples, mag_width_fixed,
        cz_lim_samples, cz_lim_fixed,
        M_TRGB, sigma_int, e_mag_obs_all, H0, r_max)
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
            r_min, r_max_eff, r2mu_100, r2z_100, n_ppc, n_hosts)
    else:
        mag_sim, cz_sim = _ppc_homogeneous_path(
            gen, H0, M_TRGB, sigma_int, sigma_v, Vext, beta,
            which_sel, mag_lim_samples, mag_lim_fixed,
            mag_width_samples, mag_width_fixed,
            cz_lim_samples, cz_lim_fixed, cz_width_samples, cz_width_fixed,
            e_mag_obs_all, e_czcmb_all,
            r_min, r_max_eff, r2mu_100, r2z_100, n_ppc, n_hosts)

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
                    r_min, r_max, r2mu_100, r2z_100, n_ppc, n_hosts):
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
    pool = _build_field_pool(field_loader, r_sphere, pool_size, gen)

    r_h_pool = pool["r_h"]
    rho_pool = pool["rho"]
    v_los_pool = pool["v_los"]
    rhat_icrs_pool = pool["rhat_icrs"]
    n_pool = len(r_h_pool)

    b1_max = float(np.max(b1))
    rho_biased_max = _smoothclip(1 + b1_max * pool["delta_max"])

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
        weight = _smoothclip(1 + b1_cand * (rho - 1))
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
        mu = _distmod(r_Mpc, h_acc, r2mu_100)

        # Measurement errors resampled from observed distribution
        e_mag = gen.choice(e_mag_obs_all, n_batch)
        e_cz = gen.choice(e_czcmb_all, n_batch)

        # Apparent magnitude
        sigma_mag = np.sqrt(e_mag**2 + sint_acc**2)
        mag_sim = gen.normal(M_acc + mu, sigma_mag)

        # Redshift with peculiar velocity from field
        z_cosmo = _redshift(r_Mpc, h_acc, r2z_100)
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
                          r_min, r_max, r2mu_100, r2z_100, n_ppc, n_hosts):
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
        mu = _distmod(r, h, r2mu_100)

        # Measurement errors resampled from observed distribution
        e_mag = gen.choice(e_mag_obs_all, batch)
        e_cz = gen.choice(e_czcmb_all, batch)

        # Apparent magnitude
        sigma_mag = np.sqrt(e_mag**2 + sint**2)
        mag_sim = gen.normal(M + mu, sigma_mag)

        # Redshift (no field velocity, only Vext)
        z_cosmo = _redshift(r, h, r2z_100)
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
