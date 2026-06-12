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

from ..cosmo.cosmography import Distance2Distmod, Distance2Redshift
from ..field import name2field_loader
from ..pvdata.field_cache import (_field_cache_dir_from_config,
                                  _field_cache_enabled_from_config)
from ..pvdata.volume_density import (_density_unit_normalization,
                                     _load_volume_data_for_H0)
from ..util import (SPEED_OF_LIGHT, fprint, galactic_to_radec, get_nested,
                    load_config, radec_to_cartesian)
from ._field_utils import build_field_pool, compute_r_max_selection, smoothclip


def _flat(x):
    """Flatten scalar samples while preserving vector-valued samples."""
    x = np.asarray(x)
    if x.ndim > 1 and x.shape[-1] != 3:
        return x.reshape(-1)
    if x.ndim > 2:
        return x.reshape(-1, x.shape[-1])
    return x


def _sample_or_default(samples, key, n, default):
    if key in samples:
        return _flat(samples[key])
    return np.full(n, default)


def _vext_samples(samples, n_post):
    """Return Cartesian external-velocity samples."""
    if "Vext" in samples:
        return _flat(samples["Vext"])
    if all(k in samples for k in ("Vext_mag", "Vext_ell", "Vext_b")):
        mag = _flat(samples["Vext_mag"])
        ell = _flat(samples["Vext_ell"])
        b = _flat(samples["Vext_b"])
        ra, dec = galactic_to_radec(ell, b)
        return mag[:, None] * radec_to_cartesian(ra, dec)
    return np.zeros((n_post, 3))


def _draw_cz(gen, mean, sigma, nu=None):
    """Draw Gaussian or Student-t redshift residuals."""
    if nu is None:
        return gen.normal(mean, sigma)
    return mean + sigma * gen.standard_t(nu, size=np.shape(mean))


def _sigma_v_from_density(rho, sigma_v_low, sigma_v_high, log_rho_t, k):
    rho = np.clip(rho, 1e-6, None)
    return sigma_v_low + (sigma_v_high - sigma_v_low) / (
        1.0 + np.exp(-k * (np.log(rho) - log_rho_t)))


def _available_field_indices(data):
    """Choose one field realization for a reconstruction PPC."""
    if "los_field_indices" in data:
        return np.asarray(data["los_field_indices"], dtype=int)
    elif "host_los_density" in data:
        n = np.asarray(data["host_los_density"]).shape[0]
        return np.arange(n, dtype=int)
    return np.array([0], dtype=int)


def _selected_field_index(data, gen, field_index=None):
    """Choose one field realization for a reconstruction PPC."""
    if field_index is not None:
        return int(field_index)
    field_indices = _available_field_indices(data)
    return int(gen.choice(field_indices))


def _field_name_config(config):
    field_name = get_nested(
        config, "io/PV_main/EDD_TRGB/reconstruction", None)
    if field_name is None:
        raise ValueError(
            "use_reconstruction=True but no reconstruction specified")
    field_config = dict(config["io"]["reconstruction_main"][field_name])
    return field_name, field_config


def _density_divisor(field_name):
    norm_info = _density_unit_normalization(field_name)
    if norm_info is None:
        return None
    return norm_info[0]


def _bias_samples(samples, config, beta, n_post):
    which_bias = get_nested(config, "model/which_bias", "linear")
    Om = get_nested(config, "model/Om", get_nested(config, "model/Om0", 0.3))
    out = {"which_bias": which_bias}
    if which_bias == "unity":
        return out
    if which_bias == "linear_from_beta":
        out["b1"] = Om**0.55 / beta
    elif "linear" in which_bias:
        out["b1"] = _sample_or_default(samples, "b1", n_post, 1.0)
    elif which_bias == "double_powerlaw":
        out["alpha_low"] = _flat(samples["alpha_low"])
        out["alpha_high"] = _flat(samples["alpha_high"])
        out["log_rho_t"] = _flat(samples["log_rho_t"])
        out["log_rho_width"] = _sample_or_default(
            samples, "log_rho_width", n_post, 1.0)
    elif which_bias == "quadratic":
        out["b1"] = _sample_or_default(samples, "b1", n_post, 1.0)
        out["b2"] = _sample_or_default(samples, "b2", n_post, 0.0)
    elif which_bias == "cubic":
        out["b1"] = _sample_or_default(samples, "b1", n_post, 1.0)
        out["b2"] = _sample_or_default(samples, "b2", n_post, 0.0)
        out["b3"] = _sample_or_default(samples, "b3", n_post, 0.0)
    else:
        raise ValueError(f"Unsupported PPC galaxy-bias model: {which_bias}")
    return out


def _bias_values(rho, bias, idx_post):
    which_bias = bias["which_bias"]
    if which_bias == "unity":
        return np.ones_like(rho)
    delta = rho - 1.0
    if "linear" in which_bias:
        b1 = bias["b1"][idx_post]
        return smoothclip(1.0 + b1 * delta)
    if which_bias == "double_powerlaw":
        alpha_low = bias["alpha_low"][idx_post]
        alpha_high = bias["alpha_high"][idx_post]
        log_rho_t = bias["log_rho_t"][idx_post]
        log_rho_width = bias["log_rho_width"][idx_post]
        log_x = np.log(np.clip(rho, 1e-6, None)) - log_rho_t
        z = log_x / log_rho_width
        log_weight = (
            alpha_low * log_x
            + ((alpha_high - alpha_low) * log_rho_width
               * np.logaddexp(0.0, z)))
        return np.exp(np.clip(log_weight, -50.0, 50.0))
    if which_bias == "quadratic":
        b1 = bias["b1"][idx_post]
        b2 = bias["b2"][idx_post]
        return smoothclip(1.0 + b1 * delta + b2 * delta**2)
    if which_bias == "cubic":
        b1 = bias["b1"][idx_post]
        b2 = bias["b2"][idx_post]
        b3 = bias["b3"][idx_post]
        return smoothclip(1.0 + b1 * delta + b2 * delta**2 + b3 * delta**3)
    raise ValueError(f"Unsupported PPC galaxy-bias model: {which_bias}")


def _bias_upper_bound(rho_support, bias, idx_post):
    """Return a conservative bias-weight bound for rejection sampling."""
    which_bias = bias["which_bias"]
    idx_post = np.asarray(idx_post)
    rho_min = float(np.min(rho_support))
    rho_max = float(np.max(rho_support))

    if which_bias == "unity":
        return np.ones_like(idx_post, dtype=float)
    if "linear" in which_bias:
        b1 = bias["b1"][idx_post]
        rho_extreme = np.where(b1 >= 0.0, rho_max, rho_min)
        return _bias_values(rho_extreme, bias, idx_post)
    if which_bias == "double_powerlaw":
        rho_t = np.exp(bias["log_rho_t"][idx_post])
        rho_eval = np.vstack([
            np.full_like(rho_t, rho_min),
            np.clip(rho_t, rho_min, rho_max),
            np.full_like(rho_t, rho_max),
        ])
        idx = idx_post[None, :]
        return np.max(_bias_values(rho_eval, bias, idx), axis=0)
    rho = rho_support[:, None]
    idx = idx_post[None, :]
    return np.max(_bias_values(rho, bias, idx), axis=0)


def _rho_support(rho, max_size=4096):
    rho = np.asarray(rho)
    if len(rho) <= max_size:
        return rho
    q = np.linspace(0.0, 1.0, max_size)
    return np.quantile(rho, q)


def _cached_volume_pool(config, field_name, field_index):
    """Load one cached H0 cut-out with radial velocities for PPC sampling."""
    _, field_config = _field_name_config(config)
    Om = get_nested(config, "model/Om", get_nested(config, "model/Om0", 0.3))
    which_bias = get_nested(config, "model/which_bias", "linear")
    grid_radius = get_nested(config, "model/selection_integral_grid_radius")
    geometry = get_nested(
        config, "model/selection_integral_geometry", "sphere")
    loaded = _load_volume_data_for_H0(
        field_name, field_config, [field_index], which_bias, Om,
        subcube_radius=grid_radius,
        voxel_subsample_fraction=get_nested(
            config, "model/density_3d_subsample_fraction", 1.0),
        voxel_subsample_seed=get_nested(
            config, "model/density_3d_subsample_seed", 42),
        load_velocity=True,
        geometry=geometry,
        cache_dir=(
            _field_cache_dir_from_config(config)
            if _field_cache_enabled_from_config(config) else None),
        cache_enabled=_field_cache_enabled_from_config(config))

    density = np.asarray(loaded["density_3d_fields"])[0]
    if loaded.get("density_3d_mode") == "log_rho":
        rho = np.exp(density)
    else:
        rho = 1.0 + density
    rho = np.clip(rho, 1e-6, None)
    r_h = np.exp(np.asarray(loaded["log_r_3d"]))
    base_weight = np.ones_like(r_h, dtype=np.float64)
    if "log_volume_weight_3d" in loaded:
        log_w = np.asarray(loaded["log_volume_weight_3d"], dtype=np.float64)
        base_weight = np.exp(log_w - np.max(log_w))
    base_weight = base_weight / np.sum(base_weight)
    rhat = np.column_stack([
        np.asarray(loaded["rhat_x_3d"]),
        np.asarray(loaded["rhat_y_3d"]),
        np.asarray(loaded["rhat_z_3d"]),
    ])
    return {
        "r_h": r_h,
        "rho": rho,
        "v_los": np.asarray(loaded["vrad_3d_fields"])[0],
        "rhat_icrs": rhat,
        "base_weight": base_weight,
        "rho_support": _rho_support(rho),
    }

###############################################################################
#                         PPC generation                                      #
###############################################################################


def generate_trgb_ppc(samples, data, config, n_ppc=None, seed=42,
                      field_index=None):
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

    H0 = _flat(samples["H0"])
    M_TRGB = _flat(samples["M_TRGB"])
    sigma_int = _flat(samples["sigma_int"])
    n_post = len(H0)
    c_star = _sample_or_default(samples, "c_star", n_post, 1.23)
    Vext = _vext_samples(samples, n_post)
    beta = _sample_or_default(samples, "beta", n_post, 0.0)
    bias = _bias_samples(samples, config, beta, n_post)

    use_density_sigma_v = get_nested(
        config, "model/use_density_dependent_sigma_v", False)
    if use_density_sigma_v:
        sigma_v = (
            _flat(samples["sigma_v_low"]),
            _flat(samples["sigma_v_high"]),
            _flat(samples["log_sigma_v_rho_t"]),
            _flat(samples["sigma_v_k"]),
        )
    else:
        sigma_v = _flat(samples["sigma_v"])

    nu_cz = _flat(samples["nu_cz"]) if (
        get_nested(config, "model/cz_likelihood", "gaussian") == "student_t"
        and "nu_cz" in samples) else None

    # Selection parameters
    which_sel = get_nested(config, "model/which_selection", None)
    mag_lim_samples = _flat(samples["mag_lim_TRGB"]) \
        if "mag_lim_TRGB" in samples else None
    mag_width_samples = _flat(samples["mag_lim_TRGB_width"]) \
        if "mag_lim_TRGB_width" in samples else None
    mag_lim_fixed = get_nested(config, "model/mag_lim_TRGB", None)
    mag_width_fixed = get_nested(config, "model/mag_lim_TRGB_width", None)

    # ---- Data ----
    mag_obs = np.asarray(data["mag_obs"])
    cz_obs = np.asarray(data["czcmb"])
    e_mag_obs_all = np.asarray(data["e_mag_obs"])
    e_czcmb_all = np.asarray(data["e_czcmb"])
    colour_dered_all = np.asarray(data["colour_dered"])
    colour_dered_mean = np.mean(colour_dered_all)
    colour_dered_std = np.std(colour_dered_all)
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
        cz_lim=None,
        h=H0 / 100, r_max=r_max,
        colour_mean=colour_dered_mean, c_star=c_star,
        colour_std=colour_dered_std)
    fprint(f"PPC: effective r_max = {r_max_eff:.1f} Mpc "
           f"(config r_max = {r_max:.1f} Mpc)")

    # ---- Check for reconstruction field ----
    use_reconstruction = get_nested(config, "model/use_reconstruction", False)

    if use_reconstruction:
        mag_sim, cz_sim = _ppc_field_path(
            gen, config, H0, M_TRGB, c_star, sigma_int, sigma_v, Vext,
            beta, bias, nu_cz, use_density_sigma_v, data, field_index,
            which_sel, mag_lim_samples, mag_lim_fixed,
            mag_width_samples, mag_width_fixed,
            e_mag_obs_all, e_czcmb_all, colour_dered_all,
            r_min, r_max_eff, r2mu, r2z, n_ppc, n_hosts)
    else:
        mag_sim, cz_sim = _ppc_homogeneous_path(
            gen, H0, M_TRGB, c_star, sigma_int, sigma_v, Vext, beta,
            nu_cz,
            which_sel, mag_lim_samples, mag_lim_fixed,
            mag_width_samples, mag_width_fixed,
            e_mag_obs_all, e_czcmb_all, colour_dered_all,
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


def _ppc_field_path(gen, config, H0, M_TRGB, c_star, sigma_int, sigma_v, Vext,
                    beta, bias, nu_cz, use_density_sigma_v, data,
                    field_index,
                    which_sel, mag_lim_samples, mag_lim_fixed,
                    mag_width_samples, mag_width_fixed,
                    e_mag_obs_all, e_czcmb_all, colour_dered_all,
                    r_min, r_max, r2mu, r2z, n_ppc, n_hosts):
    """PPC using 3D field sampling (matches mock generator)."""
    n_post = len(H0)
    field_name, field_config = _field_name_config(config)
    field_index = _selected_field_index(data, gen, field_index=field_index)
    try:
        pool = _cached_volume_pool(config, field_name, field_index)
        pool_source = "cached cut-out"
    except Exception as exc:
        fprint("PPC: cached cut-out unavailable; falling back to full field "
               f"pool ({exc}).")
        field_config = dict(field_config)
        field_config.setdefault("nsim", field_index)
        field_loader = name2field_loader(field_name)(**field_config)
        r_sphere = r_max * (float(np.max(H0)) / 100)
        pool_size = max(n_ppc * 100, 500_000)
        pool = build_field_pool(
            field_loader, r_sphere, pool_size, gen,
            density_divisor=_density_divisor(field_name))
        pool["base_weight"] = None
        pool["rho_support"] = _rho_support(pool["rho"])
        pool_source = "full field pool"

    r_h_pool = pool["r_h"]
    rho_pool = pool["rho"]
    v_los_pool = pool["v_los"]
    rhat_icrs_pool = pool["rhat_icrs"]
    n_pool = len(r_h_pool)

    # Vectorized rejection-sampling loop
    collected_mag = []
    collected_cz = []
    n_accepted = 0
    batch_size = max(int(20.0 * n_ppc), 10_000)

    fprint(f"PPC: generating {n_ppc} galaxies "
           f"(n_post={n_post}, n_hosts={n_hosts}, "
           f"field={field_name}[{field_index}], pool={n_pool}, "
           f"{pool_source})")

    while n_accepted < n_ppc:
        n_need = n_ppc - n_accepted
        batch = min(batch_size, max(n_need * 3, 1000))

        # Draw posterior sample and pool indices
        idx_post = gen.integers(0, n_post, batch)
        if pool["base_weight"] is None:
            idx_pool = gen.integers(0, n_pool, batch)
        else:
            idx_pool = gen.choice(n_pool, batch, p=pool["base_weight"])

        # Pool values
        r_h = r_h_pool[idx_pool]
        rho = rho_pool[idx_pool]
        v_los = v_los_pool[idx_pool]
        rhat = rhat_icrs_pool[idx_pool]

        # Posterior values
        h = H0[idx_post] / 100
        M = M_TRGB[idx_post]
        cs = c_star[idx_post]
        sint = sigma_int[idx_post]
        if use_density_sigma_v:
            sv_low = sigma_v[0][idx_post]
            sv_high = sigma_v[1][idx_post]
            sv_log_rho_t = sigma_v[2][idx_post]
            sv_k = sigma_v[3][idx_post]
        else:
            sv = sigma_v[idx_post]
        Vext_cand = Vext[idx_post]
        bt = beta[idx_post]

        # Distance cut: r_Mpc = r_h / h
        r_Mpc = r_h / h
        in_range = (r_Mpc >= r_min) & (r_Mpc <= r_max)

        # Density rejection
        weight = _bias_values(rho, bias, idx_post)
        weight_max = _bias_upper_bound(pool["rho_support"], bias, idx_post)
        accept_bias = gen.random(batch) < np.minimum(weight / weight_max, 1.0)

        accept = in_range & accept_bias
        if not np.any(accept):
            continue

        # Apply mask
        idx_post_acc = idx_post[accept]
        r_Mpc = r_Mpc[accept]
        h_acc = h[accept]
        M_acc = M[accept]
        cs_acc = cs[accept]
        sint_acc = sint[accept]
        if use_density_sigma_v:
            sv_acc = _sigma_v_from_density(
                rho[accept], sv_low[accept], sv_high[accept],
                sv_log_rho_t[accept], sv_k[accept])
        else:
            sv_acc = sv[accept]
        Vext_acc = Vext_cand[accept]
        bt_acc = bt[accept]
        v_los_acc = v_los[accept]
        rhat_acc = rhat[accept]
        nu_acc = None if nu_cz is None else nu_cz[idx_post_acc]
        n_batch = len(r_Mpc)

        # Distance modulus
        mu = np.asarray(r2mu(r_Mpc, h=h_acc))

        # Measurement errors and colour resampled from observed distribution
        e_mag = gen.choice(e_mag_obs_all, n_batch)
        e_cz = gen.choice(e_czcmb_all, n_batch)
        colour = gen.choice(colour_dered_all, n_batch)

        # Apparent magnitude with colour standardisation
        sigma_mag = np.sqrt(e_mag**2 + sint_acc**2)
        mag_sim = gen.normal(
            M_acc + 0.2 * (colour - cs_acc) + mu, sigma_mag)

        # Redshift with peculiar velocity from field
        z_cosmo = np.asarray(r2z(r_Mpc, h=h_acc))
        Vext_rad = np.sum(Vext_acc * rhat_acc, axis=1)
        Vpec = bt_acc * v_los_acc + Vext_rad

        cz_pred = SPEED_OF_LIGHT * (
            (1 + z_cosmo) * (1 + Vpec / SPEED_OF_LIGHT) - 1)
        sigma_cz = np.sqrt(e_cz**2 + sv_acc**2)
        cz_sim = _draw_cz(gen, cz_pred, sigma_cz, nu=nu_acc)

        # Selection
        accept_sel = _apply_selection_ppc(
            mag_sim, which_sel, idx_post_acc,
            mag_lim_samples, mag_lim_fixed,
            mag_width_samples, mag_width_fixed, gen)

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


def _ppc_homogeneous_path(gen, H0, M_TRGB, c_star, sigma_int, sigma_v,
                          Vext, beta, nu_cz,
                          which_sel, mag_lim_samples, mag_lim_fixed,
                          mag_width_samples, mag_width_fixed,
                          e_mag_obs_all, e_czcmb_all, colour_dered_all,
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
        cs = c_star[idx_post]
        sint = sigma_int[idx_post]
        sv = sigma_v[idx_post]
        Vext_cand = Vext[idx_post]
        nu = None if nu_cz is None else nu_cz[idx_post]

        # Random sky direction
        RA_rand = gen.uniform(0, 360, batch)
        dec_rand = np.rad2deg(np.arcsin(gen.uniform(-1, 1, batch)))
        rhat = radec_to_cartesian(RA_rand, dec_rand)

        # Distance modulus
        mu = np.asarray(r2mu(r, h=h))

        # Measurement errors and colour resampled from observed distribution
        e_mag = gen.choice(e_mag_obs_all, batch)
        e_cz = gen.choice(e_czcmb_all, batch)
        colour = gen.choice(colour_dered_all, batch)

        # Apparent magnitude with colour standardisation
        sigma_mag = np.sqrt(e_mag**2 + sint**2)
        mag_sim = gen.normal(M + 0.2 * (colour - cs) + mu, sigma_mag)

        # Redshift (no field velocity, only Vext)
        z_cosmo = np.asarray(r2z(r, h=h))
        Vext_rad = np.sum(Vext_cand * rhat, axis=1)
        cz_pred = SPEED_OF_LIGHT * (
            (1 + z_cosmo) * (1 + Vext_rad / SPEED_OF_LIGHT) - 1)
        sigma_cz = np.sqrt(e_cz**2 + sv**2)
        cz_sim = _draw_cz(gen, cz_pred, sigma_cz, nu=nu)

        # Selection
        accept_sel = _apply_selection_ppc(
            mag_sim, which_sel, idx_post,
            mag_lim_samples, mag_lim_fixed,
            mag_width_samples, mag_width_fixed, gen)

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


def _apply_selection_ppc(mag_sim, which_sel, idx_post,
                         mag_lim_samples, mag_lim_fixed,
                         mag_width_samples, mag_width_fixed, gen):
    """Apply selection function, returning boolean mask."""
    n = len(mag_sim)
    if which_sel == "TRGB_magnitude":
        ml = mag_lim_samples[idx_post] \
            if mag_lim_samples is not None else mag_lim_fixed
        mw = mag_width_samples[idx_post] \
            if mag_width_samples is not None else mag_width_fixed
        p_sel = norm.cdf((ml - mag_sim) / mw)
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
    return {
        "ks_mag_statistic": float(ks_mag.statistic),
        "ks_mag_pvalue": float(ks_mag.pvalue),
        "ks_cz_statistic": float(ks_cz.statistic),
        "ks_cz_pvalue": float(ks_cz.pvalue),
    }
