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
from scipy.stats import ks_2samp, norm
from tqdm.auto import tqdm

from ..field import name2field_loader
from ..pvdata.field_cache import (_field_cache_dir_from_config,
                                  _field_cache_enabled_from_config)
from ..pvdata.field_products import (
    field_smoothing_scale_from_config,
    velocity_field_smoothing_scale_from_config)
from ..pvdata.volume_density import (_density_unit_normalization,
                                     _load_volume_data_for_H0)
from ..util import (SPEED_OF_LIGHT, fprint, galactic_to_radec, get_nested,
                    load_config, radec_to_cartesian)
from ._field_utils import (build_field_pool, compute_r_max_selection,
                           galaxy_bias_params_from_values, galaxy_bias_weight)


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


def _sample_empirical(gen, values, size):
    """Draw from an empirical 1D array using integer indexing."""
    values = np.asarray(values)
    return values[gen.integers(0, len(values), size)]


def _prior_reference_value(config, name, default):
    spec = get_nested(config, f"model/priors/{name}", None)
    if isinstance(spec, dict):
        for key in ("value", "loc", "mean"):
            if key in spec:
                return spec[key]
    return default


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


class _FastDistanceConversions:
    """NumPy distance-modulus/redshift interpolation for PPC draws."""

    def __init__(self, Om0=0.3, zmin=1e-8, zmax=0.5, npoints=1000):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        self.z_grid = np.logspace(np.log10(zmin), np.log10(zmax), npoints)
        self.r_grid = cosmo.comoving_distance(self.z_grid).value
        self.log_r_grid = np.log(self.r_grid)
        self.mu_grid = cosmo.distmod(self.z_grid).value

    def distmod(self, r, h=1):
        return (
            np.interp(np.log(np.asarray(r) * h),
                      self.log_r_grid, self.mu_grid)
            - 5 * np.log10(h))

    def redshift(self, r, h=1):
        return np.interp(np.asarray(r) * h, self.r_grid, self.z_grid)


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
    if which_bias in ("uniform", "unity"):
        return out
    if which_bias == "linear_from_beta":
        out["b1"] = Om**0.55 / beta
    elif which_bias == "linear_from_beta_stochastic":
        if "b1" in samples:
            out["b1"] = _flat(samples["b1"])
        else:
            key = (
                "delta_b1_skipZ" if "delta_b1_skipZ" in samples
                else "delta_b1")
            out["b1"] = Om**0.55 / beta + _sample_or_default(
                samples, key, n_post, 0.0)
    elif which_bias == "linear":
        out["b1"] = _sample_or_default(samples, "b1", n_post, 1.0)
    elif which_bias == "powerlaw":
        out["alpha"] = _sample_or_default(samples, "alpha", n_post, 1.0)
    elif which_bias == "double_powerlaw":
        out["alpha_low"] = _flat(samples["alpha_low"])
        if "alpha_high" in samples:
            out["alpha_high"] = _flat(samples["alpha_high"])
        elif "alpha_high_skipZ" in samples:
            out["alpha_high"] = _flat(samples["alpha_high_skipZ"])
        else:
            out["alpha_high_frac"] = _flat(samples["alpha_high_frac"])
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
    if which_bias == "uniform":
        return np.ones_like(rho, dtype=float)
    params = galaxy_bias_params_from_values(bias, which_bias, idx=idx_post)
    return galaxy_bias_weight(rho, params, which_bias)


def _bias_upper_bound(rho_support, bias, idx_post):
    """Return a conservative bias-weight bound for rejection sampling."""
    which_bias = bias["which_bias"]
    idx_post = np.asarray(idx_post)
    rho_min = float(np.min(rho_support))
    rho_max = float(np.max(rho_support))

    if which_bias == "uniform":
        return np.ones_like(idx_post, dtype=float)
    if which_bias == "unity":
        bound = np.max(_bias_values(rho_support, bias, 0))
        return np.full_like(idx_post, bound, dtype=float)
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
        cache_enabled=_field_cache_enabled_from_config(config),
        field_smoothing_scale=field_smoothing_scale_from_config(config),
        velocity_field_smoothing_scale=(
            velocity_field_smoothing_scale_from_config(config)))

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


def _uniform_density_pool(gen, r_sphere, pool_size):
    """Build a unit-density, zero-velocity pool in Mpc/h coordinates."""
    r_h = r_sphere * gen.random(pool_size)**(1.0 / 3.0)
    phi = gen.uniform(0.0, 2.0 * np.pi, pool_size)
    cos_dec = gen.uniform(-1.0, 1.0, pool_size)
    sin_dec = np.sqrt(1.0 - cos_dec**2)
    rhat = np.column_stack([
        sin_dec * np.cos(phi),
        sin_dec * np.sin(phi),
        cos_dec,
    ])
    return {
        "r_h": r_h,
        "rho": np.ones(pool_size, dtype=np.float64),
        "v_los": np.zeros(pool_size, dtype=np.float64),
        "rhat_icrs": rhat,
        "base_weight": None,
        "rho_support": np.array([1.0], dtype=np.float64),
    }

###############################################################################
#                         PPC generation                                      #
###############################################################################


def generate_trgb_ppc(samples, data, config, n_ppc=None, seed=42,
                      field_index=None, progress=True):
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
    progress : bool
        Whether to show a tqdm progress bar for accepted PPC galaxies.

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
    c_star = _sample_or_default(
        samples, "c_star", n_post,
        _prior_reference_value(config, "c_star", 1.23))
    alpha_c = _sample_or_default(
        samples, "alpha_c", n_post,
        _prior_reference_value(config, "alpha_c", 0.2))
    Vext = _vext_samples(samples, n_post)
    beta = _sample_or_default(samples, "beta", n_post, 0.0)
    use_reconstruction = get_nested(config, "model/use_reconstruction", False)
    bias = (_bias_samples(samples, config, beta, n_post)
            if use_reconstruction else {"which_bias": "uniform"})

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
    mag_min_fixed = get_nested(config, "model/mag_min_TRGB", None)
    mag_lim_fixed = get_nested(config, "model/mag_lim_TRGB", None)
    mag_width_fixed = get_nested(config, "model/mag_lim_TRGB_width", None)

    # ---- Data ----
    mag_obs = np.asarray(data["mag_obs"])
    cz_obs = np.asarray(data["czcmb"])
    e_mag_obs_all = np.asarray(data["e_mag_obs"])
    e_czcmb_all = np.asarray(data["e_czcmb"])
    colour_dered_all = np.asarray(data["colour_dered"])
    e_colour_dered_all = np.asarray(
        data.get("e_colour_dered", np.zeros_like(colour_dered_all)))
    colour_dered_mean = np.mean(colour_dered_all)
    colour_dered_std = np.std(colour_dered_all)
    n_hosts = len(mag_obs)
    has_trgb_colour = "colour_dered" in data and "e_colour_dered" in data
    c_bar = _sample_or_default(
        samples, "c_bar", n_post,
        _prior_reference_value(config, "c_bar", colour_dered_mean))
    w_c = _sample_or_default(
        samples, "w_c", n_post,
        _prior_reference_value(config, "w_c",
                               max(float(colour_dered_std), 1e-3)))
    colour_model = {
        "has_colour": has_trgb_colour,
        "alpha_c": alpha_c,
        "c_bar": c_bar,
        "w_c": w_c,
        "e_colour": e_colour_dered_all,
    }

    if n_ppc is None:
        ppc_factor = get_nested(config, "model/ppc_factor", 10)
        n_ppc = ppc_factor * n_hosts

    # ---- Cosmography ----
    Om = get_nested(config, "model/Om", 0.3)
    dist = _FastDistanceConversions(Om0=Om)
    r2mu = dist.distmod
    r2z = dist.redshift

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
        colour_mean=c_bar if has_trgb_colour else None,
        c_star=c_star if has_trgb_colour else None,
        colour_std=w_c if has_trgb_colour else None,
        alpha_c=alpha_c if has_trgb_colour else 0.0)
    fprint(f"PPC: effective r_max = {r_max_eff:.1f} Mpc "
           f"(config r_max = {r_max:.1f} Mpc)")

    pool = None
    pool_label = None
    if not use_reconstruction:
        r_sphere = r_max_eff * float(np.max(H0) / 100)
        pool_size = max(n_ppc * 100, 500_000)
        pool = _uniform_density_pool(gen, r_sphere, pool_size)
        pool_label = "homogeneous"

    mag_sim, cz_sim, colour_sim = _ppc_field_path(
        gen, config, H0, M_TRGB, c_star, sigma_int, sigma_v, Vext,
        beta, bias, nu_cz, use_density_sigma_v, data, field_index,
        colour_model,
        which_sel, mag_min_fixed, mag_lim_samples, mag_lim_fixed,
        mag_width_samples, mag_width_fixed,
        e_mag_obs_all, e_czcmb_all,
        r_min, r_max_eff, r2mu, r2z, n_ppc, n_hosts, progress,
        pool=pool, pool_label=pool_label)

    out = {
        "mag_sim": mag_sim,
        "cz_sim": cz_sim,
        "mag_obs": mag_obs,
        "cz_obs": cz_obs,
    }
    if colour_sim is not None:
        out["colour_sim"] = colour_sim
        out["colour_obs"] = colour_dered_all
    return out


###############################################################################
#                    Field-based PPC path (3D density)                        #
###############################################################################


def _ppc_field_path(gen, config, H0, M_TRGB, c_star, sigma_int, sigma_v, Vext,
                    beta, bias, nu_cz, use_density_sigma_v, data,
                    field_index, colour_model,
                    which_sel, mag_min_fixed, mag_lim_samples, mag_lim_fixed,
                    mag_width_samples, mag_width_fixed,
                    e_mag_obs_all, e_czcmb_all,
                    r_min, r_max, r2mu, r2z, n_ppc, n_hosts, progress,
                    pool=None, pool_label=None):
    """PPC using a 3D pool; use rho=1 and v_los=0 for homogeneous PPC."""
    n_post = len(H0)
    if pool is None:
        field_name, field_config = _field_name_config(config)
        field_index = _selected_field_index(data, gen, field_index=field_index)
        try:
            pool = _cached_volume_pool(config, field_name, field_index)
            pool_label = f"field={field_name}[{field_index}], cached cut-out"
        except Exception as exc:
            fprint("PPC: cached cut-out unavailable; falling back to full "
                   f"field pool ({exc}).")
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
            pool_label = f"field={field_name}[{field_index}], full field pool"

    r_h_pool = pool["r_h"]
    rho_pool = pool["rho"]
    v_los_pool = pool["v_los"]
    rhat_icrs_pool = pool["rhat_icrs"]
    n_pool = len(r_h_pool)
    h_post = H0 / 100
    bias_upper = _bias_upper_bound(
        pool["rho_support"], bias, np.arange(n_post))
    base_cdf = None
    if pool["base_weight"] is not None:
        base_cdf = np.cumsum(pool["base_weight"])
        base_cdf[-1] = 1.0

    # Vectorized rejection-sampling loop
    collected_mag = []
    collected_cz = []
    collected_colour = []
    n_accepted = 0
    batch_size = max(int(20.0 * n_ppc), 10_000)

    fprint(f"PPC: generating {n_ppc} galaxies "
           f"(n_post={n_post}, n_hosts={n_hosts}, "
           f"pool={n_pool}, {pool_label})")

    with tqdm(total=n_ppc, desc="TRGB PPC", unit="gal",
              disable=not progress) as pbar:
        while n_accepted < n_ppc:
            n_need = n_ppc - n_accepted
            batch = min(batch_size, max(n_need * 3, 1000))

            # Draw posterior sample and pool indices
            idx_post = gen.integers(0, n_post, batch)
            if pool["base_weight"] is None:
                idx_pool = gen.integers(0, n_pool, batch)
            else:
                idx_pool = np.searchsorted(
                    base_cdf, gen.random(batch), side="right")

            # Pool values
            r_h = r_h_pool[idx_pool]
            rho = rho_pool[idx_pool]
            v_los = v_los_pool[idx_pool]
            rhat = rhat_icrs_pool[idx_pool]

            # Posterior values
            h = h_post[idx_post]
            M = M_TRGB[idx_post]
            cs = c_star[idx_post]
            if colour_model["has_colour"]:
                ac = colour_model["alpha_c"][idx_post]
                c0 = colour_model["c_bar"][idx_post]
                wc = colour_model["w_c"][idx_post]
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
            weight_max = bias_upper[idx_post]
            accept_bias = gen.random(batch) < np.minimum(
                weight / weight_max, 1.0)

            accept = in_range & accept_bias
            if not np.any(accept):
                continue

            # Apply mask
            idx_post_acc = idx_post[accept]
            r_Mpc = r_Mpc[accept]
            h_acc = h[accept]
            M_acc = M[accept]
            cs_acc = cs[accept]
            if colour_model["has_colour"]:
                ac_acc = ac[accept]
                c0_acc = c0[accept]
                wc_acc = wc[accept]
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

            # Measurement errors and colour population draw.
            e_mag = _sample_empirical(gen, e_mag_obs_all, n_batch)
            e_cz = _sample_empirical(gen, e_czcmb_all, n_batch)
            if colour_model["has_colour"]:
                e_colour = _sample_empirical(
                    gen, colour_model["e_colour"], n_batch)
                colour_true = gen.normal(
                    c0_acc, np.clip(wc_acc, 1e-6, None))
                colour_obs = gen.normal(colour_true, e_colour)
                colour_term = ac_acc * (colour_true - cs_acc)
            else:
                colour_obs = None
                colour_term = 0.0

            # Apparent magnitude with colour standardisation
            sigma_mag = np.sqrt(e_mag**2 + sint_acc**2)
            mag_sim = gen.normal(
                M_acc + colour_term + mu, sigma_mag)

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
                mag_min_fixed, mag_lim_samples, mag_lim_fixed,
                mag_width_samples, mag_width_fixed, gen)

            n_new = int(np.sum(accept_sel))
            collected_mag.append(mag_sim[accept_sel])
            collected_cz.append(cz_sim[accept_sel])
            if colour_obs is not None:
                collected_colour.append(colour_obs[accept_sel])
            n_accepted += n_new
            pbar.update(min(n_new, n_ppc - pbar.n))

    mag_sim = np.concatenate(collected_mag)[:n_ppc]
    cz_sim = np.concatenate(collected_cz)[:n_ppc]
    if collected_colour:
        colour_sim = np.concatenate(collected_colour)[:n_ppc]
    else:
        colour_sim = None
    fprint(f"PPC: generated {n_ppc} simulated galaxies ({pool_label}).")
    return mag_sim, cz_sim, colour_sim


###############################################################################
#                        Selection helper                                     #
###############################################################################


def _apply_selection_ppc(mag_sim, which_sel, idx_post,
                         mag_min_fixed, mag_lim_samples, mag_lim_fixed,
                         mag_width_samples, mag_width_fixed, gen):
    """Apply selection function, returning boolean mask."""
    n = len(mag_sim)
    if which_sel == "TRGB_magnitude":
        ml = mag_lim_samples[idx_post] \
            if mag_lim_samples is not None else mag_lim_fixed
        mw = mag_width_samples[idx_post] \
            if mag_width_samples is not None else mag_width_fixed
        p_sel = norm.cdf((ml - mag_sim) / mw)
        if mag_min_fixed is not None:
            p_sel -= norm.cdf((mag_min_fixed - mag_sim) / mw)
        p_sel = np.clip(p_sel, 0.0, 1.0)
        return gen.random(n) < p_sel
    return np.ones(n, dtype=bool)


###############################################################################
#                            PPC plotting                                     #
###############################################################################


def _hist_contour_levels(hist, enclosed=(0.68, 0.95)):
    """Return histogram levels enclosing the requested probability masses."""
    vals = np.asarray(hist, dtype=float).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return np.array([])

    vals = np.sort(vals)[::-1]
    cdf = np.cumsum(vals)
    cdf /= cdf[-1]
    levels = []
    for p in enclosed:
        idx = min(np.searchsorted(cdf, p), len(vals) - 1)
        levels.append(vals[idx])
    return np.unique(np.sort(levels))


def _plot_2d_contours(ax, x, y, bins, color, linestyle, label):
    """Plot smoothed 2D histogram contours for one sample."""
    from scipy.ndimage import gaussian_filter

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    hist = gaussian_filter(hist.astype(float), sigma=1.0)
    levels = _hist_contour_levels(hist)
    levels = levels[(levels > np.min(hist)) & (levels < np.max(hist))]
    if len(levels) == 0:
        return None

    xmid = 0.5 * (xedges[:-1] + xedges[1:])
    ymid = 0.5 * (yedges[:-1] + yedges[1:])
    ax.contour(xmid, ymid, hist.T, levels=levels, colors=color,
               linestyles=linestyle, linewidths=1.5)
    return ax.plot([], [], color=color, linestyle=linestyle,
                   linewidth=1.5, label=label)[0]


def plot_trgb_ppc(ppc, fname):
    """Plot 3-panel PPC comparison: 1D histograms and 2D contours.

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

    # Panel 3: 2D distribution contours
    ax = axes[2]
    bins_2d = (
        np.linspace(min(mag_obs.min(), mag_sim.min()) - 0.5,
                    max(mag_obs.max(), mag_sim.max()) + 0.5, 35),
        np.linspace(min(cz_obs.min(), cz_sim.min()) - 200,
                    max(cz_obs.max(), cz_sim.max()) + 200, 35),
    )
    handles = [
        _plot_2d_contours(ax, mag_sim, cz_sim, bins_2d, "C0", "-", "PPC"),
        _plot_2d_contours(ax, mag_obs, cz_obs, bins_2d, "k", "--",
                          "Observed"),
    ]
    handles = [h for h in handles if h is not None]
    ax.set_xlabel(r"$m_{\rm TRGB}$ [mag]")
    ax.set_ylabel(r"$cz_{\rm CMB}$ [km/s]")
    if handles:
        ax.legend(handles=handles, fontsize=8)

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
