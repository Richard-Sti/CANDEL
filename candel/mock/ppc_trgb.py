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
import contextlib
import io
import multiprocessing as mp
import os
import queue
import sys

import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import ks_2samp, norm
from tqdm.auto import tqdm

from ..field import name2field_loader
from ..pvdata.field_products import (
    field_smoothing_scale_from_config,
    velocity_field_smoothing_scale_from_config)
from ..pvdata.volume_density import _density_unit_normalization
from ..util import (SPEED_OF_LIGHT, fprint, galactic_to_radec,
                    galactic_to_radec_cartesian, get_nested, load_config,
                    radec_to_cartesian, radec_to_galactic)
from ._field_utils import (build_field_pool, build_field_pool_evaluator,
                           compute_r_max_selection,
                           galaxy_bias_params_from_values, galaxy_bias_weight)

_PPC_PROGRESS_QUEUE = None


def _flat(x):
    """Flatten scalar samples while preserving vector-valued samples."""
    x = np.asarray(x)
    if x.ndim > 1 and x.shape[-1] != 3:
        return x.reshape(-1)
    if x.ndim > 2:
        return x.reshape(-1, x.shape[-1])
    return x


def _flat_last_axis(x, width):
    """Flatten samples with a fixed-width trailing vector axis."""
    x = np.asarray(x)
    if x.ndim == 1:
        if x.shape[0] != width:
            raise ValueError(
                f"Expected trailing width {width}, got shape {x.shape}.")
        return x.reshape(1, width)
    if x.shape[-1] != width:
        raise ValueError(
            f"Expected trailing width {width}, got shape {x.shape}.")
    return x.reshape(-1, width)


def _sample_or_default(samples, key, n, default):
    if key in samples:
        return _flat(samples[key])
    return np.full(n, default)


def _sample_empirical(gen, values, size):
    """Draw from an empirical 1D array using integer indexing."""
    values = np.asarray(values)
    return values[gen.integers(0, len(values), size)]


def _positive_int_or_none(value):
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _runtime_cpu_count():
    for key in (
            "CANDEL_PPC_N_WORKERS", "HOST_DEVICES",
            "SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS",
            "OMP_NUM_THREADS"):
        value = _positive_int_or_none(os.environ.get(key))
        if value is not None:
            return value
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
            cpu_count = os.cpu_count()
            if cpu_count is not None and affinity < cpu_count:
                return affinity
        except OSError:
            pass
    return 1


def _split_counts(total, n_parts):
    q, r = divmod(int(total), int(n_parts))
    return [q + (i < r) for i in range(int(n_parts))]


def _same_ppc_value(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_same_ppc_value(a[k], b[k]) for k in a)
    try:
        aa = np.asarray(a)
        bb = np.asarray(b)
    except (TypeError, ValueError):
        return a == b
    if aa.dtype.kind == "f" or bb.dtype.kind == "f":
        return np.allclose(aa, bb)
    return np.array_equal(aa, bb)


def _merge_ppc_chunks(chunks):
    keys = sorted(set().union(*(chunk.keys() for chunk in chunks)))
    out = {}
    for key in keys:
        values = [chunk[key] for chunk in chunks if key in chunk]
        if len(values) != len(chunks):
            continue
        if key.endswith("_sim"):
            out[key] = np.concatenate([np.asarray(v) for v in values])
        elif key.endswith("_obs"):
            out[key] = np.asarray(values[0])
        elif all(_same_ppc_value(values[0], v) for v in values[1:]):
            out[key] = values[0]
        elif key == "sky_exposure":
            fprint("PPC warning: sky_exposure differs between worker "
                   "chunks; omitting aggregate exposure metadata.")
    return out


def _init_trgb_ppc_worker(progress_queue):
    global _PPC_PROGRESS_QUEUE
    _PPC_PROGRESS_QUEUE = progress_queue


def _generate_trgb_ppc_worker(kwargs):
    kwargs = dict(kwargs)
    task_id = kwargs.pop("_task_id")
    progress_queue = _PPC_PROGRESS_QUEUE
    if progress_queue is not None:
        kwargs["progress_callback"] = progress_queue.put
    with contextlib.redirect_stdout(io.StringIO()):
        return task_id, generate_trgb_ppc(**kwargs)


def _prior_reference_value(config, name, default):
    spec = get_nested(config, f"model/priors/{name}", None)
    if isinstance(spec, dict):
        for key in ("value", "loc", "mean"):
            if key in spec:
                return spec[key]
    return default


def _vext_monopole_kind(config):
    """Return the normalized Vext monopole mode from config."""
    mono = get_nested(config, "model/which_Vext_monopole", "none")
    if mono is True:
        mono = "constant"
    elif mono is False or mono is None:
        mono = "none"
    if mono == "none" and get_nested(config, "model/use_Vext_monopole", False):
        mono = "constant"
    return mono


def _check_vext_terms_supported(config):
    """Raise if the config requests Vext terms the PPC cannot reproduce.

    The PPC forward model supports the Vext dipole, a constant monopole, and
    an octupole. Sigmoid monopoles and quadrupoles still fail loudly instead
    of being silently dropped.
    """
    unsupported = []
    mono = _vext_monopole_kind(config)
    if mono not in ("none", "constant"):
        unsupported.append(f"which_Vext_monopole='{mono}'")
    if get_nested(config, "model/use_Vext_quadrupole", False):
        unsupported.append("use_Vext_quadrupole=True")
    if unsupported:
        raise NotImplementedError(
            "TRGB PPC only supports the Vext dipole, a constant monopole, "
            "and an octupole, but the config requests "
            + ", ".join(unsupported)
            + ". These Vext terms are not reproduced by the PPC "
            "forward model.")


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


def _vext_mono_samples(samples, config, n_post):
    """Return constant monopole velocity samples, or None."""
    mono = _vext_monopole_kind(config)
    if mono == "none":
        return None
    if mono != "constant":
        raise NotImplementedError(
            f"TRGB PPC does not support which_Vext_monopole={mono!r}.")
    return _sample_or_default(
        samples, "Vext_mono", n_post,
        _prior_reference_value(config, "Vext_mono", 0.0))


def _vext_oct_samples(samples, config):
    """Return octupole posterior samples as (mag, q1, q2, q3), or None."""
    if not get_nested(config, "model/use_Vext_octupole", False):
        return None

    required = (
        "Vext_oct_mag",
        "Vext_oct_q1_ell", "Vext_oct_q1_b",
        "Vext_oct_q2_ell", "Vext_oct_q2_b",
        "Vext_oct_q3_ell", "Vext_oct_q3_b",
    )
    missing = [key for key in required if key not in samples]
    if missing:
        raise ValueError(
            "PPC config requests `use_Vext_octupole=True`, but posterior "
            "samples are missing: " + ", ".join(missing))

    mag = _flat(samples["Vext_oct_mag"])
    q1 = galactic_to_radec_cartesian(
        _flat(samples["Vext_oct_q1_ell"]), _flat(samples["Vext_oct_q1_b"]))
    q2 = galactic_to_radec_cartesian(
        _flat(samples["Vext_oct_q2_ell"]), _flat(samples["Vext_oct_q2_b"]))
    q3 = galactic_to_radec_cartesian(
        _flat(samples["Vext_oct_q3_ell"]), _flat(samples["Vext_oct_q3_b"]))
    return mag, q1, q2, q3


def _octupole_radial(O_mag, q1_hat, q2_hat, q3_hat, rhat):
    """General symmetric traceless octupole radial velocity."""
    q1r = np.sum(q1_hat * rhat, axis=1)
    q2r = np.sum(q2_hat * rhat, axis=1)
    q3r = np.sum(q3_hat * rhat, axis=1)
    q1q2 = np.sum(q1_hat * q2_hat, axis=1)
    q1q3 = np.sum(q1_hat * q3_hat, axis=1)
    q2q3 = np.sum(q2_hat * q3_hat, axis=1)
    return O_mag * (
        q1r * q2r * q3r
        - (q1q2 * q3r + q1q3 * q2r + q2q3 * q1r) / 5)


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
    which_run = get_nested(config, "model/which_run", "EDD_TRGB")
    field_name = get_nested(
        config, f"io/PV_main/{which_run}/reconstruction", None)
    if field_name is None:
        raise ValueError(
            f"use_reconstruction=True but no reconstruction specified for "
            f"{which_run}.")
    field_config = dict(config["io"]["reconstruction_main"][field_name])
    return field_name, field_config


def _density_divisor(field_name):
    norm_info = _density_unit_normalization(field_name)
    if norm_info is None:
        return None
    return norm_info[0]


def _smoothing_label(scale):
    """Format an optional smoothing scale for status messages."""
    if scale is None:
        return "none"
    return f"{scale:g} Mpc/h"


def _posterior_param_label(name, values):
    """Format a compact posterior summary for a scalar parameter."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return f"{name}=nan"
    if values.size == 1 or np.allclose(values, values[0]):
        return f"{name}={values[0]:.3g}"
    q16, q50, q84 = np.percentile(values, [16, 50, 84])
    return f"{name}={q50:.3g} [{q16:.3g}, {q84:.3g}]"


def _bias_parameter_labels(bias):
    """Return printable summaries for the active galaxy-bias parameters."""
    which_bias = bias["which_bias"]
    if which_bias == "uniform":
        return ["none"]
    if which_bias == "unity":
        return ["fixed unity"]
    if which_bias in ("linear", "linear_from_beta",
                      "linear_from_beta_stochastic"):
        return [_posterior_param_label("b1", bias["b1"])]
    if which_bias == "powerlaw":
        return [_posterior_param_label("alpha", bias["alpha"])]
    if which_bias == "double_powerlaw":
        labels = [_posterior_param_label("alpha_low", bias["alpha_low"])]
        if "alpha_high" in bias:
            labels.append(_posterior_param_label(
                "alpha_high", bias["alpha_high"]))
        else:
            alpha_high = bias["alpha_low"] * bias["alpha_high_frac"]
            labels.append(_posterior_param_label(
                "alpha_high", alpha_high))
            labels.append(_posterior_param_label(
                "alpha_high_frac", bias["alpha_high_frac"]))
        labels.append(_posterior_param_label(
            "log_rho_t", bias["log_rho_t"]))
        labels.append(_posterior_param_label(
            "log_rho_width", bias["log_rho_width"]))
        return labels
    if which_bias == "quadratic":
        return [
            _posterior_param_label("b1", bias["b1"]),
            _posterior_param_label("b2", bias["b2"]),
        ]
    if which_bias == "cubic":
        return [
            _posterior_param_label("b1", bias["b1"]),
            _posterior_param_label("b2", bias["b2"]),
            _posterior_param_label("b3", bias["b3"]),
        ]
    return ["unknown"]


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
    if which_bias == "powerlaw":
        alpha = bias["alpha"][idx_post]
        rho_extreme = np.where(alpha >= 0.0, rho_max, rho_min)
        return _bias_values(rho_extreme, bias, idx_post)
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
    if which_bias == "quadratic":
        b1 = bias["b1"][idx_post]
        b2 = bias["b2"][idx_post]
        rho_vertex = np.full_like(b1, rho_min, dtype=float)
        nonzero = b2 != 0.0
        rho_vertex[nonzero] = 1.0 - b1[nonzero] / (2.0 * b2[nonzero])
        rho_vertex = np.clip(rho_vertex, rho_min, rho_max)
        rho_eval = np.vstack([
            np.full_like(b1, rho_min),
            rho_vertex,
            np.full_like(b1, rho_max),
        ])
        idx = idx_post[None, :]
        return np.max(_bias_values(rho_eval, bias, idx), axis=0)
    if which_bias == "cubic":
        b1 = bias["b1"][idx_post]
        b2 = bias["b2"][idx_post]
        b3 = bias["b3"][idx_post]
        delta_min = rho_min - 1.0
        root1 = np.full_like(b1, delta_min, dtype=float)
        root2 = np.full_like(b1, delta_min, dtype=float)
        nonzero = b3 != 0.0
        disc = b2**2 - 3.0 * b3 * b1
        valid = nonzero & (disc >= 0.0)
        sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))
        root1[valid] = (
            (-b2[valid] + sqrt_disc[valid]) / (3.0 * b3[valid]))
        root2[valid] = (
            (-b2[valid] - sqrt_disc[valid]) / (3.0 * b3[valid]))
        linear_valid = (~nonzero) & (b2 != 0.0)
        root1[linear_valid] = -b1[linear_valid] / (2.0 * b2[linear_valid])
        rho_eval = np.vstack([
            np.full_like(b1, rho_min),
            np.clip(1.0 + root1, rho_min, rho_max),
            np.clip(1.0 + root2, rho_min, rho_max),
            np.full_like(b1, rho_max),
        ])
        idx = idx_post[None, :]
        return np.max(_bias_values(rho_eval, bias, idx), axis=0)
    rho = rho_support[:, None]
    idx = idx_post[None, :]
    return np.max(_bias_values(rho, bias, idx), axis=0)


def _uniform_density_pool(gen, r_sphere, pool_size):
    """Build a unit-density, zero-velocity pool in Mpc/h coordinates."""
    r_h = r_sphere * gen.random(pool_size)**(1.0 / 3.0)
    phi = gen.uniform(0.0, 2.0 * np.pi, pool_size)
    sin_dec = gen.uniform(-1.0, 1.0, pool_size)
    cos_dec = np.sqrt(1.0 - sin_dec**2)
    rhat = np.column_stack([
        cos_dec * np.cos(phi),
        cos_dec * np.sin(phi),
        sin_dec,
    ])
    return {
        "r_h": r_h,
        "rho": np.ones(pool_size, dtype=np.float64),
        "v_los": np.zeros(pool_size, dtype=np.float64),
        "RA": np.rad2deg(phi),
        "dec": np.rad2deg(np.arcsin(sin_dec)),
        "rhat_icrs": rhat,
        "base_weight": None,
    }


def _validate_galactic_latitude_cut(b_min, source):
    """Validate an optional Galactic latitude cut in degrees."""
    if b_min is None:
        return None
    b_min = float(b_min)
    if b_min < 0.0 or b_min > 90.0:
        raise ValueError(f"`{source}` must be in [0, 90] deg.")
    if np.isclose(b_min, 0.0):
        return None
    return b_min


def _galactic_latitude_cut_from_config(config):
    """Read the PPC sky mask from the same config keys as the model."""
    which_run = get_nested(config, "model/which_run", None)
    b_min = None
    if which_run is not None:
        source = f"io/PV_main/{which_run}/b_min"
        b_min = get_nested(config, source, None)
        if b_min is not None:
            return _validate_galactic_latitude_cut(b_min, source)

    source = "model/selection_integral_b_min"
    b_min = get_nested(config, source, None)
    return _validate_galactic_latitude_cut(b_min, source)


def _sky_exposure_settings_from_config(config, nside=None, kappa=None):
    """Resolve PPC HEALPix exposure settings from arguments and config."""
    sky_config = get_nested(config, "model/TRGB_sky_exposure", {})
    if sky_config is None:
        sky_config = {}
    if not isinstance(sky_config, dict):
        sky_config = {}

    if sky_config.get("enabled", False) and "n_pix" in sky_config:
        raise ValueError(
            "`model/TRGB_sky_exposure/n_pix` is no longer supported; "
            "use `model/TRGB_sky_exposure/nside`.")

    if nside is None:
        if not sky_config.get("enabled", False):
            return 0, 48.0 if kappa is None else float(kappa)
        if "nside" not in sky_config:
            raise ValueError(
                "TRGB sky exposure requires "
                "`model/TRGB_sky_exposure/nside`.")
        nside = sky_config["nside"]

    nside = int(nside)
    if nside <= 0:
        return 0, 48.0 if kappa is None else float(kappa)

    nside = _validate_healpix_nside(nside)
    if kappa is None:
        kappa = sky_config.get("kappa", 48.0)
    kappa = float(kappa)
    if kappa <= 0:
        raise ValueError("`sky_exposure_kappa` must be positive.")
    return nside, kappa


def _apply_galactic_latitude_cut_to_pool(pool, b_min):
    """Keep only random sky positions satisfying |b| >= b_min."""
    if b_min is None or b_min <= 0.0:
        return pool
    _, b = radec_to_galactic(pool["RA"], pool["dec"])
    keep = np.abs(b) >= b_min
    n = len(pool["r_h"])
    if not np.any(keep):
        raise RuntimeError(
            f"PPC pool has no positions satisfying |b| >= {b_min:g} deg.")

    out = {}
    for key, value in pool.items():
        if value is None:
            out[key] = value
            continue
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            out[key] = arr[keep]
        else:
            out[key] = value
    return out

###############################################################################
#                         PPC generation                                      #
###############################################################################


def generate_trgb_ppc(samples, data, config, n_ppc=None, seed=42,
                      field_index=None, progress=True,
                      sky_exposure_nside=None, sky_exposure_kappa=None,
                      sky_exposure_posterior_mean=False,
                      sky_exposure_pilot_n=None, n_workers=None,
                      progress_callback=None):
    """Generate posterior predictive samples for the EDD TRGB model.

    When a reconstruction field is available, galaxies are sampled from
    the full 3D density field (matching the mock generator). Candidate
    positions are drawn in vectorised batches and accepted or rejected.

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
        Whether to show a tqdm progress bar for generated PPC galaxies.

    Returns
    -------
    dict with keys: mag_sim, cz_sim, r_sim, mag_obs, cz_obs.
    """
    if isinstance(config, str):
        config = load_config(config, replace_los_prior=False)
    _check_vext_terms_supported(config)
    sky_exposure_nside, sky_exposure_kappa = (
        _sky_exposure_settings_from_config(
            config, sky_exposure_nside, sky_exposure_kappa))
    gen = np.random.default_rng(seed)

    M_TRGB = _flat(samples["M_TRGB"])
    sigma_int = _flat(samples["sigma_int"])
    n_post = len(M_TRGB)
    H0 = _sample_or_default(
        samples, "H0", n_post, _prior_reference_value(config, "H0", 73.04))
    c_star = _sample_or_default(
        samples, "c_star", n_post,
        _prior_reference_value(config, "c_star", 1.23))
    alpha_c = _sample_or_default(
        samples, "alpha_c", n_post,
        _prior_reference_value(config, "alpha_c", 0.2))
    Vext = _vext_samples(samples, n_post)
    Vext_mono = _vext_mono_samples(samples, config, n_post)
    Vext_oct = _vext_oct_samples(samples, config)
    beta = _sample_or_default(
        samples, "beta", n_post,
        _prior_reference_value(config, "beta", 0.0))
    use_reconstruction = get_nested(config, "model/use_reconstruction", False)
    bias = (_bias_samples(samples, config, beta, n_post)
            if use_reconstruction else {"which_bias": "uniform"})
    if use_reconstruction:
        which_run = get_nested(config, "model/which_run", "EDD_TRGB")
        field_name = get_nested(
            config, f"io/PV_main/{which_run}/reconstruction", None)
        mas = None
        if field_name is not None:
            mas = get_nested(
                config,
                f"io/reconstruction_main/{field_name}/which_MAS",
                None)
        density_smoothing = _smoothing_label(
            field_smoothing_scale_from_config(config))
        velocity_smoothing = _smoothing_label(
            velocity_field_smoothing_scale_from_config(config))
        status = [
            f"reconstruction={field_name}",
            f"bias={bias['which_bias']}",
            f"density_smoothing={density_smoothing}",
            f"velocity_smoothing={velocity_smoothing}",
        ]
        if mas is not None:
            status.append(f"MAS={mas}")
    else:
        status = ["reconstruction=none", f"bias={bias['which_bias']}"]
    fprint("PPC config: " + ", ".join(status))
    fprint("PPC bias parameters: "
           + ", ".join(_bias_parameter_labels(bias)))

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
        sigma_v = _sample_or_default(
            samples, "sigma_v", n_post,
            _prior_reference_value(config, "sigma_v", 100.0))

    cz_likelihood = get_nested(config, "model/cz_likelihood", "gaussian")
    if cz_likelihood == "student_t":
        if "nu_cz" not in samples:
            raise ValueError(
                "PPC config requests Student-t cz likelihood, but posterior "
                "samples do not contain `nu_cz`.")
        nu_cz = _flat(samples["nu_cz"])
    else:
        nu_cz = None
        if "nu_cz" in samples:
            fprint("PPC warning: posterior contains `nu_cz`, but "
                   "model/cz_likelihood is not 'student_t'; using Gaussian "
                   "cz draws.")

    cz_labels = [f"likelihood={cz_likelihood}"]
    if use_reconstruction:
        cz_labels.append(_posterior_param_label("beta", beta))
    if use_density_sigma_v:
        cz_labels.extend([
            _posterior_param_label("sigma_v_low", sigma_v[0]),
            _posterior_param_label("sigma_v_high", sigma_v[1]),
            _posterior_param_label("log_sigma_v_rho_t", sigma_v[2]),
            _posterior_param_label("sigma_v_k", sigma_v[3]),
        ])
    else:
        cz_labels.append(_posterior_param_label("sigma_v", sigma_v))
    cz_labels.append(_posterior_param_label(
        "|Vext|", np.linalg.norm(Vext, axis=1)))
    if Vext_mono is not None:
        cz_labels.append(_posterior_param_label("Vext_mono", Vext_mono))
    if Vext_oct is not None:
        cz_labels.append(_posterior_param_label("Vext_oct_mag", Vext_oct[0]))
    if nu_cz is not None:
        cz_labels.append(_posterior_param_label("nu_cz", nu_cz))
    fprint("PPC cz parameters: " + ", ".join(cz_labels))

    # Selection parameters
    which_sel = get_nested(config, "model/which_selection", None)
    mag_lim_samples = _flat(samples["mag_lim_TRGB"]) \
        if "mag_lim_TRGB" in samples else None
    mag_width_samples = _flat(samples["mag_lim_TRGB_width"]) \
        if "mag_lim_TRGB_width" in samples else None
    mag_min_fixed = get_nested(config, "model/mag_min_TRGB", None)
    mag_lim_fixed = get_nested(config, "model/mag_lim_TRGB", None)
    mag_width_fixed = get_nested(config, "model/mag_lim_TRGB_width", None)
    cz_lim_samples = _flat(samples["cz_lim_selection"]) \
        if "cz_lim_selection" in samples else None
    cz_width_samples = _flat(samples["cz_lim_selection_width"]) \
        if "cz_lim_selection_width" in samples else None
    cz_lim_fixed = get_nested(config, "model/cz_lim_selection", None)
    cz_width_fixed = get_nested(config, "model/cz_lim_selection_width", None)

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
    n_ppc = int(n_ppc)

    if n_workers is None:
        n_workers = get_nested(config, "model/ppc_n_workers", None)
    if n_workers is None:
        n_workers = _runtime_cpu_count()
    n_workers = min(max(int(n_workers), 1), n_ppc)
    if n_workers > 1:
        selected_field_index = field_index
        if use_reconstruction and selected_field_index is None:
            selected_field_index = _selected_field_index(data, gen)
        counts = _split_counts(n_ppc, n_workers)
        fprint(f"PPC: splitting {n_ppc} galaxies into "
               f"{len(counts)} worker chunk(s) over {n_workers} worker(s).")
        worker_kwargs = [
            {
                "_task_id": i,
                "samples": samples,
                "data": data,
                "config": config,
                "n_ppc": count,
                "seed": seed + i,
                "field_index": selected_field_index,
                "progress": False,
                "sky_exposure_nside": sky_exposure_nside,
                "sky_exposure_kappa": sky_exposure_kappa,
                "sky_exposure_posterior_mean": (
                    sky_exposure_posterior_mean),
                "sky_exposure_pilot_n": (
                    None if sky_exposure_pilot_n is None
                    else max(1, int(round(
                        sky_exposure_pilot_n * count / n_ppc)))),
                "n_workers": 1,
            }
            for i, count in enumerate(counts)
        ]
        ctx = mp.get_context("spawn")
        progress_queue = ctx.Queue()
        with ctx.Pool(processes=n_workers,
                      initializer=_init_trgb_ppc_worker,
                      initargs=(progress_queue,)) as pool:
            async_results = [
                pool.apply_async(_generate_trgb_ppc_worker, (kwargs,))
                for kwargs in worker_kwargs
            ]
            pending = set(range(len(async_results)))
            results = []
            with tqdm(total=n_ppc, desc="TRGB PPC", unit="gal",
                      disable=not progress, file=sys.stdout,
                      miniters=1, leave=True) as pbar:
                while pending:
                    try:
                        n_new = progress_queue.get(timeout=0.2)
                        pbar.update(min(int(n_new), n_ppc - pbar.n))
                        continue
                    except queue.Empty:
                        pass
                    for i in list(pending):
                        if async_results[i].ready():
                            results.append(async_results[i].get())
                            pending.remove(i)
                while True:
                    try:
                        n_new = progress_queue.get_nowait()
                    except queue.Empty:
                        break
                    pbar.update(min(int(n_new), n_ppc - pbar.n))
        chunks = [chunk for _, chunk in sorted(results, key=lambda x: x[0])]
        return _merge_ppc_chunks(chunks)

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

    sky_exposure = None
    if sky_exposure_nside is not None and sky_exposure_nside > 0:
        if "RA_host" not in data or "dec_host" not in data:
            raise ValueError(
                "Sky-exposure PPC requires `RA_host` and `dec_host` in "
                "the observed data.")
        sky_exposure_nside = _validate_healpix_nside(sky_exposure_nside)
        sky_exposure_n_pix = int(hp.nside2npix(sky_exposure_nside))
        theta_samples = None
        theta_key = next((
            key for key in (
                "TRGB_sky_exposure_ratio",
                "TRGB_sky_exposure_theta_full",
                "TRGB_sky_exposure_theta")
            if key in samples), "TRGB_sky_exposure_theta")
        if theta_key in samples:
            theta_samples = _flat_last_axis(
                samples[theta_key], sky_exposure_n_pix)
            if len(theta_samples) == 1 and n_post > 1:
                theta_samples = np.repeat(theta_samples, n_post, axis=0)
            if len(theta_samples) != n_post:
                raise ValueError(
                    f"`{theta_key}` sample count does not match "
                    f"H0 samples: {len(theta_samples)} != {n_post}.")
            fprint(
                f"PPC: using posterior {theta_key} samples "
                "for the angular exposure.")
        baseline_key = "TRGB_sky_exposure_baseline_fraction"
        if theta_samples is not None:
            sky_exposure = sky_exposure_from_posterior_theta(
                theta_samples,
                data["RA_host"], data["dec_host"],
                nside=sky_exposure_nside,
                kappa=sky_exposure_kappa,
                posterior_mean=sky_exposure_posterior_mean,
                baseline_fraction=samples.get(baseline_key))
        else:
            n_pilot = (
                sky_exposure_pilot_n if sky_exposure_pilot_n is not None
                else n_ppc)
            fprint(
                "PPC: estimating angular selection integral for sky exposure "
                f"from {n_pilot} baseline selected galaxies.")
            pilot_gen = np.random.default_rng(seed + 1729)
            _, _, _, _, ra_pilot, dec_pilot = _ppc_field_path(
                pilot_gen, config, H0, M_TRGB, c_star, sigma_int, sigma_v,
                Vext, Vext_mono, Vext_oct,
                beta, bias, nu_cz, use_density_sigma_v, data, field_index,
                colour_model,
                which_sel, mag_min_fixed, mag_lim_samples, mag_lim_fixed,
                mag_width_samples, mag_width_fixed,
                e_mag_obs_all, e_czcmb_all,
                r_min, r_max_eff, r2mu, r2z, n_pilot, n_hosts, False,
                use_reconstruction=use_reconstruction,
                cz_lim_samples=cz_lim_samples, cz_lim_fixed=cz_lim_fixed,
                cz_width_samples=cz_width_samples,
                cz_width_fixed=cz_width_fixed)
            sky_exposure = fit_trgb_ppc_sky_exposure(
                ra_pilot, dec_pilot,
                data["RA_host"], data["dec_host"],
                nside=sky_exposure_nside,
                kappa=sky_exposure_kappa,
                gen=gen,
                posterior_mean=sky_exposure_posterior_mean)

    mag_sim, cz_sim, r_sim, colour_sim, ra_sim, dec_sim = _ppc_field_path(
        gen, config, H0, M_TRGB, c_star, sigma_int, sigma_v, Vext,
        Vext_mono, Vext_oct,
        beta, bias, nu_cz, use_density_sigma_v, data, field_index,
        colour_model,
        which_sel, mag_min_fixed, mag_lim_samples, mag_lim_fixed,
        mag_width_samples, mag_width_fixed,
        e_mag_obs_all, e_czcmb_all,
        r_min, r_max_eff, r2mu, r2z, n_ppc, n_hosts, progress,
        use_reconstruction=use_reconstruction,
        cz_lim_samples=cz_lim_samples, cz_lim_fixed=cz_lim_fixed,
        cz_width_samples=cz_width_samples, cz_width_fixed=cz_width_fixed,
        sky_exposure=sky_exposure,
        progress_callback=progress_callback)

    out = {
        "mag_sim": mag_sim,
        "cz_sim": cz_sim,
        "r_sim": r_sim,
        "ra_sim": ra_sim,
        "dec_sim": dec_sim,
        "mag_obs": mag_obs,
        "cz_obs": cz_obs,
    }
    if "RA_host" in data and "dec_host" in data:
        out["ra_obs"] = np.asarray(data["RA_host"])
        out["dec_obs"] = np.asarray(data["dec_host"])
    if colour_sim is not None:
        out["colour_sim"] = colour_sim
        out["colour_obs"] = colour_dered_all
    if sky_exposure is not None:
        out["sky_exposure"] = sky_exposure
    return out


###############################################################################
#                    Field-based PPC path (3D density)                        #
###############################################################################


def _ppc_field_path(gen, config, H0, M_TRGB, c_star, sigma_int, sigma_v, Vext,
                    Vext_mono, Vext_oct, beta, bias, nu_cz,
                    use_density_sigma_v, data, field_index, colour_model,
                    which_sel, mag_min_fixed, mag_lim_samples, mag_lim_fixed,
                    mag_width_samples, mag_width_fixed,
                    e_mag_obs_all, e_czcmb_all,
                    r_min, r_max, r2mu, r2z, n_ppc, n_hosts, progress,
                    use_reconstruction=True,
                    cz_lim_samples=None, cz_lim_fixed=None,
                    cz_width_samples=None, cz_width_fixed=None,
                    sky_exposure=None, progress_callback=None):
    """PPC using vectorised draw/accept batches."""
    n_post = len(H0)
    min_batch_size = 1000
    max_batch_size = 250_000
    field_loader = None
    field_evaluator = None
    rho_support = np.array([1.0], dtype=np.float64)
    b_min = _galactic_latitude_cut_from_config(config)
    if use_reconstruction:
        field_name, base_field_config = _field_name_config(config)
        selected_field_index = _selected_field_index(
            data, gen, field_index=field_index)
        candidate_label = f"field={field_name}[{selected_field_index}]"
    else:
        field_name = None
        base_field_config = None
        selected_field_index = None
        candidate_label = "homogeneous"
    if b_min is not None and b_min > 0.0:
        candidate_label += f", |b| >= {b_min:g} deg"
    if sky_exposure is not None:
        candidate_label += (
            f", sky exposure nside={sky_exposure['nside']}")

    def draw_candidates(n_candidate):
        nonlocal field_loader, field_evaluator, rho_support
        if not use_reconstruction:
            r_sphere = r_max * float(np.max(H0) / 100)
            candidates = _uniform_density_pool(gen, r_sphere, n_candidate)
            return _apply_galactic_latitude_cut_to_pool(candidates, b_min)

        first_candidate_batch = field_evaluator is None
        if first_candidate_batch:
            field_config = dict(base_field_config)
            field_config.setdefault("nsim", selected_field_index)
            field_loader = name2field_loader(field_name)(**field_config)
            r_sphere = r_max * (float(np.max(H0)) / 100)
            fprint("PPC: preparing field evaluator once for "
                   f"{field_name}[{selected_field_index}].")
            field_evaluator = build_field_pool_evaluator(
                field_loader,
                density_divisor=_density_divisor(field_name),
                field_smoothing_scale=field_smoothing_scale_from_config(
                    config),
                velocity_field_smoothing_scale=(
                    velocity_field_smoothing_scale_from_config(config)),
                max_radius_h=r_sphere)
            delta_max = field_evaluator.get(
                "delta_max_within_radius", field_evaluator["delta_max"])
            rho_support = np.array([
                float(field_evaluator["eps"]),
                max(float(field_evaluator["eps"]),
                    1.0,
                    1.0 + float(delta_max)),
            ], dtype=np.float64)
        r_sphere = r_max * (float(np.max(H0)) / 100)
        candidates = build_field_pool(
            field_loader, r_sphere, n_candidate, gen,
            density_divisor=_density_divisor(field_name),
            field_smoothing_scale=field_smoothing_scale_from_config(config),
            velocity_field_smoothing_scale=(
                velocity_field_smoothing_scale_from_config(config)),
            field_evaluator=field_evaluator,
            b_min=b_min,
            verbose=False)
        candidates["base_weight"] = None
        candidates = _apply_galactic_latitude_cut_to_pool(candidates, b_min)
        if first_candidate_batch:
            fprint("PPC: field evaluator ready; candidate batches reuse "
                   "the loaded field.")
        return candidates

    h_post = H0 / 100
    total_candidates = 0
    n_batches = 0

    # Vectorized rejection-sampling loop
    collected_mag = []
    collected_cz = []
    collected_r = []
    collected_colour = []
    collected_ra = []
    collected_dec = []
    n_accepted = 0
    # Adapt to low final acceptance after bias and selection rejection.
    acceptance_rate = 1.0e-3

    fprint(f"PPC: generating {n_ppc} galaxies "
           f"(n_post={n_post}, n_hosts={n_hosts}, "
           f"fresh candidate batches, {candidate_label})")

    with tqdm(total=n_ppc, desc="TRGB PPC", unit="gal",
              disable=not progress, file=sys.stdout, miniters=1,
              leave=True) as pbar:
        while n_accepted < n_ppc:
            n_need = n_ppc - n_accepted
            target_accepts = max(64, int(np.ceil(2.0 * n_need)))
            batch = int(np.ceil(
                target_accepts / max(acceptance_rate, 1.0e-5)))
            batch = min(max_batch_size, max(min_batch_size, batch))
            candidates = draw_candidates(batch)
            n_batches += 1

            # Draw posterior sample and pool indices
            batch = len(candidates["r_h"])
            total_candidates += batch
            idx_post = gen.integers(0, n_post, batch)

            # Candidate values
            r_h = candidates["r_h"]
            rho = candidates["rho"]
            v_los = candidates["v_los"]
            ra = candidates["RA"]
            dec = candidates["dec"]
            rhat = candidates["rhat_icrs"]

            # Posterior values needed for geometric and density rejection.
            h = h_post[idx_post]

            # Distance cut: r_Mpc = r_h / h
            r_Mpc = r_h / h
            in_range = (r_Mpc >= r_min) & (r_Mpc <= r_max)

            # Density rejection
            weight = _bias_values(rho, bias, idx_post)
            weight_max = _bias_upper_bound(rho_support, bias, idx_post)
            accept_bias = gen.random(batch) < np.minimum(
                weight / weight_max, 1.0)

            accept = in_range & accept_bias
            if sky_exposure is not None and np.any(accept):
                idx_accept = np.flatnonzero(accept)
                pix_accept = _sky_exposure_pixel_id(
                    ra[idx_accept], dec[idx_accept],
                    nside=sky_exposure["nside"])
                if "theta_samples" in sky_exposure:
                    idx_post_sky = idx_post[idx_accept]
                    rel_exposure = (
                        sky_exposure["theta_samples"][
                            idx_post_sky, pix_accept]
                        * sky_exposure["n_support"])
                    p_sky = (
                        rel_exposure
                        / sky_exposure[
                            "exposure_acceptance_norm_samples"][
                                idx_post_sky])
                else:
                    p_sky = (
                        sky_exposure["exposure_ratio"][pix_accept]
                        / sky_exposure["exposure_acceptance_norm"])
                keep_sky = gen.random(len(idx_accept)) < np.minimum(
                    p_sky, 1.0)
                accept_sky = np.zeros_like(accept)
                accept_sky[idx_accept[keep_sky]] = True
                accept = accept_sky

            if not np.any(accept):
                acceptance_rate *= 0.5
                continue

            # Apply mask
            idx_post_acc = idx_post[accept]
            r_Mpc = r_Mpc[accept]
            h_acc = h[accept]
            M_acc = M_TRGB[idx_post_acc]
            cs_acc = c_star[idx_post_acc]
            if colour_model["has_colour"]:
                ac_acc = colour_model["alpha_c"][idx_post_acc]
                c0_acc = colour_model["c_bar"][idx_post_acc]
                wc_acc = colour_model["w_c"][idx_post_acc]
            sint_acc = sigma_int[idx_post_acc]
            if use_density_sigma_v:
                sv_acc = _sigma_v_from_density(
                    rho[accept], sigma_v[0][idx_post_acc],
                    sigma_v[1][idx_post_acc],
                    sigma_v[2][idx_post_acc],
                    sigma_v[3][idx_post_acc])
            else:
                sv_acc = sigma_v[idx_post_acc]
            Vext_acc = Vext[idx_post_acc]
            Vext_mono_acc = (
                None if Vext_mono is None else Vext_mono[idx_post_acc])
            Vext_oct_acc = (
                None if Vext_oct is None
                else tuple(values[idx_post_acc] for values in Vext_oct))
            bt_acc = beta[idx_post_acc]
            v_los_acc = v_los[accept]
            ra_acc = ra[accept]
            dec_acc = dec[accept]
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
            if Vext_mono_acc is not None:
                Vpec = Vpec + Vext_mono_acc
            if Vext_oct_acc is not None:
                Vpec = Vpec + _octupole_radial(*Vext_oct_acc, rhat_acc)

            cz_pred = SPEED_OF_LIGHT * (
                (1 + z_cosmo) * (1 + Vpec / SPEED_OF_LIGHT) - 1)
            sigma_cz = np.sqrt(e_cz**2 + sv_acc**2)
            cz_sim = _draw_cz(gen, cz_pred, sigma_cz, nu=nu_acc)

            # Selection
            accept_sel = _apply_selection_ppc(
                mag_sim, which_sel, idx_post_acc,
                mag_min_fixed, mag_lim_samples, mag_lim_fixed,
                mag_width_samples, mag_width_fixed, gen,
                cz_sim=cz_sim, cz_lim_samples=cz_lim_samples,
                cz_lim_fixed=cz_lim_fixed, cz_width_samples=cz_width_samples,
                cz_width_fixed=cz_width_fixed)

            n_new = int(np.sum(accept_sel))
            n_report = min(n_new, n_ppc - n_accepted)
            collected_mag.append(mag_sim[accept_sel])
            collected_cz.append(cz_sim[accept_sel])
            collected_r.append(r_Mpc[accept_sel])
            collected_ra.append(ra_acc[accept_sel])
            collected_dec.append(dec_acc[accept_sel])
            if colour_obs is not None:
                collected_colour.append(colour_obs[accept_sel])
            n_accepted += n_new
            batch_acceptance_rate = n_new / batch
            acceptance_rate = (
                0.5 * acceptance_rate + 0.5 * batch_acceptance_rate)
            if progress_callback is not None and n_report > 0:
                progress_callback(n_report)
            else:
                pbar.update(n_report)

    mag_sim = np.concatenate(collected_mag)[:n_ppc]
    cz_sim = np.concatenate(collected_cz)[:n_ppc]
    r_sim = np.concatenate(collected_r)[:n_ppc]
    ra_sim = np.concatenate(collected_ra)[:n_ppc]
    dec_sim = np.concatenate(collected_dec)[:n_ppc]
    if collected_colour:
        colour_sim = np.concatenate(collected_colour)[:n_ppc]
    else:
        colour_sim = None
    fprint(f"PPC: generated {n_ppc} simulated galaxies "
           f"from {total_candidates} candidate draws "
           f"across {n_batches} batch(es) ({candidate_label}).")
    return mag_sim, cz_sim, r_sim, colour_sim, ra_sim, dec_sim


###############################################################################
#                       HEALPix angular exposure helper                       #
###############################################################################


def _validate_healpix_nside(nside):
    """Return a validated HEALPix nside value."""
    nside = int(nside)
    if nside <= 0 or not hp.isnsideok(nside):
        raise ValueError("`nside` must be a positive HEALPix nside value.")
    return nside


def _sky_exposure_pixel_id(ra, dec, nside=1):
    """Assign ICRS sky positions to HEALPix sky-exposure pixels."""
    nside = _validate_healpix_nside(nside)
    ell, b = radec_to_galactic(ra, dec)
    ell = np.nan_to_num(np.asarray(ell), nan=0.0, posinf=0.0,
                        neginf=0.0)
    b = np.nan_to_num(np.asarray(b), nan=0.0, posinf=90.0,
                      neginf=-90.0)
    ell = np.remainder(ell, 360.0)
    b = np.clip(b, -90.0, 90.0)
    theta = np.clip(0.5 * np.pi - np.deg2rad(b), 0.0, np.pi)
    phi = np.deg2rad(ell)
    return hp.ang2pix(nside, theta, phi, nest=False).astype(np.int32)


def sky_exposure_from_posterior_theta(theta_samples, ra_obs, dec_obs, *,
                                      nside=1, kappa=48.0,
                                      posterior_mean=False,
                                      baseline_fraction=None):
    """Build the PPC sky-exposure accept/reject model from posterior theta."""
    if kappa <= 0:
        raise ValueError("`kappa` must be positive.")
    nside = _validate_healpix_nside(nside)
    n_pix = int(hp.nside2npix(nside))
    theta_chain = _flat_last_axis(theta_samples, n_pix)
    theta_support = np.any(theta_chain > 0.0, axis=0)

    if baseline_fraction is not None:
        q = np.mean(_flat_last_axis(baseline_fraction, n_pix), axis=0)
        q = np.clip(q, 0.0, None)
        support = theta_support & (q > 0.0)
        q[~support] = 0.0
        q_sum = np.sum(q)
        if q_sum <= 0.0:
            raise ValueError(
                "Posterior sky-exposure baseline has empty supported mass.")
        q = q / q_sum
    else:
        support = theta_support
        n_support = int(np.sum(support))
        if n_support == 0:
            raise ValueError("Posterior sky-exposure theta has empty support.")
        q = np.zeros(n_pix, dtype=float)
        q[support] = 1.0 / n_support

    pix_obs = _sky_exposure_pixel_id(ra_obs, dec_obs, nside=nside)
    obs_counts = np.bincount(pix_obs, minlength=n_pix).astype(float)
    if np.any((~support) & (obs_counts > 0.0)):
        empty = np.flatnonzero((~support) & (obs_counts > 0.0))
        raise RuntimeError(
            "Observed TRGB host falls in a sky-exposure pixel with no "
            "posterior theta support: "
            + ", ".join(str(int(i)) for i in empty))

    n_support = int(np.sum(support))
    if n_support == 0:
        raise ValueError("Posterior sky-exposure theta has empty support.")

    theta_chain = np.array(theta_chain, dtype=float, copy=True)
    theta_chain[:, ~support] = 0.0
    theta_sums = np.sum(theta_chain, axis=1)
    if np.any(theta_sums <= 0.0):
        raise ValueError(
            "Posterior sky-exposure theta has samples outside supported mass.")
    theta_chain = theta_chain / theta_sums[:, None]
    theta = np.mean(theta_chain, axis=0)
    exposure_ratio = theta * n_support
    out = {
        "n_pix": int(n_pix),
        "n_support": int(n_support),
        "nside": int(nside),
        "kappa": float(kappa),
        "posterior_mean": bool(posterior_mean),
        "baseline_fraction": q,
        "observed_counts": obs_counts.astype(int),
        "baseline_counts": np.zeros(n_pix, dtype=int),
        "theta": theta,
        "exposure_ratio": exposure_ratio,
        "exposure_acceptance_norm": float(np.max(exposure_ratio)),
    }
    if not posterior_mean:
        out["theta_samples"] = theta_chain
        out["exposure_acceptance_norm_samples"] = np.max(
            theta_chain * n_support, axis=1)
    fprint(
        "PPC HEALPix true sky exposure forward model: "
        f"nside={nside}, n_pix={n_pix}, kappa={kappa:g}, "
        f"relative exposure range=[{np.min(exposure_ratio):.3g}, "
        f"{np.max(exposure_ratio):.3g}].")
    return out


def fit_trgb_ppc_sky_exposure(ra_baseline, dec_baseline, ra_obs, dec_obs, *,
                              nside=1, kappa=48.0, gen=None,
                              posterior_mean=False, theta_samples=None):
    """Fit a HEALPix Dirichlet angular exposure for the PPC forward model.

    The baseline pixel masses are Monte Carlo estimates of the selected-source
    angular selection integral. The final PPC samples candidates with an
    extra angular acceptance probability proportional to the exposure
    ``theta_k``.
    """
    if kappa <= 0:
        raise ValueError("`kappa` must be positive.")
    nside = _validate_healpix_nside(nside)
    n_pix = int(hp.nside2npix(nside))

    if gen is None:
        gen = np.random.default_rng()
    pix_sim = _sky_exposure_pixel_id(
        ra_baseline, dec_baseline, nside=nside)
    pix_obs = _sky_exposure_pixel_id(ra_obs, dec_obs, nside=nside)

    sim_counts = np.bincount(pix_sim, minlength=n_pix).astype(float)
    obs_counts = np.bincount(pix_obs, minlength=n_pix).astype(float)
    support = sim_counts > 0.0
    if np.any((~support) & (obs_counts > 0.0)):
        empty = np.flatnonzero((~support) & (obs_counts > 0.0))
        raise RuntimeError(
            "Cannot fit a sky exposure correction because the baseline PPC "
            "has no simulated galaxies in observed sky pixel(s): "
            + ", ".join(str(int(i)) for i in empty))
    support_idx = np.flatnonzero(support)
    n_support = len(support_idx)
    if n_support == 0:
        raise ValueError("Baseline PPC sky exposure has empty support.")

    q = np.zeros(n_pix, dtype=float)
    q[support] = sim_counts[support] / np.sum(sim_counts[support])
    alpha_prior = np.full(n_support, kappa / n_support, dtype=float)

    theta_chain = None
    importance_ess = None
    if theta_samples is not None:
        theta_chain = np.array(
            _flat_last_axis(theta_samples, n_pix), dtype=float, copy=True)
        theta_chain[:, ~support] = 0.0
        theta_sums = np.sum(theta_chain, axis=1)
        if np.any(theta_sums <= 0.0):
            raise ValueError(
                "Posterior sky-exposure theta has samples outside supported "
                "baseline mass.")
        theta_chain = theta_chain / theta_sums[:, None]
        theta = np.mean(theta_chain, axis=0)
    else:
        # True-mask posterior for observed sky pixels is proportional to
        # Dirichlet(alpha_prior) * prod_k theta_k^n_k / (q dot theta)^N.
        # Use the conjugate numerator as a proposal and importance-resample
        # the missing denominator term.
        n_importance = 8192
        proposal_alpha = alpha_prior + obs_counts[support]
        theta_prop_support = gen.dirichlet(
            proposal_alpha, size=n_importance)
        theta_prop = np.zeros((n_importance, n_pix), dtype=float)
        theta_prop[:, support_idx] = theta_prop_support
        q_dot_theta = np.clip(theta_prop @ q, 1.0e-300, None)
        logw = -np.sum(obs_counts) * np.log(q_dot_theta)
        logw = logw - np.max(logw)
        w = np.exp(logw)
        w = w / np.sum(w)
        importance_ess = float(1.0 / np.sum(w**2))
        if posterior_mean:
            theta = np.sum(theta_prop * w[:, None], axis=0)
        else:
            theta = theta_prop[gen.choice(n_importance, p=w)]

    theta = np.asarray(theta, dtype=float)
    theta = theta / np.sum(theta)
    exposure_ratio = theta * n_support
    exposure_acceptance_norm = float(np.max(exposure_ratio))
    out = {
        "n_pix": int(n_pix),
        "n_support": int(n_support),
        "nside": int(nside),
        "kappa": float(kappa),
        "posterior_mean": bool(posterior_mean),
        "baseline_fraction": q,
        "observed_counts": obs_counts.astype(int),
        "baseline_counts": sim_counts.astype(int),
        "theta": theta,
        "exposure_ratio": exposure_ratio,
        "exposure_acceptance_norm": exposure_acceptance_norm,
    }
    if theta_chain is not None and not posterior_mean:
        out["theta_samples"] = theta_chain
        out["exposure_acceptance_norm_samples"] = np.max(
            theta_chain * n_support, axis=1)
    if importance_ess is not None:
        out["importance_ess"] = importance_ess
    fprint(
        "PPC HEALPix true sky exposure forward model: "
        f"nside={nside}, n_pix={n_pix}, kappa={kappa:g}, "
        f"relative exposure range=[{np.min(exposure_ratio):.3g}, "
        f"{np.max(exposure_ratio):.3g}].")
    return out


###############################################################################
#                        Selection helper                                     #
###############################################################################


def _apply_selection_ppc(mag_sim, which_sel, idx_post,
                         mag_min_fixed, mag_lim_samples, mag_lim_fixed,
                         mag_width_samples, mag_width_fixed, gen,
                         cz_sim=None, cz_lim_samples=None, cz_lim_fixed=None,
                         cz_width_samples=None, cz_width_fixed=None):
    """Apply selection function, returning boolean mask."""
    n = len(mag_sim)
    if which_sel in ("TRGB_magnitude", "TRGB_magnitude_redshift"):
        ml = mag_lim_samples[idx_post] \
            if mag_lim_samples is not None else mag_lim_fixed
        mw = mag_width_samples[idx_post] \
            if mag_width_samples is not None else mag_width_fixed
        p_sel = norm.cdf((ml - mag_sim) / mw)
        if mag_min_fixed is not None:
            p_sel -= norm.cdf((mag_min_fixed - mag_sim) / mw)
        if which_sel == "TRGB_magnitude_redshift":
            cl = cz_lim_samples[idx_post] \
                if cz_lim_samples is not None else cz_lim_fixed
            cw = cz_width_samples[idx_post] \
                if cz_width_samples is not None else cz_width_fixed
            if cl is not None and cw is not None:
                p_sel = p_sel * norm.cdf((cl - cz_sim) / cw)
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


def _format_ppc_pvalue(pvalue):
    """Format a KS-test p-value for compact plot annotations."""
    if pvalue < 1e-3:
        return r"p<10^{-3}"
    return f"p={pvalue:.3f}"


def _plot_trgb_ppc_mnras(ppc, fname, ks_mag, ks_cz):
    """Plot a two-column MNRAS-style PPC figure."""
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    mag_sim = ppc["mag_sim"]
    cz_sim = ppc["cz_sim"]
    mag_obs = ppc["mag_obs"]
    cz_obs = ppc["cz_obs"]

    sim_color = "#2C7FB8"
    obs_color = "0.08"
    rc = {
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
    }

    with plt.style.context("science"):
        with plt.rc_context(rc):
            fig = plt.figure(figsize=(7.09, 4.35),
                             constrained_layout=True)
            gs = fig.add_gridspec(2, 2, height_ratios=(0.9, 1.25))
            ax_mag = fig.add_subplot(gs[0, 0])
            ax_cz = fig.add_subplot(gs[0, 1])
            ax_joint = fig.add_subplot(gs[1, :])

            bins_mag = np.linspace(
                min(mag_obs.min(), mag_sim.min()) - 0.5,
                max(mag_obs.max(), mag_sim.max()) + 0.5,
                36)
            ax_mag.hist(mag_sim, bins=bins_mag, density=True,
                        histtype="stepfilled", color=sim_color, alpha=0.22,
                        edgecolor=sim_color, linewidth=0.9, label="PPC")
            ax_mag.hist(mag_obs, bins=bins_mag, density=True,
                        histtype="step", color=obs_color, linewidth=1.0,
                        label="Observed")
            ax_mag.set_xlabel(r"$m_{\rm TRGB}$ [mag]")
            ax_mag.set_ylabel("Probability density")
            ax_mag.text(
                0.04, 0.94,
                rf"KS ${_format_ppc_pvalue(ks_mag.pvalue)}$",
                transform=ax_mag.transAxes, ha="left", va="top")
            ax_mag.legend(frameon=False, loc="upper right")

            bins_cz = np.linspace(
                min(cz_obs.min(), cz_sim.min()) - 200,
                max(cz_obs.max(), cz_sim.max()) + 200,
                36)
            ax_cz.hist(cz_sim, bins=bins_cz, density=True,
                       histtype="stepfilled", color=sim_color, alpha=0.22,
                       edgecolor=sim_color, linewidth=0.9, label="PPC")
            ax_cz.hist(cz_obs, bins=bins_cz, density=True, histtype="step",
                       color=obs_color, linewidth=1.0, label="Observed")
            ax_cz.set_xlabel(r"$cz_{\rm CMB}$ [km s$^{-1}$]")
            ax_cz.set_ylabel("Probability density")
            ax_cz.text(
                0.04, 0.94,
                rf"KS ${_format_ppc_pvalue(ks_cz.pvalue)}$",
                transform=ax_cz.transAxes, ha="left", va="top")

            bins_2d = (
                np.linspace(min(mag_obs.min(), mag_sim.min()) - 0.5,
                            max(mag_obs.max(), mag_sim.max()) + 0.5, 34),
                np.linspace(min(cz_obs.min(), cz_sim.min()) - 200,
                            max(cz_obs.max(), cz_sim.max()) + 200, 34),
            )
            handles = [
                _plot_2d_contours(
                    ax_joint, mag_sim, cz_sim, bins_2d, sim_color, "-",
                    "PPC"),
            ]
            obs_handle = ax_joint.scatter(
                mag_obs, cz_obs, s=12, marker="o", facecolor="white",
                edgecolor=obs_color, linewidth=0.6, zorder=3,
                label="Observed")
            handles = [h for h in handles if h is not None]
            ax_joint.set_xlabel(r"$m_{\rm TRGB}$ [mag]")
            ax_joint.set_ylabel(r"$cz_{\rm CMB}$ [km s$^{-1}$]")
            handles.append(obs_handle)
            if handles:
                labels = [handle.get_label() for handle in handles]
                ax_joint.legend(handles, labels, frameon=False,
                                loc="upper left")

            for ax in (ax_mag, ax_cz, ax_joint):
                ax.tick_params(direction="in", which="both", top=True,
                               right=True)

            fig.savefig(fname, dpi=500, bbox_inches="tight")
            plt.close(fig)


def plot_trgb_ppc(ppc, fname, *, mnras=False):
    """Plot 3-panel PPC comparison: 1D histograms and 2D contours.

    Parameters
    ----------
    ppc : dict
        Output of ``generate_trgb_ppc``.
    fname : str
        Output filename for the figure.
    mnras : bool
        If true, use a two-column MNRAS-style layout with the SciencePlots
        ``science`` style.
    """
    import matplotlib.pyplot as plt

    mag_sim = ppc["mag_sim"]
    cz_sim = ppc["cz_sim"]
    mag_obs = ppc["mag_obs"]
    cz_obs = ppc["cz_obs"]

    ks_mag = ks_2samp(mag_obs, mag_sim)
    ks_cz = ks_2samp(cz_obs, cz_sim)

    if mnras:
        _plot_trgb_ppc_mnras(ppc, fname, ks_mag, ks_cz)
        fprint(f"PPC plot saved to {fname}")
        return {
            "ks_mag_statistic": float(ks_mag.statistic),
            "ks_mag_pvalue": float(ks_mag.pvalue),
            "ks_cz_statistic": float(ks_cz.statistic),
            "ks_cz_pvalue": float(ks_cz.pvalue),
        }

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.35))
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, :]),
    ]

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
    ]
    handles = [h for h in handles if h is not None]
    obs_handle = ax.scatter(
        mag_obs, cz_obs, s=12, marker="o", facecolor="white",
        edgecolor="k", linewidth=0.6, zorder=3, label="Observed")
    handles.append(obs_handle)
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


def _wrap_mollweide_longitude(l_deg):
    """Return Galactic longitude in radians, centred on l=0 deg."""
    lon = np.remainder(np.asarray(l_deg) + 180.0, 360.0) - 180.0
    return -np.deg2rad(lon)


def _format_mollweide_axes(ax, axis_fontsize, tick_fontsize):
    """Apply the TRGB sky-distribution axis formatting."""
    ax.grid(alpha=0.45)
    ax.set_xlabel(r"$\ell$", fontsize=axis_fontsize)
    ax.set_ylabel(r"$b$", fontsize=axis_fontsize)
    ax.set_xticks(np.deg2rad([0, 120, -120]))
    ax.set_xticklabels([])
    for x, label in zip(
            np.deg2rad([0, 120, -120]),
            [r"$0^\circ$", r"$240^\circ$", r"$120^\circ$"]):
        ax.text(
            x, np.deg2rad(-6), label, ha="center", va="top",
            fontsize=tick_fontsize, color="black")
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
    ax.set_yticklabels([
        r"$-60^\circ$", r"$-30^\circ$", r"$0^\circ$",
        r"$30^\circ$", r"$60^\circ$"])
    ax.tick_params(labelsize=tick_fontsize)
    for label in ax.get_yticklabels():
        label.set_x(0.012)


def plot_trgb_ppc_sky(ppc, fname, *, mnras=False):
    """Plot the observed and retained PPC sky distributions."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter
    if mnras:
        import scienceplots  # noqa: F401

    required = ("ra_sim", "dec_sim", "ra_obs", "dec_obs")
    missing = [key for key in required if key not in ppc]
    if missing:
        raise ValueError(
            "PPC sky plot requires coordinates missing from `ppc`: "
            + ", ".join(missing))

    ell_sim, b_sim = radec_to_galactic(ppc["ra_sim"], ppc["dec_sim"])
    ell_obs, b_obs = radec_to_galactic(ppc["ra_obs"], ppc["dec_obs"])
    lon_sim = _wrap_mollweide_longitude(ell_sim)
    lat_sim = np.deg2rad(b_sim)
    lon_obs = _wrap_mollweide_longitude(ell_obs)
    lat_obs = np.deg2rad(b_obs)

    context = plt.style.context("science") if mnras else plt.style.context([])
    rc = {
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    } if mnras else {}
    figsize = (3.42, 2.45) if mnras else (7.0, 4.6)
    dpi = 500 if mnras else 200
    sim_size = 3.0 if mnras else 7.0
    obs_size = 9.0 if mnras else 18.0
    axis_fontsize = 7.5 if mnras else 10.0
    tick_fontsize = 6.5 if mnras else 8.0
    n_lon = 96 if mnras else 120
    n_lat = 48 if mnras else 60

    lon_edges = np.linspace(-np.pi, np.pi, n_lon + 1)
    lat_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_lat + 1)
    counts, _, _ = np.histogram2d(
        lon_sim, lat_sim, bins=(lon_edges, lat_edges))
    solid_angle = (
        np.diff(lon_edges)[:, None]
        * np.diff(np.sin(lat_edges))[None, :])
    density = np.divide(
        counts, solid_angle, out=np.zeros_like(counts, dtype=float),
        where=solid_angle > 0)
    density = gaussian_filter(density, sigma=(1.2, 0.9),
                              mode=("wrap", "nearest"))
    positive_density = density[density > 0]
    if len(positive_density):
        scale = np.percentile(positive_density, 99.0)
        density = np.clip(density / scale, 0.0, 1.0)
    lon_bin = np.clip(
        np.searchsorted(lon_edges, lon_sim, side="right") - 1,
        0, n_lon - 1)
    lat_bin = np.clip(
        np.searchsorted(lat_edges, lat_sim, side="right") - 1,
        0, n_lat - 1)
    point_density = density[lon_bin, lat_bin]
    point_order = np.argsort(point_density)
    cmap = plt.get_cmap("viridis")
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0)

    with context:
        with plt.rc_context(rc):
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            ax = fig.add_subplot(111, projection="mollweide")
            ax.scatter(
                lon_sim[point_order], lat_sim[point_order],
                c=point_density[point_order], s=sim_size, cmap=cmap,
                norm=norm, alpha=0.55, linewidth=0, rasterized=True,
                zorder=1)
            ax.scatter(
                lon_obs, lat_obs, s=obs_size, marker="o",
                facecolor="white", edgecolor="0.08", linewidth=0.45,
                alpha=0.90, zorder=3)
            _format_mollweide_axes(ax, axis_fontsize, tick_fontsize)
            legend_handles = [
                Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor=plt.get_cmap("viridis")(0.75),
                       markeredgecolor="none", markersize=4.0,
                       label="PPC density"),
                Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor="white", markeredgecolor="0.08",
                       markeredgewidth=0.6, markersize=4.0,
                       label="Observed"),
            ]
            ax.legend(handles=legend_handles, frameon=False,
                      loc="lower center", bbox_to_anchor=(0.5, 1.02),
                      ncol=2, handletextpad=0.4, columnspacing=1.2)
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    fprint(f"PPC sky plot saved to {fname}")
    return {
        "n_ppc": int(len(lon_sim)),
        "n_obs": int(len(lon_obs)),
    }


def plot_trgb_ppc_sky_exposure(ppc, fname, *, mnras=False):
    """Plot the coarse angular-exposure correction inferred for the PPC."""
    import matplotlib.pyplot as plt
    if mnras:
        import scienceplots  # noqa: F401

    if "sky_exposure" not in ppc:
        raise ValueError("PPC does not contain a `sky_exposure` correction.")
    exposure = ppc["sky_exposure"]
    ratio = np.asarray(exposure["exposure_ratio"], dtype=float)
    baseline = np.asarray(exposure["baseline_fraction"], dtype=float)
    observed = np.asarray(exposure["observed_counts"], dtype=float)
    observed_fraction = observed / np.sum(observed)
    x = np.arange(len(ratio))

    context = plt.style.context("science") if mnras else plt.style.context([])
    rc = {
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    } if mnras else {}
    figsize = (3.42, 2.75) if mnras else (7.0, 5.0)
    width = 0.38

    with context:
        with plt.rc_context(rc):
            fig, (ax_frac, ax_ratio) = plt.subplots(
                2, 1, figsize=figsize, sharex=True,
                gridspec_kw={"height_ratios": (1.0, 1.05)})
            ax_frac.bar(
                x - 0.5 * width, baseline, width=width,
                color="#2C7FB8", alpha=0.55, label="PPC baseline")
            ax_frac.bar(
                x + 0.5 * width, observed_fraction, width=width,
                color="0.15", alpha=0.55, label="Observed")
            ax_frac.set_ylabel("Fraction")
            ax_frac.legend(frameon=False, ncol=2, loc="upper right")

            ax_ratio.axhline(1.0, color="0.2", linewidth=0.7)
            ax_ratio.bar(x, ratio, color="#31A354", alpha=0.75)
            ax_ratio.set_xlabel("Sky pixel")
            ax_ratio.set_ylabel("Relative exposure")
            ax_ratio.set_xticks(x)
            ax_ratio.set_xticklabels([str(i) for i in x])
            for ax in (ax_frac, ax_ratio):
                ax.tick_params(direction="in", which="both", top=True,
                               right=True)
            fig.tight_layout()
            fig.savefig(fname, dpi=500 if mnras else 200,
                        bbox_inches="tight")
            plt.close(fig)

    fprint(f"PPC sky exposure plot saved to {fname}")
    return {
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
    }


def plot_trgb_ppc_distance(ppc, fname, *, mnras=False):
    """Plot the distance distribution of retained PPC galaxies."""
    import matplotlib.pyplot as plt
    if mnras:
        import scienceplots  # noqa: F401

    r_sim = np.asarray(ppc["r_sim"])
    q16, q50, q84 = np.percentile(r_sim, [16, 50, 84])
    bins = np.linspace(0.0, max(r_sim.max() * 1.05, q84 * 1.2), 36)

    context = plt.style.context("science") if mnras else plt.style.context([])
    rc = {
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    } if mnras else {}

    with context:
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(3.42, 2.35))
            ax.hist(r_sim, bins=bins, density=True, histtype="stepfilled",
                    color="#2C7FB8", alpha=0.24, edgecolor="#2C7FB8",
                    linewidth=0.9)
            ax.axvline(q50, color="0.08", linewidth=0.9,
                       label=rf"median ${q50:.1f}$ Mpc")
            ax.axvspan(q16, q84, color="0.08", alpha=0.10,
                       label=r"$16$--$84\%$")
            ax.set_xlabel(r"Distance [Mpc]")
            ax.set_ylabel("Probability density")
            ax.legend(frameon=False, loc="upper left")
            ax.tick_params(direction="in", which="both", top=True,
                           right=True)
            fig.tight_layout()
            fig.savefig(fname, dpi=500 if mnras else 200,
                        bbox_inches="tight")
            plt.close(fig)

    fprint(f"PPC distance plot saved to {fname}")
    return {
        "r_median": float(q50),
        "r_p16": float(q16),
        "r_p84": float(q84),
    }
