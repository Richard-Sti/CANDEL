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
"""Various utility functions for candel."""

try:
    # Python 3.11+
    import tomllib  # noqa
except ModuleNotFoundError:
    # Backport for <=3.10
    import tomli as tomllib

from datetime import datetime
from os.path import abspath, isabs, join

import astropy.units as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa
from astropy.coordinates import SkyCoord
from corner import corner
from jax import vmap
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

SPEED_OF_LIGHT = 299_792.458  # km / s


def fprint(*args, verbose=True, **kwargs):
    """Prints a message with a timestamp prepended."""
    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S%f")[:-6]
        print(f"{timestamp}", *args, **kwargs)


def convert_none_strings(d):
    """
    Convert all string values in a dictionary to None if they are equal to
    "none" (case insensitive). This is useful for parsing TOML files where
    "none" is used to represent None values.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            convert_none_strings(v)
        elif isinstance(v, str) and v.strip().lower() == "none":
            d[k] = None
    return d


def replace_prior_with_delta(config, param, value):
    """Replace the prior of `param` with a delta distribution at `value`."""
    fprint(f"replacing prior of `{param}` with a delta function.")
    priors = config.setdefault("model", {}).setdefault("priors", {})
    priors.pop(param, None)
    priors[param] = {
        "dist": "delta",
        "value": value
        }
    return config


def convert_to_absolute_paths(config):
    """Recursively convert relative paths in config to absolute paths."""
    root = config["root_main"]

    path_keys = {
        "fname_output",
        "los_file",
        "root",
        "path_density",
        "path_velocity",
    }

    def _recurse(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _recurse(v)
            elif k in path_keys and isinstance(v, str) and not isabs(v):
                d[k] = abspath(join(root, v))

    _recurse(config)
    return config


def load_config(config_path, replace_none=True, fill_paths=True):
    """
    Load a TOML configuration file and convert "none" strings to None.
    """
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    # Convert "none" strings to None
    if replace_none:
        config = convert_none_strings(config)

    # Assign delta priors if not using an underlying reconstruction.
    kind = config.get("pv_model", {}).get("kind", "")
    if not kind.startswith("precomputed_los"):
        config = replace_prior_with_delta(config, "alpha", 1.)
        config = replace_prior_with_delta(config, "beta", 0.)

    # Convert relative paths to absolute paths
    if fill_paths:
        config = convert_to_absolute_paths(config)

    return config


###############################################################################
#                        Coordinate transformations                           #
###############################################################################

def radec_to_cartesian(ra, dec):
    """
    Convert right ascension and declination (in degrees) to unit Cartesian
    coordinates.
    """
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    cos_dec = np.cos(dec_rad)

    x = cos_dec * np.cos(ra_rad)
    y = cos_dec * np.sin(ra_rad)
    z = np.sin(dec_rad)

    return np.column_stack([x, y, z])


def radec_to_galactic(ra, dec):
    """
    Convert right ascension and declination to galactic coordinates (all in
    degrees).
    """
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    return c.galactic.l.degree, c.galactic.b.degree


def galactic_to_radec(ell, b):
    """
    Convert galactic coordinates to right ascension and declination (all in
    degrees).
    """
    c = SkyCoord(l=ell*u.degree, b=b*u.degree, frame='galactic')
    return c.icrs.ra.degree, c.icrs.dec.degree


def galactic_to_radec_cartesian(ell, b):
    """
    Convert galactic coordinates (ell, b) in degrees to ICRS Cartesian unit
    vectors.
    """
    c = SkyCoord(l=np.atleast_1d(ell) * u.deg,
                 b=np.atleast_1d(b) * u.deg,
                 frame='galactic')
    icrs = c.icrs
    xyz = icrs.cartesian.xyz.value.T

    return xyz[0] if np.isscalar(ell) and np.isscalar(b) else xyz


def hms_to_degrees(hours, minutes=None, seconds=None):
    """Convert hours, minutes and seconds to degrees."""
    return hours * 15 + (minutes or 0) / 60 * 15 + (seconds or 0) / 3600 * 15


def dms_to_degrees(degrees, arcminutes=None, arcseconds=None):
    """Convert degrees, arcminutes and arcseconds to decimal degrees."""
    return degrees + (arcminutes or 0) / 60 + (arcseconds or 0) / 3600


###############################################################################
#                               Plotting                                      #
###############################################################################


def name2label(name):
    latex_labels = {
        "a_TFR": r"$a_\mathrm{TFR}$",
        "b_TFR": r"$b_\mathrm{TFR}$",
        "c_TFR": r"$c_\mathrm{TFR}$",
        "sigma_mu": r"$\sigma_\mu$",
        "sigma_v": r"$\sigma_v$",
        "alpha": r"$\alpha$",
        "b1": r"$b_1$",
        "b2": r"$b_2$",
        "beta": r"$\beta$",
        "Vext_mag": r"$V_\mathrm{ext}$",
        "Vext_ell": r"$\ell_\mathrm{ext}$",
        "Vext_b": r"$b_\mathrm{ext}$",
        "h": r"$h$",
        "a": r"$a$",
        "m1": r"$m_1$",
        "m2": r"$m_2$",
        "a_TFR_dipole_mag": r"$a_\mathrm{TFR, dipole}$",
        "a_TFR_dipole_ell": r"$\ell_\mathrm{TFR, dipole}$",
        "a_TFR_dipole_b": r"$b_\mathrm{TFR, dipole}$",
        "M_dipole_mag": r"$M_\mathrm{dipole}$",
        "M_dipole_ell": r"$\ell_\mathrm{dipole}$",
        "M_dipole_b": r"$b_\mathrm{dipole}$",
        "eta_prior_mean": r"$\hat{\eta}$",
        "eta_prior_std": r"$w_\eta$",
        "A_CL": r"$A_{\rm CL}$",
        "B_CL": r"$B_{\rm CL}$",
        "C_CL": r"$C_{\rm CL}$",
        "a_FP": r"$a_{\rm FP}$",
        "b_FP": r"$b_{\rm FP}$",
        "c_FP": r"$c_{\rm FP}$",
        "R_dust": r"$R_{\rm dust}$",
    }
    return latex_labels.get(name, name)


def plot_corner(samples, show_fig=True, filename=None, smooth=1, keys=None):
    """Plot a corner plot from posterior samples."""
    flat_samples = []
    labels = []

    for k, v in samples.items():
        if keys is not None and k not in keys:
            continue
        if v.ndim > 2:
            continue
        flat_samples.append(v.reshape(-1))
        labels.append(name2label(k))

    if not flat_samples:
        raise ValueError("No valid samples to plot.")

    data = np.vstack(flat_samples).T

    fig = corner(
        data,
        labels=labels,
        show_titles=True,
        smooth=smooth,
    )

    if filename is not None:
        fprint(f"saving a corner plot to {filename}")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def plot_Vext_rad_corner(samples, show_fig=True, filename=None, smooth=1):
    """Plot a corner plot of Vext_rad_{mag, ell, b} samples."""
    keys = ["Vext_rad_mag", "Vext_rad_ell", "Vext_rad_b"]
    base_labels = [r"V", r"\ell", r"b"]

    arrays = []
    labels = []

    for key, base_label in zip(keys, base_labels):
        if key not in samples:
            raise ValueError(f"Missing key: {key}")

        arr = samples[key]

        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim != 2:
            raise ValueError(f"{key} must be 2D or 3D")

        ndim = arr.shape[1]
        arrays.append(arr)

        for i in range(ndim):
            labels.append(fr"${base_label}_{{{i}}}$")

    data = np.hstack(arrays)  # shape: (nsamples_total, total_dims)

    fig = corner(data, labels=labels, show_titles=True, smooth=smooth)

    if filename is not None:
        fprint(f"saving knots corner plot to {filename}")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


###############################################################################
#                     Radial dependence of Vext                               #
###############################################################################


def interpolate_scalar_field(V, r, rbins, k=3, endpoints="not-a-knot"):
    V = jnp.asarray(V).reshape(-1, rbins.size)

    def spline_eval(y):
        spline = InterpolatedUnivariateSpline(
            rbins, y, k=k, endpoints=endpoints)
        return spline(r)

    return vmap(spline_eval)(V)


def interpolate_latitude_field(b_deg, r, rbins, k=3, endpoints="not-a-knot"):
    b_rad = jnp.deg2rad(b_deg).reshape(-1, rbins.size)
    sin_b = jnp.sin(b_rad)

    def spline_eval(y):
        spline = InterpolatedUnivariateSpline(
            rbins, y, k=k, endpoints=endpoints)
        return spline(r)

    sin_b_interp = vmap(spline_eval)(sin_b)
    return jnp.rad2deg(jnp.arcsin(jnp.clip(sin_b_interp, -1.0, 1.0)))


def interpolate_longitude_field(l_deg, r, rbins, k=3, endpoints="not-a-knot"):
    l_rad = jnp.deg2rad(l_deg).reshape(-1, rbins.size)
    sin_l = jnp.sin(l_rad)
    cos_l = jnp.cos(l_rad)

    def spline_eval(y):
        spline = InterpolatedUnivariateSpline(
            rbins, y, k=k, endpoints=endpoints)
        return spline(r)
    sin_l_interp = vmap(spline_eval)(sin_l)
    cos_l_interp = vmap(spline_eval)(cos_l)
    return jnp.rad2deg(jnp.arctan2(sin_l_interp, cos_l_interp)) % 360


def interpolate_all_radial_fields(model, Vmag, ell, b, r_eval_size=1000):
    rknot = jnp.asarray(model.kwargs_radial_Vext["rknot"])
    rmin, rmax = 0, jnp.max(rknot)
    k = model.kwargs_radial_Vext.get("k", 3)
    endpoints = model.kwargs_radial_Vext.get("endpoints", "not-a-knot")

    r = jnp.linspace(rmin, rmax, r_eval_size)

    Vmag_interp = interpolate_scalar_field(Vmag, r, rknot, k, endpoints)
    ell_interp = interpolate_longitude_field(ell, r, rknot, k, endpoints)
    b_interp = interpolate_latitude_field(b, r, rknot, k, endpoints)

    return r, Vmag_interp, ell_interp, b_interp


def plot_radial_profiles(samples, model, r_eval_size=1000, show_fig=True,
                         filename=None):
    """
    Plot the radial profiles of Vext_rad_{mag, ell, b} from the samples,
    including 1sigma and 2sigma percentile bands.
    """
    Vmag = samples["Vext_rad_mag"]
    ell = samples["Vext_rad_ell"]
    b = samples["Vext_rad_b"]

    r, V_interp, ell_interp, b_interp = interpolate_all_radial_fields(
        model, Vmag, ell, b, r_eval_size=r_eval_size
    )

    def get_percentiles(arr):
        arr = np.array(arr)
        p16, p50, p84 = np.percentile(arr, [16, 50, 84], axis=0)
        p025, p975 = np.percentile(arr, [2.5, 97.5], axis=0)
        return p025, p16, p50, p84, p975

    V025, V16, V50, V84, V975 = get_percentiles(V_interp)
    l025, l16, l50, l84, l975 = get_percentiles(ell_interp)
    b025, b16, b50, b84, b975 = get_percentiles(b_interp)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    c = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    components = [
        (V025, V16, V50, V84, V975, r"$V_{\rm dipole}~[\mathrm{km}/\mathrm{s}]$"),  # noqa
        (l025, l16, l50, l84, l975, r"$\ell_{\rm dipole}~[\mathrm{deg}]$"),
        (b025, b16, b50, b84, b975, r"$b_{\rm dipole}~[\mathrm{deg}]$"),
    ]

    for i, (lo2, lo1, med, hi1, hi2, ylabel) in enumerate(components):
        ax = axes[i]
        ax.fill_between(r, lo2, hi2, alpha=0.2, color=c)
        ax.fill_between(r, lo1, hi1, alpha=0.4, color=c)
        ax.plot(r, med, c=c)
        ax.set_xlabel(r"$r~[\mathrm{Mpc}/h]$")
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    if filename is not None:
        fprint(f"saving a radial profile plot to {filename}")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        fig.show()
    else:
        plt.close(fig)
