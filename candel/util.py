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
from os.path import abspath, basename, isabs, join, exists
from pathlib import Path
import healpy as hp
from warnings import warn

import astropy.units as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
from corner import corner
from getdist import MCSamples, plots
from h5py import File
from interpax import interp1d
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


def replace_prior_with_delta(config, param, value, verbose=True):
    """Replace the prior of `param` with a delta distribution at `value`."""
    if param not in config.get("model", {}).get("priors", {}):
        return config

    fprint(f"replacing prior of `{param}` with a delta function.",
           verbose=verbose)
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
    root_data = config.get("root_data", root)

    path_keys_root = {
        "fname_output",
    }
    path_keys_data = {
        "root",
        "los_file",
        "los_file_random",
        "path_density",
        "path_velocity",
    }

    def _recurse(d):
        for k, v in d.items():
            if isinstance(v, dict):
                _recurse(v)
            elif isinstance(v, str):
                if k in path_keys_root and not isabs(v):
                    d[k] = abspath(join(root, v))
                elif k in path_keys_data and not isabs(v):
                    d[k] = abspath(join(root_data, v))

    _recurse(config)
    return config


def load_config(config_path, replace_none=True, fill_paths=True,
                replace_los_prior=True):
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
    if replace_los_prior and not kind.startswith("precomputed_los"):
        config = replace_prior_with_delta(config, "alpha", 1.)
        config = replace_prior_with_delta(config, "beta", 0.)
        config = replace_prior_with_delta(config, "b1", 0.)
        config = replace_prior_with_delta(config, "delta_b1", 0.)

    # Convert relative paths to absolute paths
    if fill_paths:
        config = convert_to_absolute_paths(config)

    shared_params = config["inference"].get("shared_params", None)
    if shared_params and str(shared_params).lower() != "none":
        config["inference"]["shared_params"] = shared_params.split(",")

    return config


def get_nested(config, key_path, default=None):
    """Recursively access a nested value using a slash-separated key."""
    keys = key_path.split("/")
    current = config
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return default
        current = current[k]
    return current


def percent_h0_to_bulkflow(r, percent, *, H0=100.0, q0=-0.53):
    """
    Convert a fractional H0 dipole (in percent) to an equivalent bulk-flow
    magnitude evaluated at radius `r` (array-like).

    Matches the form used in `plot_Vext_radmag`: delta * (H0 r +
    q0 H0^2 r^2 / c).
    """
    frac = percent / 100.0
    r = np.asarray(r)
    return frac * (H0 * r + q0 * (H0**2) * r**2 / SPEED_OF_LIGHT)


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


def cartesian_to_radec(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to right ascension and
    declination (RA, Dec), both in degrees.
    """
    d = (x**2 + y**2 + z**2)**0.5
    dec = np.arcsin(z / d)
    ra = np.arctan2(y, x)
    ra[ra < 0] += 2 * np.pi

    ra *= 180 / np.pi
    dec *= 180 / np.pi

    return d, ra, dec


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


def supergalactic_to_radec(sgl, sgb):
    """
    Convert supergalactic coordinates (sgl, sgb) to equatorial
    right ascension and declination (RA, Dec), all in degrees.
    """
    c = SkyCoord(sgl=sgl * u.deg, sgb=sgb * u.deg, frame="supergalactic")
    return c.icrs.ra.deg, c.icrs.dec.deg


def radec_to_supergalactic(ra, dec):
    """
    Convert right ascension and declination (in degrees) to supergalactic
    coordinates in degrees.
    """
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    return c.supergalactic.sgl.deg, c.supergalactic.sgb.deg


def radec_cartesian_to_galactic(x, y, z):
    """
    Convert ICRS Cartesian vectors (x, y, z) to Galactic coordinates (ell, b)
    in degrees, and return the vector magnitude.
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    r = np.sqrt(x**2 + y**2 + z**2)

    rep = CartesianRepresentation(x * u.one, y * u.one, z * u.one)
    c_icrs = SkyCoord(rep, frame='icrs')
    gal = c_icrs.galactic

    ell = gal.l.deg
    b = gal.b.deg

    if r.size == 1:
        return r[0], ell[0], b[0]
    return r, ell, b


def hms_to_degrees(hours, minutes=None, seconds=None):
    """Convert hours, minutes and seconds to degrees."""
    return hours * 15 + (minutes or 0) / 60 * 15 + (seconds or 0) / 3600 * 15


def dms_to_degrees(degrees, arcminutes=None, arcseconds=None):
    """Convert degrees, arcminutes and arcseconds to decimal degrees."""
    return degrees + (arcminutes or 0) / 60 + (arcseconds or 0) / 3600


def heliocentric_to_cmb(z_helio, RA, dec, e_z_helio=None):
    """
    Convert heliocentric redshift to CMB redshift using the Planck 2018 CMB
    dipole.
    """
    # CMB dipole Planck 2018 values
    vsun_mag = 369  # km/s
    RA_sun = 167.942
    dec_sun = -6.944
    SPEED_OF_LIGHT = 299792.458  # km / s

    theta_sun = np.pi / 2 - np.deg2rad(dec_sun)
    phi_sun = np.deg2rad(RA_sun)

    # Convert to theat/phi in radians
    theta = np.pi / 2 - np.deg2rad(dec)
    phi = np.deg2rad(RA)

    # Unit vector in the direction of each galaxy
    n = np.asarray([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)]).T
    # CMB dipole unit vector
    vsun_normvect = np.asarray([np.sin(theta_sun) * np.cos(phi_sun),
                                np.sin(theta_sun) * np.sin(phi_sun),
                                np.cos(theta_sun)])

    # Project the CMB dipole onto the line of sight and normalize
    vsun_projected = vsun_mag * np.dot(n, vsun_normvect) / SPEED_OF_LIGHT

    zsun_tilde = np.sqrt((1 - vsun_projected) / (1 + vsun_projected))
    zcmb = (1 + z_helio) / zsun_tilde - 1

    # Optional linear error propagation
    if e_z_helio is not None:
        e_zcmb = np.abs(e_z_helio / zsun_tilde)
        return zcmb, e_zcmb

    return zcmb


###############################################################################
#                               Plotting                                      #
###############################################################################


def name2label(name):
    """
    Map internal parameter names to LaTeX labels, optionally including
    catalogue prefix.
    """
    latex_labels = {
        "a_TFR": r"$a_\mathrm{TFR}$",
        "b_TFR": r"$b_\mathrm{TFR}$",
        "c_TFR": r"$c_\mathrm{TFR}$",
        "sigma_int": r"$\sigma_L$",
        "sigma_int2": r"$\sigma_Y$",
        "sigma_LT": r"$\sigma_{LT}$",
        "sigma_YT": r"$\sigma_{YT}$",
        "rho12": r"$\rho$",
        "sigma_v": r"$\sigma_v$",
        "alpha": r"$\alpha$",
        "alpha_low": r"$\alpha_\mathrm{low}$",
        "alpha_high": r"$\alpha_\mathrm{high}$",
        "log_rho_t": r"$\ln \rho_t$",
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
        "zeropoint_dipole_mag": r"$\Delta \mathrm{ZP}$",
        "zeropoint_dipole_ell": r"$\ell_{\Delta H_0}$",
        "zeropoint_dipole_b": r"$b_{\Delta H_0}$",
        "dH_over_H_dipole": r"$\Delta H_0/H_0$",
        "dH_over_H_quad": r"$\Delta H_0/H_0$",
        "SN_absmag": r"$M_{\rm SN}$",
        "SN_alpha": r"$\mathcal{A}$",
        "SN_beta": r"$\mathcal{B}$",
        "eta_prior_mean": r"$\hat{\eta}$",
        "eta_prior_std": r"$w_\eta$",
        "A_LT": r"$a_{LT}$",
        "B_LT": r"$b_{LT}$",
        "A_YT": r"$a_{YT}$",
        "B_YT": r"$b_{YT}$",
        "A_CL": r"$a_L$",   # legacy
        "B_CL": r"$b_L$",   # legacy
        "C_CL": r"$C_{\rm CL}$",
        "A2_CL": r"$a_Y$",
        "B2_CL": r"$b_Y$",
        "a_FP": r"$a_{\rm FP}$",
        "b_FP": r"$b_{\rm FP}$",
        "c_FP": r"$c_{\rm FP}$",
        "sigma_log_theta": r"$\sigma_{\log \theta}$",
        "R_dust": r"$R_{\rm W1}$",
        "R_dist_emp": r"$R_{\rm dist}$",
        "n_dist_emp": r"$n_{\rm dist}$",
        "p_dist_emp": r"$p_{\rm dist}$",
        "Rmax_dist_emp": r"$R_{\rm max, dist}$",
        "rho_corr": r"$\rho_{\rm corr}$",
        "Vext_radmag_ell": r"$\ell_{\mathrm{Vext}}$",
        "Vext_radmag_b": r"$b_{\mathrm{Vext}}$",
    }

    # Handle radial_binned Vext parameters (e.g., Vext_radial_bin_mag__0)
    if "Vext_radial_bin" in name:
        parts = name.split("__")
        if len(parts) >= 2:
            bin_idx = parts[-1]
            if "mag" in name:
                return rf"$V_{{\mathrm{{ext}},{bin_idx}}}$"
            elif "ell" in name:
                return rf"$\ell_{{\mathrm{{ext}},{bin_idx}}}$"
            elif "b" in name:
                return rf"$b_{{\mathrm{{ext}},{bin_idx}}}$"
    
    # Handle radial_binned dipole A parameters (e.g., A_dipole_radial_bin_mag__0)
    if "A_dipole_radial_bin" in name:
        parts = name.split("__")
        if len(parts) >= 2:
            bin_idx = parts[-1]
            if "mag" in name:
                return rf"$A_{{\mathrm{{dip}},{bin_idx}}}$"
            elif "ell" in name:
                return rf"$\ell_{{A,{bin_idx}}}$"
            elif "b" in name:
                return rf"$b_{{A,{bin_idx}}}$"
    
    # Handle radial_binned A parameters (e.g., A_radial_bin__0)
    if "A_radial_bin" in name and "dipole" not in name:
        parts = name.split("__")
        if len(parts) >= 2:
            bin_idx = parts[-1]
            return rf"$A_{{\mathrm{{CL}},{bin_idx}}}$"
    
    if "/" in name:
        prefix, base = name.split("/", 1)
        base_label = latex_labels.get(base, base)
        prefix_latex = prefix.replace("_", r"\,").replace(" ", "~")
        return rf"$\mathrm{{{prefix_latex}}},\,{base_label.strip('$')}$"

    return latex_labels.get(name, name)


def name2labelgetdist(name):
    """
    Return a GetDist-compatible LaTeX label (no $...$) for a parameter,
    optionally prepending the catalogue name as plain text.

    Example:
        "CF4_W1/beta" → r"\\mathrm{CF4~W1}, \\beta"
    """
    labels = {
        "a_TFR": r"a_\mathrm{TFR}",
        "b_TFR": r"b_\mathrm{TFR}",
        "c_TFR": r"c_\mathrm{TFR}",
        "SN_absmag": r"M_{\rm SN}",
        "SN_alpha": r"\mathcal{A}",
        "SN_beta": r"\mathcal{B}",
        "sigma_int": r"\sigma_L",
        "sigma_int2": r"\sigma_Y",
        "sigma_LT": r"\sigma_{LT}",
        "sigma_YT": r"\sigma_{YT}",
        "rho12": r"\rho_{LY}",
        "sigma_v": r"\sigma_v~\left[\mathrm{km}\,\mathrm{s}^{-1}\right]",
        "alpha": r"\alpha",
        "alpha_low": r"\alpha_\mathrm{low}",
        "alpha_high": r"\alpha_\mathrm{high}",
        "log_rho_t": r"\ln \rho_t",
        "b1": r"b_1",
        "b2": r"b_2",
        "beta": r"\beta",
        "Vext_mag": r"V_\mathrm{ext}~\left[\mathrm{km}\,\mathrm{s}^{-1}\right]",  # noqa
        "Vext_ell": r"\ell_\mathrm{ext}~\left[\mathrm{deg}\right]",
        "Vext_ell_offset": r"\ell_\mathrm{ext} - 180~\left[\mathrm{deg}\right]",  # noqa
        "Vext_b":   r"b_\mathrm{ext}~\left[\mathrm{deg}\right]",
        "h": r"h",
        "a": r"a",
        "m1": r"m_1",
        "m2": r"m_2",
        "zeropoint_dipole_mag": r"\Delta_\mathrm{ZP}",         # noqa
        "zeropoint_dipole_ell": r"\ell_{\Delta H_0}~\left[\mathrm{deg}\right]",  # noqa
        "zeropoint_dipole_b": r"b_{\Delta H_0}~\left[\mathrm{deg}\right]",       # noqa
        "M_dipole_mag": r"\Delta M_\mathrm{SN}",
        "M_dipole_ell": r"\ell_{\Delta M_{\rm SN}}~\left[\mathrm{deg}\right]",
        "M_dipole_b": r"b_{\Delta M_{\rm SN}}~\left[\mathrm{deg}\right]",
        "eta_prior_mean": r"\hat{\eta}",
        "eta_prior_std": r"w_\eta",
        "A_LT": r"a_{LT}",
        "B_LT": r"b_{LT}",
        "A_YT": r"a_{YT}",
        "B_YT": r"b_{YT}",
        "A_CL": r"a_L",   # legacy
        "B_CL": r"b_L",   # legacy
        "C_CL": r"C_{\rm CL}",
        "A2_CL": r"a_Y",
        "B2_CL": r"b_Y",
        "a_FP": r"a_{\rm FP}",
        "b_FP": r"b_{\rm FP}",
        "c_FP": r"c_{\rm FP}",
        "R_dust": r"R_{\rm W1}",
        "mu_LMC": r"\mu_{\rm LMC}",
        "mu_M31": r"\mu_{\rm M31}",
        "mu_N4258": r"\mu_{\rm NGC4258}",
        "H0": r"H_0~\left[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}\right]",  # noqa
        "dZP": r"\Delta_{\rm ZP}",
        "R_dist_emp": r"R~\left[h^{-1}\,\mathrm{Mpc}\right]",
        "n_dist_emp": r"n",
        "p_dist_emp": r"p",
        "rho_corr": r"\rho_{\rm corr}",
        "dH_over_H_dipole": r"\Delta H_0/H_0",
        "dH_over_H_quad": r"\Delta H_0/H_0",
    }

    # Handle radial_binned Vext parameters (e.g., Vext_radial_bin_mag__0)
    if "Vext_radial_bin" in name:
        parts = name.split("__")
        if len(parts) >= 2:
            bin_idx = parts[-1]
            if "mag" in name:
                return rf"V_{{\mathrm{{ext}},{bin_idx}}}~\left[\mathrm{{km}}\,\mathrm{{s}}^{{-1}}\right]"
            elif "ell" in name:
                return rf"\ell_{{\mathrm{{ext}},{bin_idx}}}~\left[\mathrm{{deg}}\right]"
            elif "b" in name:
                return rf"b_{{\mathrm{{ext}},{bin_idx}}}~\left[\mathrm{{deg}}\right]"
    
    # Handle radial_binned dipole A parameters (e.g., A_dipole_radial_bin_mag__0)
    if "A_dipole_radial_bin" in name:
        parts = name.split("__")
        if len(parts) >= 2:
            bin_idx = parts[-1]
            if "mag" in name:
                return rf"A_{{\mathrm{{dip}},{bin_idx}}}~\left[\mathrm{{mag}}\right]"
            elif "ell" in name:
                return rf"\ell_{{A,{bin_idx}}}~\left[\mathrm{{deg}}\right]"
            elif "b" in name:
                return rf"b_{{A,{bin_idx}}}~\left[\mathrm{{deg}}\right]"
    
    # Handle radial_binned A parameters (e.g., A_radial_bin__0)
    if "A_radial_bin" in name and "dipole" not in name:
        parts = name.split("__")
        if len(parts) >= 2:
            bin_idx = parts[-1]
            return rf"A_{{\mathrm{{CL}},{bin_idx}}}~\left[\mathrm{{mag}}\right]"

    if "/" in name:
        prefix, base = name.split("/", 1)
        base_label = labels.get(base, base)
        prefix_latex = prefix.replace("_", r"\,").replace(" ", "~")
        return rf"\mathrm{{{prefix_latex}}},\,{base_label}"

    return labels.get(name, name)


def sort_params(keys):
    order = [
        "a_TFR", "b_TFR", "c_TFR",
        "alpha", "beta",
        "sigma_mu", "sigma_v",
        "Vext", "Vext_mag", "Vext_ell", "Vext_b"
    ]

    def sort_key(k):
        try:
            return (order.index(k), k)
        except ValueError:
            # Put unlisted keys at the end, alphabetically
            return (len(order), k)

    return sorted(keys, key=sort_key)


def plot_corner(samples, show_fig=True, filename=None, smooth=1, keys=None):
    """Plot a corner plot from posterior samples."""
    flat_samples = []
    labels = []

    for k, v in samples.items():
        if keys is not None and k not in keys:
            continue

        if k == "Vext_radmag_mag":
            nbin = v.shape[1]
            for i in range(nbin):
                flat_samples.append(v[:, i])
                labels.append(fr"$V_{{\mathrm{{ext}}, {{{i}}}}}$")

        if v.ndim > 1:
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
    """
    Plot a corner plot of Vext_rad_{mag, ell, b} samples.
    
    Handles both:
    - vector_radial_spline_uniform: Vext_rad_mag, Vext_rad_ell, Vext_rad_b (per knot)
    - vector_radial_spline_uniform_fixed_direction: Vext_rad_direction_ell/b (single), Vext_rad_mag__{i} (per knot)
    """
    # Check if using fixed direction variant
    is_fixed_direction = "Vext_rad_direction_ell" in samples
    
    if is_fixed_direction:
        # For fixed direction: plot direction + magnitudes
        ell = samples["Vext_rad_direction_ell"]
        b = samples["Vext_rad_direction_b"]
        
        # Extract per-knot magnitudes
        mag_keys = sorted([k for k in samples.keys() if k.startswith("Vext_rad_mag__")])
        
        arrays = []
        labels = []
        
        # Add direction (single value)
        arrays.append(ell[:, None])
        labels.append(r"$\ell_{\rm dir}$")
        
        arrays.append(b[:, None])
        labels.append(r"$b_{\rm dir}$")
        
        # Add magnitudes per knot
        for i, key in enumerate(mag_keys):
            arrays.append(samples[key][:, None])
            labels.append(fr"$V_{{{i}}}$")
        
        data = np.hstack(arrays)
    else:
        # Independent vectors per knot
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


def plot_corner_getdist(samples_list, labels=None, cols=None, show_fig=True,
                        filename=None, keys=None, fontsize=None,
                        legend_fontsize=None, filled=True,
                        apply_ell_offset=False, ell_zero=180,
                        mag_range=[0, None], ell_range=[0, 360],
                        b_range=[-90, 90], points=None,
                        ranges={}, truths=None):
    """Plot a GetDist triangle plot for one or more posterior samples."""

    if isinstance(samples_list, dict):
        samples_list = [samples_list]

    if labels is not None and len(labels) != len(samples_list):
        raise ValueError("Length of `labels` must match number of sample sets")

    # Build candidate key list (user-specified or inferred)
    if keys is not None:
        candidate_keys = keys
    else:
        candidate_keys = [
            k for k in samples_list[0] if samples_list[0][k].ndim == 1]

    # Include keys that are present and 1D in at least one sample set
    param_names = []
    for k in candidate_keys:
        for s in samples_list:
            if k in s and s[k].ndim == 1:
                param_names.append(k)
                break
            elif k in s and s[k].ndim > 1:
                fprint(f"[SKIP] {k} has shape {s[k].shape}")
                break

    if keys is None:
        param_names = sort_params(param_names)

    for k in param_names:
        if "_mag" in k:
            ranges[k] = mag_range

        if "_ell" in k:
            ranges[k] = ell_range

        if "_b" in k:
            ranges[k] = b_range

        if "dH_over_H" in k:
            ranges[k] = [0, None]

    gdsamples_list = []

    for samples in samples_list:
        present_params = []
        present_labels = []
        columns = []

        n_samples = len(next(iter(samples.values())))
        for k in param_names:
            if k in samples and samples[k].ndim == 1:
                col = samples[k].reshape(-1)
            else:
                col = np.full(n_samples, np.nan)

            if not np.all(np.isnan(col)):
                if "_ell" in k and apply_ell_offset:
                    col = (col - ell_zero) % 360
                label = name2labelgetdist(k)

                present_params.append(k)
                present_labels.append(label)
                columns.append(col)

        data = np.vstack(columns).T
        gds = MCSamples(
            samples=data,
            names=present_params,
            labels=present_labels,
            ranges={k: ranges[k] for k in present_params if k in ranges},
            )
        gdsamples_list.append(gds)

    # Plot styling
    settings = plots.GetDistPlotSettings()
    if fontsize is not None:
        settings.lab_fontsize = fontsize
        settings.legend_fontsize = legend_fontsize if legend_fontsize is not None else fontsize  # noqa
        settings.axes_fontsize = fontsize - 1
        settings.title_limit_fontsize = fontsize - 1

    if cols is not None:
        line_args = [{"color": c} for c in cols]
    else:
        line_args = None

    with plt.style.context("science"):
        g = plots.get_subplot_plotter(settings=settings)
        g.triangle_plot(
            gdsamples_list,
            params=param_names,
            filled=filled,
            colors=cols,
            contour_colors=cols,
            line_args=line_args,
            legend_labels=labels,
            legend_loc="upper right",
        )

        if apply_ell_offset:
            tick_positions = np.arange(0, 360, 90)
            tick_labels = [str(int((ell_zero + t) % 360)) for t in tick_positions]

            for i, param in enumerate(param_names):
                if "_ell" in param:
                    ax_diag = g.subplots[i, i]
                    ax_diag.set_xticks(tick_positions)
                    ax_diag.set_xticklabels(tick_labels)

                for j in range(i):
                    ax = g.subplots[i, j]
                    if "_ell" in param:
                        ax.set_yticks(tick_positions)
                        ax.set_yticklabels(tick_labels)
                    if "_ell" in param_names[j]:
                        ax.set_xticks(tick_positions)
                        ax.set_xticklabels(tick_labels)

        if points is not None:
            plotted_pairs = set()
            for (x_param, y_param), (x_val, y_val) in points.items():
                if x_param not in param_names or y_param not in param_names:
                    continue
                ix = param_names.index(x_param)
                iy = param_names.index(y_param)
                if iy > ix and (ix, iy) not in plotted_pairs:
                    ax = g.subplots[iy, ix]
                    ax.plot(x_val, y_val, "x", color="red", markersize=10)
                    __, labels_ = ax.get_legend_handles_labels()
                    if "Reference" not in labels_:
                        ax.legend()
                    plotted_pairs.add((ix, iy))

        if truths is not None:
            lw = 1.5 * plt.rcParams["lines.linewidth"]
            for truth_set in truths:
                truths = truth_set["dict"]
                color = truth_set.get("color", "red")
                linestyle = truth_set.get("linestyle", "--")

                # 1D panels: vertical lines
                for i, param in enumerate(param_names):
                    if param in truths:
                        val = truths[param]
                        ax = g.subplots[i, i]
                        ax.axvline(
                            val, color=color, linestyle=linestyle,
                            lw=lw, label=label)

                # 2D panels: vertical/horizontal lines and crosses
                for i, x_param in enumerate(param_names):
                    for j, y_param in enumerate(param_names):
                        if j > i:
                            ax = g.subplots[j, i]
                            x_in = x_param in truths
                            y_in = y_param in truths

                            if x_in:
                                x_val = truths[x_param]
                                ax.axvline(
                                    x_val, color=color, linestyle=linestyle,
                                    lw=lw, label=label)
                            if y_in:
                                y_val = truths[y_param]
                                ax.axhline(
                                    y_val, color=color, linestyle=linestyle,
                                    lw=lw, label=label)

        if filename is not None:
            fprint(f"[INFO] Saving GetDist triangle plot to: {filename}")
            g.export(filename, dpi=450)

        if show_fig:
            plt.show()
        else:
            plt.close()


def plot_corner_from_hdf5(fnames, keys=None, labels=None, cols=None,
                          fontsize=None, legend_fontsize=None, filled=True,
                          show_fig=True, filename=None, apply_ell_offset=False,
                          ell_zero=180, mag_range=[0, None],
                          ell_range=[0, 360], b_range=[-90, 90],
                          points=None, ranges={}, truths=None):
    """
    Plot a triangle plot from one or more HDF5 files containing posterior
    samples.
    """
    if isinstance(fnames, (str, Path)):
        fnames = [fnames]

    samples_list = []
    for fname in fnames:
        with File(fname, 'r') as f:
            grp = f["samples"]
            samples = {key: grp[key][...] for key in grp.keys()}
            samples_list.append(samples)

            full_keys = list(grp.keys())
            print(f"{basename(fname)}: {', '.join(full_keys)}")

    plot_corner_getdist(
        samples_list,
        labels=labels,
        keys=keys,
        cols=cols,
        fontsize=fontsize,
        legend_fontsize=legend_fontsize,
        filled=filled,
        show_fig=show_fig,
        filename=filename,
        apply_ell_offset=apply_ell_offset,
        ell_zero=ell_zero,
        ranges=ranges,
        mag_range=mag_range,
        ell_range=ell_range,
        b_range=b_range,
        points=points,
        truths=truths,
    )


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
    rknot = jnp.asarray(model.kwargs_Vext["rknot"])
    rmin, rmax = 0, jnp.max(rknot)
    k = model.kwargs_Vext.get("k", 3)
    endpoints = model.kwargs_Vext.get("endpoints", "not-a-knot")

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
    
    Handles both:
    - vector_radial_spline_uniform: independent directions at each knot
    - vector_radial_spline_uniform_fixed_direction: fixed direction, variable magnitude
    """
    # Check if using fixed direction variant
    is_fixed_direction = "Vext_rad_direction_ell" in samples
    
    if is_fixed_direction:
        # Fixed direction: extract single direction and per-knot magnitudes
        ell = samples["Vext_rad_direction_ell"]
        b = samples["Vext_rad_direction_b"]
        
        # Extract per-knot magnitudes
        mag_keys = sorted([k for k in samples.keys() if k.startswith("Vext_rad_mag__")])
        Vmag = np.stack([samples[k] for k in mag_keys], axis=1)
        
        # Get radial positions
        rknot = jnp.asarray(model.kwargs_Vext["rknot"])
        k_spline = model.kwargs_Vext.get("k", 3)
        endpoints = model.kwargs_Vext.get("endpoints", "not-a-knot")
        r = jnp.linspace(0, jnp.max(rknot), r_eval_size)
        
        # Interpolate magnitudes
        V_interp = interpolate_scalar_field(
            jnp.array(Vmag), r, rknot, k_spline, endpoints)
        V_interp = np.array(V_interp)
        
        # Direction is constant, just replicate for plotting
        ell_interp = np.tile(ell[:, None], (1, r_eval_size))
        b_interp = np.tile(b[:, None], (1, r_eval_size))
        r = np.array(r)
    else:
        # Independent vectors: extract per-knot mag/ell/b
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

    def add_knot_markers(ax):
        """Add vertical lines at knot/bin positions."""
        rknot = np.array(model.kwargs_Vext.get("rknot", []))
        for rk in rknot:
            ax.axvline(rk, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    def add_h0_dipole_reference(ax, ell_val, b_val, std_ell, std_b):
        """Add H0 dipole reference line and band with inferred dipole direction."""
        H0 = 100.0            # km/s/Mpc
        q0 = -0.53            # deceleration parameter
        delta1 = 0.05 * 1.15  # 5% H0 dipole
        c_light = 3e5         # km/s
        bf = delta1 * (H0 * r + q0 * H0**2 * r**2 / c_light)
        delta_u = 0.07 * 1.15
        delta_l = 0.03 * 1.15
        bu = delta_u * (H0 * r + q0 * H0**2 * r**2 / c_light)
        bl = delta_l * (H0 * r + q0 * H0**2 * r**2 / c_light)

        H0_ell_val = 123.9
        H0_b_val = 53.88
        H0_std_ell =  99.8
        H0_std_b = 17.71
        
        ax.plot(r, bf, linestyle="--", color="gray",
                label=fr"Equivalent $H_0$ dipole: $(5 \pm 2)\%$ at $(\ell, b) = ({H0_ell_val:.1f} \pm {H0_std_ell:.1f}°, {H0_b_val:.1f} \pm {H0_std_b:.1f}°)$")
        ax.fill_between(r, bl, bu, color="gray", alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    def plot_component(ax, lo2, lo1, med, hi1, hi2, ylabel, c, add_label=False, 
                      label_ell=None, label_b=None, std_ell=None, std_b=None):
        """Plot a single component with percentile bands."""
        ax.fill_between(r, lo2, hi2, alpha=0.2, color=c)
        ax.fill_between(r, lo1, hi1, alpha=0.4, color=c)
        if add_label and label_ell is not None and label_b is not None:
            if std_ell is not None and std_b is not None:
                ax.plot(r, med, c=c, 
                       label=fr"Radially varying $V_{{\rm ext}}$: $(\ell, b) = ({label_ell:.1f} \pm {std_ell:.1f}°, {label_b:.1f} \pm {std_b:.1f}°)$")
            else:
                ax.plot(r, med, c=c, 
                       label=fr"Radially varying $V_{{\rm ext}}$ at $(\ell, b) = ({label_ell:.1f}°, {label_b:.1f}°)$")
        else:
            ax.plot(r, med, c=c)
        ax.set_xlabel(r"$r~[\mathrm{Mpc}/h]$")
        ax.set_ylabel(ylabel)
        add_knot_markers(ax)

    V025, V16, V50, V84, V975 = get_percentiles(V_interp)
    c = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    
    if is_fixed_direction:
        # Only plot magnitude for fixed direction
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        
        # Calculate statistics for direction
        mean_ell = np.mean(ell)
        std_ell = np.std(ell)
        mean_b = np.mean(b)
        std_b = np.std(b)
        
        # Plot the radially varying dipole with label including uncertainties
        plot_component(ax, V025, V16, V50, V84, V975, 
                      r"$V_{\rm dipole}~[\mathrm{km}/\mathrm{s}]$", c,
                      add_label=True, label_ell=mean_ell, label_b=mean_b,
                      std_ell=std_ell, std_b=std_b)
        
        # Add H0 dipole reference with same inferred direction
        add_h0_dipole_reference(ax, mean_ell, mean_b, std_ell, std_b)
        
    else:
        # Plot all three components for independent directions
        l025, l16, l50, l84, l975 = get_percentiles(ell_interp)
        b025, b16, b50, b84, b975 = get_percentiles(b_interp)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

        components = [
            (V025, V16, V50, V84, V975, r"$V_{\rm dipole}~[\mathrm{km}/\mathrm{s}]$"),
            (l025, l16, l50, l84, l975, r"$\ell_{\rm dipole}~[\mathrm{deg}]$"),
            (b025, b16, b50, b84, b975, r"$b_{\rm dipole}~[\mathrm{deg}]$"),
        ]

        for i, (lo2, lo1, med, hi1, hi2, ylabel) in enumerate(components):
            plot_component(axes[i], lo2, lo1, med, hi1, hi2, ylabel, c)
    
    fig.tight_layout()
    if filename is not None:
        fprint(f"saving a radial profile plot to {filename}")
        fig.savefig(filename, bbox_inches="tight", dpi=450)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def _add_equator_labels(lon_step=60, lat_step=30):
    for lon in np.arange(0, 360, lon_step):
        hp.projtext(lon, -2.0, f"{lon:d}°", lonlat=True,
                    fontsize=9, ha="center", va="top")
    for lat in np.arange(-60, 61, lat_step):
        if lat == 0:
            continue
        hp.projtext(-178.0, lat, f"{lat:+d}°", lonlat=True,
                    fontsize=9, ha="right", va="center")


def _upsample_map(map_lo, nside_plot, *, nest=False):
    nside_lo = hp.npix2nside(map_lo.size)
    if nside_plot is None or nside_plot <= nside_lo:
        return map_lo
    pix_hi = np.arange(hp.nside2npix(nside_plot))
    th, ph = hp.pix2ang(nside_plot, pix_hi, nest=nest)
    # Bilinear interpolation from the coarse map:
    map_hi = hp.get_interp_val(map_lo, th, ph, nest=nest, lonlat=False)
    return map_hi


def plot_Vext_radmag(samples, model, r_eval_size=1000, show_fig=True,
                     filename=None, data=None, h0_samples=None):
    # Lazy import to avoid circular dependency at module import time.
    from .cosmography import Redshift2Distance

    Vmag = samples["Vext_radmag_mag"]
    ell = samples.get("Vext_radmag_ell", None)
    b = samples.get("Vext_radmag_b", None)
    rknot = model.kwargs_Vext["rknot"]
    method = model.kwargs_Vext["method"]

    # Attempt to extract zcmb (and Y, if present) for the distance histogram.
    zcmb = None
    Y = None
    if data is not None:
        try:
            data_dict = data.data if hasattr(data, "data") else data
            zcmb = np.asarray(data_dict.get("zcmb"))
            Y = data_dict.get("Y", None)
        except Exception:
            zcmb = None
            Y = None

    r = jnp.linspace(0.0, np.max(rknot), r_eval_size)
    V = vmap(lambda y: interp1d(r, rknot, y, method=method))(Vmag)
    V025, V16, V50, V84, V975 = np.percentile(V, [2.5, 16, 50, 84, 97.5], axis=0)

    # Compute distance range from data (if available) to align the x-axis.
    r2d = Redshift2Distance()
    dist_max = None
    r_cap = 1000.0
    if zcmb is not None:
        dist = r2d(zcmb, h=1.0)
        dist_max = float(np.nanmax(dist))
        # Extend the evaluation grid to the max of data or knots.
        r_max = max(dist_max, float(np.max(rknot)))
        r_max = min(r_max, r_cap)
        dist_max = min(dist_max, r_cap)
        r = jnp.linspace(0.0, r_max, r_eval_size)
        V = vmap(lambda y: interp1d(
            r, rknot, y, method=method, extrap=(y[0], y[-1])))(Vmag)
        V025, V16, V50, V84, V975 = np.percentile(
            V, [2.5, 16, 50, 84, 97.5], axis=0)
    else:
        dist_max = float(np.max(rknot))
        r_max = float(np.max(rknot))
        r_max = min(r_max, r_cap)
        dist_max = min(dist_max, r_cap)

    fig, (ax_hist, ax) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 2.0]})

    # Top panel: distance histogram (if data is available).
    if zcmb is not None and zcmb.size > 0 and dist_max > 0:
        dist = np.asarray(dist)
        n_bins = 25
        bins = np.linspace(0.0, dist_max, n_bins + 1)
        # If Y is present, split the histogram; otherwise single series.
        if Y is not None:
            Y_arr = np.asarray(Y)
            has_Y = Y_arr > 0
            no_Y = Y_arr <= 0
            counts_no_Y, bin_edges = np.histogram(dist[no_Y], bins=bins)
            counts_has_Y, _ = np.histogram(dist[has_Y], bins=bin_edges)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            width = bin_edges[1] - bin_edges[0]
            ax_hist.bar(bin_centers, counts_has_Y, width=width, alpha=0.7,
                        label=f'With $Y_{{SZ}}$ (N={np.sum(has_Y)})',
                        color="#7570b3", edgecolor='black', linewidth=0.5)
            ax_hist.bar(bin_centers, counts_no_Y, width=width, alpha=0.7,
                        bottom=counts_has_Y,
                        label=f'Without $Y_{{SZ}}$ (N={np.sum(no_Y)})',
                        color="#d95f02", edgecolor='black', linewidth=0.5)
            ax_hist.legend(fontsize=9, loc="upper right")
        else:
            ax_hist.hist(dist, bins=bins, color="C0", alpha=0.7, edgecolor='black')
        ax_hist.set_ylabel("Number of sources")
        ax_hist.grid(alpha=0.3)
        ax_hist.set_xlim(0.0, r_max)
    else:
        ax_hist.text(0.5, 0.5, "No redshift data provided", ha="center",
                     va="center", transform=ax_hist.transAxes)
        ax_hist.set_ylabel("Number of sources")

    # Add secondary x-axis (redshift) on the top histogram.
    def z_to_dist(z):
        return r2d(z, h=1.0)

    def dist_to_z(d):
        z_grid = np.linspace(1e-5, 1.0, 1000)
        d_grid = z_to_dist(z_grid)
        return np.interp(d, d_grid, z_grid)

    ax_top = ax_hist.secondary_xaxis('top', functions=(dist_to_z, z_to_dist))
    ax_top.set_xlabel(r"Redshift $z_\mathrm{CMB}$")

    # Bottom panel: Vext radial magnitude profile.
    ax.fill_between(r, V025, V975, alpha=0.2, color="C0")
    ax.fill_between(r, V16, V84, alpha=0.4, color="C0")
    ax.plot(r, V50, color="C0")
    ax.set_xlabel(r"$r~[h^{-1}\,\mathrm{Mpc}]$")
    ax.set_ylabel(r"$V_{\mathrm{ext}}~[\mathrm{km/s}]$")

    ax.set_xlim(r[0], r[-1])
    ax.set_ylim(None, None)

    xmin, xmax = r[0], r[-1]

    dx = 0.01 * (xmax - xmin)  # shift by 1% of the span
    knot_line_kwargs = dict(
        color="black", linestyle="--", zorder=-1, alpha=0.5)
    for rk in rknot:
        if jnp.isclose(rk, xmin):
            ax.axvline(xmin + dx, **knot_line_kwargs)
        elif jnp.isclose(rk, xmax):
            ax.axvline(xmax - dx, **knot_line_kwargs)
        else:
            ax.axvline(rk, **knot_line_kwargs)

    prior_cfg = getattr(model, "config", {})

    def _get_prior_bounds():
        """Return (lower, upper) arrays for the prior, if available."""
        cfg = get_nested(prior_cfg, "model/priors/Vext_radial_magnitude", {})
        if not cfg:
            return None
        rk_cfg = np.asarray(cfg.get("rknot", rknot))
        if rk_cfg.shape[0] != len(rknot):
            warn("rknot mismatch between model and prior; skipping prior overlay.")
            return None

        if "h0_dipole_percent" in cfg:
            mag = np.abs(percent_h0_to_bulkflow(rk_cfg, cfg["h0_dipole_percent"]))
            return -mag, mag

        if "max_modulus" in cfg:
            mag = np.abs(np.asarray(cfg["max_modulus"]))
            if mag.shape[0] != len(rk_cfg):
                warn("max_modulus length mismatch; skipping prior overlay.")
                return None
            return -mag, mag

        low = cfg.get("low", None)
        high = cfg.get("high", None)
        if low is None or high is None:
            return None
        low_arr = np.asarray(low)
        high_arr = np.asarray(high)
        if low_arr.ndim == 0:
            low_arr = np.full_like(rk_cfg, low_arr, dtype=float)
        if high_arr.ndim == 0:
            high_arr = np.full_like(rk_cfg, high_arr, dtype=float)
        if low_arr.shape[0] != len(rk_cfg) or high_arr.shape[0] != len(rk_cfg):
            warn("low/high length mismatch; skipping prior overlay.")
            return None
        return low_arr, high_arr

    def add_h0_dipole_reference(ax_ref):
        """Overlay the equivalent H0 dipole reference band."""
        dH_over_H = None
        ell_samples = None
        b_samples = None
        if h0_samples is not None:
            dH_over_H = h0_samples.get("dH_over_H_dipole", None)
            ell_samples = h0_samples.get("zeropoint_dipole_ell", None)
            b_samples = h0_samples.get("zeropoint_dipole_b", None)

        if dH_over_H is not None:
            pct = 100.0 * np.asarray(dH_over_H)
            pct_cen = float(np.mean(pct))
            pct_std = float(np.std(pct))
            pct_lo = pct_cen - pct_std
            pct_hi = pct_cen + pct_std

            bf = percent_h0_to_bulkflow(r, pct_cen)
            bu = percent_h0_to_bulkflow(r, pct_hi)
            bl = percent_h0_to_bulkflow(r, pct_lo)

            label = f"Equivalent $H_0$ dipole: ${pct_cen:.2f}\\% \\pm {pct_std:.2f}\\%$"
            if ell_samples is not None and b_samples is not None:
                H0_ell_val = float(np.mean(ell_samples))
                H0_b_val = float(np.mean(b_samples))
                H0_std_ell = float(np.std(ell_samples))
                H0_std_b = float(np.std(b_samples))
                label = (f"Equivalent $H_0$ dipole: ${pct_cen:.2f}\\% \\pm {pct_std:.2f}\\%$ "
                         f"at "
                         f"$(\\ell, b) = ({H0_ell_val:.1f} \\pm {H0_std_ell:.1f}°, "
                         f"{H0_b_val:.1f} \\pm {H0_std_b:.1f}°)$")
        else:
            pct_cen = 15.0 * 1.15
            pct_lo = pct_cen - 2.0
            pct_hi = pct_cen + 2.0

            bf = percent_h0_to_bulkflow(r, pct_cen)
            bu = percent_h0_to_bulkflow(r, pct_hi)
            bl = percent_h0_to_bulkflow(r, pct_lo)

            H0_ell_val = 123.9
            H0_b_val = 53.88
            H0_std_ell = 99.8
            H0_std_b = 17.71

            label = (f"Equivalent $H_0$ dipole: ${pct_cen:.1f}\\%$ "
                     f"at "
                     f"$(\\ell, b) = ({H0_ell_val:.1f} \\pm {H0_std_ell:.1f}°, "
                     f"{H0_b_val:.1f} \\pm {H0_std_b:.1f}°)$")

        ax_ref.plot(r, bf, linestyle="--", color="gray", label=label)
        ax_ref.fill_between(r, bl, bu, color="gray", alpha=0.3)
        ax_ref.legend(loc="best", fontsize=9)

    if ell is not None and b is not None:
        mean_ell = np.mean(ell)
        mean_b = np.mean(b)
        std_ell = np.std(ell)
        std_b = np.std(b)
        ax.plot([], [], label=(f"Radially varying $V_{{\\rm ext}}$: "
                               f"$(\\ell, b) = ({mean_ell:.1f} \\pm {std_ell:.1f}°, "
                               f"{mean_b:.1f} \\pm {std_b:.1f}°)$"),
                color="C0")

    # Always add the equivalent H0 dipole reference band (even if direction is fixed).
    add_h0_dipole_reference(ax)

    prior_bounds = _get_prior_bounds()
    if prior_bounds is not None:
        prior_lo, prior_hi = prior_bounds
        # Piecewise-linear prior envelope across knots.
        prior_lo_interp = np.interp(r, rknot, prior_lo, left=prior_lo[0], right=prior_lo[-1])
        prior_hi_interp = np.interp(r, rknot, prior_hi, left=prior_hi[0], right=prior_hi[-1])
        ax.fill_between(r, prior_lo_interp, prior_hi_interp,
                        color="black", alpha=0.08, label="Prior support")
        ax.plot(rknot, prior_lo, linestyle="--", color="black", alpha=0.5)
        ax.plot(rknot, prior_hi, linestyle="--", color="black", alpha=0.5)

    # Place the main legend at lower right if present.
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right")

    fig.tight_layout()

    if filename is not None:
        fprint(f"saving a radial Vext_mag plot to {filename}")
        fig.savefig(filename, bbox_inches="tight", dpi=450)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def plot_Vext_moll(samples_pix, fname_out, coord_in="C", coord_out="G",
                   lon_step=60, lat_step=30, eps=1e-12, nside_plot=None,
                   remove_coord_label=True, quantity_label=r"$V_{\mathrm{ext}}$",
                   unit_label=r"$\mathrm{km\ s^{-1}}$"):
    """
    Plot three stacked Mollweide maps from MCMC samples (Nsamples, Npix):
      row 1: mean
      row 2: std
      row 3: mean/std
    If nside_plot > nside(map), upsample via healpy bilinear interpolation.
    """
    mean_map = np.nanmean(samples_pix, axis=0)
    std_map = np.nanstd(samples_pix, axis=0, ddof=0)
    snr_map = mean_map / (std_map + eps)

    if nside_plot is None:
        nside_plot = 4 * hp.npix2nside(mean_map.size)

    # Upsample (optional)
    # mean_map = _upsample_map(mean_map, nside_plot)
    # std_map = _upsample_map(std_map, nside_plot)
    # snr_map = _upsample_map(snr_map, nside_plot)

    coord_arg = coord_out if coord_in == coord_out else [coord_in, coord_out]

    plt.figure(figsize=(7, 10))

    def _format_unit(prefix):
        if unit_label:
            return f"{prefix} {quantity_label} [{unit_label}]"
        return f"{prefix} {quantity_label}"

    # Mean
    hp.mollview(mean_map, nest=False, coord=coord_arg, notext=False,
                xsize=2000, cbar=True,
                unit=_format_unit("Mean"),
                title="", sub=311)
    hp.graticule(dpar=lat_step, dmer=lon_step)
    if remove_coord_label:
        ax = plt.gca()
        for t in ax.texts:
            if "Galactic" in t.get_text() or "Equatorial" in t.get_text():
                t.set_visible(False)
    _add_equator_labels(lon_step, lat_step)

    # Std
    hp.mollview(std_map, nest=False, coord=coord_arg, notext=False,
                xsize=2000, cbar=True,
                unit=_format_unit("Std"),
                title="", sub=312)
    hp.graticule(dpar=lat_step, dmer=lon_step)
    if remove_coord_label:
        ax = plt.gca()
        for t in ax.texts:
            if "Galactic" in t.get_text() or "Equatorial" in t.get_text():
                t.set_visible(False)
    _add_equator_labels(lon_step, lat_step)

    # Mean / Std
    hp.mollview(snr_map, nest=False, coord=coord_arg, notext=False,
                xsize=2000, cbar=True,
                unit=f"SNR {quantity_label}",
                title="", sub=313)
    hp.graticule(dpar=lat_step, dmer=lon_step)
    if remove_coord_label:
        ax = plt.gca()
        for t in ax.texts:
            if "Galactic" in t.get_text() or "Equatorial" in t.get_text():
                t.set_visible(False)
    _add_equator_labels(lon_step, lat_step)

    # Add padding between rows
    plt.subplots_adjust(hspace=0.35)

    plt.savefig(fname_out, dpi=450, bbox_inches="tight")
    fprint(f"saving a Mollweide map to {fname_out}")
    plt.close()


###############################################################################
#              Reading from files and other minor utilities                   #
###############################################################################


def read_gof(fname, which, raise_error=True):
    """Read goodness-of-fit statistics from a file with samples."""
    if not exists(fname) and not raise_error:
        return np.nan

    convert = which.startswith("logZ_")
    key = which.replace("logZ_", "lnZ_") if convert else which

    with File(fname, "r") as f:
        try:
            stat = float(f[f"gof/{key}"][...])
        except KeyError as e:
            raise KeyError(
                f"`{key}` not found in the file. Available keys are: "
                f"{list(f['gof'].keys())}") from e

    return stat / np.log(10) if convert else stat


def read_samples(root, fname, keys=None):
    fname = join(root, fname)

    with File(fname, "r") as f:
        if keys is None:
            keys = list(f["samples"].keys())
        elif isinstance(keys, str):
            keys = [keys]

        samples = {key: f[f"samples/{key}"][...] for key in keys}

    if isinstance(keys, list) and len(keys) == 1:
        return samples[keys[0]]
    return samples


def get_dlog_density_stats(lpA, lpB):
    """
    Compute the mean and standard deviation of the difference in log density
    between two sets of log densities of shape `(num_samples, num_objects)`.
    """
    assert lpA.ndim == lpB.ndim == 2 and lpA.shape[-1] == lpB.shape[-1]

    mu_A = np.mean(lpA, axis=0)
    mu_B = np.mean(lpB, axis=0)

    var_A = np.var(lpA, axis=0, ddof=1)
    var_B = np.var(lpB, axis=0, ddof=1)

    mean_diff = mu_A - mu_B
    std_diff = np.sqrt(var_A + var_B)

    return mean_diff, std_diff
