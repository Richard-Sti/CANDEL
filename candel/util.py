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
    import tomllib                                                              # noqa
except ModuleNotFoundError:
    # Backport for <=3.10
    import tomli as tomllib

from datetime import datetime
from os.path import abspath, isabs, join

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from corner import corner

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
        "eta_prior_mean": r"$\hat{\eta}$",
        "eta_prior_std": r"$w_\eta$",
        "A_CL": r"$A_{\rm CL}$",
        "B_CL": r"$B_{\rm CL}$",
        "C_CL": r"$C_{\rm CL}$",
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
        fprint(f"saving a corner plot to `{filename}`")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        fig.show()
    else:
        plt.close(fig)
