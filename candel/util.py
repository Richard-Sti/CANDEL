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

from datetime import datetime

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from corner import corner
import tomllib

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


def load_config(config_path):
    """
    Load a TOML configuration file and convert "none" strings to None.
    """
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    return convert_none_strings(config)


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


###############################################################################
#                               Plotting                                      #
###############################################################################


def name2label(name):
    latex_labels = {
        "a_TFR": r"$a_\mathrm{TFR}$",
        "b_TFR": r"$b_\mathrm{TFR}$",
        "sigma_mu": r"$\sigma_\mu$",
        "sigma_v": r"$\sigma_v$",
        "Vext_mag": r"$V_\mathrm{ext}$",
        "Vext_ell": r"$\ell_\mathrm{ext}$",
        "Vext_b": r"$b_\mathrm{ext}$",
        "h": r"$h$",
    }
    return latex_labels.get(name, name)


def plot_corner(samples, filename=None, smooth=1, keys=None):
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
        fig.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
