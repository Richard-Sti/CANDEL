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
"""Helpers for optional random angular scatter of tracer positions."""
import numpy as np

from ..util import fprint, get_nested, scatter_radec


def angular_position_scatter_from_config(config):
    """Return active angular-scatter settings, or ``None`` if disabled."""
    sigma = get_nested(config, "io/angular_position_scatter_deg", 0.0)
    if sigma is None:
        return None
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError(
            "`io.angular_position_scatter_deg` must be a finite "
            f"non-negative value, got {sigma!r}.")
    if sigma == 0.0:
        return None

    seed = get_nested(config, "io/angular_position_scatter_seed", 42)
    if seed is None or isinstance(seed, bool):
        raise ValueError(
            "`io.angular_position_scatter_seed` must be an integer.")
    return {"sigma_deg": sigma, "seed": int(seed)}


def catalogue_scatter_from_config(config):
    """Return active scatter settings from a catalogue config dictionary."""
    sigma = config.get("angular_position_scatter_deg", 0.0)
    if sigma is None:
        return None
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError(
            "`angular_position_scatter_deg` must be a finite non-negative "
            f"value, got {sigma!r}.")
    if sigma == 0.0:
        return None

    seed = config.get("angular_position_scatter_seed", 42)
    if seed is None or isinstance(seed, bool):
        raise ValueError("`angular_position_scatter_seed` must be an integer.")
    return {"sigma_deg": sigma, "seed": int(seed)}


def _coordinate_keys(data):
    if "RA" in data:
        ra_key = "RA"
    elif "RA_host" in data:
        ra_key = "RA_host"
    else:
        raise KeyError(
            "Cannot scatter coordinates without an `RA` or `RA_host` array.")

    if "dec" in data:
        dec_key = "dec"
    elif "DEC" in data:
        dec_key = "DEC"
    elif "dec_host" in data:
        dec_key = "dec_host"
    else:
        raise KeyError(
            "Cannot scatter coordinates without a `dec`, `DEC`, or "
            "`dec_host` array.")

    return ra_key, dec_key


def scatter_data_coordinates(data, scatter, keys=None, label="catalogue"):
    """Scatter a data dictionary's RA/dec arrays in place."""
    if scatter is None:
        return False
    if keys is None:
        keys = _coordinate_keys(data)

    ra_key, dec_key = keys
    RA, dec = scatter_radec(
        data[ra_key], data[dec_key],
        scatter["sigma_deg"], scatter["seed"])
    data[ra_key] = RA
    data[dec_key] = dec
    fprint(
        f"scattered {label} angular positions with "
        f"sigma={scatter['sigma_deg']:g} deg, seed={scatter['seed']}.")
    return True
