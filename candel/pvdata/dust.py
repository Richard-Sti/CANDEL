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
"""Dust maps support."""
import numpy as np
from astropy.coordinates import SkyCoord


def read_dustmap(RA, dec, model):
    """Read off `E(B-V)` at `RA` and `dec` for a given `model`."""
    coords = SkyCoord(RA, dec, unit="deg", frame="icrs")

    if model == "SFD":
        try:
            from dustmaps.sfd import SFDQuery
        except ImportError:
            raise ImportError("Cannot import `dustmaps`. Please install it.")
        query = SFDQuery()
    elif model == "CSFD":
        try:
            from dustmaps.csfd import CSFDQuery
        except ImportError:
            raise ImportError("Cannot import `dustmaps`. Please install it.")
        query = CSFDQuery()
    elif model == "Planck2013":
        try:
            from dustmaps.planck import PlanckQuery
        except ImportError:
            raise ImportError("Cannot import `dustmaps`. Please install it.")
        query = PlanckQuery()
    elif model == "Planck2016":
        try:
            from dustmaps.planck import PlanckGNILCQuery
        except ImportError:
            raise ImportError("Cannot import `dustmaps`. Please install it.")
        query = PlanckGNILCQuery()
    else:
        raise ValueError(f"Unsupported model: `{model}`.")

    return np.asarray(query(coords), dtype=np.float32)
