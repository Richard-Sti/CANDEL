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
import importlib

import numpy as np
from astropy.coordinates import SkyCoord

_DUST_MODELS = {
    "SFD": ("dustmaps.sfd", "SFDQuery"),
    "CSFD": ("dustmaps.csfd", "CSFDQuery"),
    "Planck2013": ("dustmaps.planck", "PlanckQuery"),
    "Planck2016": ("dustmaps.planck", "PlanckGNILCQuery"),
}


def read_dustmap(RA, dec, model):
    """Read off `E(B-V)` at `RA` and `dec` for a given `model`."""
    if model not in _DUST_MODELS:
        raise ValueError(f"Unsupported model: `{model}`.")

    module_name, class_name = _DUST_MODELS[model]
    try:
        mod = importlib.import_module(module_name)
        QueryClass = getattr(mod, class_name)
    except ImportError:
        raise ImportError("Cannot import `dustmaps`. Please install it.")

    coords = SkyCoord(RA, dec, unit="deg", frame="icrs")
    return np.asarray(QueryClass()(coords), dtype=np.float32)
