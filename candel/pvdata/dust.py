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

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

_BAYESTAR = None
_MARSHALL = None

_DUST_MODELS = {
    "SFD": ("dustmaps.sfd", "SFDQuery"),
    "CSFD": ("dustmaps.csfd", "CSFDQuery"),
    "Planck2013": ("dustmaps.planck", "PlanckQuery"),
    "Planck2016": ("dustmaps.planck", "PlanckGNILCQuery"),
}

R_H_BAYESTAR = 0.469
AKS_TO_AH = 1.55


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


def _get_bayestar():
    """Get or initialize the Bayestar19 dust map."""
    global _BAYESTAR
    if _BAYESTAR is None:
        try:
            from dustmaps.bayestar import BayestarQuery
        except ImportError as exc:
            raise ImportError(
                "Bayestar extinction queries require the optional `dustmaps` "
                "package. Install it to use MW Cepheid 3D dust maps."
            ) from exc
        _BAYESTAR = BayestarQuery(version="bayestar2019")
    return _BAYESTAR


def _get_marshall():
    """Get or initialize the Marshall06 dust map."""
    global _MARSHALL
    if _MARSHALL is None:
        try:
            from dustmaps.marshall import MarshallQuery
        except ImportError as exc:
            raise ImportError(
                "Marshall extinction queries require the optional `dustmaps` "
                "package. Install it to use MW Cepheid 3D dust maps."
            ) from exc
        _MARSHALL = MarshallQuery()
    return _MARSHALL


def _make_galactic_coords(ell, b, dist_kpc):
    """Create ``SkyCoord`` from Galactic coordinates and distances."""
    ell = np.atleast_1d(np.asarray(ell, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    dist_kpc = np.atleast_1d(np.asarray(dist_kpc, dtype=float))
    ell, b, dist_kpc = np.broadcast_arrays(ell, b, dist_kpc)
    return SkyCoord(l=ell * u.deg, b=b * u.deg,
                    distance=dist_kpc * u.kpc, frame="galactic")


def postprocess_extinction_profiles(AH):
    """Forward-fill NaN and enforce monotonicity in extinction profiles."""
    AH = np.copy(AH)
    AH[~np.isfinite(AH)] = np.nan
    for i in range(AH.shape[0]):
        row = AH[i]
        mask = np.isnan(row)
        if mask.any() and not mask.all():
            idx = np.where(~mask, np.arange(len(row)), 0)
            np.maximum.accumulate(idx, out=idx)
            row[mask] = row[idx[mask]]
        AH[i] = row
    AH = np.maximum.accumulate(AH, axis=1)
    AH = np.where(np.isfinite(AH), AH, 0.0)
    return AH


def query_AH(ell, b, dist_kpc, map_name="bayestar", return_std=False):
    """Query H-band extinction ``A_H`` from Galactic coordinates."""
    coords = _make_galactic_coords(ell, b, dist_kpc)

    if map_name == "bayestar":
        dust = _get_bayestar()
        reddening = dust(coords, mode="mean")
        A_H = R_H_BAYESTAR * reddening
        if return_std:
            samples = np.asarray(dust(coords, mode="samples"))
            A_H_std = R_H_BAYESTAR * np.nanstd(samples, axis=-1)
    elif map_name == "marshall":
        dust = _get_marshall()
        A_Ks = dust(coords)
        A_H = AKS_TO_AH * A_Ks
        if return_std:
            A_H_std = np.zeros_like(np.atleast_1d(np.asarray(A_H)),
                                    dtype=float)
    elif map_name == "bayestar+marshall":
        scalar = coords.isscalar
        dust_b = _get_bayestar()
        reddening = dust_b(coords, mode="mean")
        A_H = np.atleast_1d(R_H_BAYESTAR * np.asarray(reddening))
        if return_std:
            samples = np.asarray(dust_b(coords, mode="samples"))
            A_H_std = np.atleast_1d(
                R_H_BAYESTAR * np.nanstd(samples, axis=-1))
        nan_mask = ~np.isfinite(A_H)
        if nan_mask.any():
            dust_m = _get_marshall()
            A_Ks = dust_m(coords[nan_mask])
            A_H[nan_mask] = AKS_TO_AH * np.asarray(A_Ks)
            if return_std:
                A_H_std[nan_mask] = 0.0
        if scalar:
            A_H = A_H[0]
            if return_std:
                A_H_std = A_H_std[0]
    else:
        raise ValueError(f"Unknown dust map: {map_name}. "
                         "Use 'bayestar', 'marshall', "
                         "or 'bayestar+marshall'.")

    if not return_std:
        if np.ndim(A_H) == 0 or (hasattr(A_H, "size") and A_H.size == 1):
            return float(A_H)
        return np.asarray(A_H)

    A_H = np.asarray(A_H) if np.ndim(A_H) > 0 else float(A_H)
    A_H_std = np.asarray(A_H_std) if np.ndim(A_H_std) > 0 else float(A_H_std)
    return A_H, A_H_std


def query_AH_grid(ell, b, d_grid, map_name="bayestar", return_std=False):
    """Query ``A_H`` for multiple Galactic sightlines on a distance grid."""
    ell = np.atleast_1d(np.asarray(ell, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    d_grid = np.atleast_1d(np.asarray(d_grid, dtype=float))

    n_los = len(ell)
    n_grid = len(d_grid)
    ell_flat = np.repeat(ell, n_grid)
    b_flat = np.repeat(b, n_grid)
    d_flat = np.tile(d_grid, n_los)

    result = query_AH(ell_flat, b_flat, d_flat, map_name=map_name,
                      return_std=return_std)

    if return_std:
        AH_flat, AH_std_flat = result
        AH = np.asarray(AH_flat).reshape(n_los, n_grid)
        AH_std = np.asarray(AH_std_flat).reshape(n_los, n_grid)
        valid = np.isfinite(AH)
        return AH, valid, AH_std

    AH = np.asarray(result).reshape(n_los, n_grid)
    valid = np.isfinite(AH)
    return AH, valid


def query_reddening(ell, b, dist_kpc, map_name="bayestar"):
    """Query reddening from a 3D Galactic dust map."""
    coords = _make_galactic_coords(ell, b, dist_kpc)

    if map_name == "bayestar":
        dust = _get_bayestar()
        E_BV = dust(coords, mode="mean")
    elif map_name == "marshall":
        dust = _get_marshall()
        A_Ks = dust(coords)
        E_BV = A_Ks / 0.306
    else:
        raise ValueError(f"Unknown dust map: {map_name}. "
                         f"Use 'bayestar' or 'marshall'.")

    if np.ndim(E_BV) == 0 or (hasattr(E_BV, "size") and E_BV.size == 1):
        return float(E_BV)
    return np.asarray(E_BV)
