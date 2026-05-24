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
"""Spiral arm traces and density profiles (Drimmel et al. 2024)."""
import logging

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def get_drimmel_arm_traces(R_sun=8.122, use_extrapolated=True, ds=None):
    """Extract per-arm Galactocentric (x, y) traces for Drimmel spiral arms.

    Parameters
    ----------
    R_sun
        Solar Galactocentric distance in kpc (default 8.122).
    use_extrapolated
        If True, use the extrapolated traces (wider azimuthal coverage).
    ds
        If not None, densify each arm trace to uniform arc-length spacing
        ``ds`` [kpc] via linear interpolation.

    Returns
    -------
    arms_xy : list of (ndarray, ndarray)
        Per-arm list of (x_gc, y_gc) coordinate arrays [kpc].
    """
    try:
        from SpiralMap.models_ import spiral_drimmel_cepheids
    except ImportError as exc:
        raise ImportError(
            "Drimmel spiral-arm traces require the optional `SpiralMap` "
            "package. Install it or use a precomputed spiral cache."
        ) from exc

    model = spiral_drimmel_cepheids()
    model.xsun = -R_sun
    model.getarmlist()

    arms_xy = []
    xkey = "xgc_ex" if use_extrapolated else "xgc"
    ykey = "ygc_ex" if use_extrapolated else "ygc"

    n_raw, n_total = 0, 0
    for arm in model.arms:
        model.output_(arm)
        x = np.asarray(model.dout[xkey])
        y = np.asarray(model.dout[ykey])
        n_raw += len(x)

        if ds is not None and len(x) > 1:
            dx = np.diff(x)
            dy = np.diff(y)
            seg = np.sqrt(dx**2 + dy**2)
            s = np.concatenate([[0], np.cumsum(seg)])
            s_new = np.arange(0, s[-1], ds)
            x = np.interp(s_new, s, x)
            y = np.interp(s_new, s, y)

        arms_xy.append((x, y))
        n_total += len(x)

    logger.info(
        f"Loaded {len(model.arms)} Drimmel arm traces "
        f"({n_raw} -> {n_total} points, "
        f"extrapolated={use_extrapolated}, ds={ds})")

    return arms_xy


def compute_dist_sq_per_arm(ell, b, d_grid, arms_xy, R_sun,
                            batch_size=2000):
    """Compute squared distance to each arm on a distance grid.

    For each arm, sightline, and distance, computes the squared distance
    to the nearest trace point of that arm using a KD-tree.

    Parameters
    ----------
    ell : array, shape (n_los,)
        Galactic longitude in degrees.
    b : array, shape (n_los,)
        Galactic latitude in degrees.
    d_grid : array, shape (n_grid,)
        Distance grid in kpc.
    arms_xy : list of (ndarray, ndarray)
        Per-arm (x_gc, y_gc) trace coordinates [kpc].
    R_sun
        Solar Galactocentric distance [kpc].
    batch_size
        Process this many sightlines at a time to limit memory.

    Returns
    -------
    dist_sq : ndarray, shape (n_arms, n_los, n_grid)
        Squared distance to nearest trace point per arm [kpc^2].
    """
    ell_rad = np.deg2rad(np.asarray(ell))
    b_rad = np.deg2rad(np.asarray(b))
    d_grid = np.asarray(d_grid)

    n_arms = len(arms_xy)
    n_los = len(ell_rad)
    n_grid = len(d_grid)
    out = np.empty((n_arms, n_los, n_grid))

    trees = [cKDTree(np.column_stack(xy)) for xy in arms_xy]

    for start in range(0, n_los, batch_size):
        end = min(start + batch_size, n_los)
        cos_b = np.cos(b_rad[start:end])[:, None]
        cos_l = np.cos(ell_rad[start:end])[:, None]
        sin_l = np.sin(ell_rad[start:end])[:, None]
        d = d_grid[None, :]

        x_gc = (d * cos_b * cos_l - R_sun).ravel()
        y_gc = (d * cos_b * sin_l).ravel()
        pts = np.column_stack([x_gc, y_gc])

        for k, tree in enumerate(trees):
            dist, _ = tree.query(pts)
            out[k, start:end] = (dist**2).reshape(end - start, n_grid)

    return out
