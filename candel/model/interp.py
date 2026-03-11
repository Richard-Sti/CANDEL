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
"""
Linear interpolator for line-of-sight (LOS) data on a uniform radial grid.
"""
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp


def _check_uniform_grid(r, rtol=1e-4):
    """Verify that `r` is a uniform grid. Raises ValueError if not."""
    r_np = np.asarray(r)
    if r_np.ndim != 1 or len(r_np) < 2:
        raise ValueError(
            f"Expected 1D array with >= 2 points, got {r_np.shape}")
    dr = np.diff(r_np)
    mean_dr = (r_np[-1] - r_np[0]) / (len(r_np) - 1)
    if np.any(np.abs(dr - mean_dr) / mean_dr > rtol):
        raise ValueError(
            f"Grid is not uniform: max relative deviation "
            f"{np.max(np.abs(dr - mean_dr) / mean_dr):.2e} > rtol={rtol}")


def _uniform_idx(r_query, r_min, dr, n_steps):
    """Compute interpolation index and fractional offset on a uniform grid.

    Clamps to boundary values for queries outside [r_min, r_max].
    """
    idx_cont = jnp.clip((r_query - r_min) / dr, 0.0, n_steps - 1.0)
    idx_lo = jnp.floor(idx_cont).astype(jnp.int32).clip(0, n_steps - 2)
    t = idx_cont - idx_lo
    return idx_lo, t


class LOSInterpolator:
    """
    Vectorized 1D interpolator for line-of-sight (LOS) data on a uniform grid.

    The expected shapes are:
      - `los_r`: array of shape `(n_steps,)` — must be uniform
      - `f`: array of shape `(n_fields, n_galaxies, n_steps)`

    The interpolator can be queried with:
      - `__call__(r)` where `r` has shape `(n_galaxies,)`, returning
        `(n_fields, n_galaxies)`.
      - `interp_many(r_eval)` where `r_eval` has shape `(n_eval,)`,
        returning `(n_fields, n_galaxies, n_eval)`.

    Extrapolation:
      For `r > r_max`, values follow an exponential decay:
          f(r) = C + (A - C) * exp(-(r - r_max) / r0)
      where A is the last tabulated value, C = `extrap_constant`,
      and r0 = `r0_decay_scale`.
    """
    def __init__(self, los_r, f, r0_decay_scale=5, extrap_constant=0.):
        _check_uniform_grid(los_r)

        assert f.ndim == 3
        assert f.shape[-1] == len(los_r)

        self.f = jnp.asarray(f)
        # Store grid params in the same dtype as los_r
        self.r_min = los_r[0]
        self.r_max = los_r[-1]
        self.dr = (los_r[-1] - los_r[0]) / (len(los_r) - 1)
        self.r0_decay_scale = r0_decay_scale
        self.extrap_constant = extrap_constant
        self.n_field, self.n_gal, self.n_steps = f.shape

    @partial(jax.jit, static_argnums=0)
    def __call__(self, r):
        """Interpolate at per-galaxy radii r of shape (n_gal,).

        Returns shape (n_field, n_gal).
        """
        idx_lo, t = _uniform_idx(r, self.r_min, self.dr, self.n_steps)
        # Per-galaxy indexing: f[:, i, idx_lo[i]]
        g = jnp.arange(self.n_gal)
        val_lo = self.f[:, g, idx_lo]
        val_hi = self.f[:, g, idx_lo + 1]
        y_lin = val_lo + t * (val_hi - val_lo)

        # Exponential extrapolation for r > r_max
        A = self.f[..., -1]
        C = self.extrap_constant
        y_exp = C + (A - C) * jnp.exp(
            -(r - self.r_max) / self.r0_decay_scale)

        return jnp.where(r > self.r_max, y_exp, y_lin)

    @partial(jax.jit, static_argnums=0)
    def interp_many(self, r_eval):
        """Interpolate at shared radii r_eval of shape (n_eval,).

        Returns shape (n_field, n_gal, n_eval).
        """
        idx_lo, t = _uniform_idx(r_eval, self.r_min, self.dr, self.n_steps)
        # Shared index: f[..., idx_lo] → (n_field, n_gal, n_eval)
        val_lo = self.f[..., idx_lo]
        val_hi = self.f[..., idx_lo + 1]
        y_lin = val_lo + t * (val_hi - val_lo)

        # Exponential extrapolation
        A = self.f[..., -1][..., None]
        C = self.extrap_constant
        decay = jnp.exp(-(r_eval - self.r_max) / self.r0_decay_scale)
        y_exp = C + (A - C) * decay

        return jnp.where(r_eval > self.r_max, y_exp, y_lin)

    # Keep old name as alias
    interp_many_steps_per_galaxy = interp_many
