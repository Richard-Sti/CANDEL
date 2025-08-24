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
Interpolator for line-of-sight (LOS) data using `interpax`.
"""
from functools import partial

import jax
from jax import numpy as jnp
from jax import vmap


class LOSInterpolator:
    """
    Vectorized 1D interpolator for line-of-sight (LOS) data using JAX.

    This class precomputes interpolation functions for a batch of LOS profiles
    defined on a shared radial grid `los_r`, allowing efficient evaluation
    at per-galaxy query positions `r[i]`.

    The expected shapes are:
      - `los_r`: array of shape `(n_steps,)`
      - `f`: array of shape `(n_fields, n_galaxies, n_steps)`

    Here `n_steps` is the number of radial samples per LOS profile, and
    `n_galaxies` is the number of profiles being interpolated in parallel.
    The interpolator can then be queried with a set of galaxy radii
    `r` of shape `(n_galaxies,)` to return values of shape
    `(n_fields, n_galaxies)`.

    Extrapolation behavior:
    -----------------------
    For `r > los_r[-1]` the values follow an exponential
    approach to a constant `C = extrap_constant`:

        f(r) = C + (A - C) * exp(-(r - r_max) / r₀),

    where `A` is the last tabulated value at `r_max = los_r[-1]`, and `r₀`
    is the decay scale `r0_decay_scale`.

    This ensures that the extrapolated curve matches continuously at `r_max`
    and tends smoothly toward `C` as `r → ∞`. By default, `C = 0`, giving
    a pure exponential decay to zero.

    The method `interp_many_steps_per_galaxy` applies the same rule to
    batched arrays of evaluation radii, returning arrays of shape
    `(n_fields, n_galaxies, n_eval)` for input `r_eval` of shape `(n_eval,)`.
    """
    def __init__(self, los_r, f, r0_decay_scale=5, extrap_constant=0.):
        assert los_r.ndim == 1
        assert f.ndim == 3
        assert f.shape[-1] == los_r.shape[0]

        self.los_r = los_r
        self.f = f
        self.r0_decay_scale = r0_decay_scale
        self.extrap_constant = extrap_constant

        # Store dimensions
        self.n_field, self.n_gal, self.n_steps = f.shape

        # Define single interpolation
        r_max = self.los_r[-1]

        # Single LOS, single scalar r
        def single_interp(f_line, r_val):
            # Linear interp inside [los_r[0], r_max]; edge rule of jnp.interp
            y_lin = jnp.interp(r_val, self.los_r, f_line)

            # Exponential tail for r > r_max with amplitude fixed at last
            # sample.
            A = f_line[-1]
            C = self.extrap_constant
            y_exp = C + (A - C) * jnp.exp(-(r_val - r_max) / self.r0_decay_scale)  # noqa
            return jnp.where(r_val > r_max, y_exp, y_lin)

        # Inner vmap over galaxy axis
        vmap_gal = vmap(single_interp, in_axes=(0, 0))
        # Outer vmap over field axis, r is shared
        self._batched_interp = vmap(vmap_gal, in_axes=(0, None))

    @partial(jax.jit, static_argnums=0)
    def __call__(self, r):
        return self._batched_interp(self.f, r)

    @partial(jax.jit, static_argnums=0)
    def interp_many_steps_per_galaxy(self, r_eval):
        if r_eval.ndim != 1:
            raise ValueError(
                f"Expected r_eval to be 1D (n_eval,), got {r_eval.shape}")

        r_max = self.los_r[-1]
        A = self.f[..., -1][..., None]  # shape (n_field, n_gal, 1)

        # define interpolation over one LOS profile
        def interp_profile(y):
            return jnp.interp(r_eval, self.los_r, y)  # (n_eval,)

        # vmap over galaxies
        vmap_gal = vmap(interp_profile, in_axes=0)
        # vmap over fields
        batched_interp = vmap(vmap_gal, in_axes=0)

        y_interp = batched_interp(self.f)  # (n_field, n_gal, n_eval)

        # Exponential extrapolation
        C = self.extrap_constant
        decay = jnp.exp(-(r_eval - r_max) / self.r0_decay_scale)  # (n_eval,)
        extrap = C + (A - C) * decay  # shape (n_field, n_gal, n_eval)

        mask = r_eval > r_max  # (n_eval,)
        return jnp.where(mask, extrap, y_interp)
