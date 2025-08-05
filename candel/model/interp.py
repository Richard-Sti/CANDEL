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
from interpax import Interpolator1D
from jax import numpy as jnp
from jax import vmap


class LOSInterpolator:
    """
    Vectorized 1D interpolator for line-of-sight (LOS) data using interpax.

    This class precomputes interpolation functions for a batch of LOS profiles
    defined on a shared radial grid `los_r`, allowing efficient evaluation
    at per-galaxy query positions `r[i]`.

    The expected shapes are `los_r` of `(n_steps,)`, and `f` of
    `(n_fields, n_galaxies, n_steps)`, where `n_steps` is the number of radial
    steps and `n_galaxies` is the number of galaxies. The interpolator can then
    be queried with a radial position `r` of shape `(n_galaxies, )` to return
    an array of shape `(n_fields, n_galaxies)`.

    Note: the method `interp_many_steps_per_galaxy` supports extrapolation
    beyond the last `los_r` value using an exponential decay
    `f(r) ∝ exp(-r / r₀)` with a fixed decay scale `r₀`.
    """
    def __init__(self, los_r, f, method="linear", r0_decay_many_steps=5,
                 extrap=False):
        assert los_r.ndim == 1
        assert f.ndim == 3
        assert f.shape[-1] == los_r.shape[0]

        self.los_r = los_r
        self.f = f
        self.method = method
        self.extrap = extrap
        self.r0_decay = r0_decay_many_steps

        # store dimensions
        self.n_field, self.n_gal, self.n_steps = f.shape

        # Define single interpolation
        def single_interp(f_line, r_val):
            return Interpolator1D(
                los_r, f_line, method=method, extrap=extrap)(r_val)

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
        decay = jnp.exp(-(r_eval - r_max) / self.r0_decay)  # (n_eval,)
        extrap = A * decay  # (n_field, n_gal, n_eval)

        mask = r_eval > r_max  # (n_eval,)
        return jnp.where(mask, extrap, y_interp)
