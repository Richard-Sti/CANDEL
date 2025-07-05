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
    `(n_galaxies, n_steps)`, where `n_steps` is the number of radial steps
    and `n_galaxies` is the number of galaxies. The interpolator can then be
    queried with a radial position `r` of shape `(n_galaxies,)`.

    Note: the method `interp_many_steps_per_galaxy` supports extrapolation
    beyond the last `los_r` value using an exponential decay
    `f(r) ∝ exp(-r / r₀)` with a fixed decay scale `r₀`.
    """

    def __init__(self, los_r, f, method="cubic", r0_decay_many_steps=5,
                 extrap=False):
        assert f.ndim == 2 and los_r.ndim == 1
        assert f.shape[1] == los_r.shape[0]

        def single_interp(fi, ri):
            interp = Interpolator1D(los_r, fi, method=method, extrap=extrap)
            return interp(ri)

        self._batched_interp = jax.vmap(single_interp, in_axes=(0, 0))
        self._los_r = los_r
        self._frozen_f = f
        self.r0_decay_many_steps = r0_decay_many_steps

    @partial(jax.jit, static_argnums=0)
    def interp_many_steps_per_galaxy(self, r_eval):
        r_max = self._los_r[-1]
        y_interp = vmap(lambda y, rq: jnp.interp(rq, self._los_r, y))(
            self._frozen_f, r_eval)

        # Use the last value at r_max as amplitude A
        A = self._frozen_f[:, -1]
        extrap = A[:, None] * jnp.exp(
            -(r_eval - r_max) / self.r0_decay_many_steps)
        return jnp.where(r_eval > r_max, extrap, y_interp)

    @partial(jax.jit, static_argnums=0)
    def __call__(self, r):
        return self._batched_interp(self._frozen_f, r)
