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

    The expected shapes are `los_r` of `(n_steps, )`, and `f` of
    `(n_galaxies, n_steps)`, where `n_steps` is the number of radial steps
    and `n_galaxies` is the number of galaxies. The interpolator can then be
    queried with a radial position `r` of shape `(n_galaxies, )`.
    """

    def __init__(self, los_r, f, method="cubic", extrap=False):
        assert f.ndim == 2 and los_r.ndim == 1
        assert f.shape[1] == los_r.shape[0]

        def single_interp(fi, ri):
            interp = Interpolator1D(los_r, fi, method=method, extrap=extrap)
            return interp(ri)

        self._batched_interp = jax.vmap(single_interp, in_axes=(0, 0))
        self._los_r = los_r
        self._frozen_f = f

    @partial(jax.jit, static_argnums=0)
    def interp_many_steps_per_galaxy(self, r_eval):
        """
        Interpolate a set of steps `r_eval` of shape (`n_gal, n_eval`) for a
        set of LOS densities `los_density` of shape (`n_gal, n_steps`) at the
        radial positions `los_r` of shape (`n_steps, `). Returns an array of
        shape (`n_gal, n_eval`).
        """
        return vmap(
            lambda y, rq: jnp.interp(rq, self._los_r, y))(self._frozen_f, r_eval)  # noqa

    @partial(jax.jit, static_argnums=0)
    def __call__(self, r):
        return self._batched_interp(self._frozen_f, r)
