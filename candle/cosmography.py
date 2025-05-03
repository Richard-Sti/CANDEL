# Copyright (C) 2024 Richard Stiskalek
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
"""Various cosmography functions for converting between distance indicators."""
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from interpax import Interpolator1D
from scipy.interpolate import CubicSpline
from jax import numpy as jnp
from jax import vmap


class Distmod2Distance:
    """
    Class to build an interpolator to convert distance modulus to comoving
    distance in `Mpc / h`.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    H0 : float
        Hubble constant in `km / s / Mpc`.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, H0=100, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        mu_grid = cosmo.distmod(z_grid).value

        self._f = vmap(Interpolator1D(mu_grid, jnp.log(r_grid), extrap=False))

    def __call__(self, mu, return_log=False):
        if return_log:
            return self._f(mu)

        return jnp.exp(self._f(mu))


class Distmod2Redshift:
    """
    Class to build an interpolator to convert distance modulus to comoving
    distance in `Mpc / h`.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    H0 : float
        Hubble constant in `km / s / Mpc`.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, H0=100, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        mu_grid = cosmo.distmod(z_grid).value

        self._f = vmap(Interpolator1D(mu_grid, jnp.log(z_grid), extrap=False))

    def __call__(self, r, return_log=False):
        if return_log:
            return self._f(r)

        return jnp.exp(self._f(r))


class LogGrad_Distmod2ComovingDistance:
    """
    Class to build an interpolator to compute the log gradient of the comoving
    distance in `Mpc / h` with respect to distance modulus.

    The function is: `log (dr / dmu) | mu`.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    H0 : float
        Hubble constant in `km / s / Mpc`.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, H0=100, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        z_grid = np.logspace(np.log10(zmin_interp), np.log10(zmax_interp),
                             npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        mu_grid = cosmo.distmod(z_grid).value

        spline = CubicSpline(mu_grid, r_grid, extrapolate=False)
        drdmu = spline.derivative()(mu_grid)

        self._f = vmap(Interpolator1D(mu_grid, jnp.log(drdmu), extrap=False))

    def __call__(self, mu):
        return self._f(mu)
