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
from interpax import Interpolator1D, Interpolator2D, interp1d
from jax import numpy as jnp
from jax import vmap
from scipy.interpolate import CubicSpline

from .util import SPEED_OF_LIGHT


class Distmod2Distance:
    """
    Class to build an interpolator to convert distance modulus to comoving
    distance in `Mpc`. Choice of `h` is determined when calling the
    `__call__` method.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    is_scalar : bool
        If `True`, the interpolator is not vectorized. This is useful for
        debugging, but should be set to `False` for performance.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000, is_scalar=False):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        mu_grid = cosmo.distmod(z_grid).value

        f = Interpolator1D(mu_grid, jnp.log(r_grid), extrap=False)
        if not is_scalar:
            f = vmap(f)
        self._f = f

    def __call__(self, mu, h=1, return_log=False):
        if return_log:
            return self._f(mu + 5 * jnp.log10(h)) - jnp.log(h)

        return jnp.exp(self._f(mu + 5 * jnp.log10(h))) / h


class Distance2Distmod:
    """
    Class to build an interpolator to convert distance in `Mpc` to distance
    modulus. Choice of `h` is determined when calling the `__call__` method.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    is_scalar : bool
        If `True`, the interpolator is not vectorized. This is useful for
        debugging, but should be set to `False` for performance.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000, is_scalar=False):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        mu_grid = cosmo.distmod(z_grid).value

        f = Interpolator1D(jnp.log(r_grid), mu_grid, extrap=False)
        if not is_scalar:
            f = vmap(f)
        self._f = f

    def __call__(self, r, h=1,):
        return self._f(jnp.log(r * h)) - 5 * jnp.log10(h)


class Distance2LogAngDist:
    """
    Class to build an interpolator to convert distance in `Mpc` to log angular
    diameter distance. `h` is assumed to be one.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    is_scalar : bool
        If `True`, the interpolator is not vectorized. This is useful for
        debugging, but should be set to `False` for performance.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000, is_scalar=False):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        log_da_grid = jnp.log10(cosmo.angular_diameter_distance(z_grid).value)

        f = Interpolator1D(jnp.log(r_grid), log_da_grid, extrap=False)
        if not is_scalar:
            f = vmap(f)
        self._f = f

    def __call__(self, r):
        return self._f(jnp.log(r))


class Distance2LogLumDist:
    """
    Class to build an interpolator to convert distance in `Mpc` to log
    luminosity distance. `h` is assumed to be one.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    is_scalar : bool
        If `True`, the interpolator is not vectorized. This is useful for
        debugging, but should be set to `False` for performance.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000, is_scalar=False):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        log_dl_grid = jnp.log10(cosmo.luminosity_distance(z_grid).value)

        f = Interpolator1D(jnp.log(r_grid), log_dl_grid, extrap=False)
        if not is_scalar:
            f = vmap(f)
        self._f = f

    def __call__(self, r):
        return self._f(jnp.log(r))


class LogAngularDiameterDistance2Distmod:
    """
    Class to build an interpolator to convert log angular diameter distance in
    `Mpc` to distance modulus. Choice of `h` is determined when calling the
    `__call__` method.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        da_grid = cosmo.angular_diameter_distance(z_grid).value
        mu_grid = cosmo.distmod(z_grid).value

        self._f = vmap(
            Interpolator1D(jnp.log10(da_grid), mu_grid, extrap=False))

    def __call__(self, logdA, h=1,):
        return self._f(logdA + jnp.log10(h)) - 5 * jnp.log10(h)


class Distmod2Redshift:
    """
    Class to build an interpolator to convert distance modulus to comoving
    distance in `Mpc`. Choice of `h` is determined when calling the
    `__call__` method.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        mu_grid = cosmo.distmod(z_grid).value

        self._f = vmap(Interpolator1D(mu_grid, jnp.log(z_grid), extrap=False))

    def __call__(self, mu, h=1, return_log=False):
        if return_log:
            return self._f(mu + 5 * jnp.log10(h))

        return jnp.exp(self._f(mu + 5 * jnp.log10(h)))


class Redshift2Distance:
    """
    Class to build an interpolator to convert redshift to distance modulus.
    Choice of `h` is determined when calling the `__call__` method.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000, is_scalar=False):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value

        f = Interpolator1D(z_grid, r_grid, extrap=False)
        f_cz = Interpolator1D(
            z_grid * SPEED_OF_LIGHT, r_grid, extrap=False)

        if not is_scalar:
            f = vmap(f)
            f_cz = vmap(f_cz)

        self._f = f
        self._f_cz = f_cz

    def __call__(self, z, h=1, is_velocity=False):
        if is_velocity:
            return self._f_cz(z) / h

        return self._f(z) / h


class Redshift2Distmod:
    """
    Class to build an interpolator to convert redshift to distance
    modulus. Choice of `h` is determined when calling the
    `__call__` method.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        mu_grid = cosmo.distmod(z_grid).value

        self._f = vmap(Interpolator1D(jnp.log(z_grid), mu_grid, extrap=False))

    def __call__(self, z, h=1, ):
        return self._f(jnp.log(z)) - 5 * jnp.log10(h)


class Distance2Redshift:
    """
    Class to build an interpolator to convert comoving distance in `Mpc`
    to redshift. Choice of `h` is determined when calling the
    `__call__` method.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value

        self._f = vmap(Interpolator1D(r_grid, z_grid, extrap=False))

    def __call__(self, r, h=1):
        return self._f(r * h)


class LogGrad_Distmod2ComovingDistance:
    """
    Class to build an interpolator to compute the log gradient of the comoving
    distance in `Mpc / h` with respect to distance modulus. Choice of `h` is
    determined when calling the `__call__` method.

    The function is: `log (dr / dmu) | mu`.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.logspace(np.log10(zmin_interp), np.log10(zmax_interp),
                             npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        mu_grid = cosmo.distmod(z_grid).value

        spline = CubicSpline(mu_grid, r_grid, extrapolate=False)
        drdmu = spline.derivative()(mu_grid)

        self._f = vmap(Interpolator1D(mu_grid, jnp.log(drdmu), extrap=False))

    def __call__(self, mu, h=1):
        return self._f(mu + 5 * jnp.log10(h)) - jnp.log(h)


###############################################################################
#                   Interpolators sampling Om as well                         #
###############################################################################

class Distance2Distmod_withOm:
    """
    Interpolator to convert distance in `Mpc` to distance modulus, as a
    function of `h` and `Om`, which are specified on the fly.
    """
    def __init__(self, rmin=1e-3, rmax=500, nr=500,
                 Om_min=0.01, Om_max=0.99, nOm=500,
                 zmin_outer=1e-9, zmax_outer=0.3, method='cubic'):
        r_grid = jnp.logspace(np.log10(rmin), np.log10(rmax), nr)
        Om_grid = jnp.linspace(Om_min, Om_max, nOm)
        # z_grid = np.linspace(zmin_outer, zmax_outer, nr)
        z_grid = np.logspace(np.log10(zmin_outer), np.log10(zmax_outer), nr)

        # Precompute distance modulus grid
        mu_grid = np.empty((nr, nOm))
        for j, Om in enumerate(Om_grid):
            cosmo = FlatLambdaCDM(H0=100, Om0=Om)
            r = jnp.asarray(cosmo.comoving_distance(z_grid).value)
            mu = jnp.asarray(cosmo.distmod(z_grid).value)

            mu_grid[:, j] = interp1d(r_grid, r, mu, method=method)
            if np.any(np.isnan(mu_grid[:, j])):
                raise ValueError(
                    "The distance grid is not fully covered for "
                    f"Om = {Om:.2f}. Try increasing `redshift ranges`.")

        # Build the interpolator: f(z, Om) -> mu
        self._interp = Interpolator2D(
            x=r_grid,
            y=Om_grid,
            f=jnp.asarray(mu_grid),
            method=method,
            extrap=False,
        )

    def __call__(self, r, Om, h):
        return self._interp(r * h, Om) - 5 * jnp.log10(h)


class Distance2Redshift_withOm:
    """
    Interpolator to convert distance in `Mpc` to redshift, as a
    function of `h` and `Om`, which are specified on the fly.
    """
    def __init__(self, rmin=1e-3, rmax=500, nr=500,
                 Om_min=0.01, Om_max=0.99, nOm=500,
                 zmin_outer=1e-9, zmax_outer=0.3, method='cubic'):
        r_grid = jnp.logspace(np.log10(rmin), np.log10(rmax), nr)
        Om_grid = jnp.linspace(Om_min, Om_max, nOm)
        z_grid_fixed = np.logspace(
            np.log10(zmin_outer), np.log10(zmax_outer), nr)

        # Precompute distance modulus grid
        z_grid = np.empty((nr, nOm))
        for j, Om in enumerate(Om_grid):
            cosmo = FlatLambdaCDM(H0=100, Om0=Om)

            r = jnp.asarray(cosmo.comoving_distance(z_grid_fixed).value)
            z_grid[:, j] = interp1d(r_grid, r, z_grid_fixed, method=method)
            if np.any(np.isnan(z_grid[:, j])):
                raise ValueError(
                    "The distance grid is not fully covered for "
                    f"Om = {Om:.2f}. Try increasing `redshift ranges`.")

        # Build the interpolator: f(z, Om) -> mu
        self._interp = Interpolator2D(
            x=r_grid,
            y=Om_grid,
            f=jnp.asarray(z_grid),
            method=method,
            extrap=False,
        )

    def __call__(self, r, Om, h):
        return self._interp(r * h, Om)


###############################################################################
#                           Cosmographic expansion                            #
###############################################################################


def redshift_to_dL_cosmography(z, H0, q0=-0.55, j0=1, s0=0.055):
    """
    Calculate the luminosity distance for a given redshift using cosmographic
    expansion up to second order in redshift.
    """
    return (SPEED_OF_LIGHT * z) / H0 * (
        1
        + 0.5 * (1 - q0) * z
        - (1 / 6) * (1 - q0 - 3 * q0**2 + j0) * z**2
        + (1 / 24) * (2 - 2 * q0 - 15 * q0**2 - 15 * q0**3 + 5 * j0 + 10 * q0 *  j0 + s0) * z**3  # noqa
        )
