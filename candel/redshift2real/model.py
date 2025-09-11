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
Module for mapping observed redshift to cosmological redshift given some
calibrated density and velocity field.
"""
from abc import ABC, abstractmethod

from jax import numpy as jnp
from jax.scipy.special import logsumexp
from numpyro import factor, plate, sample
from numpyro.distributions import Normal, Uniform

from ..cosmography import Distance2Redshift
from ..model import LOSInterpolator, ln_simpson
from ..util import SPEED_OF_LIGHT, fprint, radec_to_cartesian


def log_mean_exp(logp, axis=-1):
    return logsumexp(logp, axis=axis) - jnp.log(logp.shape[axis])


class BaseRedshift2Real(ABC):
    """Base class for all models. """

    def __init__(self, RA, dec, zcmb, los_r, los_density, los_velocity,
                 calibration_samples=None, Rmin=1e-7, Rmax=300,
                 num_rgrid=101, r0_decay_scale=5, Om0=0.3):
        self.dist2redshift = Distance2Redshift(Om0=Om0)

        self.Rmin = Rmin
        self.Rmax = Rmax
        assert Rmin > 0 and Rmax > Rmin
        assert num_rgrid % 2 == 1

        self.len_input_data = len(zcmb)
        self.cz_cmb = jnp.asarray(zcmb * SPEED_OF_LIGHT)  # in km/s
        print("SETTING 1 FIELD")

        los_r = jnp.asarray(los_r)
        los_density = jnp.asarray(los_density)
        los_velocity = jnp.asarray(los_velocity)

        self.f_los_delta = LOSInterpolator(
            los_r, los_density - 1, r0_decay_scale=r0_decay_scale)
        self.f_los_velocity = LOSInterpolator(
            los_r, los_velocity, r0_decay_scale=r0_decay_scale)

        rhat = radec_to_cartesian(RA, dec)

        self.los_grid_r = jnp.linspace(Rmin, Rmax, num_rgrid)

        self.calibration_samples = calibration_samples
        calibration_keys = list(calibration_samples.keys())
        self.num_cal = calibration_samples[calibration_keys[0]].shape[0]

        # LOS Vext, (ngal, ncalibration_samples)
        if "Vext" in calibration_samples:
            self.Vext_radial = jnp.sum(
                rhat[:, None, :] * calibration_samples["Vext"][None, :, :],
                axis=-1)
        else:
            fprint("No Vext in calibration samples.")
            self.Vext_radial = jnp.zeros(
                (self.len_input_data, self.num_cal))

        self.use_im = True
        if "b1" in calibration_samples:
            self.b1 = jnp.asarray(calibration_samples["b1"])
            self.lp_norm = self.compute_linear_bias_lp_normalization(
                self.los_grid_r)
        else:
            fprint("b1 not in calibration samples. Not using inhomogeneous "
                   "Malmquist bias.")
            self.use_im = False

        self.sigma_v = jnp.asarray(calibration_samples["sigma_v"])
        if "beta" in calibration_samples:
            self.beta = jnp.asarray(calibration_samples["beta"])
        else:
            fprint("Beta not in calibration samples. Setting beta=1.")
            self.beta = jnp.ones_like(self.sigma_v)

        fprint(f"Loaded {self.len_input_data} objects and "
               f"{self.num_cal} calibration samples.")

    def compute_linear_bias_lp_normalization(self, los_grid_r):
        """
        Compute the normalization of the linear bias term in the distance
        prior.
        """
        if not self.use_im:
            return jnp.zeros((self.len_input_data, self.num_cal))

        # Density constrast along the LOS, (ngal, nrad_steps)
        los_grid_delta = self.f_los_delta.interp_many_steps_per_galaxy(los_grid_r)[0]  # noqa

        # Integrand, (ngal, ncalibration_samples, nrad_steps)
        intg = (1 + self.b1[None, :, None] * los_grid_delta[:, None, :])
        intg = jnp.log(jnp.clip(intg, 1e-5))
        intg = 2 * jnp.log(los_grid_r)[None, None, :] + intg
        return ln_simpson(intg, los_grid_r[None, None, :], axis=-1)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Redshift2Real(BaseRedshift2Real):
    """
    A model for mapping observed redshift to cosmological redshift.

    TODO: Switch to some smoother clipping for the IM.
    """

    def __call__(self, ):
        """
        `p(z_cosmo | z_CMB, calibration)` for a single object.
        """
        with plate("data", self.len_input_data):
            r = sample("r", Uniform(self.Rmin, self.Rmax))

        # Homogeneous Malmquist bias term, shape (ngal,)
        lp_r = 2 * jnp.log(r)

        if self.use_im:
            # Density constrast along the LOS, (ngal, )
            los_delta = self.f_los_delta(r)[0]
            # Distance prior, shape (ngal, ncalibration_samples)
            lp_r = (lp_r[:, None]
                    + jnp.log(jnp.clip(1 + self.b1[None, :] * los_delta[:, None], 1e-5)))  # noqa
            lp_r -= self.lp_norm
        else:
            lp_r = lp_r[:, None]

        # Cosmological redshift, shape (ngal,)
        zcosmo = self.dist2redshift(r)

        # Peculiar velocity from the reconstruction, (ngal,)
        Vpec = self.f_los_velocity(r)[0]

        # Peculiar velocity redshift (ngal, ncalibration_samples)
        zpec = (
            self.beta[None, :] * Vpec[:, None]
            + self.Vext_radial) / SPEED_OF_LIGHT
        # Predicted redshift (ngal, ncalibration_samples)
        cz_pred = SPEED_OF_LIGHT * (
            (1 + zcosmo)[:, None] * (1 + zpec) - 1)

        ll = Normal(cz_pred, self.sigma_v[None, :],).log_prob(
            self.cz_cmb[:, None])
        factor(
            "log_density",
            log_mean_exp(lp_r + ll, axis=-1) - jnp.log(ll.shape[-1])
            )
