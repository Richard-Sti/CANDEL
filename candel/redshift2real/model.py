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
from ..model import LOSInterpolator, ln_simpson, lp_galaxy_bias
from ..util import SPEED_OF_LIGHT, fprint, radec_to_cartesian


def log_mean_exp(logp, axis=-1):
    return logsumexp(logp, axis=axis) - jnp.log(logp.shape[axis])


class BaseRedshift2Real(ABC):
    """Base class for all models. """

    def __init__(self, RA, dec, zcmb, los_r, los_density, los_velocity,
                 which_bias, calibration_samples, Rmin=1e-7, Rmax=300,
                 num_rgrid=101, r0_decay_scale=5, Om0=0.3):
        self.dist2redshift = Distance2Redshift(Om0=Om0)

        self.Rmin = Rmin
        self.Rmax = Rmax
        assert Rmin > 0 and Rmax > Rmin
        assert num_rgrid % 2 == 1

        self.len_input_data = len(zcmb)
        self.cz_cmb = jnp.asarray(zcmb * SPEED_OF_LIGHT)  # in km/s

        los_r = jnp.asarray(los_r)
        los_density = jnp.asarray(los_density)
        los_velocity = jnp.asarray(los_velocity)

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
        self.which_bias = which_bias
        if which_bias is None:
            self.use_im = False
        elif which_bias == "linear":
            self.b1 = jnp.asarray(calibration_samples["b1"])
            self.f_los_delta = LOSInterpolator(
                los_r, los_density - 1, r0_decay_scale=r0_decay_scale)
            self.lp_norm = self.compute_linear_bias_lp_normalization(
                self.los_grid_r)
        elif which_bias == "double_powerlaw":
            self.alpha_low = jnp.asarray(calibration_samples["alpha_low"])
            self.alpha_high = jnp.asarray(calibration_samples["alpha_high"])
            self.log_rho_t = jnp.asarray(calibration_samples["log_rho_t"])
            self.f_los_log_density = LOSInterpolator(
                los_r, jnp.log(los_density), r0_decay_scale=r0_decay_scale)

            self.lp_norm = self.compute_double_powerlaw_bias_lp_normalization(
                self.los_grid_r)
        else:
            raise ValueError(f"Unknown bias model: {which_bias}")

        self.sigma_v = jnp.asarray(calibration_samples["sigma_v"])
        if "beta" in calibration_samples:
            self.beta = jnp.asarray(calibration_samples["beta"])
        else:
            fprint("Beta not in calibration samples. Setting beta=1.")
            self.beta = jnp.ones_like(self.sigma_v)

        fprint(f"Loaded {self.len_input_data} objects and "
               f"{self.num_cal} calibration samples.")

        self.print_sample_stats()

    def print_sample_stats(self):
        print("Calibration sample statistics:")
        print(f"sigma_v : {jnp.mean(self.sigma_v):.3f} +- {jnp.std(self.sigma_v):.3f}")  # noqa
        print(f"beta    : {jnp.mean(self.beta):.3f} +- {jnp.std(self.beta):.3f}")        # noqa
        if self.which_bias == "linear":
            print(f"b1      : {jnp.mean(self.b1):.3f} +- {jnp.std(self.b1):.3f}")          # noqa
        elif self.which_bias == "double_powerlaw":
            print(f"alpha_low  : {jnp.mean(self.alpha_low):.3f} +- {jnp.std(self.alpha_low):.3f}")      # noqa
            print(f"alpha_high : {jnp.mean(self.alpha_high):.3f} +- {jnp.std(self.alpha_high):.3f}")    # noqa
            print(f"log_rho_t  : {jnp.mean(self.log_rho_t):.3f} +- {jnp.std(self.log_rho_t):.3f}")      # noqa

    def compute_linear_bias_lp_normalization(self, los_grid_r):
        """
        Compute the normalization of the linear bias term in the distance
        prior.
        """
        print("Computing `linear` bias lp normalization...")
        # Density constrast along the LOS, (nfield, ngal, nrad_steps)
        los_grid_delta = self.f_los_delta.interp_many_steps_per_galaxy(
            los_grid_r)
        bias_params = [self.b1[None, None, :, None],]

        intg = lp_galaxy_bias(
            los_grid_delta[:, :, None, :], None, bias_params,
            galaxy_bias="linear")
        return ln_simpson(intg, los_grid_r[None, None, None, :], axis=-1)

    def compute_double_powerlaw_bias_lp_normalization(self, los_grid_r):
        """
        Compute the normalization of the double powerlaw bias term in the
        distance prior.
        """
        print("Computing `double_powerlaw` bias lp normalization...")
        los_log_density = self.f_los_log_density.interp_many_steps_per_galaxy(
            los_grid_r)
        bias_params = [self.alpha_low[None, None, :, None],
                       self.alpha_high[None, None, :, None],
                       self.log_rho_t[None, None, :, None],
                       ]
        intg = lp_galaxy_bias(
            None, los_log_density[:, :, None, :], bias_params,
            galaxy_bias="double_powerlaw")
        return ln_simpson(intg, los_grid_r[None, None, None, :], axis=-1)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Redshift2Real(BaseRedshift2Real):
    """
    A model for mapping observed redshift to cosmological redshift.
    """

    def __call__(self, ):
        """
        `p(z_cosmo | z_CMB, calibration)` for a single object.
        """
        with plate("data", self.len_input_data):
            r = sample("r", Uniform(self.Rmin, self.Rmax))

        # Homogeneous Malmquist bias term, shape (ngal,)
        lp_r = 2 * jnp.log(r)

        if self.which_bias == "linear":
            bias_params = [self.b1[None, None, :],]
            lp_r_bias = lp_galaxy_bias(
                self.f_los_delta(r)[..., None], None, bias_params,
                galaxy_bias="linear")
            lp_r = lp_r[None, :, None] + lp_r_bias - self.lp_norm
        elif self.which_bias == "double_powerlaw":
            bias_params = [
                self.alpha_low[None, None, :],
                self.alpha_high[None, None, :],
                self.log_rho_t[None, None, :],
                ]
            lp_r_bias = lp_galaxy_bias(
                None, self.f_los_log_density(r)[..., None],
                bias_params,
                galaxy_bias="double_powerlaw")
            lp_r = lp_r[None, :, None] + lp_r_bias - self.lp_norm
        else:
            lp_r = lp_r[None, :, None]

        # Cosmological redshift, shape (ngal,)
        zcosmo = self.dist2redshift(r)

        # Peculiar velocity from the reconstruction, (nfield, ngal,)
        Vpec = self.f_los_velocity(r)

        # Peculiar velocity redshift (nfield, ngal, ncalibration_samples)
        zpec = (
            self.beta[None, None, :] * Vpec[..., None]
            + self.Vext_radial[None, ...]) / SPEED_OF_LIGHT
        # Predicted redshift (nfield, ngal, ncalibration_samples)
        cz_pred = SPEED_OF_LIGHT * (
            (1 + zcosmo)[None, :, None] * (1 + zpec) - 1)

        ll = Normal(cz_pred, self.sigma_v[None, None, :],).log_prob(
            self.cz_cmb[None, :, None])
        ll += lp_r

        # Average over the calibration samples, shape (nfield, ngal)
        ll = log_mean_exp(ll, axis=-1) - jnp.log(ll.shape[-1])
        # Average over the density and velocity field samples
        factor(
            "log_density",
            log_mean_exp(ll, axis=0) - jnp.log(ll.shape[0])
            )
