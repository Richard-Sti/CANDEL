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
try:
    # Python 3.11+
    import tomllib                                                              # noqa
except ModuleNotFoundError:
    # Backport for <=3.10
    import tomli as tomllib
from abc import ABC, abstractmethod

import numpy as np
from h5py import File
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from numpyro import factor, plate, sample
from numpyro.distributions import Normal, Uniform

from ..cosmography import Distance2Redshift
from ..util import SPEED_OF_LIGHT, fprint, radec_to_cartesian


def log_mean_exp(logp, axis=-1):
    return logsumexp(logp, axis=axis) - jnp.log(logp.shape[axis])


class BaseRedshift2Real(ABC):
    """Base class for all models. """

    def __init__(self, config_path):
        self.dist2redshift = Distance2Redshift()

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        # Open the input data file.
        with File(config["input"]["fname_data"], "r") as f:
            czcmb = f["zcmb"][...] * SPEED_OF_LIGHT
            # e_czcmb = f["e_cmb"][...] * SPEED_OF_LIGHT
            RA = f["RA"][...]
            dec = f["dec"][...]

        self.len_input_data = len(czcmb)
        rhat = radec_to_cartesian(RA, dec)

        self.input_data = {
            "czcmb": czcmb,
            # "e_czcmb": e_czcmb,
            }

        # Open the calibration data file.
        with File(config["input"]["fname_calibration"], "r") as f:
            grp = f["samples"]
            Vext = grp["Vext"][...].reshape(-1, 3)
            sigma_v = grp["sigma_v"][...].reshape(-1,)

            self.len_calibration_samples = len(Vext)

            if "alpha" in grp:
                alpha = grp["alpha"][...].reshape(-1, )
            else:
                alpha = None

            if "beta" in grp:
                beta = grp["beta"][...].reshape(-1, )
            else:
                beta = None

        Vext_radial = np.sum(rhat[:, None, :] * Vext[None, :, :], axis=-1).T
        self.calibration_samples = {
            "Vext_radial": Vext_radial,
            "sigma_v": sigma_v,
            "alpha": alpha,
            "beta": beta,
            }

        self.prior_distance = Uniform(
            low=config["model"]["r_min"],
            high=config["model"]["r_max"],
        )

        for key in self.calibration_samples:
            if self.calibration_samples[key] is None:
                continue

            self.calibration_samples[key] = jnp.asarray(
                self.calibration_samples[key])

        for key in self.input_data:
            self.input_data[key] = jnp.asarray(self.input_data[key])

        fprint(f"Loaded {self.len_input_data} objects and "
               f"{self.len_calibration_samples} calibration samples.")

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SimpleRedshift2Real(BaseRedshift2Real):
    """
    A simple model for mapping observed redshift to cosmological redshift
    if the velocity field is modelled as a constant dipole.
    """

    def __call__(self, ):
        """
        `p(z_cosmo | z_CMB, calibration)` for a single object.
        """
        with plate("data", self.len_input_data):
            r = sample("r", self.prior_distance)
            lp = 2 * jnp.log(r)

            zcosmo = self.dist2redshift(r)
            zpec = self.calibration_samples["Vext_radial"] / SPEED_OF_LIGHT

            # Shape of zpec is (ncalibration_samples, ngal)
            cz_pred = SPEED_OF_LIGHT * (
                (1 + zcosmo)[None, :] * (1 + zpec) - 1)

            ll = Normal(
                cz_pred,
                self.calibration_samples["sigma_v"][:, None],).log_prob(
                    self.input_data["czcmb"][None, :])
            factor("log_density", lp + log_mean_exp(ll, axis=0))
