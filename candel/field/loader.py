# Copyright (C) 2024 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Scripts to load the existing 3D density and velocity fields so that they can
be interpolated along the line of sight of galaxies.
"""
from abc import ABC, abstractmethod

import numpy as np


class BaseFieldLoader(ABC):
    """
    Base class for loading the 3D density and velocity fields.
    """

    @abstractmethod
    def load_density(self):
        """
        Load the 3D density field.
        """
        pass

    @abstractmethod
    def load_velocity(self):
        """
        Load the 3D velocity field.
        """
        pass


class Carrick2015_FieldLoader(BaseFieldLoader):
    """
    Class to load the Carrick+2015 3D density and velocity fields, which can be
    obtained from `http://cosmicflows.iap.fr. The fields are expected to be
    in Galactic coordintes.

    Parameters
    ----------
    path_density : str
        Path to the Carrick+2015 density field.
    path_velocity : str
        Path to the Carrick+2015 velocity field.
    """

    def __init__(self, path_density, path_velocity):
        self.path_density = path_density
        self.path_velocity = path_velocity

        self.coordinate_frame = "galactic"
        self.boxsize = 400.0  # Mpc / h
        self.Omega_m = 0.3
        self.observer_pos = np.array([200., 200., 200.], dtype=np.float32)

    def load_density(self):
        # Carrick+2015 density field is in the form of overdensity
        return 1 + np.load(self.path_density).astype(np.float32)

    def load_velocity(self):
        field = np.load(self.path_velocity)

        # Because the Carrick+2015 data is in the following form:
        # "The velocities are predicted peculiar velocities in the CMB
        # frame in Galactic Cartesian coordinates, generated from the
        # \(\delta_g^*\) field with \(\beta^* = 0.43\) and an external
        # dipole \(V_\mathrm{ext} = [89,-131,17]\) (Carrick et al Table 3)
        # has already been added.""
        field[0] -= 89
        field[1] -= -131
        field[2] -= 17
        field /= 0.43

        return field.astype(np.float32)


class Manticore_FieldLoader(BaseFieldLoader):

    def __init__(self, fpath):
        pass


#         class CSiBORG2XField(BaseField):
#     """
#     CSiBORG2X `z = 0` field class based on the Manticore ICs.
#
#     Parameters
#     ----------
#     nsim : int
#         Simulation index.
#     version : str
#         Manticore version index.
#     paths : Paths, optional
#         Paths object. By default, the paths are set to the `glamdring` paths.
#     """
#     def __init__(self, nsim, version, paths=None):
#         self.version = version
#         if version == 0:
#             self.nametag = "csiborg2X"
#         elif version == 1:
#             self.nametag = "manticore_2MPP_N128_DES_V1"
#         elif version == 2:
#             self.nametag = "manticore_2MPP_MULTIBIN_N128_DES_V1"
#         elif version == 3:
#             self.nametag = "manticore_2MPP_MULTIBIN_N128_DES_V2"
#         elif version == 4:
#             self.nametag = "manticore_2MPP_MULTIBIN_N256_DES_V2"
#         else:
#             raise ValueError("Invalid Manticore version.")
#
#         super().__init__(nsim, paths, False)
#
#     def overdensity_field(self, **kwargs):
#         if self.version == 0:
#             fpath = self.paths.field(
#                 "overdensity", None, None, self.nsim, self.nametag)
#             with File(fpath, "r") as f:
#                 field = f["delta_cic"][...].astype(np.float32)
#         else:
#             raise ValueError("Invalid Manticore version to read the "
#                              "overdensity field.")
#
#         return field
#
#     def density_field(self, **kwargs):
#         if self.version == 0:
#             field = self.overdensity_field()
#             omega0 = simname2Omega_m(self.nametag)
#             rho_mean = omega0 * 277.53662724583074  # Msun / kpc^3
#             field += 1
#             field *= rho_mean
#         elif self.version in [1, 2, 3, 4]:
#             MAS = kwargs["MAS"]
#             grid = kwargs["grid"]
#             fpath = self.paths.field(
#                 "density", MAS, grid, self.nsim, self.nametag)
#
#             if MAS == "SPH":
#                 with File(fpath, "r") as f:
#                     field = f["density"][:]
#
#                 field /= (681.1 * 1e3 / grid)**3  # Convert to h^2 Msun / kpc^3
#             else:
#                 field = np.load(fpath)
#         else:
#             raise ValueError("Invalid Manticore version to read the "
#                              "density field.")
#
#         return field
#
#     def velocity_field(self, **kwargs):
#         if self.version == 0:
#             fpath = self.paths.field(
#                 "velocity", None, None, self.nsim, "csiborg2X")
#             with File(fpath, "r") as f:
#                 v0 = f["v_0"][...]
#                 v1 = f["v_1"][...]
#                 v2 = f["v_2"][...]
#                 field = np.array([v0, v1, v2])
#         elif self.version in [1, 2, 3, 4]:
#             MAS = kwargs["MAS"]
#             grid = kwargs["grid"]
#             fpath = self.paths.field(
#                 "velocity", MAS, grid, self.nsim, self.nametag)
#
#             if MAS:
#                 with File(fpath, "r") as f:
#                     density = f["density"][:]
#                     v0 = f["p0"][:] / density
#                     v1 = f["p1"][:] / density
#                     v2 = f["p2"][:] / density
#                 field = np.array([v0, v1, v2])
#             else:
#                 field = np.load(fpath)
#
#         return field
#
#     def radial_velocity_field(self, **kwargs):
#         raise RuntimeError("The radial velocity field is not available.")



###############################################################################
#             Shortcut to get the appropriate field class.                    #
###############################################################################


def name2field_loader(name):
    """
    Convert a field name to a field loader.

    Parameters
    ----------
    name : str
        Name of the field loader.

    Returns
    -------
    BaseFieldLoader
        Field loader.
    """
    if name == "Carrick2015":
        return Carrick2015_FieldLoader
    else:
        raise ValueError(f"Unknown field loader: {name}")
