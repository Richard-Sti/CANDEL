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
from os.path import join

import numpy as np
from h5py import File


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

    def __init__(self, path_density, path_velocity, **kwargs):
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
    """
    Manticore field loader class.

    Parameters
    ----------
    nsim : int
        Simulation index.
    paths : Paths, optional
        Paths object. By default, the paths are set to the `glamdring` paths.
    """

    def __init__(self, nsim, fpath_root):
        self.fname = join(fpath_root, f"mcmc_{nsim}.hdf5")

        self.coordinate_frame = "icrs"
        self.boxsize = 681.1  # Mpc / h
        self.Omega_m = 0.3111

        x0 = 0.5 * self.boxsize
        self.observer_pos = np.array([x0, x0, x0], dtype=np.float32)

    def density_field(self, **kwargs):
        with File(self.fname, "r") as f:
            field = f["density"][:]

        # Convert to h^2 Msun / kpc^3
        grid = field.shape[0]
        field /= (self.boxsize * 1e3 / grid)**3

        return field

    def velocity_field(self, **kwargs):
        with File(self.fname, "r") as f:
            density = f["density"][:]
            v0 = f["p0"][:] / density
            v1 = f["p1"][:] / density
            v2 = f["p2"][:] / density
        return np.array([v0, v1, v2])


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
    elif name.lower().startswith("manticore"):
        return Manticore_FieldLoader
    else:
        raise ValueError(f"Unknown field loader: {name}")
