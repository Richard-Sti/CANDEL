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
from astropy.io import fits
from h5py import File


def smooth_clip(x, eps=1e-3):
    return 0.5 * (x + np.sqrt(x**2 + eps**2))


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
    Class to load the Carrick+2015 3D density and velocity fields [1], which
    can be obtained from `http://cosmicflows.iap.fr. The fields are in Galactic
    coordinates.

    [1] https://arxiv.org/abs/1504.04627

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
        self.effective_resolution = 4
        self.observer_pos = np.array([200., 200., 200.], dtype=np.float32)

    def load_density(self):
        # Carrick+2015 density field is in the form of overdensity
        rho = 1 + np.load(self.path_density)
        return smooth_clip(rho, eps=1e-3).astype(np.float32)

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


class Lilow2024_FieldLoader(BaseFieldLoader):
    """
    Class to load the Lilow+2024 3D density and velocity fields [1]. The fields
    are expected to be in Galactic coordinates.

    [1] https://arxiv.org/abs/2404.02278

    Parameters
    ----------
    path_density : str
        Path to the Lilow+2024 density field.
    path_velocity_x : str
        Path to the Lilow+2024 velocity field (x-component).
    path_velocity_y : str
        Path to the Lilow+2024 velocity field (y-component).
    path_velocity_z : str
        Path to the Lilow+2024 velocity field (z-component).
    """

    def __init__(self, path_density, path_velocity_x, path_velocity_y,
                 path_velocity_z, **kwargs):
        self.path_density = path_density
        self.path_velocity = [
            path_velocity_x, path_velocity_y, path_velocity_z]

        self.coordinate_frame = "galactic"
        self.boxsize = 400.0  # Mpc / h
        self.Omega_m = 0.3175
        self.effective_resolution = 4
        self.observer_pos = np.array([200., 200., 200.], dtype=np.float32)

    def load_density(self):
        rho = np.load(self.path_density).astype(np.float32)
        return np.nan_to_num(rho, nan=1.0)

    def load_velocity(self):
        vel = np.stack(
            [np.load(f).astype(np.float32) for f in self.path_velocity])
        return np.nan_to_num(vel, nan=0.0)


class CF4_FieldLoader(BaseFieldLoader):
    """
    Class to load the CF4 3D density and velocity fields [1]. The fields
    are expected to be in Galactic coordinates.

    [1] https://arxiv.org/abs/2211.16390

    Parameters
    ----------
    folder : str
        Directory containing the CF4 FITS files
        (e.g. ".../CF4gp_23avr24_256-z008_test_100_realizations").
    nsim : int
        Realization index to load.
    """

    def __init__(self, folder, nsim, **kwargs):
        self.folder = folder
        self.nsim = int(nsim)

        self.coordinate_frame = "supergalactic"
        self.boxsize = 1000.0  # Mpc / h
        self.Omega_m = 0.3
        self.observer_pos = np.array([self.boxsize / 2] * 3, dtype=np.float32)

        fname_base = f"CF4gp_23avr24_256-z008_test_realization{1 + self.nsim}"
        self._density_path = join(self.folder, f"{fname_base}_delta.fits")
        self._velocity_path = join(self.folder, f"{fname_base}_velocity.fits")

    def load_density(self):
        rho = 1 + fits.open(self._density_path)[0].data
        return smooth_clip(rho, eps=1e-2).astype(np.float32)

    def load_velocity(self):
        vx, vy, vz = fits.open(self._velocity_path)[0].data
        return 52.0 * np.stack([vx, vy, vz], axis=0).astype(np.float32)


class CLONES_FieldLoader(BaseFieldLoader):
    """
    Class to load the CLONES z=0 density and velocity fields in supergalactic
    Cartesian coordinates.

    Parameters
    ----------
    basedir : str
        Directory containing the CLONES HDF5 files.
    fname : str
        Name of the HDF5 file to load.
    """

    def __init__(self, file_path, **kwargs):
        self.file_path = file_path

        self.coordinate_frame = "supergalactic"
        self.boxsize = 500  # Mpc / h
        self.Omega_m = 0.307115
        self.observer_pos = np.array([self.boxsize / 2] * 3, dtype=np.float32)

    def load_density(self):
        with File(self.file_path, "r") as f:
            field = f["density"][...]

        grid = field.shape[0]
        field /= (500 * 1e3 / grid)**3

        return field.astype(np.float32)

    def load_velocity(self):
        with File(self.file_path, "r") as f:
            vx = f["p0"][...] / f["density"][...]
            vy = f["p1"][...] / f["density"][...]
            vz = f["p2"][...] / f["density"][...]
            field = np.stack([vx, vy, vz], axis=0)
        return field.astype(np.float32)


class Hamlet_FieldLoader(BaseFieldLoader):
    """
    Loader for HAMLET z = 0 density and velocity fields in supergalactic
    coordinates.

    Parameters
    ----------
    nsim : int
        Simulation index starting from 0.
    fpath_root : str
        Root directory pointing to the HAMLET_V0 or HAMLET_V1 dataset.
    version : int
        Dataset version, either 0 or 1.
    """

    def __init__(self, nsim, fpath_root, version, **kwargs):
        self.nsim = int(nsim)
        self.base = fpath_root
        assert version in (0, 1)
        self.version = int(version)

        self.coordinate_frame = "supergalactic"
        self.Omega_m = 0.3
        self.H0 = 74.6
        self.dtype = np.float32

        if self.version == 0:
            folder = str(1 + (self.nsim // 2))
            self.tag = 0 if (self.nsim % 2 == 0) else 99
            self.root = join(self.base, folder)
            self.boxsize = 1000.0
            self.ngrid = 256
        elif self.version == 1:
            cluster = 1 + (self.nsim // 2)
            self.rtag, self.stag = (("R000", "S000")
                                    if (self.nsim % 2 == 0)
                                    else ("R450", "S450"))
            self.root = join(self.base,
                             f"C{cluster:03d}",
                             self.rtag,
                             self.stag,
                             "cic")
            self.boxsize = 500.0
            self.ngrid = 128
        else:
            raise ValueError(f"Unknown HAMLET version: {self.version}")

        self.observer_pos = np.array(
            [self.boxsize / 2] * 3, dtype=self.dtype)

    def _read_grid(self, fname):
        return np.fromfile(fname, dtype=self.dtype).reshape(
            (self.ngrid,) * 3)

    def load_density(self):
        if self.version == 0:
            fname = join(self.root, f"divv_{self.tag}_{self.ngrid}.bin")
            delta = self._read_grid(fname)
            rho = np.log1p(np.exp(delta))
        elif self.version == 1:
            fname = join(self.root,
                         f"cic_pos_N{self.ngrid}_{self.stag}_snap003.dat")
            rho = self._read_grid(fname)
        else:
            raise ValueError(f"Unknown HAMLET version: {self.version}")

        return rho.astype(self.dtype)

    def load_velocity(self):
        comps = []
        for c in ("x", "y", "z"):
            if self.version == 0:
                fname = join(self.root,
                             f"v{c}_{self.tag}_{self.ngrid}.bin")
            else:
                fname = join(self.root,
                             f"cic_vel{c}_N{self.ngrid}_{self.stag}"
                             f"_snap003_normed.dat")
            comps.append(self._read_grid(fname))

        return np.stack(comps, axis=0).astype(self.dtype)


class CSiBORG_FieldLoader(BaseFieldLoader):
    """
    Class to load CSiBORG1/2 z=0 SPH fields, in the ICRS frame.

    Parameters
    ----------
    nsim : int
        Simulation index (ranging from 0, not the MCMC step).
    fpath_root : str
        Root directory for the simulation files.
    version : {"csiborg1", "csiborg2"}
        Which CSiBORG version to load (sets boxsize and unit conversion).
    """

    def __init__(self, nsim, fpath_root, version, **kwargs):
        if version not in {"csiborg1", "csiborg2"}:
            raise ValueError("version must be 'csiborg1' or 'csiborg2'.")

        self.nsim = nsim
        self.flip_xz = True
        self.version = version

        index_path = join(fpath_root, f"{version}_index.txt")
        mapping = {}
        with open(index_path, "r") as f:
            for line in f:
                idx, tag = line.strip().split()
                mapping[int(idx)] = tag

        if self.nsim not in mapping:
            raise ValueError(f"nsim {self.nsim} not found in {index_path}.")

        tag = mapping[self.nsim]
        if version == "csiborg1":
            self.file_path = join(fpath_root, f"sph_ramses_{tag}_1024.hdf5")
            self.boxsize = 677.7  # Mpc / h
        elif version == "csiborg2":
            self.file_path = join(fpath_root, f"chain_{tag}_1024.hdf5")
            self.boxsize = 676.6  # Mpc / h
        else:
            raise ValueError(f"Unknown CSiBORG version: {version}")

        self.coordinate_frame = "icrs"
        self.observer_pos = np.array([self.boxsize / 2] * 3, dtype=np.float32)

    def load_density(self):
        with File(self.file_path, "r") as f:
            rho = f["density"][:]

        # Unit conversion (CSiBORG2 masses are in 1e10 Msun / h)
        if self.version == "csiborg2":
            rho = rho * 1e10  # Msun/h

        grid = rho.shape[0]
        cell = (self.boxsize * 1e3) / grid       # kpc/h per cell
        rho = rho / (cell**3)                    # -> h^2 Msun / kpc^3
        rho = rho.astype(np.float32)

        if self.flip_xz:
            rho = rho.T

        return rho

    def load_velocity(self):
        with File(self.file_path, "r") as f:
            rho = f["density"][:]
            v0 = f["p0"][:] / rho
            v1 = f["p1"][:] / rho
            v2 = f["p2"][:] / rho

        v = np.array([v0, v1, v2], dtype=np.float32)

        if self.flip_xz:
            v[0, ...] = v[0, ...].T
            v[1, ...] = v[1, ...].T
            v[2, ...] = v[2, ...].T
            v[[0, 2], ...] = v[[2, 0], ...]

        return v


class Manticore_FieldLoader(BaseFieldLoader):
    """
    Manticore field loader class, in the ICRS frame.

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

    def load_density(self):
        with File(self.fname, "r") as f:
            field = f["density"][:]

        # Convert to h^2 Msun / kpc^3
        grid = field.shape[0]
        field /= (self.boxsize * 1e3 / grid)**3

        return field

    def load_velocity(self):
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
    elif name == "Lilow2024":
        return Lilow2024_FieldLoader
    elif name == "CF4":
        return CF4_FieldLoader
    elif name == "CLONES":
        return CLONES_FieldLoader
    elif name in ["CB1", "CB2"]:
        return CSiBORG_FieldLoader
    elif name.lower().startswith("manticore"):
        return Manticore_FieldLoader
    elif name.lower().startswith("hamlet"):
        return Hamlet_FieldLoader
    else:
        raise ValueError(f"Unknown field loader: {name}")
