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


def _flip_xz(field):
    """Transpose spatial axes and swap x/z vector components (if 4D)."""
    if field.ndim == 3:
        return field.T
    field = np.transpose(field, (0, 3, 2, 1))
    field[[0, 2]] = field[[2, 0]]
    return field


class BaseFieldLoader(ABC):
    r"""
    Base class for loading 3D density and velocity fields.

    Subclasses must implement:
    - ``load_density()``: Return a 3D ``np.ndarray`` (N, N, N) of
      mass densities
      in units of :math:`h^2 M_\odot / \mathrm{kpc}^3`.
    - ``load_velocity()``: Return a 4D ``np.ndarray`` (3, N, N, N) of Cartesian
      velocity components in :math:`\mathrm{km/s}`.

    Attributes
    ----------
    boxsize : float
        Box side length in :math:`h^{-1} \mathrm{Mpc}`.
    coordinate_frame : str
        The coordinate frame of the fields (e.g., ``"icrs"``, ``"galactic"``,
        ``"supergalactic"``).
    """

    @property
    def observer_pos(self):
        """Observer position; defaults to box center. Override via
        ``self._observer_pos`` in subclass ``__init__``."""
        try:
            return self._observer_pos
        except AttributeError:
            return np.array([self.boxsize / 2] * 3, dtype=np.float32)

    @abstractmethod
    def load_density(self):
        pass

    @abstractmethod
    def load_velocity(self):
        pass


class Carrick2015_FieldLoader(BaseFieldLoader):
    """
    Class to load the Carrick+2015 3D density and velocity fields [1], which
    can be obtained from http://cosmicflows.iap.fr. The fields are in Galactic
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
        # has already been added."
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
    file_path : str
        Path to the CLONES HDF5 file.
    """

    def __init__(self, file_path, **kwargs):
        self.file_path = file_path

        self.coordinate_frame = "supergalactic"
        self.boxsize = 500  # Mpc / h
        self.Omega_m = 0.307115

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
            rho = rho.T
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

        v = np.stack(comps, axis=0).astype(self.dtype)

        if self.version == 1:
            v = _flip_xz(v)

        return v


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
            rho = _flip_xz(rho)

        return rho

    def load_velocity(self):
        with File(self.file_path, "r") as f:
            rho = f["density"][:]
            v0 = f["p0"][:] / rho
            v1 = f["p1"][:] / rho
            v2 = f["p2"][:] / rho

        v = np.array([v0, v1, v2], dtype=np.float32)

        if self.flip_xz:
            v = _flip_xz(v)

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
        self.Omega_m = 0.306

    def load_density(self):
        with File(self.fname, "r") as f:
            field = f["density"][:]

        # Convert to h^2 Msun / kpc^3
        grid = field.shape[0]
        field /= (self.boxsize * 1e3 / grid)**3

        return field.astype(np.float32)

    def load_velocity(self):
        with File(self.fname, "r") as f:
            density = f["density"][:]
            v0 = f["p0"][:] / density
            v1 = f["p1"][:] / density
            v2 = f["p2"][:] / density
        return np.array([v0, v1, v2], dtype=np.float32)


###############################################################################
#             Shortcut to get the appropriate field class.                    #
###############################################################################


_FIELD_LOADERS = {
    "Carrick2015": Carrick2015_FieldLoader,
    "Lilow2024": Lilow2024_FieldLoader,
    "CF4": CF4_FieldLoader,
    "CLONES": CLONES_FieldLoader,
    "CB1": CSiBORG_FieldLoader,
    "CB2": CSiBORG_FieldLoader,
}


def name2field_loader(name):
    """Convert a field name to a field loader class."""
    if name in _FIELD_LOADERS:
        return _FIELD_LOADERS[name]
    if name.lower().startswith("manticore"):
        return Manticore_FieldLoader
    if name.lower().startswith("hamlet"):
        return Hamlet_FieldLoader
    raise ValueError(f"Unknown field loader: {name}")
