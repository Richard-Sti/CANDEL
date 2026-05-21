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
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import join
from pathlib import Path

import numpy as np
from astropy.io import fits
from h5py import File

from ..util import fprint


@dataclass(frozen=True)
class FieldMetadata:
    """Static metadata and runtime-product policy for a field family."""

    name: str
    coordinate_frame: str
    boxsize: float
    Omega_m: float = None
    ngrid: int = None
    effective_resolution: float = None
    H0: float = None
    production_method: str = None
    storage_schema: str = None
    require_cached_products: bool = True
    cache_group: str = None
    description: str = ""

    @property
    def raw_read_allowed(self):
        return not self.require_cached_products


def available_mcmc_field_indices(fpath_root, glob="mcmc_*.hdf5",
                                 filename_regex=r"mcmc_(\d+)\.hdf5$"):
    """Return sorted field indices parsed from field product filenames."""
    root = Path(fpath_root)
    field_re = re.compile(filename_regex)
    indices = []
    for path in root.glob(glob):
        match = field_re.fullmatch(path.name)
        if match is not None:
            indices.append(int(match.group(1)))

    if not indices:
        raise FileNotFoundError(
            f"No field files matching `{glob}` found in `{root}`.")

    return sorted(indices)


def field_mas_directory(which_MAS=None):
    """Return the canonical on-disk folder for a field MAS choice."""
    if which_MAS is None:
        return "CIC"
    key = str(which_MAS).strip().replace("_", "-").lower()
    folders = {
        "cic": "CIC",
        "pcs": "PCS",
        "sph": "SPH",
        "cic-borg": "CIC_BORG",
    }
    try:
        return folders[key]
    except KeyError as exc:
        raise ValueError(
            "`which_MAS` must be one of CIC, PCS, SPH, or CIC_BORG; "
            f"got {which_MAS!r}.") from exc


def smooth_clip(x, eps=1e-3):
    """Return a differentiable positive-part approximation for ``x``.

    This is a smooth version of ``max(x, 0)`` with transition width ``eps``.
    """
    return 0.5 * (x + np.sqrt(x**2 + eps**2))


def _flip_xz(field):
    """Transpose spatial axes and swap x/z vector components (if 4D)."""
    if field.ndim == 3:
        return field.T
    field = np.transpose(field, (0, 3, 2, 1))
    field[[0, 2]] = field[[2, 0]]
    return field


def _first_hdf5_attr(attrs, names, default=None):
    for name in names:
        if name in attrs:
            value = attrs[name]
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return value
    return default


def _first_hdf5_dataset(group, names):
    for name in names:
        if name is not None and name in group:
            return name
    return None


class BaseFieldLoader(ABC):
    r"""
    Base class for loading 3D density and velocity fields.

    Subclasses must implement:
    - ``load_density()``: Return a 3D ``np.ndarray`` (N, N, N) of
      positive density-like values. The physical units and normalization are
      loader-specific.
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
            self._observer_pos = np.array(
                [self.boxsize / 2] * 3, dtype=np.float32)
            fprint(
                "field loader: observer_pos not set; using box center "
                f"{self._observer_pos}.")
            return self._observer_pos

    @abstractmethod
    def load_density(self):
        pass

    @abstractmethod
    def load_velocity(self):
        pass


class BORGFieldLoader(BaseFieldLoader):
    """Generic HDF5 loader for BORG-style gridded field products.

    Two storage methods are supported:

    - Forward/gridded products with ``overdensity`` or ``density`` plus either
      ``velocity`` or ``vx``/``vy``/``vz``.
    - N-body MAS/SPH products with ``density`` plus momentum fields
      ``p0``/``p1``/``p2``.

    Dataset names and metadata may be specified by file attributes written by
    ``scripts/BORG_fields/run_borg_fields.py``.
    """

    def __init__(self, nsim=0, fpath_root=None,
                 filename_template="mcmc_{nsim}.hdf5", file_path=None,
                 density_key=None, density_kind="auto", velocity_key=None,
                 velocity_component_keys=("vx", "vy", "vz"),
                 velocity_kind="auto", momentum_keys=("p0", "p1", "p2"),
                 boxsize=None, Omega_m=None, coordinate_frame="icrs",
                 observer_pos=None, ngrid=None, density_mass_factor=1.0,
                 flip_xz=False, **kwargs):
        self.nsim = int(nsim)
        if file_path is None:
            self.fname = join(fpath_root, filename_template.format(nsim=nsim))
        else:
            self.fname = str(file_path)
        self.file_path = self.fname
        self.density_key = density_key
        self.density_kind = density_kind
        self.velocity_key = velocity_key
        self.velocity_component_keys = tuple(velocity_component_keys)
        self.velocity_kind = velocity_kind
        self.momentum_keys = tuple(momentum_keys)
        self.density_mass_factor = float(density_mass_factor)
        self.flip_xz = bool(flip_xz)
        self._velocity_density = None

        with File(self.fname, "r") as f:
            box_value = _first_hdf5_attr(
                f.attrs, ("boxsize", "BoxSize", "box_size"), boxsize)
            if box_value is None:
                raise ValueError(
                    f"`{self.fname}` must define a boxsize attribute or the "
                    "loader must receive `boxsize`.")
            self.boxsize = self._scalar_boxsize(box_value)
            Om_value = _first_hdf5_attr(
                f.attrs, ("Omega_m", "Om", "Om0", "omega_m"), Omega_m)
            self.Omega_m = None if Om_value is None else float(Om_value)
            self.coordinate_frame = str(_first_hdf5_attr(
                f.attrs, ("coordinate_frame", "frame"), coordinate_frame))
            obs = _first_hdf5_attr(
                f.attrs, ("observer_position", "observer_pos",
                          "observer_position_borg_coordinates"),
                observer_pos)
            if obs is not None:
                self._observer_pos = np.asarray(obs, dtype=np.float32)

            key = self._density_dataset_key(f)
            shape = tuple(int(size) for size in f[key].shape)
            grid_shape = _first_hdf5_attr(f.attrs, ("grid_shape",), shape)
            self.grid_shape = tuple(int(size) for size in np.asarray(grid_shape))
            self.ngrid = int(ngrid if ngrid is not None else self.grid_shape[0])

    def _scalar_boxsize(self, value):
        value = np.asarray(value)
        if value.shape == ():
            return float(value)
        if value.size == 1:
            return float(value.reshape(-1)[0])
        if np.allclose(value, value.reshape(-1)[0]):
            return float(value.reshape(-1)[0])
        raise ValueError(
            f"`{self.fname}` has a non-cubic boxsize attribute: {value}.")

    def _attr_dataset(self, f, names):
        key = _first_hdf5_attr(f.attrs, names)
        if key is None:
            return None
        key = str(key)
        return key if key in f else None

    def _density_dataset_key(self, f):
        key = _first_hdf5_dataset(f, (self.density_key,))
        if key is None:
            key = self._attr_dataset(f, ("density_dataset",
                                         "overdensity_dataset"))
        if key is None:
            key = _first_hdf5_dataset(f, ("density", "overdensity"))
        if key is None:
            raise KeyError(
                f"No density dataset found in `{self.fname}`. Expected "
                "`density` or `overdensity` unless `density_key` is set.")
        return key

    def _velocity_dataset_key(self, f):
        key = _first_hdf5_dataset(f, (self.velocity_key,))
        if key is None:
            key = self._attr_dataset(f, ("velocity_dataset",))
        if key is None:
            key = _first_hdf5_dataset(f, ("velocity",))
        return key

    def _velocity_component_key(self, f, component):
        if component < len(self.velocity_component_keys):
            key = self.velocity_component_keys[component]
            if key is not None and key in f:
                return key
        attr_names = ("vx_dataset", "vy_dataset", "vz_dataset")
        key = self._attr_dataset(f, (attr_names[component],))
        if key is not None:
            return key
        aliases = (
            ("vx", "v_x", "velocity_x"),
            ("vy", "v_y", "velocity_y"),
            ("vz", "v_z", "velocity_z"),
        )
        return _first_hdf5_dataset(f, aliases[component])

    def _momentum_key(self, f, component):
        if component < len(self.momentum_keys):
            key = self.momentum_keys[component]
            if key is not None and key in f:
                return key
        aliases = (
            ("p0", "px", "momentum_x"),
            ("p1", "py", "momentum_y"),
            ("p2", "pz", "momentum_z"),
        )
        return _first_hdf5_dataset(f, aliases[component])

    def _density_is_overdensity(self, f, key):
        if self.density_kind == "overdensity":
            return True
        if self.density_kind in {"density", "mass_density"}:
            return False
        attr_key = self._attr_dataset(f, ("overdensity_dataset",))
        if attr_key is not None:
            return key == attr_key
        return key in {"overdensity", "density_contrast"}

    def _density_unit_volume(self, grid):
        return (self.boxsize * 1e3 / grid)**3

    def _raw_density(self, f):
        return f[self._density_dataset_key(f)][:]

    def _read_density_for_velocity(self, f):
        if self._velocity_density is None:
            self._velocity_density = self._raw_density(f)
        return self._velocity_density

    def _velocity_method(self, f):
        if self.velocity_kind in {"vector", "components", "momentum"}:
            return self.velocity_kind
        if self._velocity_dataset_key(f) is not None:
            return "vector"
        if all(self._velocity_component_key(f, i) is not None
               for i in range(3)):
            return "components"
        if all(self._momentum_key(f, i) is not None for i in range(3)):
            return "momentum"
        return None

    def load_density(self):
        with File(self.fname, "r") as f:
            key = self._density_dataset_key(f)
            field = f[key][:]
            is_overdensity = self._density_is_overdensity(f, key)

        if self.density_kind == "mass_density":
            field = field * self.density_mass_factor
            field = field / self._density_unit_volume(field.shape[0])
        elif is_overdensity:
            field = 1 + field
        field = field.astype(np.float32)
        if self.flip_xz:
            field = _flip_xz(field)
        return field

    def load_velocity(self):
        with File(self.fname, "r") as f:
            method = self._velocity_method(f)
            if method == "vector":
                key = self._velocity_dataset_key(f)
                field = f[key][:]
                if field.shape[0] == 3:
                    velocity = field.astype(np.float32)
                elif field.shape[-1] == 3:
                    velocity = np.moveaxis(field, -1, 0).astype(np.float32)
                else:
                    raise ValueError(
                        f"Velocity dataset `{key}` in `{self.fname}` must "
                        "have a component axis of length 3.")
            elif method == "components":
                comps = [f[self._velocity_component_key(f, i)][:] for i in
                         range(3)]
                velocity = np.stack(comps, axis=0).astype(np.float32)
            elif method == "momentum":
                density = self._read_density_for_velocity(f)
                comps = [f[self._momentum_key(f, i)][:] / density for i in
                         range(3)]
                velocity = np.array(comps, dtype=np.float32)
            else:
                raise KeyError(f"No velocity field found in `{self.fname}`.")

        if self.flip_xz:
            velocity = _flip_xz(velocity)
        return velocity

    def load_velocity_component(self, component):
        source_component = (2, 1, 0)[component] if self.flip_xz else component
        with File(self.fname, "r") as f:
            method = self._velocity_method(f)
            if method == "vector":
                key = self._velocity_dataset_key(f)
                field = f[key]
                if field.shape[0] == 3:
                    value = field[source_component]
                elif field.shape[-1] == 3:
                    value = field[..., source_component]
                else:
                    raise ValueError(
                        f"Velocity dataset `{key}` in `{self.fname}` must "
                        "have a component axis of length 3.")
            elif method == "components":
                key = self._velocity_component_key(f, source_component)
                value = f[key][:]
            elif method == "momentum":
                density = self._read_density_for_velocity(f)
                key = self._momentum_key(f, source_component)
                value = f[key][:] / density
            else:
                raise KeyError(f"No velocity field found in `{self.fname}`.")

        value = value.astype(np.float32)
        if self.flip_xz:
            value = _flip_xz(value)
        return value

    def clear_velocity_cache(self):
        self._velocity_density = None


class BORGSPHFieldLoader(BORGFieldLoader):
    """Loader for BORG/N-body density and momentum products."""

    def __init__(self, file_path, boxsize=None, Omega_m=None,
                 coordinate_frame="icrs", density_key="density",
                 momentum_keys=("p0", "p1", "p2"),
                 density_mass_factor=1.0, flip_xz=False, ngrid=None,
                 **kwargs):
        super().__init__(
            nsim=getattr(self, "nsim", 0), file_path=file_path,
            density_key=density_key, density_kind="mass_density",
            velocity_kind="momentum", momentum_keys=momentum_keys,
            boxsize=boxsize, Omega_m=Omega_m,
            coordinate_frame=coordinate_frame, ngrid=ngrid,
            density_mass_factor=density_mass_factor, flip_xz=flip_xz,
            **kwargs)


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
        metadata = field_metadata("Carrick2015")
        self.path_density = path_density
        self.path_velocity = path_velocity

        self.coordinate_frame = metadata.coordinate_frame
        self.boxsize = metadata.boxsize
        self.Omega_m = metadata.Omega_m
        self.effective_resolution = metadata.effective_resolution

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
        metadata = field_metadata("Lilow2024")
        self.path_density = path_density
        self.path_velocity = [
            path_velocity_x, path_velocity_y, path_velocity_z]

        self.coordinate_frame = metadata.coordinate_frame
        self.boxsize = metadata.boxsize
        self.Omega_m = metadata.Omega_m
        self.effective_resolution = metadata.effective_resolution

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
        metadata = field_metadata("CF4")
        self.folder = folder
        self.nsim = int(nsim)

        self.coordinate_frame = metadata.coordinate_frame
        self.boxsize = metadata.boxsize
        self.Omega_m = metadata.Omega_m

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
        metadata = field_metadata("CLONES")
        self.file_path = file_path

        self.coordinate_frame = metadata.coordinate_frame
        self.boxsize = metadata.boxsize
        self.Omega_m = metadata.Omega_m

    def load_density(self):
        with File(self.file_path, "r") as f:
            field = f["density"][...]

        grid = field.shape[0]
        field /= (self.boxsize * 1e3 / grid)**3

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
        metadata = field_metadata(f"HAMLET_V{self.version}")

        self.coordinate_frame = metadata.coordinate_frame
        self.Omega_m = metadata.Omega_m
        self.H0 = metadata.H0
        self.dtype = np.float32

        if self.version == 0:
            folder = str(1 + (self.nsim // 2))
            self.tag = 0 if (self.nsim % 2 == 0) else 99
            self.root = join(self.base, folder)
            self.boxsize = metadata.boxsize
            self.ngrid = metadata.ngrid
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
            self.boxsize = metadata.boxsize
            self.ngrid = metadata.ngrid
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


class ManticoreLocalSWIFT_FieldLoader(BORGSPHFieldLoader):
    """
    Manticore local SWIFT/SPH field loader, in the ICRS frame.

    Parameters
    ----------
    nsim : int
        Simulation index.
    fpath_root : str
        Directory containing ``mcmc_{nsim}.hdf5`` files.
    """

    def __init__(self, nsim, fpath_root, **kwargs):
        self.nsim = int(nsim)
        metadata = field_metadata("ManticoreLocalSWIFT")
        kwargs.setdefault("boxsize", metadata.boxsize)
        kwargs.setdefault("Omega_m", metadata.Omega_m)
        kwargs.setdefault("coordinate_frame", metadata.coordinate_frame)
        kwargs.setdefault("ngrid", metadata.ngrid)
        file_path = join(fpath_root, f"mcmc_{self.nsim}.hdf5")
        super().__init__(file_path, **kwargs)


class ManticoreLocalCOLA_FieldLoader(BORGFieldLoader):
    """
    Manticore local COLA/BORG field loader, in the ICRS frame.

    Parameters
    ----------
    nsim : int
        Simulation index.
    fpath_root : str
        Directory containing ``mcmc_{nsim}.hdf5`` files with ``overdensity``
        and ``velocity`` datasets.
    """

    def __init__(self, nsim, fpath_root, which_MAS="CIC", **kwargs):
        self.which_MAS = field_mas_directory(which_MAS)
        super().__init__(nsim, Path(fpath_root) / self.which_MAS, **kwargs)


###############################################################################
#             Shortcut to get the appropriate field class.                    #
###############################################################################


_FIELD_LOADERS = {
    "Carrick2015": Carrick2015_FieldLoader,
    "Lilow2024": Lilow2024_FieldLoader,
    "CF4": CF4_FieldLoader,
    "CLONES": CLONES_FieldLoader,
    "ManticoreLocalCOLA": ManticoreLocalCOLA_FieldLoader,
    "ManticoreLocalSWIFT": ManticoreLocalSWIFT_FieldLoader,
}

FIELD_METADATA = {
    "Carrick2015": FieldMetadata(
        name="Carrick2015",
        coordinate_frame="galactic",
        boxsize=400.0,
        Omega_m=0.3,
        effective_resolution=4.0,
        require_cached_products=False,
        cache_group="Carrick2015",
        description="Carrick+2015 2M++ density and velocity fields."),
    "Lilow2024": FieldMetadata(
        name="Lilow2024",
        coordinate_frame="galactic",
        boxsize=400.0,
        Omega_m=0.3175,
        effective_resolution=4.0,
        require_cached_products=True,
        cache_group="Lilow2024",
        description="Lilow+2024 density and velocity fields."),
    "CF4": FieldMetadata(
        name="CF4",
        coordinate_frame="supergalactic",
        boxsize=1000.0,
        Omega_m=0.3,
        require_cached_products=True,
        cache_group="CF4",
        description="Cosmicflows-4 constrained realisation fields."),
    "CLONES": FieldMetadata(
        name="CLONES",
        coordinate_frame="supergalactic",
        boxsize=500.0,
        Omega_m=0.307115,
        require_cached_products=True,
        cache_group="CLONES",
        description="CLONES z=0 density and velocity fields."),
    "HAMLET_V0": FieldMetadata(
        name="HAMLET_V0",
        coordinate_frame="supergalactic",
        boxsize=1000.0,
        Omega_m=0.3,
        ngrid=256,
        H0=74.6,
        require_cached_products=True,
        cache_group="HAMLET_V0",
        description="HAMLET version-0 z=0 fields."),
    "HAMLET_V1": FieldMetadata(
        name="HAMLET_V1",
        coordinate_frame="supergalactic",
        boxsize=500.0,
        Omega_m=0.3,
        ngrid=128,
        H0=74.6,
        require_cached_products=True,
        cache_group="HAMLET_V1",
        description="HAMLET version-1 z=0 fields."),
    "CB1": FieldMetadata(
        name="CB1",
        coordinate_frame="icrs",
        boxsize=677.7,
        Omega_m=0.307,
        ngrid=1024,
        production_method="nbody_mas_sph",
        storage_schema="density_momentum",
        require_cached_products=True,
        cache_group="CB1",
        description="CSiBORG1 z=0 SPH fields."),
    "CB2": FieldMetadata(
        name="CB2",
        coordinate_frame="icrs",
        boxsize=676.6,
        Omega_m=0.3111,
        ngrid=1024,
        production_method="nbody_mas_sph",
        storage_schema="density_momentum",
        require_cached_products=True,
        cache_group="CB2",
        description="CSiBORG2 z=0 SPH fields."),
    "ManticoreLocalCOLA": FieldMetadata(
        name="ManticoreLocalCOLA",
        coordinate_frame="icrs",
        boxsize=float("nan"),
        production_method="borg_forward_grid",
        storage_schema="overdensity_velocity",
        require_cached_products=False,
        cache_group="ManticoreLocalCOLA",
        description="Generic local BORG/COLA density and velocity fields."),
    "ManticoreLocalSWIFT": FieldMetadata(
        name="ManticoreLocalSWIFT",
        coordinate_frame="icrs",
        boxsize=681.1,
        Omega_m=0.306,
        ngrid=1024,
        production_method="nbody_mas_sph",
        storage_schema="density_momentum",
        require_cached_products=True,
        cache_group="ManticoreLocalSWIFT",
        description="Local Manticore SWIFT/SPH density and momentum fields."),
}

UNKNOWN_FIELD_METADATA = FieldMetadata(
    name="unknown",
    coordinate_frame="unknown",
    boxsize=float("nan"),
    require_cached_products=True,
    cache_group="unknown",
    description="Field has not been classified for raw runtime reads.")


def _is_manticore_local_cola(name):
    name_lower = str(name).lower()
    return (
        name_lower.startswith("manticorelocal")
        and name_lower.endswith("cola")
    )


def _is_manticore_local_swift(name):
    name_lower = str(name).lower()
    return (
        name_lower.startswith("manticorelocal")
        and name_lower.endswith("swift")
    )


def field_metadata(name):
    """Return static metadata for a supported reconstruction field."""
    if name in FIELD_METADATA:
        return FIELD_METADATA[name]

    name_lower = str(name).lower()
    if name_lower.startswith("hamlet_v0"):
        return FIELD_METADATA["HAMLET_V0"]
    if name_lower.startswith("hamlet_v1"):
        return FIELD_METADATA["HAMLET_V1"]
    if _is_manticore_local_cola(name):
        return FieldMetadata(
            name=str(name),
            coordinate_frame="icrs",
            boxsize=float("nan"),
            production_method="borg_forward_grid",
            storage_schema="overdensity_velocity",
            require_cached_products=False,
            cache_group=str(name),
            description="Generic local BORG/COLA density and velocity fields.")
    if _is_manticore_local_swift(name):
        return FieldMetadata(
            name=str(name),
            coordinate_frame="icrs",
            boxsize=float("nan"),
            production_method="nbody_mas_sph",
            storage_schema="density_momentum",
            require_cached_products=True,
            cache_group=str(name),
            description="Local Manticore SWIFT/SPH density and momentum fields.")

    return UNKNOWN_FIELD_METADATA


def field_product_policy(name):
    """Return field-product cache metadata."""
    return field_metadata(name)


def field_requires_cached_products(name):
    """Return whether field-derived products must already exist on disk."""
    return field_metadata(name).require_cached_products


def field_allows_raw_product_reads(name):
    """Return whether raw field reads are allowed to build products."""
    return field_metadata(name).raw_read_allowed


def supported_field_names():
    """Return the field names and patterns known to the loader registry."""
    return (*FIELD_METADATA, "ManticoreLocal*COLA", "ManticoreLocal*SWIFT")


def name2field_loader(name):
    """Convert a field name to a field loader class."""
    if name in _FIELD_LOADERS:
        return _FIELD_LOADERS[name]
    if name.lower().startswith("hamlet"):
        return Hamlet_FieldLoader
    if _is_manticore_local_cola(name):
        return ManticoreLocalCOLA_FieldLoader
    if _is_manticore_local_swift(name):
        return ManticoreLocalSWIFT_FieldLoader
    raise ValueError(f"Unknown field loader: {name}")
