import numpy as np
from h5py import File

from candel.field.loader import (
    BORGFieldLoader, ManticoreLocalCOLA_FieldLoader,
    ManticoreLocalSWIFT_FieldLoader,
    available_mcmc_field_indices, name2field_loader)
from candel.pvdata.volume_density import _density_unit_normalization


def test_manticore_local_cola_loader_reads_overdensity_and_velocity(tmp_path):
    fpath_root = tmp_path
    mas_root = fpath_root / "CIC"
    mas_root.mkdir()
    fname = mas_root / "mcmc_0.hdf5"

    overdensity = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10
    velocity = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)

    with File(fname, "w") as f:
        f.attrs["boxsize"] = 123.0
        f.attrs["Omega_m"] = 0.25
        f.attrs["grid_shape"] = np.array([2, 2, 2])
        f.attrs["frame"] = "icrs"
        f.create_dataset("overdensity", data=overdensity)
        f.create_dataset("velocity", data=velocity)

    loader_cls = name2field_loader("ManticoreLocalCOLA")
    assert loader_cls is ManticoreLocalCOLA_FieldLoader

    loader = loader_cls(nsim=0, fpath_root=str(fpath_root))

    np.testing.assert_allclose(loader.load_density(), 1 + overdensity)
    np.testing.assert_allclose(
        loader.load_velocity(), np.moveaxis(velocity, -1, 0))
    np.testing.assert_allclose(
        loader.load_velocity_component(1), velocity[..., 1])
    assert loader.coordinate_frame == "icrs"
    assert loader.boxsize == 123.0
    assert loader.Omega_m == 0.25
    assert loader.ngrid == 2


def test_manticore_local_cola_loader_reads_generic_borg_schema(tmp_path):
    fpath_root = tmp_path
    mas_root = fpath_root / "CIC"
    mas_root.mkdir()
    fname = mas_root / "mcmc_0.hdf5"

    density = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    vx = density + 10
    vy = density + 20
    vz = density + 30

    with File(fname, "w") as f:
        f.attrs["boxsize"] = 123.0
        f.attrs["Omega_m"] = 0.25
        f.attrs["observer_pos"] = np.array([1.0, 2.0, 3.0])
        f.attrs["frame"] = "icrs"
        f.create_dataset("density", data=density)
        f.create_dataset("vx", data=vx)
        f.create_dataset("vy", data=vy)
        f.create_dataset("vz", data=vz)

    loader_cls = name2field_loader("ManticoreLocalCOLA")
    assert loader_cls is ManticoreLocalCOLA_FieldLoader

    loader = loader_cls(nsim=0, fpath_root=str(fpath_root))

    np.testing.assert_allclose(loader.load_density(), density)
    np.testing.assert_allclose(loader.load_velocity(), np.stack([vx, vy, vz]))
    np.testing.assert_allclose(loader.load_velocity_component(2), vz)
    np.testing.assert_allclose(loader.observer_pos, [1.0, 2.0, 3.0])
    assert loader.boxsize == 123.0
    assert loader.Omega_m == 0.25
    assert loader.coordinate_frame == "icrs"


def test_manticore_local_cola_loader_respects_which_mas(tmp_path):
    fpath_root = tmp_path
    mas_root = fpath_root / "PCS"
    mas_root.mkdir()
    fname = mas_root / "mcmc_0.hdf5"

    density = np.ones((2, 2, 2), dtype=np.float32)
    velocity = np.zeros((2, 2, 2, 3), dtype=np.float32)
    with File(fname, "w") as f:
        f.attrs["boxsize"] = 123.0
        f.create_dataset("density", data=density)
        f.create_dataset("velocity", data=velocity)

    loader = ManticoreLocalCOLA_FieldLoader(
        nsim=0, fpath_root=str(fpath_root), which_MAS="PCS")

    assert loader.fname == str(fname)
    np.testing.assert_allclose(loader.load_density(), density)


def test_borg_loader_reads_forward_product_attrs(tmp_path):
    fpath_root = tmp_path
    fname = fpath_root / "mcmc_0.hdf5"

    delta = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10
    velocity = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)

    with File(fname, "w") as f:
        f.attrs["boxsize"] = 4.0
        f.attrs["grid_shape"] = np.array([2, 2, 2])
        f.attrs["Om"] = 0.31
        f.attrs["observer_position"] = np.array([2.0, 2.0, 2.0])
        f.attrs["frame"] = "icrs"
        f.attrs["overdensity_dataset"] = "delta_grid"
        f.attrs["velocity_dataset"] = "vel_grid"
        f.attrs["vx_dataset"] = "vx"
        f.attrs["vy_dataset"] = "vy"
        f.attrs["vz_dataset"] = "vz"
        f.create_dataset("delta_grid", data=delta)
        f.create_dataset("vel_grid", data=velocity)

    loader = BORGFieldLoader(nsim=0, fpath_root=str(fpath_root))

    np.testing.assert_allclose(loader.load_density(), 1 + delta)
    np.testing.assert_allclose(
        loader.load_velocity(), np.moveaxis(velocity, -1, 0))
    np.testing.assert_allclose(
        loader.load_velocity_component(0), velocity[..., 0])
    np.testing.assert_allclose(loader.observer_pos, [2.0, 2.0, 2.0])
    assert loader.boxsize == 4.0
    assert loader.Omega_m == 0.31
    assert loader.grid_shape == (2, 2, 2)
    assert loader.ngrid == 2
    assert loader.coordinate_frame == "icrs"


def test_manticore_local_swift_loader_reads_density_and_momenta(tmp_path):
    fpath_root = tmp_path
    fname = fpath_root / "mcmc_0.hdf5"

    density = np.full((2, 2, 2), 2.0, dtype=np.float32)
    p0 = density * 10
    p1 = density * 20
    p2 = density * 30

    with File(fname, "w") as f:
        f.create_dataset("density", data=density)
        f.create_dataset("p0", data=p0)
        f.create_dataset("p1", data=p1)
        f.create_dataset("p2", data=p2)

    loader_cls = name2field_loader("ManticoreLocalSWIFT")
    assert loader_cls is ManticoreLocalSWIFT_FieldLoader

    loader = loader_cls(nsim=0, fpath_root=str(fpath_root))

    expected_density = density / (loader.boxsize * 1e3 / 2)**3
    np.testing.assert_allclose(loader.load_density(), expected_density)
    np.testing.assert_allclose(
        loader.load_velocity(),
        np.stack([np.full_like(density, 10.0),
                  np.full_like(density, 20.0),
                  np.full_like(density, 30.0)]))
    np.testing.assert_allclose(
        loader.load_velocity_component(1), np.full_like(density, 20.0))
    assert loader.boxsize == 681.1
    assert loader.Omega_m == 0.306
    assert loader.coordinate_frame == "icrs"
    assert loader.fname == str(fname)


def test_available_mcmc_field_indices_reads_folder(tmp_path):
    for index in [11, 1, 42]:
        (tmp_path / f"mcmc_{index}.hdf5").touch()
    (tmp_path / "mcmc_bad.hdf5").touch()
    (tmp_path / "other_2.hdf5").touch()

    assert available_mcmc_field_indices(tmp_path) == [1, 11, 42]


def test_manticore_cola_los_density_is_not_raw_density_normalized():
    assert _density_unit_normalization("ManticoreLocalCOLA") is None
    assert _density_unit_normalization("ManticoreLocalSWIFT") is not None
