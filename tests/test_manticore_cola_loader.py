import numpy as np
from h5py import File

from candel.field.loader import ManticoreCOLA_FieldLoader, name2field_loader


def test_manticore_cola_loader_reads_overdensity_and_velocity(tmp_path):
    fpath_root = tmp_path
    fname = fpath_root / "mcmc_0.hdf5"

    overdensity = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10
    velocity = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)

    with File(fname, "w") as f:
        f.create_dataset("overdensity", data=overdensity)
        f.create_dataset("velocity", data=velocity)

    loader_cls = name2field_loader("2MPP_MULTIBIN_N256_DES_V2_COLA")
    assert loader_cls is ManticoreCOLA_FieldLoader

    loader = loader_cls(nsim=0, fpath_root=str(fpath_root))

    np.testing.assert_allclose(loader.load_density(), 1 + overdensity)
    np.testing.assert_allclose(
        loader.load_velocity(), np.moveaxis(velocity, -1, 0))
    np.testing.assert_allclose(
        loader.load_velocity_component(1), velocity[..., 1])
    assert loader.coordinate_frame == "icrs"
    assert loader.boxsize == 681.1
    assert loader.Omega_m == 0.306
