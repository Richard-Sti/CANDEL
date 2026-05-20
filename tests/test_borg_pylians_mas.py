from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


pytest.importorskip("MAS_library")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/BORG_fields/run_borg_fields.py"
SPEC = importlib.util.spec_from_file_location("run_borg_fields", SCRIPT)
run_borg_fields = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_borg_fields)


def write_particle_product(path: Path, positions: np.ndarray, velocities: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("realspace")
        scalars = group.create_group("scalars")
        scalars.create_dataset("N0", data=[4])
        scalars.create_dataset("L0", data=[4.0])
        scalars.create_dataset("L1", data=[4.0])
        scalars.create_dataset("L2", data=[4.0])
        scalars.create_dataset("corner0", data=[-2.0])
        scalars.create_dataset("corner1", data=[-2.0])
        scalars.create_dataset("corner2", data=[-2.0])
        group.create_dataset("u_pos", data=positions.astype("f4"))
        group.create_dataset("u_vel", data=velocities.astype("f4"))


def write_mcmc_metadata(path: Path) -> None:
    dtype = np.dtype([("omega_m", "f8"), ("h", "f8"), ("a0", "f8")])
    with h5py.File(path, "w") as handle:
        scalars = handle.create_group("scalars")
        scalars.create_dataset("cosmology", data=np.array([(0.31, 0.7, 1.0)], dtype=dtype))


def reference_cic(positions: np.ndarray, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resolution = 4
    boxsize = 4.0
    velocity_multiplier = 100.0
    density = np.zeros((resolution, resolution, resolution), dtype=np.float64)
    momentum = np.zeros((3, resolution, resolution, resolution), dtype=np.float64)
    density_flat = density.ravel()
    momentum_flat = momentum.reshape(3, -1)
    coords = (positions + 0.5 * boxsize) * (resolution / boxsize) - 0.5
    base = np.floor(coords).astype(np.int64)
    frac = coords - base
    velocities = velocities * velocity_multiplier

    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        ix = (base[:, 0] + dx) % resolution
        for dy in (0, 1):
            wxy = wx * (frac[:, 1] if dy else 1.0 - frac[:, 1])
            iy = (base[:, 1] + dy) % resolution
            for dz in (0, 1):
                weight = wxy * (frac[:, 2] if dz else 1.0 - frac[:, 2])
                iz = (base[:, 2] + dz) % resolution
                index = (ix * resolution + iy) * resolution + iz
                np.add.at(density_flat, index, weight)
                for axis in range(3):
                    np.add.at(momentum_flat[axis], index, weight * velocities[:, axis])

    velocity = np.zeros(density.shape + (3,), dtype=np.float64)
    nonzero = density > 0.0
    for axis in range(3):
        velocity[..., axis][nonzero] = momentum[axis][nonzero] / density[nonzero]
    return density, velocity


def test_pylians_cic_matches_centred_box_reference(tmp_path: Path) -> None:
    positions = np.array(
        [
            [-1.2, -0.7, 0.1],
            [0.4, 1.1, -1.6],
            [1.3, -1.4, 1.7],
            [-0.2, 0.5, 0.8],
        ],
        dtype=np.float32,
    )
    velocities = np.array(
        [
            [0.1, -0.2, 0.3],
            [0.0, 0.4, -0.1],
            [-0.3, 0.2, 0.5],
            [0.6, -0.4, 0.2],
        ],
        dtype=np.float32,
    )
    product = tmp_path / "product.h5"
    write_particle_product(product, positions, velocities)

    args = SimpleNamespace(sph_resolution=4, cic_chunk_size=2)
    run_borg_fields.run_cic_fields(product, "realspace", args)

    density, velocity = reference_cic(positions.astype(np.float64), velocities.astype(np.float64))
    expected_overdensity = (density / np.mean(density) - 1.0).astype(np.float32)
    with h5py.File(product, "r") as handle:
        fields = handle["realspace/cic"]
        np.testing.assert_allclose(fields["overdensity"][...], expected_overdensity, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(fields["velocity"][...], velocity.astype(np.float32), rtol=1e-5, atol=1e-5)
        assert fields.attrs["assignment_library"] == "Pylians MAS_library.MA"
        assert fields.attrs["assignment_scheme"] == "CIC"


def test_pylians_pcs_writes_product_group(tmp_path: Path) -> None:
    positions = np.array(
        [
            [-1.2, -0.7, 0.1],
            [0.4, 1.1, -1.6],
            [1.3, -1.4, 1.7],
            [-0.2, 0.5, 0.8],
        ],
        dtype=np.float32,
    )
    velocities = np.ones((positions.shape[0], 3), dtype=np.float32)
    product = tmp_path / "product.h5"
    write_particle_product(product, positions, velocities)

    args = SimpleNamespace(sph_resolution=4, cic_chunk_size=2)
    run_borg_fields.run_pylians_mas_fields(product, "realspace", args, "pcs")

    with h5py.File(product, "r") as handle:
        fields = handle["realspace/pcs"]
        assert fields["overdensity"].shape == (4, 4, 4)
        assert fields["velocity"].shape == (4, 4, 4, 3)
        assert np.isfinite(fields["overdensity"][...]).all()
        assert np.isfinite(fields["velocity"][...]).all()
        assert fields.attrs["assignment_scheme"] == "PCS"


def test_slim_product_writes_generic_borg_reader_schema(tmp_path: Path) -> None:
    positions = np.array(
        [
            [-1.2, -0.7, 0.1],
            [0.4, 1.1, -1.6],
            [1.3, -1.4, 1.7],
            [-0.2, 0.5, 0.8],
        ],
        dtype=np.float32,
    )
    velocities = np.ones((positions.shape[0], 3), dtype=np.float32)
    product = tmp_path / "product.h5"
    mcmc = tmp_path / "mcmc_1.h5"
    write_particle_product(product, positions, velocities)
    write_mcmc_metadata(mcmc)
    with h5py.File(product, "a") as handle:
        handle.attrs["source_mcmc"] = str(mcmc)
        handle["realspace"].attrs["source_mcmc"] = str(mcmc)

    args = SimpleNamespace(sph_resolution=4, cic_chunk_size=2)
    run_borg_fields.run_cic_fields(product, "realspace", args)
    run_borg_fields.slim_field_product(product, "realspace", "cic")

    with h5py.File(product, "r") as handle:
        assert "density" not in handle
        assert "density_dataset" not in handle.attrs
        for name in ("overdensity", "velocity", "vx", "vy", "vz"):
            assert name in handle
        np.testing.assert_allclose(handle["vx"][...], handle["velocity"][..., 0])
        np.testing.assert_allclose(handle["vy"][...], handle["velocity"][..., 1])
        np.testing.assert_allclose(handle["vz"][...], handle["velocity"][..., 2])
        assert handle.attrs["boxsize"] == 4.0
        assert handle.attrs["Om"] == 0.31
        assert handle.attrs["frame"] == "icrs"
        np.testing.assert_allclose(handle.attrs["observer_position"], [2.0, 2.0, 2.0])
