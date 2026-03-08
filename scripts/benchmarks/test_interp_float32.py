"""
Verify that the float32 interpolation path produces results matching
float64 to within expected precision (~1e-6 relative error).

Uses a mock field loader with known smooth fields.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from candel.field.loader import BaseFieldLoader
from candel.field.field_interp import (
    apply_gaussian_smoothing, interpolate_los_density_velocity)
from candel.util import radec_to_cartesian


class MockFieldLoader(BaseFieldLoader):
    """Deterministic smooth field for reproducible testing."""

    def __init__(self, boxsize=400.0, ngrid=64):
        self.boxsize = boxsize
        self.coordinate_frame = "galactic"
        self._ngrid = ngrid

    def load_density(self):
        N = self._ngrid
        x = np.linspace(0, 1, N)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        # Smooth density field with values ~ 0.5 to 1.5
        return (1.0 + 0.3 * np.sin(2 * np.pi * X)
                * np.cos(2 * np.pi * Y)
                * np.sin(2 * np.pi * Z)).astype(np.float64)

    def load_velocity(self):
        N = self._ngrid
        x = np.linspace(0, 1, N)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        vx = 200.0 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Z)
        vy = -150.0 * np.cos(2 * np.pi * Y) * np.sin(2 * np.pi * X)
        vz = 100.0 * np.sin(2 * np.pi * Z) * np.cos(2 * np.pi * Y)
        return np.stack([vx, vy, vz], axis=0).astype(np.float64)


def interpolate_float64_reference(field_loader, r, RA, dec):
    """Float64 reference implementation (original code path)."""
    from candel.util import radec_to_galactic
    ell, b = radec_to_galactic(RA, dec)
    rhat = radec_to_cartesian(ell, b)

    pos = (field_loader.observer_pos[None, None, :]
           + r[:, None, None] * rhat[None, :, :])
    pos_shape = pos.shape
    pos_flat = pos.reshape(-1, 3)

    eps = 1e-4
    density = np.log(field_loader.load_density() + eps)
    fill_value = np.log(1 + eps)

    ngrid = density.shape[0]
    cellsize = field_loader.boxsize / ngrid
    X = np.linspace(0.5 * cellsize, field_loader.boxsize - 0.5 * cellsize,
                    ngrid)
    grid_points = (X, X, X)

    f_density = RegularGridInterpolator(
        grid_points, density, fill_value=fill_value,
        bounds_error=False, method="linear")
    los_density = f_density(pos_flat).reshape(pos_shape[:2])
    los_density = np.exp(los_density) - eps
    los_density = np.clip(los_density, eps, None)

    velocity = field_loader.load_velocity()
    v_interp = np.empty((pos_flat.shape[0], 3))
    for i in range(3):
        f_vel = RegularGridInterpolator(
            grid_points, velocity[i], fill_value=None,
            bounds_error=False, method="linear")
        v_interp[:, i] = f_vel(pos_flat)

    v_interp = v_interp.reshape(*pos_shape)
    los_velocity = np.einsum('ijk,jk->ij', v_interp, rhat)

    return los_density.T, los_velocity.T


def main():
    loader = MockFieldLoader(boxsize=400.0, ngrid=64)
    r = np.linspace(0.1, 150.0, 201)
    # 50 random sky positions
    rng = np.random.default_rng(123)
    RA = rng.uniform(0, 360, 50)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 50)))

    # Float64 reference
    dens_ref, vel_ref = interpolate_float64_reference(loader, r, RA, dec)

    # Float32 optimized (current code)
    dens_new, vel_new = interpolate_los_density_velocity(
        loader, r, RA, dec, verbose=False)

    # Compare density
    mask = dens_ref > 1e-3  # skip near-zero values
    rel_err_dens = np.abs(dens_new[mask] - dens_ref[mask]) / dens_ref[mask]
    max_rel_dens = rel_err_dens.max()
    mean_rel_dens = rel_err_dens.mean()

    # Compare velocity
    mask_v = np.abs(vel_ref) > 1.0
    rel_err_vel = np.abs(vel_new[mask_v] - vel_ref[mask_v]) / np.abs(vel_ref[mask_v])
    max_rel_vel = rel_err_vel.max()
    mean_rel_vel = rel_err_vel.mean()

    # Absolute error for small velocities
    abs_err_vel = np.abs(vel_new - vel_ref).max()

    print("Density:")
    print(f"  max  relative error: {max_rel_dens:.2e}")
    print(f"  mean relative error: {mean_rel_dens:.2e}")
    print(f"Velocity:")
    print(f"  max  relative error: {max_rel_vel:.2e}")
    print(f"  mean relative error: {mean_rel_vel:.2e}")
    print(f"  max  absolute error: {abs_err_vel:.2e} km/s")

    # float32 precision is ~1e-7, trilinear interpolation can amplify to ~1e-5
    tol = 1e-4
    ok = max_rel_dens < tol and max_rel_vel < tol
    print(f"\nTolerance: {tol:.0e}")
    if ok:
        print("PASSED: float32 interpolation matches float64 reference.")
    else:
        print("FAILED: errors exceed tolerance.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
