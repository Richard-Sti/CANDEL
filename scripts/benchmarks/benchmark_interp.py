"""Benchmark LOS interpolation: float32 (current) vs float64 (original)."""
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from candel.field.loader import BaseFieldLoader
from candel.field.field_interp import interpolate_los_density_velocity
from candel.util import radec_to_cartesian, radec_to_galactic


class MockFieldLoader(BaseFieldLoader):
    def __init__(self, boxsize=400.0, ngrid=256):
        self.boxsize = boxsize
        self.coordinate_frame = "galactic"
        self._ngrid = ngrid
        self._density = None
        self._velocity = None

    def _ensure_fields(self):
        if self._density is not None:
            return
        rng = np.random.default_rng(42)
        N = self._ngrid
        self._density = (1.0 + 0.1 * rng.standard_normal(
            (N, N, N))).astype(np.float64)
        self._density = np.clip(self._density, 0.01, None)
        self._velocity = (rng.standard_normal(
            (3, N, N, N)) * 200).astype(np.float64)

    def load_density(self):
        self._ensure_fields()
        return self._density.copy()

    def load_velocity(self):
        self._ensure_fields()
        return self._velocity.copy()


def interpolate_float64(field_loader, r, RA, dec):
    """Original float64 code path."""
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
    X = np.linspace(0.5 * cellsize,
                    field_loader.boxsize - 0.5 * cellsize, ngrid)

    f_density = RegularGridInterpolator(
        (X, X, X), density, fill_value=fill_value,
        bounds_error=False, method="linear")
    los_density = f_density(pos_flat).reshape(pos_shape[:2])
    los_density = np.exp(los_density) - eps
    los_density = np.clip(los_density, eps, None)

    velocity = field_loader.load_velocity()
    v_interp = np.empty((pos_flat.shape[0], 3))
    for i in range(3):
        f_vel = RegularGridInterpolator(
            (X, X, X), velocity[i], fill_value=None,
            bounds_error=False, method="linear")
        v_interp[:, i] = f_vel(pos_flat)

    v_interp = v_interp.reshape(*pos_shape)
    los_velocity = np.einsum('ijk,jk->ij', v_interp, rhat)
    return los_density.T, los_velocity.T


def bench(fn, loader, r, RA, dec, n_iter=5, label=""):
    # Warmup
    fn(loader, r, RA, dec)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(loader, r, RA, dec)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(f"  {label:12s}: {times.mean():.3f} ± {times.std():.3f} s "
          f"(min {times.min():.3f} s)")
    return times.mean()


def main():
    rng = np.random.default_rng(123)

    for ngrid in [128, 256]:
        for n_gal in [500, 1000]:
            n_r = 301
            RA = rng.uniform(0, 360, n_gal)
            dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n_gal)))
            r = np.linspace(0.1, 200.0, n_r)

            loader = MockFieldLoader(boxsize=400.0, ngrid=ngrid)
            loader._ensure_fields()

            print(f"\nngrid={ngrid}, n_gal={n_gal}, n_r={n_r} "
                  f"({n_gal * n_r / 1e6:.2f}M query points)")

            t64 = bench(interpolate_float64, loader, r, RA, dec,
                        label="float64")
            t32 = bench(
                lambda l, r, ra, dec: interpolate_los_density_velocity(
                    l, r, ra, dec, verbose=False),
                loader, r, RA, dec, label="numba-f32")

            print(f"  speedup: {t64 / t32:.2f}x")


if __name__ == "__main__":
    main()
