#!/usr/bin/env python
"""Compute the local Manticore observer velocity plus TRGB Manticore Vext."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import candel


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
DEFAULT_CACHE = ROOT / "notebooks/manticore_velocity_field_cache.npz"
DEFAULT_POSTERIOR = (
    ROOT
    / "results/TRGBH0_main/table/"
    / "EDD_TRGB_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main.hdf5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def read_vext(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["samples/Vext"], dtype=np.float32)


def galactic(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag, ell, b = candel.radec_cartesian_to_galactic(
        vectors[..., 0], vectors[..., 1], vectors[..., 2])
    return np.atleast_1d(mag), np.atleast_1d(ell), np.atleast_1d(b)


def summary(vectors: np.ndarray) -> dict[str, tuple[float, float, float]]:
    mag, ell, b = galactic(vectors)
    mean_vec = np.mean(vectors, axis=0)
    _, mean_ell, _ = galactic(mean_vec[None, :])
    ell_offset = (ell - mean_ell[0] + 180.0) % 360.0 - 180.0
    ell_q = (mean_ell[0] + np.percentile(ell_offset, [16, 50, 84])) % 360.0
    return {
        "mag": tuple(np.percentile(mag, [16, 50, 84])),
        "ell": tuple(ell_q),
        "b": tuple(np.percentile(b, [16, 50, 84])),
    }


def print_ci(name: str, values: tuple[float, float, float], unit: str) -> None:
    q16, q50, q84 = values
    print(
        f"{name} = {q50:.1f} -{q50 - q16:.1f} +{q84 - q50:.1f} {unit}")


def main() -> None:
    args = parse_args()
    with np.load(args.cache, allow_pickle=False) as data:
        observer_velocity = data["observer_velocity"]
        observer_pos = data["observer_pos"]
        realisations = data["realisations"]

    vext = read_vext(args.posterior)
    rng = np.random.default_rng(args.seed)
    ireal = rng.integers(0, len(observer_velocity), size=args.draws)
    isamp = rng.integers(0, len(vext), size=args.draws)

    combined = observer_velocity[ireal] + vext[isamp]
    mean_vec = np.mean(combined, axis=0)
    mean_mag, mean_ell, mean_b = galactic(mean_vec[None, :])
    stats = summary(combined)

    print(f"Cache: {args.cache}")
    print(f"Posterior: {args.posterior}")
    print(f"Observer position [Mpc/h]: {observer_pos}")
    print(f"Manticore realisations: {len(realisations)}")
    print("")
    print(
        "Mean vector: "
        f"|v| = {mean_mag[0]:.1f} km/s, "
        f"(l, b) = ({mean_ell[0]:.1f}, {mean_b[0]:.1f}) deg")
    print("Marginal 68% intervals:")
    print_ci("|v|", stats["mag"], "km/s")
    print_ci("l", stats["ell"], "deg")
    print_ci("b", stats["b"], "deg")


if __name__ == "__main__":
    main()
