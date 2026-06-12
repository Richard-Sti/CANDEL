#!/usr/bin/env python
"""Visualise the local Manticore velocity field.

The script caches the expensive field-derived quantities:

* radial velocity on Galactic HEALPix lines of sight, volume-weighted over a
  local radial interval and averaged over realisations;
* enclosed volume-mean velocity vectors around the observer;
* the velocity interpolated at the observer position.

The inferred external velocity is read from a CANDEL posterior HDF5 file and is
added only at plotting/summary time, so the Manticore cache can be reused with
different posterior files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import healpy as hp
import h5py
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import candel
from candel.field.loader import ManticoreLocalSWIFT_FieldLoader


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
DEFAULT_FIELD_ROOT = (
    ROOT
    / "data/MANTICORE/2MPP_MULTIBIN_N256_DES_V2/sph_fields_new_feb/sph_fields"
)
DEFAULT_POSTERIOR = (
    ROOT
    / "results/TRGBH0_paper/table/"
    / "EDD_TRGB_rhoSmoothR4_MAS-PCS_sel-TRGB_magnitude_ManticoreLocalCOLA_main.hdf5"
)
DEFAULT_CACHE = ROOT / "notebooks/manticore_velocity_field_cache.npz"
DEFAULT_OUTDIR = ROOT / "notebooks/manticore_velocity_field"
CACHE_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--field-root", type=Path, default=DEFAULT_FIELD_ROOT)
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--sky-rmin", type=float, default=0.0,
                        help="Minimum radius for sky-map vrad average in Mpc/h.")
    parser.add_argument("--sky-rmax", type=float, default=15.0,
                        help="Maximum radius for sky-map vrad average in Mpc/h.")
    parser.add_argument("--sky-num-radii", type=int, default=16,
                        help="Number of radii used for the sky-map average.")
    parser.add_argument("--bulk-radii", type=float, nargs="+",
                        default=[10, 20, 30, 40, 50],
                        help="Enclosed bulk-flow radii in Mpc/h.")
    parser.add_argument("--realisations", type=int, nargs="*", default=None,
                        help="Manticore realisation indices. Default: all.")
    parser.add_argument("--max-realisations", type=int, default=None,
                        help="Optional cap after sorting available indices.")
    parser.add_argument("--posterior-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def natural_mcmc_indices(field_root: Path) -> list[int]:
    out = []
    for path in field_root.glob("mcmc_*.hdf5"):
        match = re.fullmatch(r"mcmc_(\d+)\.hdf5", path.name)
        if match:
            out.append(int(match.group(1)))
    return sorted(out)


def selected_realisations(args: argparse.Namespace) -> np.ndarray:
    if args.realisations is None:
        indices = natural_mcmc_indices(args.field_root)
    else:
        indices = sorted(set(args.realisations))
    if args.max_realisations is not None:
        indices = indices[:args.max_realisations]
    if not indices:
        raise ValueError(f"No Manticore realisations found in {args.field_root}.")
    return np.asarray(indices, dtype=np.int16)


def sky_radii(args: argparse.Namespace) -> np.ndarray:
    if args.sky_rmax < args.sky_rmin:
        raise ValueError("`--sky-rmax` must be greater than `--sky-rmin`.")
    if args.sky_num_radii < 2:
        raise ValueError("`--sky-num-radii` must be at least 2.")
    return np.linspace(args.sky_rmin, args.sky_rmax, args.sky_num_radii,
                       dtype=np.float32)


def radial_volume_average(values: np.ndarray, radii: np.ndarray) -> np.ndarray:
    weights = radii.astype(np.float32)**2
    denominator = np.trapezoid(weights, radii)
    if denominator <= 0:
        raise ValueError("The sky radial interval must have non-zero volume.")
    return np.trapezoid(
        values * weights[:, None], radii, axis=0) / denominator


def healpix_shell(nside: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
    ell = np.rad2deg(phi)
    b = 90.0 - np.rad2deg(theta)
    rhat = candel.galactic_to_radec_cartesian(ell, b).astype(np.float32)
    return ell.astype(np.float32), b.astype(np.float32), rhat


def subcube_slices(ngrid: int, boxsize: float, observer_pos: np.ndarray,
                   radius: float) -> tuple[slice, slice, slice]:
    cellsize = boxsize / ngrid
    lo = np.floor((observer_pos - radius) / cellsize - 0.5).astype(int) - 2
    hi = np.ceil((observer_pos + radius) / cellsize - 0.5).astype(int) + 3
    lo = np.clip(lo, 0, ngrid)
    hi = np.clip(hi, 0, ngrid)
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


def grid_coordinates(slices: tuple[slice, slice, slice],
                     cellsize: float) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray]:
    return tuple(
        ((np.arange(slc.start, slc.stop, dtype=np.float32) + 0.5) * cellsize)
        for slc in slices
    )


def local_distances(coords: tuple[np.ndarray, np.ndarray, np.ndarray],
                    observer_pos: np.ndarray) -> np.ndarray:
    dx = coords[0][:, None, None] - observer_pos[0]
    dy = coords[1][None, :, None] - observer_pos[1]
    dz = coords[2][None, None, :] - observer_pos[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float32)


def interpolate_component(field: np.ndarray,
                          coords: tuple[np.ndarray, np.ndarray, np.ndarray],
                          points: np.ndarray) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        coords, field, bounds_error=False, fill_value=0.0, method="linear")
    return interpolator(points).astype(np.float32)


def read_realisation(field_root: Path, nsim: int,
                     slices: tuple[slice, slice, slice]) -> tuple[np.ndarray,
                                                                  h5py.File]:
    path = field_root / f"mcmc_{nsim}.hdf5"
    h5 = h5py.File(path, "r")
    density = h5["density"][slices].astype(np.float32, copy=False)
    return density, h5


def velocity_component(h5: h5py.File, density: np.ndarray, component: int,
                       slices: tuple[slice, slice, slice]) -> np.ndarray:
    momentum = h5[f"p{component}"][slices].astype(np.float32, copy=False)
    return np.divide(momentum, density, out=momentum, where=density != 0)


def compute_cache(args: argparse.Namespace, realisations: np.ndarray) -> dict:
    loader = ManticoreLocalSWIFT_FieldLoader(
        int(realisations[0]), str(args.field_root))
    with h5py.File(loader.fname, "r") as f:
        ngrid = int(f["density"].shape[0])

    bulk_radii = np.asarray(args.bulk_radii, dtype=np.float32)
    radii_sky = sky_radii(args)
    max_radius = float(max(np.max(bulk_radii), np.max(radii_sky)))
    slices = subcube_slices(ngrid, loader.boxsize, loader.observer_pos,
                            max_radius)
    cellsize = loader.boxsize / ngrid
    coords = grid_coordinates(slices, cellsize)
    dist = local_distances(coords, loader.observer_pos)

    ell, b, rhat = healpix_shell(args.nside)
    sky_points = (
        loader.observer_pos[None, None, :]
        + radii_sky[:, None, None] * rhat[None, :, :]
    ).reshape(-1, 3)
    observer_point = loader.observer_pos[None, :]

    nreal = len(realisations)
    npix = hp.nside2npix(args.nside)
    sky_vrad = np.empty((nreal, npix), dtype=np.float32)
    bulk_vectors = np.empty((nreal, len(bulk_radii), 3), dtype=np.float32)
    observer_velocity = np.empty((nreal, 3), dtype=np.float32)

    for i, nsim in enumerate(realisations):
        print(f"Processing Manticore realisation {nsim} ({i + 1}/{nreal})",
              flush=True)
        density, h5 = read_realisation(args.field_root, int(nsim), slices)
        vrad = np.zeros(npix, dtype=np.float32)
        try:
            for comp in range(3):
                vel = velocity_component(h5, density, comp, slices)
                v_sky = interpolate_component(
                    vel, coords, sky_points).reshape(len(radii_sky), npix)
                vrad += radial_volume_average(v_sky, radii_sky) * rhat[:, comp]
                v_obs_comp = interpolate_component(vel, coords,
                                                   observer_point)[0]
                observer_velocity[i, comp] = v_obs_comp
                for j, radius in enumerate(bulk_radii):
                    mask = dist <= radius
                    if np.any(mask):
                        bulk_vectors[i, j, comp] = np.mean(vel[mask])
                    else:
                        bulk_vectors[i, j, comp] = v_obs_comp
                del vel
        finally:
            h5.close()
        sky_vrad[i] = vrad
        del density

    return {
        "cache_version": np.asarray(CACHE_VERSION, dtype=np.int16),
        "field_root": np.asarray(str(args.field_root)),
        "realisations": realisations,
        "nside": np.asarray(args.nside, dtype=np.int16),
        "sky_radii": radii_sky,
        "sky_weighting": np.asarray("volume_r2_dr"),
        "bulk_radii": bulk_radii,
        "observer_pos": loader.observer_pos.astype(np.float32),
        "boxsize": np.asarray(loader.boxsize, dtype=np.float32),
        "ell": ell,
        "b": b,
        "rhat_icrs": rhat,
        "sky_vrad": sky_vrad,
        "bulk_vectors": bulk_vectors,
        "observer_velocity": observer_velocity,
    }


def cache_matches(cache: dict, args: argparse.Namespace,
                  realisations: np.ndarray) -> bool:
    required = {
        "cache_version", "field_root", "nside", "sky_radii",
        "realisations", "bulk_radii",
    }
    if not required.issubset(cache):
        return False
    checks = [
        int(cache["cache_version"]) == CACHE_VERSION,
        str(cache["field_root"].item()) == str(args.field_root),
        int(cache["nside"]) == args.nside,
        np.allclose(cache["sky_radii"], sky_radii(args)),
        np.array_equal(cache["realisations"], realisations),
        np.allclose(cache["bulk_radii"], np.asarray(args.bulk_radii)),
    ]
    return all(checks)


def load_or_build_cache(args: argparse.Namespace) -> dict:
    realisations = selected_realisations(args)
    if args.cache.exists() and not args.rebuild_cache:
        with np.load(args.cache, allow_pickle=False) as data:
            cache = {key: data[key] for key in data.files}
        if cache_matches(cache, args, realisations):
            print(f"Using cached Manticore data: {args.cache}")
            return cache
        print("Cache metadata do not match current arguments; rebuilding.")

    cache = compute_cache(args, realisations)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.cache, **cache)
    print(f"Cached Manticore data: {args.cache}")
    return cache


def read_vext_samples(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        samples = f["samples"]
        if "Vext" in samples:
            return samples["Vext"][...].astype(np.float32)
        mag = samples["Vext_mag"][...]
        ell = samples["Vext_ell"][...]
        b = samples["Vext_b"][...]
    return (mag[:, None] * candel.galactic_to_radec_cartesian(ell, b)).astype(
        np.float32)


def vector_galactic(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray]:
    mag, ell, b = candel.radec_cartesian_to_galactic(
        vectors[..., 0], vectors[..., 1], vectors[..., 2])
    return np.atleast_1d(mag), np.atleast_1d(ell), np.atleast_1d(b)


def wrap_lon(lon: np.ndarray | float) -> np.ndarray | float:
    return np.remainder(lon, 360.0)


def vector_summary(vectors: np.ndarray) -> dict[str, float]:
    mag, ell, b = vector_galactic(vectors)
    mean_vec = np.mean(vectors, axis=0)
    mean_mag, mean_ell, mean_b = vector_galactic(mean_vec[None, :])
    mean_ell = float(mean_ell[0])
    lon_offset = (ell - mean_ell + 180.0) % 360.0 - 180.0
    qmag = np.percentile(mag, [16, 50, 84])
    qlon = wrap_lon(mean_ell + np.percentile(lon_offset, [16, 50, 84]))
    qb = np.percentile(b, [16, 50, 84])
    return {
        "mean_mag": float(mean_mag[0]),
        "mean_ell": mean_ell,
        "mean_b": float(mean_b[0]),
        "mag16": float(qmag[0]),
        "mag50": float(qmag[1]),
        "mag84": float(qmag[2]),
        "ell16": float(qlon[0]),
        "ell50": float(qlon[1]),
        "ell84": float(qlon[2]),
        "b16": float(qb[0]),
        "b50": float(qb[1]),
        "b84": float(qb[2]),
    }


def combined_draws(cache: dict, vext: np.ndarray, ndraw: int,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nreal = cache["bulk_vectors"].shape[0]
    nsamp = vext.shape[0]
    ireal = rng.integers(0, nreal, size=ndraw)
    isamp = rng.integers(0, nsamp, size=ndraw)
    bulk = cache["bulk_vectors"][ireal] + vext[isamp, None, :]
    observer = cache["observer_velocity"][ireal] + vext[isamp]
    return bulk, observer


def sky_vext_radial(vext: np.ndarray, rhat: np.ndarray) -> np.ndarray:
    mean_vext = np.mean(vext, axis=0)
    return rhat @ mean_vext


def plot_sky_map(map_data: np.ndarray, path: Path) -> None:
    vmax = float(np.nanpercentile(np.abs(map_data), 99))
    vmax = max(vmax, 1.0)
    plt.close("all")
    hp.mollview(
        map_data,
        title="",
        unit=r"$\langle V_{\rm rad}\rangle\ [\mathrm{km}\,\mathrm{s}^{-1}]$",
        cmap="coolwarm",
        min=-vmax,
        max=vmax,
        xsize=1600,
        notext=True,
        margins=(0.01, 0.01, 0.01, 0.08),
    )
    fig = plt.gcf()
    fig.set_size_inches(3.35, 2.25, forward=True)
    for text in fig.findobj(Text):
        text.set_fontsize(7)
        if r"\langle V_{\rm rad}\rangle" in text.get_text():
            x, y = text.get_position()
            text.set_position((x, y - 0.08))
    for ax in fig.axes:
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
    hp.graticule(color="0.75", alpha=0.5, linewidth=0.5)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_bulk_flow(cache: dict, bulk_plus_vext: np.ndarray,
                   outpath: Path) -> None:
    radii = cache["bulk_radii"]
    mag_plus = np.linalg.norm(bulk_plus_vext, axis=-1)
    q_plus = np.percentile(mag_plus, [16, 50, 84], axis=0)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.fill_between(radii, q_plus[0], q_plus[2], color="tab:blue", alpha=0.22,
                    linewidth=0)
    ax.plot(radii, q_plus[1], color="tab:blue", label="Manticore + Vext")
    ax.set_xlabel(r"Radius [$h^{-1}\,\mathrm{Mpc}$]")
    ax.set_ylabel(r"Bulk-flow magnitude [$\mathrm{km}\,\mathrm{s}^{-1}$]")
    ax.legend(frameon=False)
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)


def write_summary(cache: dict, vext: np.ndarray, observer_plus_vext: np.ndarray,
                  posterior_path: Path, outpath: Path) -> None:
    obs_summary = vector_summary(observer_plus_vext)
    manticore_obs_summary = vector_summary(cache["observer_velocity"])
    vext_summary = vector_summary(vext)

    lines = [
        "Manticore local velocity summary",
        "================================",
        "",
        f"Field root: {cache['field_root']}",
        f"Posterior: {posterior_path}",
        f"Realisations: {', '.join(map(str, cache['realisations']))}",
        f"Observer position [Mpc/h]: {cache['observer_pos']}",
        (
            "Sky vrad average [Mpc/h]: "
            f"{float(cache['sky_radii'][0]):.1f} to "
            f"{float(cache['sky_radii'][-1]):.1f}"
        ),
        "Sky vrad weighting: volume element (r^2 dr)",
        "",
        "Observer velocity + inferred Vext:",
        (
            "  mean vector magnitude/direction = "
            f"{obs_summary['mean_mag']:.1f} km/s toward "
            f"(l, b) = ({obs_summary['mean_ell']:.1f}, "
            f"{obs_summary['mean_b']:.1f}) deg"
        ),
        (
            "  marginal 16/50/84 percentiles: "
            f"|v| = {obs_summary['mag16']:.1f} / "
            f"{obs_summary['mag50']:.1f} / {obs_summary['mag84']:.1f} km/s, "
            f"l = {obs_summary['ell16']:.1f} / {obs_summary['ell50']:.1f} / "
            f"{obs_summary['ell84']:.1f} deg, "
            f"b = {obs_summary['b16']:.1f} / {obs_summary['b50']:.1f} / "
            f"{obs_summary['b84']:.1f} deg"
        ),
        "",
        "Manticore observer velocity only:",
        (
            "  mean vector magnitude/direction = "
            f"{manticore_obs_summary['mean_mag']:.1f} km/s toward "
            f"(l, b) = ({manticore_obs_summary['mean_ell']:.1f}, "
            f"{manticore_obs_summary['mean_b']:.1f}) deg"
        ),
        "",
        "Inferred Vext only:",
        (
            "  mean vector magnitude/direction = "
            f"{vext_summary['mean_mag']:.1f} km/s toward "
            f"(l, b) = ({vext_summary['mean_ell']:.1f}, "
            f"{vext_summary['mean_b']:.1f}) deg"
        ),
        "",
    ]
    outpath.write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.field_root = args.field_root.resolve()
    args.posterior = args.posterior.resolve()
    args.cache = args.cache.resolve()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cache = load_or_build_cache(args)
    vext = read_vext_samples(args.posterior)
    ndraw = min(args.posterior_draws, len(vext) * len(cache["realisations"]))
    bulk_plus_vext, observer_plus_vext = combined_draws(
        cache, vext, ndraw, args.seed)

    sky_mean = np.mean(cache["sky_vrad"], axis=0)
    sky_total_mean = sky_mean + sky_vext_radial(vext, cache["rhat_icrs"])

    sky_tag = (
        f"{float(cache['sky_radii'][0]):.0f}_"
        f"{float(cache['sky_radii'][-1]):.0f}Mpc_h")
    plot_sky_map(
        sky_total_mean,
        args.outdir / f"manticore_sky_vrad_{sky_tag}.png",
    )
    plot_bulk_flow(cache, bulk_plus_vext,
                   args.outdir / "manticore_bulk_flow_plus_vext.png")
    write_summary(cache, vext, observer_plus_vext, args.posterior,
                  args.outdir / "manticore_local_velocity_summary.txt")


if __name__ == "__main__":
    main()
