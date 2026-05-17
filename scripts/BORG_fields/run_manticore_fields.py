#!/usr/bin/env python3
"""Run and validate BORG forward fields for native Manticore samples."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PATHS_FILE = SCRIPT_DIR / "paths.env"


def read_paths_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing path configuration: {path}")

    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Could not parse {path}:{line_no}: {raw_line!r}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


PATHS = read_paths_file(PATHS_FILE)


def configured_path(name: str) -> Path:
    try:
        return Path(PATHS[name]).expanduser()
    except KeyError as exc:
        raise KeyError(f"Missing {name} in {PATHS_FILE}") from exc


DEFAULT_BORG_FORWARD = configured_path("BORG_FORWARD")
DEFAULT_COSMOTOOL_SPH = configured_path("COSMOTOOL_SPH")
DEFAULT_PLOT_PYTHON = configured_path("PYTHON_PATH")
DEFAULT_PLOT_SCRIPT = Path(__file__).with_name("plot_manticore_product_slices.py")
DEFAULT_RSD_COMPARISON_PLOT_SCRIPT = Path(__file__).with_name("plot_rsd_comparison.py")
DEFAULT_RSD_CROSS_SCRIPT = Path(__file__).with_name("compute_rsd_pylians_cross_correlation.py")
DEFAULT_PM_NSTEPS = 10


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--params", type=Path, help="Template params.ini.")
    parser.add_argument("--state", help="State directory name, e.g. state_6124.")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output root. Defaults to <run-root>/forward inferred from the MCMC path.",
    )
    parser.add_argument("--borg-forward", type=Path, default=DEFAULT_BORG_FORWARD)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument(
        "--mpi-launcher",
        default=os.environ.get("BORG_MPI_LAUNCHER"),
        help="Launcher template with {nprocs}, e.g. 'srun -u -n {nprocs} --mpi=pmix'.",
    )
    parser.add_argument("--nprocs", type=int, default=int(os.environ.get("NPROCS", "8")))
    parser.add_argument("--omp-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    parser.add_argument(
        "--pm-nsteps",
        type=int,
        default=DEFAULT_PM_NSTEPS,
        help=f"Override [gravity_chain_2] pm_nsteps in generated configs. Default: {DEFAULT_PM_NSTEPS}.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-patch-missing-vobs",
        dest="patch_missing_vobs",
        action="store_false",
        help="Do not create a local MCMC copy with /scalars/BORG_vobs=[0,0,0].",
    )
    parser.set_defaults(patch_missing_vobs=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one MCMC sample.")
    run.add_argument("mcmc", type=Path, help="Path to mcmc_*.h5.")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--rsd", action="store_true", help="Write redshift-space density.")
    mode.add_argument("--real-space", dest="rsd", action="store_false", help="Write real-space density.")
    run.set_defaults(rsd=False)
    run.add_argument(
        "--include-rsd",
        action="store_true",
        help="Also run/pack the RSD field into the same single HDF5 product.",
    )
    run.add_argument(
        "--single-output",
        type=Path,
        help="Single HDF5 product path. Default: <output-root>/<subchain>/<mcmc>/manticore_fields_<iteration>.h5.",
    )
    run.add_argument(
        "--no-single-output",
        action="store_true",
        help="Leave only the split rank files; do not pack a single HDF5 product.",
    )
    run.add_argument(
        "--sph-fields",
        dest="sph_fields",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run.add_argument(
        "--mas",
        choices=("sph", "cic"),
        default="sph",
        help="Mass-assignment scheme for final density/velocity fields: sph or cic. Default: sph.",
    )
    run.add_argument("--cosmotool-sph", type=Path, default=DEFAULT_COSMOTOOL_SPH)
    run.add_argument(
        "--sph-resolution",
        type=int,
        help="Grid resolution for SPH or CIC. Default: /<mode>/scalars/N0.",
    )
    run.add_argument(
        "--cic-chunk-size",
        type=int,
        default=1_000_000,
        help="Particle chunk size for CIC gridding. Default: 1000000.",
    )
    run.add_argument(
        "--sph-radius-limit",
        type=float,
        help="SPH spherical radius limit. Default: boxsize, which includes the full cube.",
    )
    run.add_argument(
        "--sph-periodic",
        type=int,
        choices=(0, 1),
        default=1,
        help="Pass periodic=0/1 to simple3DFilter. Default: 1.",
    )
    run.add_argument(
        "--sph-threads",
        type=int,
        default=int(os.environ.get("SPH_OMP_THREADS", os.environ.get("NPROCS", "8"))),
        help="OpenMP threads for CosmoTool SPH. Default: SPH_OMP_THREADS or NPROCS.",
    )
    run.add_argument(
        "--keep-sph-work",
        action="store_true",
        help="Keep intermediate CosmoTool particle and field files.",
    )
    run.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Do not write product slice plots under PRODUCT_PARENT/plots.",
    )
    run.add_argument("--plot-python", type=Path, default=DEFAULT_PLOT_PYTHON)
    run.add_argument("--plot-script", type=Path, default=DEFAULT_PLOT_SCRIPT)
    run.add_argument("--plot-slice-index", type=int, help="Default: middle slice.")
    run.set_defaults(sph_fields=True, plots=True)
    add_common_args(run)

    validate = subparsers.add_parser(
        "validate-rsd",
        help="Choose random MCMC sample(s), run RSD, and compare to /scalars/BORG_final_density.",
    )
    validate.add_argument("chain_dir", type=Path, help="Directory containing mcmc_*.h5 files.")
    validate.add_argument("--samples", type=int, default=1, help="Number of random samples. Default: 1.")
    validate.add_argument("--seed", type=int, default=12345, help="Random seed. Default: 12345.")
    validate.add_argument("--glob", default="mcmc_*.h5", help="MCMC filename glob. Default: mcmc_*.h5.")
    validate.add_argument("--atol", type=float, default=1e-3)
    validate.add_argument("--rtol", type=float, default=1e-5)
    validate.add_argument(
        "--no-rsd-comparison-plot",
        dest="rsd_comparison_plot",
        action="store_false",
        help="Do not write a generated/reference/difference density slice plot.",
    )
    validate.add_argument("--rsd-comparison-plot-dir", type=Path, help="Default: RSD output directory/plots.")
    validate.add_argument("--rsd-comparison-plot-axis", type=int, choices=(0, 1, 2), default=2)
    validate.add_argument("--rsd-comparison-plot-index", type=int, help="Default: middle slice.")
    validate.add_argument("--plot-python", type=Path, default=DEFAULT_PLOT_PYTHON)
    validate.add_argument("--rsd-comparison-plot-script", type=Path, default=DEFAULT_RSD_COMPARISON_PLOT_SCRIPT)
    validate.add_argument(
        "--no-rsd-cross-correlation",
        dest="rsd_cross_correlation",
        action="store_false",
        help="Do not compute the Pylians RSD cross-correlation against /scalars/BORG_final_density.",
    )
    validate.add_argument("--rsd-cross-output-dir", type=Path, help="Default: RSD output directory/plots.")
    validate.add_argument("--rsd-cross-boxsize", type=float, default=681.0, help="Box size in Mpc/h. Default: 681.")
    validate.add_argument("--rsd-cross-axis", type=int, default=0, help="Pylians line-of-sight axis. Default: 0.")
    validate.add_argument("--rsd-cross-threads", type=int, default=4, help="Pylians thread count. Default: 4.")
    validate.add_argument(
        "--rsd-cross-kmax",
        type=float,
        default=0.2,
        help="Maximum k for the mean-r validation statistic. Default: 0.2 h/Mpc.",
    )
    validate.add_argument(
        "--rsd-cross-min-mean-r",
        type=float,
        default=0.99,
        help="Warn if mean r(k) up to --rsd-cross-kmax is below this value. Default: 0.99.",
    )
    validate.add_argument("--rsd-cross-script", type=Path, default=DEFAULT_RSD_CROSS_SCRIPT)
    validate.set_defaults(rsd_comparison_plot=True)
    validate.set_defaults(rsd_cross_correlation=True)
    add_common_args(validate)

    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def require_file_preserve_symlink(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def iteration_from_mcmc(mcmc: Path) -> int:
    match = re.fullmatch(r"mcmc_(\d+)", mcmc.stem)
    if match is None:
        raise ValueError(f"Could not infer iteration from MCMC filename: {mcmc}")
    return int(match.group(1))


def infer_layout(mcmc: Path, output_root: Path | None) -> tuple[Path, str, Path]:
    chain_dir = mcmc.parent
    if output_root is not None:
        return output_root.expanduser().resolve(), chain_dir.name, chain_dir
    if chain_dir.parent.name != "chain":
        raise ValueError("Expected <run-root>/chain/<subchain>/mcmc_*.h5; pass --output-root.")
    return chain_dir.parent.parent / "forward", chain_dir.name, chain_dir


def find_params(chain_dir: Path, params: Path | None, state: str | None) -> Path:
    if params is not None:
        return require_file(params, "params.ini")
    if state is not None:
        return require_file(chain_dir / state / "params.ini", "params.ini")

    candidates = sorted(chain_dir.glob("state_*/params.ini"))
    if not candidates:
        raise FileNotFoundError(f"No state_*/params.ini found under {chain_dir}")
    if len(candidates) > 1:
        print(f"Using template params.ini: {candidates[0]}", flush=True)
        print("Pass --params or --state to choose a different state config.", flush=True)
    return candidates[0].resolve()


def set_ini_value(lines: list[str], section: str, key: str, value: str) -> list[str]:
    section_re = re.compile(r"^\s*\[(?P<section>[^\]]+)\]\s*$")
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
    out: list[str] = []
    in_section = False
    saw_section = False
    replaced = False

    for line in lines:
        match = section_re.match(line)
        if match:
            if in_section and not replaced:
                out.append(f"{key}={value}\n")
                replaced = True
            in_section = match.group("section") == section
            saw_section = saw_section or in_section
        if in_section and key_re.match(line):
            out.append(f"{key}={value}\n")
            replaced = True
        else:
            out.append(line)

    if in_section and not replaced:
        out.append(f"{key}={value}\n")
        replaced = True
    if not saw_section:
        raise ValueError(f"Section [{section}] was not found in params.ini")
    return out


def write_run_config(template: Path, config: Path, do_rsd: bool, console_output: Path, pm_nsteps: int) -> None:
    lines = template.read_text().splitlines(keepends=True)
    lines = set_ini_value(lines, "gravity_chain_2", "do_rsd", "True" if do_rsd else "False")
    lines = set_ini_value(lines, "gravity_chain_2", "pm_nsteps", str(pm_nsteps))
    lines = set_ini_value(lines, "system", "console_output", str(console_output))
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("".join(lines))


def has_borg_vobs(mcmc: Path) -> bool:
    import h5py

    with h5py.File(mcmc, "r") as handle:
        return "/scalars/BORG_vobs" in handle


def patch_mcmc_missing_vobs(mcmc: Path, out_dir: Path, create: bool) -> Path:
    import h5py

    if has_borg_vobs(mcmc):
        return mcmc
    patched = out_dir / "mcmc_with_BORG_vobs" / mcmc.name
    if not create:
        return patched
    if patched.is_file() and patched.stat().st_mtime >= mcmc.stat().st_mtime and has_borg_vobs(patched):
        return patched

    patched.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mcmc, patched)
    with h5py.File(patched, "a") as handle:
        handle.require_group("scalars").create_dataset("BORG_vobs", data=[0.0, 0.0, 0.0])
    return patched


def output_path(output_pattern: Path, iteration: int) -> Path:
    return Path(str(output_pattern) % iteration)


def launcher(args: argparse.Namespace) -> list[str]:
    if args.mpi_launcher is None:
        return [args.mpirun, "-np", str(args.nprocs)]
    return shlex.split(args.mpi_launcher.format(nprocs=args.nprocs))


def single_output_path(mcmc: Path, args: argparse.Namespace) -> Path:
    iteration = iteration_from_mcmc(mcmc)
    if getattr(args, "single_output", None) is not None:
        return args.single_output.expanduser().resolve()
    output_root, subchain, _ = infer_layout(mcmc, args.output_root)
    return output_root / subchain / mcmc.stem / f"manticore_fields_{iteration:04d}.h5"


def run_forward(mcmc: Path, args: argparse.Namespace, do_rsd: bool) -> tuple[Path, int, Path]:
    mcmc = require_file(mcmc, "MCMC file")
    borg_forward = require_file(args.borg_forward, "borg_forward executable")
    output_root, subchain, chain_dir = infer_layout(mcmc, args.output_root)
    params = find_params(chain_dir, args.params, args.state)
    iteration = iteration_from_mcmc(mcmc)

    mode = "rsd" if do_rsd else "realspace"
    out_dir = output_root / subchain / mcmc.stem / mode
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    mcmc_for_borg = patch_mcmc_missing_vobs(mcmc, out_dir, create=not args.dry_run) if args.patch_missing_vobs else mcmc

    config = out_dir / f"params_{mode}.ini"
    output_pattern = out_dir / "output_%04d.h5"
    if args.pm_nsteps < 1:
        raise ValueError("--pm-nsteps must be >= 1")
    write_run_config(params, config, do_rsd=do_rsd, console_output=log_dir / "borg_forward", pm_nsteps=args.pm_nsteps)

    cmd = [
        *launcher(args),
        str(borg_forward),
        "--config",
        str(config),
        "--mcmc",
        str(mcmc_for_borg),
        "--output",
        str(output_pattern),
        "--pos",
        "--vel",
        "--output_split",
    ]

    output = output_path(output_pattern, iteration)
    print(f"MCMC: {mcmc}", flush=True)
    if mcmc_for_borg != mcmc:
        print(f"MCMC passed to BORG: {mcmc_for_borg}", flush=True)
    print(f"Template config: {params}", flush=True)
    print(f"Run config: {config}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"BORG working/log directory: {log_dir}", flush=True)
    print(f"Split output files: {output}_0 ... {output}_{args.nprocs - 1}", flush=True)
    print(f"Mode: {'redshift space' if do_rsd else 'real space'}", flush=True)
    print("Command:", flush=True)
    print(" ".join(cmd), flush=True)

    if args.dry_run:
        return output_pattern, iteration, out_dir

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp_threads)
    subprocess.run(cmd, check=True, env=env, cwd=log_dir)
    validate_split_outputs(output_pattern, iteration, args.nprocs)
    print(f"Saved output directory: {out_dir}", flush=True)
    print(f"Saved datasets in each split file: /final_density, /s_hat_field, /u_pos, /u_vel", flush=True)
    return output_pattern, iteration, out_dir


def pack_split_outputs(
    output_pattern: Path,
    iteration: int,
    nprocs: int,
    product: Path,
    group_name: str,
    mcmc: Path,
) -> None:
    import h5py

    output = output_path(output_pattern, iteration)
    rank_paths = [Path(f"{output}_{rank}") for rank in range(nprocs)]
    missing = [path for path in rank_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing split output files:\n" + "\n".join(str(path) for path in missing))

    density_slabs = []
    shat_slabs = []
    particle_counts = []
    for path in rank_paths:
        with h5py.File(path, "r") as handle:
            density_slabs.append(handle["final_density"].shape[0])
            shat_slabs.append(handle["s_hat_field"].shape[0])
            particle_counts.append(handle["u_pos"].shape[0])

    product.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(rank_paths[0], "r") as first, h5py.File(product, "a") as out:
        if group_name in out:
            del out[group_name]
        group = out.create_group(group_name)
        first.copy("scalars", group, name="scalars")
        group.attrs["source_mcmc"] = str(mcmc)
        group.attrs["source_split_pattern"] = str(output) + "_<rank>"
        group.attrs["nprocs"] = nprocs
        group.attrs["iteration"] = iteration

        density_shape = (sum(density_slabs),) + first["final_density"].shape[1:]
        shat_shape = (sum(shat_slabs),) + first["s_hat_field"].shape[1:]
        particle_shape = (sum(particle_counts), 3)

        density = group.create_dataset("final_density", density_shape, dtype=first["final_density"].dtype, chunks=(1,) + density_shape[1:])
        shat = group.create_dataset("s_hat_field", shat_shape, dtype=first["s_hat_field"].dtype, chunks=(1,) + shat_shape[1:])
        pos = group.create_dataset(
            "u_pos",
            particle_shape,
            dtype=first["u_pos"].dtype,
            chunks=(min(1_000_000, max(1, particle_shape[0])), 3),
        )
        vel = group.create_dataset(
            "u_vel",
            particle_shape,
            dtype=first["u_vel"].dtype,
            chunks=(min(1_000_000, max(1, particle_shape[0])), 3),
        )

        density_offset = 0
        shat_offset = 0
        particle_offset = 0
        for path in rank_paths:
            with h5py.File(path, "r") as handle:
                n_density = handle["final_density"].shape[0]
                if n_density:
                    density[density_offset : density_offset + n_density] = handle["final_density"][...]
                    density_offset += n_density

                n_shat = handle["s_hat_field"].shape[0]
                if n_shat:
                    shat[shat_offset : shat_offset + n_shat] = handle["s_hat_field"][...]
                    shat_offset += n_shat

                n_particle = handle["u_pos"].shape[0]
                if n_particle:
                    pos[particle_offset : particle_offset + n_particle] = handle["u_pos"][...]
                    vel[particle_offset : particle_offset + n_particle] = handle["u_vel"][...]
                    particle_offset += n_particle

        out.attrs["source_mcmc"] = str(mcmc)
        out.attrs["iteration"] = iteration

    print(f"Packed {group_name} outputs into: {product}", flush=True)


def scalar_value(group, name: str) -> float:
    return float(group["scalars"][name][0])


def borg_velocity_multiplier(group) -> float:
    # BORG writes /u_vel in internal units. Its gridded velocity helpers convert
    # with ParticleBasedForwardModel::getVelocityMultiplier() = unit_v0 / a_final.
    unit_v0 = 100.0
    a_final = 1.0
    if "scalars" in group and "cosmo" in group["scalars"]:
        cosmo = group["scalars/cosmo"][0]
        if getattr(cosmo.dtype, "names", None) and "a0" in cosmo.dtype.names:
            a_final = float(cosmo["a0"])
    if a_final == 0.0:
        raise ValueError("Cannot convert BORG velocities with a_final=0")
    return unit_v0 / a_final


def dump_sph_particles(product: Path, group_name: str, particles: Path) -> float:
    import h5py
    import numpy as np

    particles.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(product, "r") as src, h5py.File(particles, "w") as out:
        group = src[group_name]
        pos = group["u_pos"]
        vel = group["u_vel"]
        velocity_multiplier = borg_velocity_multiplier(group)
        if pos.shape != vel.shape or pos.shape[1] != 3:
            raise ValueError(f"Expected matching (N, 3) u_pos/u_vel in /{group_name}")

        n_particles = int(pos.shape[0])
        dataset = out.create_dataset(
            "particles",
            (n_particles, 7),
            dtype="f4",
            chunks=(min(1_000_000, max(1, n_particles)), 7),
        )
        chunk = dataset.chunks[0]
        for start in range(0, n_particles, chunk):
            stop = min(start + chunk, n_particles)
            values = np.empty((stop - start, 7), dtype="f4")
            values[:, :3] = pos[start:stop]
            values[:, 3:6] = vel[start:stop] * velocity_multiplier
            values[:, 6] = 1.0
            dataset[start:stop] = values

        out.attrs["source_product"] = str(product)
        out.attrs["source_group"] = group_name
        out.attrs["velocity_multiplier_internal_to_kms"] = velocity_multiplier

    print(
        f"Wrote CosmoTool particle input: {particles} "
        f"(velocity multiplier={velocity_multiplier:g})",
        flush=True,
    )
    return velocity_multiplier


def write_sph_product(raw_sph: Path, product: Path, group_name: str, attrs: dict[str, float | int | str]) -> None:
    import h5py
    import numpy as np

    with h5py.File(raw_sph, "r") as src, h5py.File(product, "a") as out:
        group = out[group_name]
        if "sph" in group:
            del group["sph"]
        sph = group.create_group("sph")
        for key, value in attrs.items():
            sph.attrs[key] = value

        density = src["density"][...].astype("f4", copy=False)
        density_mean = float(np.mean(density))
        sph.attrs["density_mean_raw"] = density_mean
        sph.attrs["velocity_units"] = "km/s"
        if "velocity_multiplier_internal_to_kms" in attrs:
            sph.attrs["velocity_multiplier_internal_to_kms"] = attrs["velocity_multiplier_internal_to_kms"]

        overdensity = np.zeros_like(density, dtype="f4")
        if density_mean != 0.0:
            overdensity = density / density_mean - 1.0
        overdensity_dataset = sph.create_dataset(
            "overdensity",
            data=overdensity,
            chunks=(1,) + overdensity.shape[1:],
        )
        overdensity_dataset.attrs["quantity"] = "density contrast"
        overdensity_dataset.attrs["definition"] = "rho / mean(rho) - 1"

        velocity = sph.create_dataset(
            "velocity",
            density.shape + (3,),
            dtype="f4",
            chunks=(1,) + density.shape[1:] + (3,),
        )
        velocity.attrs["quantity"] = "mass-weighted peculiar velocity"
        velocity.attrs["units"] = "km/s"
        if "velocity_multiplier_internal_to_kms" in attrs:
            velocity.attrs["velocity_multiplier_internal_to_kms"] = attrs["velocity_multiplier_internal_to_kms"]
        nonzero = density != 0.0
        for axis in range(3):
            component = np.zeros_like(density, dtype="f4")
            momentum = src[f"p{axis}"][...].astype("f4", copy=False)
            component[nonzero] = momentum[nonzero] / density[nonzero]
            velocity[..., axis] = component

        sph.create_dataset("num_in_cell", data=src["num_in_cell"][...], chunks=(1,) + density.shape[1:])

    print(f"Wrote SPH fields to: {product}:/{group_name}/sph", flush=True)


def run_sph_fields(product: Path, group_name: str, args: argparse.Namespace) -> None:
    import h5py

    sph_binary = require_file(args.cosmotool_sph, "CosmoTool simple3DFilter")
    with h5py.File(product, "r") as handle:
        group = handle[group_name]
        resolution = args.sph_resolution or int(scalar_value(group, "N0"))
        boxsize = scalar_value(group, "L0")
        lengths = [scalar_value(group, key) for key in ("L0", "L1", "L2")]
        if any(abs(length - boxsize) > 1e-6 for length in lengths):
            raise ValueError(f"SPH gridding assumes a cubic box; got L0,L1,L2={lengths}")

    radius_limit = args.sph_radius_limit if args.sph_radius_limit is not None else boxsize
    product_work_dir = product.parent / "work" / product.stem
    work_dir = product_work_dir / group_name / "sph"
    particles = work_dir / "particles_for_cosmotool.h5"
    raw_sph = work_dir / "cosmotool_sph_fields.h5"

    velocity_multiplier = dump_sph_particles(product, group_name, particles)
    cmd = [
        str(sph_binary),
        str(particles),
        f"{radius_limit:.17g}",
        f"{boxsize:.17g}",
        str(resolution),
        f"{boxsize / 2.0:.17g}",
        f"{boxsize / 2.0:.17g}",
        f"{boxsize / 2.0:.17g}",
        str(raw_sph),
        str(args.sph_periodic),
    ]
    print("Running CosmoTool SPH:", flush=True)
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.sph_threads)
    subprocess.run(cmd, check=True, env=env)

    write_sph_product(
        raw_sph,
        product,
        group_name,
        {
            "source_particles": str(particles),
            "source_cosmotool_output": str(raw_sph),
            "cosmotool_binary": str(sph_binary),
            "resolution": resolution,
            "boxsize": boxsize,
            "radius_limit": radius_limit,
            "periodic": args.sph_periodic,
            "sph_threads": args.sph_threads,
            "mass_column_value": 1,
            "velocity_multiplier_internal_to_kms": velocity_multiplier,
        },
    )
    if not args.keep_sph_work:
        shutil.rmtree(product_work_dir, ignore_errors=True)
        print(f"Removed SPH work directory: {product_work_dir}", flush=True)


def run_cic_fields(product: Path, group_name: str, args: argparse.Namespace) -> None:
    import h5py
    import numpy as np

    with h5py.File(product, "r") as handle:
        group = handle[group_name]
        resolution = args.sph_resolution or int(scalar_value(group, "N0"))
        boxsize = scalar_value(group, "L0")
        lengths = [scalar_value(group, key) for key in ("L0", "L1", "L2")]
        if any(abs(length - boxsize) > 1e-6 for length in lengths):
            raise ValueError(f"CIC gridding assumes a cubic box; got L0,L1,L2={lengths}")
        n_particles = group["u_pos"].shape[0]
        velocity_multiplier = borg_velocity_multiplier(group)

    print(
        f"Running periodic CIC gridding: resolution={resolution}, "
        f"particles={n_particles}, chunk_size={args.cic_chunk_size}, "
        f"velocity_multiplier={velocity_multiplier:g}",
        flush=True,
    )

    density = np.zeros((resolution, resolution, resolution), dtype=np.float64)
    momentum = np.zeros((3, resolution, resolution, resolution), dtype=np.float64)
    density_flat = density.ravel()
    momentum_flat = momentum.reshape(3, -1)
    inv_cell = resolution / boxsize
    chunk_size = max(1, int(args.cic_chunk_size))

    with h5py.File(product, "r") as handle:
        group = handle[group_name]
        pos = group["u_pos"]
        vel = group["u_vel"]
        for start in range(0, n_particles, chunk_size):
            stop = min(start + chunk_size, n_particles)
            coords = (pos[start:stop] + 0.5 * boxsize) * inv_cell - 0.5
            base = np.floor(coords).astype(np.int64)
            frac = coords - base
            velocity = vel[start:stop] * velocity_multiplier

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
                            np.add.at(momentum_flat[axis], index, weight * velocity[:, axis])

            print(f"CIC deposited particles {stop} / {n_particles}", flush=True)

    velocity_field = np.zeros(density.shape + (3,), dtype=np.float32)
    nonzero = density > 0.0
    for axis in range(3):
        component = np.zeros_like(density, dtype=np.float64)
        component[nonzero] = momentum[axis][nonzero] / density[nonzero]
        velocity_field[..., axis] = component.astype(np.float32)

    density_mean = float(np.mean(density))
    overdensity = np.zeros_like(density, dtype=np.float32)
    if density_mean != 0.0:
        overdensity = (density / density_mean - 1.0).astype(np.float32)

    with h5py.File(product, "a") as handle:
        group = handle[group_name]
        if "cic" in group:
            del group["cic"]
        cic = group.create_group("cic")
        cic.attrs["resolution"] = resolution
        cic.attrs["boxsize"] = boxsize
        cic.attrs["periodic"] = 1
        cic.attrs["chunk_size"] = chunk_size
        cic.attrs["density_mean_raw"] = density_mean
        cic.attrs["mass_column_value"] = 1
        cic.attrs["velocity_units"] = "km/s"
        cic.attrs["velocity_multiplier_internal_to_kms"] = velocity_multiplier
        overdensity_dataset = cic.create_dataset(
            "overdensity",
            data=overdensity,
            chunks=(1, resolution, resolution),
        )
        overdensity_dataset.attrs["quantity"] = "density contrast"
        overdensity_dataset.attrs["definition"] = "rho / mean(rho) - 1"
        velocity_dataset = cic.create_dataset(
            "velocity",
            data=velocity_field,
            chunks=(1, resolution, resolution, 3),
        )
        velocity_dataset.attrs["quantity"] = "mass-weighted peculiar velocity"
        velocity_dataset.attrs["units"] = "km/s"

    print(f"Wrote CIC fields to: {product}:/{group_name}/cic", flush=True)


def remove_particle_datasets(product: Path, group_name: str) -> None:
    import h5py

    removed = []
    with h5py.File(product, "a") as handle:
        group = handle[group_name]
        for dataset in ("u_pos", "u_vel"):
            if dataset in group:
                del group[dataset]
                removed.append(f"/{group_name}/{dataset}")
    if removed:
        print(f"Removed particle datasets from final product: {', '.join(removed)}", flush=True)


def slim_field_product(product: Path, group_name: str, mas: str) -> None:
    import h5py
    import numpy as np

    tmp_product = product.with_suffix(product.suffix + ".slim")
    tmp_product.unlink(missing_ok=True)
    with h5py.File(product, "r") as src, h5py.File(tmp_product, "w") as out:
        fields = src[f"{group_name}/{mas}"]
        overdensity = fields["overdensity"]
        velocity = fields["velocity"]
        if overdensity.dtype != np.dtype("float32"):
            raise TypeError(f"Expected /{group_name}/{mas}/overdensity to be float32, got {overdensity.dtype}")
        if velocity.dtype != np.dtype("float32"):
            raise TypeError(f"Expected /{group_name}/{mas}/velocity to be float32, got {velocity.dtype}")

        for key, value in src.attrs.items():
            out.attrs[key] = value
        out.attrs["field_source_group"] = group_name
        out.attrs["mass_assignment"] = mas
        src.copy(overdensity, out, name="overdensity")
        src.copy(velocity, out, name="velocity")

    tmp_product.replace(product)
    print(f"Slimmed final product to: {product}:/overdensity and /velocity", flush=True)


def run_gridded_fields(product: Path, group_name: str, args: argparse.Namespace) -> None:
    if args.mas == "sph":
        run_sph_fields(product, group_name, args)
    elif args.mas == "cic":
        run_cic_fields(product, group_name, args)
    else:
        raise ValueError(f"Unknown mass-assignment scheme: {args.mas}")


def validate_split_outputs(output_pattern: Path, iteration: int, nprocs: int) -> None:
    import h5py

    output = output_path(output_pattern, iteration)
    density_slabs = 0
    particle_count = 0
    velocity_count = 0
    expected_n0 = None
    for rank in range(nprocs):
        rank_output = Path(f"{output}_{rank}")
        if not rank_output.is_file():
            raise FileNotFoundError(f"Missing split output: {rank_output}")
        with h5py.File(rank_output, "r") as handle:
            for dataset in ("final_density", "s_hat_field", "u_pos", "u_vel"):
                if dataset not in handle:
                    raise KeyError(f"Missing /{dataset} in {rank_output}")
            density_slabs += handle["final_density"].shape[0]
            particle_count += handle["u_pos"].shape[0]
            velocity_count += handle["u_vel"].shape[0]
            rank_n0 = int(handle["scalars/N0"][0])
            expected_n0 = rank_n0 if expected_n0 is None else expected_n0
            if rank_n0 != expected_n0:
                raise ValueError(f"Inconsistent N0 in {rank_output}: {rank_n0} != {expected_n0}")

    if density_slabs != expected_n0:
        raise ValueError(f"Split density slabs sum to {density_slabs}, expected {expected_n0}")
    if particle_count != velocity_count:
        raise ValueError(f"u_pos count {particle_count} != u_vel count {velocity_count}")
    print(
        "Validated split outputs: "
        f"{nprocs} files, {density_slabs} density slabs, {particle_count} particles.",
        flush=True,
    )


def compare_to_mcmc(mcmc: Path, output_pattern: Path, iteration: int, nprocs: int, atol: float, rtol: float) -> dict[str, float]:
    import h5py
    import numpy as np

    output = output_path(output_pattern, iteration)
    max_abs = 0.0
    max_ref_abs = 0.0
    sum_sq = 0.0
    count = 0
    offset = 0

    with h5py.File(mcmc, "r") as ref_handle:
        ref = ref_handle["/scalars/BORG_final_density"]
        for rank in range(nprocs):
            with h5py.File(Path(f"{output}_{rank}"), "r") as out_handle:
                slab = out_handle["final_density"][...]
            n0 = slab.shape[0]
            if n0 == 0:
                continue
            ref_slab = ref[offset : offset + n0, :, :]
            diff = slab - ref_slab
            max_abs = max(max_abs, float(np.max(np.abs(diff))))
            max_ref_abs = max(max_ref_abs, float(np.max(np.abs(ref_slab))))
            sum_sq += float(np.sum(diff * diff))
            count += int(diff.size)
            offset += n0

        if offset != ref.shape[0]:
            raise ValueError(f"Compared {offset} slabs along axis 0, expected {ref.shape[0]}")

    rms = float(np.sqrt(sum_sq / count)) if count else 0.0
    threshold = atol + rtol * max_ref_abs
    print(f"RSD reference comparison: max_abs_diff={max_abs:.16e}, rms_diff={rms:.16e}", flush=True)
    print(f"RSD reference tolerance: {threshold:.16e} (atol={atol}, rtol={rtol})", flush=True)
    passed = max_abs <= threshold
    if passed:
        print("RSD reference comparison: PASS", flush=True)
    else:
        print(
            "[WARN] RSD reference comparison: FAIL "
            "generated density does not match /scalars/BORG_final_density",
            flush=True,
        )
    return {
        "max_abs": max_abs,
        "rms": rms,
        "threshold": threshold,
        "passed": passed,
    }


def write_rsd_comparison_plot(
    mcmc: Path,
    output_pattern: Path,
    iteration: int,
    nprocs: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> Path | None:
    if not args.rsd_comparison_plot:
        return None

    plot_python = require_file_preserve_symlink(args.plot_python, "plot Python")
    plot_script = require_file(args.rsd_comparison_plot_script, "RSD comparison plot script")
    plot_dir = (
        args.rsd_comparison_plot_dir.expanduser().resolve()
        if args.rsd_comparison_plot_dir is not None
        else out_dir / "plots"
    )
    index_label = str(args.rsd_comparison_plot_index) if args.rsd_comparison_plot_index is not None else "mid"
    plot_path = plot_dir / f"rsd_comparison_mcmc_{iteration}_axis{args.rsd_comparison_plot_axis}_slice{index_label}.png"
    cmd = [
        str(plot_python),
        str(plot_script),
        "--mcmc",
        str(mcmc),
        "--output-pattern",
        str(output_pattern),
        "--iteration",
        str(iteration),
        "--nprocs",
        str(nprocs),
        "--axis",
        str(args.rsd_comparison_plot_axis),
        "--output",
        str(plot_path),
    ]
    if args.rsd_comparison_plot_index is not None:
        cmd.extend(["--index", str(args.rsd_comparison_plot_index)])

    print("Writing RSD comparison plot:", flush=True)
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] RSD comparison plot failed with exit code {exc.returncode}; continuing.", flush=True)
        return None
    return plot_path


def write_rsd_cross_correlation(
    mcmc: Path,
    output_pattern: Path,
    iteration: int,
    nprocs: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    if not args.rsd_cross_correlation:
        return None

    plot_python = require_file_preserve_symlink(args.plot_python, "plot Python")
    cross_script = require_file(args.rsd_cross_script, "RSD Pylians cross-correlation script")
    output_dir = (
        args.rsd_cross_output_dir.expanduser().resolve()
        if args.rsd_cross_output_dir is not None
        else out_dir / "plots"
    )
    metrics_json = output_dir / f"rsd_cross_correlation_mcmc_{iteration}.json"
    cmd = [
        str(plot_python),
        str(cross_script),
        "--mcmc",
        str(mcmc),
        "--output-pattern",
        str(output_pattern),
        "--iteration",
        str(iteration),
        "--nprocs",
        str(nprocs),
        "--boxsize",
        str(args.rsd_cross_boxsize),
        "--axis",
        str(args.rsd_cross_axis),
        "--threads",
        str(args.rsd_cross_threads),
        "--kmax",
        str(args.rsd_cross_kmax),
        "--min-mean-r",
        str(args.rsd_cross_min_mean_r),
        "--output-dir",
        str(output_dir),
        "--metrics-json",
        str(metrics_json),
    ]

    print("Writing RSD Pylians cross-correlation:", flush=True)
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] RSD Pylians cross-correlation failed with exit code {exc.returncode}; continuing.", flush=True)
        return None

    try:
        metrics = json.loads(metrics_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Could not read RSD cross-correlation metrics from {metrics_json}: {exc}", flush=True)
        return None

    if metrics.get("passed"):
        print("RSD Pylians cross-correlation: PASS", flush=True)
    else:
        print("[WARN] RSD Pylians cross-correlation: FAIL; continuing.", flush=True)
    return metrics


def choose_mcmc_files(chain_dir: Path, glob: str, samples: int, seed: int) -> list[Path]:
    candidates = sorted(require_dir(chain_dir, "chain directory").glob(glob))
    if not candidates:
        raise FileNotFoundError(f"No MCMC files matched {chain_dir / glob}")
    if samples < 1:
        raise ValueError("--samples must be >= 1")
    rng = random.Random(seed)
    return rng.sample(candidates, min(samples, len(candidates)))


def write_product_plots(product: Path, groups: list[str], args: argparse.Namespace) -> list[Path]:
    plot_python = require_file_preserve_symlink(args.plot_python, "plot Python")
    plot_script = require_file(args.plot_script, "plot script")
    output_dir = product.parent / "plots"
    cmd = [
        str(plot_python),
        str(plot_script),
        str(product),
        "--output-dir",
        str(output_dir),
        "--filename-prefix",
        f"{product.stem}_",
    ]
    if groups:
        cmd.extend(["--groups", *groups])
    if args.plot_slice_index is not None:
        cmd.extend(["--slice-index", str(args.plot_slice_index)])
    expected = [output_dir / f"{product.stem}_{group}_{args.mas}_field_slices.png" for group in groups]

    print("Writing field slice plots:", flush=True)
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(cmd, check=True, env=env)
    return expected


def check_plot_environment(args: argparse.Namespace) -> None:
    plot_python = require_file_preserve_symlink(args.plot_python, "plot Python")
    plot_script = require_file(args.plot_script, "plot script")
    cmd = [
        str(plot_python),
        "-c",
        (
            "import h5py, matplotlib, numpy; "
            "print('Plot Python OK:', matplotlib.__version__, numpy.__version__, h5py.__version__)"
        ),
    ]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    print("Checking plot Python before expensive field generation:", flush=True)
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    subprocess.run([str(plot_python), "-m", "py_compile", str(plot_script)], check=True, env=env)


def print_run_summary(
    mcmc: Path,
    product: Path | None,
    modes: list[str],
    split_dirs: list[Path],
    plot_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    print("", flush=True)
    print("Run summary:", flush=True)
    print(f"  Source MCMC: {mcmc}", flush=True)
    print(f"  Modes: {', '.join(modes)}", flush=True)
    print(f"  Split outputs: {', '.join(str(path) for path in split_dirs)}", flush=True)
    if product is not None:
        print(f"  Single HDF5 product: {product}", flush=True)
        if args.sph_fields:
            print("  Gridded fields: /overdensity and /velocity", flush=True)
            print("  Final product contains only field datasets: yes", flush=True)
        else:
            print("  Gridded fields: not written", flush=True)
        if plot_paths:
            print(f"  Plots: {', '.join(str(path) for path in plot_paths)}", flush=True)
        elif args.plots:
            print(f"  Plots: requested under {product.parent / 'plots'}", flush=True)
        else:
            print("  Plots: skipped", flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "run":
        mcmc = require_file(args.mcmc, "MCMC file")
        product = single_output_path(mcmc, args)
        if args.plots and not args.no_single_output and not args.dry_run:
            check_plot_environment(args)
        output_pattern, iteration, out_dir = run_forward(mcmc, args, do_rsd=args.rsd)
        if args.dry_run:
            if args.include_rsd and not args.rsd:
                run_forward(mcmc, args, do_rsd=True)
            if not args.no_single_output:
                print(f"Single HDF5 product: {product}", flush=True)
                if args.plots:
                    print(f"Plot directory: {product.parent / 'plots'}", flush=True)
                if args.sph_fields:
                    print(f"{args.mas.upper()} fields would be written to: {product}:/overdensity and /velocity", flush=True)
                    if args.mas == "sph":
                        print(f"SPH OpenMP threads: {args.sph_threads}", flush=True)
                    print("Final product will contain only these two field datasets.", flush=True)
            return
        if not args.dry_run and not args.no_single_output:
            group_name = "rsd" if args.rsd else "realspace"
            plotted_groups = [group_name]
            split_dirs = [out_dir]
            pack_split_outputs(
                output_pattern,
                iteration,
                args.nprocs,
                product,
                group_name,
                mcmc,
            )
            if args.sph_fields:
                run_gridded_fields(product, group_name, args)
                remove_particle_datasets(product, group_name)
            if args.include_rsd and not args.rsd:
                rsd_pattern, rsd_iteration, rsd_out_dir = run_forward(mcmc, args, do_rsd=True)
                split_dirs.append(rsd_out_dir)
                pack_split_outputs(rsd_pattern, rsd_iteration, args.nprocs, product, "rsd", mcmc)
                if args.sph_fields:
                    run_gridded_fields(product, "rsd", args)
                    remove_particle_datasets(product, "rsd")
                plotted_groups.append("rsd")
            plot_paths: list[Path] = []
            if args.plots:
                plot_paths = write_product_plots(product, plotted_groups, args)
            if args.sph_fields:
                if len(plotted_groups) != 1:
                    raise ValueError("Slim final products support exactly one gridded mode.")
                slim_field_product(product, plotted_groups[0], args.mas)
            print(f"Single HDF5 product: {product}", flush=True)
            print_run_summary(mcmc, product, plotted_groups, split_dirs, plot_paths, args)
        elif not args.dry_run:
            mode = "rsd" if args.rsd else "realspace"
            print_run_summary(mcmc, None, [mode], [out_dir], [], args)
        return

    mcmcs = choose_mcmc_files(args.chain_dir, args.glob, args.samples, args.seed)
    print("RSD validation samples:", flush=True)
    for mcmc in mcmcs:
        print(f"  {mcmc.resolve()}", flush=True)
    for mcmc in mcmcs:
        output_pattern, iteration, out_dir = run_forward(mcmc, args, do_rsd=True)
        if not args.dry_run:
            metrics = compare_to_mcmc(mcmc.resolve(), output_pattern, iteration, args.nprocs, args.atol, args.rtol)
            plot_path = write_rsd_comparison_plot(
                mcmc.resolve(),
                output_pattern,
                iteration,
                args.nprocs,
                out_dir,
                args,
            )
            cross_metrics = write_rsd_cross_correlation(
                mcmc.resolve(),
                output_pattern,
                iteration,
                args.nprocs,
                out_dir,
                args,
            )
            print("", flush=True)
            print("Validation summary:", flush=True)
            print(f"  Source MCMC: {mcmc.resolve()}", flush=True)
            print(f"  RSD split outputs: {output_path(output_pattern, iteration)}_<rank>", flush=True)
            if plot_path is not None:
                print(f"  RSD comparison plot: {plot_path}", flush=True)
            if cross_metrics is not None:
                cross_status = "PASS" if cross_metrics["passed"] else "FAIL"
                print(
                    "  RSD Pylians cross-correlation: "
                    f"{cross_status} mean_r(k<={cross_metrics['kmax']:.3g})="
                    f"{cross_metrics['mean_r_constrained']:.6e}, "
                    f"threshold={cross_metrics['min_mean_r']:.3g}",
                    flush=True,
                )
                print(f"  RSD cross-correlation plot: {cross_metrics['plot']}", flush=True)
                print(f"  RSD cross-correlation CSV: {cross_metrics['csv']}", flush=True)
            status = "PASS" if metrics["passed"] else "FAIL"
            print(
                f"  Comparison: {status} max_abs={metrics['max_abs']:.6e}, "
                f"rms={metrics['rms']:.6e}, threshold={metrics['threshold']:.6e}",
                flush=True,
            )
            if not metrics["passed"]:
                print("  Validation mismatch is non-fatal; continuing.", flush=True)
            if cross_metrics is not None and not cross_metrics["passed"]:
                print("  Cross-correlation mismatch is non-fatal; continuing.", flush=True)


if __name__ == "__main__":
    main()
