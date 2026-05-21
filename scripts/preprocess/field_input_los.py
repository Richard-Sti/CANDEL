# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Helpers for preparing reconstruction LOS density and velocity products."""
from os import getpid, makedirs, remove, replace
from os.path import basename, dirname, exists, join, splitext

import numpy as np
from h5py import File

import candel
from candel import fprint
from candel.pvdata.field_products import (field_smoothed_los_path,
                                          los_radial_grid_payload,
                                          reconstruction_los_label,
                                          validate_field_smoothing_scale)


def generate_random_sky(npoints, seed):
    gen = np.random.default_rng(seed)
    RA = gen.uniform(0, 360, size=npoints)
    dec = np.arcsin(gen.uniform(-1, 1, size=npoints)) * 180 / np.pi
    return RA, dec


def pv_main_los_config(config, catalogue):
    """Return the LOS template and raw catalogue-loader kwargs."""
    d = config["io"]["PV_main"][catalogue].copy()
    los_file = d.pop("los_file")
    d.pop("reconstruction", None)
    d.pop("return_all", None)
    return los_file, d


def load_los(catalogue, config, filepath=None, config_path=None):
    if "random_" in catalogue:
        d = config["io"].copy()
        los_file = d.pop("los_file_random")

        npoints = int(catalogue.replace("random_", ''))
        # This could, in principle, account for the ZoA mask.
        RA, dec = generate_random_sky(npoints, seed=42)
    elif catalogue == "CF4":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_CF4_data(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "2MTF":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_2MTF(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "SFI":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_SFI(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "PantheonPlus":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_PantheonPlus(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "PantheonPlusLane":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_PantheonPlus_Lane(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "Foundation":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_Foundation(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "LOSS":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_LOSS(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "SDSS_FP":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_SDSS_FP(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "6dF_FP":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_6dF_FP(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "EDD_TRGB":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_EDD_TRGB(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "EDD_TRGB_grouped":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_EDD_TRGB_grouped(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "SH0ES":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_SH0ES_separated(**d)
        RA, dec = data["RA_host"], data["dec_host"]
    elif catalogue == "CCHP":
        if config_path is None:
            raise ValueError("`config_path` must be provided for CCHP.")
        d = config["io"].get("CCHP", {}).copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_CCHP_from_config(
            config_path, ra_dec_only=True)
        RA, dec = data["RA"], data["DEC"]
    elif catalogue == "generic":
        if filepath is None:
            raise ValueError(
                "`filepath` must be provided for generic catalogue.")
        data = candel.pvdata.load_generic(filepath)
        RA, dec = data["RA"], data["dec"]

        # Construct LOS file path in the same directory as input
        input_dir = dirname(filepath)
        name, __ = splitext(basename(filepath))
        los_file = join(input_dir, f"{name}_LOS_<X>.hdf5")
    else:
        raise ValueError(f"Catalogue {catalogue} not supported. Please add "
                         "support for it.")

    return RA, dec, los_file


def reconstruction_field_indices(config, reconstruction):
    """Return the reconstruction realisation indices to interpolate."""
    nreal_map = {
        "Carrick2015": 1,
        "Lilow2024": 1,
        "CLONES": 1,
        "CB1": 100,
        "CB2": 20,
        "CF4": 100,
        "HAMLET_V0": 20,
        "HAMLET_V1": 20,
        }

    if reconstruction in nreal_map:
        return list(range(nreal_map[reconstruction]))
    if str(reconstruction).lower().startswith("manticorelocal"):
        candel.field.name2field_loader(reconstruction)
        field_kwargs = config["io"]["reconstruction_main"][reconstruction]
        fpath_root = field_kwargs["fpath_root"]
        if reconstruction == "ManticoreLocalCOLA":
            fpath_root = join(
                fpath_root,
                candel.field.field_mas_directory(
                    field_kwargs.get("which_MAS", "CIC")))
        return candel.field.available_mcmc_field_indices(
            fpath_root)
    raise ValueError(f"Reconstruction `{reconstruction}` not supported.")


def radial_los_grid(config, catalogue, reconstruction, verbose=True):
    """Return the radial grid for a catalogue/reconstruction LOS product."""
    radial_grid = los_radial_grid_payload(config, catalogue, reconstruction)
    if radial_grid is None:
        raise ValueError(
            f"No configured LOS radial grid for `{catalogue}`/"
            f"`{reconstruction}`.")
    rmin = radial_grid["rmin"]
    rmax = radial_grid["rmax"]
    num_steps = radial_grid["num_steps"]
    if "random_" in catalogue:
        dr = (rmax - rmin) / (num_steps - 1) if num_steps > 1 else 0.0
        fprint(f"random LOS grid: {rmin} to {rmax} Mpc/h, "
               f"dr={dr}, {num_steps} steps.", verbose=verbose)
        return np.linspace(rmin, rmax, num_steps)

    fprint(f"setting the radial grid from {rmin} to {rmax} "
           f"with {num_steps} steps.", verbose=verbose)
    return np.linspace(rmin, rmax, num_steps)


def resolve_los_output_path(los_file, reconstruction,
                            field_smoothing_scale=None, config=None):
    """Resolve a LOS template to the output path for this smoothing variant."""
    los_file = los_file.replace(
        "<X>", reconstruction_los_label(config, reconstruction))
    return field_smoothed_los_path(los_file, field_smoothing_scale)


def los_file_matches_grid(los_file, r, verbose=True):
    """Return whether an existing LOS file stores the requested radial grid."""
    if not exists(los_file):
        return False
    try:
        with File(los_file, "r") as f:
            if "r" not in f:
                fprint(f"LOS cache `{los_file}` has no `r`; rebuilding.",
                       verbose=verbose)
                return False
            stored = f["r"][...]
    except Exception as exc:
        fprint(f"LOS cache `{los_file}` could not be read ({exc}); "
               "rebuilding.", verbose=verbose)
        return False
    requested = np.asarray(r)
    if stored.shape != requested.shape or not np.allclose(stored, requested):
        fprint(f"LOS cache `{los_file}` has a stale radial grid; "
               "rebuilding.", verbose=verbose)
        return False
    return True


def _selected_field_indices(config, reconstruction, field_indices=None):
    """Return requested reconstruction field indices for one LOS product."""
    available = np.asarray(
        reconstruction_field_indices(config, reconstruction), dtype=np.int32)
    if field_indices is None:
        field_indices = config.get("io", {}).get("field_indices", None)
    if field_indices is None:
        return available.tolist()

    requested = np.asarray(field_indices, dtype=np.int32)
    if requested.ndim == 0:
        requested = requested[None]
    if requested.ndim != 1 or len(requested) == 0:
        raise ValueError(
            "`io.field_indices` must be an int or non-empty 1D list.")
    missing = [int(nsim) for nsim in requested if nsim not in available]
    if missing:
        raise ValueError(
            f"Requested field indices {missing} are not available for "
            f"`{reconstruction}`. Available: {available.tolist()}.")
    return requested.tolist()


def compute_los_file_from_coordinates(
        catalogue, reconstruction, config, RA, dec, los_template=None,
        filepath=None, field_smoothing_scale=None, overwrite=False,
        output_path=None, field_indices=None, r=None, verbose=True,
        metadata=None):
    """Compute one LOS product from already-loaded sky coordinates."""
    field_smoothing_scale = validate_field_smoothing_scale(
        field_smoothing_scale)

    nsims = _selected_field_indices(config, reconstruction, field_indices)
    if r is None:
        r = radial_los_grid(
            config, catalogue, reconstruction, verbose=verbose)
    RA = np.asarray(RA)
    dec = np.asarray(dec)
    if len(RA) != len(dec):
        raise ValueError(
            f"`RA` and `dec` must have the same length, got "
            f"{len(RA)} and {len(dec)}.")

    fprint("LOS build:", verbose=verbose)
    fprint(f"  catalogue: `{catalogue}`", verbose=verbose)
    fprint(f"  reconstruction: `{reconstruction}`", verbose=verbose)
    fprint(f"  catalogue objects: {len(RA)}", verbose=verbose)

    los_file = (
        output_path if output_path is not None
        else resolve_los_output_path(
            los_template, reconstruction, field_smoothing_scale,
            config=config))
    if los_file is None:
        raise ValueError(
            "A LOS output path is required. Provide either `output_path` "
            "or `los_template`.")
    if exists(los_file) and not overwrite:
        if los_file_matches_grid(los_file, r, verbose=verbose):
            fprint(f"  status: cached at `{los_file}`; skipping.",
                   verbose=verbose)
            return {"path": los_file, "status": "exists"}
        fprint(f"  status: stale file at `{los_file}`; overwriting.",
               verbose=verbose)

    out_dir = dirname(los_file)
    if out_dir:
        makedirs(out_dir, exist_ok=True)

    n_sims = len(nsims)
    n_gal = len(RA)
    n_r = len(r)
    is_random = "random_" in catalogue
    is_random_multireal = is_random and n_sims > 1

    loader_cls = candel.field.name2field_loader(reconstruction)
    loader_kwargs = config["io"]["reconstruction_main"][reconstruction]
    fixed_los_geometry = None
    if not is_random:
        geometry_loader = loader_cls(nsim=nsims[0], **loader_kwargs)
        fixed_los_geometry = candel.field.prepare_los_geometry(
            geometry_loader, r, RA, dec)

    fprint(f"  output: `{los_file}`", verbose=verbose)
    los_tmp_file = f"{los_file}.tmp.{getpid()}"
    dt = np.dtype(np.float32)
    dt16 = np.dtype(np.float16)
    try:
        with File(los_tmp_file, "w") as fout:
            if metadata is not None:
                for key, value in metadata.items():
                    fout.attrs[key] = value
            if is_random_multireal:
                ra_dataset = fout.create_dataset(
                    "RA", shape=(n_sims, n_gal), dtype=dt)
                dec_dataset = fout.create_dataset(
                    "dec", shape=(n_sims, n_gal), dtype=dt)
            elif is_random:
                ra_dataset = fout.create_dataset(
                    "RA", shape=(n_gal,), dtype=dt)
                dec_dataset = fout.create_dataset(
                    "dec", shape=(n_gal,), dtype=dt)
            else:
                fout.create_dataset("RA", data=RA, dtype=dt)
                fout.create_dataset("dec", data=dec, dtype=dt)
            fout.create_dataset("r", data=r, dtype=dt)
            density_dataset = fout.create_dataset(
                "los_density", shape=(n_sims, n_gal, n_r), dtype=dt,
                fillvalue=np.nan)
            velocity_dataset = fout.create_dataset(
                "los_velocity", shape=(n_sims, n_gal, n_r), dtype=dt16,
                fillvalue=np.nan)
            fout.create_dataset(
                "field_indices", data=np.asarray(nsims, dtype=np.int32))

            for i, nsim in enumerate(nsims):
                fprint(f"  field {i + 1}/{n_sims}: nsim={int(nsim)}",
                       verbose=verbose)
                loader = loader_cls(nsim=nsim, **loader_kwargs)
                if is_random:
                    RA_i, dec_i = generate_random_sky(n_gal, seed=42 + nsim)
                    los_geometry = None
                else:
                    RA_i, dec_i = RA, dec
                    los_geometry = fixed_los_geometry
                dens_i, vel_i = candel.field.interpolate_los_density_velocity(
                    loader, r, RA_i, dec_i,
                    field_smoothing_scale=field_smoothing_scale,
                    verbose=verbose, los_geometry=los_geometry)
                density_dataset[i] = dens_i.astype(np.float32)
                velocity_dataset[i] = vel_i.astype(np.float16)
                if is_random_multireal:
                    ra_dataset[i] = RA_i.astype(np.float32)
                    dec_dataset[i] = dec_i.astype(np.float32)
                elif is_random:
                    ra_dataset[:] = RA_i.astype(np.float32)
                    dec_dataset[:] = dec_i.astype(np.float32)
        replace(los_tmp_file, los_file)
    finally:
        if exists(los_tmp_file):
            remove(los_tmp_file)
    return {"path": los_file, "status": "computed"}


def compute_los_file(catalogue, reconstruction, config, filepath=None,
                     config_path=None, field_smoothing_scale=None,
                     overwrite=False, output_path=None, field_indices=None,
                     verbose=True):
    """Compute one LOS product serially using the standard file schema."""
    RA, dec, los_template = load_los(
        catalogue, config, filepath, config_path=config_path)
    return compute_los_file_from_coordinates(
        catalogue, reconstruction, config, RA, dec,
        los_template=los_template, filepath=filepath,
        field_smoothing_scale=field_smoothing_scale,
        overwrite=overwrite, output_path=output_path,
        field_indices=field_indices, verbose=verbose)
