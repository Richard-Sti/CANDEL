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
from os import makedirs
from os import replace
from os.path import basename, dirname, exists, join, splitext

import numpy as np
from h5py import File

import candel
from candel import fprint
from candel.pvdata.field_products import (
    density_smoothed_los_path,
    validate_density_smoothing_scale,
)


def generate_random_sky(npoints, seed):
    gen = np.random.default_rng(seed)
    RA = gen.uniform(0, 360, size=npoints)
    dec = np.arcsin(gen.uniform(-1, 1, size=npoints)) * 180 / np.pi
    return RA, dec


def pv_main_los_config(config, catalogue):
    """Return the LOS template and raw catalogue-loader kwargs."""
    d = config["io"]["PV_main"][catalogue].copy()
    los_file = d.pop("los_file")
    d.pop("which_host_los", None)
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
        data = candel.pvdata.load_CF4_data(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "2MTF":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_2MTF(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "SFI":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_SFI(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "PantheonPlus":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_PantheonPlus(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "PantheonPlusLane":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_PantheonPlus_Lane(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "Foundation":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_Foundation(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "LOSS":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_LOSS(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "SDSS_FP":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_SDSS_FP(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "6dF_FP":
        los_file, d = pv_main_los_config(config, catalogue)
        data = candel.pvdata.load_6dF_FP(**d)
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
    if (reconstruction == candel.field.COLA_MANTICORE_NAME
            or reconstruction.lower().startswith("manticore")):
        field_kwargs = config["io"]["reconstruction_main"][reconstruction]
        return candel.field.available_mcmc_field_indices(
            field_kwargs["fpath_root"])
    raise ValueError(f"Reconstruction `{reconstruction}` not supported.")


def radial_los_grid(config, catalogue, reconstruction, verbose=True):
    """Return the radial grid for a catalogue/reconstruction LOS product."""
    if "random_" in catalogue:
        rand_cfg = config["io"].get("reconstruction_rand_los", {})
        rmin = rand_cfg.get("rmin", 0.1)
        rmax = rand_cfg.get("rmax", 251)
        dr = rand_cfg.get("dr", 1.0)
        recon_cfg = rand_cfg.get(reconstruction, {})
        rmin = recon_cfg.get("rmin", rmin)
        rmax = recon_cfg.get("rmax", rmax)
        dr = recon_cfg.get("dr", dr)
        num_steps = round((rmax - rmin) / dr) + 1
        fprint(f"random LOS grid: {rmin} to {rmax} Mpc/h, "
               f"dr={dr}, {num_steps} steps.", verbose=verbose)
        return np.linspace(rmin, rmax, num_steps)

    d = config["io"]["reconstruction_main"]
    fprint(f"setting the radial grid from {d['rmin']} to {d['rmax']} "
           f"with {d['num_steps']} steps.", verbose=verbose)
    return np.linspace(d["rmin"], d["rmax"], d["num_steps"])


def resolve_los_output_path(los_file, reconstruction, smooth_target=None,
                            density_smoothing_scale=None):
    """Resolve a LOS template to the output path for this smoothing variant."""
    if smooth_target is not None and density_smoothing_scale is not None:
        raise ValueError(
            "`smooth_target` and `density_smoothing_scale` are mutually "
            "exclusive.")
    los_file = los_file.replace("<X>", reconstruction)
    if smooth_target is not None:
        los_file = los_file.replace(
            ".hdf5", f"_smooth_to_{smooth_target}.hdf5")
    return density_smoothed_los_path(los_file, density_smoothing_scale)


def compute_los_file(catalogue, reconstruction, config, filepath=None,
                     config_path=None, smooth_target=None,
                     density_smoothing_scale=None, overwrite=False,
                     verbose=True):
    """Compute one LOS product serially using the standard file schema."""
    density_smoothing_scale = validate_density_smoothing_scale(
        density_smoothing_scale)
    if smooth_target == 0:
        smooth_target = None
    if smooth_target is not None and density_smoothing_scale is not None:
        raise ValueError(
            "`smooth_target` and `density_smoothing_scale` are mutually "
            "exclusive.")

    nsims = reconstruction_field_indices(config, reconstruction)
    r = radial_los_grid(config, catalogue, reconstruction, verbose=verbose)
    fprint(f"loading the catalogue `{catalogue}` with "
           f"reconstruction `{reconstruction}`.", verbose=verbose)
    RA, dec, los_template = load_los(
        catalogue, config, filepath, config_path=config_path)
    fprint(f"loaded {len(RA)} galaxies from the catalogue.", verbose=verbose)

    los_file = resolve_los_output_path(
        los_template, reconstruction, smooth_target,
        density_smoothing_scale)
    if exists(los_file) and not overwrite:
        fprint(f"LOS file already exists at `{los_file}`; skipping.",
               verbose=verbose)
        return {"path": los_file, "status": "exists"}

    out_dir = dirname(los_file)
    if out_dir:
        makedirs(out_dir, exist_ok=True)

    n_sims = len(nsims)
    n_gal = len(RA)
    n_r = len(r)
    is_random_multireal = "random_" in catalogue and n_sims > 1

    loader_cls = candel.field.name2field_loader(reconstruction)
    loader_kwargs = config["io"]["reconstruction_main"][reconstruction]
    fixed_los_geometry = None
    if not is_random_multireal:
        geometry_loader = loader_cls(nsim=nsims[0], **loader_kwargs)
        fixed_los_geometry = candel.field.prepare_los_geometry(
            geometry_loader, r, RA, dec)

    fprint(f"saving the line of sight data to `{los_file}`.",
           verbose=verbose)
    los_tmp_file = f"{los_file}.tmp"
    dt = np.dtype(np.float32)
    dt16 = np.dtype(np.float16)
    with File(los_tmp_file, "w") as fout:
        if is_random_multireal:
            ra_dataset = fout.create_dataset(
                "RA", shape=(n_sims, n_gal), dtype=dt)
            dec_dataset = fout.create_dataset(
                "dec", shape=(n_sims, n_gal), dtype=dt)
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
            fprint(f"loading `{reconstruction}` for sim {nsim}.",
                   verbose=verbose)
            loader = loader_cls(nsim=nsim, **loader_kwargs)
            if is_random_multireal:
                RA_i, dec_i = generate_random_sky(n_gal, seed=42 + nsim)
                los_geometry = None
            else:
                RA_i, dec_i = RA, dec
                los_geometry = fixed_los_geometry
            dens_i, vel_i = candel.field.interpolate_los_density_velocity(
                loader, r, RA_i, dec_i, smooth_target=smooth_target,
                density_smoothing_scale=density_smoothing_scale,
                verbose=verbose, los_geometry=los_geometry)
            density_dataset[i] = dens_i.astype(np.float32)
            velocity_dataset[i] = vel_i.astype(np.float16)
            if is_random_multireal:
                ra_dataset[i] = RA_i.astype(np.float32)
                dec_dataset[i] = dec_i.astype(np.float32)
    replace(los_tmp_file, los_file)
    return {"path": los_file, "status": "computed"}
