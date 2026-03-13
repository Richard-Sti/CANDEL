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
"""
A script to compute the LOS density and radial velocity from an existing
reconstruction and a catalogue of galaxies.
"""
from argparse import ArgumentParser
from os.path import basename, dirname, join, splitext

import numpy as np
from h5py import File
from mpi4py import MPI

import candel
from candel import fprint


def generate_random_sky(npoints, seed):
    gen = np.random.default_rng(seed)
    RA = gen.uniform(0, 360, size=npoints)
    dec = np.arcsin(gen.uniform(-1, 1, size=npoints)) * 180 / np.pi
    return RA, dec


def load_los(catalogue, config, filepath=None, config_path=None):
    if "random_" in catalogue:
        d = config["io"].copy()
        los_file = d.pop("los_file_random")

        npoints = int(catalogue.replace("random_", ''))
        # This could, in principle, account for the ZoA mask.
        RA, dec = generate_random_sky(npoints, seed=42)
    elif catalogue == "CF4":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_CF4_data(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "2MTF":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_2MTF(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "SFI":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_SFI(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "PantheonPlus":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_PantheonPlus(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "PantheonPlusLane":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_PantheonPlus_Lane(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "Foundation":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_Foundation(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "LOSS":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_LOSS(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "SDSS_FP":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_SDSS_FP(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "6dF_FP":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_6dF_FP(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "EDD_TRGB":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_EDD_TRGB(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "EDD_TRGB_grouped":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_EDD_TRGB_grouped(return_all=True, **d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "SH0ES":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
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


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    verbose = rank == 0

    parser = ArgumentParser()
    parser.add_argument("--catalogue", type=str, required=True)
    parser.add_argument("--reconstruction", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smooth_target", type=float, default=None)
    parser.add_argument("--filepath", type=str, default=None,
                        help="Path to catalogue file (required for generic)")
    args = parser.parse_args()

    if args.smooth_target == 0:
        args.smooth_target = None

    config = candel.load_config(args.config)
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

    recon = args.reconstruction
    if recon in nreal_map:
        nsims = list(range(nreal_map[recon]))
    elif recon.lower().startswith("manticore"):
        nsims = list(range(30))
    else:
        if rank == 0:
            raise ValueError(f"Reconstruction `{recon}` not supported.")
        return

    fprint(f"iterating over {len(nsims)} simulations "
           f"for `{args.reconstruction}`.", verbose=verbose)

    if "random_" in args.catalogue:
        # Use the dedicated random-LOS grid with per-reconstruction dr.
        rand_cfg = config["io"].get("reconstruction_rand_los", {})
        rmin = rand_cfg.get("rmin", 0.1)
        rmax = rand_cfg.get("rmax", 251)
        dr = rand_cfg.get("dr", 1.0)
        recon_cfg = rand_cfg.get(args.reconstruction, {})
        rmin = recon_cfg.get("rmin", rmin)
        rmax = recon_cfg.get("rmax", rmax)
        dr = recon_cfg.get("dr", dr)
        num_steps = round((rmax - rmin) / dr) + 1
        r = np.linspace(rmin, rmax, num_steps)
        fprint(f"random LOS grid: {rmin} to {rmax} Mpc/h, "
               f"dr={dr}, {num_steps} steps.", verbose=verbose)
    else:
        d = config["io"]["reconstruction_main"]
        fprint(f"setting the radial grid from {d['rmin']} to {d['rmax']} "
               f"with {d['num_steps']} steps.", verbose=verbose)
        r = np.linspace(d["rmin"], d["rmax"], d["num_steps"])

    fprint(f"loading the catalogue `{args.catalogue}` with "
           f"reconstruction `{args.reconstruction}`.", verbose=verbose)
    RA, dec, los_file = load_los(
        args.catalogue, config, args.filepath, config_path=args.config)
    fprint(f"loaded {len(RA)} galaxies from the catalogue.", verbose=verbose)

    n_sims = len(nsims)
    n_gal = len(RA)
    n_r = len(r)

    # For multi-realisation random catalogues, each sim gets independent sky
    # positions drawn with seed = 42 + nsim instead of sharing seed=42.
    is_random_multireal = "random_" in args.catalogue and n_sims > 1

    # Assign work: indices in nsims handled by this rank
    my_idxs = [i for i in range(n_sims) if (i % size) == rank]

    local_results = []
    for i in my_idxs:
        nsim = nsims[i]
        print(f"[rank {rank}] loading `{args.reconstruction}` for sim {nsim}.")
        loader = candel.field.name2field_loader(args.reconstruction)(
            nsim=nsim,
            **config["io"]["reconstruction_main"][args.reconstruction])
        if is_random_multireal:
            RA_i, dec_i = generate_random_sky(n_gal, seed=42 + nsim)
        else:
            RA_i, dec_i = RA, dec
        dens_i, vel_i = candel.field.interpolate_los_density_velocity(
            loader, r, RA_i, dec_i, args.smooth_target)

        # store with the global slot index so root can place it
        local_results.append(
            (i, dens_i.astype(np.float32), vel_i.astype(np.float32),
             RA_i.astype(np.float32), dec_i.astype(np.float32)))

    # Gather lists of (i, dens, vel, RA, dec) to root
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        los_density = np.full((n_sims, n_gal, n_r), np.nan, dtype=np.float32)
        los_velocity = np.full_like(los_density, np.nan, dtype=np.float32)
        if is_random_multireal:
            all_RA = np.empty((n_sims, n_gal), dtype=np.float32)
            all_dec = np.empty((n_sims, n_gal), dtype=np.float32)

        for rank_results in all_results:
            for i, dens_i, vel_i, RA_i, dec_i in rank_results:
                los_density[i] = dens_i
                los_velocity[i] = vel_i
                if is_random_multireal:
                    all_RA[i] = RA_i
                    all_dec[i] = dec_i

        los_file = los_file.replace("<X>", args.reconstruction)
        if args.smooth_target is not None:
            los_file = los_file.replace(
                ".hdf5",
                f"_smooth_to_{args.smooth_target}.hdf5")

        fprint(f"saving the line of sight data to `{los_file}`.")
        dt = np.dtype(np.float32)
        dt16 = np.dtype(np.float16)
        with File(los_file, "w") as f:
            if is_random_multireal:
                # RA/dec shape (n_sims, n_gal): each realisation has its own
                # independent random sky positions.
                f.create_dataset("RA", data=all_RA, dtype=dt)
                f.create_dataset("dec", data=all_dec, dtype=dt)
            else:
                f.create_dataset("RA", data=RA, dtype=dt)
                f.create_dataset("dec", data=dec, dtype=dt)
            f.create_dataset("r", data=r, dtype=dt)
            f.create_dataset("los_density", data=los_density, dtype=dt)
            f.create_dataset("los_velocity", data=los_velocity, dtype=dt16)

        fprint("all finished.")


if __name__ == "__main__":
    main()
