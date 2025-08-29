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

import numpy as np
from h5py import File
from mpi4py import MPI

import candel
from candel import fprint


def load_los(catalogue, config):
    if "random_" in catalogue:
        d = config["io"].copy()
        los_file = d.pop("los_file_random")

        npoints = int(catalogue.replace("random_", ''))
        # This could, in principle, account for the ZoA mask.
        gen = np.random.default_rng(42)
        RA = gen.uniform(0, 360, size=npoints)
        dec = np.arcsin(gen.uniform(-1, 1, size=npoints)) * 180 / np.pi
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
    elif catalogue == "Clusters":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_clusters(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "SDSS_FP":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_SDSS_FP(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "SH0ES":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_SH0ES_separated(**d)
        RA, dec = data["RA_host"], data["dec_host"]
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
    args = parser.parse_args()

    config = candel.load_config(args.config)

    if args.reconstruction == "Carrick2015":
        nsims = [0]
    elif args.reconstruction.lower().startswith("manticore"):
        nsims = list(range(30))
    else:
        if rank == 0:
            raise ValueError(
                f"Reconstruction `{args.reconstruction}` not supported.")
        else:
            return

    fprint(f"iterating over {len(nsims)} simulations "
           f"for `{args.reconstruction}`.", verbose=verbose)

    d = config["io"]["reconstruction_main"]
    fprint(f"settin the radial grid from {d['rmin']} to {d['rmax']} with "
           f"{d['num_steps']} steps.", verbose=verbose)
    r = np.linspace(d["rmin"], d["rmax"], d["num_steps"])

    fprint(f"loading the catalogue `{args.catalogue}` with "
           f"reconstruction `{args.reconstruction}`.", verbose=verbose)
    RA, dec, los_file = load_los(args.catalogue, config)
    fprint(f"loaded {len(RA)} galaxies from the catalogue.", verbose=verbose)

    n_sims = len(nsims)
    n_gal = len(RA)
    n_r = len(r)

    # Assign work: indices in nsims handled by this rank
    my_idxs = [i for i in range(n_sims) if (i % size) == rank]

    local_results = []
    for i in my_idxs:
        nsim = nsims[i]
        print(f"[rank {rank}] loading `{args.reconstruction}` for sim {nsim}.")
        loader = candel.field.name2field_loader(args.reconstruction)(
            nsim=nsim,
            **config["io"]["reconstruction_main"][args.reconstruction])
        dens_i, vel_i = candel.field.interpolate_los_density_velocity(
            loader, r, RA, dec)
        # store with the global slot index so root can place it
        local_results.append(
            (i, dens_i.astype(np.float32), vel_i.astype(np.float32)))

    # Gather lists of (i, dens, vel) to root
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        los_density = np.full((n_sims, n_gal, n_r), np.nan, dtype=np.float32)
        los_velocity = np.full_like(los_density, np.nan, dtype=np.float32)

        for rank_results in all_results:
            for i, dens_i, vel_i in rank_results:
                los_density[i] = dens_i
                los_velocity[i] = vel_i

        los_file = los_file.replace("<X>", args.reconstruction)
        fprint(f"saving the line of sight data to `{los_file}`.")
        dt = np.dtype(np.float32)
        with File(los_file, "w") as f:
            f.create_dataset("RA", data=RA, dtype=dt)
            f.create_dataset("dec", data=dec, dtype=dt)
            f.create_dataset("r", data=r, dtype=dt)
            f.create_dataset("los_density", data=los_density, dtype=dt)
            f.create_dataset("los_velocity", data=los_velocity, dtype=dt)

        fprint("all finished.")


if __name__ == "__main__":
    main()
