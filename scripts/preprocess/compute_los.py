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

import candel
from candel import fprint


def load_los(catalogue, config):
    if catalogue == "CF4":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_CF4_data(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "2MTF":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_2MTF(**d)
        RA, dec = data["RA"], data["dec"]
    elif catalogue == "PantheonPlus":
        d = config["io"]["PV_main"][catalogue].copy()
        los_file = d.pop("los_file")
        data = candel.pvdata.load_PantheonPlus(**d)
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
    else:
        raise ValueError(f"Catalogue {catalogue} not supported. Please add "
                         "support for it.")

    return RA, dec, los_file


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--catalogue", type=str,
                        help="Which catalogue to use.", required=True)
    parser.add_argument("--reconstruction", type=str,
                        help="Which reconstruction to use.", required=True)
    parser.add_argument("--config", type=str,
                        help="Path to the config file with paths.",
                        required=True)
    args = parser.parse_args()

    config = candel.load_config(args.config)

    d = config["io"]["reconstruction_main"]
    fprint(f"settin the radial grid from {d['rmin']} to {d['rmax']} with "
           f"{d['num_steps']} steps.")
    r = np.linspace(d["rmin"], d["rmax"], d["num_steps"])

    fprint(f"loading the catalogue `{args.catalogue}` with "
           f"reconstruction `{args.reconstruction}`.")
    RA, dec, los_file = load_los(args.catalogue, config)
    loader = candel.field.name2field_loader(
        args.reconstruction)(**config["io"]["reconstruction_main"][args.reconstruction])  # noqa
    fprint(f"loaded {len(RA)} galaxies from the catalogue.")

    los_density, los_velocity = candel.field.interpolate_los_density_velocity(
        loader, r, RA, dec)

    los_file = los_file.replace("<X>", args.reconstruction)
    fprint(f"saving the line of sight data to `{los_file}`.")
    with File(los_file, "w") as f:
        f.create_dataset("RA", data=RA, dtype=np.float32)
        f.create_dataset("dec", data=dec, dtype=np.float32)
        f.create_dataset("r", data=r, dtype=np.float32)
        f.create_dataset("los_density", data=los_density, dtype=np.float32)
        f.create_dataset("los_velocity", data=los_velocity, dtype=np.float32)

    fprint("all finished.")
