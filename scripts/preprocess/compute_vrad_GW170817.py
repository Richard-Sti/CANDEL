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
Evaluate the Carrick2015 and Manticore radial velocity and density fields at
the positions of GW170817 posterior samples (RA, dec, luminosity distance).
"""
from argparse import ArgumentParser
from os.path import join

import numpy as np
from h5py import File
from mpi4py import MPI

import candel
from candel import fprint
from candel.field import name2field_loader
from candel.field.field_interp import build_regular_interpolator
from candel.util import (radec_to_cartesian, radec_to_galactic,
                         radec_to_supergalactic)


def direction_vectors(RA_deg, dec_deg, coordinate_frame):
    """Compute unit direction vectors in the field's native frame."""
    if coordinate_frame == "icrs":
        return radec_to_cartesian(RA_deg, dec_deg)
    elif coordinate_frame == "galactic":
        ell, b = radec_to_galactic(RA_deg, dec_deg)
        return radec_to_cartesian(ell, b)
    elif coordinate_frame == "supergalactic":
        ell, b = radec_to_supergalactic(RA_deg, dec_deg)
        return radec_to_cartesian(ell, b)
    else:
        raise ValueError(f"Unknown coordinate frame: `{coordinate_frame}`.")


def interpolate_density_velocity_at_positions(field_loader, RA_deg, dec_deg,
                                              dist):
    """
    Interpolate the density and radial velocity fields at specific
    (RA, dec, distance) positions.

    Parameters
    ----------
    field_loader : BaseFieldLoader
        A field loader instance.
    RA_deg : 1D array
        Right ascension in degrees.
    dec_deg : 1D array
        Declination in degrees.
    dist : 1D array
        Distances in Mpc/h.

    Returns
    -------
    density : 1D array
        Density at each position (1 + delta).
    vrad : 1D array
        Radial (line-of-sight) velocity at each position in km/s.
    """
    rhat = direction_vectors(RA_deg, dec_deg, field_loader.coordinate_frame)
    rhat = rhat.astype(np.float32)

    # Positions: (n_samples, 3)
    pos = (field_loader.observer_pos[None, :]
           + dist[:, None] * rhat).astype(np.float32)

    # Interpolate density (in log space for numerical stability, following
    # interpolate_los_density_velocity). Normalise to 1+delta by dividing by
    # the field mean (a no-op for Carrick which already returns 1+delta, but
    # converts Manticore's physical density to overdensity).
    eps = 1e-4
    density_field = field_loader.load_density()
    density_field /= density_field.mean()
    density_field = np.log(density_field + eps)
    fill_value = np.log(1 + eps)
    f_density = build_regular_interpolator(
        density_field, field_loader.boxsize, fill_value=fill_value)
    density = np.exp(f_density(pos)) - eps
    density = np.clip(density, eps, None).astype(np.float32)

    # Interpolate velocity components
    velocity = field_loader.load_velocity()  # (3, ngrid, ngrid, ngrid)
    v_interp = np.empty((len(dist), 3), dtype=np.float32)
    for i in range(3):
        f_vel = build_regular_interpolator(velocity[i], field_loader.boxsize)
        v_interp[:, i] = f_vel(pos)

    # Project onto the radial direction
    vrad = np.einsum('ij,ij->i', v_interp, rhat)
    return density, vrad


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    verbose = rank == 0

    parser = ArgumentParser()
    parser.add_argument("--reconstruction", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = candel.load_config(args.config)
    root_data = candel.get_root_data(config)
    recon = args.reconstruction

    # Number of realisations per reconstruction
    nreal_map = {"Carrick2015": 1, "Lilow2024": 1, "CF4": 100}
    if recon in nreal_map:
        nsims = list(range(nreal_map[recon]))
    elif recon.lower().startswith("manticore"):
        nsims = list(range(30))
    else:
        raise ValueError(f"Reconstruction `{recon}` not supported.")

    # Load GW170817 posterior samples
    fprint("loading GW170817 posterior samples.", verbose=verbose)
    data = np.genfromtxt(
        join(root_data, "high_spin_PhenomPNRT_posterior_samples.dat.gz"),
        names=True)
    n_samples = len(data)
    fprint(f"loaded {n_samples} samples.", verbose=verbose)

    # RA, dec are in radians in the file -> convert to degrees
    RA_deg = np.rad2deg(data["right_ascension"])
    dec_deg = np.rad2deg(data["declination"])
    dist_Mpc = data["luminosity_distance_Mpc"]

    # Convert distance from Mpc to Mpc/h (fields are in Mpc/h).
    # Use h = 0.674 as a fiducial value (Planck 2018). The exact value is not
    # critical because the GW posteriors will anyway be reweighted later.
    loader_cls = name2field_loader(recon)
    h = 0.674
    dist_Mpch = dist_Mpc * h
    fprint(f"using h = {h} to convert distances to Mpc/h.", verbose=verbose)

    n_sims = len(nsims)

    # Distribute realisations across MPI ranks
    my_idxs = [i for i in range(n_sims) if (i % size) == rank]
    local_results = []

    for i in my_idxs:
        nsim = nsims[i]
        fprint(f"[rank {rank}] processing realisation {nsim}.")
        loader = loader_cls(
            nsim=nsim,
            **config["io"]["reconstruction_main"][recon])
        dens_i, vrad_i = interpolate_density_velocity_at_positions(
            loader, RA_deg, dec_deg, dist_Mpch)
        local_results.append((i, dens_i, vrad_i))

    # Gather results to root
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        density_all = np.full((n_sims, n_samples), np.nan, dtype=np.float32)
        vrad_all = np.full((n_sims, n_samples), np.nan, dtype=np.float32)

        for rank_results in all_results:
            for i, dens_i, vrad_i in rank_results:
                density_all[i] = dens_i
                vrad_all[i] = vrad_i

        outfile = join(root_data, f"GW170817_fields_{recon}.hdf5")
        fprint(f"saving results to `{outfile}`.")
        with File(outfile, "w") as f:
            f.create_dataset("density", data=density_all)
            f.create_dataset("vrad", data=vrad_all)
            f.create_dataset("RA_deg", data=RA_deg.astype(np.float32))
            f.create_dataset("dec_deg", data=dec_deg.astype(np.float32))
            f.create_dataset("luminosity_distance_Mpc",
                             data=dist_Mpc.astype(np.float32))
            f.attrs["reconstruction"] = recon
            f.attrs["h"] = h
            f.attrs["n_realisations"] = n_sims

        fprint("all finished.")


if __name__ == "__main__":
    main()
