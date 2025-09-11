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
import re
from datetime import datetime
from os.path import join
from pathlib import Path

import candel
import numpy as np
from candel import read_gof
from h5py import File
from scipy.stats import norm


def compare_zeropoint_dipole_gof(fname, which, verbose=True):
    if "zeropoint_dipole" not in fname:
        raise ValueError("`zeropoint_dipole` not in filename.")

    gof_dipole = read_gof(fname, which)
    gof_no_dipole = read_gof(re.sub(r"_zeropoint_dipole(UnifComponents)?", "", fname), which)  # noqa

    if verbose:
        print(f"[DIPOLE]:    {gof_dipole}")
        print(f"[ISO]:       {gof_no_dipole}")

        # Report last modified time
        mtime = Path(fname).stat().st_mtime
        ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] File last modified: {ts}")

    return gof_dipole - gof_no_dipole


def get_bulkflow_simulation(root, simname, convert_to_galactic=True):
    f = np.load(join(root, f"enclosed_mass_{simname}.npz"))
    r, B = f["distances"], f["cumulative_velocity"]

    if convert_to_galactic:
        Bmag, Bl, Bb = candel.cartesian_to_radec(
            B[..., 0], B[..., 1], B[..., 2])
        Bl, Bb = candel.radec_to_galactic(Bl, Bb)
        B = np.stack([Bmag, Bl, Bb], axis=-1)

    return r, B


def get_bulkflow(fname, simname, root_bf, convert_to_galactic=True,
                 downsample=1, Rmax=125, include_Vext=True,
                 include_zeropoint_dipole=False, H0=73.04):
    # Read in the samples
    with File(fname, "r") as f:
        grp = f["samples"]
        Vext = grp["Vext"][...]

        try:
            beta = grp["beta"][...]
        except KeyError:
            beta = np.ones(len(Vext))

        try:
            dZP_mag = grp["zeropoint_dipole_mag"][...]
            dZP_ell = grp["zeropoint_dipole_ell"][...]
            dZP_b = grp["zeropoint_dipole_b"][...]
            dZP = candel.galactic_to_radec_cartesian(
                dZP_ell, dZP_b)
            dZP *= dZP_mag[:, None]

        except KeyError as e:
            if include_zeropoint_dipole:
                raise KeyError(
                    "Zeropoint dipole requested but not found in file.") from e
            else:
                dZP = np.zeros_like(Vext)

        sigma_v = grp["sigma_v"][...]

    # Read in the bulk flow
    names_map = {
        "CB1": "csiborg1",
        "CB2": "csiborg2_main",
        }
    f = np.load(join(
        root_bf, f"enclosed_mass_{names_map.get(simname, simname)}.npz"))

    r = f["distances"]
    # Shape of B_i is (nsims, nradial)
    Bx, By, Bz = (f["cumulative_velocity"][..., i] for i in range(3))

    # Mask out the unconstrained large scales
    Rmax = Rmax  # Mpc/h
    mask = r < Rmax
    r = r[mask]
    Bx = Bx[:, mask]
    By = By[:, mask]
    Bz = Bz[:, mask]

    Vext = Vext[::downsample]
    dZP = dZP[::downsample]
    beta = beta[::downsample]

    # Multiply the simulation velocities by beta.
    Bx = Bx[..., None] * beta
    By = By[..., None] * beta
    Bz = Bz[..., None] * beta

    # Add V_ext, shape of B_i is `(nsims, nradial, nsamples)``
    if include_Vext:
        Bx = Bx + Vext[:, 0]
        By = By + Vext[:, 1]
        Bz = Bz + Vext[:, 2]

    if include_zeropoint_dipole:
        # Note the negative sign!, shape is `(nsamples, 3)`
        dH0 = - H0 * (10**(dZP / 5) - 1)
        Bx += (0.75 * r[:, None] * dH0[:, 0][None, :])[None, ...]
        By += (0.75 * r[:, None] * dH0[:, 1][None, :])[None, ...]
        Bz += (0.75 * r[:, None] * dH0[:, 2][None, :])[None, ...]

    Bcart = np.stack([Bx, By, Bz], axis=-1)

    # Bulk flow in Cartesian coordinates at the origin, `(nsims, nsamples, 3)`.
    # We need to find the first finite point in radial distance.
    k = np.where(np.isfinite(Bcart[0, :, 0, 0]))[0][0]
    Bcart_origin = Bcart[:, k, ...]

    # Add sigma_v scatter to it
    nsim, nsample, __ = Bcart_origin.shape
    for i in range(nsample):
        Bcart_origin[:, i, :] += norm(0, sigma_v[i]).rvs(size=(nsim, 3))

    if convert_to_galactic:
        Bmag, Bl, Bb = candel.cartesian_to_radec(Bx, By, Bz)
        Bl, Bb = candel.radec_to_galactic(Bl, Bb)
        B = np.stack([Bmag, Bl, Bb], axis=-1)

        Bmag, Bl, Bb = candel.cartesian_to_radec(
            Bcart_origin[..., 0], Bcart_origin[..., 1], Bcart_origin[..., 2])
        Bl, Bb = candel.radec_to_galactic(Bl, Bb)
        Borigin = np.stack([Bmag, Bl, Bb], axis=-1)[0, ...]

    else:
        B = Bcart
        Borigin = Bcart_origin

    # Stack over the simulations
    B = np.hstack([B[i] for i in range(len(B))])
    return r, B, Borigin
