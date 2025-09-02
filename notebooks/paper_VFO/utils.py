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


import numpy as np
import candel
from os.path import join
from h5py import File
from scipy.stats import norm


def simname2label(simname):
    ltx = {"Carrick2015": "Carrick+15",
           "Lilow2024": "Lilow+24",
           "CB1": r"\texttt{CSiBORG}1",
           "CB2": r"\texttt{CSiBORG}2",
           "manticore_2MPP_MULTIBIN_N256_DES_V2": r"\texttt{Manticore-Local}",
           "CF4": "Courtois+23",
           "CLONES": "Sorce+2018",
           "HAMLET_V0": r"\texttt{HAMLET}-L",
           "HAMLET_V1": r"\texttt{HAMLET}-NL",
           }

    if isinstance(simname, list):
        names = [ltx[s] if s in ltx else s for s in simname]
        return "".join([f"{n}, " for n in names]).rstrip(", ")

    return ltx[simname] if simname in ltx else simname


def catalogue2label(catalogue):
    ltx = {"SFI_gals": r"SFI\texttt{++}",
           "2MTF": r"2MTF",
           "SFI": r"SFI\texttt{++}",
           "CF4_i": r"CF4 $i$",
           "CF4_W1": r"CF4 W1",
           "CF4_W2": r"CF4 W2",
           }

    if isinstance(catalogue, list):
        names = [ltx[s] if s in ltx else s for s in catalogue]
        return "".join([f"{n}, " for n in names]).rstrip(", ")

    return ltx[catalogue] if catalogue in ltx else catalogue


def simname2color(simname, gen=None):
    # cols = ["#1be7ffff", "#6eeb83ff", "#e4ff1aff", "#ffb800ff",
    #         "#ff5714ff", "#9b5de5ff"]

    defaults = ["tab:blue", "tab:orange", "tab:green",
                "tab:red", "tab:purple", "tab:brown",
                "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    colors_per_sim = {
        "Carrick2015": '#0C5DA5',
        "Lilow2024": '#00B945',
        "CB1": '#FF9500',
        "CB2": '#FF2C00',
        "CF4": '#845B97',
        "CLONES": '#9e9e9e',
    }

    if simname in colors_per_sim:
        return colors_per_sim[simname]

    if gen is None:
        gen = np.random.default_rng()

    return gen.choice(defaults)


def switch_paths_SN_to_no_MNR(files):
    new_files = []
    for name, paths in files:
        new_paths = []
        for p in paths:
            if "LOSS" in p or "Foundation" in p:
                new_paths.append(p.replace("MNR", "noMNR"))
            else:
                new_paths.append(p)
        new_files.append((name, new_paths))
    return new_files


def load_and_check_posteriors(files, samples, key):
    data = {}
    Nsim = len(files)
    Ncat = len(samples)

    for i in range(Nsim):
        simname, paths = files[i]
        for j in range(Ncat):
            cat = samples[j]
            fname = paths[j]

            try:
                arr = candel.read_samples("", fname, keys=key)
            except Exception as e:
                raise ValueError(
                    f"Samples for {key} [NaN/empty] sim={simname} | "
                    f"catalogue={cat} | filename={fname}") from e

            data[(simname, cat)] = arr

    return data


def load_and_clean_logZ(files, samples, stat="logZ_harmonic"):
    if not stat.startswith("logZ"):
        raise ValueError("Invalid statistic name.")

    gof = np.array([
        [candel.read_gof(f, stat, raise_error=False) for f in fs]
        for _, fs in files
    ])

    # Replace non-finite with NaN
    gof[~np.isfinite(gof)] = np.nan

    Nsim, Ncat = gof.shape
    for i in range(Nsim):
        simname, _ = files[i]
        for j in range(Ncat):
            if np.isnan(gof[i, j]):
                print(
                    f"LogZ [NaN] sim = {simname} | catalogue = {samples[j]} "
                    f"| filename = {files[i][1][j]}")

    # Subtract per-column minimum
    gof -= np.nanmin(gof, axis=0, keepdims=True)

    return gof


def strip_token_from_paths(files, token):
    new_files = []
    for name, paths in files:
        new_paths = [p.replace(f"_{token}", "") for p in paths]
        new_files.append((name, new_paths))
    return new_files


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
                 downsample=1, Rmax=125):
    # Read in the samples
    with File(fname, "r") as f:
        grp = f["samples"]
        Vext = grp["Vext"][...]

        try:
            beta = grp["beta"][...]
        except KeyError:
            beta = np.ones(len(Vext))

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
    beta = beta[::downsample]

    # Multiply the simulation velocities by beta.
    Bx = Bx[..., None] * beta
    By = By[..., None] * beta
    Bz = Bz[..., None] * beta

    # Add V_ext, shape of B_i is `(nsims, nradial, nsamples)``
    Bx = Bx + Vext[:, 0]
    By = By + Vext[:, 1]
    Bz = Bz + Vext[:, 2]

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
