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

from os.path import join

import numpy as np
from h5py import File

from .util import SPEED_OF_LIGHT


def load_CF4_data(root, which_band, best_mag_quality=True, eta_min=-0.3,
                  zcmb_max=None):
    """
    Loads the `CF4_TFR.hdf5` file from `root` and extracts fields based on
    `which_band`. Applies filters using `eta_min`, `zcmb_max`, and
    `best_mag_quality`. Returns a dictionary of cleaned and filtered data
    arrays.
    """
    with File(join(root, "CF4_TFR.hdf5"), 'r') as f:
        zcmb = f["Vcmb"][...] / SPEED_OF_LIGHT
        RA = f["RA"][...] * 360 / 24
        DEC = f["DE"][...]

        if which_band == "w1":
            mag = f["w1"][...]
            mag_quality = f["Qw"][...]
        elif which_band == "i":
            mag = f["i"][...]
            mag_quality = f["Qs"][...]
        else:
            raise ValueError("which_band must be 'w1' or 'i'.")

        eta = f["lgWmxi"][...] - 2.5
        e_eta = f["elgWi"][...]

    data = {
        "zcmb": zcmb,
        "theta": np.deg2rad(RA),
        "phi": np.pi / 2 * np.deg2rad(DEC),
        "mag": mag,
        "e_mag": np.ones_like(mag) * 0.05,
        "eta": eta,
        "e_eta": e_eta
    }

    mask = data["eta"] > eta_min
    if best_mag_quality:
        mask &= mag_quality == 5

    if zcmb_max is not None:
        mask &= data["zcmb"] < zcmb_max

    for key in data:
        data[key] = data[key][mask]

    return data


def subsample_data(data, nsamples, seed=42):
    """
    Uses `seed` to randomly select `nsamples` entries from data without
    replacement. Returns a new dictionary containing the subsampled arrays.
    """
    gen = np.random.default_rng(seed)
    ndata = len(data[list(data.keys())[0]])

    if nsamples > ndata:
        raise ValueError(f"`n_samples = {nsamples}` must be less than the "
                         f"number of data points of {ndata}.")

    mask = gen.choice(ndata, nsamples, replace=False)
    return {key: data[key][mask] for key in data}
