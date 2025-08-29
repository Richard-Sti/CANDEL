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
Script to generate many realizations of mock data.
"""
import os
from os.path import join

import candel
import numpy as np
from h5py import File

mock_dir = "/mnt/home/rstiskalek/ceph/CANDEL/data/CF4_mock"
CF4 = candel.pvdata.load_CF4_data("/mnt/home/rstiskalek/ceph/CANDEL/data/CF4",
                                  which_band="w1",)
os.makedirs(mock_dir, exist_ok=True)

distmod2dist = candel.Distmod2Distance()
distmod2redshift = candel.Distmod2Redshift()
log_grad_distmod2dist = candel.LogGrad_Distmod2ComovingDistance()

field_loader = candel.field.name2field_loader("Carrick2015")(
    path_density="/mnt/home/rstiskalek/ceph/CANDEL/data/fields/carrick2015_twompp_density.npy",   # noqa
    path_velocity="/mnt/home/rstiskalek/ceph/CANDEL/data/fields/carrick2015_twompp_velocity.npy"  # noqa
    )

kwargs = {
    'Vext_mag': 210,
    'Vext_ell': 295,
    'Vext_b': -10,
    'sigma_v': 270,
    'a_TFR': -19.72,
    'b_TFR': -10,
    'c_TFR': 12,
    'sigma_TFR': 0.32,
    'a_TFR_dipole_mag': 0.06,
    'a_TFR_dipole_ell': 142.0,
    'a_TFR_dipole_b': 52.0,
    'alpha': 1.5,
    'beta': 0.43,
    'h': 1,
    'mag': CF4['mag'] - 0.05,
    'eta': CF4['eta'],
    'mag_min': 7.5,
    'mag_max': 16.5,
    'e_mag': 0.05,
    'eta_mean': 0.0,
    'eta_std': 0.125,
    'e_eta': 0.023,
    'b_min': 7.5,
    'zcmb_max': 0.05,
    'r_h_max': 500,
    'distmod2dist': distmod2dist,
    'distmod2redshift': distmod2redshift,
    'log_grad_distmod2dist': log_grad_distmod2dist,
    'field_loader': field_loader,
    'use_data_prior': True,
    'rmin_reconstruction': 0.1,
    'rmax_reconstruction': 250,
    'num_steps_reconstruction': 501,
}

# Grid of sample sizes and seeds
nsample_list = [500, 1000, 2000, 4000, 8000, 16000, 32000]
nseeds_per_nsample = 10

index = 0
records = []

for nsamples in nsample_list:
    for i in range(nseeds_per_nsample):
        seed = 1000 * nsamples + i
        fname = join(mock_dir, f"mock_{index}.hdf5")
        print(f"[INFO] Preparing `{fname}`", flush=True)

        mock = candel.mock.gen_TFR_mock(nsamples, seed=seed, **kwargs)

        with File(fname, 'w') as f:
            grp = f.create_group("mock")
            for key, value in mock.items():
                grp.create_dataset(key, data=value, dtype=np.float32)
            for key, value in kwargs.items():
                if isinstance(value, (float, int, bool)):
                    grp.attrs[key] = value
            grp.attrs["seed"] = seed
            grp.attrs["nsamples"] = nsamples

        records.append((index, nsamples))
        index += 1

# Save index mapping
records = np.array(records, dtype=int)
np.savetxt(join(mock_dir, "mock_index.txt"), records, fmt="%d",
           header="index nsamples")
