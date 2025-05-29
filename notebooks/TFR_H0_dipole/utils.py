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
from datetime import datetime
from pathlib import Path
from os.path import basename

from h5py import File
from candel import plot_corner_getdist, read_gof


def compare_zeropoint_dipole_gof(fname, which, verbose=True):
    if "aTFRdipole" not in fname:
        raise ValueError("`aTFRdipole` not in filename.")

    gof_dipole = read_gof(fname, which)
    gof_no_dipole = read_gof(fname.replace("_aTFRdipole", ""), which)

    if verbose:
        print(f"[DIPOLE]:    {gof_dipole}")
        print(f"[ISO]:       {gof_no_dipole}")

        # Report last modified time
        mtime = Path(fname).stat().st_mtime
        ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] File last modified: {ts}")

    return gof_dipole - gof_no_dipole


def plot_corner_from_hdf5(fnames, keys=None, labels=None, fontsize=None,
                          filled=True, show_fig=True, filename=None):
    """
    Plot a triangle plot from one or more HDF5 files containing posterior
    samples.
    """
    if isinstance(fnames, (str, Path)):
        fnames = [fnames]

    samples_list = []
    for fname in fnames:
        with File(fname, 'r') as f:
            grp = f["samples"]
            samples = {key: grp[key][...] for key in grp.keys()}
            samples_list.append(samples)

            full_keys = list(grp.keys())
            print(f"{basename(fname)}: {', '.join(full_keys)}")

    plot_corner_getdist(
        samples_list,
        labels=labels,
        keys=keys,
        fontsize=fontsize,
        filled=filled,
        show_fig=show_fig,
        filename=filename
    )
