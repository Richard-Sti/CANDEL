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
from pathlib import Path

from candel import read_gof


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
