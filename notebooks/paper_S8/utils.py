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

from h5py import File


def get_key_all(fnames, key):
    values = []
    for fname in fnames:
        with File(fname, 'r') as f:
            values.append(f["samples"][key][...])

    return values


def compute_S8_all(fnames, beta2cosmo):
    S8_list = []
    for fname in fnames:
        with File(fname, 'r') as f:
            beta = f["samples"]["beta"][...]
        S8 = beta2cosmo.compute_S8(beta)
        S8_list.append(S8)

    return S8_list


def replace_token_in_paths(files, token, replacement=""):
    """Replace or remove a token in simple (path, label) tuples."""
    new_files = []
    for path, label in files:
        new_path = path.replace(token, replacement if replacement else "")
        new_files.append((new_path, label))
    return new_files
