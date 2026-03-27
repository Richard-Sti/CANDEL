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
"""Megamaser spot data loading from AAS machine-readable tables."""
from os.path import join

import numpy as np

from ..util import fprint


_MRT_FILES = {
    "CGCG074-064": "CGCG074-064_Pesce2020_mrt.txt",
}

# Kuo+2011 Table 3 fixed-width column byte ranges (1-indexed, inclusive).
_KUO2011_COLUMNS = {
    "name":     (1, 8),
    "velocity": (10, 17),
    "x":        (19, 24),
    "sigma_x":  (26, 30),
    "y":        (32, 37),
    "sigma_y":  (39, 43),
    "flux":     (45, 48),
    "e_flux":   (50, 52),
}

# LSR systemic velocities from Kuo+2011 Table 1 Keplerian fits (km/s).
_KUO2011_V_SYS_LSR = {
    "NGC 6264": 10219.0,
    "NGC 6323": 7842.0,
    "UGC 3789": 2900.0,
}

_KUO2011_DV_THRESHOLD = 200.0  # km/s threshold for systemic classification

# Gao+2016 Table 6 column byte ranges (1-indexed, inclusive).
_GAO2016_COLUMNS = {
    "velocity":  (1, 7),
    "x":         (9, 15),
    "sigma_x":   (17, 22),
    "y":         (24, 30),
    "sigma_y":   (32, 37),
    "a":         (39, 44),
    "sigma_a":   (46, 50),
}

# Heliocentric systemic velocity for NGC 5765b, derived from the CMB-frame
# v_cmb = 8525.7 km/s (Pesce+2020 Table 1) minus the helio-to-CMB dipole
# correction of 210.1 km/s at (RA, Dec) = (222.0987, 5.0672).
_NGC5765B_V_SYS_HELIO = 8315.6
_NGC5765B_DV_THRESHOLD = 200.0  # km/s threshold for systemic classification

# Column byte ranges (1-indexed, inclusive) from the MRT header.
_MRT_COLUMNS = {
    "spot_type":      (1, 1),
    "velocity":       (3, 9),
    "flux":           (11, 17),
    "e_flux":         (19, 25),
    "x":              (27, 35),
    "sigma_x":        (37, 44),
    "y":              (46, 54),
    "sigma_y":        (56, 63),
    "a":              (65, 70),
    "sigma_a":        (72, 76),
    "accel_measured":  (78, 78),
}


def _parse_mrt_line(line):
    """Parse a single data line from the MRT file using fixed-width columns."""
    row = {}
    for key, (b0, b1) in _MRT_COLUMNS.items():
        raw = line[b0 - 1:b1].strip()
        if key == "spot_type":
            row[key] = raw
        elif key == "accel_measured":
            row[key] = int(raw)
        else:
            row[key] = float(raw)
    return row


def load_NGC5765b_spots(root, v_cmb_obs=None, v_helio_to_cmb=0.0):
    """Load maser spot data for NGC 5765b from Gao+2016 Table 6.

    The table provides velocity, position (x, y), and acceleration for 192
    maser spots. Spot types are inferred from velocity offset relative to the
    heliocentric systemic velocity (~8315.6 km/s): systemic if
    |v - v_sys| < 200 km/s, otherwise blueshifted or redshifted.
    All spots have measured accelerations.

    Parameters
    ----------
    root
        Directory containing ``NGC5765b_Gao2016_table6.dat``.

    Returns
    -------
    dict with the same keys as ``load_megamaser_spots``.
    """
    fname = "NGC5765b_Gao2016_table6.dat"
    fpath = join(root, fname)
    fprint(f"loading maser spots from '{fpath}'.")

    with open(fpath) as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        row = {}
        for key, (b0, b1) in _GAO2016_COLUMNS.items():
            row[key] = float(line[b0 - 1:b1].strip())
        rows.append(row)

    n = len(rows)
    velocity = np.array([r["velocity"] for r in rows])
    x = np.array([r["x"] for r in rows])
    sigma_x = np.array([r["sigma_x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    sigma_y = np.array([r["sigma_y"] for r in rows])
    a = np.array([r["a"] for r in rows])
    sigma_a = np.array([r["sigma_a"] for r in rows])

    # Classify spot types from heliocentric velocity offsets.
    dv = velocity - _NGC5765B_V_SYS_HELIO
    is_sys = np.abs(dv) < _NGC5765B_DV_THRESHOLD
    is_red = dv >= _NGC5765B_DV_THRESHOLD
    is_blue = dv <= -_NGC5765B_DV_THRESHOLD

    spot_type = np.full(n, "s", dtype="U1")
    spot_type[is_red] = "r"
    spot_type[is_blue] = "b"

    # All spots in Gao+2016 have acceleration measurements.
    accel_measured = np.ones(n, dtype=bool)

    if v_cmb_obs is None:
        raise ValueError("v_cmb_obs must be provided for NGC5765b.")

    data = {
        "spot_type": spot_type,
        "velocity": velocity,
        "x": x,
        "sigma_x": sigma_x,
        "y": y,
        "sigma_y": sigma_y,
        "a": a,
        "sigma_a": sigma_a,
        "accel_measured": accel_measured,
        "is_systemic": is_sys,
        "is_highvel": is_red | is_blue,
        "n_spots": n,
        "galaxy_name": "NGC5765b",
        "v_cmb_obs": float(v_cmb_obs),
        "v_helio_to_cmb": float(v_helio_to_cmb),
    }

    fprint(f"loaded {n} maser spots for NGC5765b "
           f"({is_sys.sum()} systemic, {is_red.sum()} redshifted, "
           f"{is_blue.sum()} blueshifted).")
    return data


def _is_missing(s):
    """Check if a table field represents a missing value."""
    s = s.strip()
    return s in ("...", "sdotsdotsdot", "")


def _load_kuo_table2(root, fname, galaxy_label, v_sys_lsr, dv_threshold=200.0,
                     v_cmb_obs=None, v_helio_to_cmb=0.0):
    """Load maser spots from a Kuo+ Table 2 file with inline accelerations.

    The table is tab-delimited with columns: V_op, Theta_x, sigma_x,
    Theta_y, sigma_y, A, sigma_A. Missing accelerations are marked with
    ``...`` or ``sdotsdotsdot``.

    Parameters
    ----------
    root
        Directory containing the data file.
    fname
        Filename (e.g. ``"NGC6264_Kuo2013_table2.txt"``).
    galaxy_label
        Short label used in output messages.
    v_sys_lsr
        LSR systemic velocity in km/s for spot classification.
    dv_threshold
        Velocity offset threshold for systemic classification.

    Returns
    -------
    dict with the same keys as ``load_megamaser_spots``.
    """
    fpath = join(root, fname)
    fprint(f"loading maser spots from '{fpath}'.")

    with open(fpath) as f:
        lines = f.readlines()

    # Skip header lines (everything before the first numeric data line).
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("Table") or \
                stripped.startswith("(") or stripped.startswith("Notes") or \
                any(c.isalpha() for c in stripped.split("\t")[0].strip()):
            data_start = i + 1
        else:
            break

    velocities, xs, sigma_xs, ys, sigma_ys = [], [], [], [], []
    accels, sigma_accels, has_accel = [], [], []

    for line in lines[data_start:]:
        line = line.rstrip("\n")
        if not line or line.startswith("Notes"):
            break
        fields = line.split("\t")
        if len(fields) < 5:
            continue

        velocities.append(float(fields[0].strip()))
        xs.append(float(fields[1].strip()))
        sigma_xs.append(float(fields[2].strip()))
        ys.append(float(fields[3].strip()))
        sigma_ys.append(float(fields[4].strip()))

        if len(fields) >= 7 and not _is_missing(fields[5]) and \
                not _is_missing(fields[6]):
            accels.append(float(fields[5].strip()))
            sigma_accels.append(float(fields[6].strip()))
            has_accel.append(True)
        elif len(fields) >= 6 and not _is_missing(fields[5]):
            # Acceleration present but no uncertainty -> skip
            accels.append(0.0)
            sigma_accels.append(999.0)
            has_accel.append(False)
        else:
            accels.append(0.0)
            sigma_accels.append(999.0)
            has_accel.append(False)

    n = len(velocities)
    if n == 0:
        raise ValueError(f"No spots found in {fpath}.")

    velocity = np.array(velocities)
    x = np.array(xs)
    sigma_x = np.array(sigma_xs)
    y = np.array(ys)
    sigma_y = np.array(sigma_ys)
    a = np.array(accels)
    sigma_a = np.array(sigma_accels)
    accel_measured = np.array(has_accel, dtype=bool)

    # Classify spot types from LSR velocity offsets.
    dv = velocity - v_sys_lsr
    is_sys = np.abs(dv) < dv_threshold
    is_red = dv >= dv_threshold
    is_blue = dv <= -dv_threshold

    spot_type = np.full(n, "s", dtype="U1")
    spot_type[is_red] = "r"
    spot_type[is_blue] = "b"

    if v_cmb_obs is None:
        raise ValueError(f"v_cmb_obs must be provided for {galaxy_label}.")

    data = {
        "spot_type": spot_type,
        "velocity": velocity,
        "x": x,
        "sigma_x": sigma_x,
        "y": y,
        "sigma_y": sigma_y,
        "a": a,
        "sigma_a": sigma_a,
        "accel_measured": accel_measured,
        "is_systemic": is_sys,
        "is_highvel": is_red | is_blue,
        "n_spots": n,
        "galaxy_name": galaxy_label,
        "v_cmb_obs": float(v_cmb_obs),
        "v_helio_to_cmb": float(v_helio_to_cmb),
    }

    n_accel = int(accel_measured.sum())
    fprint(f"loaded {n} maser spots for {galaxy_label} "
           f"({is_sys.sum()} systemic, {is_red.sum()} redshifted, "
           f"{is_blue.sum()} blueshifted). "
           f"Accelerations: {n_accel}/{n}.")
    return data


def load_NGC6264_spots(root, v_cmb_obs=None, v_helio_to_cmb=0.0):
    """Load maser spot data for NGC 6264 from Kuo+2013 Table 2."""
    return _load_kuo_table2(
        root, "NGC6264_Kuo2013_table2.txt", "NGC6264",
        _KUO2011_V_SYS_LSR["NGC 6264"],
        v_cmb_obs=v_cmb_obs, v_helio_to_cmb=v_helio_to_cmb)


def load_NGC6323_spots(root, v_cmb_obs=None, v_helio_to_cmb=0.0):
    """Load maser spot data for NGC 6323 from Kuo+2015 Table 2."""
    return _load_kuo_table2(
        root, "NGC6323_Kuo2015_table2.txt", "NGC6323",
        _KUO2011_V_SYS_LSR["NGC 6323"],
        v_cmb_obs=v_cmb_obs, v_helio_to_cmb=v_helio_to_cmb)


def load_UGC3789_spots(root, v_cmb_obs=None, v_helio_to_cmb=0.0):
    """Load maser spot data for UGC 3789 from Reid+2013 Table 1."""
    return _load_kuo_table2(
        root, "UGC3789_Reid2013_table1.txt", "UGC3789",
        _KUO2011_V_SYS_LSR["UGC 3789"],
        v_cmb_obs=v_cmb_obs, v_helio_to_cmb=v_helio_to_cmb)


def load_megamaser_spots(root, galaxy="CGCG074-064", v_cmb_obs=None,
                         v_helio_to_cmb=0.0):
    """Load individual maser spot data for a megamaser galaxy.

    Parameters
    ----------
    root
        Directory containing the data file.
    galaxy
        Galaxy name: ``"CGCG074-064"``, ``"NGC5765b"``, etc.
    v_cmb_obs
        Observed CMB-frame recession velocity in km/s. Required.
    v_helio_to_cmb
        Heliocentric-to-CMB velocity correction in km/s (default 0).

    Returns
    -------
    dict with numpy arrays for each spot property plus derived masks,
    including ``v_cmb_obs`` and ``v_helio_to_cmb``.
    """
    if galaxy == "NGC5765b":
        return load_NGC5765b_spots(root, v_cmb_obs=v_cmb_obs,
                                   v_helio_to_cmb=v_helio_to_cmb)
    if galaxy == "NGC6264":
        return load_NGC6264_spots(root, v_cmb_obs=v_cmb_obs,
                                  v_helio_to_cmb=v_helio_to_cmb)
    if galaxy == "NGC6323":
        return load_NGC6323_spots(root, v_cmb_obs=v_cmb_obs,
                                  v_helio_to_cmb=v_helio_to_cmb)
    if galaxy == "UGC3789":
        return load_UGC3789_spots(root, v_cmb_obs=v_cmb_obs,
                                  v_helio_to_cmb=v_helio_to_cmb)

    _all = list(_MRT_FILES) + ["NGC5765b", "NGC6264", "NGC6323", "UGC3789"]
    if galaxy not in _MRT_FILES:
        raise ValueError(
            f"Unknown galaxy '{galaxy}'. Available: {_all}")

    if v_cmb_obs is None:
        raise ValueError(f"v_cmb_obs must be provided for {galaxy}.")

    fpath = join(root, _MRT_FILES[galaxy])
    fprint(f"loading maser spots from '{fpath}'.")

    with open(fpath) as f:
        lines = f.readlines()

    # Find the last separator line; data starts after it.
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            data_start = i + 1

    rows = []
    for line in lines[data_start:]:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        rows.append(_parse_mrt_line(line))

    n = len(rows)
    spot_type = np.array([r["spot_type"] for r in rows])
    velocity = np.array([r["velocity"] for r in rows])
    x = np.array([r["x"] for r in rows])
    sigma_x = np.array([r["sigma_x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    sigma_y = np.array([r["sigma_y"] for r in rows])
    a = np.array([r["a"] for r in rows])
    sigma_a = np.array([r["sigma_a"] for r in rows])
    accel_measured = np.array([r["accel_measured"] for r in rows], dtype=bool)

    data = {
        "spot_type": spot_type,
        "velocity": velocity,
        "x": x,
        "sigma_x": sigma_x,
        "y": y,
        "sigma_y": sigma_y,
        "a": a,
        "sigma_a": sigma_a,
        "accel_measured": accel_measured,
        "is_systemic": spot_type == "s",
        "is_highvel": (spot_type == "b") | (spot_type == "r"),
        "n_spots": n,
        "galaxy_name": galaxy,
        "v_cmb_obs": float(v_cmb_obs),
        "v_helio_to_cmb": float(v_helio_to_cmb),
    }

    fprint(f"loaded {n} maser spots for {galaxy} "
           f"({data['is_systemic'].sum()} systemic, "
           f"{data['is_highvel'].sum()} high-velocity).")
    return data
