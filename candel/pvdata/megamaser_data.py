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


def load_NGC5765b_spots(root, v_sys_obs=None):
    """Load maser spot data for NGC 5765b from Gao+2016 Table 6.

    The table provides velocity, position (x, y), and acceleration for 192
    maser spots. All spots have measured accelerations.

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

    cols = ["velocity", "x", "sigma_x", "y", "sigma_y", "a", "sigma_a"]
    rows = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append({k: float(v) for k, v in zip(cols, parts)})

    n = len(rows)
    velocity = np.array([r["velocity"] for r in rows])
    x = np.array([r["x"] for r in rows])
    sigma_x = np.array([r["sigma_x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    sigma_y = np.array([r["sigma_y"] for r in rows])
    a = np.array([r["a"] for r in rows])
    sigma_a = np.array([r["sigma_a"] for r in rows])

    # Flag placeholder accelerations:
    # A=1.0, sigma_a=1.0 (sentinel) and A=0.0, sigma_a=0.2 (undetected)
    accel_measured = ~(
        ((a == 1.0) & (sigma_a == 1.0))
        | ((a == 0.0) & (np.abs(sigma_a - 0.2) < 0.01))
    )

    if v_sys_obs is None:
        raise ValueError("v_sys_obs must be provided for NGC5765b.")

    data = {
        "velocity": velocity,
        "x": x,
        "sigma_x": sigma_x,
        "y": y,
        "sigma_y": sigma_y,
        "a": a,
        "sigma_a": sigma_a,
        "accel_measured": accel_measured,
        "n_spots": n,
        "galaxy_name": "NGC5765b",
        "v_sys_obs": float(v_sys_obs),
    }

    n_accel = int(accel_measured.sum())
    fprint(f"loaded {n} maser spots for NGC5765b "
           f"({n_accel} with measured acceleration).")
    return data


def _is_missing(s):
    """Check if a table field represents a missing value."""
    s = s.strip()
    return s in ("...", "sdotsdotsdot", "")


def _load_kuo_table2(root, fname, galaxy_label, v_sys_obs=None):
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

    if v_sys_obs is None:
        raise ValueError(f"v_sys_obs must be provided for {galaxy_label}.")

    data = {
        "velocity": velocity,
        "x": x,
        "sigma_x": sigma_x,
        "y": y,
        "sigma_y": sigma_y,
        "a": a,
        "sigma_a": sigma_a,
        "accel_measured": accel_measured,
        "n_spots": n,
        "galaxy_name": galaxy_label,
        "v_sys_obs": float(v_sys_obs),
    }

    n_accel = int(accel_measured.sum())
    fprint(f"loaded {n} maser spots for {galaxy_label} "
           f"({n_accel} with measured acceleration).")
    return data


def load_NGC6264_spots(root, v_sys_obs=None):
    """Load maser spot data for NGC 6264 from Kuo+2013 Table 2."""
    return _load_kuo_table2(
        root, "NGC6264_Kuo2013_table2.txt", "NGC6264",
        v_sys_obs=v_sys_obs)


def load_NGC6323_spots(root, v_sys_obs=None):
    """Load maser spot data for NGC 6323 from Kuo+2015 Table 2."""
    return _load_kuo_table2(
        root, "NGC6323_Kuo2015_table2.txt", "NGC6323",
        v_sys_obs=v_sys_obs)


def load_UGC3789_spots(root, v_sys_obs=None):
    """Load maser spot data for UGC 3789 from Reid+2013 Table 1."""
    return _load_kuo_table2(
        root, "UGC3789_Reid2013_table1.txt", "UGC3789",
        v_sys_obs=v_sys_obs)


def load_NGC4258_spots(root, v_sys_obs=472.0, flux_snr_cut=3.0):
    """Load maser spot data for NGC 4258 from Argon+2007 Table 5.

    The table has 14291 rows from 18 VLBI epochs (A–R). For each unique
    (RefVel, Chan) pair the epoch with highest flux is kept. Spots are
    classified by RefVel: ≥800 km/s → red HV, ≤-100 km/s → blue HV,
    400–650 km/s → systemic; other RefVel values are skipped.

    The e_Flux column in the file is in mJy; it is converted to Jy here.
    No accelerations are available, so ``a=0`` and ``sigma_a=999`` are set.

    Parameters
    ----------
    root
        Directory containing ``NGC4258_Argon2007_table5.dat``.
    v_sys_obs
        Observed CMB-frame recession velocity in km/s.
    flux_snr_cut
        Minimum Flux/e_Flux required to keep a spot.

    Returns
    -------
    dict with the same keys as ``load_megamaser_spots``.
    """
    fpath = join(root, "NGC4258_Argon2007_table5.dat")
    fprint(f"loading maser spots from '{fpath}'.")

    # best[key] = (flux, e_flux, cz, ewoff, e_ewoff, nsoff, e_nsoff, refvel)
    best = {}
    with open(fpath) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            chan = int(line[2:5].strip())
            flux = float(line[26:33].strip())
            e_flux = float(line[34:43].strip()) / 1000.0  # mJy -> Jy
            refvel = float(line[113:121].strip())
            key = (refvel, chan)
            if key not in best or flux > best[key][0]:
                cz = float(line[6:15].strip())
                ewoff = float(line[44:51].strip())
                e_ewoff = float(line[52:59].strip())
                nsoff = float(line[60:67].strip())
                e_nsoff = float(line[68:75].strip())
                best[key] = (flux, e_flux, cz, ewoff, e_ewoff, nsoff, e_nsoff,
                             refvel)

    velocity, x, sigma_x, y, sigma_y, spot_type = [], [], [], [], [], []
    n_skip_snr = n_skip_band = 0
    for (refvel, _chan), (flux, e_flux, cz, ew, e_ew, ns, e_ns, _rv) in \
            best.items():
        # S/N cut
        if e_flux <= 0 or flux / e_flux < flux_snr_cut:
            n_skip_snr += 1
            continue
        # Band classification
        if refvel >= 800:
            st = "r"
        elif refvel <= -100:
            st = "b"
        elif 400 <= refvel <= 650:
            st = "s"
        else:
            n_skip_band += 1
            continue
        velocity.append(cz)
        x.append(ew)
        sigma_x.append(max(e_ew, 0.001))
        y.append(ns)
        sigma_y.append(max(e_ns, 0.001))
        spot_type.append(st)

    n = len(velocity)
    n_r = spot_type.count("r")
    n_b = spot_type.count("b")
    n_s = spot_type.count("s")
    fprint(f"loaded {n} maser spots for NGC4258 "
           f"({n_r} red, {n_b} blue, {n_s} systemic; "
           f"dropped {n_skip_snr} low-SNR, {n_skip_band} ambiguous-band).")

    return {
        "velocity":        np.array(velocity),
        "x":               np.array(x),
        "sigma_x":         np.array(sigma_x),
        "y":               np.array(y),
        "sigma_y":         np.array(sigma_y),
        "a":               np.zeros(n),
        "sigma_a":         np.full(n, 999.0),
        "accel_measured":  np.zeros(n, dtype=bool),
        "spot_type":       spot_type,
        "n_spots":         n,
        "galaxy_name":     "NGC4258",
        "v_sys_obs":       float(v_sys_obs),
    }


def load_megamaser_spots(root, galaxy="CGCG074-064", v_sys_obs=None):
    """Load individual maser spot data for a megamaser galaxy.

    Parameters
    ----------
    root
        Directory containing the data file.
    galaxy
        Galaxy name: ``"CGCG074-064"``, ``"NGC5765b"``, etc.
    v_sys_obs
        Observed CMB-frame recession velocity in km/s. Required.

    Returns
    -------
    dict with numpy arrays for each spot property, including ``v_sys_obs``.
    """
    if galaxy == "NGC5765b":
        data = load_NGC5765b_spots(root, v_sys_obs=v_sys_obs)
    elif galaxy == "NGC6264":
        data = load_NGC6264_spots(root, v_sys_obs=v_sys_obs)
    elif galaxy == "NGC6323":
        data = load_NGC6323_spots(root, v_sys_obs=v_sys_obs)
    elif galaxy == "UGC3789":
        data = load_UGC3789_spots(root, v_sys_obs=v_sys_obs)
    elif galaxy == "NGC4258":
        data = load_NGC4258_spots(root, v_sys_obs=v_sys_obs)
    else:
        _all = list(_MRT_FILES) + ["NGC5765b", "NGC6264", "NGC6323",
                                   "UGC3789", "NGC4258"]
        if galaxy not in _MRT_FILES:
            raise ValueError(
                f"Unknown galaxy '{galaxy}'. Available: {_all}")

        if v_sys_obs is None:
            raise ValueError(f"v_sys_obs must be provided for {galaxy}.")

        fpath = join(root, _MRT_FILES[galaxy])
        fprint(f"loading maser spots from '{fpath}'.")

        with open(fpath) as f:
            lines = f.readlines()

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
        data = {
            "velocity": np.array([r["velocity"] for r in rows]),
            "x": np.array([r["x"] for r in rows]),
            "sigma_x": np.array([r["sigma_x"] for r in rows]),
            "y": np.array([r["y"] for r in rows]),
            "sigma_y": np.array([r["sigma_y"] for r in rows]),
            "a": np.array([r["a"] for r in rows]),
            "sigma_a": np.array([r["sigma_a"] for r in rows]),
            "accel_measured": np.array(
                [r["accel_measured"] for r in rows], dtype=bool),
            "n_spots": n,
            "galaxy_name": galaxy,
            "v_sys_obs": float(v_sys_obs),
        }

        n_accel = int(data["accel_measured"].sum())
        fprint(f"loaded {n} maser spots for {galaxy} "
               f"({n_accel} with measured acceleration).")

    # Classify spots: use spot_type from data if available, else k-means
    n = data["n_spots"]
    if "spot_type" in data:
        st = data["spot_type"]
        labels = np.array([{"b": 0, "s": 1, "r": 2}[t] for t in st])
        method = "spot_type"
    else:
        from scipy.cluster.vq import kmeans2
        centroids, lab = kmeans2(
            data["velocity"].astype(np.float64), 3, minit="++")
        order = np.argsort(centroids)
        remap = np.empty(3, dtype=int)
        remap[order] = np.arange(3)
        labels = remap[lab]
        method = "k-means on velocity"

    data["is_highvel"] = labels != 1

    # Per-spot phi bounds
    phi_lo = np.empty(n)
    phi_hi = np.empty(n)
    for k in range(n):
        if labels[k] == 0:    # blue (approaching)
            phi_lo[k], phi_hi[k] = np.pi, 2 * np.pi
        elif labels[k] == 1:  # systemic
            phi_lo[k], phi_hi[k] = -0.5 * np.pi, 0.5 * np.pi
        else:                 # red (receding)
            phi_lo[k], phi_hi[k] = 0.0, np.pi
    data["phi_lo"] = phi_lo
    data["phi_hi"] = phi_hi
    data["is_blue"] = (labels == 0)

    n_sys = int((labels == 1).sum())
    n_blue = int((labels == 0).sum())
    n_red = int((labels == 2).sum())
    fprint(f"classified spots: {n_sys} systemic, {n_blue} blue, "
           f"{n_red} red ({method}).")

    # Acceleration weights: 1/N for N spots sharing the same (a, sigma_a).
    # Spots sharing a clump-averaged acceleration should collectively
    # contribute one independent measurement, not N redundant copies.
    accel_weight = np.ones(n)
    am = data["accel_measured"]
    if am.any():
        a_vals = data["a"][am]
        sa_vals = data["sigma_a"][am]
        idx_meas = np.where(am)[0]
        # Round to avoid floating-point near-misses
        keys = np.round(a_vals, 8) + 1j * np.round(sa_vals, 8)
        unique_keys, counts = np.unique(keys, return_counts=True)
        n_shared = 0
        for uk, cnt in zip(unique_keys, counts):
            if cnt > 1:
                mask = keys == uk
                accel_weight[idx_meas[mask]] = 1.0 / cnt
                n_shared += cnt
        if n_shared > 0:
            fprint(f"accel_weight: {n_shared} spots share clump-averaged "
                   f"accelerations (weights < 1).")
    data["accel_weight"] = accel_weight

    return data
