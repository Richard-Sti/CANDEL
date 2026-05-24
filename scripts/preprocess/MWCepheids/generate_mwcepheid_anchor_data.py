#!/usr/bin/env python
# Copyright (C) 2026 Richard Stiskalek
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
"""Generate MW-Cepheid anchor products from the SH0ES FITS files."""
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2

import numpy as np
from astropy.io import fits


POPULATIONS = {
    "N4258": {
        "idx_start": 2150,
        "idx_end": 2593,
        "mu_anchor": 29.398,
    },
    "LMC": {
        "idx_start": 2648,
        "idx_end": 3130,
        "mu_anchor": 18.477,
    },
}

COL_LOGP = 41
COL_OH = 43
COL_GROUND_ZP = 45
B_W_FID = -3.285


def _default_repo_root():
    return Path(__file__).resolve().parents[3]


def _load_deltaq(path):
    if not path.exists():
        return {}

    values = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                values[parts[0]] = (float(parts[2]), float(parts[3]))
    return values


def _parse_table2(path, field_name):
    if not path.exists():
        return []

    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith(field_name):
                continue
            parts = [x.strip() for x in line.replace("\\\\", "").split("&")]
            if len(parts) >= 4:
                rows.append((parts[3].strip(), "HST" in parts[-1]))
    return rows


def _assign_q_values(out_dir, name, is_hst):
    """Return optional Q arrays when deltaq/table2 side files are available."""
    table2 = _parse_table2(out_dir / "table2.tex", name)
    deltaq = _load_deltaq(out_dir / f"{name.lower()}_hst.deltaq.dat")

    q = np.full(len(is_hst), np.nan)
    q_err = np.full(len(is_hst), np.nan)
    if not table2 or not deltaq:
        return q, q_err

    star_ids = [None] * len(is_hst)
    if len(table2) == len(is_hst):
        for i, (sid, _) in enumerate(table2):
            star_ids[i] = sid
    else:
        for flag in (True, False):
            fits_idx = [
                i for i in range(len(is_hst)) if bool(is_hst[i]) == flag]
            tab_ids = [sid for sid, is_hst_tab in table2
                       if is_hst_tab == flag]
            for sid, idx in zip(tab_ids, fits_idx):
                star_ids[idx] = sid

    for i, sid in enumerate(star_ids):
        if sid is not None and is_hst[i] and sid in deltaq:
            q[i], q_err[i] = deltaq[sid]
    return q, q_err


def generate(shoes_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    table2_src = shoes_dir / "table2.tex"
    if table2_src.exists():
        copy2(table2_src, out_dir / "table2.tex")

    y = fits.getdata(shoes_dir / "ally_shoes_ceph_topantheonwt6.0_112221.fits")
    lmat = fits.getdata(
        shoes_dir / "alll_shoes_ceph_topantheonwt6.0_112221.fits").T
    cov = fits.getdata(
        shoes_dir / "allc_shoes_ceph_topantheonwt6.0_112221.fits")

    y = np.asarray(y, dtype=np.float64)
    lmat = np.asarray(lmat, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    logp_all = lmat[:3130, COL_LOGP]
    mag_all = y[:3130].copy()
    mag_all += B_W_FID * (logp_all - 1.0)

    for name, meta in POPULATIONS.items():
        i0 = meta["idx_start"]
        i1 = meta["idx_end"]

        logp = logp_all[i0:i1]
        mw_h = mag_all[i0:i1] + meta["mu_anchor"]
        oh = lmat[i0:i1, COL_OH]
        cov_block = cov[i0:i1, i0:i1]
        mw_h_err = np.sqrt(np.diag(cov_block))
        is_hst = (lmat[i0:i1, COL_GROUND_ZP] == 0).astype(int)
        q, q_err = _assign_q_values(out_dir, name, is_hst.astype(bool))

        csv_path = out_dir / f"SH0ES_{name}_Cepheids.csv"
        header = "logP,mW_H,mW_H_err,OH,is_hst,Q,Q_err"
        data = np.column_stack([logp, mw_h, mw_h_err, oh, is_hst, q, q_err])
        np.savetxt(csv_path, data, delimiter=",", header=header,
                   comments="", fmt="%.8f")

        cov_path = out_dir / f"SH0ES_{name}_covmat.npy"
        np.save(cov_path, cov_block)

        print(f"{name}: wrote {len(logp)} Cepheids to {csv_path}")
        print(f"{name}: wrote covariance matrix to {cov_path}")


def main():
    repo = _default_repo_root()
    parser = ArgumentParser()
    parser.add_argument(
        "--shoes-dir", type=Path,
        default=repo / "data" / "SH0ES",
        help="Directory containing the raw SH0ES FITS files.")
    parser.add_argument(
        "--out-dir", type=Path,
        default=repo / "data" / "MWCepheids",
        help="Directory where MW-Cepheid anchor products are written.")
    args = parser.parse_args()

    generate(args.shoes_dir, args.out_dir)


if __name__ == "__main__":
    main()
