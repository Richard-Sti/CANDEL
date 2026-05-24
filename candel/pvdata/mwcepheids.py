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
"""Milky Way Cepheid calibration data loaders."""
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re

import jax.numpy as jnp
import numpy as np

from ..util import fprint, get_nested, load_config as load_candel_config

logger = logging.getLogger(__name__)

# Offset to convert [Fe/H] to [O/H]: [O/H] = [Fe/H] + FEH_TO_OH.
# MW Cepheids (R21 Table 1) use [Fe/H]; SH0ES anchors use [O/H].
FEH_TO_OH = 0.06


def load_data(filepath, which_subset=None):
    """Load Cepheid data directly from a file path.

    Convenience function for notebook use without needing full config.

    Parameters
    ----------
    filepath : str
        Path to data file (.csv or .dat format).
    which_subset : str, optional
        "C22" or "C27" to subset, or None for all stars.

    Returns
    -------
    CepheidData
        Loaded data container.
    """
    config = {"data": {"R21_MW": filepath}}
    if which_subset is not None:
        config["data"]["which_subset"] = which_subset
    return CepheidData(config)


class CepheidData:
    """Load and store Riess (2021) MW Cepheid data as JAX arrays.

    Supports two formats:
    - CSV: Full table with star names, HST photometry, campaign info
    - DAT: Compact format with columns:
           logP, G_mag, pi_EDR3, pi_EDR3_err, mW_H, mW_H_err, FeH, FeH_err

    Note: R21 Table 1 parallaxes already have the Lindegren Z5 correction
    applied, so no additional correction is needed.

    Parameters
    ----------
    config : dict
        Configuration dictionary (output of ``load_config``).
    """

    def __init__(self, config):
        fpath = config["data"]["R21_MW"]

        if fpath.endswith(".dat"):
            self._load_dat(fpath)
        else:
            self._load_csv(fpath)

        # Track campaign (None = mixed, "C22" or "C27" = single campaign)
        self.campaign = None

        # Optionally subset to C22 or C27 only
        which_subset = config["data"].get("which_subset", None)
        if which_subset is not None:
            self._apply_subset(which_subset)
            self.campaign = which_subset

        # Optionally exclude named stars (e.g. R21 outliers S-VUL, SV-VUL)
        exclude = config["data"].get("exclude_stars", [])
        if exclude:
            mask = ~np.isin(self.names, exclude)
            n_excluded = self.n_stars - mask.sum()
            if n_excluded > 0:
                dropped = self.names[~mask]
                logger.info(
                    f"Excluding {n_excluded} stars: "
                    f"{', '.join(dropped)}")
                self._apply_mask(mask)
            else:
                logger.warning(
                    f"exclude_stars={exclude} matched no stars")

        # Optional parallax cut for C22 stars only
        data_c22 = config.get("data", {}).get("C22", {})
        pi_cut_c22 = data_c22.get("pi_cut", None)
        if pi_cut_c22 is not None:
            self._apply_pi_cut_c22(pi_cut_c22)

        # Optional parallax cut for C27 stars only
        data_c27 = config.get("data", {}).get("C27", {})
        pi_cut_c27 = data_c27.get("pi_cut", None)
        if pi_cut_c27 is not None:
            self._apply_pi_cut_c27(pi_cut_c27)

        # Print final count after all cuts
        if self.campaign is not None:
            logger.info(
                f"Final sample ({self.campaign}): {self.n_stars} stars")
        else:
            n22 = int(self.is_c22.sum())
            n27 = int(self.is_c27.sum())
            logger.info(f"Final sample: {self.n_stars} stars "
                        f"({n22} C22 + {n27} C27)")

        # Load anchor galaxy data
        self.anchor_data = load_anchor_data(config)

    def _load_dat(self, fpath):
        """Load compact .dat format (66 stars, no names/campaigns)."""
        # Columns: logP, G_mag, pi_EDR3, pi_EDR3_err, mW_H, mW_H_err,
        #          FeH, FeH_err
        raw = np.loadtxt(fpath)
        logger.info(f"Loaded {len(raw)} stars from {fpath}")

        self.names = np.array([f"star_{i}" for i in range(len(raw))])
        self.n_stars = len(raw)

        self.logP = jnp.array(raw[:, 0])
        self.G_mag = jnp.array(raw[:, 1])
        self.pi_EDR3 = jnp.array(raw[:, 2])
        self.pi_EDR3_err = jnp.array(raw[:, 3])
        self.mW_H = jnp.array(raw[:, 4])
        self.mW_H_err = jnp.array(raw[:, 5])
        self.FeH = jnp.array(raw[:, 6])
        self.FeH_err = jnp.array(raw[:, 7])
        self.OH = self.FeH + FEH_TO_OH

        # No campaign info in .dat format
        self.is_c22 = jnp.zeros(self.n_stars, dtype=bool)
        self.is_c27 = jnp.zeros(self.n_stars, dtype=bool)

        # No HST photometry in .dat format
        self.F555W = None
        self.F555W_err = None

        # No Q index in .dat format
        self._Q_raw = None
        self.Q_err = None

        # No Galactic coordinates in .dat format
        self.ell = None
        self.b = None

    def _load_csv(self, fpath):
        """Load full CSV format with all metadata."""
        raw = np.genfromtxt(fpath, delimiter=",", names=True, dtype=None,
                            encoding="utf-8")

        # Parse parallaxes first to identify stars with EDR3
        pi_edr3 = raw["pi_EDR3"]
        pi_edr3_err = raw["pi_EDR3_err"]
        if pi_edr3.dtype.kind in ('U', 'S', 'O'):
            pi_edr3 = np.array(
                [float(x) if x.strip() != '' else np.nan for x in pi_edr3])
            pi_edr3_err = np.array(
                [float(x) if x.strip() != '' else np.nan
                 for x in pi_edr3_err])
        else:
            pi_edr3 = np.array(pi_edr3, dtype=float)
            pi_edr3_err = np.array(pi_edr3_err, dtype=float)

        # Keep only stars with EDR3 parallaxes
        keep = ~np.isnan(pi_edr3)
        names = np.array([s.strip() for s in raw["Cepheid"]])
        dropped = names[~keep]
        if len(dropped) > 0:
            logger.info(
                f"Dropping {len(dropped)} stars without EDR3: "
                f"{', '.join(dropped)}")
        raw = raw[keep]
        pi_edr3 = pi_edr3[keep]
        pi_edr3_err = pi_edr3_err[keep]

        self.names = np.array([s.strip() for s in raw["Cepheid"]])
        self.n_stars = len(self.names)
        logger.info(f"Loaded {self.n_stars} stars from {fpath}")

        # Observables
        self.logP = jnp.array(raw["logP"])
        self.mW_H = jnp.array(raw["mW_H"])
        self.mW_H_err = jnp.array(raw["mW_H_err"])
        self.FeH = jnp.array(raw["FeH"])
        self.OH = self.FeH + FEH_TO_OH

        # Not in CSV format
        self.FeH_err = None
        self.G_mag = None

        # HST photometry
        self.F555W = jnp.array(raw["F555W"])
        self.F555W_err = jnp.array(raw["F555W_err"])

        # Gaia EDR3 parallaxes (already Z5-corrected in R21 Table 1)
        self.pi_EDR3 = jnp.array(pi_edr3)
        self.pi_EDR3_err = jnp.array(pi_edr3_err)

        # Reddening-free Q index
        if "Q" in raw.dtype.names:
            self._Q_raw = jnp.array(raw["Q"])
            self.Q_err = jnp.array(raw["Q_err"])
        else:
            self._Q_raw = None
            self.Q_err = None

        # Campaign masks
        campaigns = np.array([s.strip() for s in raw["set"]])
        self.is_c22 = jnp.array(campaigns == "Cycle22")
        self.is_c27 = jnp.array(campaigns == "Cycle27")

        # Galactic coordinates (if available)
        if "ell" in raw.dtype.names and "b" in raw.dtype.names:
            self.ell = jnp.array(raw["ell"])
            self.b = jnp.array(raw["b"])
        else:
            self.ell = None
            self.b = None

    def _apply_mask(self, mask):
        """Apply a boolean mask to all data arrays.

        Parameters
        ----------
        mask : array of bool
            True for stars to keep.
        """
        mask = np.asarray(mask)
        self.names = self.names[mask]
        self.n_stars = len(self.names)

        self.logP = self.logP[mask]
        self.mW_H = self.mW_H[mask]
        self.mW_H_err = self.mW_H_err[mask]
        self.FeH = self.FeH[mask]
        self.OH = self.OH[mask]
        self.pi_EDR3 = self.pi_EDR3[mask]
        self.pi_EDR3_err = self.pi_EDR3_err[mask]

        if self.F555W is not None:
            self.F555W = self.F555W[mask]
            self.F555W_err = self.F555W_err[mask]

        if self._Q_raw is not None:
            self._Q_raw = self._Q_raw[mask]
            self.Q_err = self.Q_err[mask]

        if self.G_mag is not None:
            self.G_mag = self.G_mag[mask]
        if self.FeH_err is not None:
            self.FeH_err = self.FeH_err[mask]

        if self.ell is not None:
            self.ell = self.ell[mask]
            self.b = self.b[mask]

        self.is_c22 = self.is_c22[mask]
        self.is_c27 = self.is_c27[mask]

    def _apply_subset(self, which_subset):
        """Filter data to keep only C22 or C27 stars."""
        if which_subset == "C22":
            mask = np.asarray(self.is_c22)
        elif which_subset == "C27":
            mask = np.asarray(self.is_c27)
        else:
            raise ValueError(f"Unknown subset: {which_subset}. "
                             f"Use 'C22', 'C27', or None.")

        logger.info(f"Subsetting to {which_subset}: {mask.sum()} stars")
        self._apply_mask(mask)

    def _apply_pi_cut_c22(self, pi_cut):
        """Apply a parallax cut to C22 stars only (keep all C27 stars)."""
        is_c22 = np.asarray(self.is_c22)
        pi = np.asarray(self.pi_EDR3)

        mask = ~is_c22 | (pi > pi_cut)
        n_c22_kept = (is_c22 & mask).sum()
        n_c22_total = is_c22.sum()

        if n_c22_total > 0:
            logger.info(f"C22 pi > {pi_cut}: "
                        f"{n_c22_kept}/{n_c22_total} C22 stars kept")
        if n_c22_kept == n_c22_total:
            return
        self._apply_mask(mask)

    def _apply_pi_cut_c27(self, pi_cut):
        """Apply a parallax cut to C27 stars only (keep all C22 stars)."""
        is_c27 = np.asarray(self.is_c27)
        pi = np.asarray(self.pi_EDR3)

        mask = ~is_c27 | (pi > pi_cut)
        n_c27_kept = (is_c27 & mask).sum()
        n_c27_total = is_c27.sum()

        if n_c27_total > 0:
            logger.info(f"C27 pi > {pi_cut}: "
                        f"{n_c27_kept}/{n_c27_total} C27 stars kept")
        if n_c27_kept == n_c27_total:
            return
        self._apply_mask(mask)

    @property
    def Q(self):
        """Mean-subtracted reddening-free Q index."""
        if self._Q_raw is None:
            return None
        return self._Q_raw - jnp.mean(self._Q_raw)

    def __repr__(self):
        if self.campaign is not None:
            return f"CepheidData({self.campaign}: {self.n_stars} stars)"
        n22 = int(self.is_c22.sum())
        n27 = int(self.is_c27.sum())
        return f"CepheidData({self.n_stars} stars: {n22} C22 + {n27} C27)"

    def keys(self):
        """Return list of data attribute names."""
        return [k for k in self.__dict__ if not k.startswith("_")]

    def __getitem__(self, key):
        """Access data attributes via indexing."""
        return getattr(self, key)

    @classmethod
    def from_arrays(cls, logP, mW_H, mW_H_err, OH, pi_EDR3, pi_EDR3_err,
                    ell, b, campaign):
        """Construct CepheidData from arrays without file I/O.

        Useful for mock/synthetic data tests.
        """
        obj = object.__new__(cls)
        n = len(logP)
        obj.n_stars = n
        obj.names = np.array([f"mock_{i}" for i in range(n)])
        obj.campaign = campaign
        obj.logP = jnp.asarray(logP)
        obj.mW_H = jnp.asarray(mW_H)
        obj.mW_H_err = jnp.asarray(mW_H_err)
        obj.OH = jnp.asarray(OH)
        obj.FeH = obj.OH - FEH_TO_OH
        obj.pi_EDR3 = jnp.asarray(pi_EDR3)
        obj.pi_EDR3_err = jnp.asarray(pi_EDR3_err)
        obj.ell = jnp.asarray(ell)
        obj.b = jnp.asarray(b)
        obj.is_c22 = (jnp.ones(n, dtype=bool) if campaign == "C22"
                      else jnp.zeros(n, dtype=bool))
        obj.is_c27 = (jnp.ones(n, dtype=bool) if campaign == "C27"
                      else jnp.zeros(n, dtype=bool))
        obj.F555W = None
        obj.F555W_err = None
        obj._Q_raw = None
        obj.Q_err = None
        obj.G_mag = None
        obj.FeH_err = None
        obj.anchor_data = {}
        return obj

    def split_by_campaign(self):
        """Split data into separate C22 and C27 datasets.

        Returns
        -------
        dict
            {"C22": CepheidData, "C27": CepheidData}
            Only includes campaigns with stars present.
        """
        if self.campaign is not None:
            raise ValueError(
                f"Data already subset to {self.campaign}, cannot split.")

        result = {}

        for campaign, mask_attr in [("C22", "is_c22"), ("C27", "is_c27")]:
            mask = np.asarray(getattr(self, mask_attr))
            if mask.sum() == 0:
                continue

            # Create new data object without calling __init__
            subset = object.__new__(CepheidData)
            subset.campaign = campaign
            subset.anchor_data = {}
            subset.names = self.names[mask]
            subset.n_stars = len(subset.names)

            subset.logP = self.logP[mask]
            subset.mW_H = self.mW_H[mask]
            subset.mW_H_err = self.mW_H_err[mask]
            subset.FeH = self.FeH[mask]
            subset.OH = self.OH[mask]
            subset.pi_EDR3 = self.pi_EDR3[mask]
            subset.pi_EDR3_err = self.pi_EDR3_err[mask]

            subset.is_c22 = self.is_c22[mask]
            subset.is_c27 = self.is_c27[mask]

            # Optional attributes (None-safe)
            for attr in ("F555W", "F555W_err", "_Q_raw", "Q_err",
                         "G_mag", "FeH_err", "ell", "b"):
                val = getattr(self, attr)
                setattr(subset, attr, val[mask] if val is not None else None)

            result[campaign] = subset

        return result


def _parse_table2_ids(table2_path, field_name, n_expected):
    """Parse star IDs from SH0ES table2.tex for a given field.

    Returns an array of integer star IDs (one per row in the field),
    in the same order as table2 (which matches the extracted CSV).

    Table2 IDs have a field/visit prefix (e.g. ``A2``, ``B3``,
    ``F1-``) followed by the numeric star ID. The prefix is a single
    letter plus a digit identifying the SH0ES pointing.
    """
    # Config names may differ from table2 field names
    _field_aliases = {"NGC4258": "N4258"}
    tab_field = _field_aliases.get(field_name, field_name)

    # Known two-character field prefixes (letter + visit digit).
    # Hyphens are stripped before matching.
    _prefixes = ("A1", "A2", "B2", "B3", "C1", "C2", "F1", "F3")

    ids = []
    with open(table2_path) as f:
        for line in f:
            if not line.startswith(tab_field):
                continue
            parts = [x.strip() for x in line.replace("\\\\", "").split("&")]
            sid = parts[3].strip().replace("-", "")

            num = None
            for p in _prefixes:
                if sid.startswith(p):
                    num = int(sid[len(p):])
                    break
            if num is None:
                # Short IDs like C084, F37 — single letter + digits
                m = re.match(r"[A-Z](\d+)$", sid)
                num = int(m.group(1)) if m else -1

            ids.append(num)

    if len(ids) != n_expected:
        raise ValueError(
            f"table2.tex has {len(ids)} {tab_field} rows, "
            f"expected {n_expected}")
    return np.array(ids)


def _load_jwst_Q(jwst_files, n_csv, csv_star_ids, data_dir=""):
    """Load JWST Q index files and match to anchor CSV by star ID.

    Each JWST file has columns (whitespace-separated):
        IS  logP  Q  Q_err  [12+log(O/H)]

    Stars appearing in multiple files (same IS) are deduplicated,
    keeping the measurement with smaller Q_err. Matching is done
    on integer star IDs from table2.tex.

    Returns per-CSV-row arrays (NaN where no ID match exists).
    """
    # Concatenate all JWST files
    all_rows = []
    for fpath in jwst_files:
        if not os.path.isabs(fpath):
            fpath = os.path.join(data_dir, fpath)
        all_rows.append(np.loadtxt(fpath))
    jwst = np.vstack(all_rows)

    # Deduplicate by star ID (column 0), keeping smallest Q_err
    id_to_best = {}
    for row in jwst:
        sid = int(row[0])
        if sid not in id_to_best or row[3] < id_to_best[sid][3]:
            id_to_best[sid] = row
    jwst_lookup = {int(r[0]): r for r in id_to_best.values()}
    logger.info(f"  JWST Q: {len(jwst_lookup)} unique stars from "
                f"{len(jwst_files)} files")

    # Match by star ID
    Q_jwst = np.full(n_csv, np.nan)
    Q_err_jwst = np.full(n_csv, np.nan)
    n_matched = 0
    for i, sid in enumerate(csv_star_ids):
        if sid in jwst_lookup:
            Q_jwst[i] = jwst_lookup[sid][2]
            Q_err_jwst[i] = jwst_lookup[sid][3]
            n_matched += 1

    logger.info(f"  JWST Q: matched {n_matched}/{n_csv} anchor stars by ID")
    return Q_jwst, Q_err_jwst


class AnchorCepheidData:
    """Cepheid data for an anchor galaxy with a known geometric distance.

    Loads data extracted from the SH0ES FITS files (via
    ``scripts/extract_SH0ES_anchors.py``).

    Parameters
    ----------
    name : str
        Anchor name (e.g. ``"N4258"``, ``"LMC"``).
    csv_path : str
        Path to the CSV file (columns: logP, mW_H, mW_H_err, OH).
    covmat_path : str
        Path to the ``.npy`` covariance matrix file.
    mu_anchor : float
        Geometric distance modulus [mag].
    e_mu_anchor : float
        Uncertainty on the distance modulus [mag].
    """

    def __init__(self, name, csv_path, covmat_path,
                 mu_anchor, e_mu_anchor, mu_min=None, mu_max=None,
                 logP_min=None, scatter_subtract=None, hst_only=False,
                 jwst_Q_files=None, jwst_id_table=None, data_dir=""):
        self.name = name
        self.mu_anchor = mu_anchor
        self.e_mu_anchor = e_mu_anchor
        self.logP_min = logP_min
        self.mu_min = (mu_min if mu_min is not None
                       else mu_anchor - 10 * e_mu_anchor)
        self.mu_max = (mu_max if mu_max is not None
                       else mu_anchor + 10 * e_mu_anchor)

        raw = np.genfromtxt(csv_path, delimiter=",", names=True)
        n_raw_before_filter = len(raw)

        # Filter to HST-only photometry if requested
        if hst_only:
            if "is_hst" not in raw.dtype.names:
                raise ValueError(
                    f"hst_only=True but {csv_path} has no 'is_hst' column. "
                    f"Re-run extract_SH0ES_anchors.py to regenerate.")
            mask = raw["is_hst"].astype(bool)
            n_before = len(raw)
            raw = raw[mask]

            # Also filter the covariance matrix
            covmat_full = np.load(covmat_path)
            idx = np.where(mask)[0]
            covmat_filtered = covmat_full[np.ix_(idx, idx)]
        else:
            covmat_filtered = None

        self.n_stars = len(raw)
        if hst_only:
            logger.info(
                f"Loaded {self.n_stars}/{n_before} HST-only {name} "
                f"Cepheids from {csv_path}")
        else:
            logger.info(
                f"Loaded {self.n_stars} {name} Cepheids from {csv_path}")

        self.logP = jnp.array(raw["logP"])
        self.mW_H = jnp.array(raw["mW_H"])
        self.mW_H_err = jnp.array(raw["mW_H_err"])
        self.OH = jnp.array(raw["OH"])

        # Q index (optional — present only for anchors with matched Q data).
        # Mean-subtracted to match the MW CepheidData.Q convention.
        if "Q" in raw.dtype.names:
            Q_raw = raw["Q"]
            Q_err_raw = raw["Q_err"]
            has_Q = np.isfinite(Q_raw)
            if np.all(has_Q):
                Q_arr = jnp.array(Q_raw)
                Q_mean = float(jnp.mean(Q_arr))
                self.Q = Q_arr - Q_mean
                self.Q_err = jnp.array(Q_err_raw)
                logger.info(
                    f"  Q index: subtracted mean {Q_mean:.4f} "
                    f"from {self.n_stars} stars")
            else:
                self._Q_raw = Q_raw
                self._Q_err_raw = Q_err_raw
                self._Q_valid_mask = has_Q
                self.Q = None
                self.Q_err = None
                logger.info(
                    f"  {np.sum(~has_Q)} stars with missing Q "
                    f"(will be filtered if use_Q=true)")
        else:
            self.Q = None
            self.Q_err = None

        # Optionally replace Q with JWST measurements
        if jwst_Q_files:
            if jwst_id_table is None:
                raise ValueError(
                    "jwst_id_table (path to table2.tex) is required "
                    "when using jwst_Q_files")
            if not os.path.isabs(jwst_id_table):
                jwst_id_table = os.path.join(data_dir, jwst_id_table)
            csv_star_ids = _parse_table2_ids(
                jwst_id_table, name, n_raw_before_filter)
            if hst_only:
                csv_star_ids = csv_star_ids[mask]
            Q_jwst, Q_err_jwst = _load_jwst_Q(
                jwst_Q_files, self.n_stars, csv_star_ids, data_dir)
            has_jwst = np.isfinite(Q_jwst)
            n_replaced = int(has_jwst.sum())
            if n_replaced > 0:
                # Start from existing Q (or NaN if none)
                if "Q" in raw.dtype.names:
                    Q_merged = np.array(raw["Q"], dtype=float)
                    Q_err_merged = np.array(raw["Q_err"], dtype=float)
                else:
                    Q_merged = np.full(self.n_stars, np.nan)
                    Q_err_merged = np.full(self.n_stars, np.nan)
                Q_merged[has_jwst] = Q_jwst[has_jwst]
                Q_err_merged[has_jwst] = Q_err_jwst[has_jwst]

                has_Q = np.isfinite(Q_merged)
                if np.all(has_Q):
                    Q_arr = jnp.array(Q_merged)
                    Q_mean = float(jnp.mean(Q_arr))
                    self.Q = Q_arr - Q_mean
                    self.Q_err = jnp.array(Q_err_merged)
                    logger.info(
                        f"  JWST Q replacement: {n_replaced} stars, "
                        f"mean-subtracted (mean={Q_mean:.4f})")
                else:
                    self._Q_raw = Q_merged
                    self._Q_err_raw = Q_err_merged
                    self._Q_valid_mask = has_Q
                    self.Q = None
                    self.Q_err = None
                    logger.info(
                        f"  JWST Q replacement: {n_replaced} stars, "
                        f"{int((~has_Q).sum())} still missing")

        # Full covariance matrix
        if covmat_filtered is not None:
            self.covmat = jnp.array(covmat_filtered)
        else:
            self.covmat = jnp.array(np.load(covmat_path))
        if self.covmat.shape != (self.n_stars, self.n_stars):
            raise ValueError(
                f"Covariance shape {self.covmat.shape} does not match "
                f"n_stars={self.n_stars}")

        if scatter_subtract is not None and scatter_subtract > 0:
            self.covmat -= scatter_subtract**2 * jnp.eye(
                self.n_stars)
            logger.info(
                f"  Subtracted {scatter_subtract:.4f} mag "
                "intrinsic scatter from covariance")

        # Precompute eigendecomposition for efficient diagonal-shift
        # updates: covmat + alpha*I has eigenvalues (eig_lam + alpha)
        self.eig_lam, self.eig_Q = jnp.linalg.eigh(self.covmat)
        logger.info(f"  Covariance matrix: {self.covmat.shape}")

    @classmethod
    def from_config(cls, name, anchor_cfg, data_dir="", scatter_subtract=None):
        """Create from a config dict.

        Parameters
        ----------
        name : str
            Anchor name (e.g. ``"N4258"``).
        anchor_cfg : dict
            Must contain ``csv``, ``covmat``, ``mu``, ``e_mu``.
        data_dir : str, optional
            Base directory for resolving relative paths.

        Returns
        -------
        AnchorCepheidData
        """
        csv_path = anchor_cfg["csv"]
        covmat_path = anchor_cfg["covmat"]

        if not os.path.isabs(csv_path):
            csv_path = os.path.join(data_dir, csv_path)
        if not os.path.isabs(covmat_path):
            covmat_path = os.path.join(data_dir, covmat_path)

        return cls(
            name=name,
            csv_path=csv_path,
            covmat_path=covmat_path,
            mu_anchor=anchor_cfg["mu"],
            e_mu_anchor=anchor_cfg["e_mu"],
            mu_min=anchor_cfg.get("mu_min"),
            mu_max=anchor_cfg.get("mu_max"),
            logP_min=anchor_cfg.get("logP_min"),
            scatter_subtract=scatter_subtract,
            hst_only=anchor_cfg.get("hst_only", False),
            jwst_Q_files=anchor_cfg.get("jwst_Q_files"),
            jwst_id_table=anchor_cfg.get("jwst_id_table"),
            data_dir=data_dir,
        )

    def filter_Q_valid(self):
        """Return a copy keeping only stars with finite Q values.

        Adjusts all per-star arrays and the covariance matrix.
        """
        if self.Q is not None:
            return self  # already all valid

        mask = self._Q_valid_mask
        idx = np.where(mask)[0]
        n_drop = int(np.sum(~mask))

        new = object.__new__(type(self))
        new.name = self.name
        new.mu_anchor = self.mu_anchor
        new.e_mu_anchor = self.e_mu_anchor
        new.logP_min = self.logP_min
        new.mu_min = self.mu_min
        new.mu_max = self.mu_max

        new.logP = self.logP[idx]
        new.mW_H = self.mW_H[idx]
        new.mW_H_err = self.mW_H_err[idx]
        new.OH = self.OH[idx]
        new.n_stars = len(idx)
        Q_valid = jnp.array(self._Q_raw[mask])
        Q_mean = float(jnp.mean(Q_valid))
        new.Q = Q_valid - Q_mean
        new.Q_err = jnp.array(self._Q_err_raw[mask])
        logger.info(
            f"  Q index: subtracted mean {Q_mean:.4f} "
            f"from {new.n_stars} stars")

        new.covmat = self.covmat[jnp.ix_(idx, idx)]
        new.eig_lam, new.eig_Q = jnp.linalg.eigh(new.covmat)

        logger.info(
            f"  Filtered {self.name}: dropped {n_drop} stars "
            f"with missing Q ({new.n_stars} remaining)")
        return new

    def __repr__(self):
        return (f"AnchorCepheidData({self.name}: {self.n_stars} Cepheids, "
                f"mu={self.mu_anchor} +/- {self.e_mu_anchor})")


def load_anchor_data(config):
    """Load all anchor galaxy Cepheid datasets from config.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict
        ``{name: AnchorCepheidData}`` for each anchor in
        ``config["data"]["anchors"]``.
    """
    anchors_cfg = config.get("data", {}).get("anchors", {})
    if not anchors_cfg:
        return {}

    # Only load anchors that are enabled in the model config
    enabled = config.get("model", {}).get("anchors", [])
    if not enabled:
        return {}

    data_dir = config.get("local", {}).get("paths", {}).get("data", "")
    scatter_subtract = config.get("model", {}).get("anchor_scatter_correction")

    result = {}
    for name, acfg in anchors_cfg.items():
        if name not in enabled:
            continue
        result[name] = AnchorCepheidData.from_config(
            name, acfg, data_dir,
            scatter_subtract=scatter_subtract)

    return result


@dataclass
class MWCepheidDataset:
    """Container passed to the standalone MW Cepheid model."""

    campaigns: dict
    anchor_data: dict


def _resolve_path(root, value):
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def to_mwcepheids_config(config):
    """Translate CANDEL config shape to the standalone MW-Cepheid shape."""
    mw_cfg = get_nested(config, "io/MWCepheids")
    if mw_cfg is None:
        raise KeyError("Config is missing `io.MWCepheids`.")

    root = Path(mw_cfg["root"])
    data = {}
    r21 = mw_cfg.get("R21_MW")
    data["R21_MW"] = str(_resolve_path(root, r21)) if r21 is not None else None

    for key in ("exclude_stars", "which_subset"):
        if key in mw_cfg:
            data[key] = mw_cfg[key]
    for campaign in ("C22", "C27"):
        if campaign in mw_cfg:
            data[campaign] = mw_cfg[campaign]

    anchors = {}
    for name, acfg in mw_cfg.get("anchors", {}).items():
        anchors[name] = dict(acfg)
    data["anchors"] = anchors

    return {
        "data": data,
        "model": config["model"],
        "inference": config.get("inference", {}),
        "io": config.get("io", {}),
        "local": {"paths": {
            "data": str(root),
            "results": config.get("root_results", "."),
        }},
    }


def _load_campaigns(config):
    r21 = get_nested(config, "data/R21_MW")
    if r21 is None:
        fprint("MWCepheids: no R21_MW file configured; using anchors only.")
        return {}

    cepheids = CepheidData(config)
    if cepheids.campaign is not None:
        return {cepheids.campaign: cepheids}
    campaigns = cepheids.split_by_campaign()
    fprint("MWCepheids campaigns: "
           + ", ".join(f"{k}={v.n_stars}" for k, v in campaigns.items()))
    return campaigns


def load_MWCepheids_from_config(config_path):
    """Load MW-Cepheid calibration data from a CANDEL config path."""
    candel_config = load_candel_config(config_path, replace_los_prior=False)
    mw_config = to_mwcepheids_config(candel_config)

    campaigns = _load_campaigns(mw_config)
    anchors = load_anchor_data(mw_config)
    for name, anchor in anchors.items():
        fprint(f"MWCepheids anchor {name}: {anchor.n_stars} stars")

    if not campaigns and not anchors:
        raise ValueError("No MW Cepheid campaigns or anchors were loaded.")
    return MWCepheidDataset(campaigns, anchors)
