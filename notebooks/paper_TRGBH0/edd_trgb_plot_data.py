"""Shared data helpers for EDD TRGB paper summary plots."""
import csv
import math
from pathlib import Path
import shutil

import numpy as np


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
DATA_FILE = ROOT / "data" / "EDD_TRGB" / "EDD_TRGB.txt"
OUTDIR = ROOT / "notebooks" / "paper_TRGBH0" / "output"

DROP_NAMES = {"LMC", "SMC", "NGC4258", "NGC4258-DF6"}
PAPER_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
}


def _float_or_nan(value):
    try:
        return float(value)
    except ValueError:
        return math.nan


def _data_rows():
    with DATA_FILE.open(newline="") as handle:
        next(handle)
        yield from csv.DictReader(handle)


def load_edd_trgb_plot_data(include_sky=True):
    """Load the EDD/Rizzi-standardized TRGB sample used by the paper plots."""
    rows = []
    for row in _data_rows():
        try:
            int(row["pgc"])
        except ValueError:
            continue

        name = row["Name"]
        vcmb = _float_or_nan(row["Vcmb"])
        t814 = _float_or_nan(row["T814"])
        a814 = _float_or_nan(row["A_814"])
        m_trgb = _float_or_nan(row["M_TRGB"])

        if name in DROP_NAMES:
            continue
        if not math.isfinite(t814 - a814):
            continue
        if not math.isfinite(m_trgb):
            continue
        if abs(vcmb) >= 9999:
            continue

        rows.append((
            _float_or_nan(row["RAJ"]),
            _float_or_nan(row["DeJ"]),
            vcmb,
            t814 - a814,
        ))

    data = np.asarray(rows, dtype=float)
    out = {"czcmb": data[:, 2], "mag": data[:, 3]}
    if include_sky:
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        coords = SkyCoord(ra=data[:, 0] * u.deg, dec=data[:, 1] * u.deg)
        gal = coords.galactic
        out["ell"] = np.asarray(gal.l.deg)
        out["b"] = np.asarray(gal.b.deg)
    return out


def save_figure(fig, outname, paper_figdir=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / outname
    fig.savefig(out)
    if paper_figdir is not None:
        paper_figdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, paper_figdir / outname)
    return out
