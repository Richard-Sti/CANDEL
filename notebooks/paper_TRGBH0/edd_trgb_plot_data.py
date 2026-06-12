"""Shared data helpers for EDD TRGB paper summary plots."""
import csv
import math
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = (
    SCRIPT_DIR if SCRIPT_DIR.name == "paper_TRGBH0"
    else next(path for path in SCRIPT_DIR.parents
              if path.name == "paper_TRGBH0")
)
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from trgbh0_plot_style import OUTPUT_DIR as OUTDIR
from trgbh0_plot_style import PAPER_RC, ROOT
from trgbh0_plot_style import save_figure as save_figure_common


DATA_FILE = ROOT / "data" / "EDD_TRGB" / "EDD_TRGB.txt"

DROP_NAMES = {"LMC", "SMC", "NGC4258", "NGC4258-DF6"}


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
    return save_figure_common(fig, outname, output_dir=OUTDIR,
                              paper_figdir=paper_figdir)
