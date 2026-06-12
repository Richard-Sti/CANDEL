"""Shared paths and plotting style for TRGBH0 paper-side scripts."""

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "output"
PAPER_DIR = ROOT.parent / "Papers" / "TRGBH0"
PAPER_FIGURE_DIR = PAPER_DIR / "Figures"
TRGBH0_RESULTS = ROOT / "results" / "TRGBH0_paper"
TRGBH0_TABLE_RESULTS = TRGBH0_RESULTS / "table"
FIGURE_DPI = 500

TRGBH0_COLOURS = [
    "#ef476f",
    "#473198",
    "#a8c256",
    "#5adbff",
    "#fe9000",
]

PAPER_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
}

SMALL_FIGURE_RC = {
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
}


def trgbh0_cmap(name="trgbh0"):
    return LinearSegmentedColormap.from_list(name, TRGBH0_COLOURS)


def set_paper_rc(extra=None):
    rc = SMALL_FIGURE_RC.copy()
    if extra is not None:
        rc.update(extra)
    plt.rcParams.update(rc)


@contextmanager
def paper_style(styles=("science", "no-latex"), extra_rc=None):
    with plt.style.context(styles):
        set_paper_rc(extra_rc)
        yield


def save_pdf_png(fig, out_pdf, *, dpi=FIGURE_DPI, bbox_inches="tight"):
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches=bbox_inches)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    return out_pdf, out_png


def save_figure(fig, outname, *, output_dir=OUTPUT_DIR, paper_figdir=None,
                dpi=None, bbox_inches=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / outname
    fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches)
    if paper_figdir is not None:
        paper_figdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_figdir / Path(outname).name, dpi=dpi,
                    bbox_inches=bbox_inches)
    return out
