#!/usr/bin/env python
"""Compare fixed- and free-beta Student-t TRGBH0 posteriors."""
from argparse import ArgumentParser
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import numpy as np

from candel.plotting.corner import plot_corner_getdist
from trgbh0_plot_style import OUTPUT_DIR, TRGBH0_COLOURS, TRGBH0_TABLE_RESULTS


RESULTS = TRGBH0_TABLE_RESULTS
OUTDIR = OUTPUT_DIR

FIXED_BETA = (
    RESULTS
    / "EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_ManticoreLocalCOLA_main.hdf5"
)
FREE_BETA = (
    RESULTS
    / "EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_ManticoreLocalCOLA_beta_free_main.hdf5"
)

CORNER_KEYS = [
    "H0",
    "beta",
    "sigma_v",
    "nu_cz",
    "Vext_mag",
    "Vext_ell",
    "Vext_b",
    "alpha_low",
    "alpha_high",
]


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTDIR,
        help="Directory for the generated corner plot.",
    )
    return parser.parse_args()


def load_samples(path):
    """Load posterior samples from a CANDEL inference HDF5 file."""
    with h5py.File(path, "r") as handle:
        group = handle["samples"]
        return {key: group[key][...] for key in group.keys()}


def main():
    args = parse_args()
    for path in (FIXED_BETA, FREE_BETA):
        if not path.exists():
            raise FileNotFoundError(f"Missing posterior: {path}")

    fixed_beta_samples = load_samples(FIXED_BETA)
    free_beta_samples = load_samples(FREE_BETA)
    fixed_beta_samples["beta"] = np.ones_like(fixed_beta_samples["H0"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_corner_getdist(
        [fixed_beta_samples, free_beta_samples],
        labels=[
            r"Student-$t$, fixed $\beta=1$",
            r"Student-$t$, free $\beta$",
        ],
        cols=[TRGBH0_COLOURS[1], TRGBH0_COLOURS[0]],
        keys=CORNER_KEYS,
        filled=True,
        fontsize=13,
        legend_fontsize=15,
        ranges={
            "beta": [0.85, 1.15],
            "sigma_v": [35.0, 85.0],
            "nu_cz": [1.2, 4.4],
            "Vext_mag": [300.0, 460.0],
            "Vext_ell": [275.0, 305.0],
            "Vext_b": [-10.0, 5.0],
            "alpha_low": [0.0, None],
            "alpha_high": [0.0, None],
        },
        truths=[{
            "dict": {"beta": 1.0},
            "color": "black",
            "linestyle": ":",
        }],
        filename=str(
            args.output_dir
            / "trgbh0_student_t_fixed_vs_free_beta_corner.pdf"
        ),
        show_fig=False,
    )


if __name__ == "__main__":
    main()
