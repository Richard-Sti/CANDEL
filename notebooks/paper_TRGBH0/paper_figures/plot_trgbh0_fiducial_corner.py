#!/usr/bin/env python
"""Plot the fiducial TRGBH0 posterior corner."""
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from candel.plotting.corner import plot_corner_from_hdf5
from trgbh0_plot_style import OUTPUT_DIR, TRGBH0_TABLE_RESULTS


RESULTS = TRGBH0_TABLE_RESULTS
OUTDIR = OUTPUT_DIR

FIDUCIAL = (
    RESULTS
    / "EDD_TRGB_rhoSmoothR4_MAS-PCS_sel-TRGB_magnitude_ManticoreLocalCOLA_main.hdf5"
)

CORNER_KEYS = [
    "H0",
    "M_TRGB",
    "c_bar",
    "w_c",
    "c_star",
    "mu_LMC",
    "mu_N4258",
    "sigma_int",
    "sigma_v",
    "mag_lim_TRGB",
    "mag_lim_TRGB_width",
    "Vext_mag",
    "Vext_ell",
    "Vext_b",
    "alpha_low",
    "alpha_high",
    "log_rho_t",
    "log_rho_width",
]


def main():
    if not FIDUCIAL.exists():
        raise FileNotFoundError(f"Missing fiducial posterior: {FIDUCIAL}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plot_corner_from_hdf5(
        FIDUCIAL,
        keys=CORNER_KEYS,
        labels=[r"\texttt{ManticoreLocalCOLA}, $R_\rho=4\,h^{-1}\,\mathrm{Mpc}$"],
        filled=False,
        fontsize=18,
        legend_fontsize=24,
        ranges={
            "alpha_low": [0.0, None],
            "alpha_high": [0.0, None],
            "sigma_v": [0.0, None],
        },
        filename=str(OUTDIR / "trgbh0_manticore_density_sigma_v_corner.pdf"),
        show_fig=False,
    )


if __name__ == "__main__":
    main()
