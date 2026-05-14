#!/usr/bin/env python
"""Plot the fiducial TRGBH0 posterior corner."""
from pathlib import Path

import candel


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
RESULTS = ROOT / "results" / "TRGBH0_paper" / "table"
OUTDIR = ROOT / "notebooks" / "paper_TRGBH0" / "output"

FIDUCIAL = (
    RESULTS
    / "EDD_TRGB_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_sigv_rho_main.hdf5"
)

CORNER_KEYS = [
    "H0",
    "M_TRGB",
    "c_star",
    "mu_LMC",
    "mu_N4258",
    "sigma_int",
    "mag_lim_TRGB",
    "mag_lim_TRGB_width",
    "Vext_mag",
    "Vext_ell",
    "Vext_b",
    "alpha_low",
    "alpha_high",
    "log_rho_t",
    "sigma_v_low",
    "sigma_v_high",
    "log_sigma_v_rho_t",
    "sigma_v_k",
]


def main():
    if not FIDUCIAL.exists():
        raise FileNotFoundError(f"Missing fiducial posterior: {FIDUCIAL}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    candel.plot_corner_from_hdf5(
        FIDUCIAL,
        keys=CORNER_KEYS,
        labels=[r"\texttt{Manticore-Local}, density-dependent $\sigma_v$"],
        filled=False,
        fontsize=18,
        legend_fontsize=24,
        ranges={
            "alpha_low": [0.0, None],
            "alpha_high": [0.0, None],
            "sigma_v_low": [0.0, None],
            "sigma_v_high": [0.0, None],
            "sigma_v_k": [0.0, None],
        },
        filename=str(OUTDIR / "trgbh0_manticore_density_sigma_v_corner.pdf"),
        show_fig=False,
    )


if __name__ == "__main__":
    main()
