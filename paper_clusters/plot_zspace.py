"""2M++ z-space posterior comparison."""
from config import (
    setup_style, COLS, RESULTS_ROOT, get_figure_path, INCLUDE_MANTICORE,
    RECON_LABELS, RECON_COLORS,
)

import matplotlib.pyplot as plt
import numpy as np
import h5py
from h5py import File
from candel import plot_corner_getdist


def plot_zspace():
    """Compare dipole directions for 2M++ρ(z) and other reconstructions."""
    fnames = [
        RESULTS_ROOT / "zspace/2mpp_zspace_galaxies_LTYT_noMNR_dipA_hasY.hdf5",
        RESULTS_ROOT / "zspace/2mpp_zspace_galaxies_LTYT_noMNR_dipH0_hasY.hdf5",
    ]
    cols = [COLS[1], COLS[0]]
    labels = [
        RECON_LABELS["2mpp_zspace_galaxies"] + " ZP dipole",
        RECON_LABELS["2mpp_zspace_galaxies"] + r" $H_0$ dipole",
    ]
    contour_args = [
        {"zorder": 1},
        {"zorder": 1},
    ]
    line_args = [
        {},
        {},
    ]

    if INCLUDE_MANTICORE:
        fnames.append(RESULTS_ROOT / "zspace/manticore_LTYT_noMNR_dipH0_hasY.hdf5")
        cols.append(COLS[3])
        labels.append(RECON_LABELS["manticore"] + r" $H_0$ dipole")
        contour_args.append({"zorder": 1})
        line_args.append({})

    fnames.append(RESULTS_ROOT / "zspace/Carrick2015_LTYT_noMNR_dipH0_hasY.hdf5")
    cols.append(RECON_COLORS["Carrick2015"])
    labels.append(RECON_LABELS["Carrick2015"] + r" $H_0$ dipole")
    contour_args.append({"zorder": 3, "filled": False, "lw": 2.0})
    line_args.append({"lw": 2.0})

    fnames = [str(f) for f in fnames]

    def delta_a_to_frac(delta_a):
        delta_a = np.asarray(delta_a)
        return np.power(10.0, 0.5 * delta_a) - 1.0

    samples_list = []
    for fname in fnames:
        with File(fname, "r") as f:
            grp = f["samples"]
            samples = {}
            for key in grp.keys():
                item = grp[key]
                if isinstance(item, h5py.Dataset):
                    samples[key] = item[...]

        if "zeropoint_dipole_mag" in samples and "dH_over_H_dipole" not in samples:
            samples["dH_over_H_dipole"] = delta_a_to_frac(
                samples["zeropoint_dipole_mag"]
            )
            samples.pop("zeropoint_dipole_mag", None)

        samples_list.append(samples)

    keys = ["dH_over_H_dipole", "zeropoint_dipole_ell", "zeropoint_dipole_b"]

    plot_corner_getdist(
        samples_list,
        labels=labels,
        cols=cols,
        filled=True,
        keys=keys,
        filename=str(get_figure_path("2mpp_zspace_dipole_direction.pdf")),
        legend_fontsize=40,
        ell_zero=-90.0,
        apply_ell_offset=True,
        contour_args=contour_args,
        line_args=line_args,
    )
    plt.close("all")


def main():
    setup_style()
    plot_zspace()


if __name__ == "__main__":
    main()
