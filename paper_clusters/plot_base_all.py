"""Base model (no dipole) for different reconstructions: full posterior corner plots."""
from config import setup_style, COLS, get_results_path, get_figure_path, INCLUDE_MANTICORE

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


# All parameters to plot for YT-only model (no dipole parameters)
KEYS_ALL = [
    'A_CL', 'B_CL', 'sigma', 'sigma_v',
    'R_dist_emp', 'n_dist_emp', 'p_dist_emp',
    'Vext_mag', 'Vext_ell', 'Vext_b',
]


def plot_base_model():
    """Base model (no dipole) for different reconstructions - all parameters."""
    # Order: Carrick, [Manticore], 2M++, No recon
    fnames = [get_results_path("Carrick2015_YT_noMNR_hasY.hdf5")]
    cols = [COLS[3]]  # pink
    labels = ["Carrick2015"]
    contour_args = [{"zorder": 4, "filled": False, "lw": 2.0}]  # Carrick (front)
    line_args = [{"lw": 2.0}]

    if INCLUDE_MANTICORE:
        fnames.append(get_results_path("manticore_YT_noMNR_hasY.hdf5"))
        cols.append(COLS[1])  # orange
        labels.append("Manticore")
        contour_args.append({"zorder": 3})
        line_args.append({})

    fnames.append(get_results_path("2mpp_zspace_galaxies_YT_noMNR_hasY.hdf5"))
    cols.append(COLS[2])  # green
    labels.append(r"2M++$\rho(z)$")
    contour_args.append({"zorder": 2})
    line_args.append({})

    fnames.append(get_results_path("Vext_YT_noMNR_hasY.hdf5"))
    cols.append(COLS[0])  # purple
    labels.append("No reconstruction")
    contour_args.append({"zorder": 1})
    line_args.append({})

    fnames = [str(f) for f in fnames]
    points = {("Vext_ell", "Vext_b"): (264., 48.)}

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        points=points,
        keys=KEYS_ALL,
        filename=str(get_figure_path("reconstruction_base_all.pdf")),
        legend_fontsize=40,
        apply_ell_offset=True,
        ell_zero=180.0,
        contour_args=contour_args,
        line_args=line_args,
    )
    plt.close('all')


def main():
    setup_style()
    plot_base_model()


if __name__ == "__main__":
    main()
