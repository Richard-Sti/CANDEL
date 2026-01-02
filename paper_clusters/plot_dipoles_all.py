"""Dipoles for different reconstructions: full posterior corner plots."""
from config import setup_style, COLS, get_results_path, get_figure_path, INCLUDE_MANTICORE

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


# All parameters to plot
KEYS_ALL = [
    'A_CL', 'B_CL', 'A2_CL', 'B2_CL', 'sigma', 'sigma2', 'rho12', 'sigma_v',
    'R_dist_emp', 'n_dist_emp', 'p_dist_emp',
    'Vext_mag', 'Vext_ell', 'Vext_b',
    'dH_over_H_dipole', 'zeropoint_dipole_ell', 'zeropoint_dipole_b',
]


def plot_vext_dipole():
    """Figure (a): Vext dipole for different reconstructions - all parameters."""
    # Order: Carrick, [Manticore], 2M++, No recon
    fnames = [get_results_path("Carrick2015_LTYT_noMNR_dipVext_hasY.hdf5")]
    cols = [COLS[3]]  # pink
    labels = ["Carrick2015"]
    contour_args = [{"zorder": 4, "filled": False, "lw": 2.0}]  # Carrick (front)
    line_args = [{"lw": 2.0}]

    if INCLUDE_MANTICORE:
        fnames.append(get_results_path("manticore_LTYT_noMNR_dipVext_hasY.hdf5"))
        cols.append(COLS[1])  # orange
        labels.append("Manticore")
        contour_args.append({"zorder": 3})
        line_args.append({})

    fnames.append(get_results_path("2mpp_zspace_galaxies_LTYT_noMNR_dipVext_hasY.hdf5"))
    cols.append(COLS[2])  # green
    labels.append(r"2M++$\rho(z)$")
    contour_args.append({"zorder": 2})
    line_args.append({})

    fnames.append(get_results_path("Vext_LTYT_noMNR_dipVext_hasY.hdf5"))
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
        filename=str(get_figure_path("reconstruction_vext_all.pdf")),
        legend_fontsize=40,
        apply_ell_offset=True,
        ell_zero=180.0,
        contour_args=contour_args,
        line_args=line_args,
    )
    plt.close('all')


def plot_h0_dipole():
    """Figure (b): H0 dipole for different reconstructions - all parameters."""
    # Order: Carrick, [Manticore], 2M++, No recon
    fnames = [get_results_path("Carrick2015_LTYT_noMNR_dipH0_hasY.hdf5")]
    cols = [COLS[3]]  # pink
    labels = ["Carrick2015"]
    contour_args = [{"zorder": 4, "filled": False, "lw": 2.0}]  # Carrick (front)
    line_args = [{"lw": 2.0}]

    if INCLUDE_MANTICORE:
        fnames.append(get_results_path("manticore_LTYT_noMNR_dipH0_hasY.hdf5"))
        cols.append(COLS[1])  # orange
        labels.append("Manticore")
        contour_args.append({"zorder": 3})
        line_args.append({})

    fnames.append(get_results_path("2mpp_zspace_galaxies_LTYT_noMNR_dipH0_hasY.hdf5"))
    cols.append(COLS[2])  # green
    labels.append(r"2M++$\rho(z)$")
    contour_args.append({"zorder": 2})
    line_args.append({})

    fnames.append(get_results_path("Vext_LTYT_noMNR_dipH0_hasY.hdf5"))
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
        filename=str(get_figure_path("reconstruction_H0_all.pdf")),
        legend_fontsize=40,
        ell_zero=-90.0,
        apply_ell_offset=True,
        contour_args=contour_args,
        line_args=line_args,
    )
    plt.close('all')


def plot_zeropoint_dipole():
    """Figure (c): Zeropoint dipole for different reconstructions - all parameters."""
    # Order: Carrick, [Manticore], 2M++, No recon
    fnames = [get_results_path("Carrick2015_LTYT_noMNR_dipA_hasY.hdf5")]
    cols = [COLS[3]]  # pink
    labels = ["Carrick2015"]
    contour_args = [{"zorder": 4, "filled": False, "lw": 2.0}]  # Carrick (front)
    line_args = [{"lw": 2.0}]

    if INCLUDE_MANTICORE:
        fnames.append(get_results_path("manticore_LTYT_noMNR_dipA_hasY.hdf5"))
        cols.append(COLS[1])  # orange
        labels.append("Manticore")
        contour_args.append({"zorder": 3})
        line_args.append({})

    fnames.append(get_results_path("2mpp_zspace_galaxies_LTYT_noMNR_dipA_hasY.hdf5"))
    cols.append(COLS[2])  # green
    labels.append(r"2M++$\rho(z)$")
    contour_args.append({"zorder": 2})
    line_args.append({})

    fnames.append(get_results_path("Vext_LTYT_noMNR_dipA_hasY.hdf5"))
    cols.append(COLS[0])  # purple
    labels.append("No reconstruction")
    contour_args.append({"zorder": 1})
    line_args.append({})

    fnames = [str(f) for f in fnames]

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        keys=KEYS_ALL,
        filename=str(get_figure_path("reconstruction_zeropoint_all.pdf")),
        legend_fontsize=40,
        ell_zero=-90.0,
        apply_ell_offset=True,
        contour_args=contour_args,
        line_args=line_args,
    )
    plt.close('all')


def main():
    setup_style()
    plot_vext_dipole()
    plot_h0_dipole()
    plot_zeropoint_dipole()


if __name__ == "__main__":
    main()
