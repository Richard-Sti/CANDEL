"""Dipoles for different reconstructions: Vext and H0 corner plots."""
from config import (
    setup_style, get_results_path, get_figure_path,
    get_active_reconstructions, RECON_LABELS, RECON_COLORS, RECON_ZORDER,
)

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


def _build_corner_plot_data(relation, model, has_y=False):
    """Build lists of fnames, cols, labels, and plot args for corner plots.

    Returns data in the order needed for plotting: Carrick first (with special
    styling), then other reconstructions.
    """
    suffix = "_hasY" if has_y else ""
    model_part = f"_{model}" if model else ""

    # Build data for each active reconstruction
    recons = get_active_reconstructions()

    # Reorder so Carrick2015 is first (for layering purposes in corner plot)
    plot_order = []
    if "Carrick2015" in recons:
        plot_order.append("Carrick2015")
    plot_order.extend([r for r in recons if r != "Carrick2015"])

    fnames = []
    cols = []
    labels = []
    contour_args = []
    line_args = []

    for recon in plot_order:
        fname = get_results_path(f"{recon}_{relation}_noMNR{model_part}{suffix}.hdf5")
        fnames.append(str(fname))
        cols.append(RECON_COLORS[recon])
        labels.append(RECON_LABELS[recon])

        # Carrick gets special styling: unfilled, thick lines, on top
        if recon == "Carrick2015":
            contour_args.append({"zorder": RECON_ZORDER[recon], "filled": False, "lw": 2.0})
            line_args.append({"lw": 2.0})
        else:
            contour_args.append({"zorder": RECON_ZORDER[recon]})
            line_args.append({})

    return fnames, cols, labels, contour_args, line_args


def plot_vext_dipole():
    """Figure (a): Vext dipole for different reconstructions."""
    fnames, cols, labels, contour_args, line_args = _build_corner_plot_data(
        "LTYT", "dipVext", has_y=True
    )
    keys = ['Vext_mag', 'Vext_ell', 'Vext_b']
    points = {("Vext_ell", "Vext_b"): (264., 48.)}

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        points=points,
        keys=keys,
        filename=str(get_figure_path("reconstruction_vext.pdf")),
        legend_fontsize=40,
        apply_ell_offset=True,
        ell_zero=180.0,
        contour_args=contour_args,
        line_args=line_args,
    )
    plt.close('all')


def plot_h0_dipole():
    """Figure (b): H0 dipole for different reconstructions."""
    fnames, cols, labels, contour_args, line_args = _build_corner_plot_data(
        "LTYT", "dipH0", has_y=True
    )
    keys = ['H0_dipole_mag', 'H0_dipole_ell', 'H0_dipole_b']
    points = {("Vext_ell", "Vext_b"): (264., 48.)}

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        points=points,
        keys=keys,
        filename=str(get_figure_path("reconstruction_H0.pdf")),
        legend_fontsize=40,
        ell_zero=-90.0,
        apply_ell_offset=True,
        contour_args=contour_args,
        line_args=line_args,
    )
    plt.close('all')


def plot_zeropoint_dipole():
    """Figure (c): Zeropoint dipole for different reconstructions."""
    fnames, cols, labels, contour_args, line_args = _build_corner_plot_data(
        "LTYT", "dipA", has_y=True
    )
    keys = ['zeropoint_dipole_mag', 'zeropoint_dipole_ell', 'zeropoint_dipole_b']

    plot_corner_from_hdf5(
        fnames,
        labels=labels,
        cols=cols,
        filled=True,
        keys=keys,
        filename=str(get_figure_path("reconstruction_zeropoint.pdf")),
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
