"""Base model (no dipole) for different reconstructions: full posterior corner plots."""
from config import (
    setup_style, get_results_path, get_figure_path,
    get_active_reconstructions, RECON_LABELS, RECON_COLORS, RECON_ZORDER,
)

import matplotlib.pyplot as plt
from candel import plot_corner_from_hdf5


# All parameters to plot for YT-only model (no dipole parameters)
KEYS_ALL = [
    'A_CL', 'B_CL', 'sigma', 'sigma_v',
    'R_dist_emp', 'n_dist_emp', 'p_dist_emp',
    'Vext_mag', 'Vext_ell', 'Vext_b',
]


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


def plot_base_model():
    """Base model (no dipole) for different reconstructions - all parameters."""
    fnames, cols, labels, contour_args, line_args = _build_corner_plot_data(
        "YT", "", has_y=True
    )
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
