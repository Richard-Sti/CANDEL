"""Sigma_v 1D distribution plot."""
from config import setup_style, COLS, RESULTS_ROOT, RESULTS_FOLDER, get_figure_path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from getdist import plots, MCSamples

from candel import read_samples
try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib


def main(include_dipA=False):
    setup_style()

    # Define file patterns for each row (dipole model) and column (relation)
    # Row 0: dipVext, Row 1: dipH0, Row 2: dipA
    rows = [
        {
            "label": r"$\mathbf{\mathrm{dip}\,V_\mathrm{ext}}$",
            "relations": [
                {
                    "name": "LT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_LT_noMNR_dipVext.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LT_noMNR_dipVext.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LT_noMNR_dipVext.hdf5",
                        f"{RESULTS_FOLDER}/2mpp_zspace_galaxies_LT_noMNR_dipVext.hdf5",
                    ],
                },
                {
                    "name": "YT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_YT_noMNR_dipVext_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_YT_noMNR_dipVext_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_YT_noMNR_dipVext_hasY.hdf5",
                        f"{RESULTS_FOLDER}/2mpp_zspace_galaxies_YT_noMNR_dipVext_hasY.hdf5",
                    ],
                },
                {
                    "name": "LTYT",
                    "files": [
                        f"joint/Vext_LTYT_noMNR_dipVext_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LTYT_noMNR_dipVext_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LTYT_noMNR_dipVext_hasY.hdf5",
                        f"{RESULTS_FOLDER}/2mpp_zspace_galaxies_LTYT_noMNR_dipVext_hasY.hdf5",
                    ],
                },
            ],
        },
        {
            "label": r"$\mathbf{\mathrm{dip}\,H_0}$",
            "relations": [
                {
                    "name": "LT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_LT_noMNR_dipH0.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LT_noMNR_dipH0.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LT_noMNR_dipH0.hdf5",
                        f"{RESULTS_FOLDER}/2mpp_zspace_galaxies_LT_noMNR_dipH0.hdf5",
                    ],
                },
                {
                    "name": "YT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_YT_noMNR_dipH0_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_YT_noMNR_dipH0_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_YT_noMNR_dipH0_hasY.hdf5",
                        f"{RESULTS_FOLDER}/2mpp_zspace_galaxies_YT_noMNR_dipH0_hasY.hdf5",
                    ],
                },
                {
                    "name": "LTYT",
                    "files": [
                        f"joint/Vext_LTYT_noMNR_dipH0_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LTYT_noMNR_dipH0_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LTYT_noMNR_dipH0_hasY.hdf5",
                        f"{RESULTS_FOLDER}/2mpp_zspace_galaxies_LTYT_noMNR_dipH0_hasY.hdf5",
                    ],
                },
            ],
        },
        {
            "label": r"$\mathbf{\mathrm{dip}\,A}$",
            "relations": [
                {
                    "name": "LT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_LT_noMNR_dipA.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LT_noMNR_dipA.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LT_noMNR_dipA.hdf5",
                        f"{RESULTS_FOLDER}/2mpp_zspace_galaxies_LT_noMNR_dipA.hdf5",
                    ],
                },
                {
                    "name": "YT",
                    "files": [
                        f"{RESULTS_FOLDER}/Vext_YT_noMNR_dipA_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_YT_noMNR_dipA_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_YT_noMNR_dipA_hasY.hdf5",
                        f"{RESULTS_FOLDER}/2mpp_zspace_galaxies_YT_noMNR_dipA_hasY.hdf5",
                    ],
                },
                {
                    "name": "LTYT",
                    "files": [
                        f"joint/Vext_LTYT_noMNR_dipA_hasY.hdf5",
                        f"{RESULTS_FOLDER}/Carrick2015_LTYT_noMNR_dipA_hasY.hdf5",
                        f"{RESULTS_FOLDER}/manticore_LTYT_noMNR_dipA_hasY.hdf5",
                        f"{RESULTS_FOLDER}/2mpp_zspace_galaxies_LTYT_noMNR_dipA_hasY.hdf5",
                    ],
                },
            ],
        },
    ]
    if not include_dipA:
        rows = rows[:-1]
    # Order: No recon, Carrick, Manticore, 2M++ρ(z)
    # Colors: purple (no recon), pink (Carrick), orange (Manticore), green (2M++)
    labels = ["No reconstruction", "Carrick2015", "Manticore", r"2M++$\rho(z)$"]
    cols = [COLS[0], COLS[3], COLS[1], COLS[2]]  # purple, pink, orange, green

    def read_sigma_v_lower_bound(fname):
        toml_path = RESULTS_ROOT / fname
        toml_path = toml_path.with_suffix(".toml")
        try:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            return float(data["model"]["priors"]["sigma_v"]["low"])
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            return 0.0

    # Prior lower bound on sigma_v is 150 km/s
    sigma_v_lower = 150.0
    log_sigma_v_lower = np.log10(sigma_v_lower)

    def load_sigma_samples_reflection_kde(fnames, n_points=500):
        """Load samples and compute reflection KDE to handle boundary."""
        from scipy.stats import gaussian_kde

        all_curves = []
        for f in fnames:
            arr = read_samples(str(RESULTS_ROOT), f, keys="sigma_v")
            samples = np.log10(np.asarray(arr)).flatten()

            # Reflect samples at lower boundary
            reflected = 2 * log_sigma_v_lower - samples[samples < log_sigma_v_lower + 0.5]
            samples_extended = np.concatenate([samples, reflected])

            # Compute KDE on extended data
            kde = gaussian_kde(samples_extended)

            # Evaluate on grid above boundary
            x = np.linspace(log_sigma_v_lower, np.max(samples) + 0.1, n_points)
            y = kde(x) * 2  # Factor of 2 to correct for doubled data

            all_curves.append((x, y))

        return all_curves

    settings = plots.GetDistPlotSettings()
    settings.alpha_filled_add = -0.25
    g = plots.get_subplot_plotter(settings=settings)
    g.make_figure(nx=3, ny=len(rows), sharex=True, sharey=True)
    fig_w, fig_h = g.fig.get_size_inches()
    g.fig.set_size_inches(fig_w, fig_h * 0.85, forward=True)

    last_row_idx = len(rows) - 1
    for row_idx, row in enumerate(rows):
        for col_idx, relation in enumerate(row["relations"]):
            ax = g.get_axes((row_idx, col_idx))
            smoothed_hists = load_sigma_samples_reflection_kde(relation["files"])
            for (x, y), col in zip(smoothed_hists, cols):
                y_norm = y / np.max(y)  # Normalize to peak height of 1
                ax.plot(x, y_norm, color=col, lw=1.8)
                ax.fill_between(x, 0, y_norm, color=col, alpha=0.2)
            ax.set_xlim(left=log_sigma_v_lower)
            ax.set_ylim(bottom=0)
            ax.set_yticklabels([])
            # Relation name in top right
            ax.text(
                0.98,
                0.95,
                relation["name"],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
            )
            # Only add x-label on bottom row
            if row_idx == last_row_idx:
                ax.set_xlabel(r"$\log_{10}(\sigma_v/\mathrm{km\,s^{-1}})$")
            else:
                ax.set_xlabel("")

        # Add per-row y-axis label and a bold row label further left
        first_ax = g.subplots[row_idx][0]
        first_ax.set_ylabel(r"$P(\log_{10}\sigma_v)$", fontsize=settings.axes_fontsize)
        first_ax.text(
            -0.28,
            0.5,
            row["label"],
            transform=first_ax.transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=settings.axes_fontsize + 2,
        )

    legend_handles = [
        Line2D([0], [0], color=col, lw=1.8) for col in cols
    ]
    g.fig.legend(
        handles=legend_handles,
        labels=labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=9,
    )
    g.fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.92, wspace=0.0, hspace=-0.03)
    g.fig.savefig(str(get_figure_path("sigma_v.pdf")), bbox_inches="tight")
    plt.close('all')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot sigma_v distributions.")
    parser.add_argument(
        "--dipA",
        action="store_true",
        help="Include the dipA row in the plot.",
    )
    args = parser.parse_args()
    main(include_dipA=args.dipA)
