"""Sigma_v 1D distribution plot."""
from config import (
    setup_style, RESULTS_ROOT, RESULTS_FOLDER, get_figure_path,
    get_active_reconstructions, get_recon_labels, get_recon_colors,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
from getdist import plots, MCSamples

from candel import read_samples
try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib


def _build_sigma_v_files(relation, model, has_y=False):
    """Build file list for sigma_v plot, handling special Vext LTYT case."""
    suffix = "_hasY" if has_y else ""
    files = []
    for recon in get_active_reconstructions():
        # Special case: Vext LTYT files are in the "joint" folder
        if recon == "Vext" and relation == "LTYT":
            folder = "joint"
        else:
            folder = RESULTS_FOLDER
        fname = f"{folder}/{recon}_{relation}_noMNR_{model}{suffix}.hdf5"
        files.append(fname)
    return files


def main(include_dipA=False):
    setup_style()

    # Define file patterns for each row (relation) and column (dipole model)
    # Columns: dipVext, dipH0
    # Rows: LT, YT, LTYT
    relations = [
        {
            "name": "LT",
            "has_y": False,
        },
        {
            "name": "YT",
            "has_y": True,
        },
        {
            "name": "LTYT",
            "has_y": True,
        },
    ]
    models = [
        {"name": "dipVext", "label": r"Constant $\mathbf{V}_{\rm ext}$"},
        {"name": "dipH0", "label": r"Dipole $\delta H_0/H_0$"},
    ]

    # Get labels and colors from config
    labels = get_recon_labels()
    cols = get_recon_colors()

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
            samples = np.asarray(arr).flatten()
            log_samples = np.log10(samples)

            # Reflect samples at lower boundary
            reflected = 2 * log_sigma_v_lower - log_samples[log_samples < log_sigma_v_lower + 0.5]
            samples_extended = np.concatenate([log_samples, reflected])

            # Compute KDE on extended data
            kde = gaussian_kde(samples_extended)

            # Evaluate on grid above boundary
            x_log = np.linspace(log_sigma_v_lower, np.max(log_samples) + 0.1, n_points)
            y = kde(x_log) * 2  # Factor of 2 to correct for doubled data
            x = 10 ** x_log

            all_curves.append((x, y))

        return all_curves

    settings = plots.GetDistPlotSettings()
    settings.alpha_filled_add = -0.25
    g = plots.get_subplot_plotter(settings=settings)
    g.make_figure(nx=len(models), ny=len(relations), sharex=True, sharey=True)
    fig_w, fig_h = g.fig.get_size_inches()
    g.fig.set_size_inches(fig_w, fig_h * 0.85, forward=True)

    last_row_idx = len(relations) - 1
    for row_idx, relation in enumerate(relations):
        for col_idx, model in enumerate(models):
            ax = g.get_axes((row_idx, col_idx))
            files = _build_sigma_v_files(relation["name"], model["name"], has_y=relation["has_y"])
            smoothed_hists = load_sigma_samples_reflection_kde(files)
            for (x, y), col in zip(smoothed_hists, cols):
                y_norm = y / np.max(y)  # Normalize to peak height of 1
                ax.plot(x, y_norm, color=col, lw=1.8)
                ax.fill_between(x, 0, y_norm, color=col, alpha=0.2)
            ax.set_xscale("log")
            ax.set_xlim(left=sigma_v_lower)
            ax.set_ylim(bottom=0)
            ax.set_yticklabels([])
            # Add more x-axis ticks for better readability
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=10))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.tick_params(axis='x', labelsize=9)  # Shrink x-axis tick labels
            # Only add x-label on bottom row
            if row_idx == last_row_idx:
                ax.set_xlabel(r"$\sigma_v\ \mathrm{[km\,s^{-1}]}$")
            else:
                ax.set_xlabel("")
            if row_idx == 0:
                ax.set_title(model["label"], fontsize=12)
            # Add relation name in top right corner of each subplot
            ax.text(
                0.95,
                0.92,
                relation["name"],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
            )

        # Add per-row y-axis label
        first_ax = g.subplots[row_idx][0]
        first_ax.set_ylabel(r"$P(\sigma_v)$", fontsize=settings.axes_fontsize)

    legend_handles = [
        Line2D([0], [0], color=col, lw=1.8) for col in cols
    ]
    g.fig.legend(
        handles=legend_handles,
        labels=labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=11,
    )
    g.fig.subplots_adjust(left=0.02, right=0.995, bottom=0.10, top=0.92, wspace=0.0, hspace=-0.03)
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
