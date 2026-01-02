"""Shared configuration for paper_clusters plots and tables."""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt

# Paths
CANDEL_ROOT = Path("/Users/yasin/code/CANDEL")
RESULTS_ROOT = CANDEL_ROOT / "results"
RESULTS_FOLDER = "zspace"
FIGURES_FOLDER = CANDEL_ROOT / "paper_clusters/figures"
DATA_CONFIG_PATH = CANDEL_ROOT / "paper_clusters/data.toml"
CLUSTERS_DATA_PATH = CANDEL_ROOT / "data/Clusters/ClustersData.txt"

# Flags
INCLUDE_MANTICORE = False

# Color palette
COLS = ["#7570b3", "#d95f02", "#1b9e77", "#e7298a", "#66a61e"]

# Alternative colors used in some plots
C_WITH_Y = "#87193d"
C_NO_Y = "#1e42b9"


def setup_style():
    """Set up matplotlib style using scienceplots."""
    import importlib.util
    import pathlib

    import scienceplots  # noqa: F401 - registers styles

    pkg_file = importlib.util.find_spec("scienceplots").origin
    styles_dir = pathlib.Path(pkg_file).parent / "styles"

    user_stylelib = Path(matplotlib.get_configdir()) / "stylelib"
    user_stylelib.mkdir(parents=True, exist_ok=True)

    for f in styles_dir.glob("*.mplstyle"):
        shutil.copy2(f, user_stylelib / f.name)

    matplotlib.style.reload_library()
    plt.style.use("science")

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    })


def get_results_path(fname):
    """Get full path for a results file."""
    return RESULTS_ROOT / RESULTS_FOLDER / fname


def get_figure_path(fname):
    """Get full path for saving a figure."""
    return FIGURES_FOLDER / fname
