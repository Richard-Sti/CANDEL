"""Shared configuration for paper_clusters plots and tables."""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt

# Paths
CANDEL_ROOT = Path("/Users/yasin/code/CANDEL")
RESULTS_ROOT = CANDEL_ROOT / "results"
RESULTS_FOLDER = "nodensity2"
FIGURES_FOLDER = CANDEL_ROOT / "paper_clusters/figures"
DATA_CONFIG_PATH = CANDEL_ROOT / "paper_clusters/data.toml"
CLUSTERS_DATA_PATH = CANDEL_ROOT / "data/Clusters/ClustersData.txt"

# Flags
INCLUDE_MANTICORE = True

# Color palette
COLS = ["#7570b3", "#d95f02", "#1b9e77", "#e7298a", "#66a61e"]

# Reconstruction configuration
# Keys are the prefixes used in filenames (e.g., "Carrick2015_LT_noMNR_dipVext.hdf5")
# Order: C15 (fiducial) first, then Manticore, then No Recon
RECONSTRUCTIONS = ["Carrick2015", "manticore", "Vext"]

# Display names for plots/tables
RECON_LABELS = {
    "Vext": "No reconstruction",
    "Carrick2015": "Carrick2015 (fiducial model)",
    "manticore": "Manticore",
}

# Short labels for tables
RECON_LABELS_SHORT = {
    "Vext": "No Recon",
    "Carrick2015": "C15",
    "manticore": "Manticore",
}

# Short labels for main tables (with fiducial marker)
RECON_LABELS_SHORT_FIDUCIAL = {
    "Vext": "No Recon",
    "Carrick2015": "C15 (fiducial model)",
    "manticore": "Manticore",
}

# Colors for each reconstruction (matched to COLS palette)
RECON_COLORS = {
    "Vext": COLS[0],        # purple
    "Carrick2015": COLS[3], # pink
    "manticore": COLS[2],   # green
}

# Z-order for plotting (higher = on top)
RECON_ZORDER = {
    "Vext": 1,
    "Carrick2015": 4,
    "manticore": 3,
}


def get_active_reconstructions():
    """Return list of active reconstructions, respecting INCLUDE_MANTICORE flag."""
    if INCLUDE_MANTICORE:
        return RECONSTRUCTIONS
    return [r for r in RECONSTRUCTIONS if r != "manticore"]


def get_recon_labels():
    """Return labels for active reconstructions."""
    return [RECON_LABELS[r] for r in get_active_reconstructions()]


def get_recon_colors():
    """Return colors for active reconstructions."""
    return [RECON_COLORS[r] for r in get_active_reconstructions()]


def get_recon_zorders():
    """Return z-orders for active reconstructions."""
    return [RECON_ZORDER[r] for r in get_active_reconstructions()]

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


def build_recon_file_list(relation, model, has_y=False, folder=None):
    """Build file list for active reconstructions.

    Parameters
    ----------
    relation : str
        Relation name (e.g., "LT", "YT", "LTYT")
    model : str
        Model suffix (e.g., "dipVext", "dipH0", "base" or "")
    has_y : bool
        Whether to append "_hasY" suffix
    folder : str, optional
        Override results folder. Defaults to RESULTS_FOLDER.

    Returns
    -------
    list
        List of file paths for each active reconstruction.
    """
    if folder is None:
        folder = RESULTS_FOLDER

    suffix = "_hasY" if has_y else ""
    model_part = f"_{model}" if model and model != "base" else ""

    files = []
    for recon in get_active_reconstructions():
        fname = f"{folder}/{recon}_{relation}_noMNR{model_part}{suffix}.hdf5"
        files.append(str(RESULTS_ROOT / fname))
    return files
