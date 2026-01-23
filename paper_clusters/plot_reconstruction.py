"""Single-cluster reconstruction comparison plot."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import CLUSTERS_DATA_PATH, DATA_CONFIG_PATH, setup_style, get_figure_path
from scripts.preprocess.compare_reconstructions import (
    _interp_los,
    _load_cluster_zcmb,
    _load_los_delta,
)


def _load_cluster_names(path=CLUSTERS_DATA_PATH):
    names = np.genfromtxt(path, dtype="U32", usecols=0, skip_header=1)
    return np.asarray(names)


def _load_reconstruction_data(carrick_path, manticore_path, zspace_path, log_floor):
    delta_c, r_c = _load_los_delta(Path(carrick_path))
    delta_m, r_m = _load_los_delta(Path(manticore_path))
    delta_z, r_z = _load_los_delta(Path(zspace_path))

    if r_c.shape != r_z.shape or not np.allclose(r_c, r_z):
        delta_c = _interp_los(delta_c, r_c, r_z)
        r_c = r_z
    if r_m.shape != r_z.shape or not np.allclose(r_m, r_z):
        delta_m = _interp_los(delta_m, r_m, r_z)
        r_m = r_z

    if delta_m.ndim != 3:
        raise ValueError("Expected Manticore LOS to have shape (nsim, n_gal, n_r).")
    if delta_c.ndim == 3:
        delta_c = delta_c[0]

    n_gal = min(delta_c.shape[0], delta_m.shape[1], delta_z.shape[0])
    delta_c = delta_c[:n_gal]
    delta_m = delta_m[:, :n_gal]
    delta_z = delta_z[:n_gal]

    delta_c = np.log10(np.clip(1.0 + delta_c, log_floor, None))
    delta_z = np.log10(np.clip(1.0 + delta_z, log_floor, None))
    delta_m_log = np.log10(np.clip(1.0 + delta_m, log_floor, None))
    mean_m = np.mean(delta_m_log, axis=0)
    std_m = np.std(delta_m_log, axis=0)

    return {
        "r_c": r_c,
        "r_m": r_m,
        "r_z": r_z,
        "delta_c": delta_c,
        "delta_z": delta_z,
        "mean_m": mean_m,
        "std_m": std_m,
    }


def _plot_cluster(ax, data, cluster_index, cluster_name, r_mark):
    r_c = data["r_c"]
    r_m = data["r_m"]
    r_z = data["r_z"]
    delta_c = data["delta_c"][cluster_index]
    delta_z = data["delta_z"][cluster_index]
    mean_m = data["mean_m"][cluster_index]
    std_m = data["std_m"][cluster_index]

    # Colors: pink (Carrick), green (2M++), orange (Manticore)
    ax.plot(r_c, delta_c, color="#e7298a", lw=1.8, label="Carrick2015")
    ax.plot(r_z, delta_z, color="#1b9e77", lw=1.2, label=r"2M++$\rho(z)$")
    ax.fill_between(
        r_m,
        mean_m - std_m,
        mean_m + std_m,
        color="#d95f02",
        alpha=0.25,
        linewidth=0.0,
        label=None,
    )
    ax.plot(r_m, mean_m, color="#d95f02", lw=1.2, label="Manticore")
    ax.axvline(r_mark, color="0.5", lw=0.9, ls="--", alpha=0.7, label="Cluster position")

    ax.set_xlabel("r [Mpc/h]")
    ax.set_ylabel(r"$\log_{10}(1+\delta)$")
    ax.set_xlim(0.0, 226.0)
    ax.legend(frameon=False, ncol=2)


def plot_cluster_reconstruction(
    carrick_path=None,
    manticore_path=None,
    zspace_path=None,
    cluster_name=None,
    log_floor=1e-5,
    output=None,
    config_path=None,
    data=None,
    cluster_names=None,
    r_marks=None,
):
    setup_style()

    # Set default paths using CANDEL_ROOT
    from config import CANDEL_ROOT
    if carrick_path is None:
        carrick_path = CANDEL_ROOT / "data/Clusters/los_Clusters_Carrick2015.hdf5"
    if manticore_path is None:
        manticore_path = CANDEL_ROOT / "data/Clusters/los_Clusters_manticore.hdf5"
    if zspace_path is None:
        zspace_path = CANDEL_ROOT / "data/Clusters/los_Clusters_2mpp_zspace_galaxies.hdf5"

    if data is None:
        data = _load_reconstruction_data(carrick_path, manticore_path, zspace_path, log_floor)

    if cluster_names is None:
        cluster_names = _load_cluster_names()
    if cluster_name is None:
        raise ValueError("cluster_name is required.")
    match = np.where(np.char.lower(cluster_names) == cluster_name.lower())[0]
    if match.size == 0:
        raise ValueError(f"cluster_name '{cluster_name}' not found in cluster list.")
    cluster_index = int(match[0])

    if r_marks is None:
        if config_path is None:
            config_path = DATA_CONFIG_PATH
        r_marks = _load_cluster_zcmb(config_path)
    if cluster_index >= r_marks.shape[0]:
        raise IndexError(
            f"cluster_index {cluster_index} out of range for r_marks "
            f"(n_gal={r_marks.shape[0]})."
        )
    r_mark = float(r_marks[cluster_index])

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    _plot_cluster(ax, data, cluster_index, cluster_name, r_mark)

    fig.tight_layout()
    if output is None:
        output = get_figure_path("rho_los.pdf")
    fig.savefig(str(output))
    plt.close(fig)


def main(cluster_name="Coma"):
    """Plot LOS reconstruction for a single cluster.

    Default cluster is Coma.
    """
    plot_cluster_reconstruction(cluster_name=cluster_name)


if __name__ == "__main__":
    main()
