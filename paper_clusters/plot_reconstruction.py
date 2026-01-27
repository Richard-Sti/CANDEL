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


def _load_los_velocity(path):
    """Load LOS velocity from HDF5 file."""
    import h5py
    with h5py.File(path, "r") as f:
        vel = f["los_velocity"][...]
        r = f["r"][...]
    return vel, r


def _load_reconstruction_data(carrick_path, manticore_path, log_floor):
    delta_c, r_c = _load_los_delta(Path(carrick_path))
    delta_m, r_m = _load_los_delta(Path(manticore_path))

    # Load velocities
    vel_c, _ = _load_los_velocity(Path(carrick_path))
    vel_m, _ = _load_los_velocity(Path(manticore_path))

    # Interpolate Carrick to Manticore radial grid if needed
    if r_c.shape != r_m.shape or not np.allclose(r_c, r_m):
        delta_c = _interp_los(delta_c, r_c, r_m)
        vel_c = _interp_los(vel_c, r_c, r_m)
        r_c = r_m

    if delta_m.ndim != 3:
        raise ValueError("Expected Manticore LOS to have shape (nsim, n_gal, n_r).")
    if delta_c.ndim == 3:
        delta_c = delta_c[0]
    if vel_c.ndim == 3:
        vel_c = vel_c[0]

    n_gal = min(delta_c.shape[0], delta_m.shape[1])
    delta_c = delta_c[:n_gal]
    delta_m = delta_m[:, :n_gal]
    vel_c = vel_c[:n_gal]
    vel_m = vel_m[:, :n_gal]

    delta_c = np.log10(np.clip(1.0 + delta_c, log_floor, None))
    delta_m_log = np.log10(np.clip(1.0 + delta_m, log_floor, None))
    mean_m = np.mean(delta_m_log, axis=0)
    std_m = np.std(delta_m_log, axis=0)

    # Velocity statistics for Manticore
    mean_vel_m = np.mean(vel_m, axis=0)
    std_vel_m = np.std(vel_m, axis=0)

    return {
        "r": r_m,
        "delta_c": delta_c,
        "mean_m": mean_m,
        "std_m": std_m,
        "vel_c": vel_c,
        "mean_vel_m": mean_vel_m,
        "std_vel_m": std_vel_m,
    }


def _plot_cluster(axes, data, cluster_index, cluster_name, r_mark):
    ax_rho, ax_vel = axes
    r = data["r"]
    delta_c = data["delta_c"][cluster_index]
    mean_m = data["mean_m"][cluster_index]
    std_m = data["std_m"][cluster_index]
    vel_c = data["vel_c"][cluster_index]
    mean_vel_m = data["mean_vel_m"][cluster_index]
    std_vel_m = data["std_vel_m"][cluster_index]

    # Colors: pink (Carrick), green (Manticore)
    # Top panel: density
    ax_rho.plot(r, delta_c, color="#e7298a", lw=1.8, label="Carrick2015")
    ax_rho.fill_between(
        r,
        mean_m - std_m,
        mean_m + std_m,
        color="#1b9e77",
        alpha=0.25,
        linewidth=0.0,
    )
    ax_rho.plot(r, mean_m, color="#1b9e77", lw=1.2, label="Manticore")
    ax_rho.axvline(r_mark, color="0.5", lw=0.9, ls="--", alpha=0.7)

    ax_rho.set_ylabel(r"$\log_{10}(1+\delta)$")
    ax_rho.set_xlim(0.0, 200.0)
    ax_rho.legend(frameon=False, loc="upper right")
    ax_rho.tick_params(labelbottom=False)

    # Bottom panel: velocity (Carrick scaled by beta=0.43)
    ax_vel.plot(r, vel_c * 0.43, color="#e7298a", lw=1.8)
    ax_vel.fill_between(
        r,
        mean_vel_m - std_vel_m,
        mean_vel_m + std_vel_m,
        color="#1b9e77",
        alpha=0.25,
        linewidth=0.0,
    )
    ax_vel.plot(r, mean_vel_m, color="#1b9e77", lw=1.2)
    ax_vel.axvline(r_mark, color="0.5", lw=0.9, ls="--", alpha=0.7, label="Cluster")
    ax_vel.axhline(0, color="0.5", lw=0.5, ls=":", alpha=0.5)

    ax_vel.set_xlabel(r"$r$ [Mpc/$h$]")
    ax_vel.set_ylabel(r"$v_{\rm los}$ [km/s]")
    ax_vel.set_xlim(0.0, 200.0)
    ax_vel.legend(frameon=False, loc="upper right")


def plot_cluster_reconstruction(
    carrick_path=None,
    manticore_path=None,
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

    if data is None:
        data = _load_reconstruction_data(carrick_path, manticore_path, log_floor)

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

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})
    _plot_cluster(axes, data, cluster_index, cluster_name, r_mark)

    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.10)
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
