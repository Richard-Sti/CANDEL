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
    ax_rho.axvline(r_mark, color="k", lw=1.0, ls="-", label="Cluster position")

    ax_rho.set_ylabel(r"$\log_{10}(1+\delta)$")
    ax_rho.set_xlim(40.0, 120.0)
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
    ax_vel.axvline(r_mark, color="k", lw=1.0, ls="-")
    ax_vel.axhline(0, color="0.5", lw=0.5, ls=":", alpha=0.5)

    ax_vel.set_ylabel(r"$v_{\rm los}$ [km/s]")
    ax_vel.set_xlim(40.0, 120.0)
    ax_vel.tick_params(labelbottom=False)


def _plot_velocity_panel(ax, r, vel_c, mean_vel_m, std_vel_m, r_mark,
                         title=None, show_xlabel=True, r_mark_old=None):
    """Plot a velocity panel in the same style as the original with v_los annotation.

    Parameters
    ----------
    ax : matplotlib axis
    r : array, radial grid
    vel_c : array, Carrick velocity (already scaled by beta)
    mean_vel_m : array, Manticore mean velocity
    std_vel_m : array, Manticore std velocity
    r_mark : float, cluster position
    title : str, optional panel title (displayed in top right corner)
    show_xlabel : bool, whether to show x-axis label
    r_mark_old : float, optional old cluster position (shown as dashed line)
    """
    # Carrick (pink)
    ax.plot(r, vel_c, color="#e7298a", lw=1.8)
    # Manticore (green with uncertainty band)
    ax.fill_between(
        r,
        mean_vel_m - std_vel_m,
        mean_vel_m + std_vel_m,
        color="#1b9e77",
        alpha=0.25,
        linewidth=0.0,
    )
    ax.plot(r, mean_vel_m, color="#1b9e77", lw=1.2)
    # Old cluster position (dashed) if provided
    if r_mark_old is not None:
        ax.axvline(r_mark_old, color="0.5", lw=0.9, ls="--", alpha=0.7)
    # Current cluster marker (solid black) and zero line
    ax.axvline(r_mark, color="k", lw=1.0, ls="-")
    ax.axhline(0, color="0.5", lw=0.5, ls=":", alpha=0.5)

    # Title in top right corner
    if title:
        ax.text(0.97, 0.95, title, transform=ax.transAxes, ha='right', va='top',
                fontsize=12)
    # v_los annotation just below title
    v_at_cluster = np.interp(r_mark, r, vel_c)
    ax.text(0.97, 0.82, rf"$v_{{\rm los}} = {v_at_cluster:.0f}$ km/s",
            transform=ax.transAxes, ha='right', va='top', fontsize=12)

    if show_xlabel:
        ax.set_xlabel(r"$r$ [Mpc/$h$]")
    ax.set_ylabel(r"$v_{\rm los}$ [km/s]")
    ax.set_xlim(40.0, 120.0)
    ax.tick_params(labelbottom=show_xlabel)


def _plot_panel(ax, r, y, r_mark, panel_label, ylabel, v_at_cluster=None,
                color="#e7298a", show_ylabel=True):
    """Plot a single panel with optional velocity annotation."""
    ax.plot(r, y, color=color, lw=1.8)
    ax.axvline(r_mark, color="0.5", lw=0.9, ls="--", alpha=0.7)

    # Display velocity at cluster position (only for velocity panels)
    if v_at_cluster is not None:
        ax.text(0.95, 0.95, rf"$v_{{\rm los}} = {v_at_cluster:.0f}$ km/s",
                transform=ax.transAxes, ha='right', va='top', fontsize=12)

    ax.set_title(panel_label)
    ax.set_xlabel(r"$r$ [Mpc/$h$]")
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlim(40.0, 120.0)


def _load_carrick_data(carrick_path, log_floor):
    """Load Carrick reconstruction data only."""
    delta_c, r_c = _load_los_delta(Path(carrick_path))
    vel_c, _ = _load_los_velocity(Path(carrick_path))

    if delta_c.ndim == 3:
        delta_c = delta_c[0]
    if vel_c.ndim == 3:
        vel_c = vel_c[0]

    delta_c_log = np.log10(np.clip(1.0 + delta_c, log_floor, None))

    return {
        "r": r_c,
        "delta_c": delta_c_log,
        "vel_c": vel_c,
    }


def plot_cluster_reconstruction_4panel(
    carrick_path=None,
    cluster_name=None,
    log_floor=1e-5,
    output=None,
    config_path=None,
    shift_fraction=0.10,
):
    """Plot 4-panel figure showing density, velocity, and shift scenarios.

    Panels:
    1. Isotropic (density)
    2. Isotropic (velocity)
    3. ZP shift (cluster moves, profile unchanged)
    4. H0 shift (both cluster and profile expand)
    """
    setup_style()

    from config import CANDEL_ROOT
    if carrick_path is None:
        carrick_path = CANDEL_ROOT / "data/Clusters/los_Clusters_Carrick2015.hdf5"

    data = _load_carrick_data(carrick_path, log_floor)

    cluster_names = _load_cluster_names()
    if cluster_name is None:
        raise ValueError("cluster_name is required.")
    match = np.where(np.char.lower(cluster_names) == cluster_name.lower())[0]
    if match.size == 0:
        raise ValueError(f"cluster_name '{cluster_name}' not found in cluster list.")
    cluster_index = int(match[0])

    if config_path is None:
        config_path = DATA_CONFIG_PATH
    r_marks = _load_cluster_zcmb(config_path)
    if cluster_index >= r_marks.shape[0]:
        raise IndexError(
            f"cluster_index {cluster_index} out of range for r_marks "
            f"(n_gal={r_marks.shape[0]})."
        )
    r_mark = float(r_marks[cluster_index])

    r = data["r"]
    delta_c = data["delta_c"][cluster_index]
    vel_c = data["vel_c"][cluster_index]

    # Scale velocity by beta
    beta = 0.43
    vel_c_scaled = vel_c * beta

    # Create 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.8),
                             gridspec_kw={"wspace": 0.30})

    # Panel 1: Isotropic (density) - no velocity annotation
    _plot_panel(axes[0], r, delta_c, r_mark, "Isotropic",
                r"$\log_{10}(1+\delta)$", v_at_cluster=None, show_ylabel=True)

    # Panel 2: Isotropic (velocity baseline) - with velocity annotation
    v_at_cluster = np.interp(r_mark, r, vel_c_scaled)
    _plot_panel(axes[1], r, vel_c_scaled, r_mark, "Isotropic",
                r"$v_{\rm los}$ [km/s]", v_at_cluster=v_at_cluster, show_ylabel=True)
    axes[1].axhline(0, color="0.5", lw=0.5, ls=":", alpha=0.5)

    # Panel 3: ZP shift - only cluster moves outward
    shift = 1.0 + shift_fraction
    r_mark_zp = r_mark * shift
    v_at_cluster_zp = np.interp(r_mark_zp, r, vel_c_scaled)
    _plot_panel(axes[2], r, vel_c_scaled, r_mark_zp,
                rf"ZP shift ($\equiv \delta H_0/H_0 = -{shift_fraction:.2f}$)",
                r"$v_{\rm los}$ [km/s]", v_at_cluster=v_at_cluster_zp, show_ylabel=False)
    axes[2].axhline(0, color="0.5", lw=0.5, ls=":", alpha=0.5)

    # Panel 4: H0 shift - everything expands
    r_mark_h0 = r_mark * shift
    r_shifted = r * shift
    v_at_cluster_h0 = np.interp(r_mark_h0, r_shifted, vel_c_scaled)
    _plot_panel(axes[3], r_shifted, vel_c_scaled, r_mark_h0,
                rf"$H_0$ shift ($\delta H_0/H_0 = -{shift_fraction:.2f}$)",
                r"$v_{\rm los}$ [km/s]", v_at_cluster=v_at_cluster_h0, show_ylabel=False)
    axes[3].axhline(0, color="0.5", lw=0.5, ls=":", alpha=0.5)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.18)
    if output is None:
        output = get_figure_path("rho_los.pdf")
    fig.savefig(str(output))
    plt.close(fig)


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
    shift_fraction=0.05,
):
    """Plot 4-panel figure: density, velocity, ZP shift, H0 shift.

    The first two panels are the original Carrick vs Manticore comparison.
    The third and fourth panels show velocity with shifted cluster positions.
    """
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

    # Extract data for this cluster
    r = data["r"]
    vel_c = data["vel_c"][cluster_index] * 0.43  # Scale by beta
    mean_vel_m = data["mean_vel_m"][cluster_index]
    std_vel_m = data["std_vel_m"][cluster_index]

    # Create 4-panel figure
    fig, axes = plt.subplots(4, 1, figsize=(5.2, 9.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0.08})

    # Rows 1-2: Original density + velocity (using existing _plot_cluster)
    _plot_cluster(axes[:2], data, cluster_index, cluster_name, r_mark)
    # Add "Isotropic" label to density panel (row 1)
    axes[0].text(0.97, 0.95, "Isotropic", transform=axes[0].transAxes,
                 ha='right', va='top', fontsize=12)
    # Add "Isotropic" label and v_los annotation to velocity panel (row 2)
    axes[1].text(0.97, 0.95, "Isotropic", transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=12)
    v_at_cluster = np.interp(r_mark, r, vel_c)
    axes[1].text(0.97, 0.82, rf"$v_{{\rm los}} = {v_at_cluster:.0f}$ km/s",
                 transform=axes[1].transAxes, ha='right', va='top', fontsize=12)

    # Row 3: ZP shift - cluster moves outward, profile unchanged
    shift = 1.0 + shift_fraction
    r_mark_zp = r_mark * shift
    _plot_velocity_panel(
        axes[2], r, vel_c, mean_vel_m, std_vel_m, r_mark_zp,
        title=rf"ZP anisotropy ($\equiv \delta H_0/H_0 = -{shift_fraction:.2f}$)",
        show_xlabel=False, r_mark_old=r_mark
    )

    # Row 4: H0 shift - everything expands
    r_mark_h0 = r_mark * shift
    r_shifted = r * shift
    vel_c_shifted = vel_c  # Same values, different r grid
    mean_vel_m_shifted = mean_vel_m
    std_vel_m_shifted = std_vel_m
    _plot_velocity_panel(
        axes[3], r_shifted, vel_c_shifted, mean_vel_m_shifted, std_vel_m_shifted,
        r_mark_h0,
        title=rf"$H_0$ anisotropy ($\delta H_0/H_0 = -{shift_fraction:.2f}$)",
        show_xlabel=True, r_mark_old=r_mark
    )

    # Add figure-level legend at top with all items on one line
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.55, 0.995), fontsize=12)

    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.06)
    if output is None:
        output = get_figure_path("rho_los.pdf")
    fig.savefig(str(output))
    plt.close(fig)


def main(cluster_name="Coma"):
    """Plot 4-panel LOS reconstruction for a single cluster.

    Default cluster is Coma.
    """
    plot_cluster_reconstruction(cluster_name=cluster_name)
    # plot_cluster_reconstruction_4panel(cluster_name=cluster_name)


if __name__ == "__main__":
    main()
