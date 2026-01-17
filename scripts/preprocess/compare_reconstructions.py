"""
Compare LOS reconstructions for clusters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def compute_los_from_3d_field(
    delta_3d: np.ndarray,
    RA: np.ndarray,
    dec: np.ndarray,
    r: np.ndarray,
    box_side: float = 400.0,
    coordinate_frame: str = "galactic",
) -> np.ndarray:
    """
    Compute LOS delta from a 3D density field.

    Parameters
    ----------
    delta_3d : array
        3D overdensity field, shape (N, N, N) in galactic cartesian
    RA, dec : array
        Galaxy coordinates in ICRS degrees
    r : array
        Radial distances in h^-1 Mpc
    box_side : float
        Box side in h^-1 Mpc
    coordinate_frame : str
        "galactic" means field is in galactic cartesian, input is ICRS
        (matches Carrick2015 convention)

    Returns
    -------
    los_delta : array
        LOS delta, shape (n_gal, n_r)
    """
    from candel.util import radec_to_galactic

    N = delta_3d.shape[0]

    # Convert ICRS to galactic (l, b) - this is what Carrick's loader does
    if coordinate_frame == "galactic":
        ell, b = radec_to_galactic(RA, dec)
    else:
        # Assume input is already galactic
        ell, b = RA, dec

    # Unit vectors in galactic cartesian
    # x -> Galactic Center (l=0, b=0)
    # y -> direction of rotation (l=90, b=0)
    # z -> North Galactic Pole (b=90)
    ell_rad = np.deg2rad(ell)
    b_rad = np.deg2rad(b)
    cos_b = np.cos(b_rad)
    rhat = np.stack([
        cos_b * np.cos(ell_rad),
        cos_b * np.sin(ell_rad),
        np.sin(b_rad),
    ], axis=-1)

    # Observer at center of box
    observer_pos = np.array([0.0, 0.0, 0.0])

    # Create interpolator - use cell centers (matches CIC deposit and Carrick convention)
    cellsize = box_side / N
    coords = np.linspace(-box_side / 2 + cellsize / 2, box_side / 2 - cellsize / 2, N)
    interp = RegularGridInterpolator(
        (coords, coords, coords),
        delta_3d,
        bounds_error=False,
        fill_value=0.0,
    )

    # Compute positions along each LOS
    n_gal = len(RA)
    n_r = len(r)
    los_delta = np.zeros((n_gal, n_r), dtype=np.float32)

    for i in range(n_gal):
        positions = observer_pos + r[:, None] * rhat[i]
        los_delta[i] = interp(positions)

    return los_delta


def _load_cluster_names(path):
    names = np.genfromtxt(path, dtype="U32", usecols=0, skip_header=1)
    return np.asarray(names)


def _load_los_delta(path: Path):
    with h5py.File(path, "r") as f:
        if "los_delta" in f:
            delta = f["los_delta"][...]
        else:
            density = f["los_density"][...]
            delta = density - 1.0
        r = f["r"][...]
    lower = str(path).lower()
    if "manticore" in lower:
        density = delta + 1.0
        density = density / (0.3111 * 275.4)
        delta = density - 1.0
    elif "_cb1" in lower:
        density = delta + 1.0
        density = density / (0.307 * 275.4)
        delta = density - 1.0
    elif "_cb2" in lower:
        density = delta + 1.0
        density = density / (0.3111 * 275.4)
        delta = density - 1.0
    return delta.astype(np.float32), r.astype(np.float32)


def _interp_los(delta, r_in, r_out):
    if delta.ndim == 2:
        out = np.empty((delta.shape[0], len(r_out)), dtype=np.float32)
        for i in range(delta.shape[0]):
            out[i] = np.interp(r_out, r_in, delta[i], left=np.nan, right=np.nan)
        return out
    out = np.empty((delta.shape[0], delta.shape[1], len(r_out)), dtype=np.float32)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            out[i, j] = np.interp(r_out, r_in, delta[i, j], left=np.nan, right=np.nan)
    return out


def _load_cluster_zcmb(config_path):
    import candel
    from astropy.cosmology import FlatLambdaCDM

    config = candel.load_config(config_path)
    d = config["io"]["PV_main"]["Clusters"].copy()
    root = d.pop("root")
    data = candel.pvdata.load_clusters(root, **d)
    zcmb = data["zcmb"]
    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    r_comov = cosmo.comoving_distance(zcmb).value
    return r_comov.astype(np.float32)


def run_compare(
    carrick,
    manticore,
    zspace,
    output,
    ncols=6,
    nclusters=12,
    log_floor=1e-5,
    config_path=None,
    cluster_names_path="data/Clusters/ClustersData.txt",
    carrick_pre_iter=None,
):
    """
    Compare LOS reconstructions.

    Parameters
    ----------
    carrick : str
        Path to Carrick2015 LOS HDF5 file
    manticore : str
        Path to Manticore LOS HDF5 file
    zspace : str
        Path to 2M++ zspace LOS HDF5 file
    output : str
        Output figure path
    ncols : int
        Number of columns in figure
    nclusters : int
        Number of clusters to show
    log_floor : float
        Floor for log10(1+delta)
    config_path : str
        Path to config file for cluster redshifts
    cluster_names_path : str
        Path to cluster names file
    carrick_pre_iter : str or np.ndarray, optional
        Either path to pre-iteration Carrick LOS HDF5, or a 3D numpy array
        of the pre-iteration density field (will compute LOS on the fly)
    """
    delta_c, r_c = _load_los_delta(Path(carrick))
    delta_m, r_m = _load_los_delta(Path(manticore))
    delta_z, r_z = _load_los_delta(Path(zspace))

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

    # Handle pre-iteration Carrick field
    delta_pre = None
    r_pre = None
    if carrick_pre_iter is not None:
        if isinstance(carrick_pre_iter, (str, Path)):
            # Load from file
            delta_pre, r_pre = _load_los_delta(Path(carrick_pre_iter))
            if r_pre.shape != r_z.shape or not np.allclose(r_pre, r_z):
                delta_pre = _interp_los(delta_pre, r_pre, r_z)
                r_pre = r_z
        elif isinstance(carrick_pre_iter, np.ndarray):
            if carrick_pre_iter.ndim == 3:
                # 3D field - compute LOS on the fly
                # Load RA, dec from one of the existing LOS files
                with h5py.File(carrick, "r") as f:
                    RA = f["RA"][...]
                    dec = f["dec"][...]
                delta_pre = compute_los_from_3d_field(
                    carrick_pre_iter, RA, dec, r_z,
                    box_side=400.0, coordinate_frame="galactic",
                )
                r_pre = r_z
            elif carrick_pre_iter.ndim == 2:
                # Already LOS format
                delta_pre = carrick_pre_iter
                r_pre = r_z
        if delta_pre is not None and delta_pre.ndim == 3:
            delta_pre = delta_pre[0]

    n_gal = min(delta_c.shape[0], delta_m.shape[1], delta_z.shape[0])
    if delta_pre is not None:
        n_gal = min(n_gal, delta_pre.shape[0])
    delta_c = delta_c[:n_gal]
    delta_m = delta_m[:, :n_gal]
    delta_z = delta_z[:n_gal]
    if delta_pre is not None:
        delta_pre = delta_pre[:n_gal]

    r_marks = None
    cluster_names = None
    if cluster_names_path:
        try:
            cluster_names = _load_cluster_names(cluster_names_path)
        except OSError:
            cluster_names = None
    if cluster_names is None:
        raise ValueError("cluster names are required to label each subplot.")
    if config_path is not None:
        r_marks = _load_cluster_zcmb(config_path)[:n_gal]
        keep = r_marks < 225.0
        delta_c = delta_c[keep]
        delta_m = delta_m[:, keep]
        delta_z = delta_z[keep]
        if delta_pre is not None:
            delta_pre = delta_pre[keep]
        r_marks = r_marks[keep]
        n_gal = delta_c.shape[0]
        cluster_names = cluster_names[: len(keep)][keep]

    if nclusters is not None:
        n_gal = min(n_gal, nclusters)
        delta_c = delta_c[:n_gal]
        delta_m = delta_m[:, :n_gal]
        delta_z = delta_z[:n_gal]
        if delta_pre is not None:
            delta_pre = delta_pre[:n_gal]
        if r_marks is not None:
            r_marks = r_marks[:n_gal]
        if cluster_names is not None:
            cluster_names = cluster_names[:n_gal]

    delta_c = np.log10(np.clip(1.0 + delta_c, log_floor, None))
    delta_z = np.log10(np.clip(1.0 + delta_z, log_floor, None))
    delta_m_log = np.log10(np.clip(1.0 + delta_m, log_floor, None))
    mean_m = np.mean(delta_m_log, axis=0)
    std_m = np.std(delta_m_log, axis=0)
    if delta_pre is not None:
        delta_pre = np.log10(np.clip(1.0 + delta_pre, log_floor, None))

    r_mask = r_z <= 200.0
    if np.any(r_mask):
        diff = np.abs(delta_z[:, r_mask] - delta_c[:, r_mask])
        mean_dev = float(np.mean(diff))
        print(f"Mean |log10(1+delta) diff| zspace vs Carrick for r<=200: {mean_dev:.4f}")
        if delta_pre is not None:
            diff_pre = np.abs(delta_pre[:, r_mask] - delta_c[:, r_mask])
            mean_dev_pre = float(np.mean(diff_pre))
            print(f"Mean |log10(1+delta) diff| pre-iter vs Carrick for r<=200: {mean_dev_pre:.4f}")

    ncols = min(ncols, n_gal) if n_gal > 0 else ncols
    nrows = int(np.ceil(n_gal / ncols))
    figsize = (ncols * 3.0, nrows * 1.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        if idx >= n_gal:
            ax.axis("off")
            continue

        ax.plot(r_c, delta_c[idx], color="tab:blue", lw=0.6, label="Carrick2015")
        ax.plot(r_z, delta_z[idx], color="tab:orange", lw=0.6, label=r"2M++$\rho(z)$")
        ax.fill_between(
            r_m,
            mean_m[idx] - std_m[idx],
            mean_m[idx] + std_m[idx],
            color="tab:green",
            alpha=0.25,
            linewidth=0.0,
        )
        ax.plot(r_m, mean_m[idx], color="tab:green", lw=0.6, label="Manticore")
        if delta_pre is not None:
            ax.plot(r_pre, delta_pre[idx], color="tab:red", lw=0.6, label="Pre-iter")
        if r_marks is not None:
            ax.axvline(r_marks[idx], color="0.5", lw=0.6, ls="--", alpha=0.7)
        label = cluster_names[idx]
        ax.text(0.02, 0.85, label, transform=ax.transAxes, fontsize=6)
        if row == nrows - 1:
            ax.set_xlabel("r [Mpc/h]", fontsize=7)
        if col == 0:
            ax.set_ylabel(r"$\log_{10}(1+\delta)$", fontsize=7)
        ax.set_xlim(0.0, 226.0)

    fig.supxlabel("r [Mpc/h]", fontsize=10)
    legend_handles = [
        plt.Line2D([0], [0], color="tab:blue", lw=0.8, label="Carrick2015"),
        plt.Line2D([0], [0], color="tab:orange", lw=0.8, label=r"2M++$\rho(z)$"),
        plt.Line2D([0], [0], color="tab:green", lw=0.8, label="Manticore"),
    ]
    if delta_pre is not None:
        legend_handles.append(
            plt.Line2D([0], [0], color="tab:red", lw=0.8, label="Pre-iter Carrick")
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.94))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Carrick2015, Manticore, and 2mpp_zspace LOS delta."
    )
    parser.add_argument(
        "--carrick",
        default="data/Clusters/los_Clusters_Carrick2015.hdf5",
    )
    parser.add_argument(
        "--manticore",
        default="data/Clusters/los_Clusters_manticore.hdf5",
    )
    parser.add_argument(
        "--zspace",
        default="data/Clusters/los_Clusters_2mpp_zspace_galaxies.hdf5",
    )
    parser.add_argument(
        "--carrick-pre-iter",
        type=str,
        default=None,
        help="Path to pre-iteration Carrick LOS HDF5 or 3D .npy field",
    )
    parser.add_argument("--output", default="paper_clusters/figures/compare_reconstructions_galaxies.png")
    parser.add_argument("--ncols", type=int, default=6)
    parser.add_argument("--nclusters", type=int, default=12)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cluster-names-path", type=str, default="data/Clusters/ClustersData.txt")
    args = parser.parse_args()

    # Handle pre-iter field - can be HDF5 or .npy
    carrick_pre_iter = None
    if args.carrick_pre_iter is not None:
        if args.carrick_pre_iter.endswith(".npy"):
            carrick_pre_iter = np.load(args.carrick_pre_iter)
        else:
            carrick_pre_iter = args.carrick_pre_iter

    run_compare(
        args.carrick,
        args.manticore,
        args.zspace,
        args.output,
        ncols=args.ncols,
        nclusters=args.nclusters,
        config_path=args.config,
        cluster_names_path=args.cluster_names_path,
        carrick_pre_iter=carrick_pre_iter,
    )


if __name__ == "__main__":
    main()
