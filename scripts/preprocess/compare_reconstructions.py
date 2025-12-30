"""
Compare LOS reconstructions for clusters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


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


def run_compare(carrick, manticore, zspace, output, ncols=6, log_floor=1e-5, nclusters=12, config_path=None):
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

    n_gal = min(delta_c.shape[0], delta_m.shape[1], delta_z.shape[0], nclusters)
    delta_c = delta_c[:n_gal]
    delta_m = delta_m[:, :n_gal]
    delta_z = delta_z[:n_gal]
    r_marks = None
    if config_path is not None:
        r_marks = _load_cluster_zcmb(config_path)[:n_gal]
        keep = r_marks < 225.0
        delta_c = delta_c[keep]
        delta_m = delta_m[:, keep]
        delta_z = delta_z[keep]
        r_marks = r_marks[keep]
        n_gal = delta_c.shape[0]

    delta_c = np.log10(np.clip(1.0 + delta_c, log_floor, None))
    delta_z = np.log10(np.clip(1.0 + delta_z, log_floor, None))
    delta_m_log = np.log10(np.clip(1.0 + delta_m, log_floor, None))
    mean_m = np.mean(delta_m_log, axis=0)
    std_m = np.std(delta_m_log, axis=0)

    r_mask = r_z <= 200.0
    if np.any(r_mask):
        diff = np.abs(delta_z[:, r_mask] - delta_c[:, r_mask])
        mean_dev = float(np.mean(diff))
        print(f"Mean |log10(1+delta) diff| vs Carrick for r<=200: {mean_dev:.4f}")

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
        ax.plot(r_z, delta_z[idx], color="tab:orange", lw=0.6, label="2mpp_zspace")
        ax.fill_between(
            r_m,
            mean_m[idx] - std_m[idx],
            mean_m[idx] + std_m[idx],
            color="tab:green",
            alpha=0.25,
            linewidth=0.0,
        )
        ax.plot(r_m, mean_m[idx], color="tab:green", lw=0.6, label="Manticore")
        if r_marks is not None:
            ax.axvline(r_marks[idx], color="0.5", lw=0.6, ls="--", alpha=0.7)
        ax.text(0.02, 0.85, f"{idx}", transform=ax.transAxes, fontsize=6)
        if row == nrows - 1:
            ax.set_xlabel("r [Mpc/h]", fontsize=7)
        if col == 0:
            ax.set_ylabel("log10(1+delta)", fontsize=7)
        ax.set_xlim(0.0, 226.0)

    fig.supxlabel("r [Mpc/h]", fontsize=10)
    fig.supylabel("log10(1+delta)", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.98))
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
    parser.add_argument("--output", default="results/compare_reconstructions.png")
    parser.add_argument("--ncols", type=int, default=6)
    parser.add_argument("--nclusters", type=int, default=12)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    run_compare(
        args.carrick,
        args.manticore,
        args.zspace,
        args.output,
        ncols=args.ncols,
        nclusters=args.nclusters,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
