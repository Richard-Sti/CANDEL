"""Plot maser spots coloured by acceleration clump membership."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, ".")
from candel.pvdata.megamaser_data import load_megamaser_spots


GALAXIES = [
    ("CGCG074-064", 6541.0),
    ("NGC5765b", 8547.0),
    ("NGC6264", 10128.0),
    ("NGC6323", 7845.0),
    ("UGC3789", 3325.0),
]


def identify_clumps(data, dv_tol=10.0):
    """Return list of clumps, each a dict with indices and metadata."""
    am = data["accel_measured"]
    aw = data["accel_weight"]
    a_vals = data["a"][am]
    sa_vals = data["sigma_a"][am]
    v_vals = data["velocity"][am]
    idx_meas = np.where(am)[0]

    keys = np.round(a_vals, 8) + 1j * np.round(sa_vals, 8)
    unique_keys = np.unique(keys)

    clumps = []
    for uk in unique_keys:
        group_mask = keys == uk
        group_v = v_vals[group_mask]
        group_idx = idx_meas[group_mask]
        order = np.argsort(group_v)
        group_v = group_v[order]
        group_idx = group_idx[order]
        gaps = np.diff(group_v) > dv_tol
        clump_ids = np.concatenate([[0], np.cumsum(gaps)])
        for cid in range(clump_ids[-1] + 1):
            cmask = clump_ids == cid
            if cmask.sum() < 2:
                continue
            clumps.append({
                "indices": group_idx[cmask],
                "a": float(uk.real),
                "sigma_a": float(uk.imag),
                "N": int(cmask.sum()),
            })
    return clumps


def plot_galaxy_row(axs, data, galaxy_name):
    """Plot one row: x-y, x-v, y-v, a-v with clumps coloured."""
    v = data["velocity"]
    x = data["x"]
    y = data["y"]
    a = data["a"]
    am = data["accel_measured"]

    clumps = identify_clumps(data)
    in_clump = np.zeros(len(v), dtype=bool)
    for cl in clumps:
        in_clump[cl["indices"]] = True

    # Assign colours to clumps
    cmap = plt.cm.tab10
    clump_colors = np.full(len(v), -1, dtype=int)
    for i, cl in enumerate(clumps):
        clump_colors[cl["indices"]] = i

    s_bg, s_cl = 8, 25
    alpha_bg = 0.3

    panels = [
        (x, y, r"$\theta_x$ [mas]", r"$\theta_y$ [mas]"),
        (v, x, r"$v$ [km/s]", r"$\theta_x$ [mas]"),
        (v, y, r"$v$ [km/s]", r"$\theta_y$ [mas]"),
        (v, a, r"$v$ [km/s]", r"$a$ [km/s/yr]"),
    ]

    for ax, (px, py, xlabel, ylabel) in zip(axs, panels):
        # Background: non-clump spots
        bg = ~in_clump
        ax.scatter(px[bg], py[bg], s=s_bg, c="0.6", alpha=alpha_bg,
                   edgecolors="none", rasterized=True)

        # For a-v panel, only plot spots with measured acceleration
        if ylabel == r"$a$ [km/s/yr]":
            bg_a = bg & am
            ax.scatter(px[bg_a], py[bg_a], s=s_bg, c="0.6", alpha=alpha_bg,
                       edgecolors="none", rasterized=True, zorder=2)
            # Clear and replot only measured spots
            ax.cla()
            ax.scatter(px[bg_a], py[bg_a], s=s_bg, c="0.6", alpha=alpha_bg,
                       edgecolors="none", rasterized=True)

        # Clumps
        for i, cl in enumerate(clumps):
            idx = cl["indices"]
            color = cmap(i % 10)
            label = (f"$a$={cl['a']:+.2f}, "
                     f"$N$={cl['N']}")
            ax.scatter(px[idx], py[idx], s=s_cl, c=[color], edgecolors="k",
                       linewidths=0.3, zorder=3, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Galaxy label on leftmost panel
    axs[0].set_title(galaxy_name, fontsize=10, loc="left", fontweight="bold")

    # Legend on rightmost panel if clumps exist
    if clumps:
        handles = []
        for i, cl in enumerate(clumps):
            color = cmap(i % 10)
            handles.append(Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=color,
                markeredgecolor="k", markeredgewidth=0.3, markersize=5,
                label=f"$a$={cl['a']:+.2f}, N={cl['N']}"))
        axs[-1].legend(handles=handles, fontsize=5, loc="best",
                       framealpha=0.7, handletextpad=0.3)


if __name__ == "__main__":
    fig, axes = plt.subplots(
        5, 4, figsize=(14, 16),
        gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    for row, (galaxy, vsys) in enumerate(GALAXIES):
        data = load_megamaser_spots("data/Megamaser", galaxy, v_sys_obs=vsys)
        plot_galaxy_row(axes[row], data, galaxy)

    fig.savefig("results/Maser/accel_clumps.png", dpi=200, bbox_inches="tight")
    print("Saved results/Maser/accel_clumps.png")
