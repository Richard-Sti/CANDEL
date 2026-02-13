"""Replot radial Vext magnitude profiles for LT, YT, and LTYT."""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from interpax import interp1d

from config import get_results_path, get_figure_path, setup_style, CANDEL_ROOT
from candel.inference import postprocess_samples
from candel.model.model import ClustersModel
from candel.pvdata.data import load_PV_dataframes
from candel.util import fprint
from candel.cosmography import Redshift2Distance


SPEED_OF_LIGHT = 299_792.458  # km / s


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_nested(config, key_path, default=None):
    """Recursively access a nested value using a slash-separated key."""
    keys = key_path.split("/")
    current = config
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return default
        current = current[k]
    return current


def percent_h0_to_bulkflow(r, percent, *, H0=100.0, q0=-0.53):
    """Convert fractional H0 dipole (percent) to bulk-flow magnitude at r."""
    frac = percent / 100.0
    r = np.asarray(r)
    return frac * (H0 * r + q0 * (H0**2) * r**2 / SPEED_OF_LIGHT)


def load_samples(h5_path):
    """Load samples and log_density from a saved HDF5 file."""
    import h5py
    samples = {}
    log_density = None
    with File(h5_path, "r") as f:
        grp = f["samples"]
        for key in grp.keys():
            if isinstance(grp[key], h5py.Dataset):
                samples[key] = grp[key][()]
        if "log_density" in f:
            log_density = f["log_density"][()]
    return samples, log_density


# -----------------------------------------------------------------------------
# Main plotting function - 3 vertically stacked panels for base/fine/finest
# -----------------------------------------------------------------------------

def plot_Vext_radmag_comparison(
    all_samples, all_rknots, all_labels,
    data=None, h0_samples=None,
    all_log_densities=None, all_methods=None,
    r_eval_size=1000, show_fig=True, filename=None
):
    """Plot radial Vext magnitude with 3 vertically stacked panels.

    Parameters
    ----------
    all_samples : list of dict
        List of MCMC samples dicts (base, fine, finest).
    all_rknots : list of array
        List of knot positions for each configuration.
    all_labels : list of str
        Labels for each panel (e.g., ["Base", "Fine", "Finest"]).
    data : PVDataFrame or list
        Data for redshift histogram.
    h0_samples : dict
        Samples from dipH0 run for equivalent H0 dipole overlay.
    all_log_densities : list of array or None
        Log density arrays for each configuration (for MAP computation).
    all_methods : list of str or None
        Interpolation methods for each configuration.
    """
    r2d = Redshift2Distance()

    # Extract zcmb and Y from data
    zcmb = None
    Y = None
    if data is not None:
        try:
            if isinstance(data, list):
                zcmb_list = []
                Y_list = []
                for df in data:
                    data_dict = df.data if hasattr(df, "data") else df
                    zcmb_list.append(np.asarray(data_dict["zcmb"]))
                    Y_val = data_dict.get("Y", None)
                    if Y_val is not None:
                        Y_list.append(np.asarray(Y_val))
                    else:
                        Y_list.append(np.full(len(data_dict["zcmb"]), -1.0))
                zcmb = np.concatenate(zcmb_list)
                Y = np.concatenate(Y_list)
            else:
                data_dict = data.data if hasattr(data, "data") else data
                zcmb = np.asarray(data_dict["zcmb"])
                Y = data_dict.get("Y", None)
        except Exception:
            zcmb = None
            Y = None

    # Compute distance range
    r_cap = 1000.0
    max_rknot = max(np.max(rk) for rk in all_rknots)
    if zcmb is not None:
        dist = r2d(zcmb, h=1.0)
        dist_max = float(np.nanmax(dist))
        r_max = min(max(dist_max, max_rknot), r_cap)
        dist_max = min(dist_max, r_cap)
    else:
        dist_max = max_rknot
        r_max = min(max_rknot, r_cap)

    r = jnp.linspace(0.0, r_max, r_eval_size)

    # Compute H0 dipole reference from actual samples
    # dipH0 runs have H0_dipole_mag (fractional dH/H), H0_dipole_ell, H0_dipole_b
    h0_pct_cen = None
    h0_pct_std = None
    h0_label = None
    if h0_samples is not None:
        # Try H0_dipole_mag first (direct dipH0 runs), then dH_over_H_dipole (legacy)
        dH_over_H = h0_samples.get("H0_dipole_mag", None)
        if dH_over_H is None:
            dH_over_H = h0_samples.get("dH_over_H_dipole", None)

        if dH_over_H is not None:
            pct = 100.0 * np.asarray(dH_over_H)
            h0_pct_cen = float(np.mean(pct))
            h0_pct_std = float(np.std(pct))
            # Try both key naming conventions
            ell_samples = h0_samples.get("H0_dipole_ell", h0_samples.get("zeropoint_dipole_ell", None))
            b_samples = h0_samples.get("H0_dipole_b", h0_samples.get("zeropoint_dipole_b", None))
            if ell_samples is not None and b_samples is not None:
                H0_ell = float(np.mean(ell_samples))
                H0_b = float(np.mean(b_samples))
                H0_std_ell = float(np.std(ell_samples))
                H0_std_b = float(np.std(b_samples))
                h0_label = (f"Equiv. $H_0$ dipole: ${h0_pct_cen:.1f}\\% \\pm {h0_pct_std:.1f}\\%$ "
                            f"at $(\\ell, b) = ({H0_ell:.0f} \\pm {H0_std_ell:.0f}°, "
                            f"{H0_b:.0f} \\pm {H0_std_b:.0f}°)$")
            else:
                h0_label = f"Equiv. $H_0$ dipole: ${h0_pct_cen:.1f}\\% \\pm {h0_pct_std:.1f}\\%$"

    if h0_pct_cen is None:
        fprint("WARNING: No H0_dipole_mag or dH_over_H_dipole in h0_samples, using default")
        h0_pct_cen = 17.25
        h0_pct_std = 2.0
        h0_label = f"Equiv. $H_0$ dipole: ${h0_pct_cen:.1f}\\%$ (default)"

    # Create figure: 1 histogram + 3 profile panels
    n_panels = len(all_samples)
    fig, axes = plt.subplots(
        n_panels + 1, 1, figsize=(8, 2 + 1.5 * n_panels), sharex=True,
        gridspec_kw={"height_ratios": [0.6] + [1.0] * n_panels}
    )
    ax_hist = axes[0]
    ax_profiles = axes[1:]

    # Top panel: distance histogram
    if zcmb is not None and zcmb.size > 0 and dist_max > 0:
        dist = np.asarray(r2d(zcmb, h=1.0))
        n_bins = 25
        bins = np.linspace(0.0, dist_max, n_bins + 1)
        if Y is not None:
            Y_arr = np.asarray(Y)
            has_Y = Y_arr > 0
            no_Y = Y_arr <= 0
            counts_no_Y, bin_edges = np.histogram(dist[no_Y], bins=bins)
            counts_has_Y, _ = np.histogram(dist[has_Y], bins=bin_edges)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            width = bin_edges[1] - bin_edges[0]
            ax_hist.bar(bin_centers, counts_has_Y, width=width, alpha=0.7,
                        label=f'With $Y_{{SZ}}$ (N={np.sum(has_Y)})',
                        color="#7570b3", edgecolor='black', linewidth=0.5)
            ax_hist.bar(bin_centers, counts_no_Y, width=width, alpha=0.7,
                        bottom=counts_has_Y,
                        label=f'All (N={len(dist)})',
                        color="#d95f02", edgecolor='black', linewidth=0.5)
            ax_hist.legend(fontsize=11, loc="upper right")
        else:
            ax_hist.hist(dist, bins=bins, color="C0", alpha=0.7, edgecolor='black')
        ax_hist.set_ylabel("N")
        ax_hist.grid(alpha=0.3)
        ax_hist.set_xlim(0.0, r_max)
    else:
        ax_hist.text(0.5, 0.5, "No redshift data", ha="center",
                     va="center", transform=ax_hist.transAxes)
        ax_hist.set_ylabel("N")

    # Secondary x-axis (redshift) on histogram
    def z_to_dist(z):
        return r2d(z, h=1.0)

    def dist_to_z(d):
        z_grid = np.linspace(1e-5, 1.0, 1000)
        d_grid = z_to_dist(z_grid)
        return np.interp(d, d_grid, z_grid)

    ax_top = ax_hist.secondary_xaxis('top', functions=(dist_to_z, z_to_dist))
    ax_top.set_xlabel(r"$z_\mathrm{CMB}$")

    # Default to None lists if not provided
    if all_log_densities is None:
        all_log_densities = [None] * len(all_samples)
    if all_methods is None:
        all_methods = ["cubic"] * len(all_samples)

    # Profile panels
    for panel_idx, (samples, rknot, label, log_density, method) in enumerate(
            zip(all_samples, all_rknots, all_labels, all_log_densities,
                all_methods)):
        ax = ax_profiles[panel_idx]
        rknot = np.asarray(rknot)

        Vmag = samples["Vext_radmag_mag"]

        # H0 dipole reference
        bf = percent_h0_to_bulkflow(r, h0_pct_cen)
        bu = percent_h0_to_bulkflow(r, h0_pct_cen + h0_pct_std)
        bl = percent_h0_to_bulkflow(r, h0_pct_cen - h0_pct_std)
        if panel_idx == 0:
            ax.plot(r, bf, linestyle="--", color="gray", label=h0_label, zorder=1)
        else:
            ax.plot(r, bf, linestyle="--", color="gray", zorder=1)
        ax.fill_between(r, bl, bu, color="gray", alpha=0.2, zorder=0)

        # Violin plots at knot positions
        xmin, xmax = float(r[0]), float(r[-1])
        dx = 0.01 * (xmax - xmin)

        knot_positions = []
        knot_data = []
        for idx, rk in enumerate(rknot):
            if jnp.isclose(rk, xmin):
                x_pos = xmin + dx
            elif jnp.isclose(rk, xmax):
                x_pos = xmax - dx
            else:
                x_pos = float(rk)
            knot_positions.append(x_pos)
            knot_data.append(Vmag[:, idx])
            ax.axvline(x_pos, color="black", linestyle="--", zorder=-1, alpha=0.2)

        # Violin width based on knot spacing
        knot_spacing = np.min(np.diff(sorted(knot_positions))) if len(knot_positions) > 1 else 100
        violin_width = knot_spacing * 0.7

        parts = ax.violinplot(knot_data, positions=knot_positions, widths=violin_width,
                              showmeans=False, showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('C0')
            pc.set_edgecolor('C0')
            pc.set_alpha(0.6)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        ax.set_ylabel(r"$|V_{\mathrm{ext}}|$ [km/s]")
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, r_max)

        # Direction label from radmag samples
        ell_samples = samples.get("Vext_radmag_ell", None)
        b_samples = samples.get("Vext_radmag_b", None)
        if ell_samples is not None and b_samples is not None:
            mean_ell = float(np.mean(ell_samples))
            mean_b = float(np.mean(b_samples))
            std_ell = float(np.std(ell_samples))
            std_b = float(np.std(b_samples))
            dir_label = (f"$(\\ell, b) = ({mean_ell:.0f} \\pm {std_ell:.0f}°, "
                        f"{mean_b:.0f} \\pm {std_b:.0f}°)$")
            ax.text(0.02, 0.95, dir_label, transform=ax.transAxes, fontsize=11, va='top')

        if panel_idx == 0:
            ax.legend(loc="upper right", fontsize=11)

    # X-axis label only on bottom
    ax_profiles[-1].set_xlabel(r"$r~[h^{-1}\,\mathrm{Mpc}]$")

    fig.tight_layout()

    if filename is not None:
        fprint(f"saving radial Vext_mag plot to {filename}")
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Run function
# -----------------------------------------------------------------------------

def plot_radmag_run(label, radmag_stem, diph0_stem, fine_stem, finest_stem):
    """Load and plot base/fine/finest comparison for one relation."""
    out_path = get_figure_path(f"radmag_{label}.pdf")

    # Load base
    config_path = get_results_path(f"{radmag_stem}.toml")
    results_path = get_results_path(f"{radmag_stem}.hdf5")
    fprint(f"[{label}] loading base model from {config_path}")
    base_model = ClustersModel(str(config_path))
    base_samples_raw, base_log_density = load_samples(str(results_path))
    base_samples = postprocess_samples(base_samples_raw)
    base_rknot = base_model.kwargs_Vext["rknot"]
    base_method = base_model.kwargs_Vext.get("method", "cubic")

    # Load data
    fprint(f"[{label}] loading data")
    data = load_PV_dataframes(str(config_path), local_root=str(CANDEL_ROOT))

    # Load dipH0 samples
    diph0_path = get_results_path(f"{diph0_stem}.hdf5")
    fprint(f"[{label}] loading dipH0 samples from {diph0_path}")
    h0_samples_raw, _ = load_samples(str(diph0_path))
    h0_samples = postprocess_samples(h0_samples_raw)

    # Collect all configurations
    all_samples = [base_samples]
    all_rknots = [base_rknot]
    all_labels = ["Base"]
    all_log_densities = [base_log_density]
    all_methods = [base_method]

    # Load fine
    fine_path = get_results_path(f"{fine_stem}.hdf5")
    fine_config = get_results_path(f"{fine_stem}.toml")
    if fine_path.exists():
        fprint(f"[{label}] loading fine samples")
        fine_model = ClustersModel(str(fine_config))
        fine_samples_raw, fine_log_density = load_samples(str(fine_path))
        fine_samples = postprocess_samples(fine_samples_raw)
        all_samples.append(fine_samples)
        all_rknots.append(fine_model.kwargs_Vext["rknot"])
        all_labels.append("Fine")
        all_log_densities.append(fine_log_density)
        all_methods.append(fine_model.kwargs_Vext.get("method", "cubic"))

    # Load finest
    finest_path = get_results_path(f"{finest_stem}.hdf5")
    finest_config = get_results_path(f"{finest_stem}.toml")
    if finest_path.exists():
        fprint(f"[{label}] loading finest samples")
        finest_model = ClustersModel(str(finest_config))
        finest_samples_raw, finest_log_density = load_samples(str(finest_path))
        finest_samples = postprocess_samples(finest_samples_raw)
        all_samples.append(finest_samples)
        all_rknots.append(finest_model.kwargs_Vext["rknot"])
        all_labels.append("Finest")
        all_log_densities.append(finest_log_density)
        all_methods.append(finest_model.kwargs_Vext.get("method", "cubic"))

    fprint(f"[{label}] plotting to {out_path}")
    plot_Vext_radmag_comparison(
        all_samples, all_rknots, all_labels,
        data=data, h0_samples=h0_samples,
        all_log_densities=all_log_densities, all_methods=all_methods,
        show_fig=False, filename=str(out_path)
    )


def main():
    setup_style()
    runs = [
        ("LT", "Carrick2015_LT_noMNR_radmagVext", "Carrick2015_LT_noMNR_dipH0",
         "Carrick2015_LT_noMNR_radmagVext-fine", "Carrick2015_LT_noMNR_radmagVext-finest"),
        ("YT", "Carrick2015_YT_noMNR_radmagVext_hasY", "Carrick2015_YT_noMNR_dipH0_hasY",
         "Carrick2015_YT_noMNR_radmagVext-fine_hasY", "Carrick2015_YT_noMNR_radmagVext-finest_hasY"),
        ("LTYT", "Carrick2015_LTYT_noMNR_radmagVext_hasY", "Carrick2015_LTYT_noMNR_dipH0_hasY",
         "Carrick2015_LTYT_noMNR_radmagVext-fine_hasY", "Carrick2015_LTYT_noMNR_radmagVext-finest_hasY"),
    ]
    for label, radmag_stem, diph0_stem, fine_stem, finest_stem in runs:
        plot_radmag_run(label, radmag_stem, diph0_stem, fine_stem, finest_stem)


if __name__ == "__main__":
    main()
