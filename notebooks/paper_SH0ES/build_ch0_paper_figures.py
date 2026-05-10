#!/usr/bin/env python
"""Regenerate CH0 paper figures from the current CH0_paper chains."""

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from scipy.stats import gaussian_kde, kstest
from astropy.cosmology import FlatLambdaCDM

import candel
from candel import SPEED_OF_LIGHT


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
RESULTS = ROOT / "results" / "CH0_paper"
TABLE = RESULTS / "table"
DIST = RESULTS / "distances"
MIXED = RESULTS / "mixed_selection"
FIGURES = Path("/mnt/users/rstiskalek/Papers/CH0/Figures")
CONFIG = ROOT / "scripts" / "runs" / "configs" / "config_CH0.toml"
DATA = ROOT / "data"

COLS = ["#87193d", "#1e42b9", "#d42a29", "#05dd6b", "#ee35d5"]


def read_samples(fname, keys=None):
    fname = Path(fname)
    with h5py.File(fname, "r") as f:
        group = f["samples"]
        if keys is None:
            keys = list(group.keys())
        if isinstance(keys, str):
            return group[keys][...]
        return {key: group[key][...] for key in keys}


def h0(fname):
    return read_samples(TABLE / fname, "H0").reshape(-1)


def mean_std(x):
    x = np.asarray(x).reshape(-1)
    return float(np.mean(x)), float(np.std(x))


def kde_line(ax, samples, label, color, fill=False, ls="-", bw=1.0):
    samples = np.asarray(samples).reshape(-1)
    x = np.linspace(np.percentile(samples, 0.1), np.percentile(samples, 99.9),
                    500)
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    y = kde(x)
    ax.plot(x, y, color=color, ls=ls, label=label)
    if fill:
        ax.fill_between(x, 0, y, color=color, alpha=0.25)


def plot_h0_comparison():
    curves = [
        ("SN mag. sel.", h0(
            "CH0_sel-SN_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_paper.hdf5"),
         COLS[4]),
        ("Host redshift sel.", h0(
            "CH0_sel-redshift_manticore_2MPP_MULTIBIN_N256_DES_V2_paper.hdf5"),
         COLS[3]),
    ]
    no_sel = h0("CH0_manticore_2MPP_MULTIBIN_N256_DES_V2_paper.hdf5")

    rng = np.random.default_rng(42)
    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for label, samples, color in curves:
            kde_line(ax, samples, label, color, fill=True, bw=2.0)
        kde_line(ax, no_sel, "No selection", "gray", fill=False, ls="--",
                 bw=3.0)
        kde_line(ax, rng.normal(73.04, 1.04, 300000), "SH0ES", COLS[2],
                 fill=False, ls=":", bw=2.0)
        kde_line(ax, rng.normal(67.4, 0.5, 300000), "Planck", COLS[1],
                 fill=False, ls=":", bw=2.0)
        ax.set_xlabel(r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel("Normalised PDF")
        ax.set_xlim(64, 77.5)
        handles, legend_labels = ax.get_legend_handles_labels()
        main_handles, main_labels = [], []
        ref_handles, ref_labels = [], []
        for handle, label in zip(handles, legend_labels):
            if label in ("SH0ES", "Planck"):
                ref_handles.append(handle)
                ref_labels.append(label)
            else:
                main_handles.append(handle)
                main_labels.append(label)
        legend_main = ax.legend(
            main_handles, main_labels,
            loc="upper right",
            fontsize="large",
            title="This work (Cepheid hosts only)",
            title_fontsize="x-large",
            frameon=True,
            edgecolor="black",
            facecolor="white",
            fancybox=True,
            framealpha=1,
        )
        ax.add_artist(legend_main)
        ax.legend(ref_handles, ref_labels, loc="upper left",
                  fontsize="large")
        fig.tight_layout()
        fig.savefig(FIGURES / "H0_comparison.pdf", dpi=500,
                    bbox_inches="tight")
        plt.close(fig)


def percentile_summary(samples):
    lo, med, hi = np.percentile(np.asarray(samples).reshape(-1), [16, 50, 84])
    return med, med - lo, hi - med


def plot_h0_stacked():
    manticore_col = COLS[4]
    carrick_col = COLS[3]
    lit_col = "#444444"
    spec = [
        ("SN mag.\nselection", "Manticore", manticore_col,
         "CH0_sel-SN_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_paper.hdf5"),
        ("SN mag.\nselection", "Carrick", carrick_col,
         "CH0_sel-SN_magnitude_Carrick2015_paper.hdf5"),
        ("Host redshift\nselection", "Manticore", manticore_col,
         "CH0_sel-redshift_manticore_2MPP_MULTIBIN_N256_DES_V2_paper.hdf5"),
        ("Host redshift\nselection", "Carrick", carrick_col,
         "CH0_sel-redshift_Carrick2015_paper.hdf5"),
    ]
    literature = [
        (None, None, "", None),
        (73.04, 1.04, "SH0ES\n(Riess+ 2022)", COLS[2]),
        (73.2, 0.9, "Breuval+ 2024", lit_col),
        (72.9, 2.3, "Kenworthy+ 2022", lit_col),
        (70.4, 1.9, "CCHP", lit_col),
        (68.52, 0.62, "DESI 2024 BAO", lit_col),
        (67.4, 0.5, "Planck", COLS[1]),
        (68.22, 0.36, "Planck+ACT", COLS[1]),
        (67.24, 0.35, "Planck+SPT+ACT", lit_col),
    ]

    labels, datasets, colors, fnames = zip(*spec)
    vals = np.array([percentile_summary(h0(fname)) for fname in fnames])
    med, errm, errp = vals[:, 0], vals[:, 1], vals[:, 2]
    unique = list(dict.fromkeys(labels))
    y_positions = {label: i for i, label in enumerate(unique[::-1])}
    y = np.array([y_positions[label] for label in labels])

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(3.35, 4.5))
        lw = plt.rcParams["lines.linewidth"]
        for i, x in enumerate(med):
            offset = -0.1 if datasets[i] == "Manticore" else 0.1
            ax.errorbar(x, y[i] + offset, xerr=[[errm[i]], [errp[i]]],
                        fmt="none", ecolor=colors[i],
                        elinewidth=1.5 * lw, capsize=5,
                        label=datasets[i] if i < 2 else None)

        fid = [i for i, (label, data) in enumerate(zip(labels, datasets))
               if "SN mag." in label and data == "Manticore"][0]
        ax.axvspan(med[fid] - errm[fid], med[fid] + errp[fid],
                   color=manticore_col, alpha=0.20, zorder=-3)

        lit_base = -1
        lit_y = []
        for j, (mu, sig, label, color) in enumerate(literature):
            ypos = lit_base - j
            lit_y.append((ypos, label))
            if mu is None:
                continue
            ax.errorbar(mu, ypos, xerr=sig, fmt="none", ecolor="#222222",
                        elinewidth=1.5 * lw, capsize=5)
        ax.axhline(y=lit_base, color="black", lw=1.5 * lw, ls="--")

        yticks = list(range(len(unique))) + [yy for yy, _ in lit_y]
        yticklabels = unique[::-1] + [label for _, label in lit_y]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.text(0.05, 0.99, "This work", ha="left", va="top",
                weight="bold", transform=ax.transAxes, fontsize="small",
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="square,pad=0.2"))
        ax.text(0.05, lit_base - 0.2, "Literature", ha="left", va="top",
                weight="bold", fontsize="small",
                transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="square", linewidth=0.8))
        manticore_handle = mlines.Line2D(
            [], [], color=manticore_col, lw=2,
            label=r"\texttt{Manticore-Local}")
        carrick_handle = mlines.Line2D(
            [], [], color=carrick_col, lw=2, label=r"Carrick+2015")
        ax.legend(handles=[manticore_handle, carrick_handle],
                  loc="lower left", ncol=2, bbox_to_anchor=(-0.5, 1.03),
                  frameon=False)
        ax.set_ylim(lit_base - len(literature) + 0.5, len(unique))
        ax.set_xlabel(r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel("")
        ax.minorticks_off()
        ax.set_xlim(64, 77.8)
        fig.tight_layout()
        fig.savefig(FIGURES / "H0_stacked.pdf", dpi=500)
        plt.close(fig)


def plot_h0_proportion():
    xs, means, stds = [], [], []
    for i in range(36):
        fname = (
            MIXED
            / f"CH0_sel-SN_magnitude_or_redshift_Nmag_Nmag{i}_"
              "manticore_2MPP_MULTIBIN_N256_DES_V2_paper_mixed.hdf5"
        )
        samples = read_samples(fname, "H0")
        xs.append(i)
        means.append(np.mean(samples))
        stds.append(np.std(samples))

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(3.35, 3.0))
        ax.plot(xs, means, c=COLS[1])
        ax.fill_between(xs, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds), color=COLS[1],
                        alpha=0.18)
        ax.set_xlabel("Number of hosts selected by SN magnitude")
        ax.set_ylabel(r"$\langle H_0 \rangle ~ [\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$")
        ax.set_xlim(0, 35)
        fig.tight_layout()
        fig.savefig(FIGURES / "H0_proportion.pdf", dpi=450,
                    bbox_inches="tight")
        plt.close(fig)


def plot_manticore_corner():
    fnames = [
        TABLE / "CH0_sel-SN_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_paper.hdf5",
        TABLE / "CH0_sel-redshift_manticore_2MPP_MULTIBIN_N256_DES_V2_paper.hdf5",
    ]
    truths = [{
        "dict": {
            "H0": 73.04, "M_W": -5.8999543, "b_W": -3.298,
            "Z_W": -0.217, "dZP": -0.074, "mu_LMC": 18.475368,
            "mu_N4258": 29.390965, "mu_M31": 24.373648,
            "M_B": -19.253,
        },
        "color": "blue",
    }]
    candel.plot_corner_from_hdf5(
        fnames,
        keys=[
            "H0", "M_W", "b_W", "Z_W", "dZP", "mu_LMC", "mu_M31",
            "mu_N4258", "Vext_mag", "Vext_ell", "Vext_b", "sigma_v",
            "log_rho_t", "alpha_low", "alpha_high", "M_B",
        ],
        labels=["SN mag. sel.", "Host redshift sel."],
        apply_ell_offset=True,
        filename=str(FIGURES / "Manticore_corner.pdf"),
        fontsize=24,
        legend_fontsize=40,
        filled=False,
        ranges={"alpha_low": [0.0, None], "alpha_high": [0.0, None]},
        truths=truths,
        show_fig=False,
    )


def plot_anchor_distances():
    fnames = [
        DIST / "CH0_noVext_uniform_mu_host_no_Cepheid_redshift_paper.hdf5",
        DIST / "CH0_noVext_no_Cepheid_redshift_paper.hdf5",
        DIST / "CH0_noVext_sel-SN_magnitude_no_Cepheid_redshift_paper.hdf5",
    ]
    candel.plot_corner_from_hdf5(
        fnames,
        keys=["mu_LMC", "mu_N4258", "M_W"],
        labels=[
            r"Uniform in $\mu$",
            r"Uniform in $V$",
            r"Uniform in $V$ ($m_{\rm SN}$ sel.)",
        ],
        filled=False,
        fontsize=18,
        filename=str(FIGURES / "anchor_distances.pdf"),
        show_fig=False,
    )


def plot_mu_host(data):
    mu_r2 = read_samples(DIST / "CH0_noVext_no_Cepheid_redshift_paper.hdf5",
                         "mu_host")
    mu_r2_sn = read_samples(
        DIST / "CH0_noVext_sel-SN_magnitude_no_Cepheid_redshift_paper.hdf5",
        "mu_host")
    mu_unif = read_samples(
        DIST / "CH0_noVext_uniform_mu_host_no_Cepheid_redshift_paper.hdf5",
        "mu_host")
    r2_mean, r2_std = mu_r2.mean(axis=0), mu_r2.std(axis=0)
    sn_mean, sn_std = mu_r2_sn.mean(axis=0), mu_r2_sn.std(axis=0)
    unif_mean, unif_std = mu_unif.mean(axis=0), mu_unif.std(axis=0)
    diff_unif = r2_mean - unif_mean
    diff_sn = sn_mean - unif_mean
    err_unif = np.sqrt(r2_std**2 + unif_std**2)
    err_sn = np.sqrt(sn_std**2 + unif_std**2)
    mask = data["czcmb_cepheid_host"] < 3300

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(3.35, 3.0))
        ax.errorbar(r2_mean[mask], diff_unif[mask], xerr=r2_std[mask],
                    yerr=err_unif[mask], fmt="o", color="black",
                    ecolor="lightgray", capsize=3, ms=1.5,
                    label=r"$\mathcal{U}(V) - \mathcal{U}(\mu)$")
        ax.errorbar(r2_mean[mask], diff_sn[mask], yerr=err_sn[mask],
                    fmt="o", color=COLS[1], ecolor="#9fb0ff", capsize=3,
                    ms=1.5,
                    label=r"$\mathcal{U}(V),\,m_{\rm SN}~{\rm sel.} - \mathcal{U}(\mu)$")
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel(r"$\mu_{\rm host}^{\mathcal{U}(V)} ~ [\mathrm{mag}]$")
        ax.set_ylabel(r"$\Delta \mu ~ [\mathrm{mag}]$")
        ax.set_ylim(-0.3, 0.5)
        ax.legend(frameon=True)
        fig.tight_layout(pad=0)
        fig.savefig(FIGURES / "mu_host.pdf", bbox_inches="tight", dpi=450)
        plt.close(fig)
    print(f"Delta mu U(V)-U(mu): {diff_unif.mean():.4f} +/- {err_unif.mean():.4f}")
    print(f"Delta mu SN sel-U(mu): {diff_sn.mean():.4f} +/- {err_sn.mean():.4f}")


def plot_mu_host_cz(data):
    mu = read_samples(
        DIST / "CH0_noVext_sel-SN_magnitude_no_Cepheid_redshift_paper.hdf5",
        "mu_host")
    mu_mean, mu_std = mu.mean(axis=0), mu.std(axis=0)
    cz = data["czcmb_cepheid_host"]
    cz_err = np.ones_like(cz) * 250
    czrange = np.linspace(300, 5250, 1000)
    mu_sh0es = FlatLambdaCDM(H0=73.04, Om0=0.3).distmod(
        czrange / SPEED_OF_LIGHT).value
    mu_planck = FlatLambdaCDM(H0=67.4, Om0=0.3).distmod(
        czrange / SPEED_OF_LIGHT).value
    mask = cz < 3300

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(6.8, 3.5))
        ax.errorbar(cz[mask], mu_mean[mask], xerr=cz_err[mask],
                    yerr=mu_std[mask], fmt="o", color="black",
                    capsize=4, alpha=0.75)
        ax.plot(czrange, mu_sh0es, color=COLS[3],
                label=r"$H_0 = 73.04~\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}$")
        ax.plot(czrange, mu_planck, color=COLS[4],
                label=r"$H_0 = 67.4~\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}$")
        ax.set_xlabel(r"$c z_{\rm CMB} ~ [\mathrm{km}\,\mathrm{s}^{-1}]$")
        ax.set_ylabel(r"$\mu_{\rm host} ~ [\mathrm{mag}]$")
        ax.set_xlim(50, 3500)
        ax.set_ylim(mu_planck.min())
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "mu_host_cz.pdf", dpi=500,
                    bbox_inches="tight")
        plt.close(fig)


def plot_host_histograms():
    raise RuntimeError(
        "Do not regenerate SH0ES_host_histograms.pdf from this script. "
        "Use the detailed paper_plots.ipynb implementation with Poisson "
        "errors and selection-boundary annotations."
    )
    data_sel = candel.pvdata.load_SH0ES_separated(
        DATA / "SH0ES", cepheid_host_cz_cmb_max=3300)
    data_pp = candel.pvdata.load_PantheonPlus(
        DATA / "Pantheon+", return_all=True, removed_PV_from_covmat=False)
    bins_mag = np.linspace(9, 14.0, 12)
    bins_cz = np.linspace(0, 3300, 12)
    mask_mag_pp = data_pp["mag"] < 14
    mask_cz_pp = data_pp["zcmb"] < 3300 / SPEED_OF_LIGHT
    print("KS mag", kstest(data_sel["mag_SN_unique_Cepheid_host"],
                           data_pp["mag"][mask_mag_pp]))
    print("KS cz", kstest(data_sel["czcmb_cepheid_host"],
                          data_pp["zcmb"][mask_cz_pp] * SPEED_OF_LIGHT))
    with plt.style.context("science"):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
        axes[0].hist(data_pp["mag"][mask_mag_pp], bins=bins_mag,
                     density=True, histtype="step", label=r"Pantheon+",
                     color=COLS[0])
        axes[0].hist(data_sel["mag_SN_unique_Cepheid_host"], bins=bins_mag,
                     density=True, histtype="step",
                     label="Cepheid host galaxies", color=COLS[3])
        axes[0].set_xlabel(r"$m_{\rm SN} ~ [\mathrm{mag}]$")
        axes[0].set_ylabel("Normalised counts per bin")
        axes[0].legend()
        axes[1].hist(data_pp["zcmb"][mask_cz_pp] * SPEED_OF_LIGHT,
                     bins=bins_cz, density=True, histtype="step",
                     color=COLS[0], label=r"Pantheon+")
        axes[1].hist(data_sel["czcmb_cepheid_host"], bins=bins_cz,
                     density=True, histtype="step",
                     label="Cepheid host galaxies", color=COLS[3])
        axes[1].set_xlabel(r"$cz_{\rm CMB} ~ [\mathrm{km}\,\mathrm{s}^{-1}]$")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "SH0ES_host_histograms.pdf", dpi=450,
                    bbox_inches="tight")
        plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = candel.pvdata.load_SH0ES_from_config(CONFIG)
    # Keep SH0ES_host_histograms.pdf from the original paper_plots.ipynb
    # implementation; it has extra selection-boundary and Poisson-error
    # annotations that this rebuild script should not overwrite.
    plot_anchor_distances()
    plot_mu_host(data)
    plot_mu_host_cz(data)
    plot_h0_comparison()
    plot_h0_stacked()
    plot_h0_proportion()
    plot_manticore_corner()


if __name__ == "__main__":
    main()
