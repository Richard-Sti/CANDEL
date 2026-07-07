#!/usr/bin/env python3
"""Correlate R21 (no-Q) magnitude residuals with the Q index."""
import os
import sys

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import importlib  # noqa: E402

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402, F401
from scipy.stats import pearsonr  # noqa: E402

from candel import load_config  # noqa: E402
from candel.pvdata import CepheidData, to_mwcepheids_config  # noqa: E402

importlib.reload(scienceplots)

SAMPLES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "results", "R21", "C22+C27_anc-NGC4258", "samples.hdf5")


def load_posterior_means(path):
    with h5py.File(path, "r") as f:
        return {k: float(np.mean(f[k][:])) for k in f.keys()}


def main():
    # Load config and data
    config_path = os.path.join(
        REPO_ROOT, "scripts", "runs", "configs", "config_MWCepheids.toml")
    config = to_mwcepheids_config(
        load_config(config_path, replace_los_prior=False))

    # Both campaigns, no subset
    config_all = dict(config)
    config_all["data"] = dict(config["data"])
    config_all["data"].pop("which_subset", None)
    # Ensure anchors are loaded
    config_all.setdefault("model", {})["anchors"] = ["NGC4258"]
    data = CepheidData(config_all)

    # Posterior means from no-Q R21 run
    params = load_posterior_means(SAMPLES_PATH)
    M_H_1 = params["M_H_1"]
    b_W = params["b_W"]
    Z_W = params["Z_W"]
    delta_pi = params["delta_pi"]
    mu_N4258 = params["mu_NGC4258"]

    # --- MW campaigns ---
    split = data.split_by_campaign()
    mw_Q, mw_resid, mw_labels = [], [], []
    for campaign, camp_data in split.items():
        Q = np.asarray(camp_data.Q)
        if Q is None:
            continue

        logP = np.asarray(camp_data.logP)
        OH = np.asarray(camp_data.OH)
        mW_H = np.asarray(camp_data.mW_H)
        pi_obs = np.asarray(camp_data.pi_EDR3)

        M_pred = M_H_1 + b_W * (logP - 1) + Z_W * OH
        # Distance from corrected parallax
        pi_corr = pi_obs + delta_pi
        mu_pi = 5 * np.log10(np.clip(1.0 / pi_corr, 1e-10, None)) + 10
        resid = mW_H - M_pred - mu_pi

        mw_Q.append(Q)
        mw_resid.append(resid)
        mw_labels.append(np.full(len(Q), campaign))

    mw_Q = np.concatenate(mw_Q)
    mw_resid = np.concatenate(mw_resid)
    mw_labels = np.concatenate(mw_labels)

    # --- NGC4258 (filter stars with missing Q) ---
    anc = data.anchor_data["NGC4258"].filter_Q_valid()
    anc_Q = np.asarray(anc.Q)
    anc_logP = np.asarray(anc.logP)
    anc_OH = np.asarray(anc.OH)
    anc_mW_H = np.asarray(anc.mW_H)

    M_pred_anc = M_H_1 + b_W * (anc_logP - 1) + Z_W * anc_OH
    anc_resid = anc_mW_H - M_pred_anc - mu_N4258

    # --- Plot ---
    all_Q = np.concatenate([mw_Q, anc_Q])
    all_resid = np.concatenate([mw_resid, anc_resid])
    r, p = pearsonr(all_Q, all_resid)

    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(4, 3.2))

        # MW by campaign
        for camp, marker in [("C22", "o"), ("C27", "s")]:
            m = mw_labels == camp
            ax.scatter(mw_Q[m], mw_resid[m], s=15, marker=marker,
                       alpha=0.7, label=camp, zorder=2)

        ax.scatter(anc_Q, anc_resid, s=10, marker="D", alpha=0.5,
                   label="NGC 4258", zorder=2)

        # Best-fit line
        coeffs = np.polyfit(all_Q, all_resid, 1)
        Q_grid = np.linspace(all_Q.min(), all_Q.max(), 100)
        ax.plot(Q_grid, np.polyval(coeffs, Q_grid), "k--", lw=1, zorder=1)

        ax.set_xlabel(r"$\Delta Q$")
        ax.set_ylabel(r"$m^W_H - M_{\rm pred} - \mu$ [mag]")
        ax.legend(fontsize=7)
        ax.text(0.05, 0.95, f"r = {r:.3f}\nslope = {coeffs[0]:.3f}",
                transform=ax.transAxes, va="top", fontsize=7)

        fig.tight_layout()
        fout = os.path.join(os.path.dirname(__file__),
                            "Q_residual_correlation.pdf")
        fig.savefig(fout, dpi=200, bbox_inches="tight")
        print(f"Saved {fout}")
        plt.close(fig)


if __name__ == "__main__":
    main()
