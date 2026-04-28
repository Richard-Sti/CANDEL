# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Reproduce the figures of the joint-S8 PV paper.

Reads MCMC runs from ``results/S8`` and writes PDF figures + paper
tables. Replaces ``main_S8.ipynb``.

Examples:
    python main_S8.py --plots all
    python main_S8.py --plots s8_posterior s8_comparison
    python main_S8.py --plots all --bias both
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from os.path import dirname, abspath, join


def _heavy_imports():
    """Defer slow imports so ``--help`` returns instantly."""
    global np, interp1d, plt, sns, File, candel, posterior_agreement
    global get_key_all, compute_S8_all, compute_fsigma8_lin_all
    global makedirs, exists, md5, sys_path

    from hashlib import md5  # noqa: F401
    from os import makedirs  # noqa: F401
    from os.path import exists  # noqa: F401
    from sys import path as sys_path  # noqa: F401

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (no plt.show hangs).

    import numpy as np  # noqa: F401
    from scipy.interpolate import interp1d  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401
    import seaborn as sns  # noqa: F401
    import scienceplots  # noqa: F401  (registers the "science" mpl style)
    from h5py import File  # noqa: F401

    sys_path.insert(0, "/Users/rstiskalek/Projects/candel")
    import candel  # noqa: F401
    import posterior_agreement  # noqa: F401

    from utils import (
        get_key_all, compute_S8_all, compute_fsigma8_lin_all,
    )

    # Re-bind into module globals for the rest of the script.
    g = globals()
    g["np"] = np
    g["interp1d"] = interp1d
    g["plt"] = plt
    g["sns"] = sns
    g["File"] = File
    g["candel"] = candel
    g["posterior_agreement"] = posterior_agreement
    g["get_key_all"] = get_key_all
    g["compute_S8_all"] = compute_S8_all
    g["compute_fsigma8_lin_all"] = compute_fsigma8_lin_all
    g["makedirs"] = makedirs
    g["exists"] = exists
    g["md5"] = md5

# -----------------------------------------------------------------------------
# Constants and run paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = dirname(abspath(__file__))
ROOT = "/Users/rstiskalek/Projects/CANDEL/results/S8"
PLOTS_DIR = "/Users/rstiskalek/Projects/CANDEL/plots/S8"
SR_CACHE_DIR = join(SCRIPT_DIR, "sr_interp_cache")

# SR interp grid for sigma8_nl -> sigma8_lin. Wider than the package
# default [0.6, 1.1] so all chain values lie strictly inside (highest
# observed value across runs is sigma8_nl ~ 1.21 for Pantheon+).
SR_GRID_MIN = 0.4
SR_GRID_MAX = 1.3
SR_GRID_N = 100

COLS = ["#87193d", "#1e42b9", "#d42a29", "#05dd6b", "#ee35d5", "#f5c000"]

# Per-survey run filenames per bias model.
SURVEY_LABELS = ["CF4 TFR W1", "CF4 TFR i", "Pantheon+", "SDSS FP", "6dF FP"]
SURVEY_TAGS = ["CF4_W1", "CF4_i", "PantheonPlus", "SDSS_FP", "6dF_FP"]


def _per_survey_runs(bias):
    return [(f"precomputed_los_Carrick2015_{tag}_{bias}.hdf5", lab)
            for tag, lab in zip(SURVEY_TAGS, SURVEY_LABELS)]


# Joint run with per-survey Vext (Vext NOT shared, only sigma_v + beta shared).
# This is the configuration recommended by the referee.
def _joint_run(bias):
    return (f"precomputed_los_Carrick2015_CF4_W1,CF4_i,6dF_FP,SDSS_FP,"
            f"PantheonPlus_{bias}_shared-sigma_v+beta.hdf5")


# Extra joint variants that drop one or both FP catalogues. These runs
# exist only for ``linear`` and ``quadratic`` bias and share Vext
# (sigma_v + Vext + beta all shared).
JOINT_VARIANTS = [
    ("Joint (no 6dF)",
     "CF4_W1,CF4_i,SDSS_FP,PantheonPlus"),
    ("Joint (no 6dF, no SDSS)",
     "CF4_W1,CF4_i,PantheonPlus"),
]


def _joint_variant_run(cats, bias):
    return (f"precomputed_los_Carrick2015_{cats}_{bias}"
            f"_shared-sigma_v+Vext+beta.hdf5")

# Effective survey redshifts (computed once and cached).
# NOTE: "Joint (no 6dF)" reuses the 5-survey joint value as a placeholder;
# recompute properly when convenient.
ZEFF = {"Joint":           0.030413097,
        "Joint (no 6dF)":  0.030413097,
        "CF4 TFR W1":      0.01655924,
        "CF4 TFR i":       0.02208739,
        "SDSS FP":         0.03684078,
        "6dF FP":          0.03507398,
        "Pantheon+":       0.02632573}

# Planck S8 (TT,TE,EE+lowE+lensing).
PLANCK_S8 = (0.832, 0.013)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _sr_cache_path(b2c):
    """Cache file path keyed on cosmology parameters and grid range."""
    p = b2c.cosmo_params
    key = "_".join(f"{k}={p[k]}" for k in sorted(p))
    key += f"_grid={SR_GRID_MIN},{SR_GRID_MAX},{SR_GRID_N}"
    h = md5(key.encode()).hexdigest()[:10]
    return join(SR_CACHE_DIR, f"sr_interp_{h}.npz")


def attach_sr_interp(b2c):
    """Build (or load) the SR interpolator on ``b2c`` over a wider grid
    than the package default and with ``bounds_error=True`` so any
    out-of-range sample raises instead of silently extrapolating.
    """
    if b2c.method != "sr":
        return
    cache = _sr_cache_path(b2c)
    if exists(cache):
        d = np.load(cache)
        x, y = d["x"], d["y"]
        print(f"  loaded SR interp cache: {cache}")
    else:
        print(f"  building SR interp on [{SR_GRID_MIN}, {SR_GRID_MAX}] "
              f"with {SR_GRID_N} points (first run only)...")
        x = np.linspace(SR_GRID_MIN, SR_GRID_MAX, SR_GRID_N)
        y = np.array([b2c._find_linear_sigma8(s) for s in x])
        makedirs(SR_CACHE_DIR, exist_ok=True)
        np.savez(cache, x=x, y=y)
        print(f"  saved SR interp cache: {cache}")
    b2c._sr_interp = interp1d(x, y, kind="cubic", bounds_error=True)


def _planck_fs8_band(z, mu, sig, n_draws, seed):
    """Return (lo, mid, hi) of fsigma8(z) over Planck draws, cached on disk."""
    key = (f"planck_fs8_z={z[0]},{z[-1]},{len(z)}_n={n_draws}_seed={seed}_"
           f"mu={sorted(mu.items())}_sig={sorted(sig.items())}")
    h = md5(key.encode()).hexdigest()[:10]
    cache = join(SR_CACHE_DIR, f"planck_fs8_{h}.npz")
    if exists(cache):
        d = np.load(cache)
        print(f"  loaded Planck fsigma8 cache: {cache}")
        return d["lo"], d["mid"], d["hi"]

    from colossus.cosmology import cosmology
    from tqdm import trange

    a = 1.0 / (1.0 + z)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        d = {k: rng.normal(mu[k], sig[k]) for k in sig}
        d["flat"] = True
        draws.append(d)

    fs8_all = []
    for i in trange(len(draws), desc="Planck fsigma8"):
        cosmo = cosmology.setCosmology("tmp", draws[i])
        D = cosmo.growthFactorUnnormalized(z)
        fs8_all.append(np.gradient(np.log(D), np.log(a))
                       * cosmo.sigma(8.0, z))
    fs8_all = np.stack(fs8_all)
    lo, mid, hi = np.percentile(fs8_all, [16, 50, 84], axis=0)

    makedirs(SR_CACHE_DIR, exist_ok=True)
    np.savez(cache, lo=lo, mid=mid, hi=hi)
    print(f"  saved Planck fsigma8 cache: {cache}")
    return lo, mid, hi


def _check_files(fnames):
    missing = [f for f in fnames if not exists(f)]
    if missing:
        raise FileNotFoundError(
            "Missing run files:\n  " + "\n  ".join(missing))


def load_per_survey(bias="linear"):
    """Return ``(fnames, labels)`` for the per-survey runs of ``bias``."""
    runs = _per_survey_runs(bias)
    fnames = [join(ROOT, f) for f, _ in runs]
    labels = [lab for _, lab in runs]
    _check_files(fnames)
    return fnames, labels


def load_joint_beta(bias="linear"):
    """Return ``beta`` samples from the joint per-Vext run of ``bias``."""
    p = join(ROOT, _joint_run(bias))
    _check_files([p])
    with File(p, "r") as f:
        return f["samples"]["beta"][...]


def load_joint_variant_beta(cats, bias):
    """Return ``beta`` samples from a Vext-shared joint variant."""
    p = join(ROOT, _joint_variant_run(cats, bias))
    _check_files([p])
    with File(p, "r") as f:
        return f["samples"]["beta"][...]


def _joint_variant_rows(beta2cosmo, bias):
    """Build ``(label, beta, S8, fs8)`` tuples for each joint variant."""
    rows = []
    for lab, cats in JOINT_VARIANTS:
        b = load_joint_variant_beta(cats, bias)
        rows.append((lab, b,
                     beta2cosmo.compute_S8(b),
                     beta2cosmo.compute_fsigma8_linear(b)))
    return rows


def collect_S8_and_fs8(fnames, labels, beta2cosmo, joint=True,
                       bias="linear"):
    """Return ``(beta_list, S8_list, fs8_list, labels_with_joint)``.

    If ``joint`` is True, the joint-run derived posteriors are prepended
    with label ``"Joint"``.
    """
    beta_list = get_key_all(fnames, "beta")
    S8_list = compute_S8_all(fnames, beta2cosmo)
    fs8_list = compute_fsigma8_lin_all(fnames, beta2cosmo)

    if joint:
        beta_joint = load_joint_beta(bias)
        S8_joint = beta2cosmo.compute_S8(beta_joint)
        fs8_joint = beta2cosmo.compute_fsigma8_linear(beta_joint)
        beta_list = [beta_joint] + beta_list
        S8_list = [S8_joint] + S8_list
        fs8_list = [fs8_joint] + fs8_list
        labels = ["Joint"] + list(labels)

    return beta_list, S8_list, fs8_list, labels


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def print_table_b1_beta_S8(fnames, labels, beta_list, S8_list, fs8_list,
                           all_labels, title="Table 1",
                           extra_joint_rows=None):
    """Reproduce paper Table 1 (b1, beta*, S8, fsigma8_L) from the chains.

    ``extra_joint_rows`` is an optional list of
    ``(label, beta, S8, fs8)`` tuples for additional joint variants that
    are appended after the main "Joint" row (b1 and z_eff print as '-').
    """
    b1_list = get_key_all(fnames, "b1")
    fmt_pm = lambda mu, sd: f"{mu:.3f} +/- {sd:.3f}"
    extra_widths = [len(l) for l, *_ in (extra_joint_rows or [])]
    lab_w = max([12, *extra_widths])
    header = (f"{'Sample':<{lab_w}} {'z_eff':>6}  {'b1':>16}  "
              f"{'beta*':>16}  {'S8':>16}  {'fsigma8_L':>16}")
    sep = "-" * len(header)
    print(f"\n=== {title}: b1, beta*, S8, fsigma8_L ===")
    print(header)
    print(sep)
    for lab, b1 in zip(labels, b1_list):
        i = all_labels.index(lab)
        print(f"{lab:<{lab_w}} {ZEFF[lab]:>6.3f}  "
              f"{fmt_pm(np.mean(b1), np.std(b1)):>16}  "
              f"{fmt_pm(np.mean(beta_list[i]), np.std(beta_list[i])):>16}  "
              f"{fmt_pm(np.mean(S8_list[i]), np.std(S8_list[i])):>16}  "
              f"{fmt_pm(np.mean(fs8_list[i]), np.std(fs8_list[i])):>16}")
    i = all_labels.index("Joint")
    print(f"{'Joint':<{lab_w}} {ZEFF['Joint']:>6.3f}  {'-':>16}  "
          f"{fmt_pm(np.mean(beta_list[i]), np.std(beta_list[i])):>16}  "
          f"{fmt_pm(np.mean(S8_list[i]), np.std(S8_list[i])):>16}  "
          f"{fmt_pm(np.mean(fs8_list[i]), np.std(fs8_list[i])):>16}")
    for lab, beta, S8, fs8 in extra_joint_rows or []:
        print(f"{lab:<{lab_w}} {'-':>6}  {'-':>16}  "
              f"{fmt_pm(np.mean(beta), np.std(beta)):>16}  "
              f"{fmt_pm(np.mean(S8), np.std(S8)):>16}  "
              f"{fmt_pm(np.mean(fs8), np.std(fs8)):>16}")


def _gaussian_tension(a, b):
    """Fallback pairwise tension assuming Gaussianity."""
    return abs(np.mean(a) - np.mean(b)) / np.sqrt(np.var(a) + np.var(b))


def print_table_tension(per_survey_labels, S8_list, all_labels,
                        title="Table 2"):
    """Reproduce paper Table 2: pairwise S8 tension in sigma (lower
    triangle). Falls back to a Gaussian tension when the posterior-
    agreement estimator fails (e.g. returns inf for non-overlapping
    chains)."""
    chains = [S8_list[all_labels.index(l)] for l in per_survey_labels]
    n = len(per_survey_labels)
    print(f"\n=== {title}: pairwise S8 tension (sigma) ===")
    colw = max(len(l) for l in per_survey_labels) + 2
    print(" " * colw + "".join(f"{l:>{colw}}" for l in per_survey_labels))
    for i, lab_i in enumerate(per_survey_labels):
        cells = []
        for j in range(n):
            if j > i:
                cells.append(f"{'':>{colw}}")
            elif j == i:
                cells.append(f"{'-':>{colw}}")
            else:
                try:
                    s = posterior_agreement.compute_agreement(
                        (chains[i], chains[j])).sigma
                    if not np.isfinite(s):
                        raise ValueError("non-finite sigma")
                    cells.append(f"{s:>{colw}.2f}")
                except Exception:
                    s = _gaussian_tension(chains[i], chains[j])
                    cells.append(f"{s:>{colw - 1}.2f}*")
        print(f"{lab_i:<{colw}}" + "".join(cells))
    print("  (* = Gaussian-approximation fallback)")


def plot_s8_posterior(S8_list, labels, savedir):
    bw = 0.25
    mu, sig = PLANCK_S8
    rng = np.random.default_rng(seed=42)
    planck = rng.normal(loc=mu, scale=sig, size=100_000)

    # Drop the joint chain — dominated by SDSS, so leave only per-survey.
    pairs = [(S8, lab) for S8, lab in zip(S8_list, labels)
             if lab != "Joint"]

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(6.4, 3.5))
        for i, (S8, lab) in enumerate(pairs):
            agr = posterior_agreement.compute_agreement(
                (planck, S8)).sigma
            print(f"  {lab:<12} S8 = {np.mean(S8):.3f} ± {np.std(S8):.3f} "
                  f"| {agr:.2f} σ from Planck")
            zo = 0 if lab == "6dF FP" else 1
            sns.kdeplot(S8, label=lab, fill=True, color=COLS[i],
                        bw_method=bw, ax=ax, zorder=zo)

        planck_band = ax.axvspan(mu - sig, mu + sig, color="k", alpha=0.45,
                                 label=r"\textit{Planck} $1\sigma$",
                                 zorder=-1)
        ax.set_xlim(0.61, 1.02)
        ax.set_xlabel(r"$S_8$")
        ax.set_ylabel("Normalised PDF")

        survey_handles, survey_labels = ax.get_legend_handles_labels()
        survey_handles = [h for h, l in zip(survey_handles, survey_labels)
                          if l != r"\textit{Planck} $1\sigma$"]
        survey_labels = [l for l in survey_labels
                         if l != r"\textit{Planck} $1\sigma$"]
        leg1 = ax.legend(survey_handles, survey_labels, loc="upper left")
        ax.add_artist(leg1)
        ax.legend([planck_band], [r"\textit{Planck} $1\sigma$"],
                  loc="upper right")

        fig.tight_layout()
        out = join(savedir, "S8_posterior.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=450)
        plt.close(fig)
        print(f"  saved {out}")


def plot_s8_comparison(S8_list, labels, savedir, S8_list_q=None):
    """Stacked S8 comparison vs literature.

    If ``S8_list_q`` is given, overlay quadratic-bias values on the same
    rows as the linear ones using open markers and dashed error bars.
    """
    means_this = [np.mean(x) for x in S8_list]
    lower_this = [np.percentile(x, 16) for x in S8_list]
    upper_this = [np.percentile(x, 84) for x in S8_list]
    if S8_list_q is not None:
        means_q = [np.mean(x) for x in S8_list_q]
        lower_q = [np.percentile(x, 16) for x in S8_list_q]
        upper_q = [np.percentile(x, 84) for x in S8_list_q]
        elo_q = np.array(means_q) - np.array(lower_q)
        ehi_q = np.array(upper_q) - np.array(means_q)

    lit_blocks = {
        "pv": [
            ("Huterer+2017", 0.780, 0.087),
            ("Nusser 2017",  0.776, 0.120),
            ("Boruah+2019",  0.776, 0.033),
            ("Said+2020",    0.637, 0.054),
            ("Boubel+2024",  0.632, 0.044),
        ],
        "wl": [
            ("KiDS-Legacy",                 0.814, (0.016, 0.021)),
            ("DES-Y3, KiDS-1000,\nHSC-DR1", 0.795, (0.015, 0.017)),
            ("DES Y3, KiDS-1000",           0.790, (0.014, 0.018)),
            ("DES Y3",                      0.776, 0.017),
        ],
        "clustering": [
            ("DESI DR1", 0.836, 0.035),
            ("DES Y3",   0.778, (0.031, 0.037)),
        ],
        "clusters": [
            ("SRG/eROSITA",  0.86,  0.01),
            ("Planck-SZ",    0.774, 0.034),
            ("SPT-SZ",       0.739, 0.041),
            ("DES-Clusters", 0.650, 0.050),
        ],
        "cmb": [
            ("Planck", PLANCK_S8[0], PLANCK_S8[1]),
        ],
    }

    lit_names, lit_means, lit_errs = [], [], []
    for grp in ["pv", "wl", "clustering", "clusters", "cmb"]:
        for n, m, s in lit_blocks[grp]:
            lit_names.append(n)
            lit_means.append(m)
            lit_errs.append(s)

    def _bounds(means, errs):
        lo, hi = [], []
        for m, e in zip(means, errs):
            if isinstance(e, tuple):
                lo.append(m - e[0]); hi.append(m + e[1])
            else:
                lo.append(m - e); hi.append(m + e)
        return lo, hi

    lower_lit, upper_lit = _bounds(lit_means, lit_errs)

    names = list(labels) + lit_names
    means = means_this + lit_means
    lower = lower_this + lower_lit
    upper = upper_this + upper_lit
    err_lo = np.array(means) - np.array(lower)
    err_hi = np.array(upper) - np.array(means)

    n_this = len(labels)
    block_sizes = [n_this, len(lit_blocks["pv"]), len(lit_blocks["wl"]),
                   len(lit_blocks["clustering"]),
                   len(lit_blocks["clusters"]), len(lit_blocks["cmb"])]
    block_labels = ["This work", "Peculiar velocity\n(literature)",
                    "Weak lensing", "Clustering", "Cluster abundance",
                    "CMB"]
    tab = ["#263636", "#BBA044", "#77B6C7", "#C96F32", "#80CC14",
           "#AB36A7"][::-1]
    colors = sum(([tab[k]] * sz for k, sz in enumerate(block_sizes)), [])

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(3.5, 2.625 * 2.2))
        ax.set_ylim(0.5, len(names) + 0.5)

        # 1σ band of Planck.
        planck_m = lit_blocks["cmb"][0][1]
        planck_s = lit_blocks["cmb"][0][2]
        if isinstance(planck_s, tuple):
            ax.axvspan(planck_m - planck_s[0], planck_m + planck_s[1],
                       color=colors[-1], alpha=0.2, zorder=0)
        else:
            ax.axvspan(planck_m - planck_s, planck_m + planck_s,
                       color=colors[-1], alpha=0.2, zorder=0)

        for i, (mu, elo, ehi, c) in enumerate(
                zip(means, err_lo, err_hi, colors), start=1):
            ax.errorbar(mu, i, xerr=[[elo], [ehi]], fmt="o", color=c,
                        capsize=4, ms=3.5)

        if S8_list_q is not None:
            dy = 0.25  # vertical offset for quadratic markers
            for i in range(len(labels)):
                ax.errorbar(means_q[i], i + 1 + dy,
                            xerr=[[elo_q[i]], [ehi_q[i]]],
                            fmt="s", mfc="none", color=colors[i],
                            capsize=4, ms=3.5, linestyle="--")

        cumulative = np.cumsum(block_sizes)
        for y in cumulative[:-1] + 0.5:
            ax.axhline(y, color="0.5", linestyle="--", lw=0.8)

        starts = np.concatenate(([0], cumulative[:-1]))
        tform = ax.get_yaxis_transform()
        for start, size, lab in zip(starts, block_sizes, block_labels):
            if size == 0:
                continue
            ax.text(0.05, start + 0.8, lab, transform=tform, ha="left",
                    va="top", fontsize="small", weight="bold",
                    bbox=dict(facecolor="white", edgecolor="0.5",
                              alpha=0.7, pad=2.0))

        ax.set_yticks(range(1, len(names) + 1), names, fontsize="small")
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_xlabel(r"$S_8$")
        ax.invert_yaxis()
        fig.tight_layout()
        out = join(savedir, "S8_comparison.pdf")
        fig.savefig(out, dpi=450, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_vext_corner(fnames, labels, savedir):
    out = join(savedir, "Vext.pdf")
    candel.plot_corner_from_hdf5(
        fnames,
        fontsize=18,
        filled=True,
        labels=labels,
        cols=COLS[1:1 + len(fnames)],   # skip "Joint" colour
        keys=["Vext_mag", "Vext_ell", "Vext_b"],
        filename=out,
        show_fig=False,
    )
    print(f"  saved {out}")


def plot_fs8_z(fs8_list, labels, savedir, fs8_list_q=None):
    z = np.linspace(0.015, 0.05, 600)

    mu = dict(H0=67.36, Om0=0.315, Ob0=0.0493, sigma8=0.811, ns=0.965)
    sig = dict(H0=0.54, Om0=0.007, Ob0=0.0003, sigma8=0.006, ns=0.004)

    lo, mid, hi = _planck_fs8_band(z, mu, sig, n_draws=100, seed=0)

    literature = {
        "Stahl+2021 (SNe)":     (0.020, 0.390, 0.022),
        "Boubel+2024 (CF4)":    (0.017, 0.350, 0.030),
        "Said+2020 (SDSS+6dF)": (0.035, 0.338, 0.027),
        "Boruah+2020":          (0.023, 0.400, 0.017),
    }
    lit_markers = ["D", "^", "v", "P"]

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(6.4, 3.5))
        q_alpha = 0.45
        for i, (fs8, lab) in enumerate(zip(fs8_list, labels)):
            if lab in {"Joint", "Joint (no 6dF)"}:
                continue
            ax.errorbar(ZEFF[lab], np.median(fs8), yerr=np.std(fs8),
                        fmt="o", capsize=3, color=COLS[i], label=lab)
        if fs8_list_q is not None:
            for i, (fs8q, lab) in enumerate(zip(fs8_list_q, labels)):
                if lab in {"Joint", "Joint (no 6dF)"}:
                    continue
                ax.errorbar(ZEFF[lab] * 1.05, np.median(fs8q),
                            yerr=np.std(fs8q), fmt="s", capsize=3,
                            mfc="none", color=COLS[i], linestyle="--",
                            alpha=q_alpha)
        band = ax.fill_between(z, lo, hi, alpha=0.3)
        ax.plot(z, mid)

        lit_handles = []
        for (name, (z_lit, val, err)), m in zip(
                literature.items(), lit_markers):
            h = ax.errorbar(z_lit, val, yerr=err, fmt=m, ms=5, color="0.3",
                            mfc="none", capsize=3, label=name)
            lit_handles.append(h)

        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$f\sigma_8(z)$")
        ax.set_xlim(z.min(), z.max())

        # This work above the panel, left-aligned with axes; on the
        # right (same y) a small legend distinguishing linear vs
        # quadratic when both are plotted. Literature bottom-right.
        this_handles, this_labels = ax.get_legend_handles_labels()
        keep = [(h, l) for h, l in zip(this_handles, this_labels)
                if l not in literature]
        leg1 = ax.legend([h for h, _ in keep], [l for _, l in keep],
                         loc="lower left", bbox_to_anchor=(0.0, 1.02),
                         ncol=3, fontsize=8, frameon=False)
        ax.add_artist(leg1)

        extra_artists = [leg1]
        if fs8_list_q is not None:
            from matplotlib.lines import Line2D
            bias_handles = [
                Line2D([], [], marker="o", color="0.3", linestyle="-",
                       label="linear"),
                Line2D([], [], marker="s", mfc="none", color="0.3",
                       linestyle="--", alpha=q_alpha, label="quadratic"),
            ]
            leg_bias = ax.legend(handles=bias_handles, loc="lower right",
                                 bbox_to_anchor=(1.0, 1.02), fontsize=8,
                                 frameon=False)
            ax.add_artist(leg_bias)
            extra_artists.append(leg_bias)

        leg2 = ax.legend(lit_handles, list(literature.keys()),
                         loc="lower right", fontsize="x-small", ncol=1,
                         frameon=True)
        extra_artists.append(leg2)
        ax.text(0.02, 0.98, r"$\Lambda$CDM (Planck)",
                transform=ax.transAxes, ha="left", va="top",
                bbox=dict(facecolor=band.get_facecolor()[0], alpha=0.3,
                          edgecolor="black", boxstyle="round,pad=0.2"))
        out = join(savedir, "fs8_z.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=450,
                    bbox_extra_artists=tuple(extra_artists))
        plt.close(fig)
        print(f"  saved {out}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

ALL_PLOTS = ["s8_posterior", "s8_comparison", "vext_corner", "fs8_z"]


def main():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "--plots", nargs="+", default=["all"],
        choices=ALL_PLOTS + ["all"], metavar="PLOT",
        help=f"Which plots to make. Choices: {', '.join(ALL_PLOTS)}, all. "
             "Default: all.")
    parser.add_argument(
        "--bias", default="linear",
        choices=["linear", "quadratic", "both"],
        help="Bias model. With 'both', fs8_z and s8_comparison overlay "
             "the quadratic-bias values; other plots stay linear. "
             "Default: linear.")
    parser.add_argument(
        "--savedir", default=PLOTS_DIR,
        help=f"Output directory. Default: {PLOTS_DIR}.")
    args = parser.parse_args()

    _heavy_imports()

    plots = ALL_PLOTS if "all" in args.plots else args.plots
    makedirs(args.savedir, exist_ok=True)

    # Primary bias: 'both' uses linear as primary; quadratic is overlaid.
    primary = "linear" if args.bias in ("linear", "both") else "quadratic"
    fnames, labels = load_per_survey(primary)
    beta2cosmo = candel.cosmo.Beta2Cosmology()
    attach_sr_interp(beta2cosmo)

    # Plots that need posteriors (beta/S8/fs8) — load lazily.
    needs_posterior = {"s8_posterior", "s8_comparison", "fs8_z"}
    S8_list_q = fs8_list_q = None
    if needs_posterior & set(plots):
        beta_list, S8_list, fs8_list, all_labels = collect_S8_and_fs8(
            fnames, labels, beta2cosmo, joint=True, bias=primary)
        extra_rows = _joint_variant_rows(beta2cosmo, primary)
        print_table_b1_beta_S8(fnames, labels, beta_list, S8_list,
                               fs8_list, all_labels,
                               title=f"Table 1 [{primary}]",
                               extra_joint_rows=extra_rows)
        print_table_tension(labels, S8_list, all_labels,
                            title=f"Table 2 [{primary}]")

        if args.bias == "both":
            fnames_q, labels_q = load_per_survey("quadratic")
            beta_list_q, S8_list_q, fs8_list_q, all_labels_q = (
                collect_S8_and_fs8(fnames_q, labels_q, beta2cosmo,
                                   joint=True, bias="quadratic"))
            extra_rows_q = _joint_variant_rows(beta2cosmo, "quadratic")
            print_table_b1_beta_S8(fnames_q, labels_q, beta_list_q,
                                   S8_list_q, fs8_list_q, all_labels_q,
                                   title="Table 1 [quadratic]",
                                   extra_joint_rows=extra_rows_q)
            print_table_tension(labels_q, S8_list_q, all_labels_q,
                                title="Table 2 [quadratic]")

    # For fs8_z and s8_comparison, replace the 5-survey "Joint" entry
    # with the "Joint (no 6dF)" variant. Returns a new list with the
    # entry at the "Joint" index replaced by ``new_value``; labels list
    # gets the variant name.
    def _swap_joint(values, rows, field):
        no6df = next(r for r in rows if r[0] == "Joint (no 6dF)")
        idx = {"S8": 2, "fs8": 3}[field]
        out = list(values)
        out[all_labels.index("Joint")] = no6df[idx]
        return out

    labs_swap = ["Joint (no 6dF)" if l == "Joint" else l for l in all_labels]

    plot_files = {"s8_posterior":  "S8_posterior.pdf",
                  "s8_comparison": "S8_comparison.pdf",
                  "vext_corner":   "Vext.pdf",
                  "fs8_z":         "fs8_z.pdf"}
    saved = []

    for p in plots:
        print(f"\n--- {p} ---")
        if p == "s8_posterior":
            plot_s8_posterior(S8_list, all_labels, args.savedir)
        elif p == "s8_comparison":
            S8_swap = _swap_joint(S8_list, extra_rows, "S8")
            keep = [i for i, l in enumerate(labs_swap)
                    if l != "Joint (no 6dF)"]
            plot_s8_comparison([S8_swap[i] for i in keep],
                               [labs_swap[i] for i in keep],
                               args.savedir, S8_list_q=None)
        elif p == "vext_corner":
            plot_vext_corner(fnames, labels, args.savedir)
        elif p == "fs8_z":
            fs8_swap_q = (_swap_joint(fs8_list_q, extra_rows_q, "fs8")
                          if fs8_list_q is not None else None)
            plot_fs8_z(_swap_joint(fs8_list, extra_rows, "fs8"),
                       labs_swap, args.savedir, fs8_list_q=fs8_swap_q)
        saved.append((p, join(args.savedir, plot_files[p])))

    print("\n=== Saved plots ===")
    w = max(len(p) for p, _ in saved)
    for name, path in saved:
        print(f"  {name:<{w}}  {path}")


if __name__ == "__main__":
    main()
