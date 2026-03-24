#!/usr/bin/env python
"""SBC bias analysis for CANDEL mock inference runs.

Discovers mock run HDF5 files, computes standardised biases, PIT values
(rank-based or CDF), and produces diagnostic plots with a formal PASS/FAIL
summary.
"""

import argparse
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tarp import get_tarp_coverage


# ---------------------------------------------------------------------------
# Discovery & I/O
# ---------------------------------------------------------------------------

def discover_runs(results_dir):
    """Find all mock run indices from ``mock_NNNN_*_full.hdf5`` files.

    Returns sorted list of (index, filepath) tuples.
    """
    pattern = re.compile(r"mock_(\d{4})_.*_full\.hdf5")
    runs = []
    for fname in os.listdir(results_dir):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            runs.append((idx, os.path.join(results_dir, fname)))
    return sorted(runs, key=lambda x: x[0])


def load_samples(path):
    """Load posterior samples from a result HDF5 (``samples/`` group)."""
    out = {}
    with h5py.File(path, "r") as f:
        grp = f["samples"]
        for key in grp.keys():
            out[key] = grp[key][...]
    return out


def load_truths(results_dir, idx):
    """Load truth parameters from ``<results_dir>/mocks/mock_NNNN.hdf5``."""
    path = os.path.join(results_dir, "mocks", f"mock_{idx:04d}.hdf5")
    truths = {}
    with h5py.File(path, "r") as f:
        if "mock" in f:
            for key in f["mock"].attrs:
                val = f["mock"].attrs[key]
                if isinstance(val, (np.floating, float, int, np.integer)):
                    truths[key] = float(val)
        for key in f.attrs:
            val = f.attrs[key]
            if isinstance(val, (np.floating, float, int, np.integer)):
                truths[key] = float(val)
    return truths


# ---------------------------------------------------------------------------
# Parameter flattening (handles scalar + vector params)
# ---------------------------------------------------------------------------

def flatten_params(samples, truths):
    """Flatten parameter dicts into aligned arrays.

    Vector parameters (shape ``(n_samples, K)``) expand into scalar entries
    ``name[0], name[1], ...``.

    Returns (names, flat_samples dict, flat_truths dict).
    """
    names = []
    flat_samples = {}
    flat_truths = {}

    for key in sorted(samples.keys()):
        if key not in truths:
            continue
        s = samples[key]
        t = truths[key]

        if s.ndim == 1:
            names.append(key)
            flat_samples[key] = s
            flat_truths[key] = float(t)
        elif s.ndim == 2:
            t_arr = np.atleast_1d(t)
            for k in range(s.shape[1]):
                name = f"{key}[{k}]"
                names.append(name)
                flat_samples[name] = s[:, k]
                flat_truths[name] = float(t_arr[k])

    return names, flat_samples, flat_truths


# ---------------------------------------------------------------------------
# Circular / linear mode detection
# ---------------------------------------------------------------------------

def infer_mode(name):
    """Detect circular parameters from name.

    Returns ``("circular", period)`` or ``("linear", None)``.
    """
    low = name.lower()
    if any(k in low for k in ("ell", "phi", "ra", "lon", "longitude")):
        return ("circular", 360.0)
    return ("linear", None)


# ---------------------------------------------------------------------------
# SBC statistics
# ---------------------------------------------------------------------------

def compute_standardised_bias(samples, truth):
    """(posterior_mean − truth) / posterior_std."""
    return (np.mean(samples) - truth) / np.std(samples)


def wrap_delta(samples, truth, period):
    """Wrap differences to (−period/2, period/2]."""
    return ((samples - truth + period / 2) % period) - period / 2


def _randomised_rank(x, s):
    """Randomised rank (1..N) of *x* among *s* with uniform tie-breaking."""
    s = np.asarray(s)
    N = s.size
    lt = np.sum(s < x)
    eq = np.sum(s == x)
    u = np.random.uniform()
    rank = lt + 1 + int(np.floor(u * eq))
    return rank, N


def linear_rank_pit(samples, truth):
    r, N = _randomised_rank(truth, samples)
    return (r - 0.5) / N


def linear_cdf_pit(samples, truth):
    s = np.asarray(samples)
    return (np.sum(s < truth) + 0.5 * np.sum(s == truth)) / s.size


def circular_rank_pit(samples, truth, period=360.0, randomise=True):
    """Rank-SBC for circular variable with optional random rotation."""
    s = np.asarray(samples, float) % period
    t = truth % period
    if randomise:
        alpha = np.random.uniform(0.0, period)
        s = (s + alpha) % period
        t = (t + alpha) % period
        return linear_rank_pit(s, t)
    else:
        d = wrap_delta(s, t, period)
        return (np.sum(d < 0) + 0.5 * np.sum(d == 0)) / d.size


def circular_cdf_pit(samples, truth, period=360.0):
    d = wrap_delta(np.asarray(samples), truth, period)
    return (np.sum(d < 0) + 0.5 * np.sum(d == 0)) / d.size


def compute_pit(samples, truth, name, use_rank=True, randomise_circular=True):
    """Compute a single PIT value, dispatching on parameter mode."""
    kind, period = infer_mode(name)
    if kind == "circular":
        if use_rank:
            return circular_rank_pit(samples, truth, period,
                                     randomise=randomise_circular)
        return circular_cdf_pit(samples, truth, period)
    else:
        if use_rank:
            return linear_rank_pit(samples, truth)
        return linear_cdf_pit(samples, truth)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_posteriors(param_samples, param_truths, param_names, n_plot,
                   output_dir):
    """Per-mock subplot grid (n_plot rows × n_params cols) with KDE."""
    n_params = len(param_names)
    fig, axes = plt.subplots(n_plot, n_params,
                             figsize=(4 * n_params, 3.5 * n_plot),
                             squeeze=False)

    for i in range(n_plot):
        for j, name in enumerate(param_names):
            ax = axes[i, j]
            samples = param_samples[name][i]
            truth = param_truths[name][i]

            if samples is None or not np.isfinite(truth):
                ax.set_axis_off()
                continue

            kind, period = infer_mode(name)
            if kind == "circular":
                vals = wrap_delta(samples, truth, period)
                kde = stats.gaussian_kde(vals)
                x = np.linspace(vals.min(), vals.max(), 300)
                ax.fill_between(x, kde(x), alpha=0.4)
                ax.axvline(0.0, color="C3", ls="--")
                ax.set_xlabel(f"{name} − truth [deg]")
            else:
                kde = stats.gaussian_kde(samples)
                x = np.linspace(samples.min(), samples.max(), 300)
                ax.fill_between(x, kde(x), alpha=0.4)
                ax.axvline(truth, color="C3", ls="--")
                ax.set_xlabel(name)

            if i == 0:
                ax.set_title(name)

        axes[i, 0].set_ylabel(f"Mock {i}", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "posteriors.png"), dpi=150)
    plt.close(fig)


def plot_bias_histograms(biases, param_names, output_dir):
    """Single row figure: one subplot per parameter."""
    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    axes = np.atleast_1d(axes)
    x_std = np.linspace(-4, 4, 200)
    pdf_std = stats.norm.pdf(x_std)

    for ax, name in zip(axes, param_names):
        b = biases[name]
        ax.hist(b, bins=20, density=True, alpha=0.6)
        ax.plot(x_std, pdf_std, "k--", lw=1.2, label=r"N(0,1)")
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("(mean − truth) / std")
        ax.set_title(f"{name}\nmean={np.mean(b):.3f}, std={np.std(b):.3f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bias.png"), dpi=150)
    plt.close(fig)


def plot_pit_histograms(pits, param_names, n_bins, output_dir):
    """Single row figure with KS annotation."""
    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, param_names):
        vals = pits[name]
        valid = vals[np.isfinite(vals)]
        N = len(valid)
        ks_stat, ks_p = stats.kstest(valid, "uniform")
        ax.hist(valid, bins=n_bins, density=True, alpha=0.7, edgecolor="black")
        ax.axhline(1.0, ls="--", color="red", lw=1.5)
        # Binomial uncertainty bands on uniform density
        p_bin = 1.0 / n_bins
        bin_width = 1.0 / n_bins
        sigma_density = np.sqrt(N * p_bin * (1 - p_bin)) / (N * bin_width)
        ax.axhspan(1.0 - sigma_density, 1.0 + sigma_density,
                   alpha=0.2, color="grey")
        ax.axhspan(1.0 - 2 * sigma_density, 1.0 + 2 * sigma_density,
                   alpha=0.1, color="grey")
        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title(f"{name}\nKS p={ks_p:.3f}, n={N}")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pit.png"), dpi=150)
    plt.close(fig)


def plot_qq(pits, param_names, output_dir):
    """QQ plots against Uniform(0,1) with 95 % KS band."""
    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, param_names):
        vals = pits[name]
        valid = np.sort(vals[np.isfinite(vals)])
        N = len(valid)
        expected = (np.arange(1, N + 1) - 0.5) / N
        ks_crit = 1.36 / np.sqrt(N)

        ax.scatter(expected, valid, s=20, alpha=0.6)
        ax.plot([0, 1], [0, 1], "r--", lw=1.2)
        ax.fill_between([0, 1], [-ks_crit, 1 - ks_crit],
                        [ks_crit, 1 + ks_crit],
                        alpha=0.15, color="grey", label="95% KS band")
        ax.set_xlabel("Expected (Uniform)")
        ax.set_ylabel("Observed (PIT)")
        ks_stat, ks_p = stats.kstest(valid, "uniform")
        ax.set_title(f"{name}\nKS={ks_stat:.3f}, p={ks_p:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "qq.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SBC bias analysis for CANDEL mock runs")
    parser.add_argument("--results", required=True,
                        help="Results directory containing mock_NNNN_*_full.hdf5")
    parser.add_argument("--output", default=None,
                        help="Output directory for plots (default: <results>/bias_analysis)")
    parser.add_argument("--n-plot", type=int, default=5,
                        help="Number of mocks for detailed posterior grid")
    parser.add_argument("--rank", action="store_true", default=True,
                        dest="rank", help="Use rank-based PIT (default)")
    parser.add_argument("--no-rank", action="store_false", dest="rank",
                        help="Use CDF-based PIT instead of rank")
    parser.add_argument("--bins", type=int, default=15,
                        help="Number of PIT histogram bins")
    args = parser.parse_args()

    results_dir = args.results
    output_dir = args.output or os.path.join(results_dir, "bias_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Discover runs
    # ------------------------------------------------------------------
    runs = discover_runs(results_dir)
    if not runs:
        print(f"No mock_NNNN_*_full.hdf5 files found in {results_dir}")
        sys.exit(1)
    print(f"Found {len(runs)} runs: indices {runs[0][0]}..{runs[-1][0]}")

    # ------------------------------------------------------------------
    # Load all runs, auto-detect parameters
    # ------------------------------------------------------------------
    all_sample_dicts = []
    all_truth_dicts = []
    valid_indices = []

    for idx, fpath in runs:
        try:
            samples = load_samples(fpath)
            truths = load_truths(results_dir, idx)
            all_sample_dicts.append(samples)
            all_truth_dicts.append(truths)
            valid_indices.append(idx)
        except Exception as e:
            print(f"  Skipping mock {idx}: {e}")

    if not all_sample_dicts:
        print("No valid runs loaded.")
        sys.exit(1)

    # Parameter list: intersection of sample keys and truth keys across runs
    common_params = None
    for sd, td in zip(all_sample_dicts, all_truth_dicts):
        _, flat_s, flat_t = flatten_params(sd, td)
        keys = set(flat_s.keys()) & set(flat_t.keys())
        common_params = keys if common_params is None else common_params & keys
    param_names = sorted(common_params)

    n_runs = len(all_sample_dicts)
    print(f"Valid runs: {n_runs}")
    print(f"Parameters ({len(param_names)}): {param_names}")

    # ------------------------------------------------------------------
    # Collect per-parameter arrays
    # ------------------------------------------------------------------
    # For posterior grid: list of sample arrays per param
    param_samples = {p: [] for p in param_names}
    param_truths = {p: [] for p in param_names}
    biases = {p: [] for p in param_names}
    pits = {p: [] for p in param_names}

    for sd, td in zip(all_sample_dicts, all_truth_dicts):
        _, flat_s, flat_t = flatten_params(sd, td)
        for p in param_names:
            s = flat_s[p]
            t = flat_t[p]
            param_samples[p].append(s)
            param_truths[p].append(t)

            kind, _ = infer_mode(p)
            if kind == "circular":
                # Bias on wrapped differences
                d = wrap_delta(s, t, 360.0)
                biases[p].append(np.mean(d) / np.std(d))
            else:
                biases[p].append(compute_standardised_bias(s, t))

            pits[p].append(compute_pit(s, t, p, use_rank=args.rank))

    for p in param_names:
        biases[p] = np.array(biases[p])
        pits[p] = np.array(pits[p])

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    n_plot = min(args.n_plot, n_runs)
    print(f"\nGenerating plots ({n_plot} mocks in posterior grid)...")

    plot_posteriors(param_samples, param_truths, param_names, n_plot,
                   output_dir)
    print(f"  posteriors.png")

    plot_bias_histograms(biases, param_names, output_dir)
    print(f"  bias.png")

    plot_pit_histograms(pits, param_names, args.bins, output_dir)
    print(f"  pit.png")

    plot_qq(pits, param_names, output_dir)
    print(f"  qq.png")

    # ------------------------------------------------------------------
    # TARP coverage test
    # ------------------------------------------------------------------
    # Build arrays: samples (n_samples, n_sims, n_dims), theta (n_sims, n_dims)
    n_dims = len(param_names)
    n_samp = min(len(s) for s in param_samples[param_names[0]])
    tarp_samples = np.zeros((n_samp, n_runs, n_dims))
    tarp_theta = np.zeros((n_runs, n_dims))
    for j, p in enumerate(param_names):
        for i in range(n_runs):
            tarp_samples[:, i, j] = param_samples[p][i][:n_samp]
            tarp_theta[i, j] = param_truths[p][i]

    tarp_out = get_tarp_coverage(
        tarp_samples, tarp_theta, norm=True, bootstrap=False,
        references="random", seed=42)
    ecp, alpha = tarp_out[0], tarp_out[1]

    # Expected sigma bands from finite sample size
    sigma_binom = np.sqrt(alpha * (1 - alpha) / n_runs)

    # Plot TARP
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.fill_between(alpha, alpha - 2 * sigma_binom,
                     alpha + 2 * sigma_binom,
                     alpha=0.1, color="grey", label=r"$2\sigma$")
    ax.fill_between(alpha, alpha - sigma_binom,
                     alpha + sigma_binom,
                     alpha=0.2, color="grey", label=r"$1\sigma$")
    ax.plot(alpha, ecp, "C0-", lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Credibility level")
    ax.set_ylabel("Expected coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tarp.png"), dpi=150)
    plt.close(fig)
    print(f"  tarp.png")

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    threshold = 2.0 / np.sqrt(n_runs)
    method = "rank" if args.rank else "CDF"
    print(f"\n{'='*65}")
    print(f"  SBC Summary  (K={n_runs}, PIT method={method})")
    print(f"  PASS criteria: KS p > 0.05 AND |mean bias| < {threshold:.3f}")
    print(f"{'='*65}")
    print(f"  {'Parameter':>25s}  {'mean_bias':>10s}  {'KS_p':>8s}  {'result':>6s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*8}  {'-'*6}")

    n_pass = 0
    for p in param_names:
        mb = np.mean(biases[p])
        ks_stat, ks_p = stats.kstest(pits[p], "uniform")
        passed = ks_p > 0.05 and abs(mb) < threshold
        tag = "PASS" if passed else "FAIL"
        n_pass += int(passed)
        print(f"  {p:>25s}  {mb:>+10.3f}  {ks_p:>8.3f}  {tag:>6s}")

    print(f"{'='*65}")
    print(f"  {n_pass}/{len(param_names)} parameters passed")
    print(f"\n  Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
