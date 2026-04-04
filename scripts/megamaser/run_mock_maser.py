"""Run inference on a mock maser disk galaxy."""
import sys
sys.path.insert(0, "/Users/rstiskalek/Projects/CANDEL")

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import argparse  # noqa: E402
parser = argparse.ArgumentParser()
parser.add_argument("--host-devices", type=int, default=1)
parser.add_argument("--n-spots", type=int, default=100)
parser.add_argument("--num-warmup", type=int, default=500)
parser.add_argument("--num-samples", type=int, default=2500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--marginalise-r", action="store_true")
args = parser.parse_args()

if args.host_devices > 1:
    import numpyro
    numpyro.set_host_device_count(args.host_devices)

import tempfile  # noqa: E402
import numpy as np  # noqa: E402
import tomli_w  # noqa: E402

from candel.mock.maser_disk_mock import gen_maser_disk_mock  # noqa: E402
from candel.model.model_H0_maser import MaserDiskModel  # noqa: E402
from candel.inference.inference import run_H0_inference  # noqa: E402
from candel.util import fprint, fsection  # noqa: E402

# --- Generate mock data ---
fsection("Mock Data")
true_params = {
    "D_c": 115.0,
    "M_BH": 4.15e7,
    "i0": 72.4,
    "Omega0": 149.7,
    "di_dr": 0.0,
    "dOmega_dr": -3.2,
    "x0": -0.044,
    "y0": -0.100,
    "sigma_x_floor": 0.003,
    "sigma_y_floor": 0.003,
    "sigma_v_sys": 1.5,
    "sigma_v_hv": 1.5,
    "sigma_a_floor": 0.04,
    "H0": 73.0,
    "sigma_pec": 250.0,
}

data, tp = gen_maser_disk_mock(
    seed=args.seed, true_params=true_params, n_spots=args.n_spots,
    verbose=True)

def print_true_params(tp):
    """Print all true parameters."""
    fprint("\nTrue parameters:")
    for k in sorted(tp):
        v = tp[k]
        if isinstance(v, np.ndarray):
            continue
        fprint(f"  {k:20s} = {v}")


# Convert position params to uas for comparison with posterior
for k in ("x0", "y0", "sigma_x_floor", "sigma_y_floor"):
    if k in tp:
        tp[k] = tp[k] * 1e3

print_true_params(tp)

# --- Build config ---
# Fix all globals to true values, only sample R_phys per spot.
def _delta(v):
    return {"dist": "delta", "value": float(v)}


config = {
    "inference": {
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": min(args.host_devices, 4),
        "chain_method": "parallel" if args.host_devices > 1
        else "sequential",
        "seed": args.seed,
        "dense_mass_blocks": [
            ["D_c", "log_MBH", "dv_sys"],
            ["i0", "di_dr"],
            ["Omega0", "dOmega_dr"],
            ["x0", "y0"],
        ],
        "init_maxiter": 1000,
        "max_tree_depth": 10,
    },
    "model": {
        "which_run": "maser_disk",
        "Om": 0.315,
        "use_selection": False,
        "fit_di_dr": True,
        "marginalise_r": args.marginalise_r,
        "priors": {
            "H0": _delta(tp["H0"]),
            "sigma_pec": _delta(tp["sigma_pec"]),
            "D": {"dist": "uniform", "low": 50.0, "high": 200.0},
            "log_MBH": {"dist": "uniform", "low": 6.0, "high": 9.0},
            "R_phys": {"dist": "uniform", "low": 0.01, "high": 1.5},
            "x0": {"dist": "uniform", "low": -500.0, "high": 500.0},
            "y0": {"dist": "uniform", "low": -500.0, "high": 500.0},
            "i0": {"dist": "uniform", "low": 60.0, "high": 110.0},
            "Omega0": {"dist": "uniform", "low": 0.0, "high": 360.0},
            "dOmega_dr": {"dist": "uniform",
                          "low": -30.0, "high": 30.0},
            "di_dr": {"dist": "uniform",
                      "low": -30.0, "high": 30.0},
            "dv_sys": {"dist": "normal", "loc": 0.0, "scale": 500.0},
            "sigma_x_floor": {"dist": "uniform",
                              "low": 0.0, "high": 100.0},
            "sigma_y_floor": {"dist": "uniform",
                              "low": 0.0, "high": 100.0},
            "sigma_v_sys": {"dist": "truncated_normal",
                            "mean": 2.0, "scale": 1.0,
                            "low": 0.0, "high": 20.0},
            "sigma_v_hv": {"dist": "truncated_normal",
                           "mean": 2.0, "scale": 1.0,
                           "low": 0.0, "high": 20.0},
            "sigma_a_floor": {"dist": "uniform",
                              "low": 1e-4, "high": 5.0},
        },
    },
    "io": {
        "fname_output": "results/Maser/mock_test.hdf5",
    },
}

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()

# --- Build model and run ---
model = MaserDiskModel(tmp.name, data)

fsection("Running NUTS")
samples = run_H0_inference(
    model, save_samples=False,
    progress_bar=True)

# --- Compare posteriors to truth ---
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm as sp_norm  # noqa: E402
from corner import corner  # noqa: E402

r_ang_true = tp["r_ang_true"]
R_true = tp["R_phys_true"]
n = data["n_spots"]
has_r = "r_ang" in samples

if has_r:
    fsection("r_ang posterior vs truth")
    for i in range(n):
        s = np.array(samples["r_ang"][:, i])
        med = np.median(s)
        lo, hi = np.percentile(s, [16, 84])
        within = "OK" if lo <= r_ang_true[i] <= hi else "MISS"
        fprint(f"  spot {i:3d}: true={r_ang_true[i]:.4f}  "
               f"post={med:.4f} [{lo:.4f}, {hi:.4f}]  {within}")
else:
    fprint("r_ang marginalised — no per-spot posteriors.")

# Global parameters to track — just edit this list.
global_keys = ["D_c", "log_MBH", "dv_sys", "x0", "y0",
               "i0", "Omega0", "dOmega_dr", "di_dr",
               "sigma_x_floor", "sigma_y_floor",
               "sigma_v_sys", "sigma_v_hv", "sigma_a_floor"]

fsection("Global parameter posteriors vs truth")
for k in global_keys:
    if k not in samples:
        continue
    v_true = tp[k]
    s = np.array(samples[k])
    med = np.median(s)
    lo, hi = np.percentile(s, [16, 84])
    within = "OK" if lo <= v_true <= hi else "MISS"
    fprint(f"  {k}: true={v_true:.4f}  "
           f"post={med:.4f} [{lo:.4f}, {hi:.4f}]  {within}")

# --- Corner plot ---
global_arrays = []
global_labels = []
global_truth_vals = []
for k in global_keys:
    if k in samples:
        global_arrays.append(np.array(samples[k])[:, None])
        global_labels.append(k)
        global_truth_vals.append(tp[k])

if has_r:
    r_samples = np.array(samples["r_ang"])
    corner_data = np.hstack(global_arrays + [r_samples])
    labels = global_labels + [f"$r_{{{i}}}$" for i in range(n)]
    truths = global_truth_vals + list(r_ang_true)
else:
    corner_data = np.hstack(global_arrays)
    labels = global_labels
    truths = global_truth_vals

fig = corner(corner_data, labels=labels, truths=truths,
             truth_color="C1", show_titles=True,
             plot_datapoints=False,
             title_kwargs={"fontsize": 8},
             label_kwargs={"fontsize": 8})
fig.savefig("test_mock_corner.png", dpi=150, bbox_inches="tight")
fprint("saved corner plot to test_mock_corner.png")

# --- Bias plot ---
z_global = []
z_global_labels = []
for k, lab in zip(global_keys, global_labels):
    if k in samples:
        s = np.array(samples[k])
        z_global.append((s.mean() - tp[k]) / s.std())
        z_global_labels.append(lab)

if has_r:
    r_mean = r_samples.mean(axis=0)
    r_std = r_samples.std(axis=0)
    z_r = (r_mean - r_ang_true) / r_std
    z_all = np.concatenate([z_global, z_r])
    param_labels = z_global_labels + [
        f"$r_{{{i}}}$" for i in range(n)]
else:
    z_all = np.array(z_global)
    param_labels = z_global_labels

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.hist(z_all, bins=max(len(z_all) // 3, 5), density=True,
        alpha=0.7, label="params")
xg = np.linspace(-4, 4, 200)
ax.plot(xg, sp_norm.pdf(xg), "k-", lw=1.5,
        label=r"$\mathcal{N}(0,1)$")
ax.set_xlabel(
    r"$(\langle \theta \rangle - \theta_{\rm true}) / \sigma_\theta$")
ax.set_ylabel("density")
ax.legend()

ax = axes[1]
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.axhspan(-1, 1, color="grey", alpha=0.15)
ax.axhspan(-2, 2, color="grey", alpha=0.08)
ax.scatter(np.arange(len(z_all)), z_all, s=20, zorder=3)
ax.set_xticks(np.arange(len(z_all)))
ax.set_xticklabels(param_labels, rotation=90, fontsize=7)
ax.set_ylabel(
    r"$(\langle \theta \rangle - \theta_{\rm true}) / \sigma_\theta$")

fig.tight_layout()
fig.savefig("test_mock_bias.png", dpi=150, bbox_inches="tight")
fprint("saved bias plot to test_mock_bias.png")

# --- Correlation analysis ---
fsection("Correlation analysis")

# Collect sampled globals
sampled_globals = [k for k in global_keys if k in samples]
glob_arrays = [np.array(samples[k])[:, None] for k in sampled_globals]
n_glob = len(sampled_globals)

# Global-global correlation matrix
if n_glob > 1:
    glob_mat = np.hstack(glob_arrays)
    corr_gg = np.corrcoef(glob_mat.T)
    fprint("Global-global correlations:")
    header = "".join(f"{k:>14s}" for k in sampled_globals)
    fprint(f"  {'':14s}{header}")
    for i, ki in enumerate(sampled_globals):
        row = "".join(f"{corr_gg[i, j]:14.3f}"
                      for j in range(n_glob))
        fprint(f"  {ki:14s}{row}")

# Global-spot correlations by spot type
if has_r:
    all_samples = np.hstack(glob_arrays + [r_samples])
    corr = np.corrcoef(all_samples.T)
    spot_types = data["spot_type"]
    type_labels = {"r": "red HV", "b": "blue HV", "s": "systemic"}
    colors = {"r": "C3", "b": "C0", "s": "C2"}

    fprint("\nGlobal-spot mean |corr| by type:")
    header = "".join(f"{k:>14s}" for k in sampled_globals)
    fprint(f"  {'type':12s}{header}")
    for t, label in type_labels.items():
        mask = spot_types == t
        if not mask.any():
            continue
        row = ""
        for j in range(n_glob):
            c = corr[j, n_glob:]
            row += f"{np.mean(np.abs(c[mask])):14.3f}"
        fprint(f"  {label:12s}{row}")

    # Top 5 most correlated (global, spot) pairs
    fprint("\nTop 10 |corr| (global, spot) pairs:")
    pairs = []
    for j in range(n_glob):
        for i in range(n):
            pairs.append((abs(corr[j, n_glob + i]),
                          corr[j, n_glob + i],
                          sampled_globals[j], i, spot_types[i]))
    pairs.sort(reverse=True)
    for abs_c, c, gname, idx, stype in pairs[:10]:
        fprint(f"  {gname:14s} x spot {idx:3d} ({stype}): "
               f"corr = {c:+.3f}")

    # Plot
    fig, axes = plt.subplots(1, n_glob, figsize=(3.5 * n_glob, 3.5),
                             sharey=True)
    if n_glob == 1:
        axes = [axes]
    for j, (ax, gname) in enumerate(zip(axes, sampled_globals)):
        c = corr[j, n_glob:]
        for i in range(n):
            ax.bar(i, c[i], color=colors[spot_types[i]], width=0.8)
        ax.set_xlabel("spot index")
        ax.set_title(f"corr with {gname}")
        ax.axhline(0, color="k", lw=0.5)

    from matplotlib.patches import Patch
    handles = [Patch(color=colors[t], label=type_labels[t])
               for t in ("r", "b", "s")]
    axes[-1].legend(handles=handles, fontsize=7)
    axes[0].set_ylabel("Pearson r")

    fig.tight_layout()
    fig.savefig("test_mock_correlations.png", dpi=150,
                bbox_inches="tight")
    fprint("saved correlation plot to test_mock_correlations.png")

os.unlink(tmp.name)
