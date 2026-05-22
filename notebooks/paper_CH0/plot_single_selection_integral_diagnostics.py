#!/usr/bin/env python
"""Diagnose CH0 single-field selection-volume integrals."""

from argparse import ArgumentParser
import csv
import os
from pathlib import Path
import tomllib

import h5py
import matplotlib
import numpy as np
from scipy.special import log_ndtr, logsumexp

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import candel  # noqa: E402
from candel.pvdata.catalogues import load_SH0ES_separated  # noqa: E402
from candel.pvdata.volume_density import _load_volume_data_for_H0  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: F401,E402


ROOT = Path(__file__).resolve().parents[2]
TASK_SAMPLED = ROOT / "scripts" / "runs" / "tasks_CH0_single.txt"
TASK_FIXED = ROOT / "scripts" / "runs" / "tasks_CH0_single_fixed_bias.txt"
DEFAULT_OUTDIR = (
    Path(__file__).resolve().parent
    / "ch0_single_selection_integral_plots")

BIAS_PARAMS = ("alpha_low", "alpha_high", "log_rho_t", "log_rho_width")
FAMILY_LABELS = {
    "swift": "SWIFT/SPH",
    "cola_cic": "COLA/CIC",
}
FAMILY_COLOURS = {
    "swift": "#87193d",
    "cola_cic": "#1e42b9",
}
MODE_MARKERS = {
    "sampled": "o",
    "fixed": "s",
}
FIGURE_DPI = 500
H0_LABEL = (
    r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sampled-task-file", type=Path, default=TASK_SAMPLED,
        help="Task file with sampled-bias single-field configs.")
    parser.add_argument(
        "--fixed-task-file", type=Path, default=TASK_FIXED,
        help="Task file with fixed-bias single-field configs.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTDIR,
        help="Directory for plots and tables.")
    parser.add_argument(
        "--chunk-size", type=int, default=10,
        help="Number of 3D fields to load at once.")
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Skip missing HDF5 outputs instead of failing.")
    return parser.parse_args()


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def get_nested(mapping, keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def reconstruction_family(config):
    reconstruction = get_nested(config, ("io", "SH0ES", "reconstruction"))
    if reconstruction == "ManticoreLocalSWIFT":
        return "swift"
    if reconstruction == "ManticoreLocalCOLA":
        mas = get_nested(
            config,
            ("io", "reconstruction_main", "ManticoreLocalCOLA", "which_MAS"),
            "")
        if mas == "CIC":
            return "cola_cic"
    return None


def task_specs(task_file, mode):
    specs = []
    with repo_path(task_file).open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task, config_rel = line.split(maxsplit=1)
            config_path = repo_path(config_rel)
            with config_path.open("rb") as config_handle:
                config = tomllib.load(config_handle)
            family = reconstruction_family(config)
            if family is None:
                continue
            source = repo_path(get_nested(config, ("io", "fname_output")))
            specs.append({
                "mode": mode,
                "task": int(task),
                "family": family,
                "field": int(get_nested(config, ("io", "field_indices"))),
                "config": str(config_path),
                "source": str(source),
            })
    return specs


def finite_samples(handle, name, source):
    dataset = f"samples/{name}"
    if dataset not in handle:
        return None
    samples = np.asarray(handle[dataset], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{source}` has no finite `{name}` samples.")
    return samples


def read_scalar(handle, name, source):
    if name not in handle:
        raise KeyError(f"`{source}` does not contain `{name}`.")
    value = float(handle[name][()])
    if not np.isfinite(value):
        raise ValueError(f"`{source}` has non-finite `{name}`: {value}.")
    return value


def h0_summary(samples):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        "H0_mean": float(np.mean(samples)),
        "H0_std": float(np.std(samples, ddof=1)),
        "H0_q16": float(q16),
        "H0_q50": float(q50),
        "H0_q84": float(q84),
    }


def sample_summary(samples, prefix):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        f"{prefix}_q16": float(q16),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q84": float(q84),
        f"{prefix}_mean": float(np.mean(samples)),
    }


def read_result(spec):
    source = Path(spec["source"])
    with h5py.File(source, "r") as handle:
        h0 = finite_samples(handle, "H0", source)
        mb = finite_samples(handle, "M_B", source)
        log_s = np.asarray(
            handle["auxiliary/log_selection_integral"], dtype=float)
        ll = np.asarray(handle["auxiliary/log_likelihood_per_galaxy"],
                        dtype=float)
        ll_sel = np.asarray(
            handle["auxiliary/log_likelihood_per_galaxy_with_selection"],
            dtype=float)
        obs_sel = np.asarray(
            handle["auxiliary/log_observed_selection_per_galaxy"],
            dtype=float)
        n_hosts = ll.shape[1]
        row = {
            **spec,
            "n_hosts": int(n_hosts),
            "n_samples": int(h0.size),
            **h0_summary(h0),
            **sample_summary(mb, "M_B"),
            **sample_summary(log_s, "logS_posterior"),
            "selection_penalty_posterior_q50": float(
                -n_hosts * np.percentile(log_s, 50.0)),
            "observed_selection_sum_q50": float(
                np.median(np.sum(obs_sel, axis=1))),
            "selection_total_contrib_q50": float(
                np.median(np.sum(ll_sel - ll, axis=1))),
            "lnZ_harmonic": read_scalar(
                handle, "gof/lnZ_harmonic", source),
            "err_lnZ_harmonic": read_scalar(
                handle, "gof/err_lnZ_harmonic", source),
            "BIC": read_scalar(handle, "gof/BIC", source),
        }
        for name in BIAS_PARAMS:
            samples = finite_samples(handle, name, source)
            if samples is not None:
                row.update(sample_summary(samples, name))
            else:
                with open(spec["config"], "rb") as config_handle:
                    config = tomllib.load(config_handle)
                value = float(get_nested(
                    config, ("model", "priors", name, "value")))
                row[f"{name}_q16"] = value
                row[f"{name}_q50"] = value
                row[f"{name}_q84"] = value
                row[f"{name}_mean"] = value
        return row


def load_results(sampled_task_file, fixed_task_file, allow_missing):
    specs = [
        *task_specs(sampled_task_file, "sampled"),
        *task_specs(fixed_task_file, "fixed"),
    ]
    rows = []
    missing = []
    for spec in specs:
        if not Path(spec["source"]).is_file():
            missing.append(spec)
            continue
        rows.append(read_result(spec))
    if missing and not allow_missing:
        preview = "\n".join(spec["source"] for spec in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} result files are missing. First missing:\n"
            f"{preview}")
    return rows, missing


def reference_selection_params(rows, config_path):
    config = candel.load_config(str(config_path), replace_los_prior=False)
    root = get_nested(config, ("io", "SH0ES", "root"))
    data = load_SH0ES_separated(
        root,
        cepheid_host_cz_cmb_max=get_nested(
            config, ("io", "SH0ES", "cepheid_host_cz_cmb_max")),
        los_data_path=None,
        rand_los_data_path=None,
        volume_data=None,
        field_indices=None,
        drop_observation=get_nested(config, ("io", "SH0ES", "drop_observation")),
    )
    return {
        "H0_ref": float(np.median([row["H0_q50"] for row in rows])),
        "M_B_ref": float(np.median([row["M_B_q50"] for row in rows])),
        "e_mag_ref": float(data["mean_std_mag_SN_unique_Cepheid_host"]),
        "mag_lim": float(get_nested(config, ("model", "mag_lim_SN"))),
        "mag_width": float(get_nested(config, ("model", "mag_lim_SN_width"))),
        "grid_radius": float(get_nested(
            config, ("model", "selection_integral_grid_radius"))),
    }


def family_loader_config(family, rows):
    row = next(item for item in rows if item["family"] == family)
    config = candel.load_config(row["config"], replace_los_prior=False)
    reconstruction = get_nested(config, ("io", "SH0ES", "reconstruction"))
    field_kwargs = get_nested(
        config, ("io", "reconstruction_main", reconstruction))
    return config, reconstruction, field_kwargs


def log_galaxy_bias_from_logrho(log_rho, bias_params):
    alpha_low, alpha_high, log_rho_t, log_rho_width = bias_params
    log_x = log_rho - log_rho_t
    z = log_x / log_rho_width
    return (
        alpha_low * log_x
        + ((alpha_high - alpha_low)
           * log_rho_width
           * np.logaddexp(0.0, z))
    )


def log_density_from_model_density(density, mode):
    if mode == "log_rho":
        return density
    if mode == "delta":
        return np.log(np.clip(1.0 + density, np.finfo(float).tiny, None))
    raise ValueError(f"Unsupported 3D density mode `{mode}`.")


def controlled_integrals_for_family(
        family, rows, params, chunk_size):
    config, reconstruction, field_kwargs = family_loader_config(family, rows)
    fields = sorted({row["field"] for row in rows if row["family"] == family})
    family_rows = [row for row in rows if row["family"] == family]
    cache_dir = repo_path(get_nested(config, ("io", "field_cache_dir")))

    field_integrals = {}
    h0 = params["H0_ref"]
    h = h0 / 100.0
    mag_sigma = np.sqrt(params["e_mag_ref"]**2 + params["mag_width"]**2)

    for start in range(0, len(fields), chunk_size):
        chunk = fields[start:start + chunk_size]
        volume = _load_volume_data_for_H0(
            reconstruction,
            field_kwargs,
            np.asarray(chunk, dtype=np.int32),
            "double_powerlaw",
            get_nested(config, ("model", "Om"), 0.3),
            subcube_radius=params["grid_radius"],
            voxel_subsample_fraction=1.0,
            load_velocity=False,
            geometry=get_nested(
                config, ("model", "selection_integral_geometry"), "sphere"),
            cache_dir=str(cache_dir),
            cache_enabled=True,
            field_smoothing_scale=get_nested(
                config, ("model", "field_3d_smoothing_scale"), None),
            velocity_field_smoothing_scale=get_nested(
                config, ("model", "velocity_3d_smoothing_scale"), None),
        )

        density_fields = np.asarray(volume["density_3d_fields"], dtype=float)
        mu = np.asarray(volume["mu_at_h1_3d"], dtype=float) - 5 * np.log10(h)
        log_weight = float(volume["log_dV_3d"]) - 3 * np.log(h)
        if "log_volume_weight_3d" in volume:
            log_weight = (
                log_weight
                + np.asarray(volume["log_volume_weight_3d"], dtype=float))
        log_p_sel = log_ndtr(
            (params["mag_lim"] - (mu + params["M_B_ref"])) / mag_sigma)
        log_volume = float(logsumexp(log_weight))

        for i, field in enumerate(chunk):
            log_rho = log_density_from_model_density(
                density_fields[i], volume["density_3d_mode"])
            log_rho_integral = float(logsumexp(log_rho + log_weight))
            field_integrals[field] = {
                "family": family,
                "field": int(field),
                "log_volume_integral": log_volume,
                "volume_integral": float(np.exp(log_volume)),
                "log_density_integral": log_rho_integral,
                "density_integral": float(np.exp(log_rho_integral)),
                "mean_density": float(np.exp(log_rho_integral - log_volume)),
                "mean_delta": float(np.exp(log_rho_integral - log_volume) - 1.0),
            }

            matching = [
                row for row in family_rows if row["field"] == field]
            for row in matching:
                bias = [row[f"{name}_q50"] for name in BIAS_PARAMS]
                log_n = log_galaxy_bias_from_logrho(log_rho, bias)
                log_ng = float(logsumexp(log_n + log_weight))
                log_s = float(logsumexp(log_p_sel + log_n + log_weight))
                row["log_volume_integral"] = log_volume
                row["volume_integral"] = float(np.exp(log_volume))
                row["log_density_integral"] = log_rho_integral
                row["density_integral"] = float(np.exp(log_rho_integral))
                row["mean_density"] = field_integrals[field]["mean_density"]
                row["mean_delta"] = field_integrals[field]["mean_delta"]
                row["log_expected_galaxy_integral"] = log_ng
                row["expected_galaxy_integral"] = float(np.exp(log_ng))
                row["log_selected_galaxy_integral"] = log_s
                row["selected_galaxy_integral"] = float(np.exp(log_s))
                row["log_selected_fraction"] = log_s - log_ng
                row["selected_fraction"] = float(np.exp(log_s - log_ng))
                row["selection_penalty_controlled"] = -row["n_hosts"] * log_s

    return sorted(field_integrals.values(),
                  key=lambda item: (item["family"], item["field"]))


def add_ranks_and_deltas(rows):
    for family in FAMILY_LABELS:
        for mode in ("sampled", "fixed"):
            group = [
                row for row in rows
                if row["family"] == family and row["mode"] == mode]
            if not group:
                continue
            for key in (
                    "log_selected_galaxy_integral",
                    "log_expected_galaxy_integral",
                    "mean_density",
                    "lnZ_harmonic"):
                values = np.asarray([row[key] for row in group], dtype=float)
                med = float(np.median(values))
                order_ascending = np.argsort(values)
                order_descending = np.argsort(-values)
                for rank, index in enumerate(order_ascending, start=1):
                    group[index][f"{key}_rank_low"] = rank
                for rank, index in enumerate(order_descending, start=1):
                    group[index][f"{key}_rank_high"] = rank
                for row in group:
                    row[f"{key}_delta_from_median"] = float(row[key] - med)
            for row in group:
                row["controlled_selection_loglike_gain_vs_median"] = (
                    -row["n_hosts"]
                    * row["log_selected_galaxy_integral_delta_from_median"])
                row["posterior_selection_loglike_gain_vs_median"] = (
                    -row["n_hosts"]
                    * (row["logS_posterior_q50"]
                       - np.median([
                           item["logS_posterior_q50"] for item in group])))


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig, outdir, stem):
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_lnz_h0_vs_selection(rows, outdir):
    with plt.style.context(["science", "no-latex"]):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex="col")
        for col, family in enumerate(("swift", "cola_cic")):
            group = [row for row in rows if row["family"] == family]
            for mode in ("sampled", "fixed"):
                mode_rows = [row for row in group if row["mode"] == mode]
                x = [row["log_selected_galaxy_integral"] for row in mode_rows]
                axes[0, col].scatter(
                    x, [row["lnZ_harmonic"] for row in mode_rows],
                    c=FAMILY_COLOURS[family], marker=MODE_MARKERS[mode],
                    s=28, alpha=0.78, edgecolor="none", label=mode)
                axes[1, col].scatter(
                    x, [row["H0_q50"] for row in mode_rows],
                    c=FAMILY_COLOURS[family], marker=MODE_MARKERS[mode],
                    s=28, alpha=0.78, edgecolor="none")
            best = max(group, key=lambda row: row["lnZ_harmonic"])
            for ax in axes[:, col]:
                ax.axvline(best["log_selected_galaxy_integral"],
                           color="0.25", lw=0.8, ls=":")
            axes[0, col].set_title(FAMILY_LABELS[family])
            axes[1, col].set_xlabel(
                r"$\log \int P_{\rm sel}\,n_g\,{\rm d}V$")
        axes[0, 0].set_ylabel(r"$\ln Z_{\rm harmonic}$")
        axes[1, 0].set_ylabel(H0_LABEL)
        axes[0, 0].legend(frameon=False, loc="best")
        savefig(fig, outdir,
                "ch0_single_selection_integral_h0_lnz_vs_selected")


def plot_integral_components(rows, outdir):
    x_keys = [
        ("mean_density", r"$\langle \rho \rangle_{R<60}$"),
        ("log_expected_galaxy_integral", r"$\log \int n_g\,{\rm d}V$"),
        ("log_selected_fraction",
         r"$\log [\int P_{\rm sel}n_g{\rm d}V / \int n_g{\rm d}V]$"),
    ]
    with plt.style.context(["science", "no-latex"]):
        fig, axes = plt.subplots(
            len(x_keys), 2, figsize=(7.2, 7.0), sharey=True)
        for col, family in enumerate(("swift", "cola_cic")):
            group = [row for row in rows if row["family"] == family]
            for row_index, (key, xlabel) in enumerate(x_keys):
                ax = axes[row_index, col]
                for mode in ("sampled", "fixed"):
                    mode_rows = [
                        row for row in group if row["mode"] == mode]
                    ax.scatter(
                        [row[key] for row in mode_rows],
                        [row["lnZ_harmonic"] for row in mode_rows],
                        c=FAMILY_COLOURS[family],
                        marker=MODE_MARKERS[mode],
                        s=25, alpha=0.75, edgecolor="none", label=mode)
                ax.set_xlabel(xlabel)
                if row_index == 0:
                    ax.set_title(FAMILY_LABELS[family])
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\ln Z_{\rm harmonic}$")
        axes[0, 0].legend(frameon=False, loc="best")
        savefig(fig, outdir,
                "ch0_single_selection_integral_components")


def plot_field_sequence(rows, outdir):
    with plt.style.context(["science", "no-latex"]):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex="col")
        for col, family in enumerate(("swift", "cola_cic")):
            for mode in ("sampled", "fixed"):
                mode_rows = sorted(
                    [row for row in rows
                     if row["family"] == family and row["mode"] == mode],
                    key=lambda row: row["field"])
                axes[0, col].plot(
                    [row["field"] for row in mode_rows],
                    [row["log_selected_galaxy_integral"]
                     for row in mode_rows],
                    marker=MODE_MARKERS[mode], ms=3.5, lw=1.0,
                    color=FAMILY_COLOURS[family], alpha=0.7, label=mode)
                axes[1, col].plot(
                    [row["field"] for row in mode_rows],
                    [row["mean_density"] for row in mode_rows],
                    marker=MODE_MARKERS[mode], ms=3.5, lw=1.0,
                    color=FAMILY_COLOURS[family], alpha=0.7)
            best = max(
                [row for row in rows if row["family"] == family],
                key=lambda row: row["lnZ_harmonic"])
            for ax in axes[:, col]:
                ax.axvline(best["field"], color="0.25", lw=0.8, ls=":")
            axes[0, col].set_title(FAMILY_LABELS[family])
            axes[1, col].set_xlabel("field")
        axes[0, 0].set_ylabel(
            r"$\log \int P_{\rm sel}\,n_g\,{\rm d}V$")
        axes[1, 0].set_ylabel(r"$\langle \rho \rangle_{R<60}$")
        axes[0, 0].legend(frameon=False, loc="best")
        savefig(fig, outdir,
                "ch0_single_selection_integral_by_field")


def plot_posterior_vs_controlled(rows, outdir):
    with plt.style.context(["science", "no-latex"]):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
        for ax, family in zip(axes, ("swift", "cola_cic")):
            group = [row for row in rows if row["family"] == family]
            for mode in ("sampled", "fixed"):
                mode_rows = [row for row in group if row["mode"] == mode]
                ax.scatter(
                    [row["log_selected_galaxy_integral"]
                     for row in mode_rows],
                    [row["logS_posterior_q50"] for row in mode_rows],
                    c=FAMILY_COLOURS[family],
                    marker=MODE_MARKERS[mode],
                    s=28, alpha=0.75, edgecolor="none", label=mode)
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, color="0.3", lw=0.8, ls=":")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(FAMILY_LABELS[family])
            ax.set_xlabel(
                r"controlled $\log \int P_{\rm sel}n_g{\rm d}V$")
        axes[0].set_ylabel("posterior median log selection integral")
        axes[0].legend(frameon=False, loc="best")
        savefig(fig, outdir,
                "ch0_single_selection_integral_posterior_vs_controlled")


def plot_integral_pairwise_combinations(rows, outdir):
    variables = {
        "log_density_integral":
            r"$\log \int \rho_{\rm DM}\,{\rm d}V$",
        "log_expected_galaxy_integral":
            r"$\log \int n_g\,{\rm d}V$",
        "log_selected_galaxy_integral":
            r"$\log \int P_{\rm sel}\,n_g\,{\rm d}V$",
    }
    titles = [
        "Expected galaxies vs density",
        "Selected galaxies vs density",
        "Selected vs expected galaxies",
    ]
    pairs = [
        ("log_density_integral", "log_expected_galaxy_integral"),
        ("log_density_integral", "log_selected_galaxy_integral"),
        ("log_expected_galaxy_integral", "log_selected_galaxy_integral"),
    ]
    with plt.style.context("default"):
        plt.rcParams.update({
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        })
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
        fig.subplots_adjust(
            left=0.075, right=0.995, bottom=0.09, top=0.86,
            wspace=0.30, hspace=0.36)
        for row_index, mode in enumerate(("sampled", "fixed")):
            mode_rows = [row for row in rows if row["mode"] == mode]
            for col_index, (x_key, y_key) in enumerate(pairs):
                ax = axes[row_index, col_index]
                for family in ("swift", "cola_cic"):
                    family_rows = [
                        row for row in mode_rows if row["family"] == family]
                    ax.scatter(
                        [row[x_key] for row in family_rows],
                        [row[y_key] for row in family_rows],
                        s=24, alpha=0.72,
                        color=FAMILY_COLOURS[family],
                        marker=MODE_MARKERS[mode],
                        edgecolor="none",
                        label=FAMILY_LABELS[family])

                field21 = next(
                    row for row in mode_rows
                    if row["family"] == "swift" and row["field"] == 21)
                ax.scatter(
                    [field21[x_key]], [field21[y_key]],
                    s=180, marker="*", color="#ffcf33",
                    edgecolor="black", linewidth=0.9, zorder=5,
                    label="SWIFT field 21")
                ax.annotate(
                    "field 21",
                    xy=(field21[x_key], field21[y_key]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=8,
                    arrowprops={
                        "arrowstyle": "-",
                        "lw": 0.6,
                        "color": "0.2",
                    })

                if row_index == 0:
                    ax.set_title(titles[col_index], fontsize=10)
                if col_index == 0:
                    ax.set_ylabel(variables[y_key], fontsize=9)
                    ax.text(
                        0.03, 0.95, f"{mode} bias",
                        transform=ax.transAxes,
                        ha="left", va="top", fontsize=9,
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "facecolor": "white",
                            "edgecolor": "0.75",
                            "alpha": 0.85,
                        })
                else:
                    ax.set_ylabel(variables[y_key], fontsize=9)
                ax.set_xlabel(variables[x_key], fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(alpha=0.22, lw=0.45)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        keep = {}
        for handle, label in zip(handles, labels):
            keep.setdefault(label, handle)
        fig.legend(
            keep.values(), keep.keys(), frameon=False, ncol=3,
            loc="upper center", bbox_to_anchor=(0.5, 0.985), fontsize=9)
        savefig(fig, outdir,
                "ch0_single_selection_integral_pairwise_combinations")


def corrcoef(rows, x_key, y_key):
    if len(rows) < 3:
        return np.nan
    x = np.asarray([row[x_key] for row in rows], dtype=float)
    y = np.asarray([row[y_key] for row in rows], dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def write_summary(path, rows, field_integrals, params, missing):
    lines = [
        "# CH0 Single-Field Selection-Integral Diagnostics",
        "",
        f"Loaded result rows: {len(rows)}.",
        f"Missing outputs skipped: {len(missing)}.",
        "",
        "## Controlled Selection Parameters",
        "",
        f"- H0_ref = {params['H0_ref']:.3f} km/s/Mpc.",
        f"- M_B_ref = {params['M_B_ref']:.3f}.",
        f"- SN magnitude error = {params['e_mag_ref']:.4f} mag.",
        f"- mag_lim = {params['mag_lim']:.3f}, "
        f"mag_width = {params['mag_width']:.3f} mag.",
        f"- volume geometry = sphere, R = {params['grid_radius']:.1f} Mpc/h.",
        "",
        "The controlled integral is "
        "`S = int P_sel n_g dV_phys`, evaluated at the reference H0 and M_B.",
        "The likelihood contains `-N_host log S`, so lower `log S` is a "
        "selection-normalisation advantage.",
        "",
        "## Best-Evidence Fields",
        "",
        "| family | mode | best field | H0 | lnZ | rank_low logS | "
        "dlogS vs median | selection gain | dlogS posterior | "
        "density rank_high | mean density |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for family in ("swift", "cola_cic"):
        for mode in ("sampled", "fixed"):
            group = [
                row for row in rows
                if row["family"] == family and row["mode"] == mode]
            best = max(group, key=lambda row: row["lnZ_harmonic"])
            post_delta = (
                best["logS_posterior_q50"]
                - np.median([row["logS_posterior_q50"] for row in group]))
            lines.append(
                f"| {FAMILY_LABELS[family]} | {mode} | {best['field']} | "
                f"{best['H0_q50']:.3f} | {best['lnZ_harmonic']:.2f} | "
                f"{best['log_selected_galaxy_integral_rank_low']:.0f} | "
                f"{best['log_selected_galaxy_integral_delta_from_median']:+.3f} | "
                f"{best['controlled_selection_loglike_gain_vs_median']:+.2f} | "
                f"{post_delta:+.3f} | "
                f"{best['mean_density_rank_high']:.0f} | "
                f"{best['mean_density']:.3f} |")
    lines.extend([
        "",
        "## Correlations With Evidence",
        "",
        "| family | mode | corr(lnZ, log S) | corr(lnZ, log int n_g dV) | "
        "corr(lnZ, mean density) |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for family in ("swift", "cola_cic"):
        for mode in ("sampled", "fixed"):
            group = [
                row for row in rows
                if row["family"] == family and row["mode"] == mode]
            lines.append(
                f"| {FAMILY_LABELS[family]} | {mode} | "
                f"{corrcoef(group, 'log_selected_galaxy_integral', 'lnZ_harmonic'):+.3f} | "
                f"{corrcoef(group, 'log_expected_galaxy_integral', 'lnZ_harmonic'):+.3f} | "
                f"{corrcoef(group, 'mean_density', 'lnZ_harmonic'):+.3f} |")
    lines.extend([
        "",
        "## Field 21",
        "",
    ])
    for mode in ("sampled", "fixed"):
        row = next(
            item for item in rows
            if item["family"] == "swift"
            and item["mode"] == mode
            and item["field"] == 21)
        lines.append(
            f"- SWIFT field 21, {mode}: "
            f"logS = {row['log_selected_galaxy_integral']:.3f} "
            f"(rank {row['log_selected_galaxy_integral_rank_low']:.0f}/30 "
            f"low), selection gain vs median "
            f"{row['controlled_selection_loglike_gain_vs_median']:+.2f}, "
            f"mean density = {row['mean_density']:.3f} "
            f"(rank {row['mean_density_rank_high']:.0f}/30 high), "
            f"lnZ delta vs median "
            f"{row['lnZ_harmonic_delta_from_median']:+.2f}.")

    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("`--chunk-size` must be positive.")
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows, missing = load_results(
        args.sampled_task_file, args.fixed_task_file, args.allow_missing)
    params = reference_selection_params(rows, rows[0]["config"])
    field_integrals = []
    for family in ("swift", "cola_cic"):
        field_integrals.extend(controlled_integrals_for_family(
            family, rows, params, args.chunk_size))
    add_ranks_and_deltas(rows)

    write_csv(outdir / "ch0_single_selection_integral_rows.csv", rows)
    write_csv(outdir / "ch0_single_selection_integral_fields.csv",
              field_integrals)
    write_summary(
        outdir / "ch0_single_selection_integral_summary.txt",
        rows, field_integrals, params, missing)

    plot_lnz_h0_vs_selection(rows, outdir)
    plot_integral_components(rows, outdir)
    plot_field_sequence(rows, outdir)
    plot_posterior_vs_controlled(rows, outdir)
    plot_integral_pairwise_combinations(rows, outdir)
    print(f"wrote plots and tables to {outdir}")


if __name__ == "__main__":
    main()
