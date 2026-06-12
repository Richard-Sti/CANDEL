#!/usr/bin/env python
"""Compare matched-field H0 posteriors for fiducial and mag_min_TRGB=24 cuts."""

from argparse import ArgumentParser
from dataclasses import dataclass
import csv
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tomllib  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

try:
    import scienceplots  # noqa: F401,E402
except ModuleNotFoundError:
    scienceplots = None

from trgbh0_plot_style import (  # noqa: E402
    FIGURE_DPI,
    OUTPUT_DIR,
    ROOT,
    save_pdf_png,
    set_paper_rc,
    trgbh0_cmap,
)


FIDUCIAL_TASK_FILE = ROOT / "scripts" / "runs" / "tasks_TRGBH0_single.txt"
MAGMIN24_TASK_FILE = (
    ROOT / "scripts" / "runs" / "tasks_TRGBH0_single_magmin.txt"
)
DEFAULT_OUTDIR = OUTPUT_DIR
FIDUCIAL_MAG_MIN = 22.1
MAGMIN24 = 24.0
H0_LABEL = r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"


@dataclass(frozen=True)
class RunSpec:
    task: int
    field: int
    config: Path
    output: Path
    mag_min: float


@dataclass(frozen=True)
class H0Summary:
    spec: RunSpec
    n_h0: int
    mean: float
    std: float
    q16: float
    q50: float
    q84: float


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fiducial-task-file",
        type=Path,
        default=FIDUCIAL_TASK_FILE,
        help="Task file containing the fiducial single-field runs.",
    )
    parser.add_argument(
        "--magmin24-task-file",
        type=Path,
        default=MAGMIN24_TASK_FILE,
        help="Task file containing the mag_min_TRGB=24 single-field runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for the comparison CSV and plots.",
    )
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


def task_config_paths(task_file):
    paths = []
    with repo_path(task_file).open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            task_index, config = line.split(maxsplit=1)
            paths.append((int(task_index), repo_path(config)))
    if not paths:
        raise ValueError(f"No task configs found in `{task_file}`.")
    return paths


def active_reconstruction(config):
    return (
        get_nested(config, ("io", "PV_main", "EDD_TRGB", "reconstruction"))
        or get_nested(config, ("io", "CCHP", "reconstruction"))
    )


def is_delta_beta_one(config):
    beta_prior = get_nested(config, ("model", "priors", "beta"), default={})
    if not isinstance(beta_prior, dict):
        return False
    return (
        beta_prior.get("dist") == "delta"
        and np.isclose(float(beta_prior.get("value", np.nan)), 1.0)
    )


def is_target_config(config, mag_min):
    model = config.get("model", {})
    if active_reconstruction(config) != "ManticoreLocalCOLA":
        return False
    if get_nested(
            config,
            ("io", "reconstruction_main", "ManticoreLocalCOLA", "which_MAS"),
    ) != "PCS":
        return False
    if get_nested(model, ("cz_likelihood",), default="gaussian") != "student_t":
        return False
    if get_nested(model, ("which_selection",)) != "TRGB_magnitude":
        return False
    if not np.isclose(
            float(get_nested(model, ("field_3d_smoothing_scale",), 0.0)),
            4.0):
        return False
    if not np.isclose(
            float(get_nested(model, ("velocity_3d_smoothing_scale",), 0.0)),
            0.0):
        return False
    if not is_delta_beta_one(config):
        return False
    if not np.isclose(float(get_nested(model, ("mag_min_TRGB",), 22.1)),
                     mag_min):
        return False
    return True


def select_specs(task_file, mag_min):
    specs = {}
    for task_index, config_path in task_config_paths(task_file):
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
        if not is_target_config(config, mag_min):
            continue

        field = get_nested(config, ("io", "field_indices"))
        if not isinstance(field, int):
            continue

        spec = RunSpec(
            task=task_index,
            field=field,
            config=config_path,
            output=repo_path(get_nested(config, ("io", "fname_output"))),
            mag_min=mag_min,
        )
        if field in specs:
            raise ValueError(
                f"Multiple task configs for field {field} and mag_min={mag_min}."
            )
        specs[field] = spec
    return specs


def finite_h0_samples(output):
    with h5py.File(output, "r") as handle:
        dataset = "samples/H0"
        if dataset not in handle:
            raise ValueError(f"`{output}` does not contain `{dataset}`.")
        samples = np.asarray(handle[dataset], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{output}` has no finite H0 samples.")
    return samples


def summarise_h0(spec):
    samples = finite_h0_samples(spec.output)
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return H0Summary(
        spec=spec,
        n_h0=int(samples.size),
        mean=float(np.mean(samples)),
        std=float(np.std(samples, ddof=1)),
        q16=float(q16),
        q50=float(q50),
        q84=float(q84),
    )


def read_summaries(specs):
    summaries = {}
    missing = []
    unusable = []
    for field, spec in sorted(specs.items()):
        if not spec.output.is_file():
            missing.append(spec)
            continue
        try:
            summaries[field] = summarise_h0(spec)
        except (OSError, KeyError, ValueError) as exc:
            unusable.append((spec, f"{type(exc).__name__}: {exc}"))
    return summaries, missing, unusable


def interval_half_width(summary):
    return 0.5 * (summary.q84 - summary.q16)


def comparison_records(fiducial, magmin24):
    records = []
    for field in sorted(set(fiducial) & set(magmin24)):
        fid = fiducial[field]
        cut = magmin24[field]
        records.append({
            "field": field,
            "fiducial_task": fid.spec.task,
            "magmin24_task": cut.spec.task,
            "fiducial_mag_min_TRGB": fid.spec.mag_min,
            "magmin24_mag_min_TRGB": cut.spec.mag_min,
            "fiducial_n_H0": fid.n_h0,
            "magmin24_n_H0": cut.n_h0,
            "fiducial_H0_mean": fid.mean,
            "fiducial_H0_std": fid.std,
            "fiducial_H0_q16": fid.q16,
            "fiducial_H0_q50": fid.q50,
            "fiducial_H0_q84": fid.q84,
            "magmin24_H0_mean": cut.mean,
            "magmin24_H0_std": cut.std,
            "magmin24_H0_q16": cut.q16,
            "magmin24_H0_q50": cut.q50,
            "magmin24_H0_q84": cut.q84,
            "delta_H0_mean": cut.mean - fid.mean,
            "delta_H0_q50": cut.q50 - fid.q50,
            "delta_H0_interval_quadrature": float(np.hypot(
                interval_half_width(cut),
                interval_half_width(fid),
            )),
            "fiducial_output": str(fid.spec.output),
            "magmin24_output": str(cut.spec.output),
            "fiducial_config": str(fid.spec.config),
            "magmin24_config": str(cut.spec.config),
        })
    return records


def write_csv(records, out_csv):
    fieldnames = [
        "field",
        "fiducial_task",
        "magmin24_task",
        "fiducial_mag_min_TRGB",
        "magmin24_mag_min_TRGB",
        "fiducial_n_H0",
        "magmin24_n_H0",
        "fiducial_H0_mean",
        "fiducial_H0_std",
        "fiducial_H0_q16",
        "fiducial_H0_q50",
        "fiducial_H0_q84",
        "magmin24_H0_mean",
        "magmin24_H0_std",
        "magmin24_H0_q16",
        "magmin24_H0_q50",
        "magmin24_H0_q84",
        "delta_H0_mean",
        "delta_H0_q50",
        "delta_H0_interval_quadrature",
        "fiducial_output",
        "magmin24_output",
        "fiducial_config",
        "magmin24_config",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_comparison(records, out_pdf):
    if not records:
        return None

    fields = np.asarray([row["field"] for row in records], dtype=int)
    x = np.asarray([row["fiducial_H0_q50"] for row in records])
    y = np.asarray([row["magmin24_H0_q50"] for row in records])
    xerr = np.vstack([
        x - np.asarray([row["fiducial_H0_q16"] for row in records]),
        np.asarray([row["fiducial_H0_q84"] for row in records]) - x,
    ])
    yerr = np.vstack([
        y - np.asarray([row["magmin24_H0_q16"] for row in records]),
        np.asarray([row["magmin24_H0_q84"] for row in records]) - y,
    ])
    delta = np.asarray([row["delta_H0_q50"] for row in records])
    delta_err = np.asarray([
        row["delta_H0_interval_quadrature"] for row in records
    ])

    style = ["science", "no-latex"] if scienceplots is not None else []
    with plt.style.context(style):
        set_paper_rc()
        fig, axes = plt.subplots(
            1, 2, figsize=(7.1, 3.0), constrained_layout=True)

        norm = Normalize(
            vmin=float(np.min(fields)),
            vmax=float(np.max(fields)) if len(fields) > 1 else float(fields[0] + 1),
        )
        cmap = trgbh0_cmap("trgbh0_magmin24")
        colours = cmap(norm(fields))

        axes[0].errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="0.7",
            elinewidth=0.6, capsize=0, zorder=1)
        axes[0].scatter(
            x, y, c=fields, cmap=cmap, norm=norm, s=17, linewidths=0,
            zorder=2)
        lo = float(np.nanmin(np.concatenate([x - xerr[0], y - yerr[0]])))
        hi = float(np.nanmax(np.concatenate([x + xerr[1], y + yerr[1]])))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                     color="0.25", lw=0.7, ls="--", zorder=0)
        axes[0].set_xlim(lo - pad, hi + pad)
        axes[0].set_ylim(lo - pad, hi + pad)
        axes[0].set_xlabel(r"fiducial magnitude cut " + H0_LABEL)
        axes[0].set_ylabel(r"$m_{\rm min}=24$ " + H0_LABEL)

        axes[1].axhline(0.0, color="0.25", lw=0.7, ls="--", zorder=0)
        axes[1].errorbar(
            fields, delta, yerr=delta_err, fmt="none", ecolor="0.7",
            elinewidth=0.6, capsize=0, zorder=1)
        axes[1].scatter(fields, delta, c=colours, s=17, linewidths=0,
                        zorder=2)
        axes[1].set_xlabel("field")
        axes[1].set_ylabel(
            r"$H_0(m_{\rm min}=24)-H_0({\rm fiducial})$")

        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        cbar = fig.colorbar(scalar_map, ax=axes, pad=0.015, fraction=0.04)
        cbar.set_label("field")

        return save_pdf_png(fig, out_pdf)


def print_run_status(label, specs, summaries, missing, unusable):
    print(
        f"{label}: selected {len(specs)} configs; "
        f"{len(summaries)} usable outputs; "
        f"{len(missing)} missing outputs; {len(unusable)} unusable outputs."
    )
    if missing:
        fields = ", ".join(str(spec.field) for spec in missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        print(f"{label}: missing fields {fields}{suffix}")
    if unusable:
        fields = ", ".join(
            f"{spec.field} ({error})" for spec, error in unusable[:6]
        )
        suffix = " ..." if len(unusable) > 6 else ""
        print(f"{label}: unusable fields {fields}{suffix}")


def main():
    args = parse_args()
    fid_specs = select_specs(args.fiducial_task_file, FIDUCIAL_MAG_MIN)
    cut_specs = select_specs(args.magmin24_task_file, MAGMIN24)

    fid_summaries, fid_missing, fid_unusable = read_summaries(fid_specs)
    cut_summaries, cut_missing, cut_unusable = read_summaries(cut_specs)
    records = comparison_records(fid_summaries, cut_summaries)

    out_csv = args.output_dir / "trgbh0_magmin24_matched_fields_h0.csv"
    out_pdf = args.output_dir / "trgbh0_magmin24_matched_fields_h0.pdf"
    write_csv(records, out_csv)
    plot_paths = plot_comparison(records, out_pdf)

    print_run_status(
        "fiducial", fid_specs, fid_summaries, fid_missing, fid_unusable)
    print_run_status(
        "mag_min_TRGB=24", cut_specs, cut_summaries, cut_missing,
        cut_unusable)
    print(f"Matched fields with both outputs: {len(records)}")
    if records:
        deltas = np.asarray([row["delta_H0_q50"] for row in records])
        print(
            "Delta H0 median summary "
            f"(mag_min_TRGB=24 - fiducial): "
            f"mean={np.mean(deltas):.3f}, median={np.median(deltas):.3f}, "
            f"min={np.min(deltas):.3f}, max={np.max(deltas):.3f}"
        )
        print(f"Wrote CSV: {out_csv}")
        print(f"Wrote plots: {plot_paths[0]} and {plot_paths[1]}")
    else:
        print(f"Wrote empty CSV: {out_csv}")
        print("No plot written because no fields have both outputs yet.")


if __name__ == "__main__":
    main()
