#!/usr/bin/env python
"""Plot one-Manticore-field TRGBH0 nuisance-parameter diagnostics."""
from argparse import ArgumentParser
import csv
from dataclasses import dataclass
import re
import shutil
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import gaussian_kde, ks_2samp, pearsonr, spearmanr  # noqa: E402

import plot_trgbh0_manticore_evidence_drivers as evidence_drivers  # noqa: E402
import plot_trgbh0_manticore_tempered_evidence_h0 as tempered_evidence_h0  # noqa: E402
from trgbh0_plot_style import trgbh0_cmap  # noqa: E402


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
RESULTS = ROOT / "results" / "TRGBH0_paper" / "manticore_fields_const_sigv"
DEFAULT_OUTDIR = RESULTS / "plots"
MANTICORE_SCHEDULE = (
    Path("/mnt/extraspace/rstiskalek/MANTICORE")
    / "2MPP_MULTIBIN_N256_DES_V2"
    / "schedule_final.yaml"
)

FIELD_SET_SPECS = {
    "cola": {
        "pattern": (
            "EDD_TRGB_sel-TRGB_magnitude_"
            "COLA_manticore_2MPP_MULTIBIN_N256_DES_V2_field*_"
            "manticore_field_const_sigv.hdf5"
        ),
        "expected": 50,
        "label": "COLA",
        "suffix": "",
    },
    "non-cola": {
        "pattern": (
            "EDD_TRGB_sel-TRGB_magnitude_"
            "manticore_2MPP_MULTIBIN_N256_DES_V2_field*_"
            "manticore_field_const_sigv.hdf5"
        ),
        "expected": 30,
        "label": "non-COLA",
        "suffix": "_non_cola",
    },
}
FIELD_RE = re.compile(r"_field(\d+)_")
FIGURE_DPI = 500
PARAMS = ("H0", "sigma_int", "sigma_v", "mag_lim_TRGB",
          "mag_lim_TRGB_width")

H0_LABEL = (
    r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,"
    r"\mathrm{Mpc}^{-1}]$"
)
SIGMA_V_LABEL = r"$\sigma_v ~ [\mathrm{km}\,\mathrm{s}^{-1}]$"
SIGMA_INT_LABEL = r"$\sigma_{\rm int} ~ [\mathrm{mag}]$"
MAG_LIM_TRGB_LABEL = r"$m_{\rm lim}^{\rm TRGB} ~ [\mathrm{mag}]$"

PLOT_CHOICES = (
    "sigma-v-posterior",
    "h0-sigma-v",
    "h0-sigma-int",
    "sigma-int-sigma-v",
    "h0-mag-lim-trgb",
    "sigma-int-mag-lim-trgb",
    "h0-lnz-harmonic",
    "h0-alpha-c",
    "h0-alpha-low",
    "h0-alpha-high",
    "h0-c-star",
    "evidence-comparison",
    "evidence-drivers",
    "tempered-evidence-h0",
    "all",
)
SELF_CONTAINED_PLOTS = (
    "evidence-comparison",
    "evidence-drivers",
    "tempered-evidence-h0",
)
FIELD_NUMBER_COLOURED_PLOTS = (
    "sigma-v-posterior",
)
DEFAULT_PLOTS = (
    "h0-lnz-harmonic",
    "h0-sigma-v",
    "h0-mag-lim-trgb",
    "sigma-int-mag-lim-trgb",
)
SCATTER_PLOT_SPECS = {
    "h0-sigma-v": {
        "x_name": "sigma_v",
        "y_name": "H0",
        "x_label": SIGMA_V_LABEL,
        "y_label": H0_LABEL,
        "cmap_name": "trgbh0_manticore_fields_h0_sigma_v",
        "out_basename": "trgbh0_manticore_field_h0_sigma_v",
        "summary_basename": "manticore_field_h0_sigma_v_summary",
    },
    "h0-sigma-int": {
        "x_name": "sigma_int",
        "y_name": "H0",
        "x_label": SIGMA_INT_LABEL,
        "y_label": H0_LABEL,
        "cmap_name": "trgbh0_manticore_fields_h0_sigma_int",
        "out_basename": "trgbh0_manticore_field_h0_sigma_int",
        "summary_basename": "manticore_field_h0_sigma_int_summary",
    },
    "sigma-int-sigma-v": {
        "x_name": "sigma_int",
        "y_name": "sigma_v",
        "x_label": SIGMA_INT_LABEL,
        "y_label": SIGMA_V_LABEL,
        "cmap_name": "trgbh0_manticore_fields_sigma_int_sigma_v",
        "out_basename": "trgbh0_manticore_field_sigma_int_sigma_v",
        "summary_basename": "manticore_field_sigma_int_sigma_v_summary",
    },
    "h0-mag-lim-trgb": {
        "x_name": "mag_lim_TRGB",
        "y_name": "H0",
        "x_label": MAG_LIM_TRGB_LABEL,
        "y_label": H0_LABEL,
        "cmap_name": "trgbh0_h0_mag_lim_trgb_width",
        "out_basename": "trgbh0_manticore_field_h0_mag_lim_trgb",
        "summary_basename": "manticore_field_h0_mag_lim_trgb_summary",
    },
    "sigma-int-mag-lim-trgb": {
        "x_name": "sigma_int",
        "y_name": "mag_lim_TRGB",
        "x_label": SIGMA_INT_LABEL,
        "y_label": MAG_LIM_TRGB_LABEL,
        "cmap_name": "trgbh0_sigma_int_mag_lim_trgb_h0",
        "out_basename": "trgbh0_manticore_field_sigma_int_mag_lim_trgb",
        "summary_basename": "manticore_field_sigma_int_mag_lim_trgb_summary",
    },
    "h0-alpha-c": {
        "x_name": "alpha_c",
        "y_name": "H0",
        "x_label": r"$\alpha_c$",
        "y_label": H0_LABEL,
        "cmap_name": "trgbh0_manticore_fields_h0_alpha_c",
        "out_basename": "trgbh0_manticore_field_h0_alpha_c",
        "summary_basename": "manticore_field_h0_alpha_c_summary",
    },
    "h0-alpha-low": {
        "x_name": "alpha_low",
        "y_name": "H0",
        "x_label": r"$\alpha_{\rm low}$",
        "y_label": H0_LABEL,
        "cmap_name": "trgbh0_manticore_fields_h0_alpha_low",
        "out_basename": "trgbh0_manticore_field_h0_alpha_low",
        "summary_basename": "manticore_field_h0_alpha_low_summary",
    },
    "h0-alpha-high": {
        "x_name": "alpha_high",
        "y_name": "H0",
        "x_label": r"$\alpha_{\rm high}$",
        "y_label": H0_LABEL,
        "cmap_name": "trgbh0_manticore_fields_h0_alpha_high",
        "out_basename": "trgbh0_manticore_field_h0_alpha_high",
        "summary_basename": "manticore_field_h0_alpha_high_summary",
    },
    "h0-c-star": {
        "x_name": "c_star",
        "y_name": "H0",
        "x_label": r"$c_\star$",
        "y_label": H0_LABEL,
        "cmap_name": "trgbh0_manticore_fields_h0_c_star",
        "out_basename": "trgbh0_manticore_field_h0_c_star",
        "summary_basename": "manticore_field_h0_c_star_summary",
    },
}
PLOT_PARAMS = {
    "sigma-v-posterior": ("sigma_v",),
    "h0-lnz-harmonic": ("H0",),
    **{
        name: tuple(dict.fromkeys(
            param for param in (spec["x_name"], spec["y_name"])))
        for name, spec in SCATTER_PLOT_SPECS.items()
    },
    "evidence-comparison": (),
}


@dataclass(frozen=True)
class PlotConfig:
    field_set: str
    output_dir: Path
    summary_dir: Path
    copy_to_results: bool
    colour_by_chain: bool
    colour_by_lnz_harmonic: bool

    @property
    def field_spec(self):
        return FIELD_SET_SPECS[self.field_set]

    @property
    def field_label(self):
        return self.field_spec["label"]

    @property
    def expected_field_count(self):
        return self.field_spec["expected"]

    @property
    def field_suffix(self):
        return self.field_spec["suffix"]

    @property
    def has_colour_mode(self):
        return self.colour_by_chain or self.colour_by_lnz_harmonic

    @property
    def colour_mode_suffix(self):
        if self.colour_by_chain:
            return "_by_chain"
        if self.colour_by_lnz_harmonic:
            return "_by_lnz_harmonic"
        return ""

    @property
    def output_suffix(self):
        return self.field_suffix + self.colour_mode_suffix

    @property
    def extra_summary_fields(self):
        extra_fields = []
        if self.colour_by_chain:
            extra_fields.extend(["chain", "mcmc_step"])
        if self.colour_by_lnz_harmonic:
            extra_fields.extend(["lnZ_harmonic", "err_lnZ_harmonic"])
        return tuple(extra_fields)

    @property
    def diagnostic_name_suffix(self):
        suffix = ""
        if self.colour_by_chain:
            suffix = "-by-chain"
        elif self.colour_by_lnz_harmonic:
            suffix = "-by-lnZ-harmonic"
        if self.field_suffix:
            suffix += f"-{self.field_label}"
        return suffix


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "plots",
        nargs="*",
        help=(
            "Diagnostics to plot: sigma-v-posterior, h0-sigma-v, "
            "h0-sigma-int, sigma-int-sigma-v, h0-mag-lim-trgb, "
            "sigma-int-mag-lim-trgb, h0-lnz-harmonic, h0-alpha-c, "
            "h0-alpha-low, h0-alpha-high, h0-c-star, "
            "evidence-comparison, evidence-drivers, "
            "tempered-evidence-h0, or all. "
            "Default: h0-lnz-harmonic, h0-sigma-v, h0-mag-lim-trgb, "
            "and sigma-int-mag-lim-trgb. Scatter plots are coloured by "
            "harmonic lnZ."
        ),
    )
    parser.add_argument(
        "--field-set",
        choices=sorted(FIELD_SET_SPECS),
        default="cola",
        help="Manticore field set for the per-field diagnostics.",
    )
    parser.add_argument(
        "--colour-by-chain",
        action="store_true",
        help=(
            "Colour non-scatter diagnostics by the Manticore source chain "
            "from the schedule YAML and write separate *_by_chain outputs."
        ),
    )
    parser.add_argument(
        "--colour-by-lnZ-harmonic",
        "--colour-by-lnz-harmonic",
        dest="colour_by_lnz_harmonic",
        action="store_true",
        help=(
            "Colour non-scatter diagnostics by gof/lnZ_harmonic from each "
            "result HDF5 and write separate *_by_lnz_harmonic outputs."
        ),
    )
    parser.add_argument(
        "--manticore-schedule",
        type=Path,
        default=MANTICORE_SCHEDULE,
        help="Schedule YAML mapping field index to Manticore chain.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for PDF and PNG plot outputs.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help=(
            "Directory for CSV summaries. Default: same as --output-dir."
        ),
    )
    parser.add_argument(
        "--no-copy-to-results",
        action="store_true",
        help="Do not copy figure outputs to the results root.",
    )
    parser.add_argument(
        "--top-galaxies",
        type=int,
        default=20,
        help=(
            "Number of top galaxies to list and plot for evidence-drivers."
        ),
    )
    parser.add_argument(
        "--heatmap-galaxies",
        type=int,
        default=40,
        help=(
            "Number of galaxies to include in the evidence-drivers heatmap."
        ),
    )
    parser.add_argument(
        "--likelihood",
        choices=sorted(tempered_evidence_h0.PATTERNS),
        default="gaussian",
        help="Redshift likelihood result set for tempered-evidence-h0.",
    )
    parser.add_argument(
        "--num-beta",
        type=int,
        default=201,
        help="Number of beta values for tempered-evidence-h0.",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=20000,
        help="Number of bootstrap draws for tempered-evidence-h0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=53721,
        help="Random seed for tempered-evidence-h0.",
    )
    parser.add_argument(
        "--field-subset-fraction",
        type=float,
        default=0.8,
        help="Field-subset fraction for tempered-evidence-h0.",
    )
    parser.add_argument(
        "--target-neff",
        default="5,10,15",
        help="Comma-separated target effective field counts.",
    )
    args = parser.parse_args()
    if args.colour_by_chain and args.colour_by_lnz_harmonic:
        parser.error(
            "`--colour-by-chain` and `--colour-by-lnz-harmonic` are "
            "mutually exclusive.")
    return args


def config_from_args(args):
    return PlotConfig(
        field_set=args.field_set,
        output_dir=args.output_dir,
        summary_dir=args.summary_dir or args.output_dir,
        copy_to_results=not args.no_copy_to_results,
        colour_by_chain=args.colour_by_chain,
        colour_by_lnz_harmonic=args.colour_by_lnz_harmonic,
    )


def field_index(path):
    match = FIELD_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse field index from `{path}`.")
    return int(match.group(1))


def load_chain_metadata(path):
    with path.open() as handle:
        schedule = yaml.safe_load(handle)
    metadata = {}
    for raw_index, item in schedule.items():
        if len(item) != 1:
            raise ValueError(
                f"Schedule entry {raw_index} should contain one chain.")
        chain, info = next(iter(item.items()))
        metadata[int(raw_index)] = {
            "chain": chain,
            "mcmc_step": int(info["mcmc_step"]),
        }
    return metadata


def add_chain_metadata(rows, schedule_path):
    metadata = load_chain_metadata(schedule_path)
    for row in rows:
        field = row["field"]
        if field not in metadata:
            raise ValueError(
                f"No Manticore chain metadata for field {field} in "
                f"`{schedule_path}`.")
        row.update(metadata[field])


def read_hdf5_scalar(handle, name, path):
    try:
        value = float(handle[name][()])
    except KeyError as err:
        raise ValueError(f"`{path}` does not contain `{name}`.") from err
    if not np.isfinite(value):
        raise ValueError(f"`{path}` has non-finite `{name}`.")
    return value


def finite_samples(handle, name, path):
    samples = np.asarray(handle[f"samples/{name}"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"`{path}` has no finite {name} samples.")
    return samples


def summary(samples):
    q16, q50, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "q16": float(q16),
        "q50": float(q50),
        "q84": float(q84),
    }


def load_rows(config, params=PARAMS, include_lnz_harmonic=False):
    pattern = config.field_spec["pattern"]
    paths = sorted(RESULTS.glob(pattern), key=field_index)
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matching `{pattern}`.")

    rows = []
    missing_evidence = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            samples = {name: finite_samples(handle, name, path) for name in params}
            evidence = {}
            if include_lnz_harmonic:
                try:
                    evidence = {
                        "lnZ_harmonic": read_hdf5_scalar(
                            handle, "gof/lnZ_harmonic", path),
                        "err_lnZ_harmonic": read_hdf5_scalar(
                            handle, "gof/err_lnZ_harmonic", path),
                    }
                except ValueError as err:
                    missing_evidence.append((path, err))
                    continue
        rows.append({
            "field": field_index(path),
            "source": str(path),
            **evidence,
            **{f"n_{name}": samples[name].size for name in params},
            **{name: summary(samples[name]) for name in params},
            "samples": samples,
        })
    if include_lnz_harmonic and missing_evidence:
        for path, err in missing_evidence:
            print(f"[WARN] Skipping `{path}` for evidence-coloured plots: {err}")
    if not rows:
        raise ValueError("No rows available after applying plot filters.")
    return rows


def load_evidence_rows(field_set):
    pattern = FIELD_SET_SPECS[field_set]["pattern"]
    paths = sorted(RESULTS.glob(pattern), key=field_index)
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matching `{pattern}`.")

    rows = []
    missing_evidence = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            try:
                rows.append({
                    "field_set": FIELD_SET_SPECS[field_set]["label"],
                    "field": field_index(path),
                    "lnZ_harmonic": read_hdf5_scalar(
                        handle, "gof/lnZ_harmonic", path),
                    "err_lnZ_harmonic": read_hdf5_scalar(
                        handle, "gof/err_lnZ_harmonic", path),
                    "source": str(path),
                })
            except ValueError as err:
                missing_evidence.append((path, err))
    for path, err in missing_evidence:
        print(f"[WARN] Skipping `{path}` for evidence comparison: {err}")
    if not rows:
        raise ValueError(
            f"No evidence rows available for `{FIELD_SET_SPECS[field_set]['label']}`.")
    return rows


def write_summary(rows, params, path, extra_fields=()):
    fieldnames = ["field", *[f"n_{name}" for name in params]]
    for name in params:
        fieldnames.extend([
            f"{name}_mean",
            f"{name}_std",
            f"{name}_q16",
            f"{name}_q50",
            f"{name}_q84",
        ])
    fieldnames.extend(extra_fields)
    fieldnames.append("source")

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {"field": row["field"], "source": row["source"]}
            for name in params:
                out[f"n_{name}"] = row[f"n_{name}"]
                for key in ("mean", "std", "q16", "q50", "q84"):
                    out[f"{name}_{key}"] = row[name][key]
            for name in extra_fields:
                out[name] = row[name]
            writer.writerow(out)


def write_evidence_comparison_summary(rows_by_field_set, path):
    fieldnames = [
        "field_set", "field", "lnZ_harmonic", "err_lnZ_harmonic", "source"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rows in rows_by_field_set.values():
            for row in rows:
                writer.writerow(row)


def p_label(value):
    if value < 1e-3:
        return r"<10^{-3}"
    return f"={value:.2f}"


def kde_on_grid(samples, x_grid, bw=1.2):
    kde = gaussian_kde(samples)
    kde.set_bandwidth(kde.factor * bw)
    return kde(x_grid)


def set_paper_rc():
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.4,
    })


def copy_outputs(config, *paths):
    if not config.copy_to_results:
        return
    for path in paths:
        destination = RESULTS / path.name
        if path.resolve() != destination.resolve():
            shutil.copyfile(path, destination)


def copy_figure_outputs(config, outputs):
    figure_paths = [
        path for path in outputs.values()
        if path.suffix in {".pdf", ".png"}
    ]
    copy_outputs(config, *figure_paths)


def save_pdf_png(fig, out_pdf):
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=FIGURE_DPI, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


def fields_and_missing(rows, expected_field_count):
    fields = [row["field"] for row in rows]
    missing = [i for i in range(expected_field_count) if i not in set(fields)]
    return fields, missing


def print_outputs(name, out_pdf, out_png, summary_csv, rows, config, corr=None):
    fields, missing = fields_and_missing(rows, config.expected_field_count)
    print(f"[{name}] Wrote {out_pdf}")
    print(f"[{name}] Wrote {out_png}")
    print(f"[{name}] Wrote {summary_csv}")
    print(f"[{name}] Fields plotted: {len(fields)}; missing: {missing}")
    if corr is not None:
        pearson_r, pearson_p, spearman_r, spearman_p = corr
        print(f"[{name}] Pearson r={pearson_r:.3f}, p={pearson_p:.3g}")
        print(f"[{name}] Spearman rho={spearman_r:.3f}, p={spearman_p:.3g}")


def plot_evidence_comparison(config):
    rows_by_field_set = {
        field_set: load_evidence_rows(field_set)
        for field_set in ("cola", "non-cola")
    }
    summary_csv = config.summary_dir / "manticore_field_lnz_harmonic_comparison.csv"
    write_evidence_comparison_summary(rows_by_field_set, summary_csv)

    labels = {
        field_set: FIELD_SET_SPECS[field_set]["label"]
        for field_set in rows_by_field_set
    }
    values = {
        field_set: np.asarray(
            [row["lnZ_harmonic"] for row in rows], dtype=float)
        for field_set, rows in rows_by_field_set.items()
    }
    ks_stat, ks_p = ks_2samp(values["cola"], values["non-cola"])

    x_min = min(np.min(x) for x in values.values())
    x_max = max(np.max(x) for x in values.values())
    pad = 0.08 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)
    cmap = trgbh0_cmap("trgbh0_manticore_evidence_comparison")
    colours = {
        "cola": cmap(0.18),
        "non-cola": cmap(0.78),
    }

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(4.6, 3.15))
        for field_set in ("cola", "non-cola"):
            x = values[field_set]
            label = labels[field_set]
            ax.hist(
                x,
                bins=min(12, max(5, int(np.sqrt(x.size)) + 2)),
                density=True,
                color=colours[field_set],
                alpha=0.25,
            )
            ax.plot(
                x_grid,
                kde_on_grid(x, x_grid),
                color=colours[field_set],
                lw=1.25,
                label=(
                    rf"{label} ($N={x.size}$; "
                    rf"$\mathrm{{median}}={np.median(x):.1f}$)"
                ),
            )
            ax.axvline(
                np.median(x),
                color=colours[field_set],
                lw=0.9,
                ls=":",
            )

        delta_median = np.median(values["cola"]) - np.median(values["non-cola"])
        ax.text(
            0.03,
            0.97,
            (
                rf"$\Delta\mathrm{{median}}="
                rf"{delta_median:.1f}$" "\n"
                rf"KS $D={ks_stat:.2f}$, $p{p_label(ks_p)}$"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )
        ax.set_xlabel(r"Harmonic $\ln Z$")
        ax.set_ylabel("Density")
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", frameon=False, fontsize=6.5)

        out_pdf = (
            config.output_dir
            / "trgbh0_manticore_field_lnz_harmonic_comparison.pdf"
        )
        out_png = save_pdf_png(fig, out_pdf)

    copy_outputs(config, out_pdf, out_png)
    print(f"[evidence-comparison] Wrote {out_pdf}")
    print(f"[evidence-comparison] Wrote {out_png}")
    print(f"[evidence-comparison] Wrote {summary_csv}")
    print(
        "[evidence-comparison] "
        f"COLA fields: {values['cola'].size}; "
        f"non-COLA fields: {values['non-cola'].size}")
    print(
        "[evidence-comparison] "
        f"Delta median lnZ (COLA - non-COLA) = {delta_median:.3f}")
    print(
        "[evidence-comparison] "
        f"KS D={ks_stat:.3f}, p={ks_p:.3g}")


def plot_evidence_drivers(config, args):
    outputs = evidence_drivers.run_analysis(
        results_dir=RESULTS,
        output_dir=config.output_dir,
        summary_dir=config.summary_dir,
        field_set=config.field_set,
        top_galaxies=args.top_galaxies,
        heatmap_galaxies=args.heatmap_galaxies,
    )
    copy_figure_outputs(config, outputs)


def plot_tempered_evidence_h0(config, args):
    if config.field_set != "cola":
        raise ValueError(
            "`tempered-evidence-h0` is currently defined for the COLA "
            "Manticore field set. Use `--field-set cola`."
        )
    outputs = tempered_evidence_h0.run_analysis(
        likelihood=args.likelihood,
        results_dir=RESULTS,
        output_dir=config.output_dir,
        summary_dir=config.summary_dir,
        num_beta=args.num_beta,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        field_subset_fraction=args.field_subset_fraction,
        target_neff=args.target_neff,
    )
    copy_figure_outputs(config, outputs)


def chain_order(rows):
    chains = []
    for row in rows:
        chain = row["chain"]
        if chain not in chains:
            chains.append(chain)
    return chains


def chain_colour_map(rows, cmap_name):
    chains = chain_order(rows)
    cmap = trgbh0_cmap(cmap_name)
    if len(chains) == 1:
        samples = [0.5]
    else:
        samples = np.linspace(0.0, 1.0, len(chains))
    return {chain: cmap(value) for chain, value in zip(chains, samples)}


def lnz_colour_values(rows):
    return np.asarray([row["lnZ_harmonic"] for row in rows], dtype=float)


def evidence_norm(values):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    return plt.Normalize(vmin=vmin, vmax=vmax)


def plot_sigma_v_posterior(rows, config):
    summary_csv = (
        config.summary_dir
        / f"manticore_field_sigma_v_summary{config.output_suffix}.csv"
    )
    write_summary(
        rows, ("sigma_v",), summary_csv,
        extra_fields=config.extra_summary_fields)

    medians = np.asarray([row["sigma_v"]["q50"] for row in rows])
    all_samples = np.concatenate([row["samples"]["sigma_v"] for row in rows])
    x_min, x_max = np.percentile(all_samples, [0.2, 99.8])
    pad = 0.15 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)

    if config.colour_by_chain:
        chain_colours = chain_colour_map(rows, "trgbh0_manticore_chains")
    elif config.colour_by_lnz_harmonic:
        cmap = trgbh0_cmap("trgbh0_manticore_lnz_harmonic")
        colour_values = lnz_colour_values(rows)
        norm = plt.Normalize(vmin=np.min(colour_values),
                             vmax=np.max(colour_values))
    else:
        cmap = trgbh0_cmap("trgbh0_manticore_fields_sigma_v")
        norm = plt.Normalize(vmin=0, vmax=config.expected_field_count - 1)
    fields_list, missing = fields_and_missing(
        rows, config.expected_field_count)

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(4.7, 3.1))
        ax.hist(
            medians,
            bins=min(9, max(4, int(np.sqrt(len(medians))) + 2)),
            density=True,
            color="#b8b8b8",
            alpha=0.45,
            label=r"Field median $\sigma_v$",
            zorder=0,
        )
        for row in rows:
            ax.plot(
                x_grid,
                kde_on_grid(row["samples"]["sigma_v"], x_grid),
                color=(
                    chain_colours[row["chain"]]
                    if config.colour_by_chain else cmap(norm(
                        row["lnZ_harmonic"]
                        if config.colour_by_lnz_harmonic else row["field"]))
                ),
                lw=0.9,
                alpha=0.72,
                zorder=2,
            )
        ax.plot(
            x_grid,
            kde_on_grid(all_samples, x_grid),
            color="black",
            lw=1.35,
            label=r"Stacked posterior",
            zorder=4,
        )
        ax.axvline(
            np.median(medians),
            color="black",
            lw=1.0,
            ls=":",
            label=r"Median of field medians",
        )
        ax.set_xlabel(r"$\sigma_v ~ [\mathrm{km}\,\mathrm{s}^{-1}]$")
        ax.set_ylabel("Posterior density")
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(bottom=0)

        note = f"{config.field_label}; {len(fields_list)} fields"
        if missing:
            note += "; missing " + ", ".join(str(i) for i in missing)
        ax.set_title(note, loc="left")

        handles, labels = ax.get_legend_handles_labels()
        if config.colour_by_chain:
            for chain in chain_order(rows):
                handles.append(
                    Line2D([0], [0], color=chain_colours[chain], lw=1.0,
                           label=chain))
                labels.append(chain)
        else:
            curve_proxy = Line2D(
                [0], [0], color=cmap(norm(15)), lw=1.0,
                label=r"Individual field posterior")
            handles.insert(1, curve_proxy)
            labels.insert(1, curve_proxy.get_label())
        ax.legend(
            handles, labels,
            loc="upper right",
            fontsize=6.5,
            frameon=False,
            handlelength=1.7,
        )

        if not config.colour_by_chain:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.055)
            if config.colour_by_lnz_harmonic:
                cbar.set_label(r"Harmonic $\ln Z$")
            else:
                cbar.set_label("Manticore field")
            cbar.ax.tick_params(labelsize=7.0)

        out_pdf = (
            config.output_dir
            / (
                "trgbh0_manticore_field_sigma_v_posteriors"
                f"{config.output_suffix}.pdf"
            )
        )
        out_png = save_pdf_png(fig, out_pdf)

    copy_outputs(config, out_pdf, out_png)
    print_outputs(
        "sigma-v-posterior" + config.diagnostic_name_suffix,
        out_pdf,
        out_png,
        summary_csv,
        rows,
        config,
    )


def scatter_arrays(rows, x_name, y_name):
    x = np.asarray([row[x_name]["q50"] for row in rows])
    x_lo = np.asarray([row[x_name]["q16"] for row in rows])
    x_hi = np.asarray([row[x_name]["q84"] for row in rows])
    y = np.asarray([row[y_name]["q50"] for row in rows])
    y_lo = np.asarray([row[y_name]["q16"] for row in rows])
    y_hi = np.asarray([row[y_name]["q84"] for row in rows])
    return x, x_lo, x_hi, y, y_lo, y_hi


def plot_two_parameter_scatter(
    rows,
    config,
    name,
    x_name,
    y_name,
    x_label,
    y_label,
    cmap_name,
    out_basename,
    summary_basename,
):
    params = (x_name, y_name)
    summary_csv = (
        config.summary_dir / f"{summary_basename}{config.field_suffix}.csv"
    )
    write_summary(rows, params, summary_csv, extra_fields=("lnZ_harmonic",))

    x, x_lo, x_hi, y, y_lo, y_hi = scatter_arrays(rows, x_name, y_name)
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    cmap = trgbh0_cmap(cmap_name)
    colour_values = lnz_colour_values(rows)
    norm = evidence_norm(colour_values)
    colours = cmap(norm(colour_values))

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(4.45, 3.25))
        for i in range(len(rows)):
            ax.errorbar(
                x[i],
                y[i],
                xerr=[[x[i] - x_lo[i]], [x_hi[i] - x[i]]],
                yerr=[[y[i] - y_lo[i]], [y_hi[i] - y[i]]],
                fmt="o",
                ms=3.9,
                lw=0.75,
                elinewidth=0.55,
                capsize=1.6,
                color=colours[i],
                ecolor=colours[i],
                alpha=0.82,
                zorder=2,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{config.field_label}; {len(rows)} fields", loc="left")
        ax.text(
            0.03,
            0.97,
            (
                rf"$r={pearson_r:.2f}$, $p{p_label(pearson_p)}$" "\n"
                rf"$\rho={spearman_r:.2f}$, $p{p_label(spearman_p)}$"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.055)
        cbar.set_label(r"Harmonic $\ln Z$")
        cbar.ax.tick_params(labelsize=7.0)

        out_pdf = (
            config.output_dir / f"{out_basename}{config.field_suffix}.pdf"
        )
        out_png = save_pdf_png(fig, out_pdf)

    copy_outputs(config, out_pdf, out_png)
    print_outputs(
        f"{name}",
        out_pdf,
        out_png,
        summary_csv,
        rows,
        config,
        corr=(pearson_r, pearson_p, spearman_r, spearman_p),
    )


def plot_h0_lnz_harmonic_scatter(rows, config):
    summary_csv = (
        config.summary_dir
        / f"manticore_field_h0_lnz_harmonic_summary{config.field_suffix}.csv"
    )
    write_summary(
        rows, ("H0",), summary_csv,
        extra_fields=("lnZ_harmonic", "err_lnZ_harmonic"))

    x = lnz_colour_values(rows)
    xerr = np.asarray([row["err_lnZ_harmonic"] for row in rows], dtype=float)
    y = np.asarray([row["H0"]["q50"] for row in rows])
    y_lo = np.asarray([row["H0"]["q16"] for row in rows])
    y_hi = np.asarray([row["H0"]["q84"] for row in rows])
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    cmap = trgbh0_cmap("trgbh0_manticore_fields_h0_lnz_harmonic")
    norm = evidence_norm(x)
    colours = cmap(norm(x))

    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(4.45, 3.25))
        for i in range(len(rows)):
            ax.errorbar(
                x[i],
                y[i],
                xerr=abs(xerr[i]),
                yerr=[[y[i] - y_lo[i]], [y_hi[i] - y[i]]],
                fmt="o",
                ms=3.9,
                lw=0.75,
                elinewidth=0.55,
                capsize=1.6,
                color=colours[i],
                ecolor=colours[i],
                alpha=0.82,
                zorder=2,
            )

        ax.set_xlabel(r"Harmonic $\ln Z$")
        ax.set_ylabel(H0_LABEL)
        ax.set_title(f"{config.field_label}; {len(rows)} fields", loc="left")
        ax.text(
            0.03,
            0.97,
            (
                rf"$r={pearson_r:.2f}$, $p{p_label(pearson_p)}$" "\n"
                rf"$\rho={spearman_r:.2f}$, $p{p_label(spearman_p)}$"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.055)
        cbar.set_label(r"Harmonic $\ln Z$")
        cbar.ax.tick_params(labelsize=7.0)

        out_pdf = (
            config.output_dir
            / f"trgbh0_manticore_field_h0_lnz_harmonic{config.field_suffix}.pdf"
        )
        out_png = save_pdf_png(fig, out_pdf)

    copy_outputs(config, out_pdf, out_png)
    print_outputs(
        "h0-lnz-harmonic",
        out_pdf,
        out_png,
        summary_csv,
        rows,
        config,
        corr=(pearson_r, pearson_p, spearman_r, spearman_p),
    )


def requested_plots(items, allow_field_number_colour=False):
    if items:
        invalid = [item for item in items if item not in PLOT_CHOICES]
        if invalid:
            raise ValueError(f"Unknown plot choice(s): {invalid}.")

    if not items:
        plots = list(DEFAULT_PLOTS)
    elif "all" in items:
        plots = [
            item for item in PLOT_CHOICES
            if item != "all"
        ]
    else:
        plots = list(items)

    if allow_field_number_colour:
        return plots

    return [
        item for item in plots
        if item not in FIELD_NUMBER_COLOURED_PLOTS
    ]


def blocked_field_number_plots(items):
    if not items or "all" in items:
        return ()
    return tuple(
        item for item in items
        if item in FIELD_NUMBER_COLOURED_PLOTS
    )


def explain_blocked_field_number_plots(items):
    blocked = blocked_field_number_plots(items)
    if blocked:
        raise ValueError(
            "The following plots would be coloured by Manticore field "
            "number without a colour-mode flag: "
            f"{', '.join(blocked)}. Use --colour-by-lnZ-harmonic or "
            "--colour-by-chain.")


def plot_items_for_args(args, config):
    if not config.has_colour_mode:
        explain_blocked_field_number_plots(args.plots)
    plots = requested_plots(
        args.plots, allow_field_number_colour=config.has_colour_mode)
    if "tempered-evidence-h0" in plots and config.field_set != "cola":
        raise ValueError(
            "`tempered-evidence-h0` is currently defined for the COLA "
            "Manticore field set. Use `--field-set cola`.")
    return plots


def main():
    args = parse_args()
    config = config_from_args(args)
    try:
        plots = plot_items_for_args(args, config)
    except ValueError as err:
        raise SystemExit(f"{Path(__file__).name}: error: {err}") from None
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.summary_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_plots = [
        item for item in plots if item not in SELF_CONTAINED_PLOTS]
    rows = None
    if diagnostic_plots:
        params = tuple(dict.fromkeys(
            param for item in diagnostic_plots for param in PLOT_PARAMS[item]))
        include_lnz_harmonic = (
            any(item in SCATTER_PLOT_SPECS for item in diagnostic_plots)
            or "h0-lnz-harmonic" in diagnostic_plots
            or config.colour_by_lnz_harmonic
        )
        rows = load_rows(
            config, params, include_lnz_harmonic=include_lnz_harmonic)
        if (config.colour_by_chain
                and any(item in FIELD_NUMBER_COLOURED_PLOTS
                        for item in diagnostic_plots)):
            add_chain_metadata(rows, args.manticore_schedule)

    for item in plots:
        if item == "sigma-v-posterior":
            plot_sigma_v_posterior(rows, config)
        elif item == "h0-lnz-harmonic":
            plot_h0_lnz_harmonic_scatter(rows, config)
        elif item in SCATTER_PLOT_SPECS:
            plot_two_parameter_scatter(
                rows, config, name=item, **SCATTER_PLOT_SPECS[item])
        elif item == "evidence-comparison":
            plot_evidence_comparison(config)
        elif item == "evidence-drivers":
            plot_evidence_drivers(config, args)
        elif item == "tempered-evidence-h0":
            plot_tempered_evidence_h0(config, args)
        else:
            raise ValueError(f"Unhandled plot `{item}`.")


if __name__ == "__main__":
    main()
