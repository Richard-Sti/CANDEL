#!/usr/bin/env python
"""Per-galaxy drivers of the TRGBH0 smoothed model-variant evidence gaps.

The single-field smoothed runs share the same TRGB galaxies and the same 80
Manticore fields across model variants, so per-galaxy log-likelihoods can be
differenced *between variants, matched per field*. This decomposes the large
evidence gaps found in the variant grid (Student-t over Gaussian, sky-exposure
over none) into per-galaxy contributions, separating the magnitude/redshift
likelihood, observed-selection and selection-integral terms.
"""

import csv
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py
import matplotlib

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401,E402
from trgbh0_plot_style import (FIGURE_DPI, OUTPUT_DIR, ROOT,  # noqa: E402
                               TRGBH0_COLOURS, save_pdf_png, set_paper_rc)

DEFAULT_RESULTS_DIR = (
    ROOT / "results" / "TRGBH0_paper" / "single_fields_smoothed")
DEFAULT_OUTDIR = OUTPUT_DIR / "trgbh0_single_smoothed_evidence_drivers"
FILE_GLOB = "*_single_smoothed.hdf5"
AUX = {
    "ll": "auxiliary/log_likelihood_per_galaxy",
    "obs": "auxiliary/log_observed_selection_per_galaxy",
    "full": "auxiliary/log_likelihood_per_galaxy_with_selection",
    "logS": "auxiliary/log_selection_integral",
    "names": "auxiliary/host_names",
}
# (variant A, variant B, title, slug): report A - B, paired per field. A is
# the higher-evidence model so positive deltas favour the richer model.
DEFAULT_PAIRS = (
    ("R4 Stud sky", "R4 Gauss sky", "Student-t vs Gaussian cz",
     "studentt_vs_gaussian"),
    ("R4 Gauss sky", "R4 Gauss", "Sky-exposure on vs off", "sky_vs_nosky"),
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--pair", nargs=2, metavar=("A", "B"), action="append",
        help="Override comparison pairs (variant labels A B); repeatable.")
    parser.add_argument("--top-galaxies", type=int, default=20)
    parser.add_argument("--heatmap-galaxies", type=int, default=40)
    return parser.parse_args()


def variant_label(stem):
    tokens = [f"R{re.search(r'rhoSmoothR(\d+)', stem).group(1)}",
              "Stud" if "cz-student_t" in stem else "Gauss"]
    if "_Vmono_" in stem:
        tokens.append("Vmono")
    if "skyhp_nside1_k48" in stem:
        tokens.append("sky")
    if stem.endswith("_beta_free"):
        tokens.append("beta")
    return " ".join(tokens)


def parse_run(path):
    name = path.name
    stem = re.sub(r"_field\d+_single_smoothed\.hdf5$", "", name)
    field = int(re.search(r"_field(\d+)_", name).group(1))
    return variant_label(stem), field


def decode_names(raw):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x)
                       for x in raw])


def read_galaxy_means(path):
    with h5py.File(path, "r") as handle:
        names = decode_names(handle[AUX["names"]][()])
        ll = np.mean(np.asarray(handle[AUX["ll"]], dtype=float), axis=0)
        obs = np.mean(np.asarray(handle[AUX["obs"]], dtype=float), axis=0)
        full = np.mean(np.asarray(handle[AUX["full"]], dtype=float), axis=0)
        logS = float(np.mean(np.asarray(handle[AUX["logS"]], dtype=float)))
        lnz = float(handle["gof/lnZ_harmonic"][()])
    return {"names": names, "ll": ll, "obs": obs, "full": full,
            "logS": logS, "lnZ": lnz}


def load_variants(results_dir):
    data = {}
    names_ref = None
    for path in sorted(results_dir.glob(FILE_GLOB)):
        label, field = parse_run(path)
        row = read_galaxy_means(path)
        if names_ref is None:
            names_ref = row["names"]
        elif not np.array_equal(row["names"], names_ref):
            raise ValueError(f"Host-name order in `{path}` differs.")
        data.setdefault(label, {})[field] = row
    if not data:
        raise FileNotFoundError(f"No `{FILE_GLOB}` files in `{results_dir}`.")
    return data, names_ref


def pair_delta(data, a, b):
    """Per-galaxy A - B averaged over the shared fields."""
    fields = sorted(set(data[a]) & set(data[b]))
    if not fields:
        raise ValueError(f"No shared fields between `{a}` and `{b}`.")
    dll = np.array([data[a][f]["ll"] - data[b][f]["ll"] for f in fields])
    dobs = np.array([data[a][f]["obs"] - data[b][f]["obs"] for f in fields])
    dfull = np.array([data[a][f]["full"] - data[b][f]["full"] for f in fields])
    dlogS = np.array([data[a][f]["logS"] - data[b][f]["logS"] for f in fields])
    dlnz = np.array([data[a][f]["lnZ"] - data[b][f]["lnZ"] for f in fields])
    n_gal = dll.shape[1]
    return {
        "fields": fields, "n_gal": n_gal,
        "dll_field": dll, "dfull_field": dfull,
        "dll": dll.mean(0), "dll_std": dll.std(0, ddof=1),
        "dobs": dobs.mean(0), "dfull": dfull.mean(0),
        "dfull_std": dfull.std(0, ddof=1),
        "T_ll": float(dll.mean(0).sum()),
        "T_obs": float(dobs.mean(0).sum()),
        "T_logS": float(-n_gal * dlogS.mean()),
        "T_full": float(dfull.mean(0).sum()),
        "dlnz_mean": float(dlnz.mean()),
        "dlnz_se": float(dlnz.std(ddof=1) / np.sqrt(len(fields))),
    }


def plot_drivers(d, names, title, top_n, out_pdf):
    order = np.argsort(d["dfull"])[::-1]
    sorted_abs = np.sort(np.abs(d["dfull"]))[::-1]
    cum = np.cumsum(sorted_abs) / sorted_abs.sum()
    top = order[:top_n][::-1]

    with plt.style.context("science"):
        set_paper_rc()
        fig, (ax_bar, ax_cum) = plt.subplots(
            1, 2, figsize=(7.4, 3.3), constrained_layout=True,
            gridspec_kw={"width_ratios": [1.15, 1.0]})
        ypos = np.arange(len(top))
        ax_bar.barh(ypos, d["dfull"][top], xerr=d["dfull_std"][top],
                    color=TRGBH0_COLOURS[0], alpha=0.82,
                    error_kw={"elinewidth": 0.5, "ecolor": "0.4"})
        ax_bar.set_yticks(ypos)
        ax_bar.set_yticklabels(names[top], fontsize=5.6)
        ax_bar.axvline(0.0, color="0.3", lw=0.7)
        ax_bar.set_xlabel(r"$\Delta\langle\log\mathcal{L}_{\rm full}\rangle$"
                          " (A - B, field mean)")
        ax_bar.set_title(f"Top {top_n} galaxies", loc="left")

        ax_cum.plot(np.arange(1, cum.size + 1), cum, color=TRGBH0_COLOURS[1],
                    lw=1.1)
        ax_cum.axhline(0.9, color="0.4", lw=0.6, ls=":")
        n90 = int(np.searchsorted(cum, 0.9) + 1)
        ax_cum.axvline(n90, color="0.4", lw=0.6, ls=":")
        ax_cum.set_xlabel("Galaxies (ranked by $|\\Delta|$)")
        ax_cum.set_ylabel(r"Cumulative share of $\sum|\Delta\log\mathcal{L}|$")
        ax_cum.set_xlim(0, d["n_gal"])
        ax_cum.set_ylim(0, 1.02)
        ax_cum.set_title(f"{n90}/{d['n_gal']} galaxies reach 90%", loc="left")
        fig.suptitle(title, x=0.012, ha="left", fontsize=8.5)
        return save_pdf_png(fig, out_pdf, dpi=FIGURE_DPI)


def plot_components(d, title, out_pdf):
    totals = [d["T_ll"], d["T_obs"], d["T_logS"], d["T_full"]]
    labels = [r"$\log\mathcal{L}_{m,cz}$", r"$\log p_{\rm obs}$",
              r"$-\log S$", "total"]
    colours = [TRGBH0_COLOURS[1], TRGBH0_COLOURS[2], TRGBH0_COLOURS[3],
               TRGBH0_COLOURS[0]]
    with plt.style.context("science"):
        set_paper_rc()
        fig, (ax_bar, ax_hist) = plt.subplots(
            1, 2, figsize=(7.4, 3.2), constrained_layout=True)
        xpos = np.arange(len(totals))
        ax_bar.bar(xpos, totals, color=colours, alpha=0.84)
        ax_bar.axhline(0.0, color="0.3", lw=0.75)
        ax_bar.set_xticks(xpos)
        ax_bar.set_xticklabels(labels, rotation=20, ha="right")
        ax_bar.set_ylabel("Total delta (A - B), summed over galaxies")
        ax_bar.set_title("Evidence-gap decomposition", loc="left")
        ax_bar.text(0.97, 0.96,
                    rf"$\Delta\ln\mathcal{{Z}}={d['dlnz_mean']:+.1f}$",
                    transform=ax_bar.transAxes, ha="right", va="top",
                    fontsize=6.6)

        vals = d["dfull"]
        lo, hi = np.nanpercentile(vals, [0.5, 99.5])
        bins = np.linspace(min(lo, -0.2), max(hi, 0.2), 60)
        ax_hist.hist(vals, bins=bins, color=TRGBH0_COLOURS[0], alpha=0.7)
        ax_hist.axvline(0.0, color="0.3", lw=0.75, ls=":")
        ax_hist.set_xlabel(
            r"Per-galaxy $\Delta\langle\log\mathcal{L}_{\rm full}"
            r"\rangle$")
        ax_hist.set_ylabel("Number of galaxies")
        ax_hist.set_title("Per-galaxy delta distribution", loc="left")
        fig.suptitle(title, x=0.012, ha="left", fontsize=8.5)
        return save_pdf_png(fig, out_pdf, dpi=FIGURE_DPI)


def plot_heatmap(d, names, title, n_gal, out_pdf):
    gal_idx = np.argsort(np.abs(d["dfull"]))[::-1][:n_gal]
    mat = d["dfull_field"][:, gal_idx]
    vmax = max(np.nanpercentile(np.abs(mat), 98.0), 1e-6)
    with plt.style.context("science"):
        set_paper_rc()
        fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(np.arange(len(gal_idx)))
        ax.set_xticklabels(names[gal_idx], rotation=90, fontsize=4.6)
        ax.set_yticks(np.arange(len(d["fields"])))
        ax.set_yticklabels([str(f) for f in d["fields"]], fontsize=4.8)
        ax.set_ylabel("Manticore field")
        ax.set_title(title, loc="left")
        cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.04)
        cbar.set_label(r"$\Delta\langle\log\mathcal{L}_{\rm full}\rangle$"
                       " (A - B)")
        return save_pdf_png(fig, out_pdf, dpi=FIGURE_DPI)


def write_pair_csv(d, names, path):
    order = np.argsort(d["dfull"])[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["rank", "host_name", "delta_full", "delta_full_std",
                         "delta_ll_mcz", "delta_ll_std", "delta_obs_sel"])
        for rank, idx in enumerate(order, 1):
            writer.writerow([
                rank, names[idx], f"{d['dfull'][idx]:.5f}",
                f"{d['dfull_std'][idx]:.5f}",
                f"{d['dll'][idx]:.5f}", f"{d['dll_std'][idx]:.5f}",
                f"{d['dobs'][idx]:.5f}"])


def write_summary(results, path):
    lines = ["# TRGBH0 Smoothed Evidence-Gap Per-Galaxy Drivers", ""]
    for a, b, title, _slug, d, names in results:
        order = np.argsort(d["dfull"])[::-1]
        share = np.sort(np.abs(d["dfull"]))[::-1]
        share = np.cumsum(share) / share.sum()
        n90 = int(np.searchsorted(share, 0.9) + 1)
        top5 = ", ".join(names[order[:5]])
        lines += [
            f"## {title}  (A={a}, B={b})",
            f"Shared fields: {len(d['fields'])}; galaxies: {d['n_gal']}.",
            f"Mean dlnZ (A-B): {d['dlnz_mean']:+.2f} +- {d['dlnz_se']:.2f}.",
            "Data-fit decomposition (A-B, summed over galaxies):",
            f"  mag/redshift logL : {d['T_ll']:+.2f}",
            f"  observed selection: {d['T_obs']:+.2f}",
            f"  -log S (sel. integ): {d['T_logS']:+.2f}",
            f"  total (full)      : {d['T_full']:+.2f}",
            f"Concentration: {n90} galaxies carry 90% of |delta full|.",
            f"Top-5 driver galaxies: {top5}.",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, names = load_variants(args.results_dir)
    pairs = ([(a, b, f"{a} vs {b}", f"{a}_vs_{b}".replace(" ", "_"))
              for a, b in args.pair] if args.pair else DEFAULT_PAIRS)

    out = args.output_dir
    written = []
    results = []
    for a, b, title, slug in pairs:
        if a not in data or b not in data:
            print(f"[skip] missing variant for pair ({a}, {b}).")
            continue
        d = pair_delta(data, a, b)
        results.append((a, b, title, slug, d, names))
        csv_path = out / f"drivers_{slug}.csv"
        write_pair_csv(d, names, csv_path)
        written += [csv_path,
                    *plot_drivers(d, names, title, args.top_galaxies,
                                  out / f"drivers_{slug}_top.pdf"),
                    *plot_components(d, title,
                                     out / f"drivers_{slug}_components.pdf"),
                    *plot_heatmap(d, names, title, args.heatmap_galaxies,
                                  out / f"drivers_{slug}_heatmap.pdf")]
    if not results:
        raise SystemExit("No comparison pairs could be built.")
    summary = out / "drivers_summary.txt"
    write_summary(results, summary)
    written.append(summary)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
