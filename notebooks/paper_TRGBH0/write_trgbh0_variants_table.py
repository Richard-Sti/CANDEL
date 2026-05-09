#!/usr/bin/env python
"""Write the TRGBH0 variants table from posterior samples."""
from pathlib import Path

import h5py
import numpy as np


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
RESULTS = ROOT / "results" / "TRGBH0_paper" / "table"
PAPERDIR = Path("/mnt/users/rstiskalek/Papers/TRGBH0")


RUNS = [
    (
        "EDD_TRGB_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main",
        "\\Manticore",
        "reference",
    ),
    (
        "EDD_TRGB_sel-TRGB_magnitude_Carrick2015_main",
        "\\citetalias{Carrick_2015}",
        "C15 control",
    ),
]


def load_samples(stem):
    with h5py.File(RESULTS / f"{stem}.hdf5", "r") as handle:
        samples = {
            "H0": np.asarray(handle["samples/H0"]),
            "M_TRGB": np.asarray(handle["samples/M_TRGB"]),
            "sigma_v": np.asarray(handle["samples/sigma_v"]),
        }
        for key in ("Vext_mag", "Vext_ell", "Vext_b"):
            if f"samples/{key}" in handle:
                samples[key] = np.asarray(handle[f"samples/{key}"])
        return samples


def summarise(samples):
    q05, q16, q50, q84, q95 = np.percentile(samples, [5, 16, 50, 84, 95])
    return {
        "median": q50,
        "lo": q50 - q16,
        "hi": q84 - q50,
        "q05": q05,
        "q95": q95,
        "std": np.std(samples, ddof=1),
    }


def make_table(rows):
    def pm(stats, value_fmt):
        return (
            f"${value_fmt.format(stats['median'])}"
            f"\\pm{value_fmt.format(stats['std'])}$"
        )

    def optional_pm(stats, value_fmt):
        if stats is None:
            return "--"
        return pm(stats, value_fmt)

    lines = [
        "\\begin{table*}",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Peculiar-velocity model & $H_0$ & $M_{\\rm TRGB}$ & $\\sigma_v$ & $|\\Vext|$ & $\\ell_{\\rm ext}$ & $b_{\\rm ext}$ \\\\",
        " & $[\\kmsecMpc]$ & $[\\rm mag]$ & $[\\kmsec]$ & $[\\kmsec]$ & $[\\rm deg]$ & $[\\rm deg]$ \\\\",
        "\\midrule",
    ]
    for label, note, stats in rows:
        h0 = pm(stats["H0"], "{:.1f}")
        m_trgb = pm(stats["M_TRGB"], "{:.2f}")
        sigma_v = pm(stats["sigma_v"], "{:.0f}")
        vext_mag = optional_pm(stats.get("Vext_mag"), "{:.0f}")
        vext_ell = optional_pm(stats.get("Vext_ell"), "{:.0f}")
        vext_b = optional_pm(stats.get("Vext_b"), "{:.0f}")
        lines.append(
            f"{label} & {h0} & {m_trgb} & {sigma_v} "
            f"& {vext_mag} & {vext_ell} & {vext_b} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Posterior constraints on $H_0$ for the TRGB-only variants.",
        "Entries give posterior medians with posterior standard deviations.",
        "All runs use the EDD TRGB F814W magnitudes, the LMC and NGC\\,4258 TRGB anchors, a TRGB-magnitude selection function, and a uniform-in-volume baseline distance prior.",
        "Reconstruction-based runs additionally include inhomogeneous Malmquist weighting from the density field and evaluate the velocity reconstruction at each galaxy position.",
        "All rows use individual galaxy positions and \\ac{CMB}-frame redshifts.",
        "The~\\citetalias{Carrick_2015} cases test sensitivity to the reconstructed peculiar-velocity model.}",
        "\\label{tab:trgb_h0_variants}",
        "\\end{table*}",
        "",
    ]
    table_path = PAPERDIR / "TRGBH0_variants_table.tex"
    table_path.write_text("\n".join(lines))
    return table_path


def main():
    rows = []
    for stem, label, note in RUNS:
        rows.append(
            (
                label,
                note,
                {param: summarise(values)
                 for param, values in load_samples(stem).items()},
            )
        )

    table_path = make_table(rows)
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
