#!/usr/bin/env python
"""Write the TRGBH0 variants table from posterior samples."""
from pathlib import Path

import h5py
import numpy as np


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
RESULTS = ROOT / "results" / "TRGBH0_paper" / "table"
PAPERDIR = Path("/mnt/users/rstiskalek/Papers/TRGBH0")


ORDERED_LABELS = {
    "EDD_TRGB_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main": (
        "\\Manticore"
    ),
    "EDD_TRGB_sel-TRGB_magnitude_Carrick2015_main": (
        "\\citetalias{Carrick_2015}"
    ),
    "EDD_TRGB_sel-TRGB_magnitude_Vext_main": (
        "No reconstruction + $\\Vext$"
    ),
    "EDD_TRGB_noVext_sel-TRGB_magnitude_main": (
        "No reconstruction, $\\Vext=0$"
    ),
    "EDD_TRGB_cz-student_t_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main": (
        "\\Manticore, Student-$t$ redshifts"
    ),
    "EDD_TRGB_cz-student_t_sel-TRGB_magnitude_Carrick2015_main": (
        "\\citetalias{Carrick_2015}, Student-$t$ redshifts"
    ),
    "EDD_TRGB_Vquad_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main": (
        "\\Manticore, quadrupole"
    ),
    "EDD_TRGB_Vquad_sel-TRGB_magnitude_Carrick2015_main": (
        "\\citetalias{Carrick_2015}, quadrupole"
    ),
    "EDD_TRGB_Voct_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main": (
        "\\Manticore, octupole"
    ),
    "EDD_TRGB_Voct_sel-TRGB_magnitude_Carrick2015_main": (
        "\\citetalias{Carrick_2015}, octupole"
    ),
    "EDD_TRGB_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_sigv_rho_main": (
        "\\Manticore, density-dependent $\\sigma_v$"
    ),
    "EDD_TRGB_sel-TRGB_magnitude_Carrick2015_sigv_rho_main": (
        "\\citetalias{Carrick_2015}, density-dependent $\\sigma_v$"
    ),
}


def discover_runs():
    if not RESULTS.exists():
        raise FileNotFoundError(f"Missing results directory: {RESULTS}")

    stems = sorted(path.stem for path in RESULTS.glob("EDD_TRGB*.hdf5"))
    ordered = [stem for stem in ORDERED_LABELS if stem in stems]
    extra = [stem for stem in stems if stem not in ORDERED_LABELS]
    return [(stem, ORDERED_LABELS.get(stem, stem)) for stem in ordered + extra]


def load_samples(stem):
    with h5py.File(RESULTS / f"{stem}.hdf5", "r") as handle:
        samples = {
            "H0": np.asarray(handle["samples/H0"]),
            "M_TRGB": np.asarray(handle["samples/M_TRGB"]),
        }
        for key in (
            "sigma_v",
            "sigma_v_low",
            "sigma_v_high",
            "Vext_mag",
            "Vext_ell",
            "Vext_b",
        ):
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

    def sigma_v(stats):
        if "sigma_v" in stats:
            return pm(stats["sigma_v"], "{:.0f}")
        if "sigma_v_low" in stats and "sigma_v_high" in stats:
            low = pm(stats["sigma_v_low"], "{:.0f}").strip("$")
            high = pm(stats["sigma_v_high"], "{:.0f}").strip("$")
            return f"${low}\\,/\\,{high}$"
        return "--"

    lines = [
        "\\begin{table*}",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Variant & $H_0$ & $M_{\\rm TRGB}$ & $\\sigma_v$ & $|\\Vext|$ & $\\ell_{\\rm ext}$ & $b_{\\rm ext}$ \\\\",
        " & $[\\kmsecMpc]$ & $[\\rm mag]$ & $[\\kmsec]$ & $[\\kmsec]$ & $[\\rm deg]$ & $[\\rm deg]$ \\\\",
        "\\midrule",
    ]
    for label, stats in rows:
        h0 = pm(stats["H0"], "{:.1f}")
        m_trgb = pm(stats["M_TRGB"], "{:.2f}")
        sig_v = sigma_v(stats)
        vext_mag = optional_pm(stats.get("Vext_mag"), "{:.0f}")
        vext_ell = optional_pm(stats.get("Vext_ell"), "{:.0f}")
        vext_b = optional_pm(stats.get("Vext_b"), "{:.0f}")
        lines.append(
            f"{label} & {h0} & {m_trgb} & {sig_v} "
            f"& {vext_mag} & {vext_ell} & {vext_b} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Posterior constraints on $H_0$ and the leading nuisance parameters for the \\ac{TRGB}-only variants.",
        "Entries are posterior medians with standard deviations.",
        "Rows with density-dependent velocity dispersion report $\\sigma_{v,{\\rm low}}/\\sigma_{v,{\\rm high}}$ in the $\\sigma_v$ column.",
        "All runs use the \\ac{EDD} \\ac{TRGB} F814W magnitudes, the LMC and NGC\\,4258 \\ac{TRGB} anchors, a \\ac{TRGB}-magnitude selection function, individual galaxy positions and \\ac{CMB}-frame redshifts, and a uniform-in-volume baseline distance prior.",
        "Reconstruction-based runs additionally include inhomogeneous Malmquist weighting from the density field and evaluate the velocity reconstruction at each galaxy position.",
        "The \\Manticore\\ row is the fiducial inference.}",
        "\\label{tab:trgb_h0_variants}",
        "\\end{table*}",
        "",
    ]
    table_path = PAPERDIR / "TRGBH0_variants_table.tex"
    table_path.write_text("\n".join(lines))
    return table_path


def main():
    runs = discover_runs()
    if not runs:
        raise RuntimeError(f"No EDD_TRGB HDF5 files found in {RESULTS}")

    rows = []
    for stem, label in runs:
        rows.append(
            (
                label,
                {param: summarise(values)
                 for param, values in load_samples(stem).items()},
            )
        )

    table_path = make_table(rows)
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
