#!/usr/bin/env python
"""Write the TRGBH0 summary table from task-list posterior summaries."""
import math
import sys
from pathlib import Path

import h5py
from trgbh0_plot_style import PAPER_DIR, ROOT, TRGBH0_TABLE_RESULTS

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TASKS = ROOT / "scripts" / "runs" / "tasks_TRGBH0_main.txt"
RESULTS = TRGBH0_TABLE_RESULTS
PAPERDIR = PAPER_DIR
TABLE = PAPERDIR / "TRGBH0_variants_table.tex"
TRACK_CHANGES = True
NOT_RUN = "X"


def parse_summary(path):
    summary = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6:
            try:
                summary[fields[0]] = {
                    "std": float(fields[2]),
                    "median": float(fields[3]),
                }
            except ValueError:
                continue
    return summary


def format_parameter(summary, name, fmt):
    if name not in summary:
        raise ValueError(f"Could not find {name} row in summary")
    values = summary[name]
    return red(f"${values['median']:{fmt}}\\pm{values['std']:{fmt}}$")


def log10_evidence(path):
    with h5py.File(path, "r") as handle:
        try:
            ln_z = float(handle["gof/lnZ_harmonic"][()])
        except KeyError as err:
            raise ValueError(
                f"Could not find gof/lnZ_harmonic in {path}") from err
    return ln_z / math.log(10.0)


def format_delta_log10_evidence(value):
    return red(f"${value:.2f}$")


def red(text):
    if not TRACK_CHANGES:
        return text
    return rf"\red{{{text}}}"


def task_stems():
    for line in TASKS.read_text().splitlines():
        if not line.strip():
            continue
        _, config = line.split(maxsplit=1)
        yield Path(config).stem


def reconstruction_label(stem):
    stem_lower = stem.lower()
    if "carrick2015" in stem_lower:
        if "double_powerlaw" in stem_lower:
            if "beta_0p43" in stem_lower:
                return r"\citetalias{Carrick_2015}, double power law, $\beta=0.43$"  # noqa: E501
            if "beta_0p48" in stem_lower:
                return r"\citetalias{Carrick_2015}, double power law, $\beta=0.48$"  # noqa: E501
            return r"\citetalias{Carrick_2015}, double power law"
        label = r"\citetalias{Carrick_2015}"
        if "vmono" in stem_lower:
            label += r", $V_{\rm mono}$"
        if "voct" in stem_lower:
            label += r", octupole $\Vext$"
        return label
    if "manticore" in stem_lower:
        label = r"\Manticore, $R_\rho=4\Mpch$"
        if "beta_free" in stem_lower:
            label += r", free $\beta$"
        if "vmono" in stem_lower:
            label += r", $V_{\rm mono}$"
        if "voct" in stem_lower:
            label += r", octupole $\Vext$"
        return label
    if "Vext" in stem or stem.startswith("CCHP_sel-"):
        return r"No reconstruction, free $\Vext$"
    return "No reconstruction"


def redshift_likelihood_label(stem):
    if "cz-student_t" in stem:
        return r"Student-$t$"
    return "Gaussian"


def sky_exposure_label(stem):
    return red(r"\checkmark") if "skyhp" in stem else red("--")


def discover_rows():
    rows = []
    for stem in task_stems():
        # The redshift-free run is a distance anchor, not an H0 variant row.
        if "no_TRGB_redshift" in stem:
            continue
        summary = RESULTS / f"{stem}_summary.txt"
        samples = RESULTS / f"{stem}.hdf5"
        row = {
            "reconstruction": reconstruction_label(stem),
            "redshift_likelihood": redshift_likelihood_label(stem),
            "sky_exposure": sky_exposure_label(stem),
        }
        if summary.exists() and samples.exists():
            values = parse_summary(summary)
            row.update(
                {
                    "h0": format_parameter(values, "H0", ".2f"),
                    "m_trgb": format_parameter(values, "M_TRGB", ".2f"),
                    "sigma_int": format_parameter(values, "sigma_int", ".2f"),
                    "sigma_v": format_parameter(values, "sigma_v", ".0f"),
                    "log10_evidence": log10_evidence(samples),
                }
            )
        else:
            # Task-list variant whose posterior chains are not yet available.
            row.update(
                {
                    "h0": red(NOT_RUN),
                    "m_trgb": red(NOT_RUN),
                    "sigma_int": red(NOT_RUN),
                    "sigma_v": red(NOT_RUN),
                    "log10_evidence": None,
                    "delta_log10_evidence": red(NOT_RUN),
                }
            )
        rows.append(row)
    evidences = [row["log10_evidence"] for row in rows
                 if row["log10_evidence"] is not None]
    best = max(evidences)
    for row in rows:
        if row["log10_evidence"] is not None:
            row["delta_log10_evidence"] = format_delta_log10_evidence(
                row["log10_evidence"] - best)
    return rows


def write_table(rows):
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabularx}{\textwidth}{Xlcccccc}",
        r"\toprule",
        r"Reconstruction & Redshift likelihood & Sky exposure & $H_0$ & $M_{\rm TRGB}$ & $\sigma_{\rm int}$ & $\sigma_v$ & $\Delta\log_{10} Z_{\rm harm}$ \\",  # noqa: E501
        r" & & & $[\kmsecMpc]$ & $[\rm mag]$ & $[\rm mag]$ & $[\kmsec]$ & \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['reconstruction']} & {row['redshift_likelihood']} & "
            f"{row['sky_exposure']} & {row['h0']} & "
            f"{row['m_trgb']} & {row['sigma_int']} & {row['sigma_v']} & "
            f"{row['delta_log10_evidence']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{Posterior constraints on $H_0$ and leading nuisance parameters for the \ac{EDD} \ac{TRGB} run set.",  # noqa: E501
            r"Entries are posterior medians with standard deviations.",
            r"\red{For the \Manticore\ rows, the likelihood marginalises over the $80$ COLA density--velocity realisations through the realisation average in~\cref{eq:detected_posterior}.}",  # noqa: E501
            r"\red{The sky-exposure column marks whether the angular \ac{HST} sky-exposure selection term of~\cref{sec:angular_selection} is active.}",  # noqa: E501
            r"For Student-$t$ redshift-likelihood rows, $\sigma_v$ is the Gaussian core scale of the residual-velocity likelihood.",  # noqa: E501
            r"The evidence column reports the Bayesian evidence estimated from the posterior chains with the normalising-flow learnt harmonic-mean estimator of the \texttt{harmonic} package (\cref{sec:inference_config}), quoted as $\Delta\log_{10}Z_{\rm harm}$ relative to the highest-evidence row in the table.",  # noqa: E501
            r"For no-reconstruction rows we include the analytic full-sky angular-density correction, subtracting $N\log_{10}(4\pi)$ from the radial-only output before normalising the block where applicable.",  # noqa: E501
            r"\red{Rows correspond to the current \texttt{TRGBH0\_main} task-list entries; variants whose posterior chains are not yet available in \texttt{results/TRGBH0\_paper/table} are marked \red{X}.}}",  # noqa: E501
            r"\label{tab:trgb_h0_variants}",
            r"\end{table*}",
            "",
        ]
    )
    TABLE.write_text("\n".join(lines))


def main():
    rows = discover_rows()
    write_table(rows)
    print(f"Wrote {TABLE}")


if __name__ == "__main__":
    main()
