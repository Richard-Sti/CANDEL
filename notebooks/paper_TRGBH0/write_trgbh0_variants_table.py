#!/usr/bin/env python
"""Write the TRGBH0 summary table from task-list posterior summaries."""
from pathlib import Path


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
TASKS = ROOT / "scripts" / "runs" / "tasks_TRGBH0_main.txt"
RESULTS = ROOT / "results" / "TRGBH0_paper" / "table"
PAPERDIR = Path("/mnt/users/rstiskalek/Papers/TRGBH0")
TABLE = PAPERDIR / "TRGBH0_variants_table.tex"


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
    return f"${values['median']:{fmt}}\\pm{values['std']:{fmt}}$"


def task_stems():
    for line in TASKS.read_text().splitlines():
        if not line.strip():
            continue
        _, config = line.split(maxsplit=1)
        yield Path(config).stem


def reconstruction_label(stem):
    if "Carrick2015" in stem:
        return r"\citetalias{Carrick_2015}"
    if "manticore" in stem:
        if "beta_free" in stem:
            return r"\Manticore, free $\beta$"
        return r"\Manticore"
    if "Vext" in stem or stem.startswith("CCHP_sel-"):
        return r"No reconstruction, free $\Vext$"
    return "No reconstruction"


def selection_label(stem):
    if "sel-redshift" in stem:
        return "Redshift"
    if "sel-TRGB_magnitude" in stem:
        return r"\ac{TRGB} magnitude"
    raise ValueError(f"Unknown selection in {stem}")


def redshift_likelihood_label(stem):
    if "cz-student_t" in stem:
        return r"Student-$t$"
    return "Gaussian"


def section_label(stem):
    if stem.startswith("CCHP"):
        return r"\ac{CCHP}"
    if stem.startswith("EDD_TRGB"):
        return r"\ac{EDD}"
    raise ValueError(f"Unknown run section for {stem}")


def discover_rows():
    rows = []
    for stem in task_stems():
        summary = RESULTS / f"{stem}_summary.txt"
        if not summary.exists():
            continue
        values = parse_summary(summary)
        rows.append(
            {
                "section": section_label(stem),
                "reconstruction": reconstruction_label(stem),
                "selection": selection_label(stem),
                "redshift_likelihood": redshift_likelihood_label(stem),
                "h0": format_parameter(values, "H0", ".2f"),
                "m_trgb": format_parameter(values, "M_TRGB", ".2f"),
                "sigma_int": format_parameter(values, "sigma_int", ".2f"),
                "sigma_v": format_parameter(values, "sigma_v", ".0f"),
            }
        )
    return sorted(rows, key=lambda row: (row["section"] != r"\ac{CCHP}"))


def write_table(rows):
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\textwidth}{Xllcccc}",
        r"\toprule",
        r"Reconstruction & Selection & Redshift likelihood & $H_0$ & $M_{\rm TRGB}$ & $\sigma_{\rm int}$ & $\sigma_v$ \\",
        r" & & & $[\kmsecMpc]$ & $[\rm mag]$ & $[\rm mag]$ & $[\kmsec]$ \\",
        r"\midrule",
    ]
    current_section = None
    for row in rows:
        section = row["section"]
        if section != current_section:
            if current_section is not None:
                lines.extend([r"\addlinespace", r"\midrule"])
            lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{section}}}}} \\")
            current_section = section
        lines.append(
            f"{row['reconstruction']} & {row['selection']} & "
            f"{row['redshift_likelihood']} & {row['h0']} & "
            f"{row['m_trgb']} & {row['sigma_int']} & {row['sigma_v']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{Posterior constraints on $H_0$ and leading nuisance parameters for the \ac{CCHP} and \ac{EDD} \ac{TRGB} run sets.",
            r"Entries are posterior medians with standard deviations.",
            r"For Student-$t$ redshift-likelihood rows, $\sigma_v$ is the Gaussian core scale of the residual-velocity likelihood.",
            r"Rows are restricted to \texttt{TRGBH0\_main} task-list entries with matching posterior summaries in the table output directory.}",
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
