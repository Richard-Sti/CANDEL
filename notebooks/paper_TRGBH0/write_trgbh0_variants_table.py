#!/usr/bin/env python
"""Write the TRGBH0 posterior-summary tables from posterior samples."""
from pathlib import Path

import h5py
import numpy as np


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
RESULTS = ROOT / "results" / "TRGBH0_paper"
PAPERDIR = Path("/mnt/users/rstiskalek/Papers/TRGBH0")


COMPARISON_RUNS = [
    {
        "model": "\\Manticore",
        "extension": "Density-dependent $\\sigma_v$",
        "path": "table/EDD_TRGB_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_sigv_rho_main.hdf5",
        "fiducial": True,
    },
    {
        "model": "\\Manticore",
        "extension": "Constant $\\sigma_v$",
        "path": "table/EDD_TRGB_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main.hdf5",
    },
    {
        "model": "\\Manticore",
        "extension": "Student-$t$ redshifts",
        "path": "table/EDD_TRGB_cz-student_t_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main.hdf5",
    },
    {
        "model": "\\Manticore",
        "extension": "Quadrupole $\\Vext$",
        "path": "table/EDD_TRGB_Vquad_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main.hdf5",
    },
    {
        "model": "\\Manticore",
        "extension": "Octupole $\\Vext$",
        "path": "table/EDD_TRGB_Voct_sel-TRGB_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2_main.hdf5",
    },
    {
        "model": "\\citetalias{Carrick_2015}",
        "extension": "Density-dependent $\\sigma_v$",
        "path": "table/EDD_TRGB_sel-TRGB_magnitude_Carrick2015_sigv_rho_main.hdf5",
    },
    {
        "model": "\\citetalias{Carrick_2015}",
        "extension": "Constant $\\sigma_v$",
        "path": "table/EDD_TRGB_sel-TRGB_magnitude_Carrick2015_main.hdf5",
    },
    {
        "model": "\\citetalias{Carrick_2015}",
        "extension": "Student-$t$ redshifts",
        "path": "table/EDD_TRGB_cz-student_t_sel-TRGB_magnitude_Carrick2015_main.hdf5",
    },
    {
        "model": "\\citetalias{Carrick_2015}",
        "extension": "Quadrupole $\\Vext$",
        "path": "table/EDD_TRGB_Vquad_sel-TRGB_magnitude_Carrick2015_main.hdf5",
    },
    {
        "model": "\\citetalias{Carrick_2015}",
        "extension": "Octupole $\\Vext$",
        "path": "table/EDD_TRGB_Voct_sel-TRGB_magnitude_Carrick2015_main.hdf5",
    },
    {
        "model": "No reconstruction",
        "extension": "Gaussian redshifts, $\\Vext=0$",
        "path": "table/EDD_TRGB_noVext_sel-TRGB_magnitude_main.hdf5",
    },
    {
        "model": "No reconstruction",
        "extension": "Gaussian redshifts, free $\\Vext$",
        "path": "table/EDD_TRGB_sel-TRGB_magnitude_Vext_main.hdf5",
    },
    {
        "model": "No reconstruction",
        "extension": "Student-$t$ redshifts, free $\\Vext$",
        "path": "table/EDD_TRGB_cz-student_t_sel-TRGB_magnitude_Vext_main.hdf5",
    },
]

PARAMETER_ROWS = [
    ("H0", "$H_0$", "{:.2f}", "$\\kmsecMpc$"),
    ("M_TRGB", "$M_{\\rm TRGB}$", "{:.3f}", "mag"),
    ("c_star", "$c_\\star$", "{:.3f}", "mag"),
    ("mu_LMC", "$\\mu_{\\rm LMC}$", "{:.3f}", "mag"),
    ("mu_N4258", "$\\mu_{\\rm N4258}$", "{:.3f}", "mag"),
    ("sigma_int", "$\\sigma_{\\rm int}$", "{:.3f}", "mag"),
    ("mag_lim_TRGB", "$m_{\\rm lim}$", "{:.2f}", "mag"),
    ("mag_lim_TRGB_width", "$\\sigma_{\\rm sel}$", "{:.2f}", "mag"),
    ("Vext_mag", "$|\\Vext|$", "{:.1f}", "$\\kmsec$"),
    ("Vext_ell", "$\\ell_{\\rm ext}$", "{:.1f}", "deg"),
    ("Vext_b", "$b_{\\rm ext}$", "{:.1f}", "deg"),
    ("alpha_low", "$\\alpha_{\\rm low}$", "{:.2f}", "--"),
    ("alpha_high", "$\\alpha_{\\rm high}$", "{:.3f}", "--"),
    ("log_rho_t", "$\\ln\\rho_t$", "{:.2f}", "--"),
    ("b1", "$b_1$", "{:.2f}", "--"),
    ("beta", "$\\beta$", "{:.3f}", "--"),
    ("sigma_v", "$\\sigma_v$", "{:.1f}", "$\\kmsec$"),
    ("sigma_v_low", "$\\sigma_{v,{\\rm low}}$", "{:.1f}", "$\\kmsec$"),
    ("sigma_v_high", "$\\sigma_{v,{\\rm high}}$", "{:.1f}", "$\\kmsec$"),
    ("log_sigma_v_rho_t", "$\\ln\\rho_{v,t}$", "{:.2f}", "--"),
    ("sigma_v_k", "$k_v$", "{:.2f}", "--"),
    ("nu_cz", "$\\nu$", "{:.2f}", "--"),
]


def discover_runs():
    if not RESULTS.exists():
        raise FileNotFoundError(f"Missing results directory: {RESULTS}")

    runs = []
    for run in COMPARISON_RUNS:
        path = RESULTS / run["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing result: {path}")
        runs.append({**run, "path": path})
    return runs


def load_samples(path):
    with h5py.File(path, "r") as handle:
        samples = {}
        for key in handle["samples"].keys():
            values = np.asarray(handle[f"samples/{key}"])
            if values.ndim == 1:
                samples[key] = values
        return samples


def summarise(samples):
    samples = np.asarray(samples).reshape(-1)
    q05, q16, q50, q84, q95 = np.percentile(samples, [5, 16, 50, 84, 95])
    return {
        "median": q50,
        "lo": q50 - q16,
        "hi": q84 - q50,
        "q05": q05,
        "q95": q95,
        "std": np.std(samples, ddof=1),
        "samples": samples,
    }


def format_value(value, value_fmt):
    text = value_fmt.format(value)
    if text.startswith("-0"):
        zero_text = value_fmt.format(0.0)
        if text == "-" + zero_text:
            return zero_text
    return text


def format_pm(stats, value_fmt):
    if stats is None:
        return "--"
    return (
        f"${format_value(stats['median'], value_fmt)}"
        f"\\pm{format_value(stats['std'], value_fmt)}$"
    )


def make_compact_table(rows):
    def sigma_v(stats):
        if "sigma_v" in stats:
            return format_pm(stats["sigma_v"], "{:.0f}")
        if "sigma_v_low" in stats and "sigma_v_high" in stats:
            low = format_pm(stats["sigma_v_low"], "{:.0f}").strip("$")
            high = format_pm(stats["sigma_v_high"], "{:.0f}").strip("$")
            return f"${low}\\,/\\,{high}$"
        return "--"

    def direction(stats):
        if "Vext_ell" not in stats or "Vext_b" not in stats:
            return "--"
        ell = format_pm(stats["Vext_ell"], "{:.0f}").strip("$")
        b = format_pm(stats["Vext_b"], "{:.0f}").strip("$")
        return f"$({ell},\\,{b})$"

    def extension_parameter(stats):
        if "nu_cz" in stats:
            return f"$\\nu={format_pm(stats['nu_cz'], '{:.1f}').strip('$')}$"
        if "log_sigma_v_rho_t" in stats and "sigma_v_k" in stats:
            rho_t = summarise(np.exp(stats["log_sigma_v_rho_t"]["samples"]))
            rho = format_pm(rho_t, "{:.0f}").strip("$")
            k = format_pm(stats["sigma_v_k"], "{:.1f}").strip("$")
            return f"$\\rho_{{v,t}}={rho}$, $k_v={k}$"
        if "Vext_quad_mag" in stats:
            return (
                "$|\\Vext^{(2)}|="
                f"{format_pm(stats['Vext_quad_mag'], '{:.0f}').strip('$')}$"
            )
        if "Vext_oct_mag" in stats:
            return (
                "$|\\Vext^{(3)}|="
                f"{format_pm(stats['Vext_oct_mag'], '{:.0f}').strip('$')}$"
            )
        return "--"

    lines = [
        "\\begin{table*}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabularx}{\\textwidth}{llccccccX}",
        "\\toprule",
        "Model & Variant & $H_0$ & $M_0$ & $\\sigma_{\\rm int}$ & $\\sigma_v$ & $|\\Vext|$ & $(\\ell_{\\rm ext},b_{\\rm ext})$ & Additional parameter \\\\",
        " & & $[\\kmsecMpc]$ & $[\\rm mag]$ & $[\\rm mag]$ & $[\\kmsec]$ & $[\\kmsec]$ & $[\\rm deg]$ & \\\\",
        "\\midrule",
    ]
    previous_model = None
    for row in rows:
        model = row["model"]
        stats = row["stats"]
        if previous_model is not None and model != previous_model:
            lines.append("\\addlinespace")
        previous_model = model
        label = row["extension"]
        if row.get("fiducial"):
            label += " (fiducial)"
        lines.append(
            f"{model} & {label} & {format_pm(stats.get('H0'), '{:.1f}')} "
            f"& {format_pm(stats['M_TRGB'], '{:.2f}')} "
            f"& {format_pm(stats['sigma_int'], '{:.2f}')} "
            f"& {sigma_v(stats)} & {format_pm(stats.get('Vext_mag'), '{:.0f}')} "
            f"& {direction(stats)} & {extension_parameter(stats)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\caption{Posterior constraints on $H_0$ and leading nuisance parameters for the \\ac{TRGB}-only velocity-model variants.",
        "Entries are posterior medians with standard deviations.",
        "The fiducial row uses \\Manticore\\ with a density-dependent residual velocity dispersion.",
        "The Student-$t$ redshift rows replace the Gaussian redshift residual likelihood by a Student-$t$ likelihood, and $\\nu$ is the inferred number of degrees of freedom.",
        "Rows with density-dependent velocity dispersion report $\\sigma_{v,{\\rm low}}/\\sigma_{v,{\\rm high}}$ in the $\\sigma_v$ column.",
        "For these rows, $\\rho_{v,t}$ is the transition value of $1+\\delta$ and $k_v$ is the logistic sharpness of the transition, with larger $k_v$ giving a faster transition.",
        "The quadrupole and octupole rows add an external multipole to the dipole $\\Vext$, whose amplitude is reported as $|\\Vext^{(2)}|$ or $|\\Vext^{(3)}|$.",
        "All rows use the \\ac{EDD} \\ac{TRGB} F814W magnitudes, the LMC and NGC\\,4258 \\ac{TRGB} anchors, a \\ac{TRGB}-magnitude selection function, and a uniform-in-volume baseline distance prior.",
        "Rows with reconstructed velocity models additionally use individual galaxy positions, \\ac{CMB}-frame redshifts, inhomogeneous Malmquist weighting from the density field, and the velocity reconstruction evaluated at each galaxy position.",
        "Additional nuisance-parameter summaries are listed in~\\cref{tab:trgb_parameter_posteriors}.}",
        "\\label{tab:trgb_h0_variants}",
        "\\end{table*}",
        "",
    ]
    table_path = PAPERDIR / "TRGBH0_variants_table.tex"
    table_path.write_text("\n".join(lines))
    return table_path


def make_parameter_table(rows):
    def selection(stats):
        return (
            f"{format_pm(stats.get('mag_lim_TRGB'), '{:.2f}').strip('$')}, "
            f"{format_pm(stats.get('mag_lim_TRGB_width'), '{:.2f}').strip('$')}"
        )

    def reconstruction_params(stats):
        if "alpha_low" in stats:
            low = format_pm(stats["alpha_low"], "{:.2f}").strip("$")
            high = format_pm(stats["alpha_high"], "{:.2f}").strip("$")
            rho = format_pm(stats["log_rho_t"], "{:.2f}").strip("$")
            return (
                f"$\\alpha_{{\\rm low}}={low}$, "
                f"$\\alpha_{{\\rm high}}={high}$, "
                f"$\\ln\\rho_t={rho}$"
            )
        if "b1" in stats:
            b1 = format_pm(stats["b1"], "{:.2f}").strip("$")
            beta = format_pm(stats["beta"], "{:.3f}").strip("$")
            return f"$b_1={b1}$, $\\beta={beta}$"
        return "--"

    def likelihood_params(stats):
        parts = []
        if "nu_cz" in stats:
            parts.append(f"$\\nu={format_pm(stats['nu_cz'], '{:.2f}').strip('$')}$")
        if "log_sigma_v_rho_t" in stats:
            parts.append(
                "$\\ln\\rho_{v,t}="
                f"{format_pm(stats['log_sigma_v_rho_t'], '{:.2f}').strip('$')}$"
            )
            parts.append(
                f"$k_v={format_pm(stats['sigma_v_k'], '{:.2f}').strip('$')}$"
            )
        if "Vext_quad_mag" in stats:
            parts.append(
                "$|\\Vext^{(2)}|="
                f"{format_pm(stats['Vext_quad_mag'], '{:.0f}').strip('$')}$"
            )
        if "Vext_oct_mag" in stats:
            parts.append(
                "$|\\Vext^{(3)}|="
                f"{format_pm(stats['Vext_oct_mag'], '{:.0f}').strip('$')}$"
            )
        return ", ".join(parts) if parts else "--"

    lines = [
        "\\begin{table*}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabularx}{\\textwidth}{llccclX}",
        "\\toprule",
        "Model & Variant & $c_\\star$ & $\\mu_{\\rm LMC}$ & $\\mu_{\\rm N4258}$ & Selection & Additional nuisance parameters \\\\",
        " & & $[\\rm mag]$ & $[\\rm mag]$ & $[\\rm mag]$ & $m_{\\rm lim},\\sigma_{\\rm sel}$ & \\\\",
        "\\midrule",
    ]
    previous_model = None
    for row in rows:
        model = row["model"]
        stats = row["stats"]
        if previous_model is not None and model != previous_model:
            lines.append("\\addlinespace")
        previous_model = model
        label = row["extension"]
        if row.get("fiducial"):
            label += " (fiducial)"
        nuisance = reconstruction_params(stats)
        extra = likelihood_params(stats)
        if extra != "--":
            nuisance = f"{nuisance}; {extra}" if nuisance != "--" else extra
        lines.append(
            f"{model} & {label} "
            f"& {format_pm(stats.get('c_star'), '{:.3f}')} "
            f"& {format_pm(stats.get('mu_LMC'), '{:.3f}')} "
            f"& {format_pm(stats.get('mu_N4258'), '{:.3f}')} "
            f"& ${selection(stats)}$ "
            f"& {nuisance} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\caption{Additional scalar posterior summaries for the \\ac{TRGB}-only velocity-model variants in~\\cref{tab:trgb_h0_variants}.",
        "Entries are posterior medians with standard deviations.",
        "The selection column reports the inferred \\ac{TRGB}-magnitude selection threshold and width.",
        "\\Manticore\\ rows report the double-power-law bias parameters, while~\\citetalias{Carrick_2015} rows report the linear source-density bias and velocity-field amplitude.",
        "The final column also lists the Student-$t$ degrees of freedom, density-dependent-$\\sigma_v$ transition parameters, or external multipole amplitudes when present.}",
        "\\label{tab:trgb_parameter_posteriors}",
        "\\end{table*}",
        "",
    ]
    table_path = PAPERDIR / "TRGBH0_parameter_table.tex"
    table_path.write_text("\n".join(lines))
    return table_path


def main():
    rows = []
    for run in discover_runs():
        samples = load_samples(run["path"])
        stats = {param: summarise(values) for param, values in samples.items()}
        rows.append({**run, "stats": stats})

    for table_path in (make_compact_table(rows), make_parameter_table(rows)):
        print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
