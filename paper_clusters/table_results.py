"""Generate LaTeX results table for goodness-of-fit statistics."""
from candel import read_gof

from config import RESULTS_ROOT, RESULTS_FOLDER
from utils import stem_from_fname, pm_two_dec, quadrature, safe_get


# Model stems -> pretty names
PRETTY_NAME_MAP = {
    "manticore_Clusters_noMNR_LTYT_hasY": r"No flow or $H_0$ variation",
    "manticore_Clusters_noMNR_LTYT_dipVext_hasY": r"$\mathbf{V}_{\rm ext}$ dipole",
    "manticore_Clusters_noMNR_LTYT_quadVext_hasY": r"$\mathbf{V}_{\rm ext}$ quadrupole",
    "manticore_Clusters_noMNR_LTYT_pixVext_hasY": r"$\mathbf{V}_{\rm ext}$ pixelised",
    "manticore_Clusters_noMNR_LTYT_radVext_hasY": r"$\mathbf{V}_{\rm ext}$ radially varying dipole",
    "manticore_Clusters_noMNR_LTYT_dipA_hasY": r"$H_0$ dipole",
    "manticore_Clusters_noMNR_LTYT_quadA_hasY": r"$H_0$ quadrupole",
    "manticore_Clusters_noMNR_LTYT_pixA_hasY": r"$H_0$ pixelised",
    "manticore_Clusters_noMNR_LTYT_dipVext_dipA_hasY": r"$H_0$ dipole + $\mathbf{V}_{\rm ext}$ dipole",
}

ROW_ORDER = [
    "manticore_Clusters_noMNR_LTYT_hasY",
    "manticore_Clusters_noMNR_LTYT_dipVext_hasY",
    "manticore_Clusters_noMNR_LTYT_quadVext_hasY",
    "manticore_Clusters_noMNR_LTYT_pixVext_hasY",
    "manticore_Clusters_noMNR_LTYT_radVext_hasY",
    "manticore_Clusters_noMNR_LTYT_dipA_hasY",
    "manticore_Clusters_noMNR_LTYT_quadA_hasY",
    "manticore_Clusters_noMNR_LTYT_pixA_hasY",
    "manticore_Clusters_noMNR_LTYT_dipVext_dipA_hasY",
]


def load_task_fnames(task_file):
    """Load filenames from a task file."""
    with open(task_file) as f:
        fnames = [line.strip().split(" ")[1] for line in f if line.strip()]
        fnames = [fname.replace(".toml", ".hdf5") for fname in fnames]
    return fnames


def collect_stats(fnames):
    """Collect GOF statistics from result files."""
    stats = {}
    for fname in fnames:
        if "manticore" not in fname:
            continue
        stem = stem_from_fname(fname)
        if stem not in PRETTY_NAME_MAP:
            continue
        stats[stem] = {
            "lnZ_harmonic": safe_get(read_gof, fname, "lnZ_harmonic"),
            "err_lnZ_harmonic": safe_get(read_gof, fname, "err_lnZ_harmonic"),
            "lnZ_laplace": safe_get(read_gof, fname, "lnZ_laplace"),
            "err_lnZ_laplace": safe_get(read_gof, fname, "err_lnZ_laplace"),
            "BIC": safe_get(read_gof, fname, "BIC"),
        }
    return stats


def build_latex_table(stats, baseline_stem="manticore_Clusters_noMNR_LTYT_hasY"):
    """Build LaTeX table string."""
    zb = stats[baseline_stem].get("lnZ_harmonic")
    ezb = stats[baseline_stem].get("err_lnZ_harmonic")

    header = r"""\begin{table}
\centering
\caption{Relative goodness-of-fit referenced to the baseline model (``No flow or $H_0$ variation''). We report $\Delta\ln Z_{\rm harm} \equiv \ln Z_{\rm harm}-\ln Z_{\rm harm}^{\rm (base)}$, $\Delta\ln Z_{\rm Laplace} \equiv \ln Z_{\rm Laplace}-\ln Z_{\rm harm}^{\rm (base)}$, and $\Delta\ln Z_{\rm BIC}$ where $\ln Z_{\rm BIC}\equiv -\tfrac{1}{2}\,{\rm BIC}$, also referenced to $\ln Z_{\rm harm}^{\rm (base)}$. Uncertainties are shown where available.}
\begin{tabular}{lccc}
\hline
Model & $\Delta\ln Z_{\rm harm}$ & $\Delta\ln Z_{\rm Laplace}$ & $\Delta\ln Z_{\rm BIC}$ \\
\hline
"""

    rows = []
    for stem in ROW_ORDER:
        name = PRETTY_NAME_MAP.get(stem, stem.replace("_", r"\_"))
        s = stats.get(stem, {})

        # Delta lnZ_harm
        z = s.get("lnZ_harmonic")
        ez = s.get("err_lnZ_harmonic")
        dlh = (None if (z is None or zb is None) else (z - zb))
        sdlh = (None if (ez is None and ezb is None) else quadrature(ez, ezb))

        # Delta lnZ_Laplace (still referenced to baseline lnZ_harm)
        zl = s.get("lnZ_laplace")
        ezl = s.get("err_lnZ_laplace")
        dll = (None if (zl is None or zb is None) else (zl - zb))
        sdll = (None if (ezl is None and ezb is None) else quadrature(ezl, ezb))

        # Delta lnZ_BIC: lnZ_BIC = -BIC/2; reference to baseline lnZ_harm
        bic = s.get("BIC")
        lnZ_BIC = (None if bic is None else (-0.5 * bic))
        dlbic = (None if (lnZ_BIC is None or zb is None) else (lnZ_BIC - zb))

        row = " & ".join([
            name,
            pm_two_dec(dlh, sdlh),
            pm_two_dec(dll, sdll),
            pm_two_dec(dlbic, None),
        ]) + r" \\"
        rows.append(row)

    footer = r"""\hline
\end{tabular}
\label{tab:gof_relative_bic}
\end{table*}
"""

    return header + "\n".join(rows) + "\n" + footer


def main():
    task_file = RESULTS_ROOT.parent / "scripts/runs/tasks_0.txt"
    fnames = load_task_fnames(task_file)

    stats = collect_stats(fnames)

    if not stats:
        print("No matching files found. Check task file and result paths.")
        return

    latex_table = build_latex_table(stats)
    print(latex_table)


if __name__ == "__main__":
    main()
