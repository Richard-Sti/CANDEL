# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Corner and GetDist plotting helpers."""

from os.path import basename
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from getdist import MCSamples, plots
from h5py import File

from candel.util import fprint


def name2label(name):
    """
    Map internal parameter names to LaTeX labels, optionally including
    catalogue prefix.
    """
    latex_labels = {
        "a_TFR": r"$a_\mathrm{TFR}$",
        "b_TFR": r"$b_\mathrm{TFR}$",
        "c_TFR": r"$c_\mathrm{TFR}$",
        "sigma_int": r"$\sigma_{\rm int}$",
        "sigma_v": r"$\sigma_v$",
        "sigma_v_low": r"$\sigma_{v,{\rm low}}$",
        "sigma_v_high": r"$\sigma_{v,{\rm high}}$",
        "log_sigma_v_rho_t": r"$\ln\rho_{v,t}$",
        "sigma_v_k": r"$k_v$",
        "alpha": r"$\alpha$",
        "alpha_low": r"$\alpha_\mathrm{low}$",
        "alpha_high": r"$\alpha_\mathrm{high}$",
        "log_rho_t": r"$\ln \rho_t$",
        "log_rho_width": r"$\Delta \ln \rho$",
        "b1": r"$b_1$",
        "b2": r"$b_2$",
        "b3": r"$b_3$",
        "beta": r"$\beta$",
        "Vext_mag": r"$V_\mathrm{ext}$",
        "Vext_ell": r"$\ell_\mathrm{ext}$",
        "Vext_b": r"$b_\mathrm{ext}$",
        "logM_miss": r"$\log_{10} M_{\rm miss}$",
        "Mmiss_distance": r"$r_{\rm miss}$",
        "Mmiss_ell": r"$\ell_{\rm miss}$",
        "Mmiss_b": r"$b_{\rm miss}$",
        "h": r"$h$",
        "a": r"$a$",
        "m1": r"$m_1$",
        "m2": r"$m_2$",
        "zeropoint_dipole_mag": r"$\Delta \mathrm{ZP}$",
        "zeropoint_dipole_ell": r"$\ell_{\Delta \mathrm{ZP}}$",
        "zeropoint_dipole_b": r"$b_{\Delta \mathrm{ZP}}$",
        "SN_absmag": r"$M_{\rm SN}$",
        "SN_alpha": r"$\mathcal{A}$",
        "SN_beta": r"$\mathcal{B}$",
        "eta_prior_mean": r"$\hat{\eta}$",
        "eta_prior_std": r"$w_\eta$",
        "A_CL": r"$A_{\rm CL}$",
        "B_CL": r"$B_{\rm CL}$",
        "C_CL": r"$C_{\rm CL}$",
        "a_FP": r"$a_{\rm FP}$",
        "b_FP": r"$b_{\rm FP}$",
        "c_FP": r"$c_{\rm FP}$",
        "sigma_log_theta": r"$\sigma_{\log \theta}$",
        "R_dust": r"$R_{\rm W1}$",
        "R_dist_emp": r"$R_{\rm dist}$",
        "q_dist_emp": r"$q_{\rm dist}$",
        "Rmax_dist_emp": r"$R_{\rm max, dist}$",
        "rho_corr": r"$\rho_{\rm corr}$",
        "Vext_radmag_ell": r"$\ell_{\mathrm{Vext}}$",
        "Vext_radmag_b": r"$b_{\mathrm{Vext}}$",
        "H0": r"$H_0$",
        "M_SN": r"$M_{\rm SN}$",
        "M_TRGB": r"$M_{\rm TRGB}$",
        "mag_lim_SN": r"$m_{\rm lim}^{\rm SN}$",
        "mag_lim_SN_width": r"$\sigma_{m,{\rm lim}}^{\rm SN}$",
        "mu_LMC": r"$\mu_{\rm LMC}$",
        "c_star": r"$c_\star$",
        "c_bar": r"$\bar{c}$",
        "w_c": r"$w_c$",
        "mag_lim_TRGB": r"$m_{\rm lim}$",
        "mag_lim_TRGB_width": r"$\sigma_{\rm sel}$",
        "nu_cz": r"$\nu$",
        "mu_N4258": r"$\mu_{\rm N4258}$",
        "D_c": r"$D_c$",
        "D_A": r"$D_A$",
        "eta": r"$\eta$",
        "log_MBH": r"$\log M_{\rm BH}$",
        "M_BH": r"$M_{\rm BH}$",
        "i0": r"$i_0$",
        "di_dr": r"$\mathrm{d}i/\mathrm{d}r$",
        "d2i_dr2": r"$\mathrm{d}^2i/\mathrm{d}r^2$",
        "Omega0": r"$\Omega_0$",
        "dOmega_dr": r"$\mathrm{d}\Omega/\mathrm{d}r$",
        "d2Omega_dr2": r"$\mathrm{d}^2\Omega/\mathrm{d}r^2$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
        "dv_sys": r"$\Delta v_{\rm sys}$",
        "sigma_x_floor": r"$\sigma_{x,\mathrm{fl}}$",
        "sigma_y_floor": r"$\sigma_{y,\mathrm{fl}}$",
        "sigma_v_sys": r"$\sigma_{v,\mathrm{sys}}$",
        "sigma_v_hv": r"$\sigma_{v,\mathrm{hv}}$",
        "sigma_a_floor": r"$\sigma_{a,\mathrm{fl}}$",
        "ecc": r"$e$",
        "e_x": r"$e_x$",
        "e_y": r"$e_y$",
        "periapsis": r"$\omega$",
        "periapsis_rad": r"$\omega$",
        "dperiapsis_dr": r"$\mathrm{d}\omega/\mathrm{d}r$",
        "sigma_pec": r"$\sigma_{\rm pec}~[\mathrm{km/s}]$",
        "D_lim": r"$D_{\rm lim}$",
        "D_width": r"$\sigma_{D,\rm lim}$",
    }

    if "/" in name:
        prefix, base = name.split("/", 1)
        base_label = latex_labels.get(base, base)
        prefix_latex = prefix.replace("_", r"\,").replace(" ", "~")
        return rf"$\mathrm{{{prefix_latex}}},\,{base_label.strip('$')}$"

    return latex_labels.get(name, name)


def name2labelgetdist(name):
    """Return a GetDist-compatible LaTeX label without ``$...$``."""
    labels = {
        "a_TFR": r"a_\mathrm{TFR}",
        "b_TFR": r"b_\mathrm{TFR}",
        "c_TFR": r"c_\mathrm{TFR}",
        "SN_absmag": r"M_{\rm SN}",
        "SN_alpha": r"\mathcal{A}",
        "SN_beta": r"\mathcal{B}",
        "sigma_int": r"\sigma_{\rm int}",
        "sigma_v": r"\sigma_v~\left[\mathrm{km}\,\mathrm{s}^{-1}\right]",
        "sigma_v_low": (
            r"\sigma_{v,\rm low}~\left[\mathrm{km}\,\mathrm{s}^{-1}\right]"
        ),
        "sigma_v_high": (
            r"\sigma_{v,\rm high}~\left[\mathrm{km}\,\mathrm{s}^{-1}\right]"
        ),
        "log_sigma_v_rho_t": r"\ln\rho_{v,t}",
        "sigma_v_k": r"k_v",
        "alpha": r"\alpha",
        "alpha_low": r"\alpha_\mathrm{low}",
        "alpha_high": r"\alpha_\mathrm{high}",
        "log_rho_t": r"\ln \rho_t",
        "log_rho_width": r"\Delta \ln \rho",
        "b1": r"b_1",
        "b2": r"b_2",
        "b3": r"b_3",
        "beta": r"\beta",
        "Vext_mag": r"V_\mathrm{ext}~\left[\mathrm{km}\,\mathrm{s}^{-1}\right]",  # noqa
        "Vext_ell": r"\ell_\mathrm{ext}~\left[\mathrm{deg}\right]",
        "Vext_ell_offset": r"\ell_\mathrm{ext} - 180~\left[\mathrm{deg}\right]",  # noqa
        "Vext_b":   r"b_\mathrm{ext}~\left[\mathrm{deg}\right]",
        "logM_miss": r"\log_{10} M_{\rm miss}",
        "Mmiss_distance": r"r_{\rm miss}~\left[h^{-1}\,\mathrm{Mpc}\right]",
        "Mmiss_ell": r"\ell_{\rm miss}~\left[\mathrm{deg}\right]",
        "Mmiss_b": r"b_{\rm miss}~\left[\mathrm{deg}\right]",
        "h": r"h",
        "a": r"a",
        "m1": r"m_1",
        "m2": r"m_2",
        "zeropoint_dipole_mag": r"\Delta_\mathrm{ZP}",
        "zeropoint_dipole_ell": r"\ell_{\Delta_\mathrm{ZP}}~\left[\mathrm{deg}\right]",  # noqa
        "zeropoint_dipole_b": r"b_{\Delta_\mathrm{ZP}}~\left[\mathrm{deg}\right]",       # noqa
        "M_dipole_mag": r"\Delta M_\mathrm{SN}",
        "M_dipole_ell": r"\ell_{\Delta M_{\rm SN}}~\left[\mathrm{deg}\right]",
        "M_dipole_b": r"b_{\Delta M_{\rm SN}}~\left[\mathrm{deg}\right]",
        "eta_prior_mean": r"\hat{\eta}",
        "eta_prior_std": r"w_\eta",
        "A_CL": r"A_{\rm CL}",
        "B_CL": r"B_{\rm CL}",
        "C_CL": r"C_{\rm CL}",
        "a_FP": r"a_{\rm FP}",
        "b_FP": r"b_{\rm FP}",
        "c_FP": r"c_{\rm FP}",
        "R_dust": r"R_{\rm W1}",
        "mu_LMC": r"\mu_{\rm LMC}",
        "mu_M31": r"\mu_{\rm M31}",
        "mu_N4258": r"\mu_{\rm NGC4258}",
        "M_TRGB": r"M_{\rm TRGB}",
        "c_star": r"c_\star",
        "c_bar": r"\bar{c}",
        "w_c": r"w_c",
        "mag_lim_TRGB": r"m_{\rm lim}",
        "mag_lim_TRGB_width": r"\sigma_{\rm sel}",
        "nu_cz": r"\nu",
        "H0": r"H_0~\left[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}\right]",  # noqa
        "D_c": r"D_c",
        "D_A": r"D_A",
        "eta": r"\eta",
        "log_MBH": r"\log M_{\rm BH}",
        "M_BH": r"M_{\rm BH}",
        "i0": r"i_0",
        "di_dr": r"\mathrm{d}i/\mathrm{d}r",
        "d2i_dr2": r"\mathrm{d}^2i/\mathrm{d}r^2",
        "Omega0": r"\Omega_0",
        "dOmega_dr": r"\mathrm{d}\Omega/\mathrm{d}r",
        "d2Omega_dr2": r"\mathrm{d}^2\Omega/\mathrm{d}r^2",
        "x0": r"x_0",
        "y0": r"y_0",
        "dv_sys": r"\Delta v_{\rm sys}",
        "sigma_x_floor": r"\sigma_{x,\mathrm{fl}}",
        "sigma_y_floor": r"\sigma_{y,\mathrm{fl}}",
        "sigma_v_sys": r"\sigma_{v,\mathrm{sys}}",
        "sigma_v_hv": r"\sigma_{v,\mathrm{hv}}",
        "sigma_a_floor": r"\sigma_{a,\mathrm{fl}}",
        "ecc": r"e",
        "e_x": r"e_x",
        "e_y": r"e_y",
        "periapsis": r"\omega",
        "periapsis_rad": r"\omega",
        "dperiapsis_dr": r"\mathrm{d}\omega/\mathrm{d}r",
        "sigma_pec": r"\sigma_{\rm pec}~\left[\mathrm{km}\,\mathrm{s}^{-1}\right]",  # noqa
        "D_lim": r"D_{\rm lim}",
        "D_width": r"\sigma_{D,\rm lim}",
        "dZP": r"\Delta_{\rm ZP}",
        "R_dist_emp": r"R~\left[h^{-1}\,\mathrm{Mpc}\right]",
        "q_dist_emp": r"q",
        "rho_corr": r"\rho_{\rm corr}",
    }

    if "/" in name:
        prefix, base = name.split("/", 1)
        base_label = labels.get(base, base)
        prefix_latex = prefix.replace("_", r"\,").replace(" ", "~")
        return rf"\mathrm{{{prefix_latex}}},\,{base_label}"

    return labels.get(name, name)


def sort_params(keys):
    order = [
        "H0", "D_c", "D_A", "log_MBH", "M_BH", "eta", "dv_sys",
        "sigma_pec", "D_lim", "D_width", "x0", "y0", "i0", "Omega0",
        "di_dr", "dOmega_dr", "d2i_dr2", "d2Omega_dr2",
        "sigma_x_floor", "sigma_y_floor", "sigma_v_sys", "sigma_v_hv",
        "sigma_a_floor", "ecc", "e_x", "e_y", "periapsis",
        "periapsis_rad", "dperiapsis_dr", "a_TFR", "b_TFR", "c_TFR",
        "alpha", "beta", "sigma_int", "sigma_v", "logM_miss",
        "Mmiss_distance", "Mmiss_ell", "Mmiss_b", "Vext", "Vext_mag",
        "Vext_ell", "Vext_b"
    ]

    def sort_key(k):
        prefix, base = k.split("/", 1) if "/" in k else ("", k)
        try:
            return (order.index(base), prefix, k)
        except ValueError:
            return (len(order), prefix, k)

    return sorted(keys, key=sort_key)


def plot_corner(samples, show_fig=True, filename=None, smooth=1, keys=None):
    """Plot a corner plot from posterior samples."""
    flat_samples = []
    labels = []

    if keys is None:
        keys = sort_params(list(samples.keys()))

    for k in keys:
        if k not in samples:
            continue
        v = np.asarray(samples[k])

        if k == "Vext_radmag_mag":
            nbin = v.shape[1]
            for i in range(nbin):
                flat_samples.append(v[:, i])
                labels.append(fr"$V_{{\mathrm{{ext}}, {{{i}}}}}$")

        if v.ndim > 1:
            continue
        v = np.asarray(v).reshape(-1)
        if np.ptp(v) == 0:
            continue
        flat_samples.append(v)
        labels.append(name2label(k))

    if not flat_samples:
        raise ValueError("No valid samples to plot.")

    data = np.vstack(flat_samples).T
    fig = corner(data, labels=labels, show_titles=True, smooth=smooth)

    if filename is not None:
        fprint(f"saving a corner plot to {filename}")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def plot_Vext_rad_corner(samples, show_fig=True, filename=None, smooth=1):
    """Plot a corner plot of Vext_rad_{mag, ell, b} samples."""
    keys = ["Vext_rad_mag", "Vext_rad_ell", "Vext_rad_b"]
    base_labels = [r"V", r"\ell", r"b"]

    arrays = []
    labels = []
    for key, base_label in zip(keys, base_labels):
        if key not in samples:
            raise ValueError(f"Missing key: {key}")

        arr = samples[key]
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim != 2:
            raise ValueError(f"{key} must be 2D or 3D")

        ndim = arr.shape[1]
        arrays.append(arr)
        for i in range(ndim):
            labels.append(fr"${base_label}_{{{i}}}$")

    data = np.hstack(arrays)
    fig = corner(data, labels=labels, show_titles=True, smooth=smooth)

    if filename is not None:
        fprint(f"saving knots corner plot to {filename}")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_corner_getdist(samples_list, labels=None, cols=None, show_fig=True,
                        filename=None, keys=None, fontsize=None,
                        legend_fontsize=None, filled=True,
                        apply_ell_offset=False, mag_range=None,
                        ell_range=None, b_range=None, points=None,
                        ranges=None, truths=None):
    """Plot a GetDist triangle plot for one or more posterior samples."""
    if mag_range is None:
        mag_range = [0, None]
    if ell_range is None:
        ell_range = [0, 360]
    if b_range is None:
        b_range = [-90, 90]
    if ranges is None:
        ranges = {}

    try:
        import scienceplots  # noqa
        use_scienceplots = True
    except ImportError:
        warn("scienceplots not found, using default plotting style.")
        use_scienceplots = False

    if isinstance(samples_list, dict):
        samples_list = [samples_list]

    if labels is not None and len(labels) != len(samples_list):
        raise ValueError("Length of `labels` must match number of sample sets")

    if keys is not None:
        candidate_keys = keys
    else:
        candidate_keys = [
            k for k in samples_list[0] if samples_list[0][k].ndim == 1]

    param_names = []
    for k in candidate_keys:
        for s in samples_list:
            if k in s and s[k].ndim == 1:
                param_names.append(k)
                break
            elif k in s and s[k].ndim > 1:
                fprint(f"[SKIP] {k} has shape {s[k].shape}")
                break

    if keys is None:
        param_names = sort_params(param_names)

    for k in param_names:
        if "_mag" in k:
            ranges[k] = mag_range
        if "_ell" in k:
            ranges[k] = ell_range
        if "_b" in k:
            ranges[k] = b_range

    gdsamples_list = []
    for samples in samples_list:
        present_params = []
        present_labels = []
        columns = []

        n_samples = len(next(iter(samples.values())))
        for k in param_names:
            if k in samples and samples[k].ndim == 1:
                col = samples[k].reshape(-1)
            else:
                col = np.full(n_samples, np.nan)

            if not np.all(np.isnan(col)):
                if "_ell" in k and apply_ell_offset:
                    col = (col - 180) % 360
                    label = name2labelgetdist(k + "_offset")
                else:
                    label = name2labelgetdist(k)

                present_params.append(k)
                present_labels.append(label)
                columns.append(col)

        data = np.vstack(columns).T
        gds = MCSamples(
            samples=data,
            names=present_params,
            labels=present_labels,
            ranges={k: ranges[k] for k in present_params if k in ranges},
        )
        gdsamples_list.append(gds)

    settings = plots.GetDistPlotSettings()
    if fontsize is not None:
        settings.lab_fontsize = fontsize
        settings.legend_fontsize = (
            legend_fontsize if legend_fontsize is not None else fontsize)
        settings.axes_fontsize = fontsize - 1
        settings.title_limit_fontsize = fontsize - 1

    line_args = [{"color": c} for c in cols] if cols is not None else None

    with plt.style.context("science" if use_scienceplots else "default"):
        g = plots.get_subplot_plotter(settings=settings)
        g.triangle_plot(
            gdsamples_list,
            params=param_names,
            filled=filled,
            colors=cols,
            contour_colors=cols,
            line_args=line_args,
            legend_labels=labels,
            legend_loc="upper right",
        )

        if points is not None:
            plotted_pairs = set()
            for (x_param, y_param), (x_val, y_val) in points.items():
                if x_param not in param_names or y_param not in param_names:
                    continue
                ix = param_names.index(x_param)
                iy = param_names.index(y_param)
                if iy > ix and (ix, iy) not in plotted_pairs:
                    ax = g.subplots[iy, ix]
                    ax.plot(x_val, y_val, "x", color="red", markersize=10)
                    __, labels_ = ax.get_legend_handles_labels()
                    if "Reference" not in labels_:
                        ax.legend()
                    plotted_pairs.add((ix, iy))

        if truths is not None:
            lw = 1.5 * plt.rcParams["lines.linewidth"]
            for truth_set in truths:
                truth_vals = truth_set["dict"]
                color = truth_set.get("color", "red")
                linestyle = truth_set.get("linestyle", "--")
                truth_label = truth_set.get("label", None)

                for i, param in enumerate(param_names):
                    if param in truth_vals:
                        ax = g.subplots[i, i]
                        ax.axvline(
                            truth_vals[param], color=color,
                            linestyle=linestyle, lw=lw, label=truth_label)

                for i, x_param in enumerate(param_names):
                    for j, y_param in enumerate(param_names):
                        if j > i:
                            ax = g.subplots[j, i]
                            if x_param in truth_vals:
                                ax.axvline(
                                    truth_vals[x_param], color=color,
                                    linestyle=linestyle, lw=lw)
                            if y_param in truth_vals:
                                ax.axhline(
                                    truth_vals[y_param], color=color,
                                    linestyle=linestyle, lw=lw)

        if filename is not None:
            fprint(f"[INFO] Saving GetDist triangle plot to: {filename}")
            g.export(filename, dpi=450)

        if show_fig:
            plt.show()
        else:
            plt.close()


def plot_corner_from_hdf5(fnames, keys=None, labels=None, cols=None,
                          fontsize=None, legend_fontsize=None, filled=True,
                          show_fig=True, filename=None, apply_ell_offset=False,
                          mag_range=None, ell_range=None,
                          b_range=None, points=None, ranges=None,
                          truths=None):
    """Plot a triangle plot from HDF5 files containing posterior samples."""
    if isinstance(fnames, (str, Path)):
        fnames = [fnames]

    samples_list = []
    for fname in fnames:
        with File(fname, 'r') as f:
            grp = f["samples"]
            samples = {key: grp[key][...] for key in grp.keys()}
            samples_list.append(samples)

            full_keys = list(grp.keys())
            print(f"{basename(fname)}: {', '.join(full_keys)}")

    plot_corner_getdist(
        samples_list,
        labels=labels,
        keys=keys,
        cols=cols,
        fontsize=fontsize,
        legend_fontsize=legend_fontsize,
        filled=filled,
        show_fig=show_fig,
        filename=filename,
        apply_ell_offset=apply_ell_offset,
        ranges=ranges,
        mag_range=mag_range,
        ell_range=ell_range,
        b_range=b_range,
        points=points,
        truths=truths,
    )
