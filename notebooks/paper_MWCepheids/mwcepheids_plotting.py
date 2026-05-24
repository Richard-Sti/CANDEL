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
"""MW-Cepheid plotting utilities."""
from os.path import basename

import h5py
import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from getdist import MCSamples, plots
from scipy.stats import ks_2samp

try:
    import arviz as az
except ImportError:
    az = None

try:
    import scienceplots  # noqa
    _PLOT_STYLE = ['science', 'no-latex']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    _PLOT_STYLE = 'default'

_LABELS = {
    "M_H_1": r"$M^W_{H,1}$",
    "b_W": r"$b_W$",
    "Z_W": r"$Z_W$",
    "sigma_int": r"$\sigma_{\rm int}$",
    "sigma_int_C22": r"$\sigma_{\rm int}^{\rm C22}$",
    "sigma_int_C27": r"$\sigma_{\rm int}^{\rm C27}$",
    "delta_pi": r"$\delta_\varpi$",
    "pi_min_C27": r"$\varpi_{\rm min}^{\rm C27}$",
    "mW_min_C27": r"$m^W_{\rm min,C27}$",
    "mW_max_C22": r"$m^W_{\rm max,C22}$",
    "mW_width_C22": r"$\sigma^W_{\rm sel}$",
    "AH_max_C22": r"$A_{H,\rm max}^{\rm C22}$",
    "pi_min_C22": r"$\pi_{\rm min}^{\rm C22}$",
    "logP_min_C22": r"$\log P_{\rm min}^{\rm C22}$",
    "mu_NGC4258": r"$\mu_{\rm NGC4258}$",
    "mu_LMC": r"$\mu_{\rm LMC}$",
    "spiral_arm_frac": r"$f_{\rm spiral}$",
    "spiral_width": r"$\sigma_{\rm spiral}$",
    "mu_logP_C22": r"$\mu_{\log P}^{\rm C22}$",
    "mu_logP_C27": r"$\mu_{\log P}^{\rm C27}$",
    "mu_logP_LMC": r"$\mu_{\log P}^{\rm LMC}$",
    "mu_logP_NGC4258": r"$\mu_{\log P}^{\rm N4258}$",
    "sigma_logP_C22": r"$\sigma_{\log P}^{\rm C22}$",
    "sigma_logP_C27": r"$\sigma_{\log P}^{\rm C27}$",
    "sigma_logP_LMC": r"$\sigma_{\log P}^{\rm LMC}$",
    "sigma_logP_NGC4258": r"$\sigma_{\log P}^{\rm N4258}$",
    "mu_OH_C22": r"$\mu_{[{\rm O/H}]}^{\rm C22}$",
    "mu_OH_C27": r"$\mu_{[{\rm O/H}]}^{\rm C27}$",
    "mu_OH_LMC": r"$\mu_{[{\rm O/H}]}^{\rm LMC}$",
    "mu_OH_NGC4258": r"$\mu_{[{\rm O/H}]}^{\rm N4258}$",
    "sigma_OH_C22": r"$\sigma_{[{\rm O/H}]}^{\rm C22}$",
    "sigma_OH_C27": r"$\sigma_{[{\rm O/H}]}^{\rm C27}$",
    "sigma_OH_LMC": r"$\sigma_{[{\rm O/H}]}^{\rm LMC}$",
    "sigma_OH_NGC4258": r"$\sigma_{[{\rm O/H}]}^{\rm N4258}$",
}

# GetDist labels (no $..$ wrapping — GetDist adds them automatically)
_GETDIST_LABELS = {
    "M_H_1": r"M^W_{H,1}",
    "b_W": r"b_W",
    "Z_W": r"Z_W",
    "sigma_int": r"\sigma_{\rm int}",
    "sigma_int_C22": r"\sigma_{\rm int}^{\rm C22}",
    "sigma_int_C27": r"\sigma_{\rm int}^{\rm C27}",
    "delta_pi": r"\delta_\varpi",
    "pi_min_C27": r"\varpi_{\rm min}^{\rm C27}",
    "mW_min_C27": r"m^W_{\rm min,C27}",
    "mW_max_C22": r"m^W_{\rm max,C22}",
    "mW_width_C22": r"\sigma^W_{\rm sel}",
    "AH_max_C22": r"A_{H,\rm max}^{\rm C22}",
    "pi_min_C22": r"\pi_{\rm min}^{\rm C22}",
    "logP_min_C22": r"\log P_{\rm min}^{\rm C22}",
    "mu_NGC4258": r"\mu_{\rm NGC4258}",
    "mu_LMC": r"\mu_{\rm LMC}",
    "spiral_arm_frac": r"f_{\rm spiral}",
    "spiral_width": r"\sigma_{\rm spiral}",
    "mu_logP_C22": r"\mu_{\log P}^{\rm C22}",
    "mu_logP_C27": r"\mu_{\log P}^{\rm C27}",
    "mu_logP_LMC": r"\mu_{\log P}^{\rm LMC}",
    "mu_logP_NGC4258": r"\mu_{\log P}^{\rm N4258}",
    "sigma_logP_C22": r"\sigma_{\log P}^{\rm C22}",
    "sigma_logP_C27": r"\sigma_{\log P}^{\rm C27}",
    "sigma_logP_LMC": r"\sigma_{\log P}^{\rm LMC}",
    "sigma_logP_NGC4258": r"\sigma_{\log P}^{\rm N4258}",
    "mu_OH_C22": r"\mu_{[{\rm O/H}]}^{\rm C22}",
    "mu_OH_C27": r"\mu_{[{\rm O/H}]}^{\rm C27}",
    "mu_OH_LMC": r"\mu_{[{\rm O/H}]}^{\rm LMC}",
    "mu_OH_NGC4258": r"\mu_{[{\rm O/H}]}^{\rm N4258}",
    "sigma_OH_C22": r"\sigma_{[{\rm O/H}]}^{\rm C22}",
    "sigma_OH_C27": r"\sigma_{[{\rm O/H}]}^{\rm C27}",
    "sigma_OH_LMC": r"\sigma_{[{\rm O/H}]}^{\rm LMC}",
    "sigma_OH_NGC4258": r"\sigma_{[{\rm O/H}]}^{\rm N4258}",
    "c_W": r"c_W",
    "sigma_int_anchor": r"\sigma_{\rm int}^{\rm anc}",
}


def _is_varying(x):
    """Check if array contains varying (non-constant) values."""
    x = np.asarray(x).ravel()
    return not np.allclose(x, x[0])


def print_summary(mcmc, prob=0.9):
    """Print MCMC summary with 3 decimal places.

    Similar to numpyro's mcmc.print_summary() but with 3 decimal places.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object.
    prob : float, optional
        Probability mass for credible interval (default 0.9).
    """
    from numpyro.diagnostics import summary

    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    samples_flat = mcmc.get_samples(group_by_chain=False)
    stats = summary(samples_by_chain, prob=prob)

    # Header
    lo = (1 - prob) / 2 * 100
    hi = (1 + prob) / 2 * 100
    print()
    print(f"{'':>15} {'mean':>10} {'std':>10} {'median':>10} "
          f"{f'{lo:.1f}%':>10} {f'{hi:.1f}%':>10} {'n_eff':>10} {'r_hat':>10}")
    print("-" * 95)

    for name, values in stats.items():
        # Skip array parameters (like per-star distances)
        if values["mean"].ndim > 0:
            continue

        # Skip deterministic (constant) parameters
        if not _is_varying(samples_flat[name]):
            continue

        mean = values["mean"]
        std = values["std"]
        median = values["median"]
        ci_lo = values[f"{lo:.1f}%"]
        ci_hi = values[f"{hi:.1f}%"]
        n_eff = values["n_eff"]
        r_hat = values["r_hat"]

        print(f"{name:>15} {mean:>10.3f} {std:>10.3f} {median:>10.3f} "
              f"{ci_lo:>10.3f} {ci_hi:>10.3f} {n_eff:>10.1f} {r_hat:>10.3f}")

    print()
    num_samples = mcmc.num_samples
    num_chains = mcmc.num_chains
    print(f"Number of samples: {num_samples} × {num_chains} chains")

    # Report divergences
    extra_fields = mcmc.get_extra_fields()
    if "diverging" in extra_fields:
        n_divergent = int(extra_fields["diverging"].sum())
        n_total = num_samples * num_chains
        pct = 100 * n_divergent / n_total
        if n_divergent > 0:
            print(f"WARNING: {n_divergent} divergent transitions ({pct:.1f}%)")
        else:
            print("No divergent transitions")


def plot_trace(mcmc, keys=None, exclude=None, filename=None):
    """Plot trace plots for selected parameters.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object.
    keys : list of str, optional
        Parameter names to plot. Defaults to global parameters.
    exclude : list of str, optional
        Parameter names to exclude.
    filename : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    axes : array of Axes
    """
    if az is None:
        raise ImportError("`arviz` is required for plot_trace.")
    samples = mcmc.get_samples()
    if keys is None:
        keys = list(samples.keys())
    if exclude is not None:
        keys = [k for k in keys if k not in exclude]

    # Filter out deterministic (constant) parameters
    keys = [k for k in keys if _is_varying(samples[k])]

    with plt.style.context(_PLOT_STYLE):
        idata = az.from_numpyro(mcmc)
        labeller = az.labels.MapLabeller(var_name_map=_LABELS)
        axes = az.plot_trace(
            idata,
            var_names=keys,
            labeller=labeller,
            compact=True,
        )

        fig = axes.ravel()[0].get_figure()
        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")

        plt.close(fig)
        return fig


def plot_distance_comparison(samples, data, filename=None):
    """Plot posterior distance vs observed distance from parallax.

    Parameters
    ----------
    samples : dict
        Posterior samples from MCMC (must contain "d" and "delta_pi").
    data : CepheidData
        Loaded Cepheid data.
    filename : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    fig : Figure
    """
    # Find distance samples (d, d_C22, or d_C27)
    if "d" in samples:
        d_key = "d"
    elif f"d_{data.campaign}" in samples:
        d_key = f"d_{data.campaign}"
    else:
        raise ValueError("No distance samples found. This plot requires the "
                         "forward model (model_type = 'forward').")

    # Posterior distances from MCMC
    d_samples = np.asarray(samples[d_key])
    d_median = np.median(d_samples, axis=0)
    d_16, d_84 = np.percentile(d_samples, [16, 84], axis=0)

    # Observed distance from parallax (using median delta_pi)
    # Model: pi_EDR3 + delta_pi = 1/d, so d = 1/(pi_EDR3 + delta_pi)
    delta_pi_median = np.median(samples["delta_pi"])
    pi_corrected = np.asarray(data.pi_EDR3) + delta_pi_median
    d_obs = 1.0 / pi_corrected

    # Error from parallax measurement uncertainty: d_err = pi_err / pi^2
    pi_err = np.asarray(data.pi_EDR3_err)
    d_obs_err = pi_err / pi_corrected**2

    with plt.style.context(_PLOT_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1],
                                              "hspace": 0.05})

        # Top panel: d_posterior vs d_obs
        ax = axes[0]
        ax.errorbar(
            d_obs, d_median,
            xerr=d_obs_err,
            yerr=[d_median - d_16, d_84 - d_median],
            fmt="o", ms=3, lw=0.8, alpha=0.7,
        )
        lim = [min(np.min(d_obs - d_obs_err), np.min(d_16)) * 0.9,
               max(np.max(d_obs + d_obs_err), np.max(d_84)) * 1.1]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$d$ posterior median [kpc]")

        # Bottom panel: ratio
        ax = axes[1]
        ratio = d_median / d_obs
        ratio_err_lo = (d_median - d_16) / d_obs
        ratio_err_hi = (d_84 - d_median) / d_obs
        ax.errorbar(
            d_obs, ratio,
            xerr=d_obs_err,
            yerr=[ratio_err_lo, ratio_err_hi],
            fmt="o", ms=3, lw=0.8, alpha=0.7,
        )
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"$1/(\pi_{\rm EDR3} + \Delta_\pi)$ [kpc]")
        ax.set_ylabel(r"$d_{\rm post} / d_{\rm obs}$")

        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")

        plt.close(fig)
        return fig


def plot_corner(mcmc, keys=None, exclude=None, filename=None, smooth=1):
    """Plot a corner plot for selected parameters.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object.
    keys : list of str, optional
        Parameter names to plot. Defaults to global parameters.
    exclude : list of str, optional
        Parameter names to exclude.
    filename : str, optional
        If provided, save the figure to this path.
    smooth : float, optional
        Smoothing scale for the corner plot.

    Returns
    -------
    fig : Figure
    """
    samples = mcmc.get_samples(group_by_chain=False)
    if keys is None:
        keys = list(samples.keys())
    if exclude is not None:
        keys = [k for k in keys if k not in exclude]

    # Filter out array parameters (e.g. per-star distances) and constants
    keys = [k for k in keys if np.asarray(samples[k]).ndim == 1]
    keys = [k for k in keys if _is_varying(samples[k])]

    arr = np.column_stack([np.asarray(samples[k]) for k in keys])
    labels = [_LABELS.get(k, k) for k in keys]

    with plt.style.context(_PLOT_STYLE):
        fig = corner(
            arr,
            labels=labels,
            show_titles=True,
            smooth=smooth,
        )

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")

        plt.close(fig)
        return fig


def plot_corner_getdist(samples_list, labels=None, cols=None, show_fig=True,
                        filename=None, keys=None, fontsize=None,
                        legend_fontsize=None, filled=True, ranges=None,
                        truths=None, linestyles=None, scales=None,
                        fig_width_inch=None, alphas=None):
    """Plot a GetDist triangle plot for one or more posterior sample dicts.

    Parameters
    ----------
    samples_list : dict or list of dict
        Posterior samples. Each dict maps parameter names to 1D arrays.
    labels : list of str, optional
        Legend labels for each sample set.
    cols : list of str, optional
        Colours for each sample set.
    show_fig : bool
        Whether to call plt.show().
    filename : str, optional
        If provided, save the figure to this path.
    keys : list of str, optional
        Parameter names to plot. Defaults to all 1D, varying parameters.
    fontsize : int, optional
        Font size for axis labels.
    legend_fontsize : int, optional
        Font size for legend.
    filled : bool
        Whether to fill contours.
    ranges : dict, optional
        Parameter ranges for GetDist.
    truths : list of dict, optional
        Each entry has ``"dict"`` (param->value), ``"color"``, ``"linestyle"``.
    """
    if ranges is None:
        ranges = {}
    if scales is None:
        scales = {}

    # Build scaled label overrides
    _SCALE_LABELS = {
        "delta_pi": {1000: r"\delta_\varpi \; [\mu{\rm as}]"},
    }

    if isinstance(samples_list, dict):
        samples_list = [samples_list]

    if labels is not None and len(labels) != len(samples_list):
        raise ValueError(
            "Length of `labels` must match number of sample sets")

    # Build candidate key list
    if keys is not None:
        candidate_keys = keys
    else:
        candidate_keys = [
            k for k in samples_list[0]
            if np.asarray(samples_list[0][k]).ndim == 1]

    # Keep keys that are 1D and varying in at least one sample set
    param_names = []
    for k in candidate_keys:
        for s in samples_list:
            arr = np.asarray(s.get(k, np.array([])))
            if arr.ndim == 1 and len(arr) > 0 and _is_varying(arr):
                param_names.append(k)
                break

    gdsamples_list = []
    for samples in samples_list:
        present_params = []
        present_labels = []
        columns = []

        n_samples = len(next(iter(samples.values())))
        for k in param_names:
            arr = np.asarray(samples.get(k, np.array([])))
            if arr.ndim == 1 and len(arr) == n_samples:
                col = arr * scales.get(k, 1.0)
            else:
                col = np.full(n_samples, np.nan)

            if not np.all(np.isnan(col)):
                label = _GETDIST_LABELS.get(k, k)
                if k in scales and k in _SCALE_LABELS:
                    label = _SCALE_LABELS[k].get(scales[k], label)
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

    # Plot styling
    settings = plots.GetDistPlotSettings()
    settings.lw1 = settings.lw1 * 2.0
    if fontsize is not None:
        settings.lab_fontsize = fontsize
        settings.legend_fontsize = (
            legend_fontsize if legend_fontsize is not None else fontsize)
        settings.axes_fontsize = fontsize - 1
        settings.title_limit_fontsize = fontsize - 1

    n = len(gdsamples_list)
    if cols is not None or linestyles is not None:
        _cols = cols if cols is not None else [None] * n
        _ls = linestyles if linestyles is not None else ["-"] * n
        line_args = [{"color": c, "ls": ls} for c, ls in zip(_cols, _ls)]
    else:
        line_args = None

    with plt.style.context(_PLOT_STYLE):
        kw = {"settings": settings}
        if fig_width_inch is not None:
            kw["width_inch"] = fig_width_inch
        g = plots.get_subplot_plotter(**kw)
        tri_kw = dict(
            params=param_names,
            filled=filled,
            colors=cols,
            contour_colors=cols,
            line_args=line_args,
            legend_labels=labels,
            legend_loc="upper right",
        )
        if alphas is not None:
            tri_kw["alphas"] = alphas
        g.triangle_plot(gdsamples_list, **tri_kw)

        if truths is not None:
            from matplotlib.lines import Line2D
            lw = settings.lw1 * 0.5
            truth_handles = []
            for truth_set in truths:
                tvals = truth_set["dict"]
                color = truth_set.get("color", "red")
                linestyle = truth_set.get("linestyle", "--")
                label = truth_set.get("label", None)

                scaled_tvals = {
                    k: v * scales.get(k, 1.0) for k, v in tvals.items()}

                for i, param in enumerate(param_names):
                    if param in scaled_tvals:
                        g.subplots[i, i].axvline(
                            scaled_tvals[param], color=color,
                            linestyle=linestyle, lw=lw)

                for i, x_param in enumerate(param_names):
                    for j, y_param in enumerate(param_names):
                        if j > i:
                            ax = g.subplots[j, i]
                            if x_param in scaled_tvals:
                                ax.axvline(
                                    scaled_tvals[x_param], color=color,
                                    linestyle=linestyle, lw=lw)
                            if y_param in scaled_tvals:
                                ax.axhline(
                                    scaled_tvals[y_param], color=color,
                                    linestyle=linestyle, lw=lw)

                if label is not None:
                    truth_handles.append(
                        Line2D([], [], color=color, ls=linestyle,
                               lw=lw, label=label))

            # Append truth entries to the existing GetDist legend
            if truth_handles:
                leg = None
                if g.fig.legends:
                    leg = g.fig.legends[-1]
                elif hasattr(g, 'legend'):
                    leg = g.legend

                if leg is not None:
                    handles = leg.legend_handles + truth_handles
                    labels_ = ([t.get_text() for t in leg.get_texts()]
                               + [h.get_label() for h in truth_handles])
                    leg.remove()
                g.fig.legend(
                    handles if leg is not None else truth_handles,
                    labels_ if leg is not None else
                    [h.get_label() for h in truth_handles],
                    fontsize=settings.legend_fontsize,
                    loc="upper right",
                    frameon=False)

        if filename is not None:
            g.export(filename, dpi=500)

        if show_fig:
            plt.show()
        else:
            plt.close()


def plot_corner_from_hdf5(fnames, keys=None, labels=None, cols=None,
                          fontsize=None, legend_fontsize=None, filled=True,
                          show_fig=True, filename=None, ranges=None,
                          truths=None, linestyles=None, scales=None,
                          fig_width_inch=None, alphas=None):
    """Plot a GetDist triangle plot from one or more HDF5 sample files.

    Parameters
    ----------
    fnames : str or list of str
        Path(s) to HDF5 files containing posterior samples.
    keys : list of str, optional
        Parameter names to plot.
    labels : list of str, optional
        Legend labels for each file.
    cols : list of str, optional
        Colours for each file.
    fontsize : int, optional
        Font size for axis labels.
    legend_fontsize : int, optional
        Font size for legend.
    filled : bool
        Whether to fill contours.
    show_fig : bool
        Whether to call plt.show().
    filename : str, optional
        If provided, save the figure to this path.
    ranges : dict, optional
        Parameter ranges for GetDist.
    truths : list of dict, optional
        Truth values to overplot.
    """
    if isinstance(fnames, str):
        fnames = [fnames]

    samples_list = []
    for fname in fnames:
        with h5py.File(fname, "r") as f:
            # New format: samples in "samples" group; old: at top level
            grp = f["samples"] if "samples" in f else f
            samples = {key: grp[key][...] for key in grp.keys()}
            samples_list.append(samples)
            print(f"{basename(fname)}: {', '.join(grp.keys())}")

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
        ranges=ranges,
        truths=truths,
        linestyles=linestyles,
        scales=scales,
        fig_width_inch=fig_width_inch,
        alphas=alphas,
    )


def plot_ppc(ppc_results, labels=None, cols=None, histtypes=None,
             show_ks=None, show_scatter=None, pi_ymax=None,
             mW_xlim=None, pi_xlim=None, logP_xlim=None,
             filename=None,
             scatter_size=1, scatter_alpha=0.2, ncols=4, figsize=None):
    """Plot posterior predictive checks for one or more MW-Cepheid runs."""
    if isinstance(ppc_results, dict):
        ppc_results = [ppc_results]
    n_runs = len(ppc_results)

    if labels is None:
        labels = [f"PPC {i}" for i in range(n_runs)]
    elif isinstance(labels, str):
        labels = [labels]

    default_cols = ["#c52233", "#3c91e6", "#2e8b57", "#f2a359", "#0e0004"]
    if cols is None:
        cols = default_cols[:n_runs]
    elif isinstance(cols, str):
        cols = [cols]

    if histtypes is None:
        histtypes = ["bar"] * n_runs
    elif isinstance(histtypes, str):
        histtypes = [histtypes] * n_runs

    if show_ks is None:
        show_ks = list(range(n_runs))
    if show_scatter is None:
        show_scatter = list(range(n_runs))

    mW_obs = ppc_results[0]["mW_obs"]
    pi_obs = ppc_results[0]["pi_obs"]
    logP_obs = ppc_results[0]["logP_obs"]

    nrows = int(np.ceil(4 / ncols))
    if figsize is None:
        figsize = (4.5 * ncols, 4 * nrows)

    with plt.style.context(_PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.ravel()

        obs_kw = dict(bins="auto", density=True, alpha=0.9,
                      histtype="step", lw=1.2, color="k",
                      label=r"$\mathrm{Observed}$")

        ks_results = []
        for i, (ppc, lab, col, ht) in enumerate(
                zip(ppc_results, labels, cols, histtypes)):
            mW_sim = ppc["mW_sim"]
            pi_sim = ppc["pi_sim"]
            logP_sim = ppc["logP_sim"]

            ks_mW = ks_2samp(mW_obs, mW_sim)
            ks_pi = ks_2samp(pi_obs, pi_sim)
            ks_logP = ks_2samp(logP_obs, logP_sim)
            ks_results.append((ks_mW.pvalue, ks_pi.pvalue, ks_logP.pvalue))

            hist_kw = dict(bins="auto", density=True, alpha=0.8,
                           color=col, label=lab, histtype=ht)
            if ht == "step":
                hist_kw["lw"] = 1.5

            axes[0].hist(mW_sim, **hist_kw)
            axes[1].hist(pi_sim, **hist_kw)
            axes[2].hist(logP_sim, **hist_kw)
            if i in show_scatter:
                axes[3].scatter(pi_sim, mW_sim, s=scatter_size,
                                alpha=scatter_alpha, color=col,
                                rasterized=True, label=lab)

            print(f"  {lab}: KS(mW) p={ks_mW.pvalue:.4f}, "
                  f"KS(pi) p={ks_pi.pvalue:.4f}, "
                  f"KS(logP) p={ks_logP.pvalue:.4f}")

        ks_fs = 9
        ks_count = 0
        for j, (p_mW, p_pi, p_logP) in enumerate(ks_results):
            if j not in show_ks:
                continue
            dy = ks_count * 0.08
            c = cols[j]
            axes[0].text(0.97, 0.92 - dy, f"$p_{{\\rm KS}} = {p_mW:.3f}$",
                         transform=axes[0].transAxes, fontsize=ks_fs,
                         color=c, va="top", ha="right")
            axes[1].text(0.97, 0.92 - dy, f"$p_{{\\rm KS}} = {p_pi:.3f}$",
                         transform=axes[1].transAxes, fontsize=ks_fs,
                         color=c, va="top", ha="right")
            axes[2].text(0.97, 0.92 - dy, f"$p_{{\\rm KS}} = {p_logP:.3f}$",
                         transform=axes[2].transAxes, fontsize=ks_fs,
                         color=c, va="top", ha="right")
            ks_count += 1

        axes[0].hist(mW_obs, **obs_kw)
        axes[1].hist(pi_obs, **obs_kw)
        axes[2].hist(logP_obs, **obs_kw)
        axes[3].scatter(
            pi_obs, mW_obs, s=10, alpha=0.9, edgecolor="k",
            lw=0.2, color="k", zorder=10,
            label=r"$\mathrm{Observed}$")

        axes[0].set_xlabel(r"$m^W_H \; [{\rm mag}]$")
        axes[0].set_ylabel(r"$\mathrm{Normalised \; counts}$")
        axes[1].set_xlabel(r"$\varpi \; [{\rm mas}]$")
        axes[1].set_ylabel(r"$\mathrm{Normalised \; counts}$")
        if pi_ymax is not None:
            axes[1].set_ylim(top=pi_ymax)
        axes[2].set_xlabel(r"$\log P / (1 \; {\rm day})$")
        axes[2].set_ylabel(r"$\mathrm{Normalised \; counts}$")
        axes[3].set_xlabel(r"$\varpi \; [{\rm mas}]$")
        axes[3].set_ylabel(r"$m^W_H \; [{\rm mag}]$")

        if mW_xlim is not None:
            axes[0].set_xlim(*mW_xlim)
            axes[3].set_ylim(*mW_xlim)
        if pi_xlim is not None:
            axes[1].set_xlim(*pi_xlim)
            axes[3].set_xlim(*pi_xlim)
        if logP_xlim is not None:
            axes[2].set_xlim(*logP_xlim)

        handles, leg_labels = axes[0].get_legend_handles_labels()
        handles = [handles[-1]] + handles[:-1]
        leg_labels = [leg_labels[-1]] + leg_labels[:-1]
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)
        fig.legend(handles, leg_labels, fontsize=10, ncol=len(handles),
                   loc="upper center", frameon=False)

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")

        plt.close(fig)
        return fig
