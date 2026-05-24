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
"""Generic plotting helpers for likelihood/selection diagnostics."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_FIELD_COLOURS = [
    "#ef476f",
    "#473198",
    "#a8c256",
    "#5adbff",
    "#fe9000",
]


def finite_corr(x, y):
    """Return the Pearson correlation of finite entries, or NaN."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 3:
        return np.nan
    if np.nanstd(x[finite]) == 0.0 or np.nanstd(y[finite]) == 0.0:
        return np.nan
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def field_cmap(name="manticore_fields", colours=None):
    """Return a continuous colormap for field-index diagnostics."""
    if colours is None:
        colours = DEFAULT_FIELD_COLOURS
    return LinearSegmentedColormap.from_list(name, colours)


def _as_1d(name, values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional.")
    return arr


def _finite_argmax(values):
    values = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return None
    return int(finite[np.argmax(values[finite])])


def plot_raw_selection_evidence(
        raw_likelihood,
        selection_term,
        evidence,
        field_ids=None,
        *,
        total_likelihood=None,
        raw_label="raw total likelihood",
        selection_label=r"selection normalisation, $-\log S$",
        evidence_label=r"harmonic $\ln Z$",
        field_label="Manticore field",
        selection_title="Selection term",
        evidence_title=None,
        best_evidence_label="best lnZ",
        max_selection_label="max -logS",
        show_constant_total=True,
        constant_total_quantiles=(20.0, 50.0, 80.0),
        figsize=(7.2, 3.35),
        cmap=None,
        norm=None):
    """Plot raw likelihood, selection term, and evidence for matched fields.

    The caller supplies already-reduced arrays.  This helper does not know
    about HDF5 files, model classes, or paper-specific result directories.
    """
    raw = _as_1d("raw_likelihood", raw_likelihood)
    selection = _as_1d("selection_term", selection_term)
    evidence = _as_1d("evidence", evidence)
    if not (raw.size == selection.size == evidence.size):
        raise ValueError("`raw_likelihood`, `selection_term`, and "
                         "`evidence` must have the same length.")
    if raw.size == 0:
        raise ValueError("Input arrays must not be empty.")

    if field_ids is None:
        fields = np.arange(raw.size, dtype=float)
    else:
        fields = _as_1d("field_ids", field_ids)
        if fields.size != raw.size:
            raise ValueError("`field_ids` must match the data length.")

    if total_likelihood is None:
        constant_levels = raw + selection
        constant_total_label = "dashed: constant raw + selection"
    else:
        total = _as_1d("total_likelihood", total_likelihood)
        if total.size != raw.size:
            raise ValueError("`total_likelihood` must match the data length.")
        residual = total - raw - selection
        if np.nanstd(residual) < 1e-8:
            constant_levels = total
            constant_total_label = "dashed: constant total"
        else:
            constant_levels = raw + selection
            constant_total_label = "dashed: constant raw + selection"

    if cmap is None:
        cmap = field_cmap()
    if norm is None:
        finite_fields = fields[np.isfinite(fields)]
        if finite_fields.size == 0:
            field_min, field_max = 0.0, 1.0
        else:
            field_min = float(np.min(finite_fields))
            field_max = float(np.max(finite_fields))
        if field_min == field_max:
            field_max = field_min + 1.0
        norm = plt.Normalize(vmin=field_min, vmax=field_max)

    best_evidence_idx = _finite_argmax(evidence)
    max_selection_idx = _finite_argmax(selection)

    with plt.rc_context({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
    }):
        fig, axes = plt.subplots(
            1, 2, figsize=figsize, sharex=True, constrained_layout=True)
        ax_sel, ax_evidence = axes

        scatter_kw = {
            "c": fields,
            "cmap": cmap,
            "norm": norm,
            "s": 32,
            "alpha": 0.88,
            "edgecolor": "0.15",
            "linewidth": 0.3,
        }
        sc = ax_sel.scatter(raw, selection, **scatter_kw)
        ax_evidence.scatter(raw, evidence, **scatter_kw)

        if show_constant_total:
            xlim = ax_sel.get_xlim()
            ylim = ax_sel.get_ylim()
            xgrid = np.linspace(xlim[0], xlim[1], 200)
            for level in np.nanpercentile(
                    constant_levels, constant_total_quantiles):
                ax_sel.plot(
                    xgrid, level - xgrid, color="0.55", lw=0.65,
                    ls="--", alpha=0.55)
            ax_sel.set_xlim(xlim)
            ax_sel.set_ylim(ylim)
            ax_sel.text(
                0.03, 0.04, constant_total_label,
                transform=ax_sel.transAxes, ha="left", va="bottom",
                fontsize=6.6, color="0.35")

        highlights = [
            (best_evidence_idx, "o", best_evidence_label, (4, 4)),
            (max_selection_idx, "s", max_selection_label, (4, -11)),
        ]
        for idx, marker, label, offset in highlights:
            if idx is None:
                continue
            field = f"{fields[idx]:.0f}"
            ax_sel.scatter(
                raw[idx], selection[idx], s=74, marker=marker,
                facecolor="none", edgecolor="black", linewidth=1.0,
                zorder=4)
            ax_sel.annotate(
                f"{label}: field {field}", (raw[idx], selection[idx]),
                xytext=offset, textcoords="offset points", fontsize=7.0,
                color="black")
            ax_evidence.scatter(
                raw[idx], evidence[idx], s=74, marker=marker,
                facecolor="none", edgecolor="black", linewidth=1.0,
                zorder=4)
            ax_evidence.annotate(
                field, (raw[idx], evidence[idx]), xytext=(4, 4),
                textcoords="offset points", fontsize=7.0, color="black")

        ax_sel.set_xlabel(raw_label)
        ax_sel.set_ylabel(selection_label)
        ax_sel.set_title(selection_title, loc="left")

        if evidence_title is None:
            evidence_title = rf"Evidence; $r={finite_corr(raw, evidence):.2f}$"
        ax_evidence.set_xlabel(raw_label)
        ax_evidence.set_ylabel(evidence_label)
        ax_evidence.set_title(evidence_title, loc="left")

        cbar = fig.colorbar(
            sc, ax=axes.ravel().tolist(), pad=0.012, fraction=0.055)
        cbar.set_label(field_label)
        cbar.ax.tick_params(labelsize=7.0)

    return fig, axes
