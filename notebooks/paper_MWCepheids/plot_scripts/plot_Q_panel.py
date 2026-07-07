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
"""Plot Q (reddening-free index) against mW, logP and [O/H]."""
from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401

plt.style.use(['science', 'no-latex'])

repo_root = abspath(join(dirname(abspath(__file__)), "..", "..", ".."))
data_dir = join(repo_root, "data", "MWCepheids")
df = pd.read_csv(join(data_dir, "Riess2021_Table1_with_coords.csv"))
df["OH"] = df["FeH"] + 0.06

c22 = df[df["set"] == "Cycle22"]
c27 = df[df["set"] == "Cycle27"]

outliers = ["CP-CEP", "DR-VEL", "GQ-ORI"]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

xkeys = ["mW_H", "logP", "OH"]
xerr_keys = ["mW_H_err", None, None]
xlabels = [r"$m_W^H$ [mag]", r"$\log P$ [days]", r"$[\mathrm{O/H}]$ [dex]"]

for ax, xk, xek, xlab in zip(axes, xkeys, xerr_keys, xlabels):
    for subset, marker, col, label in [
        (c22, 'o', '#3c91e6', 'C22'),
        (c27, 's', '#c52233', 'C27'),
    ]:
        xerr = subset[xek] if xek else None
        ax.errorbar(subset[xk], subset["Q"], xerr=xerr, yerr=subset["Q_err"],
                    fmt=marker, ms=3, color=col, elinewidth=0.6, capsize=0,
                    alpha=0.85, label=label)

    for _, row in df[df["Cepheid"].isin(outliers)].iterrows():
        xval = row[xk] if xk != "OH" else row["FeH"] + 0.06
        ax.annotate(row["Cepheid"], (xval, row["Q"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=5.5)

    ax.set_xlabel(xlab)

axes[0].set_ylabel(r"$Q$")
axes[2].legend(frameon=True)

fig.tight_layout()
fig.savefig(join(data_dir, "..", "Q_panel.png"), dpi=300)
print("Saved Q_panel.png")
plt.show()
