# Copyright (C) 2025 Richard Stiskalek
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


import random


def simname2label(simname):
    ltx = {"Carrick2015": "Carrick+15",
           "Lilow2024": "Lilow+24",
           "CB1": r"\texttt{CSiBORG}1",
           "CB2": r"\texttt{CSiBORG}2",
           "manticore_2MPP_MULTIBIN_N256_DES_V2": r"\texttt{Manticore-Local}",
           "CF4": "Courtois+23",
           "CLONES": "Sorce+2018",
           }

    if isinstance(simname, list):
        names = [ltx[s] if s in ltx else s for s in simname]
        return "".join([f"{n}, " for n in names]).rstrip(", ")

    return ltx[simname] if simname in ltx else simname


def catalogue2label(catalogue):
    ltx = {"SFI_gals": r"SFI\texttt{++}",
           "CF4_TFR_not2MTForSFI_i": r"CF4 $i$-band",
           "CF4_TFR_i": r"CF4 TFR $i$",
           "CF4_TFR_w1": r"CF4 TFR W1",
           "CF4_TFR_w2": r"CF4 TFR W2",
           }

    if isinstance(catalogue, list):
        names = [ltx[s] if s in ltx else s for s in catalogue]
        return "".join([f"{n}, " for n in names]).rstrip(", ")

    return ltx[catalogue] if catalogue in ltx else catalogue


def simname2color(simname):
    cols = ["#1be7ffff", "#6eeb83ff", "#e4ff1aff", "#ffb800ff",
            "#ff5714ff", "#9b5de5ff"]

    defaults = ["tab:blue", "tab:orange", "tab:green",
                "tab:red", "tab:purple", "tab:brown",
                "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    colors = {
        "Carrick2015": cols[0],
        "Lilow2024": cols[1],
        "CB1": cols[2],
        "CB2": cols[3],
        "CF4": cols[4],
        "CLONES": cols[5],
    }

    return colors.get(simname, random.choice(defaults))
