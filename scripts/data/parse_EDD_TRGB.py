"""Parse EDD TRGB catalog into a clean CSV and make summary plots."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from candel.util import SPEED_OF_LIGHT

fdir = Path(__file__).resolve().parents[2] / "data" / "EDD_TRGB"
fin = fdir / "EDD_TRGB.txt"

with open(fin) as f:
    lines = f.readlines()

# Line 1: source categories, line 2: column names, line 3: format strings,
# line 4: units, line 5: descriptions, lines 6+: data
columns = lines[1].strip().split(",")
units = lines[3].strip().split(",")
formats = lines[2].strip().split(",")

# Determine which columns are numeric from the format strings
numeric_cols = set()
for i, fmt in enumerate(formats):
    fmt = fmt.strip()
    if "f" in fmt or "d" in fmt:
        numeric_cols.add(i)

rows = []
for line in lines[5:]:
    parts = line.strip().split(",")
    row = []
    for i, val in enumerate(parts):
        val = val.strip().strip('"')
        if val == "" or val == "---":
            row.append(np.nan if i in numeric_cols else "")
        elif i in numeric_cols:
            row.append(float(val))
        else:
            row.append(val)
    rows.append(row)

df = pd.DataFrame(rows, columns=columns)
print(f"Parsed {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")
print(f"Units:   {units[:10]}...")

# Extract galaxy names from the Name/CMD path
df["name"] = df["Name/CMD"].str.extract(r"LV/([^/]+)/")[0]

# Convert sexagesimal RA (HHMMSS.S) / DEC (±DDMMSS.S) to degrees
ra_str = df["RA2000"].str.replace(
    r"(\d{2})(\d{2})(\d+\.?\d*)", r"\1h\2m\3s", regex=True)
dec_str = df["DEC2000"].str.replace(
    r"([+-]?)(\d{2})(\d{2})(\d+\.?\d*)", r"\1\2d\3m\4s", regex=True)
coords = SkyCoord(ra_str, dec_str)
df["RA"] = coords.ra.deg
df["dec"] = coords.dec.deg

# Compute redshift from heliocentric velocity
df["z_helio"] = df["v"] / SPEED_OF_LIGHT

n_with_v = df["v"].notna().sum()
print(f"Have velocities for {n_with_v}/{len(df)} galaxies")

fout = fdir / "EDD_TRGB.csv"
df.to_csv(fout, index=False)
print(f"Saved to {fout}")

# --- Plots ---

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histogram of distances
dist = df["D814"].dropna()
axs[0, 0].hist(dist, bins=30, edgecolor="black", linewidth=0.5)
axs[0, 0].set_xlabel(r"$D_{814}$ [Mpc]")
axs[0, 0].set_ylabel("Count")

# 2. Distance vs apparent TRGB magnitude
mask = df["D814"].notna() & df["T814"].notna()
axs[0, 1].scatter(df.loc[mask, "D814"], df.loc[mask, "T814"], s=8, alpha=0.7)
axs[0, 1].set_xlabel(r"$D_{814}$ [Mpc]")
axs[0, 1].set_ylabel(r"$m_{814}^{\rm TRGB}$ [mag]")

# 3. Histogram of heliocentric velocities
vhel = df["v"].dropna()
axs[1, 0].hist(vhel, bins=30, edgecolor="black", linewidth=0.5)
axs[1, 0].set_xlabel(r"$v_{\rm helio}$ [km/s]")
axs[1, 0].set_ylabel("Count")

# 4. Apparent TRGB magnitude vs heliocentric velocity
mask = df["v"].notna() & df["T814"].notna()
axs[1, 1].scatter(df.loc[mask, "v"], df.loc[mask, "T814"], s=8, alpha=0.7)
axs[1, 1].set_xlabel(r"$v_{\rm helio}$ [km/s]")
axs[1, 1].set_ylabel(r"$m_{814}^{\rm TRGB}$ [mag]")

fig.tight_layout()
fplot = Path(__file__).resolve().parent / "EDD_TRGB_summary.png"
fig.savefig(fplot, dpi=200, bbox_inches="tight")
print(f"Saved plot to {fplot}")
plt.close(fig)
