"""
Plot the phi integrand shape at several r_ang values for a representative
spot, showing why uniform phi grids fail at wrong r_ang.
"""
import numpy as np
import jax
import jax.numpy as jnp
import tempfile
import os
import tomli
import tomli_w
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from candel.model.model_H0_maser import (
    MaserDiskModel, C_v, C_a,
)
from candel.model.integration import trapz_log_weights
from candel.pvdata.megamaser_data import load_megamaser_spots

jax.config.update("jax_platform_name", "gpu")

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
with open(CONFIG_PATH, "rb") as f:
    master_cfg = tomli.load(f)

GALAXY = "NGC5765b"

# Build model
cfg = master_cfg.copy()
cfg["model"] = master_cfg["model"].copy()
cfg["model"]["marginalise_r"] = False
gcfg = master_cfg["model"]["galaxies"][GALAXY]
data = load_megamaser_spots(
    master_cfg["io"]["maser_data"]["root"], galaxy=GALAXY,
    v_sys_obs=gcfg["v_sys_obs"])
for key in ("D_lo", "D_hi"):
    if key in gcfg:
        data[key] = float(gcfg[key])
tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(cfg, tmp)
tmp.close()
model = MaserDiskModel(tmp.name, data)
os.unlink(tmp.name)

# Physical params from MAP init
init = gcfg["init"]
H0 = float(init["H0"])
D_c = float(init["D_c"])
eta = float(init["eta"])
h = H0 / 100.0
z_cosmo = float(model.distance2redshift(jnp.atleast_1d(D_c), h=h).squeeze())
D_A = D_c / (1.0 + z_cosmo)
M_BH = 10.0**(eta + np.log10(D_A) - 7.0)
x0 = float(init["x0"])
y0 = float(init["y0"])
v_sys = model.v_sys_obs + float(init.get("dv_sys", 0.0))

args = (
    x0, y0, D_A, M_BH, v_sys,
    model._r_ang_ref,
    jnp.asarray(np.deg2rad(float(init["i0"]))),
    jnp.asarray(np.deg2rad(float(init["di_dr"]))),
    jnp.asarray(np.deg2rad(float(init["Omega0"]))),
    jnp.asarray(np.deg2rad(float(init["dOmega_dr"]))),
    jnp.asarray(float(init["sigma_x_floor"]) ** 2),
    jnp.asarray(float(init["sigma_y_floor"]) ** 2),
    jnp.asarray(float(init["sigma_v_sys"]) ** 2),
    jnp.asarray(float(init["sigma_v_hv"]) ** 2),
    jnp.asarray(float(init["sigma_a_floor"]) ** 2),
)

# r_ang centering estimates
sin_i = np.abs(np.sin(np.deg2rad(float(init["i0"]))))
dv = np.asarray(model._all_v) - v_sys
r_est = M_BH * (C_v * sin_i) ** 2 / (D_A * (dv ** 2 + 1e-30))
r_min = float(model._r_ang_lo)
r_max = float(model._r_ang_hi)
r_est = np.clip(r_est, r_min, r_max)

# Pick one HV spot near median r
is_hv = np.asarray(model.is_highvel)
hv_idx = np.where(is_hv)[0]
median_r = np.median(r_est[hv_idx])
spot = hv_idx[np.argmin(np.abs(r_est[hv_idx] - median_r))]
r_spot = r_est[spot]
print(f"Spot {spot}: r_est={r_spot:.4f} mas, "
      f"v={float(model._all_v[spot]):.1f} km/s (HV)")

# Fine phi grid
n_phi = 10001
phi = np.linspace(0, 2 * np.pi, n_phi)
sin_phi = jnp.sin(phi)
cos_phi = jnp.cos(phi)

scales = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

fig, axes = plt.subplots(len(scales), 1, figsize=(8, 2.2 * len(scales)),
                         sharex=True)

for ax, scale in zip(axes, scales):
    r_ang = jnp.clip(jnp.asarray(r_est * scale), r_min, r_max)

    ll = model._phi_integrand(r_ang, sin_phi, cos_phi, *args)
    ll_spot = np.array(ll[spot])
    ll_spot = ll_spot - ll_spot.max()

    ax.plot(np.rad2deg(phi), ll_spot, lw=0.8)
    ax.set_ylabel(r"$\Delta \ln L$")
    ax.set_ylim(-50, 5)
    ax.axhline(0, color="k", lw=0.3)

    # Annotate with FWHM
    above = ll_spot > -0.5
    if above.any():
        fwhm = np.rad2deg(phi[above][-1] - phi[above][0])
        ax.set_title(f"scale={scale:.1f}x  (r={float(r_ang[spot]):.3f} mas)  "
                     f"FWHM={fwhm:.2f}°", fontsize=9, loc="left")
    else:
        ax.set_title(f"scale={scale:.1f}x  (r={float(r_ang[spot]):.3f} mas)  "
                     f"FWHM<{360/n_phi:.3f}°", fontsize=9, loc="left")

axes[-1].set_xlabel(r"$\phi$ [deg]")
axes[-1].xaxis.set_major_locator(MultipleLocator(60))
axes[-1].set_xlim(0, 360)

fig.suptitle(f"{GALAXY} spot {spot} (HV, r_est={r_spot:.3f} mas)",
             fontsize=11)
plt.tight_layout()
plt.savefig("results/Maser/phi_integrand_vs_r.png", dpi=150)
print("Saved results/Maser/phi_integrand_vs_r.png")
