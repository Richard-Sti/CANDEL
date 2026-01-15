"""
Minimal script to compute δ with ψ(r) correction and compare to Carrick.
"""
import numpy as np
import healpy as hp
from scipy.special import gammaincc, gamma as gamma_func
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Parameters
NGRID = 257  # Match Carrick resolution
BOXSIZE = 400.0
SIGMA_SMOOTH = 4.0
R_2MRS_CUTOFF = 125.0
M_STAR = -23.2457
ALPHA = -0.8778
BIAS_CONST = 0.73
BIAS_SLOPE = 0.24
Rmax = BOXSIZE / 2
C_LIGHT = 299792.458  # km/s
H0 = 100.0  # km/s/Mpc (h=1)
Q0 = -0.55  # deceleration parameter for LCDM

# Load data
print("Loading data...")
aquila = np.load('data/Carrick_reconstruction_2015/2m++_0Runs.npy', allow_pickle=True)
delta_carrick = np.load('data/Carrick_reconstruction_2015/dField_0Runs.npy')
coverage_map = hp.read_map('coverage_aquila_filled.fits')
nside = hp.get_nside(coverage_map)

# ψ(r) functions
def psi_func(r, m_lim):
    r = np.maximum(r, 0.1)
    M_lim = m_lim - 5 * np.log10(r) - 25
    x = 10**(0.4 * (M_STAR - M_lim))
    L_ratio = gammaincc(ALPHA+3, x) * gamma_func(ALPHA+3) / (gammaincc(ALPHA+2, x) * gamma_func(ALPHA+2) + 1e-12)
    return BIAS_CONST + BIAS_SLOPE * L_ratio

# Galaxy data
gal_l = np.deg2rad(aquila['gal_long'])
gal_b = np.deg2rad(aquila['gal_lat'])
# distance column is already converged real-space distance from Carrick's iteration (in km/s units)
r_gal = aquila['distance'] / 100.0  # km/s -> Mpc/h
flag_2mrs = aquila['flag_2mrs_mask_final']
L_Lstar = 10**(-0.4 * (aquila['AbsMag'] - M_STAR))
# Use weight (includes FoF suppression)
w = aquila['weight'] * L_Lstar
w[(flag_2mrs == 1) & (r_gal > R_2MRS_CUTOFF)] = 0.0
print(f"Distance range: {r_gal.min():.1f} - {r_gal.max():.1f} Mpc/h")

# Deposit on grid using CIC
cell = BOXSIZE / NGRID
obs = BOXSIZE / 2
xg = r_gal * np.cos(gal_b) * np.cos(gal_l) + obs
yg = r_gal * np.cos(gal_b) * np.sin(gal_l) + obs
zg = r_gal * np.sin(gal_b) + obs

# CIC: distribute weight to 8 neighboring cells
rho = np.zeros((NGRID, NGRID, NGRID))
xc, yc, zc = xg / cell - 0.5, yg / cell - 0.5, zg / cell - 0.5
ix0 = np.floor(xc).astype(int)
iy0 = np.floor(yc).astype(int)
iz0 = np.floor(zc).astype(int)
ddx, ddy, ddz = xc - ix0, yc - iy0, zc - iz0

for di in [0, 1]:
    for dj in [0, 1]:
        for dk in [0, 1]:
            wx = (1 - ddx) if di == 0 else ddx
            wy = (1 - ddy) if dj == 0 else ddy
            wz = (1 - ddz) if dk == 0 else ddz
            ixx = np.clip(ix0 + di, 0, NGRID-1)
            iyy = np.clip(iy0 + dj, 0, NGRID-1)
            izz = np.clip(iz0 + dk, 0, NGRID-1)
            np.add.at(rho, (ixx, iyy, izz), w * wx * wy * wz)

print("Deposited (CIC)")

# Build 3D coordinates
print("Building 3D mask...")
ii, jj, kk = np.mgrid[0:NGRID, 0:NGRID, 0:NGRID]
xv = (ii + 0.5) * cell - obs
yv = (jj + 0.5) * cell - obs
zv = (kk + 0.5) * cell - obs
rv = np.sqrt(xv**2 + yv**2 + zv**2)

# Galactic coords for each voxel
l_vox = np.arctan2(yv, xv)
b_vox = np.arcsin(np.clip(zv / np.maximum(rv, 1e-10), -1, 1))
theta_vox = np.pi/2 - b_vox
phi_vox = np.mod(l_vox, 2*np.pi)

# Coverage lookup (vectorized)
pix = hp.ang2pix(nside, theta_vox.ravel(), phi_vox.ravel())
is_deep = (coverage_map[pix] == 1).reshape(NGRID, NGRID, NGRID)
is_2mrs = ~is_deep

# Masks
zeroed = is_2mrs & (rv > R_2MRS_CUTOFF)
valid = rv <= Rmax
effective = valid & ~zeroed
print(f"Zeroed fraction: {100*zeroed[valid].mean():.1f}%")

# Delta - use effective volume for normalization (before smoothing)
rho_mean = rho[effective].mean()
delta = np.where(effective, rho / rho_mean - 1, 0.0)

# ψ correction BEFORE smoothing (Carrick order) - uniform 11.5
rv_clip = np.clip(rv, 1, Rmax)
psi = psi_func(rv_clip, 11.5)
delta_psi = np.where(effective, delta / psi, 0.0)

# Smooth AFTER ψ correction (4 Mpc/h) - single pass
delta_corr = gaussian_filter(delta_psi, SIGMA_SMOOTH / cell)
delta = gaussian_filter(delta, SIGMA_SMOOTH / cell)  # Also smooth uncorrected for comparison

print(f"δ range: [{delta.min():.2f}, {delta.max():.2f}]")
print(f"δ_corr range: [{delta_corr.min():.2f}, {delta_corr.max():.2f}]")

# Carrick radial coords
N_c = delta_carrick.shape[0]
cell_c = BOXSIZE / N_c
ic, jc, kc = np.mgrid[0:N_c, 0:N_c, 0:N_c]
rc = np.sqrt(((ic+0.5)*cell_c - obs)**2 + ((jc+0.5)*cell_c - obs)**2 + ((kc+0.5)*cell_c - obs)**2)

# Radial profiles
print("Computing profiles...")
r_bins = np.linspace(0, Rmax, 51)
r_cen = 0.5 * (r_bins[:-1] + r_bins[1:])

prof_uncorr = np.zeros(len(r_cen))
prof_corr = np.zeros(len(r_cen))
prof_2mrs = np.zeros(len(r_cen))
prof_deep = np.zeros(len(r_cen))
prof_carrick = np.zeros(len(r_cen))

for i in range(len(r_cen)):
    shell = (rv >= r_bins[i]) & (rv < r_bins[i+1])
    m = shell & effective
    if m.sum() > 0:
        prof_uncorr[i] = delta[m].mean()
        prof_corr[i] = delta_corr[m].mean()

    m2 = shell & is_2mrs & (rv <= R_2MRS_CUTOFF)
    prof_2mrs[i] = delta_corr[m2].mean() if m2.sum() > 0 else np.nan

    md = shell & is_deep
    prof_deep[i] = delta_corr[md].mean() if md.sum() > 0 else np.nan

    mc = (rc >= r_bins[i]) & (rc < r_bins[i+1])
    prof_carrick[i] = delta_carrick[mc].mean() if mc.sum() > 0 else np.nan

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(r_cen, prof_carrick, 'g-', lw=2, label='Carrick (dField_0Runs)')
ax1.plot(r_cen, prof_uncorr, 'b--', lw=2, label='Ours (uncorrected)')
ax1.plot(r_cen, prof_corr, 'r-', lw=2, label='Ours (ψ-corrected)')
ax1.axhline(0, color='k', ls='--', alpha=0.5)
ax1.axvline(R_2MRS_CUTOFF, color='gray', ls=':', alpha=0.7)
ax1.set_xlabel('r [Mpc/h]')
ax1.set_ylabel('<δ(r)>')
ax1.set_title('Mean overdensity vs Carrick')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, Rmax)

ax2.plot(r_cen, prof_carrick, 'g-', lw=2, label='Carrick')
ax2.plot(r_cen, prof_2mrs, 'b-', lw=2, label='2MRS (ψ₁₁.₅)')
ax2.plot(r_cen, prof_deep, 'r-', lw=2, label='Deep (ψ₁₂.₅)')
ax2.axhline(0, color='k', ls='--', alpha=0.5)
ax2.axvline(R_2MRS_CUTOFF, color='gray', ls=':', alpha=0.7)
ax2.set_xlabel('r [Mpc/h]')
ax2.set_ylabel('<δ(r)>')
ax2.set_title('ψ-corrected by region')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, Rmax)

plt.tight_layout()
plt.savefig('minimal_delta_psi.png', dpi=150, bbox_inches='tight')
print("\nSaved to minimal_delta_psi.png")
