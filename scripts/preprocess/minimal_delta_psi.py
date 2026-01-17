"""
Minimal script to compute δ with ψ(r) correction and compare to Carrick.
Includes LOS comparison to 10 random clusters.
"""
import numpy as np
import healpy as hp
import h5py
from scipy.special import gammaincc, gamma as gamma_func
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

def radec_to_galactic(ra, dec):
    """Convert RA, Dec (ICRS) to galactic l, b (all in degrees)."""
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    return c.galactic.l.degree, c.galactic.b.degree

# Parameters
NGRID = 257  # Match Carrick resolution
BOXSIZE = 400.0
SIGMA_SMOOTH = 4.0
R_2MRS_CUTOFF = 125.0
M_STAR = -23.2457  # Row 0
ALPHA = -0.8778    # Row 0
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

def compute_los_delta(delta_3d, ell_deg, b_deg, r_arr, box_side=400.0):
    """Interpolate 3D delta field along LOS directions."""
    N = delta_3d.shape[0]
    cell = box_side / N
    coords = np.linspace(-box_side/2 + cell/2, box_side/2 - cell/2, N)
    interp = RegularGridInterpolator((coords, coords, coords), delta_3d,
                                      bounds_error=False, fill_value=0.0)
    # Unit vectors in galactic cartesian
    ell_rad, b_rad = np.deg2rad(ell_deg), np.deg2rad(b_deg)
    cos_b = np.cos(b_rad)
    rhat = np.stack([cos_b * np.cos(ell_rad), cos_b * np.sin(ell_rad), np.sin(b_rad)], axis=-1)
    # Interpolate along each LOS
    n_gal = len(ell_deg)
    los_delta = np.zeros((n_gal, len(r_arr)))
    for i in range(n_gal):
        positions = r_arr[:, None] * rhat[i]
        los_delta[i] = interp(positions)
    return los_delta

def velocity_from_density_fft(delta, beta, box_side_mpc, zero_pad=True):
    """Compute velocity field from density using FFT (Carrick Eq. 1)."""
    N = delta.shape[0]
    if zero_pad:
        N_pad = 2 * N
        delta_padded = np.zeros((N_pad, N_pad, N_pad), dtype=np.float64)
        delta_padded[:N, :N, :N] = delta
        box_side_pad = box_side_mpc * 2
    else:
        delta_padded = delta
        N_pad = N
        box_side_pad = box_side_mpc
    dx_pad = box_side_pad / N_pad

    # FFT of density
    delta_k = np.fft.rfftn(delta_padded)

    # Wave vectors
    kx = 2.0 * np.pi * np.fft.fftfreq(N_pad, d=dx_pad)
    ky = 2.0 * np.pi * np.fft.fftfreq(N_pad, d=dx_pad)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N_pad, d=dx_pad)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0  # Avoid division by zero

    # v_k = +i * β * H0 * (k/k²) * δ_k
    H0 = 100.0
    prefactor = 1j * beta * H0 / K2
    vx_k = prefactor * KX * delta_k
    vy_k = prefactor * KY * delta_k
    vz_k = prefactor * KZ * delta_k
    vx_k[0, 0, 0] = vy_k[0, 0, 0] = vz_k[0, 0, 0] = 0.0

    # IFFT back to real space
    vx = np.fft.irfftn(vx_k, s=(N_pad, N_pad, N_pad))
    vy = np.fft.irfftn(vy_k, s=(N_pad, N_pad, N_pad))
    vz = np.fft.irfftn(vz_k, s=(N_pad, N_pad, N_pad))

    if zero_pad:
        vx, vy, vz = vx[:N, :N, :N], vy[:N, :N, :N], vz[:N, :N, :N]

    return np.stack([vx, vy, vz], axis=-1).astype(np.float32)

def compute_los_velocity(vel_3d, ell_deg, b_deg, r_arr, box_side=400.0):
    """Interpolate 3D velocity field along LOS and project to LOS component."""
    N = vel_3d.shape[0]
    cell = box_side / N
    coords = np.linspace(-box_side/2 + cell/2, box_side/2 - cell/2, N)

    # Create interpolators for each component
    interp_vx = RegularGridInterpolator((coords, coords, coords), vel_3d[..., 0],
                                         bounds_error=False, fill_value=0.0)
    interp_vy = RegularGridInterpolator((coords, coords, coords), vel_3d[..., 1],
                                         bounds_error=False, fill_value=0.0)
    interp_vz = RegularGridInterpolator((coords, coords, coords), vel_3d[..., 2],
                                         bounds_error=False, fill_value=0.0)

    # Unit vectors in galactic cartesian
    ell_rad, b_rad = np.deg2rad(ell_deg), np.deg2rad(b_deg)
    cos_b = np.cos(b_rad)
    rhat = np.stack([cos_b * np.cos(ell_rad), cos_b * np.sin(ell_rad), np.sin(b_rad)], axis=-1)

    n_gal = len(ell_deg)
    los_vel = np.zeros((n_gal, len(r_arr)))
    for i in range(n_gal):
        positions = r_arr[:, None] * rhat[i]
        vx = interp_vx(positions)
        vy = interp_vy(positions)
        vz = interp_vz(positions)
        # Project onto LOS direction
        los_vel[i] = vx * rhat[i, 0] + vy * rhat[i, 1] + vz * rhat[i, 2]
    return los_vel

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

# ============================================================
# Alternate field using lumWeight instead of weight
# ============================================================
print("Computing alternate field with lumWeight...")
w_lum = aquila['lumWeight'] * L_Lstar
w_lum[(flag_2mrs == 1) & (r_gal > R_2MRS_CUTOFF)] = 0.0

# Deposit with CIC
rho_lum = np.zeros((NGRID, NGRID, NGRID))
for di in [0, 1]:
    for dj in [0, 1]:
        for dk in [0, 1]:
            wx = (1 - ddx) if di == 0 else ddx
            wy = (1 - ddy) if dj == 0 else ddy
            wz = (1 - ddz) if dk == 0 else ddz
            ixx = np.clip(ix0 + di, 0, NGRID-1)
            iyy = np.clip(iy0 + dj, 0, NGRID-1)
            izz = np.clip(iz0 + dk, 0, NGRID-1)
            np.add.at(rho_lum, (ixx, iyy, izz), w_lum * wx * wy * wz)

# Delta and ψ correction
rho_lum_mean = rho_lum[effective].mean()
delta_lum = np.where(effective, rho_lum / rho_lum_mean - 1, 0.0)
delta_lum_psi = np.where(effective, delta_lum / psi, 0.0)
delta_lum_corr = gaussian_filter(delta_lum_psi, SIGMA_SMOOTH / cell)
print(f"δ_lum range: [{delta_lum_corr.min():.2f}, {delta_lum_corr.max():.2f}]")
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

# ============================================================
# LOS comparison to clusters
# ============================================================
print("\nLoading cluster data for LOS comparison...")
with h5py.File('data/Clusters/los_Clusters_Carrick2015.hdf5', 'r') as f:
    cluster_RA = f['RA'][...]
    cluster_dec = f['dec'][...]
    r_los = f['r'][...]
print(f"Loaded {len(cluster_RA)} clusters, r_los from 0 to {r_los.max():.0f} Mpc/h")

# Load Carrick dField_1Runs for comparison
print("Loading dField_1Runs.npy...")
delta_carrick_1 = np.load('data/Carrick_reconstruction_2015/dField_1Runs.npy')

# Load make_2mpp_carrick fields (pre and post iteration)
print("Loading make_2mpp fields (iter0=pre, iter2=post)...")
delta_make2mpp_iter0 = np.load('delta_make2mpp_iter0.npy')
delta_make2mpp_iter2 = np.load('delta_make2mpp_iter2.npy')

# Convert to galactic
cluster_ell, cluster_b = radec_to_galactic(cluster_RA, cluster_dec)

# Load cluster names
cluster_names = np.genfromtxt('data/Clusters/ClustersData.txt', dtype='U32', usecols=0, skip_header=1)

# Select 10 random clusters
np.random.seed(42)
n_clusters_plot = 10
n_total = len(cluster_RA)
idx = np.random.choice(n_total, min(n_clusters_plot, n_total), replace=False)
idx = np.sort(idx)
print(f"Selected {len(idx)} clusters for LOS comparison")

# Compute LOS profiles for all fields
print("Computing LOS profiles...")
los_ours = compute_los_delta(delta_corr, cluster_ell, cluster_b, r_los, BOXSIZE)
los_lumweight = compute_los_delta(delta_lum_corr, cluster_ell, cluster_b, r_los, BOXSIZE)
los_carrick_1 = compute_los_delta(delta_carrick_1, cluster_ell, cluster_b, r_los, BOXSIZE)
los_make2mpp_iter0 = compute_los_delta(delta_make2mpp_iter0, cluster_ell, cluster_b, r_los, BOXSIZE)
los_make2mpp_iter2 = compute_los_delta(delta_make2mpp_iter2, cluster_ell, cluster_b, r_los, BOXSIZE)

# Plot LOS comparison (2 rows x 5 cols)
fig2, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i >= len(idx):
        ax.axis('off')
        continue
    ci = idx[i]
    # Convert to log10(1+delta) for plotting
    log_carrick = np.log10(np.clip(1 + los_carrick_1[ci], 1e-5, None))
    log_ours = np.log10(np.clip(1 + los_ours[ci], 1e-5, None))
    log_lumweight = np.log10(np.clip(1 + los_lumweight[ci], 1e-5, None))
    log_iter0 = np.log10(np.clip(1 + los_make2mpp_iter0[ci], 1e-5, None))
    log_iter2 = np.log10(np.clip(1 + los_make2mpp_iter2[ci], 1e-5, None))

    ax.plot(r_los, log_carrick, 'b-', lw=1.2, label='dField_1Runs')
    ax.plot(r_los, log_ours, 'r-', lw=1.2, label='minimal (weight)')
    ax.plot(r_los, log_lumweight, 'c-.', lw=1.0, label='minimal (lumweight)')
    ax.plot(r_los, log_iter0, 'g--', lw=1.0, label='make2mpp iter0')
    ax.plot(r_los, log_iter2, 'm:', lw=1.0, label='make2mpp iter2')
    ax.set_xlim(0, 200)
    ax.axhline(0, color='k', ls='--', alpha=0.3)

    name = cluster_names[ci] if ci < len(cluster_names) else f'Cluster {ci}'
    ax.text(0.05, 0.9, name, transform=ax.transAxes, fontsize=8)

    if i >= 5:
        ax.set_xlabel('r [Mpc/h]')
    if i % 5 == 0:
        ax.set_ylabel(r'$\log_{10}(1+\delta)$')

# Add legend to first panel
axes[0].legend(loc='lower right', fontsize=6)

fig2.suptitle('LOS δ comparison to 10 random clusters', fontsize=12)
plt.tight_layout()
plt.savefig('los_comparison_clusters.png', dpi=150, bbox_inches='tight')
print("Saved to los_comparison_clusters.png")

# ============================================================
# LOS velocity comparison
# ============================================================
print("\n" + "="*60)
print("LOS Velocity Comparison")
print("="*60)

# Load dField_0Runs and compute velocity via FFT
print("Loading dField_0Runs.npy...")
delta_carrick_0 = np.load('data/Carrick_reconstruction_2015/dField_0Runs.npy')

print("Computing velocity from δ using FFT (β=1.0, vField is stored unscaled)...")
BETA_COMPUTE = 1.0  # vField is stored with β*=1, user scales by β*=0.43 when using
vel_computed = velocity_from_density_fft(delta_carrick_0, BETA_COMPUTE, BOXSIZE, zero_pad=True)
print(f"Computed velocity shape: {vel_computed.shape}, range: [{vel_computed.min():.0f}, {vel_computed.max():.0f}] km/s")

# Load reference vField_0Runs (shape: 3, 257, 257, 257)
print("Loading vField_0Runs.npy...")
vel_carrick_raw = np.load('data/Carrick_reconstruction_2015/vField_0Runs.npy')
# Transpose to (257, 257, 257, 3) to match our format
vel_carrick = np.moveaxis(vel_carrick_raw, 0, -1)
print(f"Carrick velocity shape: {vel_carrick.shape}, range: [{vel_carrick.min():.0f}, {vel_carrick.max():.0f}] km/s")

# Compute LOS velocity profiles for same clusters
print("Computing LOS velocity profiles...")
los_vel_computed = compute_los_velocity(vel_computed, cluster_ell, cluster_b, r_los, BOXSIZE)
los_vel_carrick = compute_los_velocity(vel_carrick, cluster_ell, cluster_b, r_los, BOXSIZE)

# Plot LOS velocity comparison (2 rows x 5 cols)
fig3, axes3 = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
axes3 = axes3.flatten()

for i, ax in enumerate(axes3):
    if i >= len(idx):
        ax.axis('off')
        continue
    ci = idx[i]

    ax.plot(r_los, los_vel_carrick[ci], 'b-', lw=1.2, label='vField_0Runs')
    ax.plot(r_los, los_vel_computed[ci], 'r--', lw=1.2, label='FFT from δ')
    ax.set_xlim(0, 200)
    ax.axhline(0, color='k', ls='--', alpha=0.3)

    name = cluster_names[ci] if ci < len(cluster_names) else f'Cluster {ci}'
    ax.text(0.05, 0.9, name, transform=ax.transAxes, fontsize=8)

    if i >= 5:
        ax.set_xlabel('r [Mpc/h]')
    if i % 5 == 0:
        ax.set_ylabel(r'$v_{LOS}$ [km/s]')

# Add legend to first panel
axes3[0].legend(loc='lower right', fontsize=7)

fig3.suptitle('LOS velocity comparison (β=1.0, unscaled)', fontsize=12)
plt.tight_layout()
plt.savefig('los_velocity_clusters.png', dpi=150, bbox_inches='tight')
print("Saved to los_velocity_clusters.png")
