"""
Compare luminosity weights computed from Aquila catalogue to stored lumWeight.
Deposits on grid and compares to Carrick, matching minimal_delta_psi.py exactly.
"""
import numpy as np
from scipy import special as sps
from scipy.special import gammaincc, gamma as gamma_func
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import healpy as hp

# =============================================================================
# Parameters - match minimal_delta_psi.py exactly
# =============================================================================
NGRID = 257
BOXSIZE = 400.0
SIGMA_SMOOTH = 4.0
R_2MRS_CUTOFF = 125.0
Rmax = BOXSIZE / 2
N_PAD = 288  # Reduced FFT padding (same as carrick_iterated.py)

# LG apex direction (Carrick 2015)
L_APEX = 276.0  # deg
B_APEX = 30.0   # deg
V_LG = 627.0    # km/s, LG velocity in CMB frame
C_LIGHT = 299792.458
H0 = 100.0
OMEGA_M = 0.30
OMEGA_DE = 0.70

# Schechter params from Lavaux & Hudson 2011 Table 2 (CMB frame)
# M∈[-25,-17], cz∈[750,20000]: α = -0.80 ± 0.01, M* = -23.22 ± 0.01
MSTAR = -23.22
ALPHA = -0.80

# Bias parameters (Westover 2007)
BIAS_CONST = 0.73
BIAS_SLOPE = 0.24

ABS_MAGLIM_BRIGHT = -26.0
ABS_MAGLIM_FAINT = -17.0

print(f"Using Schechter params (Lavaux+11 CMB): M*={MSTAR:.2f}, α={ALPHA:.2f}")

# =============================================================================
# Cosmological luminosity distance
# =============================================================================
def luminosity_distance(z):
    if z <= 0:
        return 0.0
    def integrand(zp):
        return 1.0 / np.sqrt(OMEGA_M * (1+zp)**3 + OMEGA_DE)
    eta, _ = quad(integrand, 0, z)
    return (C_LIGHT / H0) * (1 + z) * eta

def luminosity_distance_vec(z_arr):
    return np.array([luminosity_distance(z) for z in z_arr])

# =============================================================================
# Schechter LF integrals
# =============================================================================
def gamma_upper(s, x):
    x = np.asarray(x, dtype=np.float64)
    if s > 0:
        return sps.gammaincc(s, x) * sps.gamma(s)
    k = int(np.ceil(1.0 - s))
    sp = s + k
    result = sps.gammaincc(sp, x) * sps.gamma(sp)
    for _ in range(k):
        result = (result - np.power(x, sp - 1) * np.exp(-x)) / (sp - 1)
        sp -= 1
    return result

def integral_lw_lumfun(M):
    x = 10.0 ** (0.4 * (MSTAR - M))
    a = ALPHA + 2.0
    return gamma_upper(a, x)

def integral_lw_lumfun_2m(M_faint, M_bright):
    return integral_lw_lumfun(M_faint) - integral_lw_lumfun(M_bright)

# =============================================================================
# ψ(r) function - match minimal_delta_psi.py exactly
# =============================================================================
def psi_func(r, m_lim):
    r = np.maximum(r, 0.1)
    M_lim = m_lim - 5 * np.log10(r) - 25
    x = 10**(0.4 * (MSTAR - M_lim))
    L_ratio = gammaincc(ALPHA+3, x) * gamma_func(ALPHA+3) / (gammaincc(ALPHA+2, x) * gamma_func(ALPHA+2) + 1e-12)
    return BIAS_CONST + BIAS_SLOPE * L_ratio

# =============================================================================
# Load data
# =============================================================================
print("Loading Aquila catalogue...")
aquila = np.load('data/Carrick_reconstruction_2015/2m++_0Runs.npy', allow_pickle=True)
print(f"  {len(aquila)} galaxies")

print("Loading coverage map...")
coverage_map = hp.read_map('coverage_aquila_filled.fits', verbose=False)
nside = hp.get_nside(coverage_map)

print("Loading Carrick field...")
delta_carrick = np.load('data/Carrick_reconstruction_2015/dField_0Runs.npy')

# Extract fields
K2MRS = aquila['K2MRS']
gal_l = np.deg2rad(aquila['gal_long'])
gal_b = np.deg2rad(aquila['gal_lat'])

# Use catalogue distance directly (already computed as best_velcmb - v_LG·cos(θ))
distance_kms = aquila['distance']

c1_all = aquila['c1_all']
c2_all = aquila['c2_all']
lumWeight_stored = aquila['lumWeight']
flag_2mrs = aquila['flag_2mrs_mask_final']

# =============================================================================
# Compute weights following C++ logic
# =============================================================================
print("Computing cosmological distances...")
z = np.maximum(distance_kms / C_LIGHT, 100.0 / C_LIGHT)
d_L = luminosity_distance_vec(z)
mu = 5.0 * np.log10(np.maximum(d_L, 0.01) * 1e5)

print("Computing weights...")
m_b, m_f = 11.5, 12.5
M = K2MRS - mu
Mb = np.clip(m_b - mu, ABS_MAGLIM_BRIGHT, ABS_MAGLIM_FAINT)
Mf = np.clip(m_f - mu, ABS_MAGLIM_BRIGHT, ABS_MAGLIM_FAINT)

cb = c1_all.copy()
cf = c2_all.copy()
cb[cb < 0.5] = 0
cf[cf < 0.5] = 0
cf[np.isnan(cf)] = 0
cb[np.isnan(cb)] = 0

numer = (cf * integral_lw_lumfun_2m(Mf, Mb) +
         cb * integral_lw_lumfun_2m(Mb, ABS_MAGLIM_BRIGHT))
denom = integral_lw_lumfun_2m(ABS_MAGLIM_FAINT, ABS_MAGLIM_BRIGHT)
getWeight_2m = numer / denom
w_lum_computed = 1.0 / np.maximum(getWeight_2m, 1e-10)

# =============================================================================
# Compare weights
# =============================================================================
r_mpc = distance_kms / 100.0
valid = (lumWeight_stored > 0) & (r_mpc < 200) & np.isfinite(w_lum_computed)
ratio_lum = w_lum_computed[valid] / lumWeight_stored[valid]
print(f"\nLUMINOSITY weights:")
print(f"  Median ratio (computed/stored): {np.median(ratio_lum):.3f}")
print(f"  Correlation: {np.corrcoef(lumWeight_stored[valid], w_lum_computed[valid])[0,1]:.4f}")

# =============================================================================
# Plot weight comparison
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.scatter(r_mpc[valid], lumWeight_stored[valid], s=1, alpha=0.1, c='blue')
ax.set_xlabel('r [Mpc/h]')
ax.set_ylabel('lumWeight')
ax.set_title('Stored (Aquila)')
ax.set_xlim(0, 200)
ax.set_ylim(0, 10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(r_mpc[valid], w_lum_computed[valid], s=1, alpha=0.1, c='red')
ax.set_xlabel('r [Mpc/h]')
ax.set_ylabel('lumWeight')
ax.set_title(f'Computed (α={ALPHA:.3f}, M*={MSTAR:.2f})')
ax.set_xlim(0, 200)
ax.set_ylim(0, 10)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.scatter(lumWeight_stored[valid], w_lum_computed[valid], s=1, alpha=0.1)
ax.plot([0, 10], [0, 10], 'r--', lw=2, label='1:1')
ax.set_xlabel('lumWeight (stored)')
ax.set_ylabel('lumWeight (computed)')
ax.set_title('1:1 comparison')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('compare_lumweight_aquila.png', dpi=150)
print("\nSaved compare_lumweight_aquila.png")

# =============================================================================
# Deposit on grid (matching minimal_delta_psi.py)
# =============================================================================
print("\n" + "="*60)
print("Depositing galaxies on grid...")
print("="*60)

# Galaxy positions
gal_l = np.deg2rad(aquila['gal_long'])
gal_b = np.deg2rad(aquila['gal_lat'])

# L/L* from stored AbsMag
L_Lstar = 10**(-0.4 * (aquila['AbsMag'] - MSTAR))

# Total weights: lumWeight * L/L*
w_stored = lumWeight_stored * L_Lstar
w_computed = w_lum_computed * L_Lstar

# Zero out 2MRS galaxies beyond cutoff
w_stored[(flag_2mrs == 1) & (r_mpc > R_2MRS_CUTOFF)] = 0.0
w_computed[(flag_2mrs == 1) & (r_mpc > R_2MRS_CUTOFF)] = 0.0

# Zero out galaxies that Carrick excluded (lumWeight=0 in Run0 catalogue)
carrick_excluded = lumWeight_stored == 0
w_computed[carrick_excluded] = 0.0
print(f"Excluded {carrick_excluded.sum()} galaxies with lumWeight=0 in Run0")

print(f"Distance range: {r_mpc.min():.1f} - {r_mpc.max():.1f} Mpc/h")

# Grid setup
cell = BOXSIZE / NGRID
obs = BOXSIZE / 2

# CIC deposit function with positions as argument
def deposit_cic(weights, r_dist):
    xg = r_dist * np.cos(gal_b) * np.cos(gal_l) + obs
    yg = r_dist * np.cos(gal_b) * np.sin(gal_l) + obs
    zg = r_dist * np.sin(gal_b) + obs
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
                np.add.at(rho, (ixx, iyy, izz), weights * wx * wy * wz)
    return rho

print("Depositing stored weights...")
rho_stored = deposit_cic(w_stored, r_mpc)
print("Depositing computed weights...")
rho_computed = deposit_cic(w_computed, r_mpc)

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

# Coverage lookup
pix = hp.ang2pix(nside, theta_vox.ravel(), phi_vox.ravel())
is_deep = (coverage_map[pix] == 1).reshape(NGRID, NGRID, NGRID)
is_2mrs = ~is_deep

# Masks
zeroed = is_2mrs & (rv > R_2MRS_CUTOFF)
valid_mask = rv <= Rmax
effective = valid_mask & ~zeroed
print(f"Zeroed fraction: {100*zeroed[valid_mask].mean():.1f}%")

# Delta - use effective volume for normalization (before smoothing)
rho_stored_mean = rho_stored[effective].mean()
rho_computed_mean = rho_computed[effective].mean()

delta_stored = np.where(effective, rho_stored / rho_stored_mean - 1, 0.0)
delta_computed = np.where(effective, rho_computed / rho_computed_mean - 1, 0.0)

# ψ correction BEFORE smoothing (uniform 11.5 like minimal_delta_psi.py)
rv_clip = np.clip(rv, 1, Rmax)
psi = psi_func(rv_clip, 11.5)
delta_stored_psi = np.where(effective, delta_stored / psi, 0.0)
delta_computed_psi = np.where(effective, delta_computed / psi, 0.0)

# Smooth AFTER ψ correction
delta_stored_corr = gaussian_filter(delta_stored_psi, SIGMA_SMOOTH / cell)
delta_computed_corr = gaussian_filter(delta_computed_psi, SIGMA_SMOOTH / cell)

print(f"δ_stored range: [{delta_stored_corr.min():.2f}, {delta_stored_corr.max():.2f}]")
print(f"δ_computed range: [{delta_computed_corr.min():.2f}, {delta_computed_corr.max():.2f}]")

# Carrick radial coords
N_c = delta_carrick.shape[0]
cell_c = BOXSIZE / N_c
ic, jc, kc = np.mgrid[0:N_c, 0:N_c, 0:N_c]
rc = np.sqrt(((ic+0.5)*cell_c - obs)**2 + ((jc+0.5)*cell_c - obs)**2 + ((kc+0.5)*cell_c - obs)**2)

# =============================================================================
# Radial profiles
# =============================================================================
print("Computing radial profiles...")
r_bins = np.linspace(0, Rmax, 41)
r_cen = 0.5 * (r_bins[:-1] + r_bins[1:])

prof_stored = np.zeros(len(r_cen))
prof_computed = np.zeros(len(r_cen))
prof_carrick = np.zeros(len(r_cen))

for i in range(len(r_cen)):
    shell = (rv >= r_bins[i]) & (rv < r_bins[i+1])
    m = shell & effective
    if m.sum() > 0:
        prof_stored[i] = delta_stored_corr[m].mean()
        prof_computed[i] = delta_computed_corr[m].mean()

    mc = (rc >= r_bins[i]) & (rc < r_bins[i+1])
    prof_carrick[i] = delta_carrick[mc].mean() if mc.sum() > 0 else np.nan

# =============================================================================
# Plot comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(r_cen, prof_carrick, 'g-', lw=2, label='Carrick (dField_0Runs)')
ax.plot(r_cen, prof_stored, 'b-', lw=2, label='Stored lumWeight')
ax.plot(r_cen, prof_computed, 'r--', lw=2, label='Computed lumWeight')
ax.axhline(0, color='k', ls='--', alpha=0.5)
ax.axvline(R_2MRS_CUTOFF, color='gray', ls=':', alpha=0.7, label='2MRS cutoff')
ax.set_xlabel('r [Mpc/h]')
ax.set_ylabel('<δ(r)>')
ax.set_title('Mean overdensity radial profile (ψ-corrected)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, Rmax)

ax = axes[1]
ax.plot(r_cen, prof_computed - prof_stored, 'k-', lw=2)
ax.axhline(0, color='r', ls='--', alpha=0.5)
ax.axvline(R_2MRS_CUTOFF, color='gray', ls=':', alpha=0.7)
ax.set_xlabel('r [Mpc/h]')
ax.set_ylabel('δ_computed - δ_stored')
ax.set_title('Difference (computed - stored)')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, Rmax)

plt.tight_layout()
plt.savefig('compare_delta_aquila.png', dpi=150)
print("\nSaved compare_delta_aquila.png")

# =============================================================================
# LOS comparison to clusters
# =============================================================================
print("\n" + "="*60)
print("Computing LOS profiles to clusters...")
print("="*60)

import h5py
from scipy.interpolate import RegularGridInterpolator

# Load cluster LOS data
los_file = 'data/Clusters/los_Clusters_Carrick2015.hdf5'
print(f"Loading cluster LOS data from {los_file}...")
with h5py.File(los_file, 'r') as f:
    cluster_RA = f['RA'][:]
    cluster_dec = f['dec'][:]
    los_r = f['r'][:]
    los_density_carrick = f['los_density'][0]  # Shape (312, 251)
    los_velocity_carrick = f['los_velocity'][0]

n_clusters = len(cluster_RA)
n_r_los = len(los_r)
print(f"  {n_clusters} clusters, {n_r_los} radial bins from {los_r.min():.1f} to {los_r.max():.1f} Mpc/h")

# IMPORTANT: The Carrick LOS file was generated using Carrick2015_FieldLoader.load_velocity()
# which divides by β*=0.43 to remove the pre-applied scaling. So LOS file has v/β*.
# Our FFT velocities use delta_to_velocity() with beta=0.43, so they output v (with β*).
# To compare fairly, multiply Carrick LOS velocities by β*.
BETA_STAR = 0.43
los_velocity_carrick = los_velocity_carrick * BETA_STAR
print(f"  Applied β*={BETA_STAR} scaling to Carrick LOS velocities")

# Load Carrick velocity field (stores v/β*, need to scale by β*)
print("Loading Carrick velocity field...")
v_carrick = np.load('data/Carrick_reconstruction_2015/vField_0Runs.npy') * BETA_STAR
print(f"  Shape: {v_carrick.shape}")

# Compute velocity fields from delta using linear theory
print("Computing velocity fields from δ using linear theory...")

def velocity_from_density_fft(delta, beta, box_side_mpc, n_pad=N_PAD):
    """Compute velocity field from density using FFT (Carrick Eq. 1)."""
    N = delta.shape[0]
    if n_pad > N:
        delta_padded = np.zeros((n_pad, n_pad, n_pad), dtype=np.float64)
        delta_padded[:N, :N, :N] = delta
        box_side_pad = box_side_mpc * n_pad / N
    else:
        delta_padded = delta
        n_pad = N
        box_side_pad = box_side_mpc
    dx_pad = box_side_pad / n_pad

    # FFT of density
    delta_k = np.fft.rfftn(delta_padded)

    # Wave vectors
    kx = 2.0 * np.pi * np.fft.fftfreq(n_pad, d=dx_pad)
    ky = 2.0 * np.pi * np.fft.fftfreq(n_pad, d=dx_pad)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n_pad, d=dx_pad)
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
    vx = np.fft.irfftn(vx_k, s=(n_pad, n_pad, n_pad))
    vy = np.fft.irfftn(vy_k, s=(n_pad, n_pad, n_pad))
    vz = np.fft.irfftn(vz_k, s=(n_pad, n_pad, n_pad))

    if n_pad > N:
        vx, vy, vz = vx[:N, :N, :N], vy[:N, :N, :N], vz[:N, :N, :N]

    return np.stack([vx, vy, vz], axis=0)

# BETA_STAR already defined above when scaling Carrick LOS velocities
v_stored = velocity_from_density_fft(delta_stored_corr, BETA_STAR, BOXSIZE)
v_computed = velocity_from_density_fft(delta_computed_corr, BETA_STAR, BOXSIZE)
print(f"  v_stored: |v| max = {np.sqrt(v_stored[0]**2 + v_stored[1]**2 + v_stored[2]**2).max():.1f} km/s")
print(f"  v_computed: |v| max = {np.sqrt(v_computed[0]**2 + v_computed[1]**2 + v_computed[2]**2).max():.1f} km/s")

# Convert RA/dec to Galactic coordinates
def radec_to_galactic(ra, dec):
    """Convert RA/dec (degrees) to Galactic l,b (degrees)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    return c.galactic.l.degree, c.galactic.b.degree

cluster_l, cluster_b = radec_to_galactic(cluster_RA, cluster_dec)
print(f"  Converted to Galactic coords")

# Unit vectors for each cluster (in Galactic frame)
l_rad = np.deg2rad(cluster_l)
b_rad = np.deg2rad(cluster_b)
rhat = np.stack([
    np.cos(b_rad) * np.cos(l_rad),
    np.cos(b_rad) * np.sin(l_rad),
    np.sin(b_rad)
], axis=1)  # Shape (n_clusters, 3)

# Build interpolators for the computed delta field
print("Building interpolators...")
coords = np.linspace(0, BOXSIZE, NGRID+1)[:-1] + cell/2  # Cell centers
coords_c = np.linspace(0, BOXSIZE, N_c+1)[:-1] + cell_c/2

# Use the smoothed, psi-corrected fields
interp_stored = RegularGridInterpolator(
    (coords, coords, coords), delta_stored_corr,
    bounds_error=False, fill_value=0.0, method='linear')

interp_computed = RegularGridInterpolator(
    (coords, coords, coords), delta_computed_corr,
    bounds_error=False, fill_value=0.0, method='linear')

interp_carrick = RegularGridInterpolator(
    (coords_c, coords_c, coords_c), delta_carrick,
    bounds_error=False, fill_value=0.0, method='linear')

# Build velocity interpolators for stored lumWeight (from FFT)
interp_vx_stored = RegularGridInterpolator(
    (coords, coords, coords), v_stored[0],
    bounds_error=False, fill_value=0.0, method='linear')
interp_vy_stored = RegularGridInterpolator(
    (coords, coords, coords), v_stored[1],
    bounds_error=False, fill_value=0.0, method='linear')
interp_vz_stored = RegularGridInterpolator(
    (coords, coords, coords), v_stored[2],
    bounds_error=False, fill_value=0.0, method='linear')

# Build velocity interpolators for my computed weights (from FFT)
interp_vx_computed = RegularGridInterpolator(
    (coords, coords, coords), v_computed[0],
    bounds_error=False, fill_value=0.0, method='linear')
interp_vy_computed = RegularGridInterpolator(
    (coords, coords, coords), v_computed[1],
    bounds_error=False, fill_value=0.0, method='linear')
interp_vz_computed = RegularGridInterpolator(
    (coords, coords, coords), v_computed[2],
    bounds_error=False, fill_value=0.0, method='linear')

# Build velocity interpolators for Carrick vField_0Runs
interp_vx_carrick = RegularGridInterpolator(
    (coords_c, coords_c, coords_c), v_carrick[0],
    bounds_error=False, fill_value=0.0, method='linear')
interp_vy_carrick = RegularGridInterpolator(
    (coords_c, coords_c, coords_c), v_carrick[1],
    bounds_error=False, fill_value=0.0, method='linear')
interp_vz_carrick = RegularGridInterpolator(
    (coords_c, coords_c, coords_c), v_carrick[2],
    bounds_error=False, fill_value=0.0, method='linear')

# Compute LOS profiles
print("Computing LOS density and velocity profiles...")
los_density_stored = np.zeros((n_clusters, n_r_los))  # From Aquila stored lumWeight
los_density_computed = np.zeros((n_clusters, n_r_los))  # From my computed weights
los_density_carrick_interp = np.zeros((n_clusters, n_r_los))  # From Carrick dField
los_velocity_carrick_interp = np.zeros((n_clusters, n_r_los))  # From Carrick vField
los_velocity_stored = np.zeros((n_clusters, n_r_los))  # From FFT of stored δ
los_velocity_computed = np.zeros((n_clusters, n_r_los))  # From FFT of computed δ

for i_cl in range(n_clusters):
    # Positions along LOS
    pos = obs + los_r[:, None] * rhat[i_cl]  # Shape (n_r, 3)

    # Interpolate density from all sources
    los_density_stored[i_cl] = interp_stored(pos)
    los_density_computed[i_cl] = interp_computed(pos)
    los_density_carrick_interp[i_cl] = interp_carrick(pos)

    # Interpolate velocity from stored δ (FFT)
    vx_s = interp_vx_stored(pos)
    vy_s = interp_vy_stored(pos)
    vz_s = interp_vz_stored(pos)
    los_velocity_stored[i_cl] = (
        vx_s * rhat[i_cl, 0] + vy_s * rhat[i_cl, 1] + vz_s * rhat[i_cl, 2]
    )

    # Interpolate velocity from computed δ (FFT)
    vx_c = interp_vx_computed(pos)
    vy_c = interp_vy_computed(pos)
    vz_c = interp_vz_computed(pos)
    los_velocity_computed[i_cl] = (
        vx_c * rhat[i_cl, 0] + vy_c * rhat[i_cl, 1] + vz_c * rhat[i_cl, 2]
    )

    # Interpolate velocity from Carrick vField_0Runs
    vx_k = interp_vx_carrick(pos)
    vy_k = interp_vy_carrick(pos)
    vz_k = interp_vz_carrick(pos)
    los_velocity_carrick_interp[i_cl] = (
        vx_k * rhat[i_cl, 0] + vy_k * rhat[i_cl, 1] + vz_k * rhat[i_cl, 2]
    )

print(f"  Computed LOS profiles for {n_clusters} clusters")

# =============================================================================
# Compute LG→CMB frame correction for each cluster direction
# =============================================================================
# LG apex unit vector
l_apex_rad = np.deg2rad(L_APEX)
b_apex_rad = np.deg2rad(B_APEX)
apex_rhat = np.array([
    np.cos(b_apex_rad) * np.cos(l_apex_rad),
    np.cos(b_apex_rad) * np.sin(l_apex_rad),
    np.sin(b_apex_rad)
])

# Cluster unit vectors
cluster_l_rad = np.deg2rad(cluster_l)
cluster_b_rad = np.deg2rad(cluster_b)
cluster_rhat = np.stack([
    np.cos(cluster_b_rad) * np.cos(cluster_l_rad),
    np.cos(cluster_b_rad) * np.sin(cluster_l_rad),
    np.sin(cluster_b_rad)
], axis=1)

# Angular distance from apex
cos_apex_dist = np.dot(cluster_rhat, apex_rhat)
apex_dist_deg = np.rad2deg(np.arccos(np.clip(cos_apex_dist, -1, 1)))

# Projection of LG velocity onto each cluster LOS: delta_r = v_LG·r̂ / H0  (Mpc/h)
v_lg_proj = V_LG * np.dot(cluster_rhat, apex_rhat)  # km/s per cluster
delta_r_lg = v_lg_proj / H0
print(f"\nLG→CMB frame correction:")
print(f"  delta_r range: [{delta_r_lg.min():.2f}, {delta_r_lg.max():.2f}] Mpc/h")
print(f"  Apex clusters (θ<30°): shift right by ~{delta_r_lg[apex_dist_deg < 30].mean():.1f} Mpc/h")
print(f"  Anti-apex clusters (θ>150°): shift left by ~{delta_r_lg[apex_dist_deg > 150].mean():.1f} Mpc/h")

# =============================================================================
# Select random sample of clusters for individual LOS plots
# =============================================================================
np.random.seed(42)
n_sample = 12
sample_idx = np.random.choice(n_clusters, n_sample, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, idx in enumerate(sample_idx):
    ax = axes[i]
    dr = delta_r_lg[idx]  # Frame correction for this cluster
    # Carrick LOS file (CMB frame) - no shift needed
    ax.plot(los_r, los_density_carrick[idx], 'g-', lw=2, label='Carrick LOS (CMB)')
    # Our profiles - LG frame (dashed, faint)
    ax.plot(los_r, 1 + los_density_carrick_interp[idx], 'c--', lw=1.5, alpha=0.4, label='dField interp (LG)')
    ax.plot(los_r, 1 + los_density_stored[idx], 'b--', lw=1.5, alpha=0.4, label='Aquila (LG)')
    ax.plot(los_r, 1 + los_density_computed[idx], 'r--', lw=1.5, alpha=0.4, label='My weights (LG)')
    # Our profiles - shifted to CMB frame (solid)
    ax.plot(los_r + dr, 1 + los_density_carrick_interp[idx], 'c-', lw=2, alpha=0.8, label='dField interp (CMB)')
    ax.plot(los_r + dr, 1 + los_density_stored[idx], 'b-', lw=2, alpha=0.8, label='Aquila (CMB)')
    ax.plot(los_r + dr, 1 + los_density_computed[idx], 'r-', lw=2, alpha=0.8, label='My weights (CMB)')
    ax.axhline(1, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel(r'$1 + \delta$')
    ax.set_title(f'Cluster {idx}: l={cluster_l[idx]:.1f}°, b={cluster_b[idx]:.1f}°, Δr={dr:.1f}')
    ax.set_xlim(0, 200)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=5, ncol=2)

plt.suptitle('LOS Density: Dashed=LG frame, Solid=CMB frame', fontsize=12)
plt.tight_layout()
plt.savefig('los_comparison_clusters.png', dpi=150)
print("\nSaved los_comparison_clusters.png")

# =============================================================================
# LOS velocity comparison for same clusters
# =============================================================================
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, idx in enumerate(sample_idx):
    ax = axes[i]
    dr = delta_r_lg[idx]  # Frame correction for this cluster
    # Carrick LOS file (CMB frame) - no shift needed
    ax.plot(los_r, los_velocity_carrick[idx], 'g-', lw=2, label='Carrick LOS (CMB)')
    # Our profiles - LG frame (dashed, faint)
    ax.plot(los_r, los_velocity_carrick_interp[idx], 'c--', lw=1.5, alpha=0.4, label='vField interp (LG)')
    ax.plot(los_r, los_velocity_stored[idx], 'b--', lw=1.5, alpha=0.4, label='Aquila (LG)')
    ax.plot(los_r, los_velocity_computed[idx], 'r--', lw=1.5, alpha=0.4, label='My weights (LG)')
    # Our profiles - shifted to CMB frame (solid)
    ax.plot(los_r + dr, los_velocity_carrick_interp[idx], 'c-', lw=2, alpha=0.8, label='vField interp (CMB)')
    ax.plot(los_r + dr, los_velocity_stored[idx], 'b-', lw=2, alpha=0.8, label='Aquila (CMB)')
    ax.plot(los_r + dr, los_velocity_computed[idx], 'r-', lw=2, alpha=0.8, label='My weights (CMB)')
    ax.axhline(0, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel(r'$v_{los}$ [km/s]')
    ax.set_title(f'Cluster {idx}: l={cluster_l[idx]:.1f}°, b={cluster_b[idx]:.1f}°, Δr={dr:.1f}')
    ax.set_xlim(0, 200)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=5, ncol=2)

plt.suptitle('LOS Velocity: Dashed=LG frame, Solid=CMB frame', fontsize=12)
plt.tight_layout()
plt.savefig('los_velocity_clusters.png', dpi=150)
print("Saved los_velocity_clusters.png")

# =============================================================================
# Compute per-cluster MSE and find worst clusters
# =============================================================================
print("\nComputing per-cluster MSE...")

# MSE WITHOUT frame correction (LG frame comparison)
mse_density_lg = np.mean((los_density_carrick - (1 + los_density_computed))**2, axis=1)
mse_velocity_lg = np.mean((los_velocity_carrick - los_velocity_computed)**2, axis=1)

# MSE WITH frame correction (CMB frame comparison)
# Need to interpolate shifted profiles back to los_r grid
from scipy.interpolate import interp1d
mse_density_cmb = np.zeros(n_clusters)
mse_velocity_cmb = np.zeros(n_clusters)

for i_cl in range(n_clusters):
    dr = delta_r_lg[i_cl]
    r_shifted = los_r + dr

    # Interpolate our profiles (shifted to CMB) back to los_r grid
    # Use linear interpolation, extrapolate with boundary values
    f_den = interp1d(r_shifted, 1 + los_density_computed[i_cl],
                     kind='linear', bounds_error=False, fill_value='extrapolate')
    f_vel = interp1d(r_shifted, los_velocity_computed[i_cl],
                     kind='linear', bounds_error=False, fill_value='extrapolate')

    den_cmb = f_den(los_r)
    vel_cmb = f_vel(los_r)

    mse_density_cmb[i_cl] = np.mean((los_density_carrick[i_cl] - den_cmb)**2)
    mse_velocity_cmb[i_cl] = np.mean((los_velocity_carrick[i_cl] - vel_cmb)**2)

# Compare LG vs CMB frame MSE
print(f"\nMSE Comparison (My weights vs Carrick LOS file):")
print(f"  Density MSE (LG frame):  mean={mse_density_lg.mean():.4f}, median={np.median(mse_density_lg):.4f}")
print(f"  Density MSE (CMB frame): mean={mse_density_cmb.mean():.4f}, median={np.median(mse_density_cmb):.4f}")
print(f"  Density improvement: {100*(1 - mse_density_cmb.mean()/mse_density_lg.mean()):.1f}%")
print(f"  Velocity MSE (LG frame):  mean={mse_velocity_lg.mean():.0f}, median={np.median(mse_velocity_lg):.0f}")
print(f"  Velocity MSE (CMB frame): mean={mse_velocity_cmb.mean():.0f}, median={np.median(mse_velocity_cmb):.0f}")
print(f"  Velocity improvement: {100*(1 - mse_velocity_cmb.mean()/mse_velocity_lg.mean()):.1f}%")

# Use CMB-frame MSE for finding worst clusters
mse_per_cluster_density = mse_density_cmb
mse_per_cluster_velocity = mse_velocity_cmb

# Find 12 worst MSE clusters for density
worst_den_idx = np.argsort(mse_per_cluster_density)[-12:][::-1]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, idx in enumerate(worst_den_idx):
    ax = axes[i]
    dr = delta_r_lg[idx]  # Frame correction for this cluster
    # Carrick LOS file (CMB frame) - no shift needed
    ax.plot(los_r, los_density_carrick[idx], 'g-', lw=2, label='Carrick LOS (CMB)')
    # Our profiles - LG frame (dashed, faint)
    ax.plot(los_r, 1 + los_density_carrick_interp[idx], 'c--', lw=1.5, alpha=0.4, label='dField interp (LG)')
    ax.plot(los_r, 1 + los_density_stored[idx], 'b--', lw=1.5, alpha=0.4, label='Aquila (LG)')
    ax.plot(los_r, 1 + los_density_computed[idx], 'r--', lw=1.5, alpha=0.4, label='My weights (LG)')
    # Our profiles - shifted to CMB frame (solid)
    ax.plot(los_r + dr, 1 + los_density_carrick_interp[idx], 'c-', lw=2, alpha=0.8, label='dField interp (CMB)')
    ax.plot(los_r + dr, 1 + los_density_stored[idx], 'b-', lw=2, alpha=0.8, label='Aquila (CMB)')
    ax.plot(los_r + dr, 1 + los_density_computed[idx], 'r-', lw=2, alpha=0.8, label='My weights (CMB)')
    ax.axhline(1, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel(r'$1 + \delta$')
    ax.set_title(f'Cluster {idx}: l={cluster_l[idx]:.1f}°, b={cluster_b[idx]:.1f}°\n'
                f'MSE={mse_per_cluster_density[idx]:.4f}, apex={apex_dist_deg[idx]:.0f}°, Δr={dr:.1f}')
    ax.set_xlim(0, 200)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=5, ncol=2)

plt.suptitle('LOS Density: 12 WORST MSE Clusters - Dashed=LG, Solid=CMB', fontsize=14)
plt.tight_layout()
plt.savefig('los_density_worst.png', dpi=150)
plt.close()
print("Saved los_density_worst.png")

# Find 12 worst MSE clusters for velocity
worst_vel_idx = np.argsort(mse_per_cluster_velocity)[-12:][::-1]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, idx in enumerate(worst_vel_idx):
    ax = axes[i]
    dr = delta_r_lg[idx]  # Frame correction for this cluster
    # Carrick LOS file (CMB frame) - no shift needed
    ax.plot(los_r, los_velocity_carrick[idx], 'g-', lw=2, label='Carrick LOS (CMB)')
    # Our profiles - LG frame (dashed, faint)
    ax.plot(los_r, los_velocity_carrick_interp[idx], 'c--', lw=1.5, alpha=0.4, label='vField interp (LG)')
    ax.plot(los_r, los_velocity_stored[idx], 'b--', lw=1.5, alpha=0.4, label='Aquila (LG)')
    ax.plot(los_r, los_velocity_computed[idx], 'r--', lw=1.5, alpha=0.4, label='My weights (LG)')
    # Our profiles - shifted to CMB frame (solid)
    ax.plot(los_r + dr, los_velocity_carrick_interp[idx], 'c-', lw=2, alpha=0.8, label='vField interp (CMB)')
    ax.plot(los_r + dr, los_velocity_stored[idx], 'b-', lw=2, alpha=0.8, label='Aquila (CMB)')
    ax.plot(los_r + dr, los_velocity_computed[idx], 'r-', lw=2, alpha=0.8, label='My weights (CMB)')
    ax.axhline(0, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel(r'$v_{los}$ [km/s]')
    ax.set_title(f'Cluster {idx}: l={cluster_l[idx]:.1f}°, b={cluster_b[idx]:.1f}°\n'
                f'MSE={mse_per_cluster_velocity[idx]:.0f}, apex={apex_dist_deg[idx]:.0f}°, Δr={dr:.1f}')
    ax.set_xlim(0, 200)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=5, ncol=2)

plt.suptitle('LOS Velocity: 12 WORST MSE Clusters - Dashed=LG, Solid=CMB', fontsize=14)
plt.tight_layout()
plt.savefig('los_velocity_worst.png', dpi=150)
plt.close()
print("Saved los_velocity_worst.png")

# =============================================================================
# Radially binned comparison plots
# =============================================================================
print("\nComputing radially binned statistics...")

# Bin the LOS data by radial distance
r_bin_edges = np.linspace(0, 200, 21)
r_bin_cen = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])

# For each radial bin, compute mean and std across all clusters and LOS points
mean_delta_carrick_los = np.zeros(len(r_bin_cen))
mean_delta_stored = np.zeros(len(r_bin_cen))
mean_delta_computed = np.zeros(len(r_bin_cen))
std_delta_carrick_los = np.zeros(len(r_bin_cen))
std_delta_stored = np.zeros(len(r_bin_cen))
std_delta_computed = np.zeros(len(r_bin_cen))

mean_vel_carrick_los = np.zeros(len(r_bin_cen))
mean_vel_stored = np.zeros(len(r_bin_cen))
mean_vel_computed = np.zeros(len(r_bin_cen))
std_vel_carrick_los = np.zeros(len(r_bin_cen))
std_vel_stored = np.zeros(len(r_bin_cen))
std_vel_computed = np.zeros(len(r_bin_cen))

for i in range(len(r_bin_cen)):
    mask = (los_r >= r_bin_edges[i]) & (los_r < r_bin_edges[i+1])
    if mask.sum() == 0:
        continue

    # Flatten across all clusters for this radial bin
    # Carrick LOS file stores 1+δ, convert to δ for comparison
    d_carrick = los_density_carrick[:, mask].flatten() - 1  # Convert to δ
    d_stored = los_density_stored[:, mask].flatten()
    d_computed = los_density_computed[:, mask].flatten()

    v_carrick = los_velocity_carrick[:, mask].flatten()
    v_stored = los_velocity_stored[:, mask].flatten()
    v_computed = los_velocity_computed[:, mask].flatten()

    mean_delta_carrick_los[i] = np.mean(d_carrick)
    mean_delta_stored[i] = np.mean(d_stored)
    mean_delta_computed[i] = np.mean(d_computed)
    std_delta_carrick_los[i] = np.std(d_carrick)
    std_delta_stored[i] = np.std(d_stored)
    std_delta_computed[i] = np.std(d_computed)

    mean_vel_carrick_los[i] = np.mean(v_carrick)
    mean_vel_stored[i] = np.mean(v_stored)
    mean_vel_computed[i] = np.mean(v_computed)
    std_vel_carrick_los[i] = np.std(v_carrick)
    std_vel_stored[i] = np.std(v_stored)
    std_vel_computed[i] = np.std(v_computed)

# Plot radially binned comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mean density
ax = axes[0, 0]
ax.plot(r_bin_cen, mean_delta_carrick_los, 'g-', lw=2, label='Carrick LOS file')
ax.plot(r_bin_cen, mean_delta_stored, 'b-', lw=2, label='Aquila lumWeight')
ax.plot(r_bin_cen, mean_delta_computed, 'r--', lw=2, label='My weights')
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.set_xlabel('r [Mpc/h]')
ax.set_ylabel(r'$\langle\delta\rangle$')
ax.set_title('Mean LOS overdensity (all clusters)')
ax.legend()
ax.grid(True, alpha=0.3)

# Density scatter
ax = axes[0, 1]
ax.fill_between(r_bin_cen,
                mean_delta_carrick_los - std_delta_carrick_los,
                mean_delta_carrick_los + std_delta_carrick_los,
                alpha=0.2, color='green', label='Carrick ±1σ')
ax.fill_between(r_bin_cen,
                mean_delta_stored - std_delta_stored,
                mean_delta_stored + std_delta_stored,
                alpha=0.2, color='blue', label='Aquila ±1σ')
ax.fill_between(r_bin_cen,
                mean_delta_computed - std_delta_computed,
                mean_delta_computed + std_delta_computed,
                alpha=0.2, color='red', label='My weights ±1σ')
ax.plot(r_bin_cen, mean_delta_carrick_los, 'g-', lw=2)
ax.plot(r_bin_cen, mean_delta_stored, 'b-', lw=2)
ax.plot(r_bin_cen, mean_delta_computed, 'r--', lw=2)
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.set_xlabel('r [Mpc/h]')
ax.set_ylabel(r'$\delta$')
ax.set_title('LOS overdensity scatter (all clusters)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Mean velocity
ax = axes[1, 0]
ax.plot(r_bin_cen, mean_vel_carrick_los, 'g-', lw=2, label='Carrick LOS file')
ax.plot(r_bin_cen, mean_vel_stored, 'b-', lw=2, label='Aquila lumWeight')
ax.plot(r_bin_cen, mean_vel_computed, 'r--', lw=2, label='My weights')
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.set_xlabel('r [Mpc/h]')
ax.set_ylabel(r'$\langle v_{los}\rangle$ [km/s]')
ax.set_title('Mean LOS velocity (all clusters)')
ax.legend()
ax.grid(True, alpha=0.3)

# Velocity scatter
ax = axes[1, 1]
ax.fill_between(r_bin_cen,
                mean_vel_carrick_los - std_vel_carrick_los,
                mean_vel_carrick_los + std_vel_carrick_los,
                alpha=0.2, color='green', label='Carrick ±1σ')
ax.fill_between(r_bin_cen,
                mean_vel_stored - std_vel_stored,
                mean_vel_stored + std_vel_stored,
                alpha=0.2, color='blue', label='Aquila ±1σ')
ax.fill_between(r_bin_cen,
                mean_vel_computed - std_vel_computed,
                mean_vel_computed + std_vel_computed,
                alpha=0.2, color='red', label='My weights ±1σ')
ax.plot(r_bin_cen, mean_vel_carrick_los, 'g-', lw=2)
ax.plot(r_bin_cen, mean_vel_stored, 'b-', lw=2)
ax.plot(r_bin_cen, mean_vel_computed, 'r--', lw=2)
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.set_xlabel('r [Mpc/h]')
ax.set_ylabel(r'$v_{los}$ [km/s]')
ax.set_title('LOS velocity scatter (all clusters)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('los_comparison.png', dpi=150)
print("Saved los_comparison.png")

# =============================================================================
# 1:1 scatter plots at specific radii
# =============================================================================
print("\nCreating 1:1 scatter plots...")

# Select specific radii for comparison
r_compare = [50, 100, 150]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, r_val in enumerate(r_compare):
    # Find closest r index
    r_idx = np.argmin(np.abs(los_r - r_val))
    actual_r = los_r[r_idx]

    # Carrick LOS file stores 1+δ, convert to δ
    d_carrick_r = los_density_carrick[:, r_idx] - 1
    d_stored_r = los_density_stored[:, r_idx]
    d_computed_r = los_density_computed[:, r_idx]

    # Top row: Aquila lumWeight vs Carrick
    ax = axes[0, i]
    ax.scatter(d_carrick_r, d_stored_r, s=10, alpha=0.5, c='blue')
    lim = max(np.abs(d_carrick_r).max(), np.abs(d_stored_r).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=2, label='1:1')
    ax.set_xlabel(r'$\delta$ (Carrick LOS file)')
    ax.set_ylabel(r'$\delta$ (Aquila lumWeight)')
    ax.set_title(f'r={actual_r:.0f} Mpc/h')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    corr = np.corrcoef(d_carrick_r, d_stored_r)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')

    # Bottom row: My weights vs Carrick
    ax = axes[1, i]
    ax.scatter(d_carrick_r, d_computed_r, s=10, alpha=0.5, c='red')
    lim = max(np.abs(d_carrick_r).max(), np.abs(d_computed_r).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=2, label='1:1')
    ax.set_xlabel(r'$\delta$ (Carrick LOS file)')
    ax.set_ylabel(r'$\delta$ (My weights)')
    ax.set_title(f'r={actual_r:.0f} Mpc/h')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    corr = np.corrcoef(d_carrick_r, d_computed_r)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')

plt.suptitle('LOS Density: Aquila lumWeight (top) vs My weights (bottom)', fontsize=14)
plt.tight_layout()
plt.savefig('los_1to1_comparison.png', dpi=150)
print("Saved los_1to1_comparison.png")

print("\n" + "="*60)
print("All comparisons complete!")
print("="*60)
