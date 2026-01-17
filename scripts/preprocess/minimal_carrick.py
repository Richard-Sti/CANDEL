"""
Minimal Carrick reconstruction script.
Builds separate density fields for deep and 2MRS regions.
"""
import numpy as np
import healpy as hp
from scipy import special as sps
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# =============================================================================
# Parameters
# =============================================================================
DATA_DIR = "data/2M++"
N = 257
BOX_SIDE = 400.0
RMAX = BOX_SIDE / 2.0
R_2MRS_CUTOFF = 125.0
CMIN = 0.5
H0 = 100.0
C_LIGHT = 299792.458

# Schechter LF parameters
ALPHA = -0.85
MSTAR = -23.25
M_MIN = -20.0

# =============================================================================
# Load galaxies
# =============================================================================
print("Loading galaxies...")
with open(f"{DATA_DIR}/2m++.txt") as f:
    lines = f.readlines()

header_idx = next(i for i, line in enumerate(lines) if "Designation" in line and "|" in line)
data = np.genfromtxt(f"{DATA_DIR}/2m++.txt", delimiter="|", skip_header=header_idx + 2, usecols=[3, 4, 5, 7, 9])

l_deg = data[:, 0]
b_deg = data[:, 1]
K2Mpp = data[:, 2]
vcmb = data[:, 3]
gid = np.where(np.isfinite(data[:, 4]), data[:, 4], -1).astype(int)

# Filter valid
valid = np.isfinite(l_deg) & np.isfinite(b_deg) & np.isfinite(vcmb) & np.isfinite(K2Mpp) & (vcmb > 0)
l_deg, b_deg, K2Mpp, vcmb, gid = l_deg[valid], b_deg[valid], K2Mpp[valid], vcmb[valid], gid[valid]
print(f"  {len(vcmb)} galaxies after filtering")

# =============================================================================
# Compute L/L* from ORIGINAL positions (before group centering)
# =============================================================================
print("Computing L/L* from original positions...")
vcmb_orig = vcmb.copy()
z_orig = vcmb_orig / C_LIGHT
r_orig = (C_LIGHT / H0) * z_orig
r_orig_safe = np.maximum(r_orig, 0.1)
mu_orig = 5.0 * np.log10(r_orig_safe) + 25.0
M_abs_orig = K2Mpp - mu_orig
L_over_Lstar = 10.0 ** (-0.4 * (M_abs_orig - MSTAR))

# =============================================================================
# Load groups and collapse FoF
# =============================================================================
print("Loading groups...")
groups = np.genfromtxt(f"{DATA_DIR}/2m++_groups.txt", delimiter="|", skip_header=12, usecols=[0, 1, 2, 6])
group_gid = groups[:, 0].astype(int)
group_l, group_b, group_v = groups[:, 1], groups[:, 2], groups[:, 3]
group_dict = {gid: i for i, gid in enumerate(group_gid)}

for i in range(len(vcmb)):
    if gid[i] >= 0 and gid[i] in group_dict:
        idx = group_dict[gid[i]]
        l_deg[i], b_deg[i], vcmb[i] = group_l[idx], group_b[idx], group_v[idx]

valid = vcmb > 0
l_deg, b_deg, K2Mpp, vcmb, gid = l_deg[valid], b_deg[valid], K2Mpp[valid], vcmb[valid], gid[valid]
L_over_Lstar = L_over_Lstar[valid]  # Keep original L/L*
print(f"  {len(vcmb)} galaxies after FoF collapse")

# =============================================================================
# Load completeness maps and assign m_lim
# =============================================================================
print("Loading completeness maps...")
map11 = hp.read_map(f"{DATA_DIR}/incompleteness_11_5.fits")
map12 = hp.read_map(f"{DATA_DIR}/incompleteness_12_5.fits")
print(f"  Map NSIDE: {hp.get_nside(map12)}, NPIX: {len(map12)}")

# Convert galactic (l, b) to equatorial (RA, DEC) for HEALPix query
# The completeness maps appear to be in equatorial coordinates
from astropy.coordinates import SkyCoord
import astropy.units as u

coords_gal = SkyCoord(l=l_deg*u.deg, b=b_deg*u.deg, frame='galactic')
coords_eq = coords_gal.icrs
ra_deg = coords_eq.ra.deg
dec_deg = coords_eq.dec.deg

# HEALPix in equatorial: theta = 90° - dec, phi = ra
theta = np.deg2rad(90.0 - dec_deg)
phi = np.deg2rad(ra_deg)
pix11 = hp.ang2pix(hp.get_nside(map11), theta, phi)
pix12 = hp.ang2pix(hp.get_nside(map12), theta, phi)

c11 = map11[pix11]
c12 = map12[pix12]
use_deep = c12 > 0

m_lim = np.where(use_deep, 12.5, 11.5)
comp = np.where(use_deep, c12, c11)
comp = np.where(np.isfinite(comp) & (comp > 0), comp, CMIN)
comp = np.clip(comp, CMIN, 1.0)

print(f"  {np.sum(~use_deep)} in 2MRS-only, {np.sum(use_deep)} in deep regions")

# Debug: check where deep regions are in galactic coords
deep_mean_l = np.mean(l_deg[use_deep])
deep_mean_b = np.mean(b_deg[use_deep])
mrs_mean_l = np.mean(l_deg[~use_deep])
mrs_mean_b = np.mean(b_deg[~use_deep])
print(f"  Deep mean (l,b): ({deep_mean_l:.1f}, {deep_mean_b:.1f})")
print(f"  2MRS mean (l,b): ({mrs_mean_l:.1f}, {mrs_mean_b:.1f})")

# Plot the completeness map to verify angular positions
hp.mollview(map12, title="Deep completeness map (map12)", coord='G')
plt.savefig('completeness_map12.png', dpi=100)
plt.close()
print("  Saved completeness_map12.png")


# =============================================================================
# Compute distances
# =============================================================================
z_obs = vcmb / C_LIGHT
r_mpc = (C_LIGHT / H0) * z_obs

# =============================================================================
# Compute weights
# =============================================================================
print("Computing weights...")

# Angular weight
w_ang = 1.0 / comp

# Selection function S(r) = Gamma(alpha+2, x_lim) / Gamma(alpha+2, x_min)
def gamma_upper(s, x):
    """Upper incomplete gamma function."""
    if s > 0:
        return sps.gammaincc(s, x) * sps.gamma(s)
    k = int(np.ceil(1.0 - s))
    sp = s + k
    result = sps.gammaincc(sp, x) * sps.gamma(sp)
    for _ in range(k):
        result = (result - np.power(x, sp - 1) * np.exp(-x)) / (sp - 1)
        sp -= 1
    return result

# Selection function uses GROUP position (where galaxy is deposited)
r_safe = np.maximum(r_mpc, 0.1)
mu = 5.0 * np.log10(r_safe) + 25.0
M_lim = m_lim - mu

x_lim = 10.0 ** (-0.4 * (M_lim - MSTAR))
x_min = 10.0 ** (-0.4 * (M_MIN - MSTAR))

a = ALPHA + 2.0
S_r = gamma_upper(a, x_lim) / gamma_upper(a, x_min)
S_r = np.clip(S_r, 1e-6, 1.0)

w_L = 1.0 / S_r

# L_over_Lstar already computed from ORIGINAL positions (before group centering)
# Total weight
w_total = w_ang * w_L * L_over_Lstar

# =============================================================================
# Compute positions
# =============================================================================
ell_rad = np.deg2rad(l_deg)
b_rad = np.deg2rad(b_deg)
cos_b = np.cos(b_rad)
x = r_mpc * cos_b * np.cos(ell_rad)
y = r_mpc * cos_b * np.sin(ell_rad)
z = r_mpc * np.sin(b_rad)

dx = BOX_SIDE / N
ix = np.floor((x + RMAX) / dx).astype(int)
iy = np.floor((y + RMAX) / dx).astype(int)
iz = np.floor((z + RMAX) / dx).astype(int)

# =============================================================================
# Deposit DEEP region
# =============================================================================
print("Depositing DEEP region...")
w_deep = w_total.copy()
w_deep[~use_deep] = 0.0  # Zero 2MRS galaxies
w_deep[r_mpc > RMAX] = 0.0

valid_deep = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (iz >= 0) & (iz < N) & (w_deep > 0)
rho_deep = np.zeros((N, N, N), dtype=np.float64)
np.add.at(rho_deep, (ix[valid_deep], iy[valid_deep], iz[valid_deep]), w_deep[valid_deep])
print(f"  Deposited {np.sum(valid_deep)} deep galaxies")

# =============================================================================
# Deposit 2MRS region
# =============================================================================
print("Depositing 2MRS region...")
w_2mrs = w_total.copy()
w_2mrs[use_deep] = 0.0  # Zero deep galaxies
w_2mrs[r_mpc > R_2MRS_CUTOFF] = 0.0  # Zero beyond 125 Mpc/h
w_2mrs[r_mpc > RMAX] = 0.0

valid_2mrs = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (iz >= 0) & (iz < N) & (w_2mrs > 0)
rho_2mrs = np.zeros((N, N, N), dtype=np.float64)
np.add.at(rho_2mrs, (ix[valid_2mrs], iy[valid_2mrs], iz[valid_2mrs]), w_2mrs[valid_2mrs])
print(f"  Deposited {np.sum(valid_2mrs)} 2MRS galaxies")

# =============================================================================
# Gaussian smoothing
# =============================================================================
print("Applying Gaussian smoothing (4 Mpc/h)...")
sigma_voxels = 4.0 / dx
rho_deep = gaussian_filter(rho_deep, sigma=sigma_voxels, mode='constant')
rho_2mrs = gaussian_filter(rho_2mrs, sigma=sigma_voxels, mode='constant')

# =============================================================================
# Build 3D masks
# =============================================================================
print("Building 3D masks...")
coords = np.linspace(-RMAX + dx/2, RMAX - dx/2, N)
xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
rr = np.sqrt(xx**2 + yy**2 + zz**2)

rr_safe = np.maximum(rr, 1e-10)
# In galactic cartesian: x->GC(l=0), y->l=90, z->NGP(b=90)
# b = arcsin(z/r), l = atan2(y, x)
gal_b_3d = np.arcsin(zz / rr_safe)
gal_l_3d = np.arctan2(yy, xx)
gal_l_3d = np.where(gal_l_3d < 0, gal_l_3d + 2*np.pi, gal_l_3d)

# Convert galactic (l, b) to equatorial (RA, DEC) for HEALPix query
# The completeness maps might be in equatorial coordinates
from astropy.coordinates import SkyCoord
import astropy.units as u

print("Converting galactic to equatorial for HEALPix query...")
gal_l_deg_3d = np.rad2deg(gal_l_3d)
gal_b_deg_3d = np.rad2deg(gal_b_3d)
coords_gal = SkyCoord(l=gal_l_deg_3d.ravel()*u.deg, b=gal_b_deg_3d.ravel()*u.deg, frame='galactic')
coords_eq = coords_gal.icrs
ra_3d = coords_eq.ra.deg.reshape(rr.shape)
dec_3d = coords_eq.dec.deg.reshape(rr.shape)

# HEALPix in equatorial: theta = 90° - dec, phi = ra
theta_3d = np.deg2rad(90.0 - dec_3d)
phi_3d = np.deg2rad(ra_3d)
pix_3d = hp.ang2pix(hp.get_nside(map12), theta_3d.ravel(), phi_3d.ravel()).reshape(rr.shape)

is_2mrs_only_3d = map12[pix_3d] <= 0
is_deep_3d = ~is_2mrs_only_3d

# Debug: check if 3D grid mapping is correct
print("Debug: checking 3D grid -> galactic (l, b) -> equatorial (RA, DEC) -> deep")
test_indices = [
    (N//2 + 50, N//2, N//2),  # x=positive, y=0, z=0 -> l=0
    (N//2 - 50, N//2, N//2),  # x=negative, y=0, z=0 -> l=180
    (N//2, N//2 + 50, N//2),  # x=0, y=positive, z=0 -> l=90
    (N//2, N//2, N//2 + 50),  # x=0, y=0, z=positive -> b=90
]
for (i, j, k) in test_indices:
    x_val, y_val, z_val = xx[i,j,k], yy[i,j,k], zz[i,j,k]
    l = gal_l_deg_3d[i,j,k]
    b = gal_b_deg_3d[i,j,k]
    ra = ra_3d[i,j,k]
    dec = dec_3d[i,j,k]
    deep = is_deep_3d[i,j,k]
    print(f"  grid[{i},{j},{k}] = ({x_val:.0f},{y_val:.0f},{z_val:.0f}) -> (l={l:.0f}, b={b:.0f}) -> (RA={ra:.0f}, DEC={dec:.0f}) -> deep={deep}")

# Masks for normalization and profiles
valid_deep_3d = (rr <= RMAX) & is_deep_3d
valid_2mrs_3d = (rr <= R_2MRS_CUTOFF) & is_2mrs_only_3d
valid_2mrs_outer_3d = (rr > R_2MRS_CUTOFF) & (rr <= RMAX) & is_2mrs_only_3d

# =============================================================================
# Normalize
# =============================================================================
print("Normalizing...")
rho_deep_mean = np.mean(rho_deep[valid_deep_3d])
rho_2mrs_mean = np.mean(rho_2mrs[valid_2mrs_3d])

print(f"  Deep mean: {rho_deep_mean:.6f}")
print(f"  2MRS mean (r<=125): {rho_2mrs_mean:.6f}")

# Set 2MRS outer region to mean (so normalized = 1)
rho_2mrs[valid_2mrs_outer_3d] = rho_2mrs_mean

# =============================================================================
# Compute radial profiles
# =============================================================================
print("Computing radial profiles...")
r_edges = np.linspace(0, RMAX, 41)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

mean_rho_deep = np.zeros(len(r_centers))
mean_rho_2mrs = np.zeros(len(r_centers))
std_rho_deep = np.zeros(len(r_centers))
std_rho_2mrs = np.zeros(len(r_centers))

for i in range(len(r_centers)):
    # Deep region profile
    shell_deep = (rr >= r_edges[i]) & (rr < r_edges[i+1]) & is_deep_3d & (rr <= RMAX)
    if np.any(shell_deep):
        mean_rho_deep[i] = np.mean(rho_deep[shell_deep])
        std_rho_deep[i] = np.std(rho_deep[shell_deep])

    # 2MRS region profile
    shell_2mrs = (rr >= r_edges[i]) & (rr < r_edges[i+1]) & is_2mrs_only_3d & (rr <= RMAX)
    if np.any(shell_2mrs):
        mean_rho_2mrs[i] = np.mean(rho_2mrs[shell_2mrs])
        std_rho_2mrs[i] = np.std(rho_2mrs[shell_2mrs])

# Load Carrick for comparison
print("Loading Carrick field...")
delta_carrick = np.load("data/fields/carrick2015_twompp_density.npy")
rho_carrick = 1.0 + delta_carrick

mean_rho_carrick_deep = np.zeros(len(r_centers))
mean_rho_carrick_2mrs = np.zeros(len(r_centers))
std_rho_carrick_deep = np.zeros(len(r_centers))
std_rho_carrick_2mrs = np.zeros(len(r_centers))

for i in range(len(r_centers)):
    shell_deep = (rr >= r_edges[i]) & (rr < r_edges[i+1]) & is_deep_3d & (rr <= RMAX)
    if np.any(shell_deep):
        mean_rho_carrick_deep[i] = np.mean(rho_carrick[shell_deep])
        std_rho_carrick_deep[i] = np.std(rho_carrick[shell_deep])

    shell_2mrs = (rr >= r_edges[i]) & (rr < r_edges[i+1]) & is_2mrs_only_3d & (rr <= RMAX)
    if np.any(shell_2mrs):
        mean_rho_carrick_2mrs[i] = np.mean(rho_carrick[shell_2mrs])
        std_rho_carrick_2mrs[i] = np.std(rho_carrick[shell_2mrs])

# =============================================================================
# Plot
# =============================================================================
print("Plotting...")
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Normalize to mean=1
norm_deep = rho_deep_mean
norm_2mrs = rho_2mrs_mean
norm_carrick_deep = np.mean(mean_rho_carrick_deep[mean_rho_carrick_deep > 0])
norm_carrick_2mrs = np.mean(mean_rho_carrick_2mrs[mean_rho_carrick_2mrs > 0])

# Top panel: Mean profiles
ax = axes[0]
ax.plot(r_centers, mean_rho_deep / norm_deep, 'b-', lw=2, label='Ours - Deep')
ax.plot(r_centers, mean_rho_2mrs / norm_2mrs, 'g-', lw=2, label='Ours - 2MRS')
ax.plot(r_centers, mean_rho_carrick_deep / norm_carrick_deep, 'b--', lw=1.5, alpha=0.7, label='Carrick - Deep')
ax.plot(r_centers, mean_rho_carrick_2mrs / norm_carrick_2mrs, 'g--', lw=1.5, alpha=0.7, label='Carrick - 2MRS')

ax.axhline(1, color='gray', ls=':', alpha=0.5)
ax.axvline(R_2MRS_CUTOFF, color='orange', ls='--', lw=2, alpha=0.7, label='2MRS cutoff (125)')

ax.set_ylabel('Mean ρ / <ρ>', fontsize=12)
ax.set_title('Radial density profiles: Deep vs 2MRS regions')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 2.5)

# Bottom panel: Std profiles
ax = axes[1]
ax.plot(r_centers, std_rho_deep / norm_deep, 'b-', lw=2, label='Ours - Deep')
ax.plot(r_centers, std_rho_2mrs / norm_2mrs, 'g-', lw=2, label='Ours - 2MRS')
ax.plot(r_centers, std_rho_carrick_deep / norm_carrick_deep, 'b--', lw=1.5, alpha=0.7, label='Carrick - Deep')
ax.plot(r_centers, std_rho_carrick_2mrs / norm_carrick_2mrs, 'g--', lw=1.5, alpha=0.7, label='Carrick - 2MRS')

ax.axvline(R_2MRS_CUTOFF, color='orange', ls='--', lw=2, alpha=0.7)

ax.set_xlabel('r [Mpc/h]', fontsize=12)
ax.set_ylabel('Std ρ / <ρ>', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 200)

plt.tight_layout()
plt.savefig('minimal_carrick_profiles.png', dpi=150)
plt.close()
print("Saved minimal_carrick_profiles.png")

# =============================================================================
# Compute psi(r) for bias normalization (Carrick Eq. 8)
# =============================================================================
print("Computing psi(r) bias normalization...")

# Luminosity-dependent bias: b/b* = 0.73 + 0.24 * L/L* (Westover 2007)
B0, B1 = 0.73, 0.24

def compute_psi(r_vals, m_lim_val, alpha=ALPHA, Mstar=MSTAR, M_min=M_MIN):
    """
    Compute psi(r) = <b*L*S> / <L*S> for bias normalization.
    This is the luminosity-weighted mean bias at distance r.
    """
    psi = np.zeros_like(r_vals)

    for i, r in enumerate(r_vals):
        if r < 0.1:
            psi[i] = 1.0
            continue

        mu = 5.0 * np.log10(r) + 25.0
        M_lim_r = m_lim_val - mu

        # Integrate over luminosity using quadrature
        # L/L* from L_min to L_lim (visible galaxies)
        L_min = 10.0 ** (-0.4 * (M_min - Mstar))  # Minimum L/L* considered
        L_lim = 10.0 ** (-0.4 * (M_lim_r - Mstar))  # Limiting L/L* at this distance

        if L_lim <= L_min:
            psi[i] = 1.0
            continue

        # Use log-spaced L values for integration
        n_L = 100
        log_L = np.linspace(np.log10(L_lim), np.log10(L_min * 100), n_L)  # Up to 100 L*
        L_vals = 10.0 ** log_L
        L_vals = L_vals[L_vals >= L_lim]  # Only visible galaxies

        if len(L_vals) < 2:
            psi[i] = 1.0
            continue

        # Schechter function: phi(L) dL ~ L^alpha * exp(-L) dL
        phi = L_vals ** alpha * np.exp(-L_vals)

        # Bias: b/b* = B0 + B1 * L/L*
        b = B0 + B1 * L_vals

        # Numerator: integral of b * L * phi
        # Denominator: integral of L * phi
        numerator = np.trapz(b * L_vals * phi, L_vals)
        denominator = np.trapz(L_vals * phi, L_vals)

        if denominator > 0:
            psi[i] = numerator / denominator
        else:
            psi[i] = 1.0

    return psi

# Compute psi on radial grid
r_psi = np.linspace(0.1, RMAX, 200)
psi_deep = compute_psi(r_psi, 12.5)
psi_2mrs = compute_psi(r_psi, 11.5)

# Plot psi(r) for both cutoffs
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_psi, psi_deep, 'b-', lw=2, label=r'Deep ($m_{\rm lim}=12.5$)')
ax.plot(r_psi, psi_2mrs, 'g-', lw=2, label=r'2MRS ($m_{\rm lim}=11.5$)')
ax.axhline(1, color='gray', ls=':', alpha=0.5)
ax.axvline(R_2MRS_CUTOFF, color='orange', ls='--', lw=2, alpha=0.7, label='2MRS cutoff (125)')
ax.set_xlabel('r [Mpc/h]', fontsize=12)
ax.set_ylabel(r'$\psi(r)$', fontsize=12)
ax.set_title(r'Luminosity-weighted mean bias $\psi(r) = \langle b \cdot L \rangle / \langle L \rangle$')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 200)
plt.tight_layout()
plt.savefig('minimal_carrick_psi.png', dpi=150)
plt.close()
print("Saved minimal_carrick_psi.png")

# Apply psi(r) normalization to 3D fields
# delta* = delta / psi(r) to normalize to b* bias
print("Applying psi(r) normalization to 3D fields...")

# Interpolate psi to 3D grid
psi_deep_3d = np.interp(rr.ravel(), r_psi, psi_deep).reshape(rr.shape)
psi_2mrs_3d = np.interp(rr.ravel(), r_psi, psi_2mrs).reshape(rr.shape)

# Convert rho to delta and apply psi normalization
delta_deep = (rho_deep / rho_deep_mean - 1.0) / psi_deep_3d
delta_2mrs = (rho_2mrs / rho_2mrs_mean - 1.0) / psi_2mrs_3d

# Set delta=0 beyond 200 Mpc/h
delta_deep[rr > RMAX] = 0.0
delta_2mrs[rr > RMAX] = 0.0

# Check delta ranges
print(f"Delta deep: min={np.min(delta_deep):.2f}, max={np.max(delta_deep):.2f}, std={np.std(delta_deep):.3f}")
print(f"Delta 2MRS: min={np.min(delta_2mrs):.2f}, max={np.max(delta_2mrs):.2f}, std={np.std(delta_2mrs):.3f}")
print(f"Carrick:    min={np.min(delta_carrick):.2f}, max={np.max(delta_carrick):.2f}, std={np.std(delta_carrick):.3f}")

# Carrick is already delta (normalized to b*)
delta_carrick_field = delta_carrick  # Already loaded as delta

# =============================================================================
# Plot Supergalactic Plane slice
# =============================================================================
print("Plotting Supergalactic Plane slice...")
from scipy.interpolate import RegularGridInterpolator
from astropy.coordinates import CartesianRepresentation, Galactic, Supergalactic
import astropy.units as u

# Rotation matrix from Supergalactic to Galactic cartesian
sg_x = Supergalactic(CartesianRepresentation(1, 0, 0, unit=u.Mpc))
sg_y = Supergalactic(CartesianRepresentation(0, 1, 0, unit=u.Mpc))
sg_z = Supergalactic(CartesianRepresentation(0, 0, 1, unit=u.Mpc))

gal_x = sg_x.transform_to(Galactic()).cartesian
gal_y = sg_y.transform_to(Galactic()).cartesian
gal_z = sg_z.transform_to(Galactic()).cartesian

R_sg2gal = np.array([
    [gal_x.x.value, gal_y.x.value, gal_z.x.value],
    [gal_x.y.value, gal_y.y.value, gal_z.y.value],
    [gal_x.z.value, gal_y.z.value, gal_z.z.value],
])

# Create interpolators
interp_deep = RegularGridInterpolator((coords, coords, coords), delta_deep,
                                       bounds_error=False, fill_value=np.nan)
interp_2mrs = RegularGridInterpolator((coords, coords, coords), delta_2mrs,
                                       bounds_error=False, fill_value=np.nan)
interp_carrick = RegularGridInterpolator((coords, coords, coords), delta_carrick_field,
                                          bounds_error=False, fill_value=np.nan)
interp_is_deep = RegularGridInterpolator((coords, coords, coords), is_deep_3d.astype(float),
                                          bounds_error=False, fill_value=0.0)

# Grid in supergalactic plane (SGZ=0)
sg_extent = 200.0
n_pix = 400
sg_coords = np.linspace(-sg_extent, sg_extent, n_pix)
SGX, SGY = np.meshgrid(sg_coords, sg_coords, indexing='ij')
SGZ = np.zeros_like(SGX)

sg_pos = np.stack([SGX, SGY, SGZ], axis=-1)
gal_pos = np.einsum('ij,...j->...i', R_sg2gal, sg_pos)

# Sample fields
slice_deep = interp_deep(gal_pos)
slice_2mrs = interp_2mrs(gal_pos)
slice_carrick = interp_carrick(gal_pos)
is_deep_slice = interp_is_deep(gal_pos) > 0.5

# Compute galactic (l, b) for each pixel
gal_r = np.sqrt(np.sum(gal_pos**2, axis=-1))
gal_r_safe = np.maximum(gal_r, 1e-10)
gal_b = np.arcsin(gal_pos[..., 2] / gal_r_safe)
gal_l = np.arctan2(gal_pos[..., 1], gal_pos[..., 0])
gal_l = np.where(gal_l < 0, gal_l + 2*np.pi, gal_l)

# ZoA mask
zoa_mask = np.abs(gal_b) < np.deg2rad(5.0)

# Debug: check specific SG positions
print("Debug: SG positions -> galactic coords -> deep?")
test_sg = [(100, 0), (0, 100), (-100, 0), (0, -100), (100, 100)]
for sgx, sgy in test_sg:
    # Find nearest pixel
    ix = int((sgx + sg_extent) / (2*sg_extent) * n_pix)
    iy = int((sgy + sg_extent) / (2*sg_extent) * n_pix)
    ix = max(0, min(n_pix-1, ix))
    iy = max(0, min(n_pix-1, iy))
    gp = gal_pos[ix, iy]
    l_deg = np.rad2deg(gal_l[ix, iy])
    b_deg = np.rad2deg(gal_b[ix, iy])
    deep = is_deep_slice[ix, iy]
    print(f"  SG({sgx},{sgy}) -> gal({gp[0]:.0f},{gp[1]:.0f},{gp[2]:.0f}) -> (l={l_deg:.0f},b={b_deg:.0f}) -> deep={deep}")

# Combine deep + 2MRS (use deep where available, else 2MRS)
slice_ours = np.where(is_deep_slice, slice_deep, slice_2mrs)
# Don't mask ZoA - show the full field

print(f"  Deep fraction: {np.mean(is_deep_slice):.3f}, ZoA fraction: {np.mean(zoa_mask):.3f}")

# Debug: Check how galactic coords in the SG plane compare to expectation
# The SG plane intersects the galactic plane at a ~60 degree angle
# Let's verify the rotation matrix is correct by checking key directions
print("Debug: Verifying supergalactic-galactic transformation")
# SGX should be toward SGL=0, SGB=0, which in galactic is near l=137.37°, b=0°
# SGY should be toward SGL=90, SGB=0
sg_test = [
    (100, 0, 0, "SGX (SGL=0)"),
    (0, 100, 0, "SGY (SGL=90)"),
    (-100, 0, 0, "-SGX (SGL=180)"),
    (0, -100, 0, "-SGY (SGL=270)"),
]
for sgx, sgy, sgz, name in sg_test:
    sg_vec = np.array([sgx, sgy, sgz])
    gal_vec = R_sg2gal @ sg_vec
    r = np.sqrt(np.sum(gal_vec**2))
    l = np.rad2deg(np.arctan2(gal_vec[1], gal_vec[0]))
    if l < 0:
        l += 360
    b = np.rad2deg(np.arcsin(gal_vec[2] / r))
    print(f"  {name}: gal=({gal_vec[0]:.1f},{gal_vec[1]:.1f},{gal_vec[2]:.1f}) -> l={l:.1f}, b={b:.1f}")

# Plot 4-panel: our field, Carrick, our mask in SG plane, galaxy scatter plot colored by deep/2MRS
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12))

vmin, vmax = -1.0, 2.0
extent = [-sg_extent, sg_extent, -sg_extent, sg_extent]

# 1. Our field
ax = axes4[0, 0]
im = ax.imshow(slice_ours.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax.set_xlabel('SGX [Mpc/h]')
ax.set_ylabel('SGY [Mpc/h]')
ax.set_title(r'Ours $\delta^*$')
ax.set_aspect('equal')
plt.colorbar(im, ax=ax, label=r'$\delta^*$')

# 2. Carrick field
ax = axes4[0, 1]
im = ax.imshow(slice_carrick.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax.set_xlabel('SGX [Mpc/h]')
ax.set_ylabel('SGY [Mpc/h]')
ax.set_title(r'Carrick2015 $\delta^*$')
ax.set_aspect('equal')
plt.colorbar(im, ax=ax, label=r'$\delta^*$')

# 3. Our deep/2MRS mask in SG plane
ax = axes4[1, 0]
mask_plot = np.zeros_like(is_deep_slice, dtype=float)
mask_plot[is_deep_slice] = 1.0  # Deep = red
mask_plot[~is_deep_slice] = 0.0  # 2MRS = blue
im = ax.imshow(mask_plot.T, origin='lower', extent=extent, vmin=0, vmax=1, cmap='coolwarm')
ax.set_xlabel('SGX [Mpc/h]')
ax.set_ylabel('SGY [Mpc/h]')
ax.set_title('Our mask in SG plane (red=deep, blue=2MRS)')
ax.set_aspect('equal')
plt.colorbar(im, ax=ax)

# 4. Galaxy scatter plot in SG plane, colored by deep/2MRS
# Transform galaxy positions from galactic cartesian to supergalactic cartesian
R_gal2sg = R_sg2gal.T  # Inverse is transpose for rotation matrices
gal_xyz = np.stack([x, y, z], axis=-1)  # Shape (N, 3)
sg_xyz = (R_gal2sg @ gal_xyz.T).T  # (3, 3) @ (3, N) -> (3, N) -> transpose to (N, 3)

ax = axes4[1, 1]
# Filter to near SG plane (|SGZ| < 30 Mpc/h)
near_sg_plane = np.abs(sg_xyz[:, 2]) < 30
sg_x_gal = sg_xyz[near_sg_plane, 0]
sg_y_gal = sg_xyz[near_sg_plane, 1]
use_deep_plane = use_deep[near_sg_plane]
ax.scatter(sg_x_gal[~use_deep_plane], sg_y_gal[~use_deep_plane], s=1, c='blue', alpha=0.3, label='2MRS')
ax.scatter(sg_x_gal[use_deep_plane], sg_y_gal[use_deep_plane], s=1, c='red', alpha=0.3, label='Deep')
ax.set_xlabel('SGX [Mpc/h]')
ax.set_ylabel('SGY [Mpc/h]')
ax.set_xlim(-sg_extent, sg_extent)
ax.set_ylim(-sg_extent, sg_extent)
ax.set_title('Galaxies near SG plane (|SGZ|<30)')
ax.set_aspect('equal')
ax.legend(loc='upper right', markerscale=5)

plt.tight_layout()
plt.savefig('minimal_carrick_mask_compare.png', dpi=150)
plt.close()
print("Saved minimal_carrick_mask_compare.png")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vmin, vmax = -1.0, 2.0
extent = [-sg_extent, sg_extent, -sg_extent, sg_extent]

ax = axes[0]
im = ax.imshow(slice_ours.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax.set_xlabel('SGX [Mpc/h]')
ax.set_ylabel('SGY [Mpc/h]')
ax.set_title(r'Ours $\delta^*$ (deep only)')
ax.set_aspect('equal')
plt.colorbar(im, ax=ax, label=r'$\delta^*$')

ax = axes[1]
im = ax.imshow(slice_carrick.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax.set_xlabel('SGX [Mpc/h]')
ax.set_ylabel('SGY [Mpc/h]')
ax.set_title(r'Carrick2015 $\delta^*$')
ax.set_aspect('equal')
plt.colorbar(im, ax=ax, label=r'$\delta^*$')

plt.tight_layout()
plt.savefig('minimal_carrick_SGP.png', dpi=150)
plt.close()
print("Saved minimal_carrick_SGP.png")
