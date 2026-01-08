"""
Minimal Carrick reconstruction script.
"""
import numpy as np
import healpy as hp
from scipy import special as sps
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
print(f"  {len(vcmb)} galaxies after FoF collapse")

# =============================================================================
# Load completeness maps and assign m_lim
# =============================================================================
print("Loading completeness maps...")
map11 = hp.read_map(f"{DATA_DIR}/incompleteness_11_5.fits")
map12 = hp.read_map(f"{DATA_DIR}/incompleteness_12_5.fits")

theta = np.deg2rad(90.0 - b_deg)
phi = np.deg2rad(l_deg)
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

r_safe = np.maximum(r_mpc, 0.1)
mu = 5.0 * np.log10(r_safe) + 25.0
M_lim = m_lim - mu

x_lim = 10.0 ** (-0.4 * (M_lim - MSTAR))
x_min = 10.0 ** (-0.4 * (M_MIN - MSTAR))

a = ALPHA + 2.0
S_r = gamma_upper(a, x_lim) / gamma_upper(a, x_min)
S_r = np.clip(S_r, 1e-6, 1.0)

w_L = 1.0 / S_r

# Galaxy luminosity L/L*
M_abs = K2Mpp - mu
L_over_Lstar = 10.0 ** (-0.4 * (M_abs - MSTAR))

# Total weight
w_total = w_ang * w_L * L_over_Lstar

# =============================================================================
# Apply masks
# =============================================================================
print("Applying masks...")

# Zero beyond 200 Mpc/h
w_total[r_mpc > RMAX] = 0.0

# Zero ALL galaxies in 2MRS-only regions (map12 <= 0)
is_2mrs_region = ~use_deep
w_total[is_2mrs_region] = 0.0

n_zeroed = np.sum(w_total == 0)
print(f"  {n_zeroed} galaxies zeroed ({100*n_zeroed/len(w_total):.1f}%)")

# =============================================================================
# Compute positions and deposit (simple NGP for clarity)
# =============================================================================
print("Depositing onto grid...")

# Galactic Cartesian coordinates
ell_rad = np.deg2rad(l_deg)
b_rad = np.deg2rad(b_deg)
cos_b = np.cos(b_rad)
x = r_mpc * cos_b * np.cos(ell_rad)
y = r_mpc * cos_b * np.sin(ell_rad)
z = r_mpc * np.sin(b_rad)

# Grid indices (NGP = nearest grid point)
dx = BOX_SIDE / N
ix = np.floor((x + RMAX) / dx).astype(int)
iy = np.floor((y + RMAX) / dx).astype(int)
iz = np.floor((z + RMAX) / dx).astype(int)

# Clip to valid range
valid_idx = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (iz >= 0) & (iz < N) & (w_total > 0)
ix, iy, iz = ix[valid_idx], iy[valid_idx], iz[valid_idx]
w_dep = w_total[valid_idx]

# Deposit
rho = np.zeros((N, N, N), dtype=np.float64)
np.add.at(rho, (ix, iy, iz), w_dep)

print(f"  Deposited {np.sum(valid_idx)} galaxies")

# =============================================================================
# Gaussian smoothing (Carrick uses 4 Mpc/h)
# =============================================================================
print("Applying Gaussian smoothing (4 Mpc/h)...")
from scipy.ndimage import gaussian_filter
sigma_voxels = 4.0 / dx  # Convert Mpc/h to voxels
rho = gaussian_filter(rho, sigma=sigma_voxels, mode='constant')
print(f"  Smoothing sigma = {sigma_voxels:.2f} voxels")

# =============================================================================
# Build 3D masks
# =============================================================================
print("Building 3D masks...")

coords = np.linspace(-RMAX + dx/2, RMAX - dx/2, N)
xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
rr = np.sqrt(xx**2 + yy**2 + zz**2)

# Angular mask from HEALPix
rr_safe = np.maximum(rr, 1e-10)
theta_3d = np.arccos(zz / rr_safe)
phi_3d = np.arctan2(yy, xx)
phi_3d = np.where(phi_3d < 0, phi_3d + 2*np.pi, phi_3d)
pix_3d = hp.ang2pix(hp.get_nside(map12), theta_3d.ravel(), phi_3d.ravel()).reshape(rr.shape)

is_2mrs_only_3d = map12[pix_3d] <= 0

# Mask: valid if r <= 200 AND NOT in 2MRS-only region
valid_3d = (rr <= RMAX) & ~is_2mrs_only_3d

print(f"  {np.sum(valid_3d)} valid voxels ({100*np.mean(valid_3d):.1f}%)")

# =============================================================================
# Compute radial profiles of rho (no normalization)
# =============================================================================
print("Computing radial profiles of rho...")

r_edges = np.linspace(0, RMAX, 41)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

mean_rho = np.zeros(len(r_centers))
std_rho = np.zeros(len(r_centers))

for i in range(len(r_centers)):
    shell = (rr >= r_edges[i]) & (rr < r_edges[i+1]) & valid_3d
    if np.any(shell):
        mean_rho[i] = np.mean(rho[shell])
        std_rho[i] = np.std(rho[shell])

print(f"  Global mean rho (valid): {np.mean(rho[valid_3d]):.4f}")

# Load Carrick and compute (1+delta) for comparison
print("Loading Carrick field...")
delta_carrick = np.load("data/fields/carrick2015_twompp_density.npy")
rho_carrick = 1.0 + delta_carrick  # Convert delta to density

mean_rho_carrick = np.zeros(len(r_centers))
std_rho_carrick = np.zeros(len(r_centers))

for i in range(len(r_centers)):
    shell = (rr >= r_edges[i]) & (rr < r_edges[i+1]) & valid_3d
    if np.any(shell):
        mean_rho_carrick[i] = np.mean(rho_carrick[shell])
        std_rho_carrick[i] = np.std(rho_carrick[shell])

# =============================================================================
# Plot
# =============================================================================
print("Plotting...")

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Normalize both to mean=1 for shape comparison
norm_ours = np.mean(mean_rho[mean_rho > 0])
norm_carrick = np.mean(mean_rho_carrick[mean_rho_carrick > 0])

# Mean rho
ax = axes[0]
ax.plot(r_centers, mean_rho / norm_ours, 'b-', lw=2, label='Ours (normalized)')
ax.plot(r_centers, mean_rho_carrick / norm_carrick, 'r--', lw=2, label='Carrick (normalized)')
ax.axhline(1, color='gray', ls=':', alpha=0.5)
ax.axvline(R_2MRS_CUTOFF, color='orange', ls='--', alpha=0.7, label='2MRS cutoff')
ax.set_ylabel('Mean ρ / <ρ>', fontsize=12)
ax.legend()
ax.set_title('Radial profiles (deep regions only, normalized to mean=1)')
ax.grid(True, alpha=0.3)

# Std rho (also normalized)
ax = axes[1]
ax.plot(r_centers, std_rho / norm_ours, 'b-', lw=2, label='Ours')
ax.plot(r_centers, std_rho_carrick / norm_carrick, 'r--', lw=2, label='Carrick')
ax.axvline(R_2MRS_CUTOFF, color='orange', ls='--', alpha=0.7)
ax.set_xlabel('r [Mpc/h]', fontsize=12)
ax.set_ylabel('Std ρ / <ρ>', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('minimal_carrick_profiles.png', dpi=150)
plt.close()
print("Saved minimal_carrick_profiles.png")
