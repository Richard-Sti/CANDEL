"""
Carrick2015 reconstruction with iterative RSD correction.

Combines:
- Correct equatorial coordinate query for HEALPix completeness maps
- Iterative beta adiabatic reconstruction (0 -> 1)
- Distance averaging to suppress oscillations
"""
import numpy as np
import healpy as hp
from scipy import special as sps
from scipy.ndimage import gaussian_filter
from collections import deque
from astropy.coordinates import SkyCoord
import astropy.units as u
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
Q0 = -0.55  # Deceleration parameter for LCDM

# Schechter LF parameters
ALPHA = -0.85
MSTAR = -23.25
M_MIN = -20.0

# Bias parameters (Westover 2007)
BIAS_CONST = 0.73
BIAS_SLOPE = 0.24

# Iteration parameters
N_ITERATIONS = 11  # beta from 0 to 1 (use 101 for full run)
N_AVG = 5  # Average last N distance estimates


def gamma_upper(s, x):
    """Upper incomplete gamma function."""
    x = np.asarray(x)
    if s > 0:
        return sps.gammaincc(s, x) * sps.gamma(s)
    k = int(np.ceil(1.0 - s))
    sp = s + k
    result = sps.gammaincc(sp, x) * sps.gamma(sp)
    for _ in range(k):
        result = (result - np.power(x, sp - 1) * np.exp(-x)) / (sp - 1)
        sp -= 1
    return result


def compute_selection_function(r_mpc, m_lim):
    """Compute selection function S(r) = Gamma(alpha+2, x_lim) / Gamma(alpha+2, x_min)."""
    r_safe = np.maximum(r_mpc, 0.1)
    mu = 5.0 * np.log10(r_safe) + 25.0
    M_lim = m_lim - mu

    x_lim = 10.0 ** (-0.4 * (M_lim - MSTAR))
    x_min = 10.0 ** (-0.4 * (M_MIN - MSTAR))

    a = ALPHA + 2.0
    S = gamma_upper(a, x_lim) / gamma_upper(a, x_min)
    return np.clip(S, 1e-6, 1.0)


def compute_psi(r_mpc, m_lim):
    """Compute bias normalization psi(r) = b_const + b_slope * <L/L*>."""
    r_safe = np.maximum(r_mpc, 0.1)
    mu = 5.0 * np.log10(r_safe) + 25.0
    M_lim = m_lim - mu
    x_lim = 10.0 ** (-0.4 * (M_lim - MSTAR))

    a2 = ALPHA + 2.0
    a3 = ALPHA + 3.0
    gamma_a2 = gamma_upper(a2, x_lim)
    gamma_a3 = gamma_upper(a3, x_lim)
    gamma_a2 = np.where(np.abs(gamma_a2) < 1e-12, 1e-12, gamma_a2)

    L_mean = gamma_a3 / gamma_a2
    return BIAS_CONST + BIAS_SLOPE * L_mean


def velocity_from_density_fft(delta, beta, box_side):
    """Compute velocity field from density using FFT."""
    N = delta.shape[0]

    # Zero-pad to avoid wrap-around
    N_pad = 2 * N
    delta_padded = np.zeros((N_pad, N_pad, N_pad), dtype=np.float64)
    delta_padded[:N, :N, :N] = delta
    dx_pad = box_side * 2 / N_pad

    delta_k = np.fft.rfftn(delta_padded)

    kx = 2.0 * np.pi * np.fft.fftfreq(N_pad, d=dx_pad)
    ky = 2.0 * np.pi * np.fft.fftfreq(N_pad, d=dx_pad)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N_pad, d=dx_pad)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0

    prefactor = 1j * beta * H0 / K2
    vx_k = prefactor * KX * delta_k
    vy_k = prefactor * KY * delta_k
    vz_k = prefactor * KZ * delta_k
    vx_k[0, 0, 0] = 0.0
    vy_k[0, 0, 0] = 0.0
    vz_k[0, 0, 0] = 0.0

    vx = np.fft.irfftn(vx_k, s=(N_pad, N_pad, N_pad))[:N, :N, :N]
    vy = np.fft.irfftn(vy_k, s=(N_pad, N_pad, N_pad))[:N, :N, :N]
    vz = np.fft.irfftn(vz_k, s=(N_pad, N_pad, N_pad))[:N, :N, :N]

    return np.stack([vx, vy, vz], axis=-1).astype(np.float32)


def trilinear_interp(field, positions, box_side):
    """Trilinear interpolation of 3D field at positions."""
    N = field.shape[0]
    Rmax = box_side / 2.0
    dx = box_side / N

    coords = (positions + Rmax) / dx
    i0 = np.floor(coords).astype(np.int32)
    f = coords - i0
    i0 = np.clip(i0, 0, N - 2)
    i1 = i0 + 1

    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    wx0, wx1 = 1.0 - fx, fx
    wy0, wy1 = 1.0 - fy, fy
    wz0, wz1 = 1.0 - fz, fz

    if field.ndim == 4:  # Vector field
        result = np.zeros((positions.shape[0], 3), dtype=np.float32)
        for c in range(3):
            fld = field[:, :, :, c]
            v000 = fld[i0[:, 0], i0[:, 1], i0[:, 2]]
            v100 = fld[i1[:, 0], i0[:, 1], i0[:, 2]]
            v010 = fld[i0[:, 0], i1[:, 1], i0[:, 2]]
            v110 = fld[i1[:, 0], i1[:, 1], i0[:, 2]]
            v001 = fld[i0[:, 0], i0[:, 1], i1[:, 2]]
            v101 = fld[i1[:, 0], i0[:, 1], i1[:, 2]]
            v011 = fld[i0[:, 0], i1[:, 1], i1[:, 2]]
            v111 = fld[i1[:, 0], i1[:, 1], i1[:, 2]]
            result[:, c] = (v000*wx0*wy0*wz0 + v100*wx1*wy0*wz0 +
                           v010*wx0*wy1*wz0 + v110*wx1*wy1*wz0 +
                           v001*wx0*wy0*wz1 + v101*wx1*wy0*wz1 +
                           v011*wx0*wy1*wz1 + v111*wx1*wy1*wz1)
        return result
    else:  # Scalar field
        v000 = field[i0[:, 0], i0[:, 1], i0[:, 2]]
        v100 = field[i1[:, 0], i0[:, 1], i0[:, 2]]
        v010 = field[i0[:, 0], i1[:, 1], i0[:, 2]]
        v110 = field[i1[:, 0], i1[:, 1], i0[:, 2]]
        v001 = field[i0[:, 0], i0[:, 1], i1[:, 2]]
        v101 = field[i1[:, 0], i0[:, 1], i1[:, 2]]
        v011 = field[i0[:, 0], i1[:, 1], i1[:, 2]]
        v111 = field[i1[:, 0], i1[:, 1], i1[:, 2]]
        return (v000*wx0*wy0*wz0 + v100*wx1*wy0*wz0 +
                v010*wx0*wy1*wz0 + v110*wx1*wy1*wz0 +
                v001*wx0*wy0*wz1 + v101*wx1*wy0*wz1 +
                v011*wx0*wy1*wz1 + v111*wx1*wy1*wz1)


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

valid = np.isfinite(l_deg) & np.isfinite(b_deg) & np.isfinite(vcmb) & np.isfinite(K2Mpp) & (vcmb > 0)
l_deg, b_deg, K2Mpp, vcmb, gid = l_deg[valid], b_deg[valid], K2Mpp[valid], vcmb[valid], gid[valid]
print(f"  {len(vcmb)} galaxies after filtering")

# =============================================================================
# Compute L/L* from ORIGINAL positions (before group centering)
# =============================================================================
print("Computing L/L* from original positions...")
z_orig = vcmb / C_LIGHT
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
group_dict = {gid_val: i for i, gid_val in enumerate(group_gid)}

for i in range(len(vcmb)):
    if gid[i] >= 0 and gid[i] in group_dict:
        idx = group_dict[gid[i]]
        l_deg[i], b_deg[i], vcmb[i] = group_l[idx], group_b[idx], group_v[idx]

valid = vcmb > 0
l_deg, b_deg, K2Mpp, vcmb, gid = l_deg[valid], b_deg[valid], K2Mpp[valid], vcmb[valid], gid[valid]
L_over_Lstar = L_over_Lstar[valid]
print(f"  {len(vcmb)} galaxies after FoF collapse")

# =============================================================================
# Load completeness maps (EQUATORIAL coordinates!)
# =============================================================================
print("Loading completeness maps...")
map11 = hp.read_map(f"{DATA_DIR}/incompleteness_11_5.fits", verbose=False)
map12 = hp.read_map(f"{DATA_DIR}/incompleteness_12_5.fits", verbose=False)

# Convert galactic (l, b) to equatorial (RA, DEC) for HEALPix query
coords_gal = SkyCoord(l=l_deg*u.deg, b=b_deg*u.deg, frame='galactic')
coords_eq = coords_gal.icrs
ra_deg = coords_eq.ra.deg
dec_deg = coords_eq.dec.deg

theta = np.deg2rad(90.0 - dec_deg)
phi = np.deg2rad(ra_deg)
pix11 = hp.ang2pix(hp.get_nside(map11), theta, phi)
pix12 = hp.ang2pix(hp.get_nside(map12), theta, phi)

c11 = map11[pix11]
c12 = map12[pix12]
use_deep = c12 > 0

m_lim = np.where(use_deep, 12.5, 11.5)
comp = np.where(use_deep, c12, c11)
comp = np.clip(np.where(np.isfinite(comp) & (comp > 0), comp, CMIN), CMIN, 1.0)

print(f"  {np.sum(~use_deep)} in 2MRS-only, {np.sum(use_deep)} in deep regions")

# Angular weight
w_ang = 1.0 / comp

# =============================================================================
# Initial distances
# =============================================================================
z_obs = vcmb / C_LIGHT
r_mpc = (C_LIGHT / H0) * (z_obs - (1.0 + Q0) / 2.0 * z_obs**2)

# Unit direction vectors in galactic cartesian
ell_rad = np.deg2rad(l_deg)
b_rad = np.deg2rad(b_deg)
cos_b = np.cos(b_rad)
rhat = np.stack([
    cos_b * np.cos(ell_rad),
    cos_b * np.sin(ell_rad),
    np.sin(b_rad)
], axis=-1)

# =============================================================================
# Build 3D grid and masks
# =============================================================================
print("Building 3D grid...")
dx = BOX_SIDE / N
coords = np.linspace(-RMAX + dx/2, RMAX - dx/2, N)
xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
rr = np.sqrt(xx**2 + yy**2 + zz**2)

# Build 3D deep/2MRS mask using equatorial coordinates
print("Building 3D masks (equatorial query)...")
rr_safe = np.maximum(rr, 1e-10)
gal_b_3d = np.arcsin(zz / rr_safe)
gal_l_3d = np.arctan2(yy, xx)
gal_l_3d = np.where(gal_l_3d < 0, gal_l_3d + 2*np.pi, gal_l_3d)

# Convert to equatorial
gal_l_deg_3d = np.rad2deg(gal_l_3d)
gal_b_deg_3d = np.rad2deg(gal_b_3d)
coords_gal_3d = SkyCoord(l=gal_l_deg_3d.ravel()*u.deg, b=gal_b_deg_3d.ravel()*u.deg, frame='galactic')
coords_eq_3d = coords_gal_3d.icrs
ra_3d = coords_eq_3d.ra.deg.reshape(rr.shape)
dec_3d = coords_eq_3d.dec.deg.reshape(rr.shape)

theta_3d = np.deg2rad(90.0 - dec_3d)
phi_3d = np.deg2rad(ra_3d)
pix_3d = hp.ang2pix(hp.get_nside(map12), theta_3d.ravel(), phi_3d.ravel()).reshape(rr.shape)

is_2mrs_only_3d = map12[pix_3d] <= 0
is_deep_3d = ~is_2mrs_only_3d

# Masks
valid_region = rr <= RMAX
zeroed_region = is_2mrs_only_3d & (rr > R_2MRS_CUTOFF)
nonmasked = valid_region & ~zeroed_region

# m_lim for 3D psi computation
m_lim_3d = np.where(is_2mrs_only_3d, 11.5, 12.5)

# =============================================================================
# Iterative reconstruction
# =============================================================================
print(f"Starting iterative reconstruction ({N_ITERATIONS} iterations)...")
beta_values = np.linspace(0.0, 1.0, N_ITERATIONS)
distance_history = deque(maxlen=N_AVG)
sigma_voxels = 4.0 / dx

for i_iter, beta in enumerate(beta_values):
    # Compute weights with current distances
    S_r = compute_selection_function(r_mpc, m_lim)
    w_L = 1.0 / S_r

    # Total weight
    w_total = w_ang * w_L * L_over_Lstar

    # Zero out 2MRS beyond cutoff and outside box
    w_total[(~use_deep) & (r_mpc > R_2MRS_CUTOFF)] = 0.0
    w_total[r_mpc > RMAX] = 0.0

    # Compute positions
    positions = r_mpc[:, None] * rhat
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Grid indices
    ix = np.floor((x + RMAX) / dx).astype(int)
    iy = np.floor((y + RMAX) / dx).astype(int)
    iz = np.floor((z + RMAX) / dx).astype(int)

    # Deposit
    valid_dep = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (iz >= 0) & (iz < N) & (w_total > 0)
    rho = np.zeros((N, N, N), dtype=np.float64)
    np.add.at(rho, (ix[valid_dep], iy[valid_dep], iz[valid_dep]), w_total[valid_dep])

    # Normalize
    rho_mean = np.mean(rho[nonmasked])
    rho_mean = max(rho_mean, 1e-10)
    delta = rho / rho_mean - 1.0
    delta = np.where(nonmasked, delta, 0.0)

    # Bias normalization
    psi_3d = compute_psi(rr.ravel(), m_lim_3d.ravel()).reshape(rr.shape)
    psi_3d = np.maximum(psi_3d, 0.1)
    delta = delta / psi_3d
    delta = np.where(nonmasked, delta, 0.0)

    # Smooth
    delta = gaussian_filter(delta, sigma=sigma_voxels, mode='constant')

    if beta > 0:
        # Compute velocity
        velocity = velocity_from_density_fft(delta, beta, BOX_SIDE)

        # Interpolate at galaxy positions
        v_at_gal = trilinear_interp(velocity, positions, BOX_SIDE)
        v_los = np.sum(v_at_gal * rhat, axis=1)

        # Update distances
        z_cos = (1.0 + z_obs) / (1.0 + v_los / C_LIGHT) - 1.0
        z_cos = np.maximum(z_cos, 1e-8)
        r_new = (C_LIGHT / H0) * (z_cos - (1.0 + Q0) / 2.0 * z_cos**2)

        # Average
        distance_history.append(r_new.copy())
        if len(distance_history) > 1:
            r_mpc = np.mean(np.array(distance_history), axis=0)
        else:
            r_mpc = r_new

    if i_iter % 20 == 0 or i_iter == N_ITERATIONS - 1:
        print(f"  Iteration {i_iter}/{N_ITERATIONS-1}, beta={beta:.2f}, delta std={np.std(delta[nonmasked]):.3f}")

# =============================================================================
# Final output
# =============================================================================
print("Reconstruction complete.")

# Set delta=0 outside valid region
delta[~valid_region] = 0.0

# Load Carrick for comparison
print("Loading Carrick for comparison...")
delta_carrick = np.load("data/fields/carrick2015_twompp_density.npy")

print(f"\nFinal statistics:")
print(f"  Ours:    min={np.min(delta):.2f}, max={np.max(delta):.2f}, std={np.std(delta[nonmasked]):.3f}")
print(f"  Carrick: min={np.min(delta_carrick):.2f}, max={np.max(delta_carrick):.2f}, std={np.std(delta_carrick):.3f}")

# Save
np.save("carrick_iterated_delta.npy", delta.astype(np.float32))
print("Saved carrick_iterated_delta.npy")

# =============================================================================
# Plot comparison
# =============================================================================
print("Plotting supergalactic plane comparison...")
from scipy.interpolate import RegularGridInterpolator
from astropy.coordinates import CartesianRepresentation, Galactic, Supergalactic

# Rotation matrix
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

# Interpolators
interp_ours = RegularGridInterpolator((coords, coords, coords), delta, bounds_error=False, fill_value=np.nan)
interp_carrick = RegularGridInterpolator((coords, coords, coords), delta_carrick, bounds_error=False, fill_value=np.nan)

# SG plane grid
sg_extent = 200.0
n_pix = 400
sg_coords = np.linspace(-sg_extent, sg_extent, n_pix)
SGX, SGY = np.meshgrid(sg_coords, sg_coords, indexing='ij')
SGZ = np.zeros_like(SGX)

sg_pos = np.stack([SGX, SGY, SGZ], axis=-1)
gal_pos = np.einsum('ij,...j->...i', R_sg2gal, sg_pos)

slice_ours = interp_ours(gal_pos)
slice_carrick = interp_carrick(gal_pos)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
vmin, vmax = -1.0, 2.0
extent = [-sg_extent, sg_extent, -sg_extent, sg_extent]

ax = axes[0]
im = ax.imshow(slice_ours.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax.set_xlabel('SGX [Mpc/h]')
ax.set_ylabel('SGY [Mpc/h]')
ax.set_title(r'Ours (iterated) $\delta^*$')
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
plt.savefig('carrick_iterated_SGP.png', dpi=150)
plt.close()
print("Saved carrick_iterated_SGP.png")
