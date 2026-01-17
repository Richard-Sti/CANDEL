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
CMIN = 0.1  # Minimum completeness
H0 = 100.0
C_LIGHT = 299792.458
Q0 = -0.55  # Deceleration parameter for LCDM

# Schechter LF parameters (Carrick2015 values)
ALPHA = -0.85
MSTAR = -23.25
M_MIN = -20.0  # Faintest absolute magnitude for selection function integration

# Local Group velocity relative to CMB
V_LG = 627.0  # km/s
L_APEX = 276.0  # degrees (galactic)
B_APEX = 30.0  # degrees (galactic)

# Bias parameters (Westover 2007)
BIAS_CONST = 0.73
BIAS_SLOPE = 0.24

# Iteration parameters
N_ITERATIONS = 1  # beta from 0 to 0.43
BETA_MAX = 0.43
N_AVG = 5  # Average last N distance estimates
PLOT_EVERY = 10  # Save diagnostic plots every N iterations

# ZoA cloning
CLONE_ZOA = True  # Enable Zone of Avoidance cloning


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


def compute_selection_function(r_mpc, m_lim, alpha=None, Mstar=None):
    """Compute selection function S(r) = Gamma(alpha+2, x_lim) / Gamma(alpha+2, x_min)."""
    if alpha is None:
        alpha = ALPHA
    if Mstar is None:
        Mstar = MSTAR

    r_safe = np.maximum(r_mpc, 0.1)
    mu = 5.0 * np.log10(r_safe) + 25.0
    M_lim = m_lim - mu

    x_lim = 10.0 ** (-0.4 * (M_lim - Mstar))
    x_min = 10.0 ** (-0.4 * (M_MIN - Mstar))

    a = alpha + 2.0
    S = gamma_upper(a, x_lim) / gamma_upper(a, x_min)
    return np.clip(S, 1e-6, 1.0)


def compute_psi(r_mpc, m_lim, alpha=None, Mstar=None):
    """Compute bias normalization psi(r) = b_const + b_slope * <L/L*>."""
    if alpha is None:
        alpha = ALPHA
    if Mstar is None:
        Mstar = MSTAR

    r_safe = np.maximum(r_mpc, 0.1)
    mu = 5.0 * np.log10(r_safe) + 25.0
    M_lim = m_lim - mu
    x_lim = 10.0 ** (-0.4 * (M_lim - Mstar))

    a2 = alpha + 2.0
    a3 = alpha + 3.0
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


def clone_zoa_galaxies(l_deg, b_deg, vcmb, K2Mpp, gid, w_total):
    """
    Clone galaxies from strips above/below ZoA to fill masked region.
    Uses MIRROR approach (Lavaux/Carrick): mirror across ZoA boundary.

    For |ℓ| > 30° (narrow ZoA): mirror 5° < |b| < 10° across b=±5° into |b| < 5°
    For |ℓ| < 30° (wide ZoA):   mirror 10° < |b| < 20° across b=±10° into |b| < 10°

    Mirror formula: clone_b = 2 * boundary - source_b
    This means north sources go to south ZoA and vice versa.

    Returns: (l_ext, b_ext, vcmb_ext, K_ext, gid_ext, w_ext, is_clone, source_idx)
    source_idx[i] = index of source galaxy for clone i (or -1 for originals)
    """
    n_orig = len(l_deg)

    # Normalize longitude to [-180, 180] for easier GC detection
    l_norm = np.mod(l_deg + 180, 360) - 180
    near_gc = np.abs(l_norm) < 30  # Near Galactic center

    # Collect clones and their source indices
    clone_l, clone_b, clone_v, clone_K, clone_gid, clone_w = [], [], [], [], [], []
    clone_src_idx = []  # Track which original galaxy each clone comes from

    # ===== NARROW ZOA (|ℓ| > 30°): mirror across b=±5° =====
    # North: 5° < b < 10° → mirror across b=5 → -5° < clone_b < 0°
    src_n = (~near_gc) & (b_deg > 5) & (b_deg < 10)
    if np.any(src_n):
        clone_l.append(l_deg[src_n])
        clone_b.append(10.0 - b_deg[src_n])  # 2*5 - b = 10 - b
        clone_v.append(vcmb[src_n])
        clone_K.append(K2Mpp[src_n])
        clone_gid.append(gid[src_n])
        clone_w.append(w_total[src_n])
        clone_src_idx.append(np.where(src_n)[0])

    # South: -10° < b < -5° → mirror across b=-5 → 0° < clone_b < 5°
    src_s = (~near_gc) & (b_deg > -10) & (b_deg < -5)
    if np.any(src_s):
        clone_l.append(l_deg[src_s])
        clone_b.append(-10.0 - b_deg[src_s])  # 2*(-5) - b = -10 - b
        clone_v.append(vcmb[src_s])
        clone_K.append(K2Mpp[src_s])
        clone_gid.append(gid[src_s])
        clone_w.append(w_total[src_s])
        clone_src_idx.append(np.where(src_s)[0])

    # ===== WIDE ZOA (|ℓ| < 30°): mirror across b=±10° =====
    # North: 10° < b < 20° → mirror across b=10 → -10° < clone_b < 0°
    src_n_wide = near_gc & (b_deg > 10) & (b_deg < 20)
    if np.any(src_n_wide):
        clone_l.append(l_deg[src_n_wide])
        clone_b.append(20.0 - b_deg[src_n_wide])  # 2*10 - b = 20 - b
        clone_v.append(vcmb[src_n_wide])
        clone_K.append(K2Mpp[src_n_wide])
        clone_gid.append(gid[src_n_wide])
        clone_w.append(w_total[src_n_wide])
        clone_src_idx.append(np.where(src_n_wide)[0])

    # South: -20° < b < -10° → mirror across b=-10 → 0° < clone_b < 10°
    src_s_wide = near_gc & (b_deg > -20) & (b_deg < -10)
    if np.any(src_s_wide):
        clone_l.append(l_deg[src_s_wide])
        clone_b.append(-20.0 - b_deg[src_s_wide])  # 2*(-10) - b = -20 - b
        clone_v.append(vcmb[src_s_wide])
        clone_K.append(K2Mpp[src_s_wide])
        clone_gid.append(gid[src_s_wide])
        clone_w.append(w_total[src_s_wide])
        clone_src_idx.append(np.where(src_s_wide)[0])

    # Concatenate
    if clone_l:
        n_clones = sum(len(c) for c in clone_l)
        l_ext = np.concatenate([l_deg] + clone_l)
        b_ext = np.concatenate([b_deg] + clone_b)
        v_ext = np.concatenate([vcmb] + clone_v)
        K_ext = np.concatenate([K2Mpp] + clone_K)
        gid_ext = np.concatenate([gid] + clone_gid)
        w_ext = np.concatenate([w_total] + clone_w)
        is_clone = np.concatenate([np.zeros(n_orig, dtype=bool),
                                   np.ones(n_clones, dtype=bool)])
        # Source index: -1 for originals, actual index for clones
        source_idx = np.concatenate([np.full(n_orig, -1, dtype=int),
                                     np.concatenate(clone_src_idx)])
    else:
        l_ext, b_ext, v_ext, K_ext, gid_ext, w_ext = l_deg, b_deg, vcmb, K2Mpp, gid, w_total
        is_clone = np.zeros(n_orig, dtype=bool)
        source_idx = np.full(n_orig, -1, dtype=int)

    return l_ext, b_ext, v_ext, K_ext, gid_ext, w_ext, is_clone, source_idx


def match_fibre_collision_pairs(l_deg, b_deg, vcmb, fc_flag):
    """
    Match fibre collision clones to their source galaxies.
    Sources have same vcmb and are ~3 arcmin away.

    Returns: fc_source_idx array where fc_source_idx[i] = source index for clone i,
             or -1 if not a clone or no match found
    """
    fc_source_idx = np.full(len(l_deg), -1, dtype=int)
    is_fc_clone = fc_flag == 1
    is_real = fc_flag != 1

    for i in np.where(is_fc_clone)[0]:
        # Find real galaxies with same vcmb (within 1 km/s)
        same_v = is_real & (np.abs(vcmb - vcmb[i]) < 1)
        if not np.any(same_v):
            continue

        # Find nearest one (angular separation in degrees)
        dl = (l_deg[same_v] - l_deg[i]) * np.cos(np.deg2rad(b_deg[i]))
        db = b_deg[same_v] - b_deg[i]
        ang_sep = np.sqrt(dl**2 + db**2)

        min_idx = np.argmin(ang_sep)
        if ang_sep[min_idx] < 0.5:  # Within 0.5 degrees (30 arcmin)
            fc_source_idx[i] = np.where(same_v)[0][min_idx]

    return fc_source_idx


def fit_schechter_lf(M_abs, r_mpc, m_lim, vcmb, M_bright=-25.0, M_faint=-17.0,
                     v_min=750.0, v_max=20000.0):
    """
    Fit Schechter LF using STY maximum likelihood (Lavaux 2011 method).

    The likelihood for galaxy i with absolute magnitude M_i at distance r_i:
    L_i = Phi(M_i) / integral_{M_lim(r_i)}^{M_faint} Phi(M) dM

    where M_lim(r) = m_lim - 5*log10(r) - 25 is the faintest observable abs mag.

    Parameters
    ----------
    M_abs : array
        Absolute magnitudes of galaxies
    r_mpc : array
        Distances in Mpc/h
    m_lim : array
        Apparent magnitude limit for each galaxy (11.5 or 12.5)
    vcmb : array
        CMB-frame velocities in km/s
    M_bright : float
        Brightest absolute magnitude for fitting (default -25)
    M_faint : float
        Faintest absolute magnitude for fitting (default -21)
    v_min, v_max : float
        Velocity range for fitting (default 5000-20000 km/s, Lavaux 2011)

    Returns
    -------
    alpha, Mstar : fitted Schechter parameters
    """
    from scipy.optimize import minimize

    # Lavaux 2011 cuts: velocity range and magnitude range
    valid = (np.isfinite(M_abs) & (r_mpc > 0.1) & np.isfinite(m_lim) &
             (vcmb >= v_min) & (vcmb <= v_max) &
             (M_abs >= M_bright) & (M_abs <= M_faint))
    M_fit = M_abs[valid]
    r_fit = r_mpc[valid]
    mlim_fit = m_lim[valid]

    print(f"    LF fitting: {valid.sum()} galaxies (v=[{v_min},{v_max}], M=[{M_bright},{M_faint}])")
    print(f"      M_abs range: [{M_fit.min():.2f}, {M_fit.max():.2f}], median={np.median(M_fit):.2f}")

    # Compute M_lim for each galaxy
    mu_fit = 5.0 * np.log10(r_fit) + 25.0
    M_lim_fit = mlim_fit - mu_fit  # Faintest observable abs mag at this distance

    def neg_log_likelihood(params):
        alpha, Mstar = params

        # Phi(M) = C * 10^(0.4*(1+alpha)*(Mstar-M)) * exp(-10^(0.4*(Mstar-M)))
        # where C = 0.4 * ln(10)

        x = 10.0 ** (-0.4 * (M_fit - Mstar))
        log_phi = (1 + alpha) * (-0.4 * np.log(10) * (M_fit - Mstar)) - x

        # Normalization integral: integral from M_lim to M_faint
        # = Gamma(alpha+2, x_lim) - Gamma(alpha+2, x_faint)
        # where x = 10^(0.4*(Mstar-M))
        x_lim = 10.0 ** (-0.4 * (M_lim_fit - Mstar))
        x_faint = 10.0 ** (-0.4 * (M_faint - Mstar))

        # Use incomplete gamma function
        a = alpha + 2.0
        norm_integral = gamma_upper(a, x_lim) - gamma_upper(a, x_faint)
        norm_integral = np.maximum(norm_integral, 1e-30)

        log_norm = np.log(norm_integral)

        # Log likelihood
        log_L = np.sum(log_phi - log_norm)

        # Return negative for minimization
        return -log_L

    # Minimize
    result = minimize(neg_log_likelihood, x0=[-0.85, -23.25],
                      bounds=[(-2.0, -0.1), (-25.0, -21.0)],
                      method='L-BFGS-B')

    return result.x[0], result.x[1]


# =============================================================================
# Load galaxies
# =============================================================================
print("Loading galaxies...")
with open(f"{DATA_DIR}/2m++.txt") as f:
    lines = f.readlines()

header_idx = next(i for i, line in enumerate(lines) if "Designation" in line and "|" in line)
# Load columns: l, b, K, Vcmb, GID, ZoA_flag, FC_clone_flag, M0, M1, M2
data = np.genfromtxt(f"{DATA_DIR}/2m++.txt", delimiter="|", skip_header=header_idx + 2,
                     usecols=[3, 4, 5, 7, 9, 12, 13, 14, 15, 16], filling_values=0)

l_deg = data[:, 0]
b_deg = data[:, 1]
K2Mpp = data[:, 2]
vcmb = data[:, 3]
gid = np.where(np.isfinite(data[:, 4]), data[:, 4], -1).astype(int)
zoa_flag = np.where(np.isfinite(data[:, 5]), data[:, 5], 0).astype(int)
fc_clone_flag = np.where(np.isfinite(data[:, 6]), data[:, 6], 0).astype(int)
# Survey flags: M0=2MRS exclusive, M1=SDSS, M2=6dFGRS
is_2mrs_only = np.where(np.isfinite(data[:, 7]), data[:, 7], 0).astype(int) == 1
is_sdss = np.where(np.isfinite(data[:, 8]), data[:, 8], 0).astype(int) == 1
is_6df = np.where(np.isfinite(data[:, 9]), data[:, 9], 0).astype(int) == 1
print(f"  ZoA clones in catalogue: {np.sum(zoa_flag == 1)}")
print(f"  Fibre collision clones in catalogue: {np.sum(fc_clone_flag == 1)}")
print(f"  Survey flags: 2MRS-only={is_2mrs_only.sum()}, SDSS={is_sdss.sum()}, 6dF={is_6df.sum()}")

valid = np.isfinite(l_deg) & np.isfinite(b_deg) & np.isfinite(vcmb) & np.isfinite(K2Mpp) & (vcmb > 0)
l_deg, b_deg, K2Mpp, vcmb, gid = l_deg[valid], b_deg[valid], K2Mpp[valid], vcmb[valid], gid[valid]
zoa_flag, fc_clone_flag = zoa_flag[valid], fc_clone_flag[valid]
is_2mrs_only = is_2mrs_only[valid]
print(f"  {len(vcmb)} galaxies after filtering")

# =============================================================================
# Compute L/L* from ORIGINAL positions in LG frame (before group centering)
# =============================================================================
print("Computing L/L* from original positions (LG frame)...")

# Convert CMB to LG frame for L/L* calculation
l_rad_orig = np.deg2rad(l_deg)
b_rad_orig = np.deg2rad(b_deg)
l_apex_rad = np.deg2rad(L_APEX)
b_apex_rad = np.deg2rad(B_APEX)
cos_theta_orig = (np.sin(b_rad_orig) * np.sin(b_apex_rad) +
                  np.cos(b_rad_orig) * np.cos(b_apex_rad) * np.cos(l_rad_orig - l_apex_rad))
v_lg_orig = vcmb - V_LG * cos_theta_orig

# Use LG frame for distance/luminosity calculation
z_orig = v_lg_orig / C_LIGHT
r_orig = (C_LIGHT / H0) * z_orig
r_orig_safe = np.maximum(r_orig, 0.1)
mu_orig = 5.0 * np.log10(r_orig_safe) + 25.0
M_abs_orig = K2Mpp - mu_orig
L_over_Lstar = 10.0 ** (-0.4 * (M_abs_orig - MSTAR))
print(f"  Mean L/L* = {np.mean(L_over_Lstar):.2f}")

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
zoa_flag, fc_clone_flag = zoa_flag[valid], fc_clone_flag[valid]
is_2mrs_only = is_2mrs_only[valid]
L_over_Lstar = L_over_Lstar[valid]
print(f"  {len(vcmb)} galaxies after FoF collapse")

# =============================================================================
# Remove galaxies in ZoA target region (before cloning)
# =============================================================================
if CLONE_ZOA:
    # ZoA target regions:
    # - |ℓ| > 30°: |b| < 5° (narrow ZoA)
    # - |ℓ| < 30°: |b| < 10° (wide ZoA near Galactic center)
    l_norm = np.mod(l_deg + 180, 360) - 180
    near_gc = np.abs(l_norm) < 30

    # Galaxies in ZoA target region
    in_narrow_zoa = (~near_gc) & (np.abs(b_deg) < 5)
    in_wide_zoa = near_gc & (np.abs(b_deg) < 10)
    in_zoa = in_narrow_zoa | in_wide_zoa

    n_removed = in_zoa.sum()
    keep = ~in_zoa
    l_deg, b_deg, K2Mpp, vcmb, gid = l_deg[keep], b_deg[keep], K2Mpp[keep], vcmb[keep], gid[keep]
    zoa_flag, fc_clone_flag = zoa_flag[keep], fc_clone_flag[keep]
    is_2mrs_only = is_2mrs_only[keep]
    L_over_Lstar = L_over_Lstar[keep]
    print(f"  Removed {n_removed} galaxies in ZoA target region, {len(vcmb)} remaining")

# =============================================================================
# Match fibre collision clones to sources
# =============================================================================
print("Matching fibre collision clones to sources...")
fc_source_idx = match_fibre_collision_pairs(l_deg, b_deg, vcmb, fc_clone_flag)
n_fc_clones = np.sum(fc_clone_flag == 1)
n_fc_matched = np.sum(fc_source_idx >= 0)
print(f"  Matched {n_fc_matched}/{n_fc_clones} fibre collision clones to sources")

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

# Use catalogue flags for m_lim (2MRS=11.5, deep=12.5)
m_lim = np.where(is_2mrs_only, 11.5, 12.5)
# Use appropriate completeness map based on survey
comp = np.where(is_2mrs_only, c11, c12)
comp = np.clip(np.where(np.isfinite(comp) & (comp > 0), comp, CMIN), CMIN, 1.0)

print(f"  {np.sum(is_2mrs_only)} in 2MRS-only, {np.sum(~is_2mrs_only)} in deep regions")

# Angular weight
w_ang = 1.0 / comp

# FC clones inherit w_ang and m_lim from their source (once, position-based)
fc_matched = fc_source_idx >= 0
if np.any(fc_matched):
    w_ang[fc_matched] = w_ang[fc_source_idx[fc_matched]]
    m_lim[fc_matched] = m_lim[fc_source_idx[fc_matched]]
    print(f"  FC clones: inherited w_ang and m_lim from sources")

# =============================================================================
# Initial distances (LG frame)
# =============================================================================
# Convert CMB to LG frame
ell_rad = np.deg2rad(l_deg)
b_rad = np.deg2rad(b_deg)
cos_theta = (np.sin(b_rad) * np.sin(b_apex_rad) +
             np.cos(b_rad) * np.cos(b_apex_rad) * np.cos(ell_rad - l_apex_rad))
v_lg = vcmb - V_LG * cos_theta

# Use LG frame for initial distances
z_obs = v_lg / C_LIGHT
z_obs = np.maximum(z_obs, 1e-6)  # Avoid negative redshifts
r_mpc = (C_LIGHT / H0) * (z_obs - (1.0 + Q0) / 2.0 * z_obs**2)
r_mpc = np.maximum(r_mpc, 0.1)  # Minimum distance
print(f"  Using LG frame: mean shift = {np.mean(V_LG * cos_theta):.1f} km/s")

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
beta_values = np.linspace(0.0, BETA_MAX, N_ITERATIONS)
distance_history = deque(maxlen=N_AVG)
sigma_voxels = 4.0 / dx

import time as _time

for i_iter, beta in enumerate(beta_values):
    # Fit Schechter LF with current distances
    r_safe = np.maximum(r_mpc, 0.1)
    mu_current = 5.0 * np.log10(r_safe) + 25.0
    M_abs_current = K2Mpp - mu_current

    # Fit Schechter LF at first iteration only (use fixed values after)
    if i_iter == 0:
        alpha_fit, Mstar_fit = fit_schechter_lf(M_abs_current, r_mpc, m_lim, vcmb)
        print(f"    LF: alpha={alpha_fit:.3f}, M*={Mstar_fit:.2f}")
        # Store for later iterations
        _alpha_stored, _Mstar_stored = alpha_fit, Mstar_fit
    else:
        alpha_fit, Mstar_fit = _alpha_stored, _Mstar_stored

    # Recompute L/L* with fitted M*
    L_over_Lstar = 10.0 ** (-0.4 * (M_abs_current - Mstar_fit))

    # Compute weights with current distances and fitted LF
    S_r = compute_selection_function(r_mpc, m_lim, alpha=alpha_fit, Mstar=Mstar_fit)
    w_L = 1.0 / S_r

    # Total weight
    w_total = w_ang * w_L * L_over_Lstar

    # Zero out 2MRS beyond cutoff and outside box
    w_total[is_2mrs_only & (r_mpc > R_2MRS_CUTOFF)] = 0.0
    w_total[r_mpc > RMAX] = 0.0

    # Clone ZoA galaxies
    if CLONE_ZOA:
        l_ext, b_ext, v_ext, K_ext, gid_ext, w_ext, is_clone, source_idx = clone_zoa_galaxies(
            l_deg, b_deg, vcmb, K2Mpp, gid, w_total
        )
        n_clones = is_clone.sum()
        if i_iter == 0:
            print(f"    ZoA cloning: {n_clones} clones added")

        # Compute rhat for all (including clones with shifted b)
        l_rad_ext = np.deg2rad(l_ext)
        b_rad_ext = np.deg2rad(b_ext)
        cos_b_ext = np.cos(b_rad_ext)
        rhat_ext = np.stack([
            cos_b_ext * np.cos(l_rad_ext),
            cos_b_ext * np.sin(l_rad_ext),
            np.sin(b_rad_ext)
        ], axis=-1)

        # Build r_mpc for extended array:
        # - Original galaxies: use current r_mpc (updated by iteration)
        # - Clones: inherit r_mpc from source galaxy
        r_mpc_dep = np.zeros(len(l_ext))
        r_mpc_dep[:len(r_mpc)] = r_mpc  # Original galaxies
        r_mpc_dep[is_clone] = r_mpc[source_idx[is_clone]]  # Clones inherit from source

        # Compute positions
        positions = r_mpc_dep[:, None] * rhat_ext
        w_dep = w_ext
    else:
        # No cloning
        positions = r_mpc[:, None] * rhat
        w_dep = w_total
        is_clone = np.zeros(len(w_total), dtype=bool)
        source_idx = np.full(len(w_total), -1, dtype=int)

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Compare NGP vs CIC
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # NGP
    rho_ngp = np.zeros((N, N, N), dtype=np.float64)
    ix = np.floor((x + RMAX) / dx).astype(int)
    iy = np.floor((y + RMAX) / dx).astype(int)
    iz = np.floor((z + RMAX) / dx).astype(int)
    valid_ngp = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (iz >= 0) & (iz < N) & (w_dep > 0)
    np.add.at(rho_ngp, (ix[valid_ngp], iy[valid_ngp], iz[valid_ngp]), w_dep[valid_ngp])

    # CIC
    rho_cic = np.zeros((N, N, N), dtype=np.float64)
    cic_coords = (positions + RMAX) / dx
    i0 = np.floor(cic_coords).astype(int)
    f = cic_coords - i0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0[:, 0] + di
                jj = i0[:, 1] + dj
                kk = i0[:, 2] + dk
                wx = (1.0 - f[:, 0]) if di == 0 else f[:, 0]
                wy = (1.0 - f[:, 1]) if dj == 0 else f[:, 1]
                wz = (1.0 - f[:, 2]) if dk == 0 else f[:, 2]
                w_cic = wx * wy * wz * w_dep
                valid = (ii >= 0) & (ii < N) & (jj >= 0) & (jj < N) & (kk >= 0) & (kk < N) & (w_cic > 0)
                np.add.at(rho_cic, (ii[valid], jj[valid], kk[valid]), w_cic[valid])

    if i_iter == 0:
        print(f"    NGP: sum={rho_ngp.sum():.1f}, max={rho_ngp.max():.1f}")
        print(f"    CIC: sum={rho_cic.sum():.1f}, max={rho_cic.max():.1f}")

    rho = rho_cic  # Use CIC

    # Normalize
    rho_mean = np.mean(rho[nonmasked])
    rho_mean = max(rho_mean, 1e-10)
    if i_iter == 0:
        print(f"    DEBUG: rho sum={rho.sum():.1f}, max={rho.max():.1f}, mean_nonmasked={rho_mean:.4f}")
    delta = rho / rho_mean - 1.0
    delta = np.where(nonmasked, delta, 0.0)
    if i_iter == 0:
        print(f"    DEBUG: delta (pre-psi) max={delta[nonmasked].max():.1f}, std={delta[nonmasked].std():.3f}")

    # Bias normalization (using fitted LF parameters)
    psi_3d = compute_psi(rr.ravel(), m_lim_3d.ravel(), alpha=alpha_fit, Mstar=Mstar_fit).reshape(rr.shape)
    psi_3d = np.maximum(psi_3d, 0.1)
    delta = delta / psi_3d
    delta = np.where(nonmasked, delta, 0.0)
    if i_iter == 0:
        print(f"    DEBUG: delta (post-psi) max={delta[nonmasked].max():.1f}, std={delta[nonmasked].std():.3f}")

    # Smooth
    delta = gaussian_filter(delta, sigma=sigma_voxels, mode='constant')
    if i_iter == 0:
        print(f"    DEBUG: delta (smoothed) max={delta[nonmasked].max():.1f}, std={delta[nonmasked].std():.3f}")

    if beta > 0:
        # Compute velocity
        velocity = velocity_from_density_fft(delta, beta, BOX_SIDE)

        # Interpolate at ORIGINAL galaxy positions only (not clones)
        # Clones inherit distances from source galaxies, not from velocity field
        positions_orig = r_mpc[:, None] * rhat
        v_at_gal = trilinear_interp(velocity, positions_orig, BOX_SIDE)
        v_los = np.sum(v_at_gal * rhat, axis=1)

        # Update distances for original galaxies only
        z_cos = (1.0 + z_obs) / (1.0 + v_los / C_LIGHT) - 1.0
        z_cos = np.maximum(z_cos, 1e-8)
        r_new = (C_LIGHT / H0) * (z_cos - (1.0 + Q0) / 2.0 * z_cos**2)

        # Average
        distance_history.append(r_new.copy())
        if len(distance_history) > 1:
            r_mpc = np.mean(np.array(distance_history), axis=0)
        else:
            r_mpc = r_new

        # FC clones inherit r_mpc from their source galaxy
        if np.any(fc_matched):
            r_mpc[fc_matched] = r_mpc[fc_source_idx[fc_matched]]

    delta_std = np.std(delta[nonmasked])
    print(f"  {i_iter}/{N_ITERATIONS}, beta={beta:.2f}, std={delta_std:.3f}")

    # Save diagnostic plots every PLOT_EVERY iterations
    if i_iter % PLOT_EVERY == 0 or i_iter == N_ITERATIONS - 1:
        # Plot psi(r) and w_L(r) on first iteration
        if i_iter == 0:
            r_plot = np.linspace(1, 200, 200)
            psi_2mrs = compute_psi(r_plot, np.full_like(r_plot, 11.5))
            psi_deep = compute_psi(r_plot, np.full_like(r_plot, 12.5))
            S_2mrs = compute_selection_function(r_plot, np.full_like(r_plot, 11.5))
            S_deep = compute_selection_function(r_plot, np.full_like(r_plot, 12.5))
            wL_2mrs = 1.0 / S_2mrs
            wL_deep = 1.0 / S_deep

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Left panel: psi(r)
            ax1.plot(r_plot, psi_2mrs, 'b-', lw=2, label='2MRS (m=11.5)')
            ax1.plot(r_plot, psi_deep, 'r-', lw=2, label='Deep (m=12.5)')
            ax1.axvline(R_2MRS_CUTOFF, color='orange', ls='--', alpha=0.7, label='2MRS cutoff')
            ax1.set_xlabel('r [Mpc/h]')
            ax1.set_ylabel(r'$\psi(r) = b_{const} + b_{slope} \langle L/L^* \rangle$')
            ax1.set_title('Bias normalization factor')
            ax1.legend()
            ax1.set_xlim(0, 200)
            ax1.grid(True, alpha=0.3)

            # Right panel: w_L(r) = 1/S(r)
            ax2.plot(r_plot, wL_2mrs, 'b-', lw=2, label='2MRS (m=11.5)')
            ax2.plot(r_plot, wL_deep, 'r-', lw=2, label='Deep (m=12.5)')
            ax2.axvline(R_2MRS_CUTOFF, color='orange', ls='--', alpha=0.7, label='2MRS cutoff')
            ax2.set_xlabel('r [Mpc/h]')
            ax2.set_ylabel(r'$w_L(r) = 1/S(r)$')
            ax2.set_title('Luminosity selection weight')
            ax2.legend()
            ax2.set_xlim(0, 200)
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('psi_r_diagnostic.png', dpi=100)
            plt.close()
            print(f"    Saved psi_r_diagnostic.png")

            # ZoA cloning diagnostic plot
            if CLONE_ZOA and is_clone.sum() > 0:
                fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={'projection': 'mollweide'})

                # Wrap longitude for mollweide
                l_rad_plot = np.deg2rad(l_ext)
                l_rad_plot = np.where(l_rad_plot > np.pi, l_rad_plot - 2*np.pi, l_rad_plot)
                b_rad_plot = np.deg2rad(b_ext)

                # Original galaxies
                ax.scatter(l_rad_plot[~is_clone], b_rad_plot[~is_clone], s=1, alpha=0.2, c='blue', label=f'Original (n={np.sum(~is_clone)})')
                # Clones
                ax.scatter(l_rad_plot[is_clone], b_rad_plot[is_clone], s=3, alpha=0.7, c='red', label=f'Clones (n={is_clone.sum()})')

                # Mark ZoA boundaries
                b_lines = [5, -5, 10, -10]
                for b_line in b_lines:
                    ax.axhline(np.deg2rad(b_line), color='orange' if abs(b_line) == 5 else 'green',
                              ls='--' if abs(b_line) == 5 else ':', alpha=0.5)
                ax.axhline(0, color='gray', ls='-', alpha=0.3)

                ax.set_title('ZoA Cloning: Original (blue) vs Clones (red)')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('zoa_cloning_diagnostic.png', dpi=150)
                plt.close()
                print(f"    Saved zoa_cloning_diagnostic.png")

            # Scatter plot of combined weights vs r (each point = galaxy, excluding masked)
            # Side-by-side comparison with aquila weights
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Split by 2MRS vs deep, excluding masked galaxies (2MRS beyond 125)
            is_2mrs_gal = is_2mrs_only
            is_masked = is_2mrs_gal & (r_mpc > R_2MRS_CUTOFF)  # 2MRS galaxies beyond 125
            w_combined = w_ang * w_L  # Angular x luminosity weight
            # 2MRS galaxies within 125
            sel_2mrs = is_2mrs_gal & ~is_masked
            # Deep galaxies (all used)
            sel_deep = ~is_2mrs_gal

            # Left panel: Computed w_ang * w_L
            ax = axes[0]
            ax.scatter(r_mpc[sel_2mrs], w_combined[sel_2mrs], s=1, alpha=0.3, c='blue', label=f'2MRS (m=11.5, n={sel_2mrs.sum()})')
            ax.scatter(r_mpc[sel_deep], w_combined[sel_deep], s=1, alpha=0.3, c='red', label=f'Deep (m=12.5, n={sel_deep.sum()})')
            ax.axvline(R_2MRS_CUTOFF, color='orange', ls='--', lw=2, label='2MRS cutoff')
            ax.set_xlabel('r [Mpc/h]')
            ax.set_ylabel(r'$w_{ang} \times w_L$ (angular $\times$ luminosity weight)')
            ax.set_title(f'Computed $w_{{ang}} \\times w_L$ (excluding {is_masked.sum()} masked)')
            ax.legend()
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 10.5)
            ax.grid(True, alpha=0.3)

            # Right panel: Aquila weight column for comparison
            ax = axes[1]
            try:
                aquila = np.load("data/Carrick_reconstruction_2015/2m++_0Runs.npy")
                aquila_weight = aquila['weight']
                aquila_r = aquila['distance'] / 100.0
                aquila_K = aquila['K2MRS']
                aquila_is_2mrs = aquila_K < 11.75
                aquila_sel_2mrs = aquila_is_2mrs & (aquila_weight > 0) & (aquila_r < R_2MRS_CUTOFF)
                aquila_sel_deep = ~aquila_is_2mrs & (aquila_weight > 0) & (aquila_r < 200)

                ax.scatter(aquila_r[aquila_sel_2mrs], aquila_weight[aquila_sel_2mrs], s=1, alpha=0.3, c='blue',
                          label=f'2MRS (n={aquila_sel_2mrs.sum()})')
                ax.scatter(aquila_r[aquila_sel_deep], aquila_weight[aquila_sel_deep], s=1, alpha=0.3, c='red',
                          label=f'Deep (n={aquila_sel_deep.sum()})')
                ax.axvline(R_2MRS_CUTOFF, color='orange', ls='--', lw=2, label='2MRS cutoff')
                ax.set_xlabel('r [Mpc/h]')
                ax.set_ylabel('Aquila weight')
                ax.set_title('Aquila weight column')
                ax.legend()
                ax.set_xlim(0, 200)
                ax.set_ylim(0, 10.5)
                ax.grid(True, alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f'Could not load aquila data:\n{e}',
                       transform=ax.transAxes, ha='center', va='center')

            plt.tight_layout()
            plt.savefig('weights_vs_r.png', dpi=100)
            plt.close()
            print(f"    Saved weights_vs_r.png")

            # Histogram of w_ang for 2MRS galaxies
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(w_ang[sel_2mrs], bins=100, alpha=0.7, label=f'2MRS (n={sel_2mrs.sum()})')
            ax.set_xlabel(r'$w_{ang} = 1/\mathrm{completeness}$')
            ax.set_ylabel('Count')
            ax.set_title('Angular completeness weights for 2MRS galaxies')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('w_ang_hist_2mrs.png', dpi=100)
            plt.close()
            print(f"    Saved w_ang_hist_2mrs.png")

            # Print w_ang statistics
            print(f"    w_ang for 2MRS: min={w_ang[sel_2mrs].min():.3f}, max={w_ang[sel_2mrs].max():.3f}, "
                  f"median={np.median(w_ang[sel_2mrs]):.3f}")
            print(f"    w_ang < 1.05: {(w_ang[sel_2mrs] < 1.05).sum()} ({100*(w_ang[sel_2mrs] < 1.05).sum()/sel_2mrs.sum():.1f}%)")
            print(f"    w_ang >= 1.5: {(w_ang[sel_2mrs] >= 1.5).sum()} ({100*(w_ang[sel_2mrs] >= 1.5).sum()/sel_2mrs.sum():.1f}%)")
            # Check if bimodality is from completeness
            print(f"    completeness for 2MRS: min={comp[sel_2mrs].min():.3f}, max={comp[sel_2mrs].max():.3f}")

            # Sky plot of all galaxies in galactic coordinates
            fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})

            # All galaxies (not masked)
            all_gals = sel_2mrs | sel_deep
            l_rad_all = np.deg2rad(l_deg)
            b_rad_all = np.deg2rad(b_deg)
            # Wrap l to [-pi, pi] for mollweide
            l_rad_all = np.where(l_rad_all > np.pi, l_rad_all - 2*np.pi, l_rad_all)

            # Deep galaxies in green
            ax.scatter(l_rad_all[sel_deep], b_rad_all[sel_deep], s=1, alpha=0.3, c='green', label=f'Deep (n={sel_deep.sum()})')

            # 2MRS galaxies colored by w_ang (orange to red colormap)
            w_ang_2mrs = w_ang[sel_2mrs]
            sc = ax.scatter(l_rad_all[sel_2mrs], b_rad_all[sel_2mrs], s=3, alpha=0.7,
                           c=w_ang_2mrs, cmap='YlOrRd', vmin=1.0, vmax=2.0, label=f'2MRS (n={sel_2mrs.sum()})')
            cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, shrink=0.5)
            cbar.set_label(r'$w_{ang}$ (2MRS)')

            ax.axhline(0, color='gray', ls='--', alpha=0.5)
            ax.set_title('All galaxies in Galactic coords: Deep (green), 2MRS (colored by w_ang)')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('high_wang_sky.png', dpi=100)
            plt.close()
            print(f"    Saved high_wang_sky.png")

            # Print |b| statistics to confirm ZoA
            high_wang = w_ang[sel_2mrs] >= 1.5
            low_wang = w_ang[sel_2mrs] < 1.5
            b_high = np.abs(b_deg[sel_2mrs][high_wang])
            b_low = np.abs(b_deg[sel_2mrs][low_wang])
            print(f"    |b| for high w_ang: mean={b_high.mean():.1f}, median={np.median(b_high):.1f}, <10deg: {(b_high < 10).sum()} ({100*(b_high < 10).sum()/len(b_high):.1f}%)")
            print(f"    |b| for low w_ang:  mean={b_low.mean():.1f}, median={np.median(b_low):.1f}, <10deg: {(b_low < 10).sum()} ({100*(b_low < 10).sum()/len(b_low):.1f}%)")

        # Load Carrick for comparison
        delta_carrick = np.load("data/fields/carrick2015_twompp_density.npy")

        # Compute radial profiles for All, 2MRS-only, and Deep regions
        r_edges = np.linspace(0, RMAX, 21)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        n_bins = len(r_centers)

        # Masks for different regions
        mask_all = nonmasked
        mask_2mrs = nonmasked & is_2mrs_only_3d
        mask_deep = nonmasked & is_deep_3d

        # Arrays for mean and std
        mean_ours_all, mean_carrick_all = np.zeros(n_bins), np.zeros(n_bins)
        mean_ours_2mrs, mean_carrick_2mrs = np.zeros(n_bins), np.zeros(n_bins)
        mean_ours_deep, mean_carrick_deep = np.zeros(n_bins), np.zeros(n_bins)
        std_ours_all, std_carrick_all = np.zeros(n_bins), np.zeros(n_bins)
        std_ours_2mrs, std_carrick_2mrs = np.zeros(n_bins), np.zeros(n_bins)
        std_ours_deep, std_carrick_deep = np.zeros(n_bins), np.zeros(n_bins)

        for i_r in range(n_bins):
            shell_base = (rr >= r_edges[i_r]) & (rr < r_edges[i_r+1])

            shell = shell_base & mask_all
            if np.any(shell):
                mean_ours_all[i_r] = np.mean(1 + delta[shell])
                mean_carrick_all[i_r] = np.mean(1 + delta_carrick[shell])
                std_ours_all[i_r] = np.std(delta[shell])
                std_carrick_all[i_r] = np.std(delta_carrick[shell])

            shell = shell_base & mask_2mrs
            if np.any(shell):
                mean_ours_2mrs[i_r] = np.mean(1 + delta[shell])
                mean_carrick_2mrs[i_r] = np.mean(1 + delta_carrick[shell])
                std_ours_2mrs[i_r] = np.std(delta[shell])
                std_carrick_2mrs[i_r] = np.std(delta_carrick[shell])

            shell = shell_base & mask_deep
            if np.any(shell):
                mean_ours_deep[i_r] = np.mean(1 + delta[shell])
                mean_carrick_deep[i_r] = np.mean(1 + delta_carrick[shell])
                std_ours_deep[i_r] = np.std(delta[shell])
                std_carrick_deep[i_r] = np.std(delta_carrick[shell])

        # Normalize means
        norm_ours = np.mean(mean_ours_all[mean_ours_all > 0])
        norm_carrick = np.mean(mean_carrick_all[mean_carrick_all > 0])

        # Plot 2x3 grid: (mean, std) x (all, 2MRS, deep)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        titles = ['All regions', '2MRS only (m<11.5)', 'Deep only (m<12.5)']
        mean_data = [(mean_ours_all, mean_carrick_all),
                     (mean_ours_2mrs, mean_carrick_2mrs),
                     (mean_ours_deep, mean_carrick_deep)]
        std_data = [(std_ours_all, std_carrick_all),
                    (std_ours_2mrs, std_carrick_2mrs),
                    (std_ours_deep, std_carrick_deep)]

        for col, (title, (m_ours, m_car), (s_ours, s_car)) in enumerate(zip(titles, mean_data, std_data)):
            # Mean plot (top row)
            ax = axes[0, col]
            valid = m_ours > 0
            ax.plot(r_centers[valid], m_ours[valid] / norm_ours, 'b-', lw=2, label='Ours')
            ax.plot(r_centers[valid], m_car[valid] / norm_carrick, 'r--', lw=2, label='Carrick')
            ax.axhline(1, color='gray', ls=':', alpha=0.5)
            ax.axvline(R_2MRS_CUTOFF, color='orange', ls='--', alpha=0.7)
            ax.set_xlabel('r [Mpc/h]')
            ax.set_ylabel(r'Mean $\rho / \langle\rho\rangle$')
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 200)
            ax.set_ylim(0.5, 2.0)
            ax.grid(True, alpha=0.3)

            # Std plot (bottom row)
            ax = axes[1, col]
            valid = s_ours > 0
            std_carrick_nonmasked = np.std(delta_carrick[nonmasked])
            ax.plot(r_centers[valid], s_ours[valid], 'b-', lw=2, label=f'Ours (total={delta_std:.3f})')
            ax.plot(r_centers[valid], s_car[valid], 'r--', lw=2, label=f'Carrick (total={std_carrick_nonmasked:.3f})')
            ax.axvline(R_2MRS_CUTOFF, color='orange', ls='--', alpha=0.7)
            ax.set_xlabel('r [Mpc/h]')
            ax.set_ylabel(r'Std($\delta$)')
            ax.legend(fontsize=8)
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 3)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Iteration {i_iter}/{N_ITERATIONS}, beta={beta:.2f}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'iter_{i_iter:03d}_profile.png', dpi=100)
        plt.close()

        # Plot supergalactic slice
        from scipy.interpolate import RegularGridInterpolator
        from astropy.coordinates import CartesianRepresentation, Galactic, Supergalactic

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

        interp_ours = RegularGridInterpolator((coords, coords, coords), delta, bounds_error=False, fill_value=np.nan)
        interp_carrick = RegularGridInterpolator((coords, coords, coords), delta_carrick, bounds_error=False, fill_value=np.nan)

        sg_extent = 200.0
        n_pix = 400
        sg_coords = np.linspace(-sg_extent, sg_extent, n_pix)
        SGX, SGY = np.meshgrid(sg_coords, sg_coords, indexing='ij')
        SGZ = np.zeros_like(SGX)
        sg_pos = np.stack([SGX, SGY, SGZ], axis=-1)
        gal_pos = np.einsum('ij,...j->...i', R_sg2gal, sg_pos)

        slice_ours = interp_ours(gal_pos)
        slice_carrick = interp_carrick(gal_pos)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        vmin, vmax = -1.0, 15.0
        extent = [-sg_extent, sg_extent, -sg_extent, sg_extent]

        ax = axes[0]
        im = ax.imshow(slice_ours.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
        ax.set_xlabel('SGX [Mpc/h]')
        ax.set_ylabel('SGY [Mpc/h]')
        ax.set_title(f'Ours iter {i_iter}, std={delta_std:.3f}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

        ax = axes[1]
        im = ax.imshow(slice_carrick.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
        ax.set_xlabel('SGX [Mpc/h]')
        ax.set_ylabel('SGY [Mpc/h]')
        ax.set_title(f'Carrick2015, std={np.std(delta_carrick[nonmasked]):.3f}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(f'iter_{i_iter:03d}_SGP.png', dpi=100)
        plt.close()

        print(f"    Saved iter_{i_iter:03d}_profile.png and iter_{i_iter:03d}_SGP.png")

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
print(f"  Carrick: min={np.min(delta_carrick):.2f}, max={np.max(delta_carrick):.2f}, std={np.std(delta_carrick[nonmasked]):.3f}")

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
vmin, vmax = -1.0, 15.0
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
