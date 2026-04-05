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
"""Megamaser disk forward model for H0 inference.

Implements the warped Keplerian disk model from Pesce et al. (2020),
arXiv:2001.04581. All JAX functions are JIT-compilable and auto-differentiable.

Per-spot phi is always marginalised numerically on an arcsin-spaced
grid. For high-velocity spots, a reflection trick exploits
the V(phi) = V(pi - phi) symmetry to halve the velocity evaluations.
Optionally, r can also be marginalised on a log-spaced grid, eliminating
all per-spot latent variables. See candel/model/phi_marginalisation.md.

All operations are fully batched over spots — no vmap or lax.scan.
All angles are in RADIANS inside physics functions.
"""
import numpy as _np
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm
from numpyro import factor, handlers, plate, sample
from numpyro.distributions import Uniform
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize_scalar

from ..util import SPEED_OF_LIGHT, fprint, fsection, get_nested
from .base_model import ModelBase
from .pv_utils import rsample
from .integration import ln_trapz_precomputed, trapz_log_weights


# -----------------------------------------------------------------------
# Disk physics constants
# -----------------------------------------------------------------------

C_v = 0.9420       # km/s: sqrt(GM_sun / (1 mas * 1 Mpc)) * 1e-3
C_a = 1.872e-4     # km/s/yr: GM_sun * yr / ((1 mas * 1 Mpc)^2 * 1e3)
C_g = 1.974e-11    # dimensionless: 2*GM_sun / (c^2 * 1 mas * 1 Mpc)
LOG_2PI = 1.8378770664093453  # jnp.log(2 * pi), precomputed


# -----------------------------------------------------------------------
# Spot classification and disk PA estimation
# -----------------------------------------------------------------------


def classify_spots(v, n_clusters=3):
    """Classify spots into blue, systemic, red via k-means on velocity."""
    centroids, labels = kmeans2(v.astype(_np.float64), n_clusters, minit="++")
    order = _np.argsort(centroids)
    remap = _np.empty(n_clusters, dtype=int)
    remap[order] = _np.arange(n_clusters)
    labels = remap[labels]  # 0=blue, 1=systemic, 2=red
    return labels


def estimate_omega(x, y, v, is_hv):
    """Estimate disk PA by maximising |corr(impact, v)| for HV spots."""
    xh, yh, vh = x[is_hv], y[is_hv], v[is_hv]

    def neg_abs_corr(omega):
        b = xh * _np.sin(omega) + yh * _np.cos(omega)
        return -_np.abs(_np.corrcoef(b, vh)[0, 1])

    result = minimize_scalar(neg_abs_corr, bounds=(0, _np.pi),
                             method="bounded")
    # Pick the sign so that positive impact -> higher velocity (receding)
    omega = result.x
    b = xh * _np.sin(omega) + yh * _np.cos(omega)
    if _np.corrcoef(b, vh)[0, 1] < 0:
        omega = (omega + _np.pi) % (2 * _np.pi)
    return omega


# -----------------------------------------------------------------------
# Grid construction (pure numpy, called once at init)
# -----------------------------------------------------------------------


def _build_phi_half_grid_hv(G_half=251, s_min=0.0001, s_max=0.999,
                            n_patch=8):
    """Arcsin-spaced half-grid on [0, pi/2] for HV spots."""
    s = _np.linspace(s_min, s_max, G_half)
    phi = _np.arcsin(s)
    phi_cut = phi[-(n_patch + 1)]
    phi[-n_patch:] = _np.linspace(phi_cut, _np.pi / 2, n_patch + 2)[1:-1]
    phi = _np.append(phi, _np.pi / 2)
    return phi


def _build_phi_grid_sys(G=501, s_max=0.999, n_patch=10):
    """Arcsin-spaced grid on [-pi/2, pi/2] for systemic spots."""
    s = _np.linspace(-s_max, s_max, G)
    phi = _np.arcsin(s)
    phi_cut_lo = phi[n_patch]
    phi[:n_patch] = _np.linspace(-_np.pi / 2, phi_cut_lo, n_patch + 2)[1:-1]
    phi_cut_hi = phi[-(n_patch + 1)]
    phi[-n_patch:] = _np.linspace(phi_cut_hi, _np.pi / 2, n_patch + 2)[1:-1]
    phi = _np.concatenate([[-_np.pi / 2], phi, [_np.pi / 2]])
    return phi


def _build_r_grid(r_min, r_max, n_r=251):
    """Log-spaced radius grid."""
    return _np.logspace(_np.log10(r_min), _np.log10(r_max), n_r)




# -----------------------------------------------------------------------
# Disk physics functions
# -----------------------------------------------------------------------


# Conversion: 1 mas at 1 Mpc = 4.848e-3 pc
PC_PER_MAS_MPC = 4.848e-3


def warp_geometry(r_ang, r_ang_ref, i0_rad, di_dr_rad,
                  Omega0_rad, dOmega_dr_rad):
    """Evaluate warped inclination and position angle at angular radius.

    The expansion is about r_ang_ref (in mas), so i0 and Omega0 are
    the values at that angular radius. The warp rates di/dr and
    dOmega/dr are in radians per mas.

    Parameters
    ----------
    r_ang : angular radius in mas
    r_ang_ref : reference angular radius in mas (expansion centre)
    i0_rad, di_dr_rad : inclination at r_ang_ref and warp rate (rad/mas)
    Omega0_rad, dOmega_dr_rad : position angle at r_ang_ref and
        warp rate (rad/mas)

    Returns
    -------
    i, Omega : inclination and position angle at r_ang, in radians
    """
    dr = r_ang - r_ang_ref
    i = i0_rad + di_dr_rad * dr
    Omega = Omega0_rad + dOmega_dr_rad * dr
    return i, Omega


def predict_position(r_ang, phi, x0, y0, i, Omega):
    """Predict sky-plane position of maser spots.

    Note: used by the mock generator. The inference hot path uses
    the fused ``_compute_observables`` instead.

    All angles (phi, i, Omega) in radians. Positions in mas.

    Returns
    -------
    X, Y : predicted sky-plane coordinates in mas
    """
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    sin_O = jnp.sin(Omega)
    cos_O = jnp.cos(Omega)
    cos_i = jnp.cos(i)

    X = x0 + r_ang * (sin_phi * sin_O - cos_phi * cos_O * cos_i)
    Y = y0 + r_ang * (sin_phi * cos_O + cos_phi * sin_O * cos_i)
    return X, Y


def predict_velocity_los(r_ang, phi, D, M_BH, v_sys, i):
    """Predict line-of-sight velocity of maser spots (optical convention).

    Note: used by the mock generator. The inference hot path uses
    the fused ``_compute_observables`` instead.

    Includes Keplerian orbital velocity, relativistic Doppler, gravitational
    redshift, and systemic redshift.

    Parameters
    ----------
    r_ang : orbital radius in mas
    phi : azimuthal angle in radians
    D : angular-diameter distance in Mpc
    M_BH : black hole mass in M_sun
    v_sys : systemic velocity in km/s
    i : inclination in radians

    Returns
    -------
    V_obs : observed velocity in km/s (optical convention)
    """
    v_kep = C_v * jnp.sqrt(M_BH / (r_ang * D))

    v_z = v_kep * jnp.sin(phi) * jnp.sin(i)

    beta = v_kep / SPEED_OF_LIGHT
    gamma = 1.0 / jnp.sqrt(1.0 - beta**2)
    one_plus_z_D = gamma * (1.0 + v_z / SPEED_OF_LIGHT)

    one_plus_z_g = 1.0 / jnp.sqrt(1.0 - C_g * M_BH / (r_ang * D))

    z_0 = v_sys / SPEED_OF_LIGHT

    V_obs = SPEED_OF_LIGHT * (
        one_plus_z_D * one_plus_z_g * (1.0 + z_0) - 1.0)
    return V_obs


def predict_acceleration_los(r_ang, phi, D, M_BH, i):
    """Predict line-of-sight centripetal acceleration.

    Note: used by the mock generator. The inference hot path uses
    the fused ``_compute_observables`` instead.

    Parameters
    ----------
    r_ang : orbital radius in mas
    phi : azimuthal angle in radians
    D : angular-diameter distance in Mpc
    M_BH : black hole mass in M_sun
    i : inclination in radians

    Returns
    -------
    A_z : line-of-sight acceleration in km/s/yr
    """
    a_mag = C_a * M_BH / (r_ang**2 * D**2)
    A_z = a_mag * jnp.cos(phi) * jnp.sin(i)
    return A_z


def _compute_observables(r_ang, sin_phi, cos_phi, x0, y0, D, M_BH,
                         v_sys, sin_i, cos_i, sin_O, cos_O):
    """Fused computation of all observables. Avoids duplicate trig/sqrt.

    All outputs broadcast to the shape of (r_ang * sin_phi).
    """
    # Position
    X = x0 + r_ang * (sin_phi * sin_O - cos_phi * cos_O * cos_i)
    Y = y0 + r_ang * (sin_phi * cos_O + cos_phi * sin_O * cos_i)

    # Velocity (reuse v_kep for acceleration)
    rD = r_ang * D
    v_kep = C_v * jnp.sqrt(M_BH / rD)
    v_z = v_kep * sin_phi * sin_i
    beta = v_kep / SPEED_OF_LIGHT
    gamma = 1.0 / jnp.sqrt(1.0 - beta * beta)
    one_plus_z_D = gamma * (1.0 + v_z / SPEED_OF_LIGHT)
    one_plus_z_g = 1.0 / jnp.sqrt(1.0 - C_g * M_BH / rD)
    V = SPEED_OF_LIGHT * (
        one_plus_z_D * one_plus_z_g * (1.0 + v_sys / SPEED_OF_LIGHT)
        - 1.0)

    # Acceleration (reuse v_kep^2 / rD = GM / r^2 D^2 in code units)
    a_mag = v_kep * v_kep / rD * (C_a / (C_v * C_v))
    A = a_mag * cos_phi * sin_i

    return X, Y, V, A


def _precompute_r_quantities(r_ang, D, M_BH, sin_i, cos_i, sin_O, cos_O):
    """Precompute r-dependent quantities for the phi integration.

    Returns quantities that depend only on r (not phi), to be
    broadcast into the phi dimension by the caller. This avoids
    3 expensive sqrt calls per (r, phi) grid point.

    All inputs/outputs have shape (..., N_r) or broadcastable.
    """
    rD = r_ang * D
    v_kep = C_v * jnp.sqrt(M_BH / rD)  # 1 sqrt

    beta = v_kep / SPEED_OF_LIGHT
    gamma = 1.0 / jnp.sqrt(1.0 - beta * beta)  # 1 sqrt

    one_plus_z_g = 1.0 / jnp.sqrt(1.0 - C_g * M_BH / rD)  # 1 sqrt

    a_mag = v_kep * v_kep / rD * (C_a / (C_v * C_v))

    # Precompute position projection coefficients (multiply-add only).
    # X = x0 + r * (sin_phi * pA - cos_phi * pB)
    # Y = y0 + r * (sin_phi * pC + cos_phi * pD)
    pA = sin_O
    pB = cos_O * cos_i
    pC = cos_O
    pD = sin_O * cos_i

    return v_kep, gamma, one_plus_z_g, a_mag, pA, pB, pC, pD


def _observables_from_precomputed(sin_phi, cos_phi, x0, y0, v_sys,
                                  sin_i, r_ang,
                                  v_kep, gamma, one_plus_z_g, a_mag,
                                  pA, pB, pC, pD):
    """Compute observables using precomputed r-dependent quantities.

    Only multiply-add operations — no sqrt, no division.
    """
    X = x0 + r_ang * (sin_phi * pA - cos_phi * pB)
    Y = y0 + r_ang * (sin_phi * pC + cos_phi * pD)

    v_z = v_kep * sin_phi * sin_i
    one_plus_z_D = gamma * (1.0 + v_z / SPEED_OF_LIGHT)
    V = SPEED_OF_LIGHT * (
        one_plus_z_D * one_plus_z_g * (1.0 + v_sys / SPEED_OF_LIGHT)
        - 1.0)

    A = a_mag * cos_phi * sin_i

    return X, Y, V, A


def _observables_no_accel(sin_phi, cos_phi, x0, y0, v_sys,
                          sin_i, r_ang, v_kep, gamma, one_plus_z_g,
                          pA, pB, pC, pD):
    """Position + velocity only. Skips acceleration entirely for spots
    without measured acceleration."""
    X = x0 + r_ang * (sin_phi * pA - cos_phi * pB)
    Y = y0 + r_ang * (sin_phi * pC + cos_phi * pD)

    v_z = v_kep * sin_phi * sin_i
    one_plus_z_D = gamma * (1.0 + v_z / SPEED_OF_LIGHT)
    V = SPEED_OF_LIGHT * (
        one_plus_z_D * one_plus_z_g * (1.0 + v_sys / SPEED_OF_LIGHT)
        - 1.0)

    return X, Y, V


def _chi2_4obs(x_obs, X, inv_var_x, y_obs, Y, inv_var_y,
               v_obs, V, inv_var_v, a_obs, A, inv_var_a):
    """Sum of 4 chi-squared terms. No log(var) — that's added per-spot."""
    dx = x_obs - X
    dy = y_obs - Y
    dv = v_obs - V
    da = a_obs - A
    return (dx * dx * inv_var_x + dy * dy * inv_var_y
            + dv * dv * inv_var_v + da * da * inv_var_a)


def _chi2_3obs(x_obs, X, inv_var_x, y_obs, Y, inv_var_y,
               v_obs, V, inv_var_v):
    """Sum of 3 chi-squared terms (no acceleration)."""
    dx = x_obs - X
    dy = y_obs - Y
    dv = v_obs - V
    return (dx * dx * inv_var_x + dy * dy * inv_var_y
            + dv * dv * inv_var_v)


def _stable_log_2cosh(x):
    """log(2*cosh(x)), numerically stable for large |x|.

    = log(exp(x) + exp(-x)) = |x| + log(1 + exp(-2|x|))
    = |x| + softplus(-2|x|)
    """
    return jnp.abs(x) + jnp.log1p(jnp.exp(-2.0 * jnp.abs(x)))


def _hv_reflected_ll(x_obs, y_obs, v_obs, a_obs,
                     inv_vx, inv_vy, inv_vv, inv_va,
                     Xmid, Xdel, Ymid, Ydel, V, A1):
    """Log-likelihood for HV reflected pair using log-cosh trick.

    For the two reflected phi modes (phi, pi-phi), positions differ by
    ±delta while velocity and |acceleration| are shared. Instead of
    computing two full chi² + logaddexp, we decompose into:

        chi2_1 = chi2_v + S + C     (mode 1)
        chi2_2 = chi2_v + S - C     (mode 2)

    where S is the symmetric part and C the cross term. Then:

        logaddexp(-0.5*chi2_1, -0.5*chi2_2)
            = -0.5*(chi2_v + S) + log(2*cosh(0.5*C))

    This avoids computing X1,X2,Y1,Y2 and two separate chi² calls.
    """
    dx_mid = x_obs - Xmid
    dy_mid = y_obs - Ymid

    # Symmetric part: (dx_mid² + Xdel²)*inv_vx + (dy_mid² + Ydel²)*inv_vy
    S = ((dx_mid * dx_mid + Xdel * Xdel) * inv_vx
         + (dy_mid * dy_mid + Ydel * Ydel) * inv_vy)

    # Cross term: 2*(dx_mid*Xdel*inv_vx - dy_mid*Ydel*inv_vy)
    C = 2.0 * (dx_mid * Xdel * inv_vx - dy_mid * Ydel * inv_vy)

    # Add acceleration if measured (inv_va > 0).
    # A1 = a_mag*cos_phi*sin_i, A2 = -A1.
    # chi2_a1 = (a_obs-A1)², chi2_a2 = (a_obs+A1)²
    # S_a = (a_obs² + A1²)*inv_va, C_a = -2*a_obs*A1*inv_va
    S = S + (a_obs * a_obs + A1 * A1) * inv_va
    C = C - 2.0 * a_obs * A1 * inv_va

    chi2_v = (v_obs - V) ** 2 * inv_vv

    return -0.5 * (chi2_v + S) + _stable_log_2cosh(0.5 * C)


def _spot_ll_syst(r_ang, sin_phi, cos_phi,
                  x_obs, y_obs, v_obs, a_obs,
                  inv_var_x, inv_var_y, inv_var_v, inv_var_a,
                  log_norm,
                  x0, y0, D, M_BH, v_sys,
                  sin_i, cos_i, sin_O, cos_O):
    """Per-spot log-likelihood for systemic spots.

    Warp trig (sin_i, cos_i, sin_O, cos_O) is precomputed on the
    (N, N_r) grid by the caller, avoiding redundant recomputation
    across the phi grid.

    Returns
    -------
    ll : (N_sys, ..., G_sys) per-spot per-phi log-likelihood
    """
    X, Y, V, A = _compute_observables(
        r_ang, sin_phi, cos_phi, x0, y0, D, M_BH, v_sys,
        sin_i, cos_i, sin_O, cos_O)

    chi2 = _chi2_4obs(x_obs, X, inv_var_x, y_obs, Y, inv_var_y,
                      v_obs, V, inv_var_v, a_obs, A, inv_var_a)
    return log_norm - 0.5 * chi2


def _spot_ll_hv(r_ang,
                sin_phi1, cos_phi1, sin_phi2, cos_phi2,
                x_obs, y_obs, v_obs, a_obs,
                inv_var_x, inv_var_y, inv_var_v, inv_var_a,
                log_norm,
                x0, y0, D, M_BH, v_sys,
                sin_i, cos_i, sin_O, cos_O):
    """Log-likelihood for a pair of reflected phi modes.

    Velocity is computed once (sin(phi1) == sin(phi2) by construction).
    Warp trig is precomputed by the caller.

    Returns
    -------
    ll : (N_hv, ..., G_hv) per-spot per-phi log-likelihood
    """
    # Velocity: same for both modes (sin_phi1 == sin_phi2)
    X1, Y1, V, A1 = _compute_observables(
        r_ang, sin_phi1, cos_phi1, x0, y0, D, M_BH, v_sys,
        sin_i, cos_i, sin_O, cos_O)

    # Mode 2: only position and acceleration differ
    X2 = x0 + r_ang * (sin_phi2 * sin_O - cos_phi2 * cos_O * cos_i)
    Y2 = y0 + r_ang * (sin_phi2 * cos_O + cos_phi2 * sin_O * cos_i)
    A2 = -A1

    chi2_v = (v_obs - V) ** 2 * inv_var_v

    chi2_1 = (_chi2_4obs(x_obs, X1, inv_var_x, y_obs, Y1, inv_var_y,
                         v_obs, V, inv_var_v, a_obs, A1, inv_var_a)
              - chi2_v)  # pos + accel only
    chi2_2 = (_chi2_4obs(x_obs, X2, inv_var_x, y_obs, Y2, inv_var_y,
                         v_obs, V, inv_var_v, a_obs, A2, inv_var_a)
              - chi2_v)  # pos + accel only

    # logaddexp avoids materialising a stacked (2, ...) array
    ll = (log_norm - 0.5 * chi2_v
          + jnp.logaddexp(-0.5 * chi2_1, -0.5 * chi2_2))
    return ll


# -----------------------------------------------------------------------
# Model classes
# -----------------------------------------------------------------------


class MaserDiskModel(ModelBase):
    """Megamaser disk H0 model with marginalised per-spot nuisance params.

    Phi is always marginalised on an arcsin-spaced grid.
    If marginalise_r is True, r is also marginalised on a log grid,
    leaving only ~16 global parameters for NUTS.
    """

    def __init__(self, config_path, data):
        super().__init__(config_path)
        fsection("Maser Disk Model")
        self._load_and_set_priors()

        self.n_spots = data["n_spots"]
        self.is_highvel = jnp.asarray(data["is_highvel"])

        # Default per-spot velocity measurement error if not provided
        if "sigma_v" not in data:
            data["sigma_v"] = _np.ones(data["n_spots"])
            fprint("sigma_v not in data, defaulting to 1 km/s.")

        self._set_data_arrays(
            data, skip_keys=("accel_measured", "is_highvel", "is_systemic",
                             "is_blue", "is_red", "spot_type",
                             "phi_lo", "phi_hi",
                             "n_spots", "galaxy_name", "v_sys_obs"))

        if "v_sys_obs" not in data:
            raise ValueError(
                "data must contain 'v_sys_obs' (CMB-frame recession "
                "velocity in km/s).")
        self.v_sys_obs = float(data["v_sys_obs"])

        # Spot index arrays for systemic / red / blue subsets
        is_hv_np = _np.asarray(data["is_highvel"])
        is_blue_np = _np.asarray(data.get("is_blue", _np.zeros(
            self.n_spots, dtype=bool)))
        is_red_np = is_hv_np & ~is_blue_np
        self._idx_sys = jnp.where(~self.is_highvel)[0]
        self._idx_red = jnp.where(jnp.asarray(is_red_np))[0]
        self._idx_blue = jnp.where(jnp.asarray(is_blue_np))[0]
        self._n_sys = int((~is_hv_np).sum())
        self._n_red = int(is_red_np.sum())
        self._n_blue = int(is_blue_np.sum())

        # Pre-masked data arrays per subset (avoids indexing during inference)
        # Store sigma^2 directly since only variances are needed downstream.
        for label, idx in [("sys", self._idx_sys),
                           ("red", self._idx_red),
                           ("blue", self._idx_blue)]:
            for key in ("x", "y", "velocity", "a"):
                setattr(self, f"_{key}_{label}",
                        getattr(self, key)[idx])
            for key in ("sigma_x", "sigma_y", "sigma_v", "sigma_a"):
                setattr(self, f"_{key}2_{label}",
                        getattr(self, key)[idx]**2)

        # Split each type into with-accel and without-accel sub-groups.
        # Spots without acceleration get a 3-obs chi² (no A computed).
        accel_meas = _np.asarray(data.get(
            "accel_measured", self.sigma_a < 1e4))
        for label, idx_arr in [("sys", self._idx_sys),
                               ("red", self._idx_red),
                               ("blue", self._idx_blue)]:
            idx_np = _np.asarray(idx_arr)
            has_a = accel_meas[idx_np]
            setattr(self, f"_idx_{label}_a",
                    jnp.asarray(idx_np[has_a]))
            setattr(self, f"_idx_{label}_noa",
                    jnp.asarray(idx_np[~has_a]))
            setattr(self, f"_n_{label}_a", int(has_a.sum()))
            setattr(self, f"_n_{label}_noa", int((~has_a).sum()))
            # Pre-masked data for each sub-group
            for key in ("x", "y", "velocity"):
                arr = getattr(self, key)
                setattr(self, f"_{key}_{label}_a", arr[idx_np[has_a]])
                setattr(self, f"_{key}_{label}_noa", arr[idx_np[~has_a]])
            setattr(self, f"_a_{label}_a",
                    self.a[idx_np[has_a]])
            for key in ("sigma_x", "sigma_y", "sigma_v"):
                arr = getattr(self, key)
                setattr(self, f"_{key}2_{label}_a",
                        arr[idx_np[has_a]]**2)
                setattr(self, f"_{key}2_{label}_noa",
                        arr[idx_np[~has_a]]**2)
            setattr(self, f"_sigma_a2_{label}_a",
                    self.sigma_a[idx_np[has_a]]**2)

        n_a = sum(getattr(self, f"_n_{l}_a") for l in ("sys", "red", "blue"))
        n_noa = sum(getattr(self, f"_n_{l}_noa")
                    for l in ("sys", "red", "blue"))
        fprint(f"accel split: {n_a} with, {n_noa} without.")

        # ---- Phi grids and precomputed trig ----
        phi_half = _build_phi_half_grid_hv()
        phi_sys = _build_phi_grid_sys()

        # Systemic: trig of full grid
        self._sin_phi_sys = jnp.sin(jnp.asarray(phi_sys))
        self._cos_phi_sys = jnp.cos(jnp.asarray(phi_sys))

        # HV: trig of half-grid and reflected grids
        sin_half = jnp.sin(jnp.asarray(phi_half))
        cos_half = jnp.cos(jnp.asarray(phi_half))
        # Red: phi1 = phi_half, phi2 = pi - phi_half
        # sin(pi-x) = sin(x), cos(pi-x) = -cos(x)
        self._sin_phi1_red = sin_half
        self._cos_phi1_red = cos_half
        self._sin_phi2_red = sin_half
        self._cos_phi2_red = -cos_half
        # Blue: phi1 = pi + phi_half, phi2 = 2*pi - phi_half
        # sin(pi+x) = -sin(x), cos(pi+x) = -cos(x)
        # sin(2*pi-x) = -sin(x), cos(2*pi-x) = cos(x)
        self._sin_phi1_blue = -sin_half
        self._cos_phi1_blue = -cos_half
        self._sin_phi2_blue = -sin_half
        self._cos_phi2_blue = cos_half

        self._log_w_phi_hv = jnp.asarray(trapz_log_weights(phi_half))
        self._log_w_phi_sys = jnp.asarray(trapz_log_weights(phi_sys))

        # Inverse permutation for concat+gather (replaces .at[idx].set).
        # Order: sys_a, sys_noa, red_a, red_noa, blue_a, blue_noa
        order = jnp.concatenate([
            self._idx_sys_a, self._idx_sys_noa,
            self._idx_red_a, self._idx_red_noa,
            self._idx_blue_a, self._idx_blue_noa])
        self._inv_order = jnp.argsort(order)

        # ---- Physical radius grid (for Mode 2) ----
        self.marginalise_r = get_nested(
            self.config, "model/marginalise_r", False)
        if self.marginalise_r:
            R_min = float(get_nested(
                self.config, "model/priors/R_phys/low", 0.01))
            R_max = float(get_nested(
                self.config, "model/priors/R_phys/high", 1.5))
            R_grid = _build_r_grid(R_min, R_max)
            self._R_phys_grid = jnp.asarray(R_grid)
            # Trapezoidal weights with flat (uniform) R_phys prior
            self._log_w_R = jnp.asarray(
                trapz_log_weights(jnp.asarray(R_grid)))
            # r_ang grid set after D_A_est is computed below

        # ---- Selection function grid ----
        D_min = float(get_nested(self.config, "model/priors/D/low", 10.0))
        D_max = float(get_nested(self.config, "model/priors/D/high", 200.0))
        self._sel_D_grid = jnp.linspace(D_min, D_max, 501)
        self._sel_log_w = jnp.asarray(trapz_log_weights(self._sel_D_grid))
        self._sel_lp_vol = 2.0 * jnp.log(self._sel_D_grid)
        self.use_selection = get_nested(
            self.config, "model/use_selection", False)

        self.fit_di_dr = get_nested(
            self.config, "model/fit_di_dr", False)

        # Reference angular radius for warp expansion (mas).
        self._r_ang_ref = 0.0

        # Fixed r_ang bounds for Mode 1 (estimated from v_sys_obs).
        # Uses cosmographic D_A to convert R_phys bounds to angular.
        z_est = self.v_sys_obs / SPEED_OF_LIGHT
        q0 = -0.55
        D_c_est = (SPEED_OF_LIGHT * z_est / 73.0
                   * (1 + 0.5 * (1 - q0) * z_est))
        D_A_est = D_c_est / (1 + z_est)
        R_lo = float(get_nested(
            self.config, "model/priors/R_phys/low", 0.01))
        R_hi = float(get_nested(
            self.config, "model/priors/R_phys/high", 1.5))
        self._r_ang_lo = R_lo / (D_A_est * PC_PER_MAS_MPC)
        self._r_ang_hi = R_hi / (D_A_est * PC_PER_MAS_MPC)
        fprint(f"r_ang bounds: [{self._r_ang_lo:.3f}, "
               f"{self._r_ang_hi:.3f}] mas "
               f"(D_A_est = {D_A_est:.1f} Mpc)")

        # Fixed r_ang grid for Mode 2 (from R_phys grid at D_A_est).
        # D_A_est ~ v_sys / (H0 * (1+z)); good enough since the grid
        # just needs to cover the plausible angular radius range.
        if self.marginalise_r:
            self._r_ang_grid = jnp.asarray(
                self._R_phys_grid / (D_A_est * PC_PER_MAS_MPC))
            self._log_w_R = jnp.asarray(
                trapz_log_weights(self._r_ang_grid))
            fprint(f"r_ang grid: [{float(self._r_ang_grid[0]):.4f}, "
                   f"{float(self._r_ang_grid[-1]):.4f}] mas")

        mode = "r+phi" if self.marginalise_r else "phi only"
        fprint(f"loaded {self.n_spots} maser spots "
               f"({self._n_sys} systemic, {self._n_red} red, "
               f"{self._n_blue} blue).")
        fprint(f"marginalisation mode: {mode}")
        fprint(f"phi grids: HV half={len(phi_half)}, "
               f"sys={len(phi_sys)}")
        if self.marginalise_r:
            fprint(f"r_ang grid: {len(self._r_ang_grid)} log-spaced, "
                   f"R_phys in [{R_min:.3f}, {R_max:.3f}] pc")
        fprint(f"use_selection = {self.use_selection}")
        fprint(f"fit_di_dr = {self.fit_di_dr}")

    def _eval_marginal_phi(self, r_ang, x0, y0, D_A, M_BH, v_sys,
                           r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
                           sigma_x_floor2, sigma_y_floor2,
                           var_v_sys, var_v_hv, sigma_a_floor2):
        """Evaluate per-spot log-likelihood marginalised over phi.

        Mode 1 (sample r_ang): r_ang is (N,)
        Mode 2 (marginalise R_phys): r_ang is (N, n_r)

        All sigma/var arguments are pre-squared.

        Returns
        -------
        ll : (N,) or (N, n_r) per-spot marginalised log-likelihood
        """
        # rpad: adds trailing dim for phi broadcasting on r_ang
        # dpad: adds dims so 1D data arrays broadcast with r_ang + phi
        rpad = (slice(None),) * r_ang.ndim + (None,)
        n_extra = r_ang.ndim
        dpad = (slice(None),) + (None,) * n_extra

        # Precompute ALL r-dependent quantities on the (N,) or (N, N_r)
        # grid. The phi loop then only does multiply-add + logsumexp.
        i_r, Omega_r = warp_geometry(
            r_ang, r_ang_ref, i0, di_dr, Omega0, dOmega_dr)
        sin_i = jnp.sin(i_r)
        cos_i = jnp.cos(i_r)
        sin_O = jnp.sin(Omega_r)
        cos_O = jnp.cos(Omega_r)
        v_kep, gamma, z_g_factor, a_mag, pA, pB, pC, pD = \
            _precompute_r_quantities(r_ang, D_A, M_BH,
                                     sin_i, cos_i, sin_O, cos_O)

        def _r_precomp(idx):
            """Slice precomputed r-quantities for a spot subset."""
            return (r_ang[idx][rpad], sin_i[idx][rpad],
                    v_kep[idx][rpad], gamma[idx][rpad],
                    z_g_factor[idx][rpad], a_mag[idx][rpad],
                    pA[idx][rpad], pB[idx][rpad],
                    pC[idx][rpad], pD[idx][rpad])

        def _obs_4(idx, sp, cp):
            """X, Y, V, A from precomputed r-quantities."""
            return _observables_from_precomputed(
                sp, cp, x0, y0, v_sys, *_r_precomp(idx))

        def _obs_3(idx, sp, cp):
            """X, Y, V only — no acceleration computed."""
            (r_sub, si, vk, gm, zg, _, pa, pb, pc, pd) = _r_precomp(idx)
            return _observables_no_accel(
                sp, cp, x0, y0, v_sys, si, r_sub, vk, gm, zg,
                pa, pb, pc, pd)

        def _lnorm_3(sx2, sy2, sv2):
            return -0.5 * (3 * LOG_2PI + jnp.log(sx2 + sigma_x_floor2)
                           + jnp.log(sy2 + sigma_y_floor2)
                           + jnp.log(sv2))

        def _lnorm_4(sx2, sy2, sv2, sa2):
            return (_lnorm_3(sx2, sy2, sv2)
                    - 0.5 * (LOG_2PI + jnp.log(sa2 + sigma_a_floor2)))

        # ---- Systemic spots WITH accel ----
        def _sys_block(idx_attr, x_d, y_d, v_d, a_d,
                       sx2, sy2, sv2, sa2, var_v_floor, has_accel):
            vx = sx2[dpad] + sigma_x_floor2
            vy = sy2[dpad] + sigma_y_floor2
            vv = sv2[dpad] + var_v_floor
            idx = getattr(self, idx_attr)
            if has_accel:
                va = sa2[dpad] + sigma_a_floor2
                X, Y, V, A = _obs_4(idx, self._sin_phi_sys,
                                     self._cos_phi_sys)
                chi2 = _chi2_4obs(
                    x_d[dpad], X, 1.0 / vx, y_d[dpad], Y, 1.0 / vy,
                    v_d[dpad], V, 1.0 / vv, a_d[dpad], A, 1.0 / va)
                lnorm = _lnorm_4(sx2[dpad], sy2[dpad], vv, sa2[dpad])
            else:
                X, Y, V = _obs_3(idx, self._sin_phi_sys,
                                  self._cos_phi_sys)
                chi2 = _chi2_3obs(
                    x_d[dpad], X, 1.0 / vx, y_d[dpad], Y, 1.0 / vy,
                    v_d[dpad], V, 1.0 / vv)
                lnorm = _lnorm_3(sx2[dpad], sy2[dpad], vv)
            return ln_trapz_precomputed(lnorm - 0.5 * chi2,
                                        self._log_w_phi_sys, axis=-1)

        def _hv_block(idx_attr, sp1, cp1, sp2, cp2, log_w_phi,
                      x_d, y_d, v_d, a_d,
                      sx2, sy2, sv2, sa2, var_v_floor, has_accel):
            vx = sx2[dpad] + sigma_x_floor2
            vy = sy2[dpad] + sigma_y_floor2
            vv = sv2[dpad] + var_v_floor
            inv_vx, inv_vy, inv_vv = 1.0 / vx, 1.0 / vy, 1.0 / vv
            idx = getattr(self, idx_attr)
            (r_sub, si, vk, gm, zg, am,
             pa_s, pb_s, pc_s, pd_s) = _r_precomp(idx)

            # Position midpoint ± delta (sp1==sp2, cp2=-cp1).
            Xmid = x0 + r_sub * sp1 * pa_s
            Xdel = r_sub * cp1 * pb_s
            Ymid = y0 + r_sub * sp1 * pc_s
            Ydel = r_sub * cp1 * pd_s

            # Velocity (shared by both modes — depends only on sin_phi).
            v_z = vk * sp1 * si
            one_plus_z_D = gm * (1.0 + v_z / SPEED_OF_LIGHT)
            V = SPEED_OF_LIGHT * (
                one_plus_z_D * zg * (1.0 + v_sys / SPEED_OF_LIGHT)
                - 1.0)

            # Acceleration (only for spots with measurements).
            A1 = am * cp1 * si if has_accel else jnp.zeros(())
            inv_va = 1.0 / (sa2[dpad] + sigma_a_floor2) if has_accel \
                else jnp.zeros(())

            # log-cosh trick replaces logaddexp of two full chi²
            ll = _hv_reflected_ll(
                x_d[dpad], y_d[dpad], v_d[dpad],
                a_d[dpad] if has_accel else jnp.zeros(()),
                inv_vx, inv_vy, inv_vv, inv_va,
                Xmid, Xdel, Ymid, Ydel, V, A1)

            if has_accel:
                lnorm = _lnorm_4(sx2[dpad], sy2[dpad], vv, sa2[dpad])
            else:
                lnorm = _lnorm_3(sx2[dpad], sy2[dpad], vv)

            return ln_trapz_precomputed(
                lnorm + ll, log_w_phi, axis=-1)

        results = []

        # ---- Systemic: with accel, then without ----
        for suffix, has_a in [("_a", True), ("_noa", False)]:
            n = getattr(self, f"_n_sys{suffix}")
            if n > 0:
                kw = dict(
                    x_d=getattr(self, f"_x_sys{suffix}"),
                    y_d=getattr(self, f"_y_sys{suffix}"),
                    v_d=getattr(self, f"_velocity_sys{suffix}"),
                    a_d=getattr(self, f"_a_sys_a", None) if has_a else None,
                    sx2=getattr(self, f"_sigma_x2_sys{suffix}"),
                    sy2=getattr(self, f"_sigma_y2_sys{suffix}"),
                    sv2=getattr(self, f"_sigma_v2_sys{suffix}") + var_v_sys,
                    sa2=getattr(self, f"_sigma_a2_sys_a", None) if has_a
                        else None,
                    var_v_floor=0.0,  # already added above
                    has_accel=has_a)
                results.append(_sys_block(f"_idx_sys{suffix}", **kw))

        # ---- Red HV: with accel, then without ----
        for suffix, has_a in [("_a", True), ("_noa", False)]:
            n = getattr(self, f"_n_red{suffix}")
            if n > 0:
                kw = dict(
                    sp1=self._sin_phi1_red, cp1=self._cos_phi1_red,
                    sp2=self._sin_phi2_red, cp2=self._cos_phi2_red,
                    log_w_phi=self._log_w_phi_hv,
                    x_d=getattr(self, f"_x_red{suffix}"),
                    y_d=getattr(self, f"_y_red{suffix}"),
                    v_d=getattr(self, f"_velocity_red{suffix}"),
                    a_d=getattr(self, f"_a_red_a", None) if has_a else None,
                    sx2=getattr(self, f"_sigma_x2_red{suffix}"),
                    sy2=getattr(self, f"_sigma_y2_red{suffix}"),
                    sv2=getattr(self, f"_sigma_v2_red{suffix}") + var_v_hv,
                    sa2=getattr(self, f"_sigma_a2_red_a", None) if has_a
                        else None,
                    var_v_floor=0.0,
                    has_accel=has_a)
                results.append(_hv_block(f"_idx_red{suffix}", **kw))

        # ---- Blue HV: with accel, then without ----
        for suffix, has_a in [("_a", True), ("_noa", False)]:
            n = getattr(self, f"_n_blue{suffix}")
            if n > 0:
                kw = dict(
                    sp1=self._sin_phi1_blue, cp1=self._cos_phi1_blue,
                    sp2=self._sin_phi2_blue, cp2=self._cos_phi2_blue,
                    log_w_phi=self._log_w_phi_hv,
                    x_d=getattr(self, f"_x_blue{suffix}"),
                    y_d=getattr(self, f"_y_blue{suffix}"),
                    v_d=getattr(self, f"_velocity_blue{suffix}"),
                    a_d=getattr(self, f"_a_blue_a", None) if has_a
                        else None,
                    sx2=getattr(self, f"_sigma_x2_blue{suffix}"),
                    sy2=getattr(self, f"_sigma_y2_blue{suffix}"),
                    sv2=getattr(self, f"_sigma_v2_blue{suffix}") + var_v_hv,
                    sa2=getattr(self, f"_sigma_a2_blue_a", None) if has_a
                        else None,
                    var_v_floor=0.0,
                    has_accel=has_a)
                results.append(_hv_block(f"_idx_blue{suffix}", **kw))

        # Concat + gather replaces .at[idx].set() scatter ops
        return jnp.concatenate(results, axis=0)[self._inv_order]

    def _sample_galaxy(self, shared_params, h):
        """Sample all per-galaxy parameters and accumulate log-likelihood.

        Parameters
        ----------
        shared_params : dict
            Shared parameters (H0, sigma_pec, and optionally D_lim, D_width)
            passed through to rsample.
        h : reduced Hubble constant (H0 / 100)
        """
        D_c = rsample("D_c", self.priors["D"], shared_params)

        z_cosmo = self.distance2redshift(
            jnp.atleast_1d(D_c), h=h).squeeze()
        D_A = D_c / (1 + z_cosmo)

        log_MBH = rsample("log_MBH", self.priors["log_MBH"], shared_params)
        M_BH = 10.0**log_MBH
        # x0, y0 sampled in uas, converted to mas for physics
        x0 = rsample("x0", self.priors["x0"], shared_params) * 1e-3
        y0 = rsample("y0", self.priors["y0"], shared_params) * 1e-3

        i0_deg = rsample("i0", self.priors["i0"], shared_params)
        Omega0_deg = rsample("Omega0", self.priors["Omega0"], shared_params)
        dOmega_dr_deg = rsample(
            "dOmega_dr", self.priors["dOmega_dr"], shared_params)

        i0 = jnp.deg2rad(i0_deg)
        Omega0 = jnp.deg2rad(Omega0_deg)
        dOmega_dr = jnp.deg2rad(dOmega_dr_deg)

        if self.fit_di_dr:
            di_dr_deg = rsample("di_dr", self.priors["di_dr"], shared_params)
            di_dr = jnp.deg2rad(di_dr_deg)
        else:
            di_dr = jnp.array(0.0)

        # sigma floors sampled in uas, converted to mas^2
        sigma_x_floor2 = (rsample(
            "sigma_x_floor", self.priors["sigma_x_floor"],
            shared_params) * 1e-3)**2
        sigma_y_floor2 = (rsample(
            "sigma_y_floor", self.priors["sigma_y_floor"],
            shared_params) * 1e-3)**2
        var_v_sys = rsample(
            "sigma_v_sys", self.priors["sigma_v_sys"], shared_params)**2
        var_v_hv = rsample(
            "sigma_v_hv", self.priors["sigma_v_hv"], shared_params)**2
        sigma_a_floor2 = rsample(
            "sigma_a_floor", self.priors["sigma_a_floor"], shared_params)**2

        dv_sys = rsample("dv_sys", self.priors["dv_sys"], shared_params)
        v_sys = self.v_sys_obs + dv_sys

        args = (x0, y0, D_A, M_BH, v_sys,
                self._r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
                sigma_x_floor2, sigma_y_floor2, var_v_sys, var_v_hv,
                sigma_a_floor2)

        if self.marginalise_r:
            r_all = jnp.broadcast_to(
                self._r_ang_grid[None, :],
                (self.n_spots, len(self._r_ang_grid)))

            ll_per_r = self._eval_marginal_phi(r_all, *args)

            ll_per_spot = ln_trapz_precomputed(
                ll_per_r, self._log_w_R, axis=-1)
            ll_disk = jnp.sum(ll_per_spot)

        else:
            # Sample angular radii with fixed bounds (set at init from
            # cosmographic D_A estimate). Jacobian correction recovers
            # the uniform prior on R_phys.
            with plate("spots", self.n_spots):
                r_spots = sample(
                    "r_ang", Uniform(self._r_ang_lo, self._r_ang_hi))
            # factor("prior_R_correction",
            #        self.n_spots * jnp.log(D_A * PC_PER_MAS_MPC))

            ll_per_spot = self._eval_marginal_phi(r_spots, *args)
            ll_disk = jnp.sum(ll_per_spot)

        factor("ll_disk", ll_disk)

        return D_c

    def __call__(self):
        if self.use_selection:
            raise RuntimeError(
                "Selection function must be applied in JointMaserModel, "
                "not MaserDiskModel.")

        H0 = rsample("H0", self.priors["H0"])
        sigma_pec = rsample("sigma_pec", self.priors["sigma_pec"])
        shared = {"H0": H0, "sigma_pec": sigma_pec}
        self._sample_galaxy(shared, H0 / 100.0)


class JointMaserModel(ModelBase):
    """Joint multi-galaxy megamaser H0 model.

    Shares H0, sigma_pec, and selection parameters across all galaxies;
    all other parameters are per-galaxy and scoped via handlers.scope.
    """

    def __init__(self, config_path, data_list):
        super().__init__(config_path)
        fsection("Joint Maser Disk Model")
        self._load_and_set_priors()

        self.models = [MaserDiskModel(config_path, data) for data in data_list]
        self.galaxy_names = [d["galaxy_name"] for d in data_list]
        self.use_selection = any(m.use_selection for m in self.models)
        fprint(f"loaded {len(self.models)} galaxies: "
               f"{', '.join(self.galaxy_names)}")

    def __call__(self):
        H0 = rsample("H0", self.priors["H0"])
        sigma_pec = rsample("sigma_pec", self.priors["sigma_pec"])
        h = H0 / 100.0

        shared = {"H0": H0, "sigma_pec": sigma_pec}

        if self.use_selection:
            D_lim = rsample("D_lim", self.priors["D_lim"])
            D_width = rsample("D_width", self.priors["D_width"])
            shared["D_lim"] = D_lim
            shared["D_width"] = D_width

            # Selection normalisation (same for all galaxies)
            m0 = self.models[0]
            log_sel_grid = jax_norm.logcdf(
                (D_lim - m0._sel_D_grid) / D_width)
            log_integrand = log_sel_grid + m0._sel_lp_vol
            log_Z_sel = ln_trapz_precomputed(
                log_integrand, m0._sel_log_w, axis=-1)

        for model, gname in zip(self.models, self.galaxy_names):
            with handlers.scope(prefix=gname):
                D_c = model._sample_galaxy(shared, h)

            if self.use_selection:
                log_sel_this = jax_norm.logcdf((D_lim - D_c) / D_width)
                factor(f"ll_sel_{gname}", log_sel_this - log_Z_sel)
