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
import jax.numpy as jnp
import numpy as _np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as jax_norm
from numpyro import deterministic, factor, handlers, plate, sample
from numpyro.distributions import Uniform

from ..util import SPEED_OF_LIGHT, fprint, fsection, get_nested
from .base_model import ModelBase
from .integration import (ln_trapz_precomputed, trapz_log_weights,
                          simpson_log_weights)
from .pv_utils import rsample
from .utils import normal_logpdf_var

# -----------------------------------------------------------------------
# Disk physics constants
# -----------------------------------------------------------------------

# Internal units: M_BH in 1e7 M_sun, sky positions in μas, r_ang in mas, D in Mpc.
M_BH_UNIT = 1e7
C_v = 2978.8656    # km/s: sqrt(G * 1e7 M_sun / (1 mas * 1 Mpc))
C_a = 1.872e3      # km/s/yr: 1e7 M_sun * G * yr / (1 mas * 1 Mpc)^2
C_g = 1.974e-4     # dimensionless: 2*G * 1e7 M_sun / (c^2 * 1 mas * 1 Mpc)
LOG_2PI = 1.8378770664093453  # jnp.log(2 * pi), precomputed


# -----------------------------------------------------------------------
# Spot classification and disk PA estimation
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Grid construction (pure numpy, called once at init)
# -----------------------------------------------------------------------


def _build_phi_half_grid_hv(G=201, n_patch=8):
    """Arccos-spaced half-grid on [0, pi/2] for HV spots.

    Uniform in cos(phi) gives density proportional to sin(phi) — dense
    near phi=pi/2 where high-velocity masers sit (maximum LOS velocity),
    sparse near phi=0.  The sparse low-phi tail is patched with a short
    linear segment to avoid a coarse gap there.

    G must be odd (for Simpson's rule).  n_patch must be even so the
    patch–arccos junction sits on a Simpson panel boundary.
    Setting c_min=0 makes the last point arccos(0) = pi/2 exactly,
    avoiding the tiny spacing that an appended endpoint would create.
    """
    if G % 2 == 0:
        G += 1
    c = _np.linspace(0.9999, 0.0, G)      # cos(phi): ~1 -> 0
    phi = _np.arccos(c)                     # phi: ~0 -> pi/2
    # Near phi=0 arccos is sparse; replace first n_patch with linear spacing
    phi_cut = phi[n_patch]
    phi[:n_patch] = _np.linspace(phi[0], phi_cut, n_patch + 2)[1:-1]
    return phi


def _build_phi_grid_sys(n_inner=201, inner_deg=30.0, n_wing=100,
                        n_back_inner=101, n_back_wing=50):
    """Two-cluster grid on [-pi, pi] for systemic spots.

    Front cluster (phi~0): arcsin-spaced, dense at phi=0.
    Back cluster (phi~pi): arcsin-spaced, dense at phi=pi. Coarser than
    front — only needed for spots with poorly constrained acceleration.
    Linear wings connect the two clusters.
    """
    inner_rad = _np.deg2rad(inner_deg)

    def _cluster(center, n_in, n_w):
        s = _np.linspace(-0.999, 0.999, n_in)
        phi_in = center + _np.arcsin(s) * (inner_rad / (_np.pi / 2))
        lo = _np.linspace(center - _np.pi / 2, center - inner_rad,
                          n_w + 1)[:-1]
        hi = _np.linspace(center + inner_rad, center + _np.pi / 2,
                          n_w + 1)[1:]
        return _np.concatenate([lo, phi_in, hi])

    front = _cluster(0.0, n_inner, n_wing)
    back = _cluster(_np.pi, n_back_inner, n_back_wing)
    # Wrap back from [pi/2, 3pi/2] to [-pi, pi]
    back = _np.where(back > _np.pi, back - 2 * _np.pi, back)

    return _np.unique(_np.concatenate([front, back]))


def _build_r_grid(r_min, r_max, n_r=101, scale=0.3):
    """Sinh-spaced radius grid in [r_min, r_max].

    Parameterisation: log(r) = log(r_center) + sinh(t)*scale, t uniform,
    where r_center = sqrt(r_min*r_max) is the geometric centre of the range.
    This makes the grid densest at the centre of [r_min, r_max] in log-space,
    independent of the galaxy's absolute angular radius scale.
    Benchmarks show |ΔlogL| < 0.001 nats vs a 1001-point reference at
    n_r=101; see convergence_grids.py.
    """
    logr_lo = _np.log(r_min)
    logr_hi = _np.log(r_max)
    logr_c = 0.5 * (logr_lo + logr_hi)  # geometric centre
    t_lo = _np.arcsinh((logr_lo - logr_c) / scale)
    t_hi = _np.arcsinh((logr_hi - logr_c) / scale)
    t = _np.linspace(t_lo, t_hi, n_r)
    return _np.exp(logr_c + _np.sinh(t) * scale)


# -----------------------------------------------------------------------
# Disk physics functions
# -----------------------------------------------------------------------

# Conversion: 1 mas at 1 Mpc = 4.848e-3 pc
PC_PER_MAS_MPC = 4.848e-3


def warp_geometry(r_ang, r_ang_ref, i0_rad, di_dr_rad,
                  Omega0_rad, dOmega_dr_rad,
                  d2i_dr2_rad=0.0, d2Omega_dr2_rad=0.0):
    """Evaluate warped inclination and position angle at angular radius.

    The expansion is about r_ang_ref (in mas), so i0 and Omega0 are
    the values at that angular radius. The warp rates di/dr and
    dOmega/dr are in radians per mas. Optional quadratic terms
    d2i/dr2 and d2Omega/dr2 are in radians per mas^2.

    Parameters
    ----------
    r_ang : angular radius in mas
    r_ang_ref : reference angular radius in mas (expansion centre)
    i0_rad, di_dr_rad : inclination at r_ang_ref and warp rate (rad/mas)
    Omega0_rad, dOmega_dr_rad : position angle at r_ang_ref and
        warp rate (rad/mas)
    d2i_dr2_rad, d2Omega_dr2_rad : quadratic warp rates (rad/mas^2),
        default 0 (linear warp only)

    Returns
    -------
    i, Omega : inclination and position angle at r_ang, in radians
    """
    dr = r_ang - r_ang_ref
    dr2 = dr * dr
    i = i0_rad + di_dr_rad * dr + 0.5 * d2i_dr2_rad * dr2
    Omega = Omega0_rad + dOmega_dr_rad * dr + 0.5 * d2Omega_dr2_rad * dr2
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

    R = r_ang * 1e3  # mas → μas for position projection
    X = x0 + R * (sin_phi * sin_O - cos_phi * cos_O * cos_i)
    Y = y0 + R * (sin_phi * cos_O + cos_phi * sin_O * cos_i)
    return X, Y


def predict_velocity_los(r_ang, phi, D, M_BH, v_sys, i, ecc=0.0, omega=0.0):
    """Predict line-of-sight velocity of maser spots (optical convention).

    Note: used by the mock generator. The inference hot path uses
    the fused ``_compute_observables`` instead.

    Includes Keplerian orbital velocity, relativistic Doppler, gravitational
    redshift, and systemic redshift. Supports eccentric orbits.

    Parameters
    ----------
    r_ang : orbital radius in mas
    phi : azimuthal angle in radians
    D : angular-diameter distance in Mpc
    M_BH : black hole mass in 1e7 M_sun
    v_sys : systemic velocity in km/s
    i : inclination in radians
    ecc : orbital eccentricity
    omega : argument of periapsis (phi of periapsis) in radians

    Returns
    -------
    V_obs : observed velocity in km/s (optical convention)
    """
    v_kep = C_v * jnp.sqrt(M_BH / (r_ang * D))

    cos_f = jnp.cos(phi - omega)
    ecc_fac = ((jnp.sin(phi) + ecc * jnp.sin(omega))
               / jnp.sqrt(1.0 + ecc * cos_f))
    v_z = v_kep * ecc_fac * jnp.sin(i)

    beta_c2 = (v_kep / SPEED_OF_LIGHT)**2
    beta_e2 = (beta_c2 * (1.0 + ecc**2 + 2.0 * ecc * cos_f)
               / (1.0 + ecc * cos_f))
    gamma = 1.0 / jnp.sqrt(1.0 - beta_e2)

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
    M_BH : black hole mass in 1e7 M_sun
    i : inclination in radians

    Returns
    -------
    A_z : line-of-sight acceleration in km/s/yr
    """
    a_mag = C_a * M_BH / (r_ang**2 * D**2)
    A_z = a_mag * jnp.cos(phi) * jnp.sin(i)
    return A_z


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
    R = r_ang * 1e3
    X = x0 + R * (sin_phi * pA - cos_phi * pB)
    Y = y0 + R * (sin_phi * pC + cos_phi * pD)

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
    R = r_ang * 1e3
    X = x0 + R * (sin_phi * pA - cos_phi * pB)
    Y = y0 + R * (sin_phi * pC + cos_phi * pD)

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



# -----------------------------------------------------------------------
# Per-spot adaptive r-integration
# -----------------------------------------------------------------------


def _adaptive_r_integrate(
        sin_phi, cos_phi,
        x_d, y_d, v_d, a_d,
        var_x, var_y, var_v, var_a, accel_w,
        has_accel,
        x0, y0, D_A, M_BH, v_sys,
        r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
        d2i_dr2, d2Omega_dr2,
        r_min, r_max, n_local, K_sigma,
        ecc=None, periapsis0=None, dperiapsis_dr=0.0):
    """Per-spot adaptive r-integration for one phi solution.

    For each (spot, phi), finds the integrand peak in r via Fisher
    information, builds a local trapezoidal grid, and integrates.
    Supports circular and eccentric orbits.

    Returns log_I of shape (N, n_phi).
    """
    eps = 1e-30
    inv_vx = 1.0 / var_x
    inv_vy = 1.0 / var_y
    inv_vv = 1.0 / var_v

    dx = (x_d - x0)[:, None]
    dy = (y_d - y0)[:, None]
    dv = (v_d - v_sys)[:, None]
    dv2 = dv * dv + eps

    sp = sin_phi[None, :]
    cp = cos_phi[None, :]

    # ---- Centering: iterative Fisher-optimal r_peak ----
    def _center(si, ci, sO, cO):
        # fX, fY are projection factors: X_μas = x0 + r_mas * 1e3 * fX
        # Scale by 1e3 so centering gives r_pos in mas.
        fX = 1e3 * (sp * sO - cp * cO * ci)
        fY = 1e3 * (sp * cO + cp * sO * ci)
        b_pos = fX * fX * inv_vx[:, None] + fY * fY * inv_vy[:, None]
        a_pos = dx * fX * inv_vx[:, None] + dy * fY * inv_vy[:, None]
        r_pos = a_pos / (b_pos + eps)
        b_pos = jnp.where(
            (r_pos < r_min) | (r_pos > r_max), 0.0, b_pos)
        r_pos = jnp.clip(r_pos, r_min, r_max)

        r_vel = jnp.clip(
            M_BH * (C_v * jnp.abs(sp) * si) ** 2 / (D_A * dv2),
            r_min, r_max)
        b_vel = dv2 / (4 * r_vel * r_vel + eps) * inv_vv[:, None]
        b_vel = jnp.where(r_vel >= r_max * 0.99, 0.0, b_vel)

        b_tot = b_pos + b_vel
        r_c = jnp.where(
            b_tot > eps,
            (b_pos * r_pos + b_vel * r_vel) / (b_tot + eps),
            r_vel)
        return jnp.clip(r_c, r_min, r_max), b_pos, b_vel

    i_ref, Om_ref = warp_geometry(
        r_ang_ref, r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
        d2i_dr2, d2Omega_dr2)
    r_c, b_pos, b_vel = _center(
        jnp.sin(i_ref), jnp.cos(i_ref),
        jnp.sin(Om_ref), jnp.cos(Om_ref))

    for _ in range(3):
        i_r, Om_r = warp_geometry(
            r_c, r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
            d2i_dr2, d2Omega_dr2)
        r_c, b_pos, b_vel = _center(
            jnp.sin(i_r), jnp.cos(i_r),
            jnp.sin(Om_r), jnp.cos(Om_r))

    # Use the WIDER of position/velocity sigmas for grid width.
    # Combined sigma is too tight — non-Gaussian tails of the weaker
    # constraint extend beyond the combined Gaussian envelope.
    b_min = (4.0 / (r_max - r_min)) ** 2
    sigma_pos = 1.0 / jnp.sqrt(jnp.maximum(b_pos, b_min))
    sigma_vel = 1.0 / jnp.sqrt(jnp.maximum(b_vel, b_min))
    sigma_r = jnp.maximum(sigma_pos, sigma_vel)
    sigma_r = jnp.minimum(sigma_r, (r_max - r_min) / 4)

    # ---- Local grid: uniform (N, n_phi, n_local) ----
    t = jnp.linspace(-K_sigma, K_sigma, n_local)
    r_loc_raw = r_c[:, :, None] + sigma_r[:, :, None] * t[None, None, :]
    # Mask out-of-range points instead of clipping to avoid boundary
    # pile-up that biases the trapezoidal weights.
    valid = (r_loc_raw >= r_min) & (r_loc_raw <= r_max)
    r_loc = jnp.clip(r_loc_raw, r_min, r_max)

    h = sigma_r * (2.0 * K_sigma / (n_local - 1))
    w_end = jnp.ones(n_local).at[0].set(0.5).at[-1].set(0.5)
    log_w = jnp.log(h)[:, :, None] + jnp.log(w_end)[None, None, :]

    # ---- Observables at r_loc ----
    i_r, Om_r = warp_geometry(
        r_loc, r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
        d2i_dr2, d2Omega_dr2)
    sin_i = jnp.sin(i_r)
    cos_i = jnp.cos(i_r)
    sin_O = jnp.sin(Om_r)
    cos_O = jnp.cos(Om_r)

    sp3 = sin_phi[None, :, None]
    cp3 = cos_phi[None, :, None]

    R_loc = r_loc * 1e3
    X = x0 + R_loc * (sp3 * sin_O - cp3 * cos_O * cos_i)
    Y = y0 + R_loc * (sp3 * cos_O + cp3 * sin_O * cos_i)

    rD = r_loc * D_A
    v_kep = C_v * jnp.sqrt(M_BH / rD)

    if ecc is not None:
        omega_r = periapsis0 + dperiapsis_dr * (r_loc - r_ang_ref)
        sin_om = jnp.sin(omega_r)
        cos_om = jnp.cos(omega_r)
        cos_f = cp3 * cos_om + sp3 * sin_om
        ecc_fac = (sp3 + ecc * sin_om) / jnp.sqrt(1.0 + ecc * cos_f)
        v_z = v_kep * ecc_fac * sin_i
        beta_c2 = (v_kep / SPEED_OF_LIGHT) ** 2
        beta_e2 = (beta_c2 * (1.0 + ecc ** 2 + 2.0 * ecc * cos_f)
                   / (1.0 + ecc * cos_f))
        gamma = 1.0 / jnp.sqrt(1.0 - beta_e2)
    else:
        v_z = v_kep * sp3 * sin_i
        beta = v_kep / SPEED_OF_LIGHT
        gamma = 1.0 / jnp.sqrt(1.0 - beta * beta)

    zpg = 1.0 / jnp.sqrt(1.0 - C_g * M_BH / rD)
    V = SPEED_OF_LIGHT * (
        gamma * (1.0 + v_z / SPEED_OF_LIGHT) * zpg
        * (1.0 + v_sys / SPEED_OF_LIGHT) - 1.0)

    # ---- chi² ----
    chi2 = ((x_d[:, None, None] - X) ** 2 * inv_vx[:, None, None]
            + (y_d[:, None, None] - Y) ** 2 * inv_vy[:, None, None]
            + (v_d[:, None, None] - V) ** 2 * inv_vv[:, None, None])

    if has_accel:
        A = C_a * M_BH / (r_loc ** 2 * D_A ** 2) * cp3 * sin_i
        inv_va = 1.0 / var_a
        chi2 = chi2 + ((a_d[:, None, None] - A) ** 2
                       * inv_va[:, None, None] * accel_w[:, None, None])

    # ---- lnorm + integrate ----
    lnorm = -0.5 * (3 * LOG_2PI + jnp.log(var_x) + jnp.log(var_y)
                     + jnp.log(var_v))
    if has_accel:
        lnorm = lnorm - 0.5 * (LOG_2PI + jnp.log(var_a)) * accel_w

    log_f = lnorm[:, None, None] - 0.5 * chi2
    # Mask out-of-range grid points so they don't contribute
    log_f = jnp.where(valid, log_f, -jnp.inf)
    return logsumexp(log_f + log_w, axis=-1)


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

        self._resolve_per_galaxy_priors(data)

        self.n_spots = data["n_spots"]
        self.is_highvel = jnp.asarray(data["is_highvel"])

        # Default per-spot velocity measurement error if not provided
        if "sigma_v" not in data:
            sv_default = float(get_nested(
                self.config, "model/sigma_v_default", 0.25))
            data["sigma_v"] = sv_default * _np.ones(data["n_spots"])
            fprint(f"sigma_v not in data, defaulting to {sv_default} km/s.")

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

        # Split each type into with-accel and without-accel sub-groups.
        # Spots without acceleration get a 3-obs chi² (no A computed).
        accel_meas = _np.asarray(data.get(
            "accel_measured", self.sigma_a < 1e4))
        use_clump_weight = get_nested(
            self.config, "model/use_clump_weight", True)
        if use_clump_weight:
            accel_w = jnp.asarray(data.get(
                "accel_weight", _np.ones(self.n_spots)))
        else:
            accel_w = jnp.ones(self.n_spots)
            fprint("clump weighting disabled (use_clump_weight = false)")
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
            setattr(self, f"_accel_w_{label}_a",
                    accel_w[idx_np[has_a]])

        _labels = ("sys", "red", "blue")
        n_a = sum(getattr(self, f"_n_{lb}_a") for lb in _labels)
        n_noa = sum(getattr(self, f"_n_{lb}_noa") for lb in _labels)
        fprint(f"accel split: {n_a} with, {n_noa} without.")

        # All-spot arrays in original data order (for adaptive r grids)
        self._all_x = jnp.asarray(data["x"])
        self._all_y = jnp.asarray(data["y"])
        self._all_sigma_x2 = jnp.asarray(data["sigma_x"])**2
        self._all_sigma_y2 = jnp.asarray(data["sigma_y"])**2
        self._all_v = jnp.asarray(self.velocity)
        self._all_a = jnp.asarray(self.a)
        self._all_sigma_a = jnp.asarray(self.sigma_a)
        self._all_has_accel = jnp.asarray(accel_meas)
        self._all_sigma_a2 = self._all_sigma_a**2

        # Per-spot velocity variance and acceleration weight (for adaptive phi)
        all_sigma_v2 = _np.zeros(self.n_spots)
        all_accel_w = _np.zeros(self.n_spots)
        for prefix in ("sys", "red", "blue"):
            for suffix in ("_a", "_noa"):
                attr = f"_idx_{prefix}{suffix}"
                if not hasattr(self, attr):
                    continue
                idx = getattr(self, attr)
                if len(idx) == 0:
                    continue
                all_sigma_v2[_np.asarray(idx)] = _np.asarray(
                    getattr(self, f"_sigma_v2_{prefix}{suffix}"))
                if suffix == "_a":
                    all_accel_w[_np.asarray(idx)] = _np.asarray(
                        getattr(self, f"_accel_w_{prefix}_a"))
        self._all_sigma_v2 = jnp.asarray(all_sigma_v2)
        self._all_accel_w = jnp.asarray(all_accel_w)

        # ---- Phi grids and precomputed trig ----
        G_half = int(get_nested(self.config, "model/G_phi_half", 251))
        n_inner_sys = int(get_nested(self.config, "model/n_inner_sys", 201))
        inner_deg_sys = float(get_nested(
            self.config, "model/inner_deg_sys", 30.0))
        n_wing_sys = int(get_nested(self.config, "model/n_wing_sys", 100))
        phi_half = _build_phi_half_grid_hv(G=G_half)
        phi_sys = _build_phi_grid_sys(n_inner=n_inner_sys,
                                      inner_deg=inner_deg_sys,
                                      n_wing=n_wing_sys)

        # Store phi grids for phi prior computation
        self._phi_half = jnp.asarray(phi_half)
        self._phi_sys = jnp.asarray(phi_sys)

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

        # Simpson's rule for HV (O(h^4)), trapezoidal for systemic
        self._log_w_phi_hv = jnp.asarray(simpson_log_weights(phi_half))
        self._log_w_phi_sys = jnp.asarray(trapz_log_weights(phi_sys))

        use_ecc = get_nested(self.config, "model/use_ecc", False)
        use_qw = get_nested(self.config, "model/use_quadratic_warp", False)

        # Per-galaxy overrides
        gname = data.get("galaxy_name", "")
        gal_cfg = get_nested(self.config, f"model/galaxies/{gname}", {})
        self.use_ecc = gal_cfg.get("use_ecc", use_ecc)
        self.use_quadratic_warp = gal_cfg.get(
            "use_quadratic_warp", use_qw)
        fprint(f"use_ecc = {self.use_ecc}")
        fprint(f"use_quadratic_warp = {self.use_quadratic_warp}")

        # Phi method: "default" (arccos/two-cluster grids),
        # "adaptive" (per-spot sinh), "bruteforce" (uniform grid).
        # Legacy: adaptive_phi=True maps to "adaptive".
        legacy_adaptive = gal_cfg.get("adaptive_phi", False)
        self.phi_method = gal_cfg.get(
            "phi_method", "adaptive" if legacy_adaptive else "default")
        if self.phi_method == "adaptive":
            self._n_phi_adaptive = int(get_nested(
                self.config, "model/n_phi_adaptive", 1001))
            self._K_sigma_phi = float(get_nested(
                self.config, "model/K_sigma_phi", 20.0))
            fprint(f"phi_method = adaptive, n_phi={self._n_phi_adaptive}, "
                   f"K_sigma={self._K_sigma_phi}")
        elif self.phi_method == "bruteforce":
            self._n_phi_bruteforce = int(gal_cfg.get(
                "n_phi_bruteforce", 30001))
            fprint(f"phi_method = bruteforce, "
                   f"n_phi={self._n_phi_bruteforce}")
        else:
            fprint(f"phi_method = default")
        # Backward compat
        self.adaptive_phi = self.phi_method == "adaptive"

        # Inverse permutation for concat+gather (replaces .at[idx].set).
        # Order: sys_a, sys_noa, red_a, red_noa, blue_a, blue_noa
        order = jnp.concatenate([
            self._idx_sys_a, self._idx_sys_noa,
            self._idx_red_a, self._idx_red_noa,
            self._idx_blue_a, self._idx_blue_noa])
        self._inv_order = jnp.argsort(order)

        # ---- Radius grid setup (for Mode 2) ----
        marginalise_r_global = get_nested(
            self.config, "model/marginalise_r", False)
        self.marginalise_r = gal_cfg.get(
            "marginalise_r", marginalise_r_global)
        self._adaptive_r = get_nested(
            self.config, "model/adaptive_r", True)
        if self.marginalise_r:
            self._R_phys_lo = float(get_nested(
                self.config, "model/R_phys_lo", 0.01))
            self._R_phys_hi = float(get_nested(
                self.config, "model/R_phys_hi", 2.0))
            self._n_r = int(get_nested(self.config, "model/n_r", 251))
            # Spot-aware sinh grid: center on median log(r_proj),
            # scale from data spread with conservative floor of 0.3.
            r_proj = _np.sqrt(
                _np.asarray(data["x"])**2 + _np.asarray(data["y"])**2)
            r_proj = r_proj / 1e3  # μas → mas
            r_proj = r_proj[r_proj > 1e-3]
            self._r_logr_c = float(_np.median(_np.log(r_proj)))
            self._r_scale = float(
                max(0.5 * _np.std(_np.log(r_proj)), 0.3))
            fprint(f"r grid: spot-aware sinh, logr_c={self._r_logr_c:.3f} "
                   f"(r={_np.exp(self._r_logr_c):.3f} mas), "
                   f"scale={self._r_scale:.3f}")
            if self._adaptive_r:
                self._n_r_local = int(get_nested(
                    self.config, "model/n_r_local", 201))
                self._K_sigma = float(get_nested(
                    self.config, "model/K_sigma", 5.0))
                fprint(f"adaptive r: {self._n_r_local} local points, "
                       f"K={self._K_sigma}")

        # ---- Selection function grid ----
        D_min = float(get_nested(self.config, "model/priors/D/low", 10.0))
        D_max = float(get_nested(self.config, "model/priors/D/high", 200.0))
        self._sel_D_grid = jnp.linspace(D_min, D_max, 501)
        self._sel_log_w = jnp.asarray(trapz_log_weights(self._sel_D_grid))
        self._sel_lp_vol = 2.0 * jnp.log(self._sel_D_grid)
        self.use_selection = get_nested(
            self.config, "model/use_selection", False)

        # di_dr is always sampled (inclination warp).

        # Reference angular radius for warp expansion (mas).
        # Per-galaxy override or median projected HV-spot radius.
        r_ang_ref_cfg = gal_cfg.get("r_ang_ref", None)
        if r_ang_ref_cfg is not None:
            self._r_ang_ref = float(r_ang_ref_cfg)
            fprint(f"warp pivot r_ang_ref = {self._r_ang_ref:.3f} mas "
                   f"(from config)")
        else:
            x_hv = _np.asarray(data["x"])[is_hv_np]  # μas
            y_hv = _np.asarray(data["y"])[is_hv_np]
            r_ang_hv = _np.sqrt(x_hv**2 + y_hv**2) / 1e3  # μas → mas
            self._r_ang_ref = float(_np.median(r_ang_hv))
            fprint(f"warp pivot r_ang_ref = {self._r_ang_ref:.3f} mas "
                   f"(median projected radius of {len(r_ang_hv)} HV spots)")

        # Fixed r_ang bounds for Mode 1 (estimated from v_sys_obs).
        # Uses cosmographic D_A to convert R_phys bounds to angular.
        z_est = self.v_sys_obs / SPEED_OF_LIGHT
        q0 = -0.55
        D_c_est = (SPEED_OF_LIGHT * z_est / 73.0
                   * (1 + 0.5 * (1 - q0) * z_est))
        D_A_est = D_c_est / (1 + z_est)
        R_lo = float(get_nested(
            self.config, "model/R_phys_lo", 0.01))
        R_hi = float(get_nested(
            self.config, "model/R_phys_hi", 2.0))
        self._r_ang_lo = R_lo / (D_A_est * PC_PER_MAS_MPC)
        self._r_ang_hi = R_hi / (D_A_est * PC_PER_MAS_MPC)
        fprint(f"r_ang bounds: [{self._r_ang_lo:.3f}, "
               f"{self._r_ang_hi:.3f}] mas "
               f"(D_A_est = {D_A_est:.1f} Mpc)")

        # At each likelihood evaluation, the r_ang grid is built
        # directly in r_ang space from sinh(t) (precomputed) shifted
        # to the current D_A. Trapezoid weights in r_ang space.
        if self.marginalise_r:
            r_lo_est = self._R_phys_lo / (D_A_est * PC_PER_MAS_MPC)
            r_hi_est = self._R_phys_hi / (D_A_est * PC_PER_MAS_MPC)
            fprint(f"r_ang grid (at D_A_est): "
                   f"[{r_lo_est:.4f}, {r_hi_est:.4f}] mas")

        if self._adaptive_r and self.marginalise_r:
            mode = "r+phi (adaptive)"
        elif self.marginalise_r:
            mode = "r+phi"
        else:
            mode = "phi only"
        fprint(f"loaded {self.n_spots} maser spots "
               f"({self._n_sys} systemic, {self._n_red} red, "
               f"{self._n_blue} blue).")
        fprint(f"marginalisation mode: {mode}")
        fprint(f"phi grids: HV half={len(phi_half)}, "
               f"sys={len(phi_sys)}")
        if self.marginalise_r:
            fprint(f"r grid: {self._n_r} points, "
                   f"R_phys in [{self._R_phys_lo:.3f}, "
                   f"{self._R_phys_hi:.3f}] pc")
        fprint(f"use_selection = {self.use_selection}")

    def build_r_ang_grid(self, D_A):
        """Build spot-aware sinh-spaced r_ang grid at a given D_A."""
        conv = D_A * PC_PER_MAS_MPC
        logr_lo = jnp.log(self._R_phys_lo / conv)
        logr_hi = jnp.log(self._R_phys_hi / conv)
        t_lo = jnp.arcsinh((logr_lo - self._r_logr_c) / self._r_scale)
        t_hi = jnp.arcsinh((logr_hi - self._r_logr_c) / self._r_scale)
        t = jnp.linspace(t_lo, t_hi, self._n_r)
        return jnp.exp(self._r_logr_c + jnp.sinh(t) * self._r_scale)

    def _resolve_per_galaxy_priors(self, data):
        """Set per-galaxy D prior from data dict."""
        self._D_c_volume = False
        if "D_lo" in data and "D_hi" in data:
            lo, hi = float(data["D_lo"]), float(data["D_hi"])
            self.priors["D"] = Uniform(lo, hi)
            D_prior_type = get_nested(
                self.config, "model/D_c_prior", "uniform")
            self._D_c_volume = D_prior_type == "volume"
            fprint(f"D prior: {'volume' if self._D_c_volume else 'uniform'}"
                   f"({lo:.1f}, {hi:.1f})")

    def _eval_marginal_phi(self, r_ang, x0, y0, D_A, M_BH, v_sys,
                           r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
                           sigma_x_floor2, sigma_y_floor2,
                           var_v_sys, var_v_hv,
                           sigma_a_floor2,
                           log_w_r=None,
                           ecc=None, periapsis0=None, dperiapsis_dr=0.0,
                           d2i_dr2=0.0, d2Omega_dr2=0.0):
        """Evaluate per-spot log-likelihood marginalised over phi [and r].

        Mode 1 (sample r_ang): r_ang is (N,), log_w_r is None.
            Returns (N,) per-spot marginalised-phi log-likelihood.
        Mode 2 (marginalise r): r_ang is (N, n_r), log_w_r is (n_r,).
            Fuses the phi and r logsumexp into a single 2D reduction,
            returning (N,) per-spot fully-marginalised log-likelihood.

        All sigma/var arguments are pre-squared.
        """
        # rpad: adds trailing dim for phi broadcasting on r_ang
        # dpad: adds dims so 1D data arrays broadcast with r_ang + phi
        rpad = (slice(None),) * r_ang.ndim + (None,)
        n_extra = r_ang.ndim
        dpad = (slice(None),) + (None,) * n_extra

        # Precompute ALL r-dependent quantities on the (N,) or (N, N_r)
        # grid. The phi loop then only does multiply-add + logsumexp.
        i_r, Omega_r = warp_geometry(
            r_ang, r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
            d2i_dr2, d2Omega_dr2)
        sin_i = jnp.sin(i_r)
        cos_i = jnp.cos(i_r)
        sin_O = jnp.sin(Omega_r)
        cos_O = jnp.cos(Omega_r)
        v_kep, gamma, z_g_factor, a_mag, pA, pB, pC, pD = \
            _precompute_r_quantities(r_ang, D_A, M_BH,
                                     sin_i, cos_i, sin_O, cos_O)

        # Periapsis angle warp: omega(r) = periapsis0 + dperiapsis_dr*(r-r_ref)
        if ecc is not None:
            omega_r = periapsis0 + dperiapsis_dr * (r_ang - r_ang_ref)
            sin_omega = jnp.sin(omega_r)
            cos_omega = jnp.cos(omega_r)

        def _r_precomp(idx):
            """Slice precomputed r-quantities for a spot subset.

            Return order matches _observables_from_precomputed args
            after (sin_phi, cos_phi, x0, y0, v_sys).
            """
            return (sin_i[idx][rpad], r_ang[idx][rpad],
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
            (si, r_sub, vk, gm, zg, _, pa, pb, pc, pd) = _r_precomp(idx)
            return _observables_no_accel(
                sp, cp, x0, y0, v_sys, si, r_sub, vk, gm, zg,
                pa, pb, pc, pd)

        def _obs_4_ecc(idx, sp, cp):
            """X, Y, V (eccentric), A from precomputed r-quantities."""
            (si, r_sub, vk, _, zg, am, pa, pb, pc, pd) = _r_precomp(idx)
            sw = sin_omega[idx][rpad]
            cw = cos_omega[idx][rpad]
            R_sub = r_sub * 1e3
            X = x0 + R_sub * (sp * pa - cp * pb)
            Y = y0 + R_sub * (sp * pc + cp * pd)

            # Velocity components
            cos_f = cp * cw + sp * sw
            ecc_fac = (sp + ecc * sw) / jnp.sqrt(1.0 + ecc * cos_f)
            v_z = vk * ecc_fac * si

            # Precise Lorentz factor for eccentric orbit
            beta_c2 = (vk / SPEED_OF_LIGHT)**2
            beta_e2 = (beta_c2
                       * (1.0 + ecc**2 + 2.0 * ecc * cos_f)
                       / (1.0 + ecc * cos_f))
            gm_e = 1.0 / jnp.sqrt(1.0 - beta_e2)

            one_plus_z_D = gm_e * (1.0 + v_z / SPEED_OF_LIGHT)
            V = SPEED_OF_LIGHT * (
                one_plus_z_D * zg
                * (1.0 + v_sys / SPEED_OF_LIGHT) - 1.0)
            A = am * cp * si
            return X, Y, V, A

        def _obs_3_ecc(idx, sp, cp):
            """X, Y, V (eccentric) only — no acceleration."""
            (si, r_sub, vk, _, zg, _, pa, pb, pc, pd) = _r_precomp(idx)
            sw = sin_omega[idx][rpad]
            cw = cos_omega[idx][rpad]
            R_sub = r_sub * 1e3
            X = x0 + R_sub * (sp * pa - cp * pb)
            Y = y0 + R_sub * (sp * pc + cp * pd)

            cos_f = cp * cw + sp * sw
            ecc_fac = (sp + ecc * sw) / jnp.sqrt(1.0 + ecc * cos_f)
            v_z = vk * ecc_fac * si

            beta_c2 = (vk / SPEED_OF_LIGHT)**2
            beta_e2 = (beta_c2
                       * (1.0 + ecc**2 + 2.0 * ecc * cos_f)
                       / (1.0 + ecc * cos_f))
            gm_e = 1.0 / jnp.sqrt(1.0 - beta_e2)

            one_plus_z_D = gm_e * (1.0 + v_z / SPEED_OF_LIGHT)
            V = SPEED_OF_LIGHT * (
                one_plus_z_D * zg
                * (1.0 + v_sys / SPEED_OF_LIGHT) - 1.0)
            return X, Y, V

        def _lnorm_3(sx2, sy2, sv2, sv_floor2):
            return -0.5 * (3 * LOG_2PI + jnp.log(sx2 + sigma_x_floor2)
                           + jnp.log(sy2 + sigma_y_floor2)
                           + jnp.log(sv2 + sv_floor2))

        def _lnorm_4(sx2, sy2, sv2, sv_floor2, sa2, sa_floor2):
            return (_lnorm_3(sx2, sy2, sv2, sv_floor2)
                    - 0.5 * (LOG_2PI + jnp.log(sa2 + sa_floor2)))

        # Combined phi[+r] weights for fused 2D logsumexp in Mode 2.
        if log_w_r is not None:
            if log_w_r.ndim == 2:
                # Per-spot weights: (N, n_r) -> (N, n_r, n_phi)
                log_w_2d_sys = (log_w_r[:, :, None]
                                + self._log_w_phi_sys[None, None, :])
                log_w_2d_hv = (log_w_r[:, :, None]
                               + self._log_w_phi_hv[None, None, :])
            else:
                log_w_2d_sys = log_w_r[:, None] + self._log_w_phi_sys[None, :]
                log_w_2d_hv = log_w_r[:, None] + self._log_w_phi_hv[None, :]

        # ---- Systemic spots ----
        def _sys_block(idx_attr, log_w_r, log_w_2d,
                       x_d, y_d, v_d, a_d,
                       sx2, sy2, sv2, sv_floor2, sa2, has_accel,
                       sa_floor2, aw=None):
            vx = sx2[dpad] + sigma_x_floor2
            vy = sy2[dpad] + sigma_y_floor2
            vv = sv2[dpad] + sv_floor2
            idx = getattr(self, idx_attr)
            if has_accel:
                va = sa2[dpad] + sa_floor2
                inv_va = 1.0 / va
                X, Y, V, A = _obs_4(
                    idx, self._sin_phi_sys, self._cos_phi_sys)
                chi2_xyv = _chi2_3obs(
                    x_d[dpad], X, 1.0 / vx, y_d[dpad], Y, 1.0 / vy,
                    v_d[dpad], V, 1.0 / vv)
                da = a_d[dpad] - A
                chi2_a = da * da * inv_va
                lnorm_xyv = _lnorm_3(sx2[dpad], sy2[dpad], sv2[dpad],
                                     sv_floor2)
                lnorm_a = -0.5 * (LOG_2PI + jnp.log(va))
                w = aw[dpad]
                ll = (lnorm_xyv + w * lnorm_a
                      - 0.5 * chi2_xyv - 0.5 * w * chi2_a)
            else:
                X, Y, V = _obs_3(
                    idx, self._sin_phi_sys, self._cos_phi_sys)
                chi2 = _chi2_3obs(
                    x_d[dpad], X, 1.0 / vx, y_d[dpad], Y, 1.0 / vy,
                    v_d[dpad], V, 1.0 / vv)
                lnorm = _lnorm_3(sx2[dpad], sy2[dpad], sv2[dpad],
                                 sv_floor2)
                ll = lnorm - 0.5 * chi2
            if log_w_r is not None:
                w2d = log_w_2d[idx] if log_w_2d.ndim == 3 else log_w_2d
                return logsumexp(ll + w2d, axis=(-2, -1))
            return ln_trapz_precomputed(ll, self._log_w_phi_sys, axis=-1)

        def _hv_block(idx_attr, sp1, cp1, sp2, cp2,
                      log_w_r, log_w_2d,
                      x_d, y_d, v_d, a_d,
                      sx2, sy2, sv2, sv_floor2, sa2, has_accel,
                      sa_floor2, aw=None):
            vx = sx2[dpad] + sigma_x_floor2
            vy = sy2[dpad] + sigma_y_floor2
            vv = sv2[dpad] + sv_floor2
            inv_vx, inv_vy, inv_vv = 1.0 / vx, 1.0 / vy, 1.0 / vv
            idx = getattr(self, idx_attr)
            r_sub = r_ang[idx][rpad]
            R_sub = r_sub * 1e3
            pa_s, pb_s = pA[idx][rpad], pB[idx][rpad]
            pc_s, pd_s = pC[idx][rpad], pD[idx][rpad]

            if has_accel:
                va = sa2[dpad] + sa_floor2
                inv_va = 1.0 / va
                X1, Y1, V, A1 = _obs_4(idx, sp1, cp1)
                X2 = x0 + R_sub * (sp2 * pa_s - cp2 * pb_s)
                Y2 = y0 + R_sub * (sp2 * pc_s + cp2 * pd_s)
                A2 = -A1
                chi2_v = (v_d[dpad] - V) ** 2 * inv_vv
                # x,y chi² per solution (no v, no a)
                dx1 = x_d[dpad] - X1
                dy1 = y_d[dpad] - Y1
                chi2_xy1 = dx1 * dx1 * inv_vx + dy1 * dy1 * inv_vy
                dx2 = x_d[dpad] - X2
                dy2 = y_d[dpad] - Y2
                chi2_xy2 = dx2 * dx2 * inv_vx + dy2 * dy2 * inv_vy
                da1 = a_d[dpad] - A1
                da2 = a_d[dpad] - A2
                chi2_a1 = da1 * da1 * inv_va
                chi2_a2 = da2 * da2 * inv_va
                w = aw[dpad]
                lnorm_xyv = _lnorm_3(sx2[dpad], sy2[dpad], sv2[dpad],
                                     sv_floor2)
                lnorm_a = -0.5 * (LOG_2PI + jnp.log(va))
                lnorm = lnorm_xyv + w * lnorm_a
                ll = (lnorm - 0.5 * chi2_v
                      + jnp.logaddexp(
                          -0.5 * (chi2_xy1 + w * chi2_a1),
                          -0.5 * (chi2_xy2 + w * chi2_a2)))
            else:
                X1, Y1, V = _obs_3(idx, sp1, cp1)
                X2 = x0 + R_sub * (sp2 * pa_s - cp2 * pb_s)
                Y2 = y0 + R_sub * (sp2 * pc_s + cp2 * pd_s)
                chi2_v = (v_d[dpad] - V) ** 2 * inv_vv
                chi2_1 = (_chi2_3obs(
                    x_d[dpad], X1, inv_vx, y_d[dpad], Y1, inv_vy,
                    v_d[dpad], V, inv_vv) - chi2_v)
                chi2_2 = (_chi2_3obs(
                    x_d[dpad], X2, inv_vx, y_d[dpad], Y2, inv_vy,
                    v_d[dpad], V, inv_vv) - chi2_v)
                lnorm = _lnorm_3(sx2[dpad], sy2[dpad], sv2[dpad],
                                 sv_floor2)
                ll = (lnorm - 0.5 * chi2_v
                      + jnp.logaddexp(-0.5 * chi2_1, -0.5 * chi2_2))
            if log_w_r is not None:
                w2d = log_w_2d[idx] if log_w_2d.ndim == 3 else log_w_2d
                return logsumexp(ll + w2d, axis=(-2, -1))
            return ln_trapz_precomputed(ll, self._log_w_phi_hv, axis=-1)

        def _sys_block_ecc(idx_attr, log_w_r, log_w_2d,
                           x_d, y_d, v_d, a_d,
                           sx2, sy2, sv2, sv_floor2, sa2, has_accel,
                           sa_floor2, aw=None):
            vx = sx2[dpad] + sigma_x_floor2
            vy = sy2[dpad] + sigma_y_floor2
            vv = sv2[dpad] + sv_floor2
            idx = getattr(self, idx_attr)
            if has_accel:
                va = sa2[dpad] + sa_floor2
                inv_va = 1.0 / va
                X, Y, V, A = _obs_4_ecc(
                    idx, self._sin_phi_sys, self._cos_phi_sys)
                chi2_xyv = _chi2_3obs(
                    x_d[dpad], X, 1.0 / vx, y_d[dpad], Y, 1.0 / vy,
                    v_d[dpad], V, 1.0 / vv)
                da = a_d[dpad] - A
                chi2_a = da * da * inv_va
                lnorm_xyv = _lnorm_3(sx2[dpad], sy2[dpad], sv2[dpad],
                                     sv_floor2)
                lnorm_a = -0.5 * (LOG_2PI + jnp.log(va))
                w = aw[dpad]
                ll = (lnorm_xyv + w * lnorm_a
                      - 0.5 * chi2_xyv - 0.5 * w * chi2_a)
            else:
                X, Y, V = _obs_3_ecc(
                    idx, self._sin_phi_sys, self._cos_phi_sys)
                chi2 = _chi2_3obs(
                    x_d[dpad], X, 1.0 / vx, y_d[dpad], Y, 1.0 / vy,
                    v_d[dpad], V, 1.0 / vv)
                lnorm = _lnorm_3(sx2[dpad], sy2[dpad], sv2[dpad],
                                 sv_floor2)
                ll = lnorm - 0.5 * chi2
            if log_w_r is not None:
                w2d = log_w_2d[idx] if log_w_2d.ndim == 3 else log_w_2d
                return logsumexp(ll + w2d, axis=(-2, -1))
            return ln_trapz_precomputed(ll, self._log_w_phi_sys, axis=-1)

        def _hv_block_ecc(idx_attr, sp1, cp1, sp2, cp2,
                          log_w_r, log_w_2d,
                          x_d, y_d, v_d, a_d,
                          sx2, sy2, sv2, sv_floor2, sa2, has_accel,
                          sa_floor2, aw=None):
            """HV block with eccentricity: V1 != V2, no reflection shortcut."""
            vx = sx2[dpad] + sigma_x_floor2
            vy = sy2[dpad] + sigma_y_floor2
            vv = sv2[dpad] + sv_floor2
            inv_vx, inv_vy, inv_vv = 1.0 / vx, 1.0 / vy, 1.0 / vv
            idx = getattr(self, idx_attr)
            if has_accel:
                va = sa2[dpad] + sa_floor2
                inv_va = 1.0 / va
                X1, Y1, V1, A1 = _obs_4_ecc(idx, sp1, cp1)
                X2, Y2, V2, A2 = _obs_4_ecc(idx, sp2, cp2)
                chi2_xyv1 = _chi2_3obs(
                    x_d[dpad], X1, inv_vx, y_d[dpad], Y1, inv_vy,
                    v_d[dpad], V1, inv_vv)
                chi2_xyv2 = _chi2_3obs(
                    x_d[dpad], X2, inv_vx, y_d[dpad], Y2, inv_vy,
                    v_d[dpad], V2, inv_vv)
                da1 = a_d[dpad] - A1
                da2 = a_d[dpad] - A2
                chi2_a1 = da1 * da1 * inv_va
                chi2_a2 = da2 * da2 * inv_va
                w = aw[dpad]
                lnorm_xyv = _lnorm_3(sx2[dpad], sy2[dpad], sv2[dpad],
                                     sv_floor2)
                lnorm_a = -0.5 * (LOG_2PI + jnp.log(va))
                lnorm = lnorm_xyv + w * lnorm_a
                ll = lnorm + jnp.logaddexp(
                    -0.5 * (chi2_xyv1 + w * chi2_a1),
                    -0.5 * (chi2_xyv2 + w * chi2_a2))
            else:
                X1, Y1, V1 = _obs_3_ecc(idx, sp1, cp1)
                X2, Y2, V2 = _obs_3_ecc(idx, sp2, cp2)
                chi2_1 = _chi2_3obs(
                    x_d[dpad], X1, inv_vx, y_d[dpad], Y1, inv_vy,
                    v_d[dpad], V1, inv_vv)
                chi2_2 = _chi2_3obs(
                    x_d[dpad], X2, inv_vx, y_d[dpad], Y2, inv_vy,
                    v_d[dpad], V2, inv_vv)
                lnorm = _lnorm_3(sx2[dpad], sy2[dpad], sv2[dpad],
                                 sv_floor2)
                ll = lnorm + jnp.logaddexp(-0.5 * chi2_1, -0.5 * chi2_2)
            if log_w_r is not None:
                w2d = log_w_2d[idx] if log_w_2d.ndim == 3 else log_w_2d
                return logsumexp(ll + w2d, axis=(-2, -1))
            return ln_trapz_precomputed(ll, self._log_w_phi_hv, axis=-1)

        results = []

        # ---- Systemic: with accel, then without ----
        for suffix, has_a in [("_a", True), ("_noa", False)]:
            n = getattr(self, f"_n_sys{suffix}")
            if n > 0:
                kw = dict(
                    log_w_r=log_w_r,
                    log_w_2d=log_w_2d_sys if log_w_r is not None else None,
                    x_d=getattr(self, f"_x_sys{suffix}"),
                    y_d=getattr(self, f"_y_sys{suffix}"),
                    v_d=getattr(self, f"_velocity_sys{suffix}"),
                    a_d=getattr(self, "_a_sys_a", None) if has_a else None,
                    sx2=getattr(self, f"_sigma_x2_sys{suffix}"),
                    sy2=getattr(self, f"_sigma_y2_sys{suffix}"),
                    sv2=getattr(self, f"_sigma_v2_sys{suffix}"),
                    sv_floor2=var_v_sys,
                    sa2=(getattr(self, "_sigma_a2_sys_a", None)
                         if has_a else None),
                    has_accel=has_a,
                    sa_floor2=sigma_a_floor2,
                    aw=(getattr(self, "_accel_w_sys_a", None)
                        if has_a else None))
                sys_fn = _sys_block_ecc if ecc is not None else _sys_block
                results.append(sys_fn(f"_idx_sys{suffix}", **kw))

        # ---- Red and Blue HV: with accel, then without ----
        for color in ["red", "blue"]:
            for suffix, has_a in [("_a", True), ("_noa", False)]:
                if getattr(self, f"_n_{color}{suffix}") == 0:
                    continue
                kw = dict(
                    sp1=getattr(self, f"_sin_phi1_{color}"),
                    cp1=getattr(self, f"_cos_phi1_{color}"),
                    sp2=getattr(self, f"_sin_phi2_{color}"),
                    cp2=getattr(self, f"_cos_phi2_{color}"),
                    log_w_r=log_w_r,
                    log_w_2d=(log_w_2d_hv if log_w_r is not None
                              else None),
                    x_d=getattr(self, f"_x_{color}{suffix}"),
                    y_d=getattr(self, f"_y_{color}{suffix}"),
                    v_d=getattr(self, f"_velocity_{color}{suffix}"),
                    a_d=(getattr(self, f"_a_{color}_a", None)
                         if has_a else None),
                    sx2=getattr(self, f"_sigma_x2_{color}{suffix}"),
                    sy2=getattr(self, f"_sigma_y2_{color}{suffix}"),
                    sv2=getattr(self, f"_sigma_v2_{color}{suffix}"),
                    sv_floor2=var_v_hv,
                    sa2=(getattr(self, f"_sigma_a2_{color}_a", None)
                         if has_a else None),
                    has_accel=has_a,
                    sa_floor2=sigma_a_floor2,
                    aw=(getattr(self, f"_accel_w_{color}_a", None)
                        if has_a else None))
                hv_fn = _hv_block_ecc if ecc is not None else _hv_block
                results.append(hv_fn(f"_idx_{color}{suffix}", **kw))

        # Concat + gather replaces .at[idx].set() scatter ops
        return jnp.concatenate(results, axis=0)[self._inv_order]

    def _eval_adaptive_phi_r(self, x0, y0, D_A, M_BH, v_sys,
                              r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
                              sigma_x_floor2, sigma_y_floor2,
                              var_v_sys, var_v_hv,
                              sigma_a_floor2,
                              ecc=None, periapsis0=None,
                              dperiapsis_dr=0.0,
                              d2i_dr2=0.0, d2Omega_dr2=0.0):
        """Per-spot adaptive r + phi integration.

        Each spot gets a sinh-spaced r grid centered on its projected
        radius, then delegates to _eval_marginal_phi for phi integration.
        """
        conv = D_A * PC_PER_MAS_MPC
        r_min = self._R_phys_lo / conv
        r_max = self._R_phys_hi / conv
        n_local = self._n_r_local

        # Per-spot r centering using the observable that best constrains
        # r for each spot type:
        #   HV:    velocity → r_vel = M*(C_v*sin_i)^2 / (D*dv^2)
        #   sys+a: acceleration → r_acc = sqrt(C_a*M*sin_i / (D^2*|a|))
        #   sys-a: median HV r_vel (fallback, broad peak)
        sin_i = jnp.abs(jnp.sin(i0))
        eps = 1e-30

        # Velocity-based (assumes sin(phi)≈1 for HV)
        dv = self._all_v - v_sys
        r_vel = M_BH * (C_v * sin_i) ** 2 / (D_A * (dv ** 2 + eps))
        r_vel = jnp.clip(r_vel, r_min, r_max)

        # Acceleration-based (assumes cos(phi)≈1 for systemic)
        r_acc = jnp.sqrt(
            C_a * M_BH * sin_i / (D_A ** 2 * (jnp.abs(self._all_a) + eps)))
        r_acc = jnp.clip(r_acc, r_min, r_max)

        # Acceleration S/N: use total sigma (per-spot + floor)
        sigma_a_total = jnp.sqrt(self._all_sigma_a**2 + sigma_a_floor2)
        accel_snr = jnp.abs(self._all_a) / (sigma_a_total + eps)
        # Only trust acceleration-based r if S/N >= 2
        accel_good = self._all_has_accel & (accel_snr >= 2.0)

        # Fallback for unconstrained spots: geometric midpoint of r range
        # with large scale → nearly uniform grid in log-r. The centering
        # location doesn't matter much because the large scale distributes
        # points approximately uniformly across the full range.
        logr_mid = 0.5 * (jnp.log(r_min) + jnp.log(r_max))
        r_fallback = jnp.exp(logr_mid)
        # scale = quarter of total log-range → density varies only ~2x
        # from center to edge, effectively covering the full range
        scale_broad = 0.25 * (jnp.log(r_max) - jnp.log(r_min))

        # Assemble: HV→r_vel, sys+good_accel→r_acc, unconstrained→midpoint
        r_est = jnp.where(
            self.is_highvel, r_vel,
            jnp.where(accel_good, r_acc, r_fallback))
        r_est = jnp.clip(r_est, r_min * 1.01, r_max * 0.99)
        logr_c = jnp.log(r_est)

        # Per-spot scale in log-r:
        #   HV:    from dr/dv: σ(log r) ≈ 2σ_v/|dv|, floor 0.05
        #   sys+a: from dr/da: σ(log r) ≈ σ_a/(2|a|), floor 0.1
        #   unconstrained: scale_broad (nearly uniform over full range)
        sigma_v_eff = jnp.sqrt(var_v_hv)
        sigma_a_eff = jnp.sqrt(sigma_a_floor2)
        sigma_log_vel = 2.0 * sigma_v_eff / (jnp.abs(dv) + eps)
        sigma_log_acc = sigma_a_eff / (2.0 * jnp.abs(self._all_a) + eps)
        scale = jnp.where(
            self.is_highvel,
            jnp.maximum(sigma_log_vel, 0.05),
            jnp.where(accel_good,
                       jnp.maximum(sigma_log_acc, 0.1),
                       scale_broad))

        # Per-spot sinh grid in log-r space
        logr_lo = jnp.log(r_min)
        logr_hi = jnp.log(r_max)
        t_lo = jnp.arcsinh((logr_lo - logr_c) / scale)
        t_hi = jnp.arcsinh((logr_hi - logr_c) / scale)
        u = jnp.linspace(0.0, 1.0, n_local)
        t_grid = t_lo[:, None] + (t_hi - t_lo)[:, None] * u[None, :]
        r_all = jnp.exp(logr_c[:, None] + jnp.sinh(t_grid) * scale[:, None])

        # Per-spot trapezoidal weights (with floor to avoid log(0))
        h = jnp.diff(r_all, axis=-1)
        h_left = jnp.concatenate(
            [jnp.zeros((self.n_spots, 1)), h], axis=-1)
        h_right = jnp.concatenate(
            [h, jnp.zeros((self.n_spots, 1))], axis=-1)
        w = (h_left + h_right) / 2
        log_w_r = jnp.log(jnp.maximum(w, 1e-30))

        # Delegate to existing phi integration
        ecc_kw = {}
        if ecc is not None:
            ecc_kw = dict(ecc=ecc, periapsis0=periapsis0,
                          dperiapsis_dr=dperiapsis_dr)
        return self._eval_marginal_phi(
            r_all, x0, y0, D_A, M_BH, v_sys,
            r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor2, sigma_y_floor2,
            var_v_sys, var_v_hv, sigma_a_floor2,
            log_w_r=log_w_r,
            d2i_dr2=d2i_dr2, d2Omega_dr2=d2Omega_dr2,
            **ecc_kw)

    def _eval_bruteforce_phi(self, r_ang, x0, y0, D_A, M_BH, v_sys,
                             r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
                             sigma_x_floor2, sigma_y_floor2,
                             var_v_sys, var_v_hv, sigma_a_floor2,
                             d2i_dr2=0.0, d2Omega_dr2=0.0):
        """Mode 1 phi-marginal via uniform brute-force grid.

        Integrates over phi on a uniform [0, 2pi] grid using the core
        _phi_integrand. No specialized grids or adaptive centering.

        Parameters
        ----------
        r_ang : (N,) sampled angular radii per spot

        Returns
        -------
        (N,) per-spot phi-marginalised log-likelihood
        """
        n_phi = self._n_phi_bruteforce
        phi = jnp.linspace(0.0, 2 * jnp.pi, n_phi)
        log_w = trapz_log_weights(phi)
        ll = self._phi_integrand(
            r_ang, jnp.sin(phi), jnp.cos(phi),
            x0, y0, D_A, M_BH, v_sys,
            r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor2, sigma_y_floor2,
            var_v_sys, var_v_hv, sigma_a_floor2,
            d2i_dr2=d2i_dr2, d2Omega_dr2=d2Omega_dr2)
        return logsumexp(ll + log_w[None, :], axis=-1)

    def _phi_integrand(self, r_ang, sin_phi, cos_phi,
                       x0, y0, D_A, M_BH, v_sys,
                       r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
                       sigma_x_floor2, sigma_y_floor2,
                       var_v_sys, var_v_hv, sigma_a_floor2,
                       d2i_dr2=0.0, d2Omega_dr2=0.0):
        """Per-spot log-likelihood integrand at given (r, phi) points.

        Core physics: geometry → observables → chi2 → log-integrand.
        Uses model's stored spot data arrays. No integration — callers
        combine with quadrature weights and logsumexp.

        Parameters
        ----------
        r_ang : (N,) per-spot angular radius
        sin_phi, cos_phi : (n_phi,) shared or (N, n_phi) per-spot

        Returns
        -------
        (N, n_phi) log f(r_i, phi_j | data_i, theta)
        """
        i_r, Om_r = warp_geometry(
            r_ang, r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
            d2i_dr2, d2Omega_dr2)
        sin_i = jnp.sin(i_r)
        cos_i = jnp.cos(i_r)
        sin_O = jnp.sin(Om_r)
        cos_O = jnp.cos(Om_r)
        v_kep, gamma, z_g, a_mag, pA, pB, pC, pD = \
            _precompute_r_quantities(
                r_ang, D_A, M_BH, sin_i, cos_i, sin_O, cos_O)

        X, Y, V, A = _observables_from_precomputed(
            sin_phi, cos_phi, x0, y0, v_sys,
            sin_i[:, None], r_ang[:, None],
            v_kep[:, None], gamma[:, None], z_g[:, None], a_mag[:, None],
            pA[:, None], pB[:, None], pC[:, None], pD[:, None])

        var_x = self._all_sigma_x2 + sigma_x_floor2
        var_y = self._all_sigma_y2 + sigma_y_floor2
        var_v = self._all_sigma_v2 + jnp.where(
            self.is_highvel, var_v_hv, var_v_sys)
        var_a = self._all_sigma_a2 + sigma_a_floor2

        # Position residuals in float64 to avoid catastrophic cancellation
        # (NGC4258: x~4000 μas, σ_x~3 μas → 4 sig. digits in float32).
        f64 = jnp.float64
        dx = self._all_x[:, None].astype(f64) - X.astype(f64)
        dy = self._all_y[:, None].astype(f64) - Y.astype(f64)
        chi2_pos = (dx * dx / var_x[:, None].astype(f64)
                    + dy * dy / var_y[:, None].astype(f64))
        chi2 = (chi2_pos.astype(X.dtype)
                + (self._all_v[:, None] - V) ** 2 / var_v[:, None])

        chi2_a = ((self._all_a[:, None] - A) ** 2
                  / var_a[:, None]
                  * self._all_accel_w[:, None])
        chi2 = chi2 + chi2_a * self._all_has_accel[:, None]

        lnorm = -0.5 * (3 * LOG_2PI + jnp.log(var_x)
                         + jnp.log(var_y) + jnp.log(var_v))
        lnorm_a = (-0.5 * (LOG_2PI + jnp.log(var_a))
                   * self._all_accel_w
                   * self._all_has_accel)

        return (lnorm + lnorm_a)[:, None] - 0.5 * chi2

    def _eval_adaptive_phi_mode1(
            self, r_ang, x0, y0, D_A, M_BH, v_sys,
            r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor2, sigma_y_floor2,
            var_v_sys, var_v_hv, sigma_a_floor2,
            ecc=None, periapsis0=None, dperiapsis_dr=0.0,
            d2i_dr2=0.0, d2Omega_dr2=0.0):
        """Mode 1 per-spot log-likelihood with adaptive phi grid.

        At each spot's sampled r_ang, finds phi_peak via the 2x2 position
        solve (exact when s*^2+c*^2=1 at the correct r), builds a local
        sinh-spaced phi grid, and integrates numerically.

        Parameters
        ----------
        r_ang : (N,) sampled angular radii per spot

        Returns
        -------
        (N,) per-spot phi-marginalised log-likelihood
        """
        eps = 1e-30
        n_phi = self._n_phi_adaptive
        K = self._K_sigma_phi
        N = r_ang.shape[0]

        # Warp at each spot's sampled r
        i_r, Om_r = warp_geometry(
            r_ang, r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
            d2i_dr2, d2Omega_dr2)
        cos_i = jnp.cos(i_r)
        sin_O = jnp.sin(Om_r)
        cos_O = jnp.cos(Om_r)

        # Position coefficients: (N,)
        a1 = sin_O
        a2 = -cos_O * cos_i
        b1 = cos_O
        b2 = sin_O * cos_i

        # 2x2 solve for phi* at each spot's r
        dx = self._all_x - x0
        dy = self._all_y - y0
        R_ang = r_ang * 1e3  # mas → μas for position solve
        rhs_x = dx / (R_ang + eps)
        rhs_y = dy / (R_ang + eps)
        det = a1 * b2 - a2 * b1
        safe_det = jnp.where(jnp.abs(det) > eps, det, eps)
        s_star = (rhs_x * b2 - rhs_y * a2) / safe_det
        c_star = (rhs_y * a1 - rhs_x * b1) / safe_det
        phi_star = jnp.arctan2(s_star, c_star)

        # sigma_phi from position Hessian
        sf = jnp.sin(phi_star)
        cf = jnp.cos(phi_star)
        dXdphi = R_ang * (a1 * cf - a2 * sf)
        dYdphi = R_ang * (b1 * cf - b2 * sf)
        var_x = self._all_sigma_x2 + sigma_x_floor2
        var_y = self._all_sigma_y2 + sigma_y_floor2
        H = dXdphi**2 / var_x + dYdphi**2 / var_y
        sigma_phi = jnp.clip(1.0 / jnp.sqrt(H + eps), 1e-6, jnp.pi / 2)

        # Per-spot sinh grid: (N, n_phi)
        u = jnp.linspace(0.0, 1.0, n_phi)
        t_lo = jnp.arcsinh(-K * jnp.ones(N))
        t_hi = jnp.arcsinh(K * jnp.ones(N))
        t = t_lo[:, None] + (t_hi - t_lo)[:, None] * u[None, :]
        phi_grid = phi_star[:, None] + jnp.sinh(t) * sigma_phi[:, None]

        sin_phi = jnp.sin(phi_grid)
        cos_phi = jnp.cos(phi_grid)

        # Trapezoidal weights
        h = jnp.diff(phi_grid, axis=-1)
        h_l = jnp.concatenate([jnp.zeros((N, 1)), h], axis=-1)
        h_r = jnp.concatenate([h, jnp.zeros((N, 1))], axis=-1)
        log_w_phi = jnp.log(jnp.maximum((h_l + h_r) / 2, 1e-30))

        ll = self._phi_integrand(
            r_ang, sin_phi, cos_phi,
            x0, y0, D_A, M_BH, v_sys,
            r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
            sigma_x_floor2, sigma_y_floor2,
            var_v_sys, var_v_hv, sigma_a_floor2,
            d2i_dr2=d2i_dr2, d2Omega_dr2=d2Omega_dr2)

        return logsumexp(ll + log_w_phi, axis=-1)

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
        if self._D_c_volume:
            factor("D_c_volume", 2 * jnp.log(D_c))

        z_cosmo = self.distance2redshift(
            jnp.atleast_1d(D_c), h=h).squeeze()
        D_A = D_c / (1 + z_cosmo)

        # Redshift likelihood: cz_cosmo vs v_sys_obs with sigma_pec scatter
        sigma_pec = shared_params["sigma_pec"]
        cz_cosmo = SPEED_OF_LIGHT * z_cosmo
        factor("ll_redshift",
               normal_logpdf_var(cz_cosmo, self.v_sys_obs, sigma_pec**2))

        eta = rsample("eta", self.priors["eta"],
                      shared_params)
        log_MBH = deterministic("log_MBH", eta + jnp.log10(D_A))
        M_BH = 10.0**(log_MBH - 7.0)  # in units of 1e7 M_sun
        x0 = rsample("x0", self.priors["x0"], shared_params)  # μas
        y0 = rsample("y0", self.priors["y0"], shared_params)  # μas

        i0_deg = rsample("i0", self.priors["i0"], shared_params)
        Omega0_deg = rsample("Omega0", self.priors["Omega0"], shared_params)
        dOmega_dr_deg = rsample(
            "dOmega_dr", self.priors["dOmega_dr"], shared_params)

        i0 = jnp.deg2rad(i0_deg)
        Omega0 = jnp.deg2rad(Omega0_deg)
        dOmega_dr = jnp.deg2rad(dOmega_dr_deg)

        di_dr_deg = rsample("di_dr", self.priors["di_dr"], shared_params)
        di_dr = jnp.deg2rad(di_dr_deg)

        sigma_x_floor2 = rsample(
            "sigma_x_floor", self.priors["sigma_x_floor"],
            shared_params)**2  # μas²
        sigma_y_floor2 = rsample(
            "sigma_y_floor", self.priors["sigma_y_floor"],
            shared_params)**2  # μas²
        var_v_sys = rsample(
            "sigma_v_sys", self.priors["sigma_v_sys"], shared_params)**2
        var_v_hv = rsample(
            "sigma_v_hv", self.priors["sigma_v_hv"], shared_params)**2
        sigma_a_floor2 = rsample(
            "sigma_a_floor", self.priors["sigma_a_floor"],
            shared_params)**2

        dv_sys = rsample("dv_sys", self.priors["dv_sys"], shared_params)
        v_sys = self.v_sys_obs + dv_sys

        # Eccentricity parameters (optional)
        ecc_kw = {}
        if self.use_ecc:
            ecc = rsample("ecc", self.priors["ecc"], shared_params)
            periapsis_deg = rsample(
                "periapsis", self.priors["periapsis"], shared_params)
            periapsis0 = jnp.deg2rad(periapsis_deg)
            dperiapsis_dr_deg = rsample(
                "dperiapsis_dr", self.priors["dperiapsis_dr"],
                shared_params)
            dperiapsis_dr = jnp.deg2rad(dperiapsis_dr_deg)
            ecc_kw = dict(ecc=ecc, periapsis0=periapsis0,
                          dperiapsis_dr=dperiapsis_dr)

        # Quadratic warp terms (optional)
        quad_kw = {}
        if self.use_quadratic_warp:
            d2i_dr2_deg = rsample(
                "d2i_dr2", self.priors["d2i_dr2"], shared_params)
            d2Omega_dr2_deg = rsample(
                "d2Omega_dr2", self.priors["d2Omega_dr2"], shared_params)
            quad_kw = dict(d2i_dr2=jnp.deg2rad(d2i_dr2_deg),
                           d2Omega_dr2=jnp.deg2rad(d2Omega_dr2_deg))

        args = (x0, y0, D_A, M_BH, v_sys,
                self._r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
                sigma_x_floor2, sigma_y_floor2, var_v_sys, var_v_hv,
                sigma_a_floor2)

        if self.phi_method != "default" and self.marginalise_r:
            raise ValueError(
                f"phi_method='{self.phi_method}' requires Mode 1 "
                f"(marginalise_r=False).")

        if self.marginalise_r:
            if self._adaptive_r:
                ll_per_spot = self._eval_adaptive_phi_r(
                    *args, **ecc_kw, **quad_kw)
            else:
                r_ang_grid = self.build_r_ang_grid(D_A)
                log_w_r = trapz_log_weights(r_ang_grid)
                r_all = jnp.broadcast_to(
                    r_ang_grid[None, :],
                    (self.n_spots, len(r_ang_grid)))
                ll_per_spot = self._eval_marginal_phi(
                    r_all, *args, log_w_r=log_w_r, **ecc_kw, **quad_kw)
            ll_disk = jnp.sum(ll_per_spot)

        else:
            # Sample in log-r for better NUTS geometry, but retain
            # uniform prior on r_ang via Jacobian: p(r) = const →
            # p(log r) ∝ r → factor(log r) corrects uniform-in-log to
            # uniform-in-r.
            log_r_lo = jnp.log(self._r_ang_lo)
            log_r_hi = jnp.log(self._r_ang_hi)
            with plate("spots", self.n_spots):
                log_r = sample(
                    "log_r_ang", Uniform(log_r_lo, log_r_hi))
            r_spots = deterministic("r_ang", jnp.exp(log_r))
            # Jacobian: uniform in r ↔ p(log r) ∝ r = exp(log r)
            factor("ll_r_jacobian", jnp.sum(log_r))

            if self.phi_method == "adaptive":
                ll_per_spot = self._eval_adaptive_phi_mode1(
                    r_spots, *args, **ecc_kw, **quad_kw)
            elif self.phi_method == "bruteforce":
                ll_per_spot = self._eval_bruteforce_phi(
                    r_spots, *args, **quad_kw)
            else:
                ll_per_spot = self._eval_marginal_phi(
                    r_spots, *args, **ecc_kw, **quad_kw)
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
