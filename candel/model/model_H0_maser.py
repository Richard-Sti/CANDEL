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
from numpyro.distributions import TruncatedNormal, Uniform

from ..util import SPEED_OF_LIGHT, fprint, fsection, get_nested
from .base_model import ModelBase
from .integration import ln_trapz_precomputed, trapz_log_weights
from .pv_utils import rsample

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


def estimate_from_data(data, H0=73.0):
    """Estimate D_c and log10(M_BH) from spot data.

    D_c from Hubble flow with second-order correction.
    log_MBH from per-spot Keplerian relation (biased low by sin^2(i)).
    """
    v_sys_obs = data["v_sys_obs"]
    is_hv = data["is_highvel"]

    z_est = v_sys_obs / SPEED_OF_LIGHT
    q0 = -0.5275
    D_c = float(SPEED_OF_LIGHT * z_est / H0
                * (1 + 0.5 * (1 - q0) * z_est))
    D_A = D_c / (1 + z_est)

    r_ang_hv = _np.sqrt(data["x"][is_hv]**2 + data["y"][is_hv]**2)
    dv_hv = _np.abs(data["velocity"][is_hv] - v_sys_obs)
    M_per_spot = dv_hv**2 * r_ang_hv * D_A / C_v**2
    log_MBH = float(_np.log10(max(float(_np.median(M_per_spot)), 1.0)))

    fprint(f"data estimates: D_c = {D_c:.1f} Mpc, "
           f"log_MBH = {log_MBH:.2f}")
    return D_c, log_MBH


# -----------------------------------------------------------------------
# Grid construction (pure numpy, called once at init)
# -----------------------------------------------------------------------


def _build_phi_half_grid_hv(G_half=251, c_min=0.0001, c_max=0.9999,
                            n_patch=8):
    """Arccos-spaced half-grid on [0, pi/2] for HV spots.

    Uniform in cos(phi) gives density proportional to sin(phi) — dense
    near phi=pi/2 where high-velocity masers sit (maximum LOS velocity),
    sparse near phi=0.  The sparse low-phi tail is patched with a short
    linear segment to avoid a coarse gap there.
    """
    c = _np.linspace(c_max, c_min, G_half)   # cos(phi): 1 -> 0
    phi = _np.arccos(c)                        # phi: 0 -> pi/2
    # Near phi=0 arccos is sparse; replace first n_patch with linear spacing
    phi_cut = phi[n_patch]
    phi[:n_patch] = _np.linspace(phi[0], phi_cut, n_patch + 2)[1:-1]
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

        # Resolve data-estimate priors using spot data
        D_c_est, log_MBH_est = estimate_from_data(data)
        self._resolve_data_estimate_priors(
            D_c_est, log_MBH_est, data["v_sys_obs"])

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

        _labels = ("sys", "red", "blue")
        n_a = sum(getattr(self, f"_n_{lb}_a") for lb in _labels)
        n_noa = sum(getattr(self, f"_n_{lb}_noa") for lb in _labels)
        fprint(f"accel split: {n_a} with, {n_noa} without.")

        # ---- Phi grids and precomputed trig ----
        G_half = int(get_nested(self.config, "model/G_phi_half", 251))
        G_sys = int(get_nested(self.config, "model/G_phi_sys", 501))
        phi_half = _build_phi_half_grid_hv(G_half=G_half)
        phi_sys = _build_phi_grid_sys(G=G_sys)

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

        self._log_w_phi_hv = jnp.asarray(trapz_log_weights(phi_half))
        self._log_w_phi_sys = jnp.asarray(trapz_log_weights(phi_sys))

        # ---- Phi prior (optional) ----
        self.phi_prior = get_nested(self.config, "model/phi_prior", False)

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
            n_r = int(get_nested(self.config, "model/n_r", 251))
            R_grid = _build_r_grid(R_min, R_max, n_r=n_r)
            R_phys_grid = jnp.asarray(R_grid)
            # r_ang grid set after D_A_est is computed below

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
        # Pivot at the median projected HV-spot radius to decorrelate i0 from di_dr.
        x_hv = _np.asarray(data["x"])[is_hv_np]
        y_hv = _np.asarray(data["y"])[is_hv_np]
        r_ang_hv = _np.sqrt(x_hv**2 + y_hv**2)
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
                R_phys_grid / (D_A_est * PC_PER_MAS_MPC))
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
        fprint(f"phi_prior = {self.phi_prior}")

    def _resolve_data_estimate_priors(self, D_c_est, log_MBH_est, v_sys_obs):
        """Replace data-estimate prior sentinels with concrete dists."""
        from candel.model.utils import VolumePrior

        p = self.priors.get("D")
        if isinstance(p, dict) and p.get("type") == "data_estimate_uniform":
            hw = p["half_width"]
            lo = max(D_c_est - hw, 1.0)
            hi = D_c_est + hw
            self.priors["D"] = Uniform(lo, hi)
            fprint(f"D prior: U({lo:.1f}, {hi:.1f})")
        elif isinstance(p, dict) and p.get("type") == "data_estimate_volume":
            hw = p["half_width"]
            lo = max(D_c_est - hw, 1.0)
            hi = D_c_est + hw
            self.priors["D"] = VolumePrior(lo, hi)
            fprint(f"D prior: Volume({lo:.1f}, {hi:.1f})")

        p = self.priors.get("log_MBH")
        if isinstance(p, dict) and p.get("type") == "data_estimate_truncated_normal":  # noqa
            self.priors["log_MBH"] = TruncatedNormal(
                log_MBH_est, p["scale"], low=p["low"], high=p["high"])
            fprint(f"log_MBH prior: TruncatedNormal("
                   f"{log_MBH_est:.2f}, {p['scale']}, "
                   f"[{p['low']}, {p['high']}])")

        p = self.priors.get("eta")
        if isinstance(p, dict) and p.get("type") == "data_estimate_uniform":
            z_est = v_sys_obs / SPEED_OF_LIGHT
            D_A_est = D_c_est / (1 + z_est)
            log_mod_est = log_MBH_est - _np.log10(D_A_est)
            hw = p["half_width"]
            self.priors["eta"] = Uniform(
                log_mod_est - hw, log_mod_est + hw)
            fprint(f"eta prior: U({log_mod_est - hw:.3f}, "
                   f"{log_mod_est + hw:.3f})  (est={log_mod_est:.3f})")

    def _eval_marginal_phi(self, r_ang, x0, y0, D_A, M_BH, v_sys,
                           r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
                           sigma_x_floor2, sigma_y_floor2,
                           var_v_sys, var_v_hv,
                           sigma_a_floor2,
                           log_w_r=None,
                           phi_mu_red=None, phi_sigma_red=None,
                           phi_mu_blue=None, phi_sigma_blue=None,
                           phi_mu_sys=None, phi_sigma_sys=None):
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
            r_ang, r_ang_ref, i0, di_dr, Omega0, dOmega_dr)
        sin_i = jnp.sin(i_r)
        cos_i = jnp.cos(i_r)
        sin_O = jnp.sin(Omega_r)
        cos_O = jnp.cos(Omega_r)
        v_kep, gamma, z_g_factor, a_mag, pA, pB, pC, pD = \
            _precompute_r_quantities(r_ang, D_A, M_BH,
                                     sin_i, cos_i, sin_O, cos_O)

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

        def _lnorm_3(sx2, sy2, sv2, sv_floor2):
            return -0.5 * (3 * LOG_2PI + jnp.log(sx2 + sigma_x_floor2)
                           + jnp.log(sy2 + sigma_y_floor2)
                           + jnp.log(sv2 + sv_floor2))

        def _lnorm_4(sx2, sy2, sv2, sv_floor2, sa2, sa_floor2):
            return (_lnorm_3(sx2, sy2, sv2, sv_floor2)
                    - 0.5 * (LOG_2PI + jnp.log(sa2 + sa_floor2)))

        # Phi prior weights (truncated Gaussian on phi grids).
        # For HV: prior on phi_half ∈ [0, π/2]; symmetric under reflection
        # so both modes get the same weight → reflection trick preserved.
        # For sys: prior on phi ∈ [-π/2, π/2].
        def _phi_prior_weights(phi_grid, base_w, mu, sigma):
            """Add truncated Gaussian log-prior to trapezoidal weights."""
            if sigma is None:
                return base_w
            lp = -0.5 * ((phi_grid - mu) / sigma)**2
            # Stable log(CDF(b) - CDF(a)): use log1p(-exp(logcdf_a -
            # logcdf_b)) + logcdf_b to avoid catastrophic cancellation
            # when mu is far outside [grid[0], grid[-1]].
            logcdf_hi = jax_norm.logcdf((phi_grid[-1] - mu) / sigma)
            logcdf_lo = jax_norm.logcdf((phi_grid[0] - mu) / sigma)
            log_Z = logcdf_hi + jnp.log1p(-jnp.exp(logcdf_lo - logcdf_hi))
            # Clamp log_Z to avoid -inf when mu is far from the grid.
            # Dtype-aware: log(tiny) is ~ -87 for f32, -708 for f64.
            log_Z = jnp.maximum(
                log_Z, jnp.log(jnp.finfo(log_Z.dtype).tiny) + 10)
            return base_w + lp - jnp.log(sigma) - log_Z

        log_w_phi_red = _phi_prior_weights(
            self._phi_half, self._log_w_phi_hv, phi_mu_red, phi_sigma_red)
        log_w_phi_blue = _phi_prior_weights(
            self._phi_half, self._log_w_phi_hv, phi_mu_blue, phi_sigma_blue)
        log_w_phi_sys = _phi_prior_weights(
            self._phi_sys, self._log_w_phi_sys, phi_mu_sys, phi_sigma_sys)

        # Combined phi[+r] weights for fused 2D logsumexp in Mode 2.
        if log_w_r is not None:
            log_w_2d_sys = log_w_r[:, None] + log_w_phi_sys[None, :]
            log_w_2d_red = log_w_r[:, None] + log_w_phi_red[None, :]
            log_w_2d_blue = log_w_r[:, None] + log_w_phi_blue[None, :]

        # ---- Systemic spots ----
        def _sys_block(idx_attr, log_w_r, log_w_2d,
                       x_d, y_d, v_d, a_d,
                       sx2, sy2, sv2, sv_floor2, sa2, has_accel,
                       sa_floor2):
            vx = sx2[dpad] + sigma_x_floor2
            vy = sy2[dpad] + sigma_y_floor2
            vv = sv2[dpad] + sv_floor2
            idx = getattr(self, idx_attr)
            if has_accel:
                va = sa2[dpad] + sa_floor2
                X, Y, V, A = _obs_4(
                    idx, self._sin_phi_sys, self._cos_phi_sys)
                chi2 = _chi2_4obs(
                    x_d[dpad], X, 1.0 / vx, y_d[dpad], Y, 1.0 / vy,
                    v_d[dpad], V, 1.0 / vv, a_d[dpad], A, 1.0 / va)
                lnorm = _lnorm_4(sx2[dpad], sy2[dpad], sv2[dpad],
                                 sv_floor2, sa2[dpad], sa_floor2)
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
                return logsumexp(ll + log_w_2d, axis=(-2, -1))
            return ln_trapz_precomputed(ll, log_w_phi_sys, axis=-1)

        def _hv_block(idx_attr, sp1, cp1, sp2, cp2, log_w_phi,
                      log_w_r, log_w_2d,
                      x_d, y_d, v_d, a_d,
                      sx2, sy2, sv2, sv_floor2, sa2, has_accel,
                      sa_floor2):
            vx = sx2[dpad] + sigma_x_floor2
            vy = sy2[dpad] + sigma_y_floor2
            vv = sv2[dpad] + sv_floor2
            inv_vx, inv_vy, inv_vv = 1.0 / vx, 1.0 / vy, 1.0 / vv
            idx = getattr(self, idx_attr)
            r_sub = r_ang[idx][rpad]
            pa_s, pb_s = pA[idx][rpad], pB[idx][rpad]
            pc_s, pd_s = pC[idx][rpad], pD[idx][rpad]

            if has_accel:
                va = sa2[dpad] + sa_floor2
                inv_va = 1.0 / va
                X1, Y1, V, A1 = _obs_4(idx, sp1, cp1)
                X2 = x0 + r_sub * (sp2 * pa_s - cp2 * pb_s)
                Y2 = y0 + r_sub * (sp2 * pc_s + cp2 * pd_s)
                A2 = -A1
                chi2_v = (v_d[dpad] - V) ** 2 * inv_vv
                chi2_1 = (_chi2_4obs(
                    x_d[dpad], X1, inv_vx, y_d[dpad], Y1, inv_vy,
                    v_d[dpad], V, inv_vv,
                    a_d[dpad], A1, inv_va) - chi2_v)
                chi2_2 = (_chi2_4obs(
                    x_d[dpad], X2, inv_vx, y_d[dpad], Y2, inv_vy,
                    v_d[dpad], V, inv_vv,
                    a_d[dpad], A2, inv_va) - chi2_v)
                lnorm = _lnorm_4(sx2[dpad], sy2[dpad], sv2[dpad],
                                 sv_floor2, sa2[dpad], sa_floor2)
            else:
                X1, Y1, V = _obs_3(idx, sp1, cp1)
                X2 = x0 + r_sub * (sp2 * pa_s - cp2 * pb_s)
                Y2 = y0 + r_sub * (sp2 * pc_s + cp2 * pd_s)
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
                return logsumexp(ll + log_w_2d, axis=(-2, -1))
            return ln_trapz_precomputed(ll, log_w_phi, axis=-1)

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
                    sa_floor2=sigma_a_floor2)
                results.append(_sys_block(f"_idx_sys{suffix}", **kw))

        # ---- Red and Blue HV: with accel, then without ----
        hv_colors = [
            ("red", log_w_phi_red,
             log_w_2d_red if log_w_r is not None else None),
            ("blue", log_w_phi_blue,
             log_w_2d_blue if log_w_r is not None else None),
        ]
        for color, lw_phi, lw_2d in hv_colors:
            for suffix, has_a in [("_a", True), ("_noa", False)]:
                if getattr(self, f"_n_{color}{suffix}") == 0:
                    continue
                kw = dict(
                    sp1=getattr(self, f"_sin_phi1_{color}"),
                    cp1=getattr(self, f"_cos_phi1_{color}"),
                    sp2=getattr(self, f"_sin_phi2_{color}"),
                    cp2=getattr(self, f"_cos_phi2_{color}"),
                    log_w_phi=lw_phi,
                    log_w_r=log_w_r,
                    log_w_2d=lw_2d,
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
                    sa_floor2=sigma_a_floor2)
                results.append(
                    _hv_block(f"_idx_{color}{suffix}", **kw))

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

        eta = rsample("eta", self.priors["eta"],
                               shared_params)
        log_MBH = deterministic("log_MBH",
                                eta + jnp.log10(D_A))
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

        di_dr_deg = rsample("di_dr", self.priors["di_dr"], shared_params)
        di_dr = jnp.deg2rad(di_dr_deg)

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
            "sigma_a_floor", self.priors["sigma_a_floor"],
            shared_params)**2

        dv_sys = rsample("dv_sys", self.priors["dv_sys"], shared_params)
        v_sys = self.v_sys_obs + dv_sys

        # Phi prior parameters (optional)
        phi_kw = {}
        if self.phi_prior:
            phi_mu_red = rsample(
                "phi_mu_red", self.priors["phi_mu_red"], shared_params)
            phi_sigma_red = rsample(
                "phi_sigma_red", self.priors["phi_sigma_red"], shared_params)
            phi_mu_blue = rsample(
                "phi_mu_blue", self.priors["phi_mu_blue"], shared_params)
            phi_sigma_blue = rsample(
                "phi_sigma_blue", self.priors["phi_sigma_blue"], shared_params)
            phi_mu_sys = rsample(
                "phi_mu_sys", self.priors["phi_mu_sys"], shared_params)
            phi_sigma_sys = rsample(
                "phi_sigma_sys", self.priors["phi_sigma_sys"], shared_params)
            phi_kw = dict(
                phi_mu_red=jnp.deg2rad(phi_mu_red),
                phi_sigma_red=jnp.deg2rad(phi_sigma_red),
                phi_mu_blue=jnp.deg2rad(phi_mu_blue),
                phi_sigma_blue=jnp.deg2rad(phi_sigma_blue),
                phi_mu_sys=jnp.deg2rad(phi_mu_sys),
                phi_sigma_sys=jnp.deg2rad(phi_sigma_sys))

        args = (x0, y0, D_A, M_BH, v_sys,
                self._r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
                sigma_x_floor2, sigma_y_floor2, var_v_sys, var_v_hv,
                sigma_a_floor2)

        if self.marginalise_r:
            r_all = jnp.broadcast_to(
                self._r_ang_grid[None, :],
                (self.n_spots, len(self._r_ang_grid)))

            ll_per_spot = self._eval_marginal_phi(
                r_all, *args, log_w_r=self._log_w_R, **phi_kw)
            ll_disk = jnp.sum(ll_per_spot)

        else:
            with plate("spots", self.n_spots):
                r_spots = sample(
                    "r_ang", Uniform(self._r_ang_lo, self._r_ang_hi))

            ll_per_spot = self._eval_marginal_phi(r_spots, *args, **phi_kw)
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
