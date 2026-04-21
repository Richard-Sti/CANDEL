# Copyright (C) 2026 Richard Stiskalek
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
"""CPU+MPI mode2-like runner for a single maser galaxy (NGC4258-oriented).

Layout:
  - Each MPI rank owns a disjoint subset of maser spots. If the
    communicator size equals the spot count after any `--max-spots`
    truncation, this is one rank per spot; otherwise ranks hold
    round-robin subsets `spots[r::SIZE]`. Every rank builds its own
    `MaserDiskModel` instance (forced to `mode="mode2"` even for
    galaxies with `forbid_marginalise_r=true`).
  - All ranks execute in lockstep. Rank 0 drives either differential
    evolution (MAP) or ultranest (posterior). For each parameter vector,
    rank 0 broadcasts a packed float64 buffer, every rank evaluates its
    one-spot log-marginal with the existing
    `MaserDiskModel._eval_phi_marginal`, and `Reduce(SUM)` returns the
    galaxy log-likelihood to rank 0.
  - Log-prior on global parameters is added on rank 0 only after the
    reduction (DE) or handled by the ultranest prior transform (NS).

Usage (inside the package venv, glamdring or any MPI host):
    export OMP_NUM_THREADS=1
    mpirun -n <N_SPOTS> python scripts/megamaser/run_mode2_mpi.py \\
        NGC4258 --method de|ns [options]
"""
import os
import sys

# ---- Single-thread each rank BEFORE importing numpy/jax ----
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["JAX_PLATFORMS"] = "cpu"
# Force XLA to use a single CPU thread per JAX process. With many MPI
# ranks per node, we don't want each JAX process grabbing all cores.
_xla = os.environ.get("XLA_FLAGS", "")
_xla_additions = ("--xla_cpu_multi_thread_eigen=false "
                  "intra_op_parallelism_threads=1")
if _xla_additions not in _xla:
    os.environ["XLA_FLAGS"] = (_xla + " " + _xla_additions).strip()

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import tomli
import tomli_w

from mpi4py import MPI

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from h5py import File as H5File

from candel.model.model_H0_maser import MaserDiskModel, PC_PER_MAS_MPC
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import fprint, fsection, get_nested


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

# Silence non-root ranks at the file-descriptor level so that Python print,
# C extensions (JAX, MPI banners), and numpy warnings all go to /dev/null.
if RANK != 0:
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 1)
    os.dup2(_devnull, 2)
    os.close(_devnull)


def _log(msg):
    if RANK == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Parameter spec & priors
# ---------------------------------------------------------------------------

_BASE_PARAMS = (
    "D_c", "eta", "dv_sys", "x0", "y0",
    "i0", "Omega0", "dOmega_dr", "di_dr",
    "sigma_x_floor", "sigma_y_floor",
    "sigma_v_sys", "sigma_v_hv", "sigma_a_floor",
)
_ECC_CART_PARAMS = ("e_x", "e_y", "dperiapsis_dr")
_QUAD_WARP_PARAMS = ("d2i_dr2", "d2Omega_dr2")
_PRIOR_KEY = {"D_c": "D"}  # parameter name -> prior key used by model.priors


def _prior_for(model, name):
    """Look up the numpyro distribution object for a sampled site."""
    return model.priors[_PRIOR_KEY.get(name, name)]


def build_param_spec(model):
    """Ordered list of sampled parameter names for this model."""
    names = list(_BASE_PARAMS)
    if model.use_ecc:
        if not model.ecc_cartesian:
            raise ValueError(
                "Only ecc_cartesian=True is supported by run_mode2_mpi.")
        names.extend(_ECC_CART_PARAMS)
    if model.use_quadratic_warp:
        names.extend(_QUAD_WARP_PARAMS)
    return names


# ---------------------------------------------------------------------------
# Data subsetting for per-rank spot subsets
# ---------------------------------------------------------------------------

def _subset_spots(full_data, indices):
    """Return a copy of `full_data` containing only the listed spots.

    `indices` is a numpy array / list of integer indices into the full
    spot arrays. Scalar keys (galaxy_name, v_sys_obs, velocity_frame,
    D_lo, D_hi, …) are passed through unchanged.
    """
    idx = np.asarray(indices, dtype=int)
    out = {}
    for k, v in full_data.items():
        if isinstance(v, np.ndarray):
            out[k] = v[idx].copy()
        elif isinstance(v, list):
            out[k] = [v[int(i)] for i in idx]
        else:
            out[k] = v
    out["n_spots"] = int(idx.size)
    return out


# ---------------------------------------------------------------------------
# Config overrides (force mode2 for the target galaxy)
# ---------------------------------------------------------------------------

def _build_runtime_config(master_cfg, galaxy, grid_overrides, full_data):
    """Copy master config and force mode2 for `galaxy`.

    `grid_overrides` is a dict of CLI overrides for n_phi_*/n_r_* keys; any
    non-None entry replaces both the global default and the per-galaxy
    override for that key.

    `full_data` is the galaxy data dict (all spots); used to pin
    `r_ang_ref*` from the full HV-spot median so per-rank n_spots=1
    instances do not re-derive it from a single-row subset.
    """
    cfg = {
        "inference": dict(master_cfg.get("inference", {})),
        "model": dict(master_cfg["model"]),
        "io": dict(master_cfg["io"]),
        "optimise": dict(master_cfg.get("optimise", {})),
    }
    cfg["model"]["galaxies"] = {
        k: dict(v) if isinstance(v, dict) else v
        for k, v in master_cfg["model"]["galaxies"].items()
    }

    gcfg = cfg["model"]["galaxies"].get(galaxy, {})
    # Force mode2 for this galaxy and lift the block on r marginalisation.
    gcfg["mode"] = "mode2"
    gcfg.pop("forbid_marginalise_r", None)

    # Pin warp pivots from the full galaxy (n_spots=1 subsets would fail
    # `_configure_warp_pivots`, which defaults to the median HV radius).
    is_hv = np.asarray(full_data["is_highvel"])
    x_hv = np.asarray(full_data["x"])[is_hv]
    y_hv = np.asarray(full_data["y"])[is_hv]
    r_ang_hv = np.sqrt(x_hv**2 + y_hv**2) / 1e3  # μas → mas
    r_median = float(np.median(r_ang_hv))
    gcfg.setdefault("r_ang_ref", r_median)
    gcfg.setdefault("r_ang_ref_i", r_median)
    gcfg.setdefault("r_ang_ref_Omega", r_median)
    gcfg.setdefault("r_ang_ref_periapsis", r_median / 2.0)

    cfg["model"]["galaxies"][galaxy] = gcfg

    # Apply grid defaults from [mode2_mpi] (these override the [model]
    # defaults that are tuned for GPU). Order of precedence (lowest to
    # highest): [model] global -> [mode2_mpi] global -> [mode2_mpi.galaxies.<gal>]
    # -> CLI override in `grid_overrides`.
    mpi_sec = master_cfg.get("mode2_mpi", {})
    gal_mpi_sec = mpi_sec.get("galaxies", {}).get(galaxy, {})
    for k in ("n_phi_hv_high", "n_phi_hv_low", "n_phi_sys",
              "n_r_local", "n_r_brute"):
        if k in gal_mpi_sec:
            cfg["model"][k] = int(gal_mpi_sec[k])
            gcfg[k] = int(gal_mpi_sec[k])
        elif k in mpi_sec:
            cfg["model"][k] = int(mpi_sec[k])
            gcfg[k] = int(mpi_sec[k])

    # CLI overrides take precedence over everything.
    for k, v in grid_overrides.items():
        if v is None:
            continue
        cfg["model"][k] = int(v)
        gcfg[k] = int(v)

    return cfg


# ---------------------------------------------------------------------------
# Per-rank likelihood function (JIT-compiled)
# ---------------------------------------------------------------------------

def _build_phys_args_jax(model, names, flat):
    """Convert a flat parameter array into (phys_args, phys_kw) for
    `_eval_phi_marginal`. Mirrors `phys_from_sample` but uses jnp, so it
    traces under JIT. `names` is a static Python tuple giving the order
    of parameters in `flat`.
    """
    # Closure: build a name -> index lookup once at JIT trace time.
    idx = {n: i for i, n in enumerate(names)}

    D_c = flat[idx["D_c"]]
    H0_ref = float(get_nested(model.config, "model/H0_ref", 73.0))
    h = H0_ref / 100.0
    z_cosmo = model.distance2redshift(jnp.atleast_1d(D_c), h=h).squeeze()
    D_A = D_c / (1.0 + z_cosmo)

    eta = flat[idx["eta"]]
    M_BH = jnp.power(10.0, eta + jnp.log10(D_A) - 7.0)
    v_sys = model.v_sys_obs + flat[idx["dv_sys"]]

    i0 = jnp.deg2rad(flat[idx["i0"]])
    di_dr = jnp.deg2rad(flat[idx["di_dr"]])
    Omega0 = jnp.deg2rad(flat[idx["Omega0"]])
    dOmega_dr = jnp.deg2rad(flat[idx["dOmega_dr"]])

    phys_args = (
        flat[idx["x0"]], flat[idx["y0"]],
        D_A, M_BH, v_sys,
        model._r_ang_ref_i, model._r_ang_ref_Omega,
        model._r_ang_ref_periapsis,
        i0, di_dr, Omega0, dOmega_dr,
        flat[idx["sigma_x_floor"]] ** 2,
        flat[idx["sigma_y_floor"]] ** 2,
        flat[idx["sigma_v_sys"]] ** 2,
        flat[idx["sigma_v_hv"]] ** 2,
        flat[idx["sigma_a_floor"]] ** 2,
    )

    phys_kw = {}
    if model.use_quadratic_warp:
        phys_kw["d2i_dr2"] = jnp.deg2rad(flat[idx["d2i_dr2"]])
        phys_kw["d2Omega_dr2"] = jnp.deg2rad(flat[idx["d2Omega_dr2"]])
    if model.use_ecc:
        e_x = flat[idx["e_x"]]
        e_y = flat[idx["e_y"]]
        phys_kw["ecc"] = jnp.sqrt(e_x * e_x + e_y * e_y)
        phys_kw["periapsis0"] = jnp.arctan2(e_y, e_x)
        phys_kw["dperiapsis_dr"] = jnp.deg2rad(flat[idx["dperiapsis_dr"]])

    return phys_args, phys_kw


def make_per_rank_logL(model, names):
    """Return a JIT-compiled callable mapping a flat parameter array to
    a scalar log-marginal for this rank's assigned spots (summed over
    them). `names` is the order the array entries must be in (see
    `build_param_spec`).
    """
    names_t = tuple(names)

    # Finite floor replacing NaN / -inf from pathological prior draws so
    # Allreduce(SUM) and downstream samplers never see a non-finite total.
    _LL_FLOOR = -1e18

    def _logL(flat):
        phys_args, phys_kw = _build_phys_args_jax(model, names_t, flat)
        D_A = phys_args[2]
        M_BH = phys_args[3]
        v_sys = phys_args[4]
        sigma_a_floor2 = phys_args[16]
        i0 = phys_args[8]
        var_v_hv = phys_args[15]
        spot_groups = model._build_r_grids_mode2(
            D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
        ll_arr = model._eval_phi_marginal(spot_groups, phys_args, phys_kw)
        ll = jnp.sum(ll_arr)  # sum across this rank's assigned spots
        return jnp.where(jnp.isfinite(ll), ll, _LL_FLOOR)

    return jax.jit(_logL)


# ---------------------------------------------------------------------------
# Priors: log-density, bounds, and unit-cube transform (NS)
# ---------------------------------------------------------------------------

def _dist_bounds(model, name, n_sigma_unbounded=5.0):
    """Return (lo, hi) bounds for a prior, used by DE and by NS when
    clipping an unbounded Normal to a finite box.
    """
    dist_name = model.prior_dist_name[_PRIOR_KEY.get(name, name)]
    d = _prior_for(model, name)
    if dist_name == "uniform":
        return float(d.low), float(d.high)
    if dist_name == "truncated_normal":
        return float(d.low), float(d.high)
    if dist_name == "sine_angle":
        return float(d.low), float(d.high)
    if dist_name == "normal":
        mu = float(d.loc)
        sd = float(d.scale)
        return mu - n_sigma_unbounded * sd, mu + n_sigma_unbounded * sd
    raise ValueError(f"Unsupported prior '{dist_name}' for '{name}'")


def log_prior(model, names, flat):
    """Sum log-prior over all sampled sites plus the Cartesian-ecc
    Jacobian factor (mirrors `factor('ecc_cartesian_jac', ...)` in the
    numpyro model). `flat` is a 1-D numpy array ordered to match `names`.
    Returns a Python float.
    """
    total = 0.0
    for i, n in enumerate(names):
        d = _prior_for(model, n)
        lp = float(np.asarray(d.log_prob(jnp.asarray(float(flat[i])))))
        if not np.isfinite(lp):
            return -np.inf
        total += lp
    if model.use_ecc and model.ecc_cartesian:
        idx = {n: i for i, n in enumerate(names)}
        e_x = float(flat[idx["e_x"]])
        e_y = float(flat[idx["e_y"]])
        r2 = e_x * e_x + e_y * e_y
        total += float(np.log(4.0 / np.pi) - 0.5 * np.log(r2 + 1e-6))
    return total


def prior_transform_factory(model, names):
    """Return a function u ∈ [0,1]^D -> physical parameter array.

    For ultranest. Each dim uses the exact CDF inverse of its prior,
    which pushes most of the probability mass where the prior lives.
    """
    from scipy.stats import truncnorm, norm

    kinds = []
    for n in names:
        d = _prior_for(model, n)
        dn = model.prior_dist_name[_PRIOR_KEY.get(n, n)]
        if dn == "uniform":
            kinds.append(("uniform", float(d.low), float(d.high)))
        elif dn == "truncated_normal":
            mu = float(d.base_dist.loc)
            sd = float(d.base_dist.scale)
            a = (float(d.low) - mu) / sd
            b = (float(d.high) - mu) / sd
            kinds.append(("tnorm", mu, sd, a, b))
        elif dn == "normal":
            kinds.append(("norm", float(d.loc), float(d.scale)))
        elif dn == "sine_angle":
            kinds.append(("sine", float(d.low), float(d.high)))
        else:
            raise ValueError(
                f"Unsupported prior kind '{dn}' for '{n}' in NS transform")

    def transform(u):
        """Vectorised: u has shape (..., ndim). Returns same shape."""
        u = np.asarray(u)
        x = np.empty_like(u)
        for k, spec in enumerate(kinds):
            uk = u[..., k]
            kind = spec[0]
            if kind == "uniform":
                _, lo, hi = spec
                x[..., k] = lo + uk * (hi - lo)
            elif kind == "tnorm":
                _, mu, sd, a, b = spec
                x[..., k] = truncnorm.ppf(uk, a, b, loc=mu, scale=sd)
            elif kind == "norm":
                _, mu, sd = spec
                x[..., k] = norm.ppf(uk, loc=mu, scale=sd)
            elif kind == "sine":
                _, lo_deg, hi_deg = spec
                lo_rad = np.deg2rad(lo_deg)
                hi_rad = np.deg2rad(hi_deg)
                cos_val = np.cos(hi_rad) + uk * (
                    np.cos(lo_rad) - np.cos(hi_rad))
                x[..., k] = np.rad2deg(np.arccos(cos_val))
        return x

    return transform


def _prior_signature(model, names):
    """Serializable signature of the active priors for checkpoint safety."""
    sig = {}
    for name in names:
        key = _PRIOR_KEY.get(name, name)
        dist_name = model.prior_dist_name[key]
        d = _prior_for(model, name)
        entry = {"kind": dist_name}
        if dist_name == "uniform":
            entry["low"] = float(d.low)
            entry["high"] = float(d.high)
        elif dist_name == "truncated_normal":
            entry["loc"] = float(d.base_dist.loc)
            entry["scale"] = float(d.base_dist.scale)
            entry["low"] = float(d.low)
            entry["high"] = float(d.high)
        elif dist_name == "normal":
            entry["loc"] = float(d.loc)
            entry["scale"] = float(d.scale)
        elif dist_name == "sine_angle":
            entry["low"] = float(d.low)
            entry["high"] = float(d.high)
        else:
            raise ValueError(
                f"Unsupported prior '{dist_name}' for '{name}'")
        sig[name] = entry
    return sig


def _de_checkpoint_meta(*, galaxy, names, bounds, model, gcfg, n_spots,
                        mpi_size):
    """Compatibility metadata stored in DE checkpoints.

    Resuming a DE population after changing priors, parameter order, grid
    settings, galaxy, or MPI decomposition is unsafe. Store enough runtime
    metadata to fail closed on those changes.
    """
    grid = {}
    for key in ("n_phi_hv_high", "n_phi_hv_low", "n_phi_sys",
                "n_r_local", "n_r_brute"):
        val = gcfg.get(key)
        grid[key] = None if val is None else int(val)
    return {
        "version": 1,
        "galaxy": str(galaxy),
        "names": list(names),
        "bounds": [[float(lo), float(hi)] for lo, hi in bounds],
        "priors": _prior_signature(model, names),
        "model_flags": {
            "use_ecc": bool(model.use_ecc),
            "ecc_cartesian": bool(model.ecc_cartesian),
            "use_quadratic_warp": bool(model.use_quadratic_warp),
        },
        "grid": grid,
        "n_spots": int(n_spots),
        "mpi_size": int(mpi_size),
    }


# ---------------------------------------------------------------------------
# Collective log-likelihood under MPI
# ---------------------------------------------------------------------------

class CollectiveLikelihood:
    """MPI wrapper around per-rank subset log-likelihoods.

    Communication is explicit and typed:
      - rank 0 broadcasts a small int64 header `(cmd, batch_size)`;
      - for eval commands, rank 0 broadcasts a contiguous float64
        parameter buffer of shape `(batch_size, ndim)`;
      - every rank computes its local spot contribution(s);
      - `Reduce(SUM)` returns the total likelihood vector to rank 0 only.

    This avoids per-call pickling overhead from `Comm.bcast/allreduce` on
    Python objects and lets ultranest batches share a single collective.
    """

    _CMD_STOP = 0
    _CMD_EVAL = 1

    def __init__(self, per_rank_logL, ndim):
        self._logL = per_rank_logL
        self._ndim = int(ndim)
        self._header = np.zeros(2, dtype=np.int64)
        self._param_buf = None
        self._send_ll = None
        self._recv_ll = None

    def _ensure_param_buf(self, batch_size):
        shape = (int(batch_size), self._ndim)
        if self._param_buf is None or self._param_buf.shape != shape:
            self._param_buf = np.empty(shape, dtype=np.float64)
        return self._param_buf

    def _ensure_ll_buf(self, batch_size):
        shape = (int(batch_size),)
        if self._send_ll is None or self._send_ll.shape != shape:
            self._send_ll = np.empty(shape, dtype=np.float64)
        if RANK == 0 and (self._recv_ll is None or self._recv_ll.shape != shape):
            self._recv_ll = np.empty(shape, dtype=np.float64)

    def _bcast_header(self, cmd, batch_size):
        if RANK == 0:
            self._header[:] = (int(cmd), int(batch_size))
        COMM.Bcast(self._header, root=0)
        return int(self._header[0]), int(self._header[1])

    def _run_eval(self, batch_size):
        params = self._ensure_param_buf(batch_size)
        self._ensure_ll_buf(batch_size)
        COMM.Bcast(params, root=0)

        for i in range(batch_size):
            self._send_ll[i] = float(self._logL(params[i]))

        recv = self._recv_ll if RANK == 0 else None
        COMM.Reduce(self._send_ll, recv, op=MPI.SUM, root=0)
        if RANK == 0:
            return float(recv[0]) if batch_size == 1 else recv.copy()
        return None

    def evaluate(self, flat_array):
        """Collectively evaluate one parameter vector.

        Rank 0 must pass a shape `(ndim,)` array; other ranks must pass
        `None`. Returns the scalar total log-likelihood on rank 0 and
        `None` elsewhere.
        """
        if RANK == 0:
            arr = np.asarray(flat_array, dtype=np.float64)
            if arr.shape != (self._ndim,):
                raise ValueError(
                    f"Expected shape ({self._ndim},), got {arr.shape}")
            params = self._ensure_param_buf(1)
            params[0] = arr
        elif flat_array is not None:
            raise ValueError("Non-root ranks must pass None to evaluate().")

        cmd, batch = self._bcast_header(self._CMD_EVAL if RANK == 0 else 0,
                                        1 if RANK == 0 else 0)
        if cmd != self._CMD_EVAL:
            raise RuntimeError(f"Unexpected collective command {cmd}")
        return self._run_eval(batch)

    def evaluate_many(self, flat_batch):
        """Collectively evaluate a batch of parameter vectors.

        Rank 0 must pass shape `(B, ndim)` or `(ndim,)`; other ranks must
        pass `None`. Returns a shape `(B,)` float64 array on rank 0 and
        `None` elsewhere.
        """
        if RANK == 0:
            arr = np.asarray(flat_batch, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2 or arr.shape[1] != self._ndim:
                raise ValueError(
                    f"Expected shape (B, {self._ndim}), got {arr.shape}")
            params = self._ensure_param_buf(arr.shape[0])
            params[:] = np.ascontiguousarray(arr)
            batch_size = int(arr.shape[0])
        elif flat_batch is not None:
            raise ValueError(
                "Non-root ranks must pass None to evaluate_many().")
        else:
            batch_size = 0

        cmd, batch = self._bcast_header(self._CMD_EVAL if RANK == 0 else 0,
                                        batch_size)
        if cmd != self._CMD_EVAL:
            raise RuntimeError(f"Unexpected collective command {cmd}")
        out = self._run_eval(batch)
        if RANK == 0:
            return np.atleast_1d(np.asarray(out, dtype=np.float64))
        return None

    def service_once(self):
        """Service one root-issued command on worker ranks.

        Returns `False` after a STOP command; otherwise `True`.
        """
        cmd, batch = self._bcast_header(0, 0)
        if cmd == self._CMD_STOP:
            return False
        if cmd != self._CMD_EVAL:
            raise RuntimeError(f"Unknown collective command {cmd}")
        self._run_eval(batch)
        return True

    def stop(self):
        if RANK == 0:
            self._bcast_header(self._CMD_STOP, 0)

    def worker_loop(self):
        while self.service_once():
            pass


def _benchmark_likelihood(coll, per_rank_logL, init_vec, *, n_probe=10,
                          has_spots=True):
    """Report steady-state local and collective likelihood timings."""
    init_arr = np.ascontiguousarray(init_vec, dtype=np.float64)

    if has_spots:
        t0 = time.perf_counter()
        for _ in range(n_probe):
            _ = float(per_rank_logL(init_arr))
        t_local = (time.perf_counter() - t0) / n_probe
    else:
        t_local = np.nan
    all_local = COMM.gather(t_local, root=0)

    COMM.Barrier()
    t0 = time.perf_counter() if RANK == 0 else None
    for _ in range(n_probe):
        if RANK == 0:
            coll.evaluate(init_arr)
        else:
            coll.service_once()
    if RANK == 0:
        t_wall = (time.perf_counter() - t0) / n_probe
    else:
        t_wall = None
    COMM.Barrier()

    if RANK == 0:
        if t_wall >= 1.0:
            fprint(f"likelihood wall: {t_wall:.2f} s/call ({n_probe} probes)")
        elif t_wall >= 1e-3:
            fprint(f"likelihood wall: {t_wall*1e3:.1f} ms/call ({n_probe} probes)")
        else:
            fprint(f"likelihood wall: {t_wall*1e6:.1f} μs/call ({n_probe} probes)")

        all_local = np.asarray(all_local, dtype=np.float64)
        active_local = all_local[np.isfinite(all_local)]
        if active_local.size:
            fprint("per-rank local logL: "
                   f"min={active_local.min()*1e3:.3f} ms  "
                   f"mean={active_local.mean()*1e3:.3f} ms  "
                   f"max={active_local.max()*1e3:.3f} ms")


# ---------------------------------------------------------------------------
# Differential evolution (rand/1/bin)
# ---------------------------------------------------------------------------

def run_de(coll, model, names, bounds, *, popsize, maxiter, F, CR, tol, seed,
           checkpoint_path=None, checkpoint_every=25, resume=False,
           checkpoint_meta=None):
    """Rand/1/bin DE on rank 0 using `coll.evaluate` as the collective
    log-posterior. Returns the best sample dict and its log-posterior.

    If ``checkpoint_path`` is set, the full DE state (population, fitness,
    priors, generation index, best, RNG state, compatibility metadata) is
    pickled every ``checkpoint_every`` generations. ``resume=True`` loads
    that checkpoint if present and continues from the saved generation.
    """
    import pickle

    rng = np.random.default_rng(seed)
    D = len(names)
    lo = np.asarray([b[0] for b in bounds])
    hi = np.asarray([b[1] for b in bounds])

    fprint(f"DE config: D={D} params, popsize={popsize}, maxiter={maxiter}, "
           f"F={F}, CR={CR}, tol={tol}, seed={seed}, "
           f"checkpoint_every={checkpoint_every}")
    fprint(f"DE params: {', '.join(names)}")

    resumed = False
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        if state["popsize"] != popsize or state["D"] != D:
            raise RuntimeError(
                f"Checkpoint shape mismatch: popsize {state['popsize']}→"
                f"{popsize}, D {state['D']}→{D}")
        saved_meta = state.get("meta")
        if checkpoint_meta is not None:
            if saved_meta is None:
                raise RuntimeError(
                    "Checkpoint lacks compatibility metadata and cannot be "
                    "safely resumed with the current runner. Start a fresh "
                    "run or resume from a checkpoint created by this version.")
            if saved_meta != checkpoint_meta:
                diff_keys = sorted(
                    k for k in set(saved_meta) | set(checkpoint_meta)
                    if saved_meta.get(k) != checkpoint_meta.get(k))
                diff_txt = ", ".join(diff_keys[:6])
                if len(diff_keys) > 6:
                    diff_txt += ", ..."
                raise RuntimeError(
                    f"Checkpoint metadata mismatch for {checkpoint_path}. "
                    "Resume requires the same galaxy, priors, parameter "
                    f"order, grid settings, spot count, and MPI size. "
                    f"Differing fields: {diff_txt}")
        pop = state["pop"]
        fit = state["fit"]
        prior_vals = state["prior_vals"]
        post = state["post"]
        best_x = state["best_x"]
        best_post = state["best_post"]
        start_gen = state["gen"] + 1
        rng.bit_generator.state = state["rng_state"]
        fprint(f"Resumed DE from {checkpoint_path} at gen {start_gen - 1} "
               f"(best log-post = {best_post:.4f})")
        resumed = True

    if not resumed:
        # Fresh start: uniform-in-box initial population.
        pop = lo + rng.random((popsize, D)) * (hi - lo)
        fit = np.empty(popsize)
        for k in range(popsize):
            fit[k] = coll.evaluate(pop[k])
        prior_vals = np.array([log_prior(model, names, pop[k])
                               for k in range(popsize)])
        post = fit + prior_vals
        best_idx = int(np.argmax(post))
        best_x = pop[best_idx].copy()
        best_post = float(post[best_idx])
        fprint(f"DE init: best log-post = {best_post:.4f} "
               f"(ll = {fit[best_idx]:.4f}, lp = {prior_vals[best_idx]:.4f})")
        start_gen = 1

    def _save_checkpoint(gen):
        if not checkpoint_path:
            return
        tmp = checkpoint_path + ".tmp"
        state = dict(
            pop=pop, fit=fit, prior_vals=prior_vals, post=post,
            best_x=best_x, best_post=best_post, gen=gen,
            rng_state=rng.bit_generator.state,
            popsize=popsize, D=D, meta=checkpoint_meta)
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp, checkpoint_path)

    prev_best = best_post
    gen = start_gen - 1  # ensures `gen` is defined if the loop never runs
    for gen in range(start_gen, maxiter + 1):
        for k in range(popsize):
            # Select 3 distinct indices not equal to k.
            idxs = [j for j in range(popsize) if j != k]
            a, b_, c = rng.choice(idxs, size=3, replace=False)
            mutant = pop[a] + F * (pop[b_] - pop[c])
            # Reflect to bounds.
            mutant = np.where(mutant < lo, lo + (lo - mutant), mutant)
            mutant = np.where(mutant > hi, hi - (mutant - hi), mutant)
            mutant = np.clip(mutant, lo, hi)
            # Binomial crossover with at least one inherited dim.
            mask = rng.random(D) < CR
            if not mask.any():
                mask[rng.integers(D)] = True
            trial = np.where(mask, mutant, pop[k])

            ll_trial = coll.evaluate(trial)
            lp_trial = log_prior(model, names, trial)
            post_trial = ll_trial + lp_trial
            if post_trial > post[k]:
                pop[k] = trial
                fit[k] = ll_trial
                prior_vals[k] = lp_trial
                post[k] = post_trial
                if post_trial > best_post:
                    best_post = post_trial
                    best_x = trial.copy()

        if gen % 25 == 0 or gen == 1:
            fprint(f"DE gen {gen:>5d}: best log-post = {best_post:.4f} "
                   f"(pop std = {np.std(post):.3f})")

        if checkpoint_path and gen % checkpoint_every == 0:
            _save_checkpoint(gen)
            fprint(f"  checkpoint saved at gen {gen}")

        if gen > 50 and abs(best_post - prev_best) < tol and np.std(post) < tol:
            fprint(f"DE converged at gen {gen} (delta < {tol})")
            break
        prev_best = best_post

    # Final checkpoint on normal exit (only if the loop ran at least once).
    if gen >= start_gen:
        _save_checkpoint(gen)

    return {n: float(v) for n, v in zip(names, best_x)}, best_post


# ---------------------------------------------------------------------------
# Ultranest driver
# ---------------------------------------------------------------------------

def run_ns(coll, names, model, out_dir, *, min_live, max_ncalls, seed,
           resume=False, stepsampler="none", step_nsteps=None):
    import ultranest
    import builtins as _builtins
    from ultranest.stepsampler import (RegionSliceSampler, SliceSampler,
                                       generate_mixture_random_direction)

    transform = prior_transform_factory(model, names)
    nsteps = max(1, len(names) if step_nsteps is None else int(step_nsteps))

    idx_lookup = {n: i for i, n in enumerate(names)}

    def loglike(params):
        """Ultranest always passes a batch (shape (B, ndim)); evaluate
        the whole batch through one MPI collective and return a (B,) array.
        """
        arr = np.asarray(params, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        out = coll.evaluate_many(arr)
        have_ecc_jac = model.use_ecc and model.ecc_cartesian
        ix = idx_lookup["e_x"] if have_ecc_jac else None
        iy = idx_lookup["e_y"] if have_ecc_jac else None
        if have_ecc_jac:
            # Mirror the numpyro `factor('ecc_cartesian_jac', ...)`;
            # kept in the likelihood so the prior transform stays a
            # pure CDF-inverse of the Cartesian Uniforms.
            e_x = arr[:, ix]
            e_y = arr[:, iy]
            r2 = e_x * e_x + e_y * e_y
            out = out + (np.log(4.0 / np.pi) - 0.5 * np.log(r2 + 1e-6))
        return out

    # `vectorized=True` tells ultranest to pass a (B, ndim) batch directly
    # (without its internal `vectorize()` row-by-row wrapper, which would
    # stack scalars into an (B, 1) 2-D return and break the shape check).
    # Ultranest natively supports resume via log_dir. Pass resume="resume"
    # to pick up an existing run, or "overwrite" to start fresh.
    # Prevent UltraNest from importing mpi4py in its constructor. This
    # wrapper already uses MPI for the collective likelihood service, with
    # rank 0 inside UltraNest and workers parked in `worker_loop()`. If
    # UltraNest auto-detects MPI, it will issue its own collectives and
    # deadlock immediately because the workers are not inside UltraNest.
    import sys as _sys
    _real_import = _builtins.__import__
    def _guarded_import(name, globals=None, locals=None, fromlist=(),
                        level=0):
        if name == "mpi4py" or name.startswith("mpi4py."):
            raise ImportError(
                "mpi4py hidden from UltraNest in mode2 MPI wrapper")
        return _real_import(name, globals, locals, fromlist, level)
    _mpi4py_backup = _sys.modules.pop("mpi4py", None)
    _mpi4py_mpi_backup = _sys.modules.pop("mpi4py.MPI", None)
    resume_flag = "resume" if resume else "overwrite"
    # Ultranest uses the module-global NumPy RNG internally, so seed it
    # explicitly to make `--seed` effective for NS runs as well.
    _np_rng_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        _builtins.__import__ = _guarded_import
        sampler = ultranest.ReactiveNestedSampler(
            names, loglike, transform, log_dir=str(out_dir),
            resume=resume_flag, vectorized=True)
        _builtins.__import__ = _real_import
        if stepsampler == "region-slice":
            sampler.stepsampler = RegionSliceSampler(
                nsteps=nsteps, check_nsteps="move-distance")
        elif stepsampler == "slice-mixture":
            sampler.stepsampler = SliceSampler(
                nsteps=nsteps,
                generate_direction=generate_mixture_random_direction,
                check_nsteps="move-distance")
        elif stepsampler != "none":
            raise ValueError(f"Unknown NS stepsampler '{stepsampler}'")
        result = sampler.run(
            min_num_live_points=min_live, max_ncalls=max_ncalls,
            viz_callback=False, show_status=True)
    finally:
        _builtins.__import__ = _real_import
        if _mpi4py_backup is not None:
            _sys.modules["mpi4py"] = _mpi4py_backup
        if _mpi4py_mpi_backup is not None:
            _sys.modules["mpi4py.MPI"] = _mpi4py_mpi_backup
        np.random.set_state(_np_rng_state)

    return sampler, result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_de_toml(out_path, galaxy, best_dict, best_logpost):
    lines = [f"# DE MAP (CPU+MPI, mode2) for {galaxy}",
             f"# best log-posterior = {best_logpost:.4f}",
             f"[model.galaxies.{galaxy}.init]"]
    for k in sorted(best_dict):
        v = best_dict[k]
        lines.append(f"{k} = {round(float(v), 4)}")
    Path(out_path).write_text("\n".join(lines) + "\n")


def _print_de_summary(galaxy, best_dict, best_post, runtime):
    fsection(f"DE MAP summary ({galaxy}, {runtime:.1f}s)")
    fprint(f"best log-posterior = {best_post:.4f}")
    for k in sorted(best_dict):
        fprint(f"  {k:20s} = {best_dict[k]:12.4f}")


def _print_ns_summary(galaxy, result, names, runtime):
    fsection(f"NS posterior summary ({galaxy}, {runtime:.1f}s)")
    fprint(f"log Z = {result['logz']:.3f} ± {result['logzerr']:.3f}")
    fprint(f"effective sample size = {result.get('ess', float('nan')):.0f}")
    pts = np.asarray(result["samples"])
    for i, n in enumerate(names):
        col = pts[:, i]
        lo, med, hi = np.quantile(col, [0.16, 0.5, 0.84])
        fprint(f"  {n:20s} = {med:12.4f}  "
               f"[-{med - lo:8.4f} +{hi - med:8.4f}]  (16/84%)")


def _save_ns_h5(out_path, sampler, result, names):
    with H5File(out_path, "w") as f:
        grp = f.create_group("samples")
        pts = np.asarray(result["samples"])
        for i, n in enumerate(names):
            grp.create_dataset(n, data=pts[:, i])
        if "weighted_samples" in result:
            wgrp = f.create_group("weighted_samples")
            wpts = np.asarray(result["weighted_samples"]["points"])
            wlog = np.asarray(result["weighted_samples"]["logw"])
            for i, n in enumerate(names):
                wgrp.create_dataset(n, data=wpts[:, i])
            wgrp.create_dataset("logw", data=wlog)
        f.attrs["log_Z"] = float(result["logz"])
        f.attrs["log_Z_err"] = float(result["logzerr"])
        f.attrs["ess"] = float(result.get("ess", np.nan))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_config = str(Path(__file__).resolve().with_name("config_maser.toml"))
    parser = argparse.ArgumentParser()
    parser.add_argument("galaxy", type=str)
    parser.add_argument("--method", choices=["de", "ns"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    # Grid overrides (use the mode2 default from config if None).
    parser.add_argument("--n-phi-hv-high", type=int, default=None)
    parser.add_argument("--n-phi-hv-low", type=int, default=None)
    parser.add_argument("--n-phi-sys", type=int, default=None)
    parser.add_argument("--n-r-local", type=int, default=None)
    parser.add_argument("--n-r-brute", type=int, default=None)
    # DE options.
    parser.add_argument("--de-popsize", type=int, default=150)
    parser.add_argument("--de-maxiter", type=int, default=2000)
    parser.add_argument("--de-F", type=float, default=0.7)
    parser.add_argument("--de-CR", type=float, default=0.9)
    parser.add_argument("--de-tol", type=float, default=1e-4)
    # NS options.
    parser.add_argument("--ns-min-live", type=int, default=400)
    parser.add_argument("--ns-max-ncalls", type=int, default=None)
    parser.add_argument("--ns-stepsampler",
                        choices=["none", "region-slice", "slice-mixture"],
                        default="none")
    parser.add_argument("--ns-nsteps", type=int, default=None)
    # Misc.
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=default_config)
    parser.add_argument("--max-spots", type=int, default=None,
                        help="Truncate loaded data to the first N spots "
                             "(smoke-testing the MPI pipeline).")
    # Checkpointing.
    parser.add_argument("--resume", action="store_true",
                        help="DE: resume from <out>/*_de_checkpoint.pkl if "
                             "present. NS: resume ultranest log_dir.")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="DE: pickle state every N generations "
                             "(default 25).")
    args = parser.parse_args()

    if args.resume and args.out_dir is None:
        parser.error("--resume requires --out-dir so the exact previous run "
                     "is unambiguous.")
    if args.max_spots is not None and args.max_spots < 1:
        parser.error("--max-spots must be >= 1")
    if args.method == "de":
        if args.de_popsize < 4:
            parser.error("--de-popsize must be >= 4")
        if args.de_maxiter < 1:
            parser.error("--de-maxiter must be >= 1")
        if args.checkpoint_every < 1:
            parser.error("--checkpoint-every must be >= 1")
    else:
        if args.ns_min_live < 1:
            parser.error("--ns-min-live must be >= 1")
        if args.ns_max_ncalls is not None and args.ns_max_ncalls < 1:
            parser.error("--ns-max-ncalls must be >= 1")
        if args.ns_nsteps is not None and args.ns_nsteps < 1:
            parser.error("--ns-nsteps must be >= 1")

    if RANK == 0:
        fsection(f"mode2-MPI runner ({args.galaxy}, method={args.method})")
        fprint(f"MPI size: {SIZE} ranks; jax backend: {jax.default_backend()}")

    # ---- Load master config ----
    with open(args.config_path, "rb") as f:
        master_cfg = tomli.load(f)

    if args.galaxy not in master_cfg["model"]["galaxies"]:
        if RANK == 0:
            print(f"Unknown galaxy '{args.galaxy}'.", flush=True)
        sys.exit(1)

    grid_overrides = {
        "n_phi_hv_high": args.n_phi_hv_high,
        "n_phi_hv_low": args.n_phi_hv_low,
        "n_phi_sys": args.n_phi_sys,
        "n_r_local": args.n_r_local,
        "n_r_brute": args.n_r_brute,
    }
    gcfg_src = master_cfg["model"]["galaxies"][args.galaxy]
    v_sys_obs = gcfg_src["v_sys_obs"]

    # ---- Load full galaxy data (all ranks load it) ----
    full = load_megamaser_spots(
        master_cfg["io"]["maser_data"]["root"], args.galaxy,
        v_sys_obs=v_sys_obs)
    if "D_lo" in gcfg_src and "D_hi" in gcfg_src:
        full["D_lo"] = float(gcfg_src["D_lo"])
        full["D_hi"] = float(gcfg_src["D_hi"])

    if args.max_spots is not None and args.max_spots < full["n_spots"]:
        _keep = int(args.max_spots)
        for k, v in list(full.items()):
            if isinstance(v, np.ndarray) and v.shape[:1] == (full["n_spots"],):
                full[k] = v[:_keep].copy()
            elif isinstance(v, list) and len(v) == full["n_spots"]:
                full[k] = list(v[:_keep])
        full["n_spots"] = _keep
        _log(f"truncated to first {_keep} spots (--max-spots)")

    cfg = _build_runtime_config(
        master_cfg, args.galaxy, grid_overrides, full)
    gcfg = cfg["model"]["galaxies"][args.galaxy]

    n_spots = full["n_spots"]
    my_indices = np.arange(RANK, n_spots, SIZE, dtype=int)
    my_count = int(my_indices.size)
    if RANK == 0:
        per_rank_counts = np.array(
            [len(range(r, n_spots, SIZE)) for r in range(SIZE)],
            dtype=int)
        if SIZE > n_spots:
            idle = int((per_rank_counts == 0).sum())
            fprint(f"spot distribution across {SIZE} ranks: "
                   f"min={per_rank_counts.min()}, "
                   f"mean={per_rank_counts.mean():.2f}, "
                   f"max={per_rank_counts.max()}, "
                   f"idle={idle}, sum={per_rank_counts.sum()} "
                   f"(expected n_spots={n_spots})")
        else:
            fprint(f"spot distribution across {SIZE} ranks: "
                   f"min={per_rank_counts.min()}, "
                   f"mean={per_rank_counts.mean():.2f}, "
                   f"max={per_rank_counts.max()}, "
                   f"sum={per_rank_counts.sum()} "
                   f"(expected n_spots={n_spots})")

    # ---- Per-rank data subset + model ----
    # Suppress MaserDiskModel init output on all ranks (including rank 0) via
    # fd-level redirect — Python-level sys.stdout redirect is insufficient
    # because output may be buffered or written at the C level.
    if my_count > 0:
        my_data = _subset_spots(full, my_indices)
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".toml", delete=False)
        tomli_w.dump(cfg, tmp)
        tmp.close()
        _saved1, _saved2 = os.dup(1), os.dup(2)
        _dn = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_dn, 1); os.dup2(_dn, 2); os.close(_dn)
        model = MaserDiskModel(tmp.name, my_data)
        os.dup2(_saved1, 1); os.dup2(_saved2, 2)
        os.close(_saved1); os.close(_saved2)
        os.unlink(tmp.name)
        # `_resolve_per_galaxy_priors` installs a per-galaxy Uniform under
        # priors["D"] but leaves prior_dist_name["D"] unset; register it so
        # `_dist_bounds`/`prior_transform_factory` look it up uniformly.
        if "D" in model.priors and "D" not in model.prior_dist_name:
            model.prior_dist_name["D"] = "uniform"
    else:
        model = None

    # Print galaxy-level model summary from the full data (not per-rank).
    if RANK == 0:
        is_hv_arr  = np.asarray(full["is_highvel"])
        has_a_arr  = np.isfinite(np.asarray(full["a"])) & (np.asarray(full["a"]) != 0)
        n_sys      = int((~is_hv_arr).sum())
        n_sys_cons = int((~is_hv_arr & has_a_arr).sum())
        n_hv       = int(is_hv_arr.sum())
        fprint(f"galaxy spot summary: {n_spots} total — "
               f"{n_sys} sys ({n_sys_cons} w/accel, {n_sys-n_sys_cons} w/o), "
               f"{n_hv} HV")
        fprint(f"mode: mode2 (marginalise r+φ)  "
               f"n_phi_hv_high={gcfg.get('n_phi_hv_high')}  "
               f"n_phi_sys={gcfg.get('n_phi_sys')}  "
               f"n_r_local={gcfg.get('n_r_local')}  "
               f"n_r_brute={gcfg.get('n_r_brute')}")

    # Build `names` and `init_vec` on rank 0 and broadcast so every rank
    # agrees on the parameter order.
    if RANK == 0:
        names = build_param_spec(model)
        init_cfg = gcfg.get("init", {})
        if not init_cfg:
            print(f"ERROR: {args.galaxy} has no [init] section.", flush=True)
            sys.exit(1)
        init_vec = np.empty(len(names), dtype=np.float64)
        for i, n in enumerate(names):
            if n in init_cfg:
                init_vec[i] = float(init_cfg[n])
            else:
                b = _dist_bounds(model, n)
                init_vec[i] = 0.5 * (b[0] + b[1])
    else:
        names = None
        init_vec = None
    names = COMM.bcast(names, root=0)
    init_vec = COMM.bcast(init_vec, root=0)

    _log(f"sampled parameters ({len(names)}): {', '.join(names)}")

    # ---- Build per-rank log-likelihood ----
    if model is not None:
        per_rank_logL = make_per_rank_logL(model, names)
    else:
        per_rank_logL = lambda _flat: 0.0

    # Warm the JIT cache: each rank compiles its own logL once.
    t0 = time.perf_counter()
    _ = float(per_rank_logL(init_vec))
    my_jit_time = time.perf_counter() - t0
    all_jit = COMM.gather(my_jit_time, root=0)
    COMM.Barrier()
    if RANK == 0:
        fprint(f"JIT warm-up: min={min(all_jit):.1f}s  "
               f"max={max(all_jit):.1f}s  mean={sum(all_jit)/len(all_jit):.1f}s")

    coll = CollectiveLikelihood(per_rank_logL, len(names))
    _benchmark_likelihood(
        coll, per_rank_logL, init_vec, has_spots=(model is not None))

    # ---- Dispatch ----
    if args.out_dir:
        out_root = os.path.abspath(args.out_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_root = os.path.abspath(os.path.join(
            cfg["io"]["root_output"],
            f"{args.galaxy}_mode2_mpi_{args.method}_{stamp}_{os.getpid()}"))
    file_tag = f"{args.galaxy}_mode2_mpi"
    Path(out_root).mkdir(parents=True, exist_ok=True)

    if RANK == 0:
        bounds = [_dist_bounds(model, n) for n in names]
        ckpt_meta = _de_checkpoint_meta(
            galaxy=args.galaxy, names=names, bounds=bounds, model=model,
            gcfg=gcfg, n_spots=n_spots, mpi_size=SIZE)
        stop_sent = False
        fprint(f"output dir: {out_root}")

        try:
            if args.method == "de":
                fsection("Running differential evolution")
                ckpt_path = os.path.join(
                    out_root, f"{file_tag}_de_checkpoint.pkl")
                t_start = time.perf_counter()
                best_dict, best_post = run_de(
                    coll, model, names, bounds,
                    popsize=args.de_popsize, maxiter=args.de_maxiter,
                    F=args.de_F, CR=args.de_CR, tol=args.de_tol,
                    seed=args.seed,
                    checkpoint_path=ckpt_path,
                    checkpoint_every=args.checkpoint_every,
                    resume=args.resume,
                    checkpoint_meta=ckpt_meta)
                dt = time.perf_counter() - t_start
                coll.stop()
                stop_sent = True
                out_toml = os.path.join(out_root, f"{file_tag}_de_map.toml")
                _save_de_toml(out_toml, args.galaxy, best_dict, best_post)
                _print_de_summary(args.galaxy, best_dict, best_post, dt)
                fprint(f"Saved best-fit TOML to {out_toml}")
                fprint(f"Checkpoint at {ckpt_path}")

            else:  # NS
                fsection("Running ultranest")
                if args.ns_stepsampler == "none":
                    fprint("NS proposal: default MLFriends region sampler "
                           "(no stepsampler)")
                else:
                    ns_nsteps = len(names) if args.ns_nsteps is None else args.ns_nsteps
                    if args.ns_stepsampler == "region-slice":
                        label = "RegionSliceSampler"
                    else:
                        label = "SliceSampler(mixture directions)"
                    fprint(f"NS proposal: {label}(nsteps={ns_nsteps})")
                ns_dir = os.path.join(out_root, f"{file_tag}_ultranest")
                t_start = time.perf_counter()
                sampler, result = run_ns(
                    coll, names, model, ns_dir,
                    min_live=args.ns_min_live, max_ncalls=args.ns_max_ncalls,
                    seed=args.seed, resume=args.resume,
                    stepsampler=args.ns_stepsampler,
                    step_nsteps=args.ns_nsteps)
                dt = time.perf_counter() - t_start
                coll.stop()
                stop_sent = True
                out_h5 = os.path.join(out_root, f"{file_tag}_ns.hdf5")
                _save_ns_h5(out_h5, sampler, result, names)
                _print_ns_summary(args.galaxy, result, names, dt)
                fprint(f"Saved NS samples to {out_h5}")
        finally:
            if not stop_sent:
                try:
                    coll.stop()
                except Exception:
                    pass

    else:
        coll.worker_loop()


if __name__ == "__main__":
    main()
