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
"""Generic maser disk inference script (NUTS or NSS nested sampling).

Usage:
    python run_maser_disk.py <galaxy_name> [--sampler nuts|nss] [options]

All settings (priors, sampler config) are read from config_maser.toml.
CLI args override config values where provided.
"""
import hashlib
import json
import os
import sys

import tomli

with open(os.path.join(os.path.dirname(__file__), "../../local_config.toml"), "rb") as f:
    _lcfg = tomli.load(f)
ld = os.environ.get("LD_LIBRARY_PATH", "")
needed = [p for p in _lcfg.get("gpu_ld_library_path", []) if p not in ld]
if needed:
    os.environ["LD_LIBRARY_PATH"] = ":".join(needed) + (f":{ld}" if ld else "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Disable JAX's 75% GPU pre-allocation so memory is allocated on demand.
# Without this, the BFC allocator exhausts the pre-allocated pool on
# galaxies with many spots (e.g. UGC3789: 156 spots).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import tempfile
import time
from importlib import metadata as importlib_metadata

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_maser.toml")
with open(_CONFIG_PATH, "rb") as f:
    master_cfg = tomli.load(f)


def _cli_value(flag):
    prefix = f"{flag}="
    for i, arg in enumerate(sys.argv):
        if arg.startswith(prefix):
            return arg[len(prefix):]
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _cli_galaxy():
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--sampler", "--seed", "--num-warmup", "--num-samples",
                   "--num-chains", "--n-live", "--num-mcmc-steps",
                   "--num-delete", "--termination", "--devices", "--mode",
                   "--f-grid", "--init-method", "--r-ang-init",
                   "--checkpoint-interval-minutes"}:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        return arg
    return None


def _effective_run_from_argv():
    """Infer the early sampler/mode choice before importing JAX."""
    galaxy = _cli_galaxy()
    sampler = _cli_value("--sampler") or master_cfg["inference"].get(
        "sampler", "nss")
    cli_mode = _cli_value("--mode")
    model_cfg = master_cfg["model"]
    gal_cfg = model_cfg.get("galaxies", {}).get(galaxy, {})
    mode = cli_mode or gal_cfg.get("mode") or model_cfg.get("mode", "mode2")
    return galaxy, sampler, mode


# Float64 must be enabled before importing JAX.
_galaxy_early, _sampler_early, _mode_early = _effective_run_from_argv()
_force_f64_mode1_nuts = (
    _sampler_early == "nuts" and _mode_early == "mode1")
_enable_f64 = "--f64" in sys.argv or _force_f64_mode1_nuts
if _enable_f64:
    if "--f64" in sys.argv:
        sys.argv.remove("--f64")
    import jax
    jax.config.update("jax_enable_x64", True)
    if _force_f64_mode1_nuts:
        print("float64 enabled (required for mode1 NUTS)", flush=True)
    else:
        print("float64 enabled (--f64)", flush=True)

import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import tomli_w
from h5py import File as H5File
from jax import random
from numpyro.infer import NUTS
from numpyro.infer.initialization import init_to_value

from candel.inference.checkpointed_nuts import run_checkpointed_nuts
from candel.inference.inference import print_clean_summary, _setup_dense_mass
from candel.inference.nested import print_nested_summary, run_nss
from candel.model.model_H0_maser import (JointMaserModel, MaserDiskModel,
                                          radius_from_los_acceleration,
                                          remap_warp_to_r0)
from candel.plotting.corner import plot_corner, sort_params
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import data_path, fprint, fsection, get_nested, results_path

_devs = jax.devices()
_dev_names = ", ".join(d.device_kind for d in _devs)
_precision = "float64" if jax.config.jax_enable_x64 else "float32"
print(f"JAX platform: {jax.default_backend()}, devices: {_devs} "
      f"({_dev_names}), precision: {_precision}", flush=True)

inf_cfg = master_cfg["inference"]


def _json_ready(value):
    """Convert arrays and mappings to JSON-stable Python containers."""
    if isinstance(value, dict):
        return {
            str(k): _json_ready(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stable_hash(value):
    """Return a deterministic SHA256 hash for JSON-compatible metadata."""
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path):
    """Return the SHA256 hash of a file, or ``None`` if unreadable."""
    if path is None:
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _package_version(package):
    """Return an installed package version, or ``None`` if absent."""
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _dense_mass_metadata(dense_mass):
    """Normalise NumPyro dense-mass settings for checkpoint metadata."""
    if isinstance(dense_mass, bool):
        return dense_mass
    return _json_ready(dense_mass)


def _combine_chain_samples(chain_samples):
    """Concatenate sequential NUTS chain outputs along the sample axis."""
    keys = chain_samples[0].keys()
    return {
        key: np.concatenate(
            [np.asarray(samples[key]) for samples in chain_samples], axis=0)
        for key in keys
    }


# >>> f-grid helpers
_FGRID_KEYS = (
    "n_phi_hv_high",
    "n_phi_hv_low",
    "n_phi_sys",
    "n_phi_hv_high_mode1",
    "n_phi_hv_low_mode1",
    "n_phi_sys_mode1",
    "n_r_local",
    "n_r_global",
    "n_r_scan",
)


def _round_to_odd(n, f):
    """Scale n by f, round to the nearest odd int, floor at 3."""
    m = max(3, int(round(n * f)))
    if m % 2 == 0:
        m += 1
    return m


def _apply_fgrid(cfg, f):
    """In-place scale of grid-size keys in cfg['model'] and every
    cfg['model']['galaxies'][*] sub-dict. Only keys in _FGRID_KEYS are
    touched; every other value is left alone."""
    m = cfg["model"]
    for k in _FGRID_KEYS:
        if k in m:
            m[k] = _round_to_odd(int(m[k]), f)
    for g in m.get("galaxies", {}).values():
        if not isinstance(g, dict):
            continue
        for k in _FGRID_KEYS:
            if k in g:
                g[k] = _round_to_odd(int(g[k]), f)
# <<< f-grid helpers


def _mode1_r_ang_data_init(model, init_params, seed):
    """Current data-driven stochastic Mode 1 r_ang initialisation."""
    rng = np.random.default_rng(seed)
    from scipy.stats import truncnorm as _truncnorm

    D_c = float(init_params["D_c"])
    H0_ref = float(get_nested(model.config, "model/H0_ref", 73.0))
    z_cosmo = float(model.distance2redshift(
        jnp.atleast_1d(jnp.asarray(D_c)), h=H0_ref / 100.0).squeeze())
    D_A = D_c / (1.0 + z_cosmo)
    _lo, _hi = model.r_ang_range(D_A)
    r_lo, r_hi = float(_lo), float(_hi)

    eta = float(init_params["eta"])
    x0 = float(init_params["x0"])   # μas
    y0 = float(init_params["y0"])   # μas
    i0_rad = np.deg2rad(float(init_params["i0"]))
    sin_i = abs(np.sin(i0_rad))
    M_BH = 10.0 ** (eta + np.log10(D_A) - 7.0)  # 1e7 Msun

    is_hv = np.asarray(model.is_highvel, dtype=bool)
    has_accel = np.asarray(model._all_has_accel, dtype=bool)

    # HV: group mean/sd from projected sky radii.
    x_hv = np.asarray(model._all_x)[is_hv]  # μas
    y_hv = np.asarray(model._all_y)[is_hv]  # μas
    r_hv = np.clip(np.sqrt((x_hv - x0)**2 + (y_hv - y0)**2) / 1e3,
                   r_lo * 1.01, r_hi * 0.99)
    mu_hv = float(np.mean(r_hv))
    sd_hv = max(float(np.std(r_hv)), 0.1 * mu_hv)

    # Systemic: group mean/sd from spots with accel measurements.
    sys_accel = (~is_hv) & has_accel
    if sys_accel.any():
        a_sys = np.abs(np.asarray(model._all_a)[sys_accel])
        r_sys = np.asarray(radius_from_los_acceleration(
            a_sys + 1e-30, sin_i, D_A, M_BH))
        r_sys = np.clip(r_sys, r_lo * 1.01, r_hi * 0.99)
        mu_sys = float(np.mean(r_sys))
        sd_sys = max(float(np.std(r_sys)), 0.1 * mu_sys)
    else:
        mu_sys, sd_sys = mu_hv, sd_hv
        fprint("  r_ang: no systemic accel, "
               "using HV distribution for systemic spots")

    r_ang = np.empty(model.n_spots)
    for mask, mu, sd, label in [
        (is_hv, mu_hv, sd_hv, "HV"),
        (~is_hv, mu_sys, sd_sys, "sys"),
    ]:
        n = int(mask.sum())
        if n == 0:
            continue
        a = (r_lo - mu) / sd
        b = (r_hi - mu) / sd
        r_ang[mask] = _truncnorm.rvs(
            a, b, loc=mu, scale=sd, size=n, random_state=rng)
        fprint(f"  r_ang {label}: n={n}, mu={mu:.3f}, "
               f"sd={sd:.3f} mas")

    fprint(f"  r_ang: data-driven init "
           f"[{r_ang.min():.3f}, {r_ang.max():.3f}] mas "
           f"| D_c={D_c:.1f} Mpc")
    return jnp.asarray(r_ang)


def _mode1_r_ang_peak_init(model, init_params):
    """Initialise Mode 1 r_ang at conditional phi-marginal peaks."""
    r_ang, diag = model.mode1_r_ang_conditional_peak(init_params)
    r_np = np.asarray(r_ang)
    fprint(f"  r_ang: conditional-peak init "
           f"[{r_np.min():.3f}, {r_np.max():.3f}] mas "
           f"| D_A={diag['D_A']:.1f} Mpc")
    return r_ang


def _add_mode1_r_ang_init(model, init_params, seed, method):
    """Attach r_ang to init_params when Mode 1 samples it."""
    if model.mode != "mode1" or "r_ang" in init_params:
        return
    if method == "data":
        init_params["r_ang"] = _mode1_r_ang_data_init(
            model, init_params, seed)
    elif method == "peak":
        init_params["r_ang"] = _mode1_r_ang_peak_init(
            model, init_params)
    else:
        raise ValueError(f"Unknown r_ang init method: {method!r}")

# ---- Parse args (CLI overrides config) ----
parser = argparse.ArgumentParser()
parser.add_argument("galaxy", type=str)
parser.add_argument("--sampler", type=str, default=None,
                    choices=["nuts", "nss"])
parser.add_argument("--seed", type=int, default=None)
# NUTS
parser.add_argument("--num-warmup", type=int, default=None,
                    help="Number of NUTS warmup steps (default: from config)")
parser.add_argument("--num-samples", type=int, default=None,
                    help="Number of NUTS samples per chain (default: from config)")
parser.add_argument("--num-chains", type=int, default=None,
                    help="Number of NUTS chains (default: from config); "
                         "chains run sequentially")
# NSS
parser.add_argument("--n-live", type=int, default=None)
parser.add_argument("--num-mcmc-steps", type=int, default=None)
parser.add_argument("--num-delete", type=int, default=None)
parser.add_argument("--termination", type=float, default=None)
parser.add_argument("--max-steps", type=int, default=None,
                    help="Maximum HRSS stepping-out steps per side "
                         "(default: from config or 10)")
parser.add_argument("--max-shrinkage", type=int, default=None,
                    help="Maximum HRSS shrinkage proposals per transition "
                         "(default: from config or 100)")
parser.add_argument("--stale-retries", type=int, default=None,
                    help="Extra full replacement-chain attempts when a chain "
                         "has no accepted HRSS transition "
                         "(default: from config or 1)")
parser.add_argument("--cov-jitter", type=float, default=None,
                    help="Relative eigenvalue floor for live-point covariance "
                         "regularisation; use 0 for the old behaviour "
                         "(default: from config or 1e-6)")
parser.add_argument("--devices", type=str, default=None,
                    help="Local devices used by sampler parallel work: "
                         "auto, 1, or N. auto uses multiple non-CPU devices "
                         "only when they are visible.")
parser.add_argument("--mode", type=str, default=None,
                    choices=["mode1", "mode2"],
                    help="Sampling mode: mode1 samples r and marginalises "
                         "phi, mode2 marginalises both (default: from config)")
parser.add_argument("--f-grid", type=float, default=1.0,
                    help="Scale every integer phi/r grid size by this "
                         "factor; results are rounded to the nearest "
                         "odd int (min 3). Applies to global [model] "
                         "keys and per-galaxy overrides.")
parser.add_argument("--init-method", type=str, default=None,
                    choices=["config", "median", "sample"],
                    help="NUTS initialisation method (default: config)  "
                         "config:  globals from config, r_ang data-driven from sky positions / accelerations  "
                         "median:  median of N prior draws for globals, r_ang data-driven  "
                         "sample:  globals sampled from priors, r_ang data-driven")
parser.add_argument("--r-ang-init", type=str, default=None,
                    choices=["data", "peak"],
                    help="Mode 1 r_ang initialisation: data draws from the "
                         "existing rough data-driven distribution; peak "
                         "uses the conditional phi-marginal peak from the "
                         "Mode 2 centring optimiser.")
parser.add_argument("--no-ecc", action="store_true",
                    help="Disable eccentricity model (override config)")
parser.add_argument("--add-ecc", "-add-ecc", action="store_true",
                    help="Enable eccentricity model (override config)")
parser.add_argument("--no-quadratic-warp", action="store_true",
                    help="Disable quadratic warp (override config)")
parser.add_argument("--add-quadratic", "--add-quadratic-warp",
                    "-add-quadratic",
                    dest="add_quadratic_warp", action="store_true",
                    help="Enable quadratic warp (override config)")
parser.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint "
                         "(NUTS warmup/sampling, NSS)")
parser.add_argument("--checkpoint-interval-minutes", type=float, default=15.0,
                    help="Checkpoint interval in minutes "
                         "(default: 15). Applies to warmup and sampling.")
parser.add_argument("--f64", action="store_true", default=_enable_f64,
                    help="Enable JAX float64. Automatically enabled for "
                         "mode1 NUTS.")
args = parser.parse_args()
if args.no_ecc and args.add_ecc:
    parser.error("--no-ecc and --add-ecc are mutually exclusive")
if args.no_quadratic_warp and args.add_quadratic_warp:
    parser.error("--no-quadratic-warp and --add-quadratic are mutually "
                 "exclusive")

galaxy = args.galaxy
sampler = args.sampler or inf_cfg.get("sampler", "nss")
seed = args.seed or inf_cfg.get("seed", 42)

# Build descriptive output tag
_mcfg = master_cfg["model"]

_tags = []

# Distance prior
_dc_prior = _mcfg.get("D_c_prior", "uniform")
_tags.append("Dvol" if _dc_prior == "volume" else "Dflat")

# Mode tag: always included so filenames are unambiguous.
# Resolve in priority order: CLI → per-galaxy config → global config.
_gal_mode = _mcfg.get("galaxies", {}).get(args.galaxy, {}).get("mode", None)
_mode_tag = args.mode or _gal_mode or _mcfg.get("mode", "mode2")
_tags.append(_mode_tag)

if args.f_grid != 1.0:
    _tags.append(f"fg{args.f_grid:g}")

# Model feature overrides
if args.no_ecc:
    _tags.append("noecc")
if args.add_ecc:
    _tags.append("ecc")
if args.no_quadratic_warp:
    _tags.append("noqw")
if args.add_quadratic_warp:
    _tags.append("qw")

dist_tag = "_".join(_tags)

# ---- Validate galaxy ----
galaxies = master_cfg["model"]["galaxies"]
is_joint = (galaxy == "joint")

if not is_joint and galaxy not in galaxies:
    print(f"Unknown galaxy '{galaxy}'. "
          f"Available: {list(galaxies.keys())} + ['joint']", flush=True)
    sys.exit(1)

# ---- Load data ----
if is_joint:
    # Exclude NGC4258 (geometric parallax, not in the MCP sample)
    galaxy_names = [g for g in galaxies if g != "NGC4258"]
    fsection(f"Loading joint data ({len(galaxy_names)} galaxies)")
    data_list = []
    for gname in galaxy_names:
        gcfg_g = galaxies[gname]
        d = load_megamaser_spots(data_path("data", "Megamaser"), gname,
                                 v_sys_obs=gcfg_g["v_sys_obs"])
        if "D_lo" in gcfg_g and "D_hi" in gcfg_g:
            d["D_lo"] = float(gcfg_g["D_lo"])
            d["D_hi"] = float(gcfg_g["D_hi"])
        data_list.append(d)
    n_spots = sum(d["n_spots"] for d in data_list)
else:
    gcfg = galaxies[galaxy]
    v_sys_obs = gcfg["v_sys_obs"]
    fsection(f"Loading {galaxy} data")
    data = load_megamaser_spots(data_path("data", "Megamaser"), galaxy, v_sys_obs=v_sys_obs)
    if "D_lo" in gcfg and "D_hi" in gcfg:
        data["D_lo"] = float(gcfg["D_lo"])
        data["D_hi"] = float(gcfg["D_hi"])

# ---- Build model config from master config ----
dense_mass_blocks = inf_cfg.get("dense_mass_blocks", [
    ["D_c", "eta", "dv_sys"],
    ["i0", "di_dr", "Omega0", "dOmega_dr"],
    ["x0", "y0"],
])

config = {
    "inference": {
        "num_warmup": args.num_warmup or inf_cfg.get("num_warmup", 2500),
        "num_samples": args.num_samples or inf_cfg.get("num_samples", 2000),
        "num_chains": args.num_chains or inf_cfg.get("num_chains", 1),
        "chain_method": "sequential",
        "seed": seed,
        "dense_mass_blocks": dense_mass_blocks,
        "max_tree_depth": inf_cfg.get("max_tree_depth", 10),
    },
    "model": master_cfg["model"],
    "io": master_cfg["io"],
}
if args.mode is not None:
    config["model"]["mode"] = args.mode
if (args.no_ecc or args.add_ecc
        or args.no_quadratic_warp or args.add_quadratic_warp):
    gals = config["model"].setdefault("galaxies", {})
    gcfg_ov = gals.setdefault(galaxy if not is_joint else "", {})
    if args.no_ecc:
        gcfg_ov["use_ecc"] = False
        fprint("CLI override: use_ecc = False")
    if args.add_ecc:
        gcfg_ov["use_ecc"] = True
        fprint("CLI override: use_ecc = True")
    if args.no_quadratic_warp:
        gcfg_ov["use_quadratic_warp"] = False
        fprint("CLI override: use_quadratic_warp = False")
    if args.add_quadratic_warp:
        gcfg_ov["use_quadratic_warp"] = True
        fprint("CLI override: use_quadratic_warp = True")

# Joint: apply prior overrides from [joint.priors] config section
if is_joint:
    joint_priors = master_cfg.get("joint", {}).get("priors", {})
    for pname, pval in joint_priors.items():
        config["model"]["priors"][pname] = pval
        fprint(f"joint override: {pname} -> {pval}")

if args.f_grid != 1.0:
    _apply_fgrid(config, args.f_grid)
    m = config["model"]
    fprint(f"f-grid={args.f_grid:g}: "
           f"n_phi_hv_high={m.get('n_phi_hv_high')}, "
           f"n_phi_hv_low={m.get('n_phi_hv_low')}, "
           f"n_phi_sys={m.get('n_phi_sys')}, "
           f"n_phi_hv_high_mode1={m.get('n_phi_hv_high_mode1')}, "
           f"n_phi_hv_low_mode1={m.get('n_phi_hv_low_mode1')}, "
           f"n_phi_sys_mode1={m.get('n_phi_sys_mode1')}, "
           f"n_r_local={m.get('n_r_local')}, "
           f"n_r_global={m.get('n_r_global')}, "
           f"n_r_scan={m.get('n_r_scan')}")

if not is_joint:
    _model_cfg = config["model"]
    _gal_cfg = _model_cfg.get("galaxies", {}).get(galaxy, {})
    _mode = args.mode or _gal_cfg.get("mode") or _model_cfg.get("mode", "mode2")
    _spot_batch = _gal_cfg.get(
        "mode2_spot_batch", _model_cfg.get("mode2_spot_batch"))
    if _mode == "mode2":
        if _spot_batch is None:
            print("ERROR: mode2_spot_batch must be set explicitly in "
                  "config_maser.toml for mode2 runs. Recommended value: 16.",
                  flush=True)
            sys.exit(1)
        if int(_spot_batch) <= 0:
            print("ERROR: mode2_spot_batch must be a positive integer.",
                  flush=True)
            sys.exit(1)

tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
tomli_w.dump(config, tmp)
tmp.close()
if is_joint:
    model = JointMaserModel(tmp.name, data_list)
else:
    model = MaserDiskModel(tmp.name, data)
os.unlink(tmp.name)

# ---- Run sampler ----
if sampler == "nss" and model.mode != "mode2":
    print("ERROR: nested sampling requires mode2 "
          f"(current mode: {model.mode})", flush=True)
    sys.exit(1)

if sampler == "nss" and is_joint:
    print("ERROR: nested sampling is impractical for the joint model "
          "(~80 params). Use --sampler nuts.", flush=True)
    sys.exit(1)

if not is_joint:
    n_spots = data["n_spots"]

if sampler == "nuts":
    num_warmup = args.num_warmup or inf_cfg.get("num_warmup", 2500)
    num_samples = args.num_samples or inf_cfg.get("num_samples", 2000)
    num_chains = args.num_chains or inf_cfg.get("num_chains", 1)
    if num_chains < 1:
        raise ValueError("num_chains must be positive.")
    if model.mode == "mode1" and not jax.config.jax_enable_x64:
        raise RuntimeError("mode1 NUTS megamaser sampling requires float64.")

    if is_joint:
        fsection(f"Running NUTS (joint, {n_spots} spots)")
        # Build init from per-galaxy config init sections
        init_params = {}
        for gname in galaxy_names:
            ginit = galaxies[gname].get("init", {})
            for k, v in ginit.items():
                init_params[f"{gname}/{k}"] = jnp.asarray(v)
        if init_params:
            init_strategy = init_to_value(values=init_params)
            fprint(f"NUTS init from config ({len(init_params)} params)")
        else:
            raise ValueError(
                "Joint NUTS requires per-galaxy [init] sections in config. "
                f"None found for: {galaxy_names}")
    else:
        fsection(f"Running NUTS ({galaxy}, {n_spots} spots)")
        init_cfg = gcfg.get("init", {})
        init_method = args.init_method or inf_cfg.get("init_method", "config")
        r_ang_init_method = args.r_ang_init or inf_cfg.get(
            "r_ang_init_method", "data")

        _init_desc = {
            "config": "globals from config",
            "median": "median of N prior draws for globals",
            "sample": "globals from prior",
        }
        fprint(f"Initialisation: {init_method} "
               f"({_init_desc.get(init_method, '?')})")
        if model.mode == "mode1":
            fprint(f"  r_ang init: {r_ang_init_method}")

        if init_method == "config":
            if not init_cfg:
                print("ERROR: no [init] section for", galaxy, flush=True)
                sys.exit(1)
            init_params = {k: jnp.asarray(v) for k, v in init_cfg.items()}

            # Remove params that aren't sampled.
            if not is_joint:
                init_params.pop("H0", None)
                init_params.pop("sigma_pec", None)
            if not model.use_ecc:
                for _k in ("e_x", "e_y", "ecc", "periapsis",
                           "dperiapsis_dr"):
                    init_params.pop(_k, None)
            if not model.use_quadratic_warp:
                for _k in ("d2i_dr2", "d2Omega_dr2"):
                    init_params.pop(_k, None)

            _add_mode1_r_ang_init(
                model, init_params, seed, r_ang_init_method)

            init_strategy = init_to_value(values=init_params)
            for k, v in sorted(init_params.items()):
                v = jnp.asarray(v)
                if v.ndim == 0:
                    fprint(f"  {k:20s} = {float(v):12.4f}")
                else:
                    fprint(f"  {k:20s} = [{len(v)} values]")
        elif init_method == "sample":
            # Sample globals from priors, then r_ang conditional on sampled D_c.
            rng_key = random.PRNGKey(seed)
            init_params = {}
            _global_priors = [
                ("D", "D_c"), ("eta", "eta"), ("x0", "x0"), ("y0", "y0"),
                ("i0", "i0"), ("Omega0", "Omega0"),
                ("dOmega_dr", "dOmega_dr"), ("di_dr", "di_dr"),
                ("sigma_x_floor", "sigma_x_floor"),
                ("sigma_y_floor", "sigma_y_floor"),
                ("sigma_v_sys", "sigma_v_sys"), ("sigma_v_hv", "sigma_v_hv"),
                ("sigma_a_floor", "sigma_a_floor"), ("dv_sys", "dv_sys"),
            ]
            for prior_key, site_name in _global_priors:
                rng_key, subkey = random.split(rng_key)
                init_params[site_name] = model.priors[prior_key].sample(subkey)
            if model.use_ecc:
                if model.ecc_cartesian:
                    for k in ("e_x", "e_y"):
                        rng_key, subkey = random.split(rng_key)
                        init_params[k] = model.priors[k].sample(subkey)
                else:
                    rng_key, subkey = random.split(rng_key)
                    init_params["ecc"] = model.priors["ecc"].sample(subkey)
                    init_params["periapsis_rad"] = jnp.array(0.0)
                rng_key, subkey = random.split(rng_key)
                init_params["dperiapsis_dr"] = model.priors["dperiapsis_dr"].sample(subkey)
            if model.use_quadratic_warp:
                for k in ("d2i_dr2", "d2Omega_dr2"):
                    rng_key, subkey = random.split(rng_key)
                    init_params[k] = model.priors[k].sample(subkey)
            _add_mode1_r_ang_init(
                model, init_params, seed, r_ang_init_method)
            init_strategy = init_to_value(values=init_params)
            for k, v in sorted(init_params.items()):
                v = jnp.asarray(v)
                if v.ndim == 0:
                    fprint(f"  {k:20s} = {float(v):12.4f}")
                else:
                    fprint(f"  {k:20s} = [{len(v)} values]")
        elif init_method == "median":
            N = int(inf_cfg.get("median_num_samples", 100))
            rng_key = random.PRNGKey(seed)
            init_params = {}
            _global_priors = [
                ("D", "D_c"), ("eta", "eta"), ("x0", "x0"), ("y0", "y0"),
                ("i0", "i0"), ("Omega0", "Omega0"),
                ("dOmega_dr", "dOmega_dr"), ("di_dr", "di_dr"),
                ("sigma_x_floor", "sigma_x_floor"),
                ("sigma_y_floor", "sigma_y_floor"),
                ("sigma_v_sys", "sigma_v_sys"), ("sigma_v_hv", "sigma_v_hv"),
                ("sigma_a_floor", "sigma_a_floor"), ("dv_sys", "dv_sys"),
            ]
            for prior_key, site_name in _global_priors:
                rng_key, subkey = random.split(rng_key)
                draws = model.priors[prior_key].sample(subkey, sample_shape=(N,))
                init_params[site_name] = jnp.median(draws, axis=0)
            if model.use_ecc:
                if model.ecc_cartesian:
                    for k in ("e_x", "e_y"):
                        rng_key, subkey = random.split(rng_key)
                        draws = model.priors[k].sample(subkey, sample_shape=(N,))
                        init_params[k] = jnp.median(draws, axis=0)
                else:
                    rng_key, subkey = random.split(rng_key)
                    draws = model.priors["ecc"].sample(subkey, sample_shape=(N,))
                    init_params["ecc"] = jnp.median(draws, axis=0)
                    init_params["periapsis_rad"] = jnp.array(0.0)
                rng_key, subkey = random.split(rng_key)
                draws = model.priors["dperiapsis_dr"].sample(subkey, sample_shape=(N,))
                init_params["dperiapsis_dr"] = jnp.median(draws, axis=0)
            if model.use_quadratic_warp:
                for k in ("d2i_dr2", "d2Omega_dr2"):
                    rng_key, subkey = random.split(rng_key)
                    draws = model.priors[k].sample(subkey, sample_shape=(N,))
                    init_params[k] = jnp.median(draws, axis=0)
            _add_mode1_r_ang_init(
                model, init_params, seed, r_ang_init_method)
            init_strategy = init_to_value(values=init_params)
            for k, v in sorted(init_params.items()):
                v = jnp.asarray(v)
                if v.ndim == 0:
                    fprint(f"  {k:20s} = {float(v):12.4f}")
                else:
                    fprint(f"  {k:20s} = [{len(v)} values]")
        else:
            raise ValueError(f"Unknown init_method: {init_method!r}")
    # Dense mass matrix: joint uses full dense; single-galaxy uses
    # per-galaxy dense_mass_globals flag for a single globals block.
    t0 = time.time()
    if is_joint:
        _dense_mass = True
        fprint("Dense mass: full dense (joint mode)")
    elif not is_joint and gcfg.get("dense_mass", True) is False:
        _dense_mass = False
        fprint("Dense mass: diagonal (per-galaxy override)")
    elif not is_joint and (
            inf_cfg.get("dense_mass_single_block", False)
            or gcfg.get("dense_mass_globals", False)):
        # Collect all scalar (non-r_ang) param names from init config,
        # excluding params that aren't sampled in single-galaxy mode
        # or that are disabled by CLI overrides.
        _skip = {"r_ang", "H0", "sigma_pec"}
        if not model.use_ecc:
            _skip |= {"e_x", "e_y", "ecc", "periapsis", "dperiapsis_dr"}
        if not model.use_quadratic_warp:
            _skip |= {"d2i_dr2", "d2Omega_dr2"}
        _global_names = [k for k in init_cfg if k not in _skip]
        _block_names = list(_global_names)
        if (model.mode == "mode1"
                and (inf_cfg.get("dense_mass_include_r_ang", False)
                     or gcfg.get("dense_mass_include_r_ang", False))):
            _block_names.append("r_ang")
        _dense_mass = [tuple(_block_names)]
        fprint(f"Dense mass: 1 block with {len(_block_names)} site(s): "
               f"{_block_names}")
        if "r_ang" in _block_names:
            fprint(f"  includes all {model.n_spots} r_ang components")
    else:
        _dense_mass = _setup_dense_mass(
            config["inference"], None, model, model_kwargs={})
        if isinstance(_dense_mass, list):
            fprint(f"Dense mass: {len(_dense_mass)} blocks")
            for i, block in enumerate(_dense_mass):
                fprint(f"  block {i}: {list(block)}")
        elif isinstance(_dense_mass, bool):
            fprint(f"Dense mass: {'full dense' if _dense_mass else 'diagonal'}")

    if is_joint:
        _tap = float(inf_cfg.get("target_accept_prob", 0.9))
        _init_step_size = float(inf_cfg.get("init_step_size", 1.0))
    else:
        _tap = float(gcfg.get("target_accept_prob",
                              inf_cfg.get("target_accept_prob", 0.9)))
        _init_step_size = float(gcfg.get(
            "init_step_size", inf_cfg.get("init_step_size", 1.0)))
    fprint(f"NUTS: target_accept_prob={_tap}, init_step_size={_init_step_size}")
    nuts_kernel = NUTS(model,
                       max_tree_depth=inf_cfg.get("max_tree_depth", 10),
                       target_accept_prob=_tap,
                       step_size=_init_step_size,
                       dense_mass=_dense_mass,
                       init_strategy=init_strategy)

    _nuts_ckpt_dir = results_path(
        master_cfg["io"].get("root_output", "results/Maser"),
        "nuts_checkpoints", galaxy if not is_joint else "joint")
    os.makedirs(_nuts_ckpt_dir, exist_ok=True)
    _nuts_ckpt_base = os.path.join(_nuts_ckpt_dir, f"nuts_ckpt_{dist_tag}")
    fprint(f"Checkpoints: {_nuts_ckpt_dir}")
    fprint(f"Checkpoint file pattern: {_nuts_ckpt_base}_chain<N>.pkl")
    _checkpoint_interval_seconds = args.checkpoint_interval_minutes * 60.0
    fprint(f"Checkpoint interval: {args.checkpoint_interval_minutes:g} "
           "minutes (warmup and sampling)")
    fprint(f"NUTS chains: {num_chains} sequential chain(s), each with an "
           "independent checkpoint file")

    _chain_keys = ([random.PRNGKey(seed)] if num_chains == 1
                   else list(random.split(random.PRNGKey(seed), num_chains)))
    _chain_samples = []
    _chain_extra_fields = []
    _nuts_settings = {
        "max_tree_depth": int(inf_cfg.get("max_tree_depth", 10)),
        "target_accept_prob": _tap,
        "init_step_size": _init_step_size,
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "num_chains": int(num_chains),
        "chain_method": "sequential",
    }
    _init_metadata = {
        "method": "joint_config" if is_joint else init_method,
        "r_ang_method": None if is_joint else r_ang_init_method,
        "init_params_hash": _stable_hash(init_params),
    }
    _base_checkpoint_metadata = {
        "galaxy": galaxy,
        "mode": model.mode,
        "dist_tag": dist_tag,
        "sampler": "nuts",
        "seed": int(seed),
        "nuts_settings": _nuts_settings,
        "dense_mass": _dense_mass_metadata(_dense_mass),
        "dense_mass_hash": _stable_hash(_dense_mass_metadata(_dense_mass)),
        "config_hash": _stable_hash(config),
        "model_config_hash": _stable_hash(config["model"]),
        "inference_config_hash": _stable_hash(config["inference"]),
        "data_hash": _stable_hash(data_list if is_joint else data),
        "init": _init_metadata,
        "model_class": type(model).__name__,
        "use_ecc": bool(getattr(model, "use_ecc", False)),
        "use_quadratic_warp": bool(
            getattr(model, "use_quadratic_warp", False)),
        "n_spots": int(n_spots),
        "package_versions": {
            "candel": _package_version("candel"),
            "jax": jax.__version__,
            "numpyro": numpyro.__version__,
            "numpy": np.__version__,
        },
        "code_hashes": {
            "run_maser_disk": _file_hash(__file__),
            "checkpointed_nuts": _file_hash(
                sys.modules[run_checkpointed_nuts.__module__].__file__),
            "model_module": _file_hash(
                sys.modules[type(model).__module__].__file__),
        },
    }
    for _chain_idx, _chain_key in enumerate(_chain_keys):
        _nuts_ckpt_path = f"{_nuts_ckpt_base}_chain{_chain_idx}.pkl"
        if args.resume and os.path.isfile(_nuts_ckpt_path):
            fprint(f"Resuming chain {_chain_idx} from {_nuts_ckpt_path}")
        elif args.resume:
            fprint(f"--resume: no NUTS checkpoint found for chain "
                   f"{_chain_idx} at {_nuts_ckpt_path}, starting fresh")

        _metadata = dict(_base_checkpoint_metadata)
        _metadata["chain_index"] = int(_chain_idx)
        _metadata["chain_rng_key"] = np.asarray(_chain_key).tolist()
        fprint(f"NUTS chain {_chain_idx + 1}/{num_chains}: "
               f"{_nuts_ckpt_path}")
        result = run_checkpointed_nuts(
            nuts_kernel, _chain_key,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=1, chain_method="sequential",
            progress_bar=True, checkpoint_path=_nuts_ckpt_path,
            resume=args.resume,
            checkpoint_interval_seconds=_checkpoint_interval_seconds,
            checkpoint_metadata=_metadata)
        _chain_samples.append(result.samples)
        _chain_extra_fields.append(result.extra_fields)
    dt = time.time() - t0

    samples = _combine_chain_samples(_chain_samples)
    _extra_fields = _combine_chain_samples(_chain_extra_fields)
    n_div = int(_extra_fields['diverging'].sum())
    print(f"\nWall time: {dt:.0f}s, Divergences: {n_div}", flush=True)
    suffix = f"nuts_{dist_tag}"
    meta = None

elif sampler == "nss":
    n_live = args.n_live or inf_cfg.get("n_live", 5000)
    num_mcmc_steps = args.num_mcmc_steps or inf_cfg.get("num_mcmc_steps", 0)
    num_delete = args.num_delete or inf_cfg.get("num_delete", 250)
    termination = args.termination or inf_cfg.get("termination", -3)
    devices = args.devices or inf_cfg.get("devices", "auto")
    max_steps = args.max_steps or inf_cfg.get("max_steps", 10)
    max_shrinkage = args.max_shrinkage or inf_cfg.get("max_shrinkage", 100)
    stale_retries = (args.stale_retries if args.stale_retries is not None
                     else inf_cfg.get("stale_retries", 1))
    cov_jitter = (args.cov_jitter if args.cov_jitter is not None
                  else inf_cfg.get("cov_jitter", 1e-6))

    if num_mcmc_steps == 0:
        num_mcmc_steps = None  # run_nss will use ndim

    _label = "joint" if is_joint else galaxy
    fsection(f"Running NSS ({_label}, {n_spots} spots)")
    fprint(f"n_live={n_live}, mcmc_steps={num_mcmc_steps}, "
           f"num_delete={num_delete}")
    fprint(f"max_steps={max_steps}, max_shrinkage={max_shrinkage}, "
           f"stale_retries={stale_retries}, cov_jitter={cov_jitter:g}")
    fprint(f"devices={devices}")

    _nss_ckpt_dir = results_path(
        master_cfg["io"].get("root_output", "results/Maser"),
        "nss_checkpoints", galaxy if not is_joint else "joint")
    os.makedirs(_nss_ckpt_dir, exist_ok=True)
    _nss_ckpt_path = os.path.join(_nss_ckpt_dir, f"nss_ckpt_{dist_tag}.npz")
    _nss_resume = None
    if args.resume and os.path.isfile(_nss_ckpt_path):
        _nss_resume = _nss_ckpt_path
        fprint(f"Resuming from {_nss_ckpt_path}")
    elif args.resume:
        fprint(f"--resume: no checkpoint found at {_nss_ckpt_path}, "
               "starting fresh")
    fprint(f"Checkpoints: {_nss_ckpt_dir}")
    fprint(f"Checkpoint file: {_nss_ckpt_path}")
    _checkpoint_interval_seconds = args.checkpoint_interval_minutes * 60.0
    fprint(f"Checkpoint interval: {args.checkpoint_interval_minutes:g} "
           "minutes")

    t0 = time.time()
    samples = run_nss(
        model, model_kwargs={},
        n_live=n_live, num_mcmc_steps=num_mcmc_steps,
        num_delete=num_delete,
        termination=termination, seed=seed,
        checkpoint_dir=_nss_ckpt_dir, checkpoint_path=_nss_ckpt_path,
        resume_path=_nss_resume,
        checkpoint_interval=_checkpoint_interval_seconds,
        devices=devices,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
        stale_retries=stale_retries,
        cov_jitter=cov_jitter,
    )
    dt = time.time() - t0

    meta = samples.pop("__nested__")
    print(f"\nWall time: {dt:.0f}s", flush=True)
    print(f"log Z = {meta['log_Z']:.2f} +/- {meta['log_Z_err']:.2f}",
          flush=True)
    print(f"n_eff = {meta['n_eff']}", flush=True)
    suffix = f"nss_{dist_tag}"

# ---- Print results ----
fsection("Results")
if is_joint:
    # Print shared params
    for k in ["H0", "sigma_pec", "D_lim", "D_width"]:
        if k in samples:
            s = np.asarray(samples[k])
            print(f"  {k:20s} = {s.mean():10.3f} +/- {s.std():8.3f}",
                  flush=True)
    # Print per-galaxy D_c
    for gname in galaxy_names:
        k = f"{gname}/D_c"
        if k in samples:
            s = np.asarray(samples[k])
            print(f"  {k:20s} = {s.mean():10.3f} +/- {s.std():8.3f}",
                  flush=True)
else:
    param_keys = ['D_c', 'log_MBH', 'i0', 'di_dr', 'Omega0', 'dOmega_dr',
                  'x0', 'y0', 'dv_sys',
                  'sigma_x_floor', 'sigma_y_floor',
                  'sigma_v_sys', 'sigma_v_hv',
                  'sigma_a_floor']
    for k in param_keys:
        if k in samples:
            s = np.asarray(samples[k])
            print(f"  {k:20s} = {s.mean():10.3f} +/- {s.std():8.3f}",
                  flush=True)

    D_c = np.asarray(samples['D_c'])
    H0_ref = float(get_nested(master_cfg, "model/H0_ref", 73.0))
    z_cosmo = D_c * H0_ref / 299792.458
    D_A = D_c / (1 + z_cosmo)
    if 'log_MBH' in samples:
        log_MBH = np.asarray(samples['log_MBH'])
    else:
        log_MBH = np.asarray(samples['eta']) + np.log10(D_A)
    M_BH = 10**log_MBH
    print(f"\n  D_A = {D_A.mean():.1f} +/- {D_A.std():.1f} Mpc", flush=True)
    print(f"  M_BH = {M_BH.mean():.2e} +/- {M_BH.std():.2e} M_sun",
          flush=True)

    # Implied H0 from Hubble law (no peculiar velocity correction)
    v_sys_obs = float(gcfg["v_sys_obs"])
    dv_sys = np.asarray(samples['dv_sys']) if 'dv_sys' in samples else np.zeros(1)
    v_sys = v_sys_obs + dv_sys
    H0_implied = v_sys_obs / D_c

    frame = data.get("velocity_frame", "unknown")
    print(f"  v_sys ({frame}) = {v_sys.mean():.1f} +/- {v_sys.std():.1f} km/s",
          flush=True)

    from candel.pvdata.megamaser_data import v_sys_to_cmb
    ra_gal = float(gcfg["ra"])
    dec_gal = float(gcfg["dec"])
    v_cmb = v_sys_to_cmb(v_sys, frame, ra_gal, dec_gal)
    if v_cmb is not None:
        print(f"  v_sys (cmb) = {v_cmb.mean():.1f} +/- {v_cmb.std():.1f} km/s",
              flush=True)
    else:
        print(f"  v_sys (cmb) = unknown frame, conversion skipped",
              flush=True)

    print(f"  H0 (v_sys_obs/D_c) = {H0_implied.mean():.1f} "
          f"+/- {H0_implied.std():.1f} km/s/Mpc", flush=True)

    # Warp angles re-expressed at r = 0 (same units as sampled: deg, deg/mas)
    _w0 = remap_warp_to_r0(
        samples, model._r_ang_ref_i, model._r_ang_ref_Omega)
    print(f"\n  Warp at r=0 (pivot was r_i={model._r_ang_ref_i:.3f}, "
          f"r_Ω={model._r_ang_ref_Omega:.3f} mas):", flush=True)
    for _k, _v in _w0.items():
        print(f"  {_k:20s} = {_v.mean():10.3f} +/- {_v.std():8.3f}",
              flush=True)

# Summary table
fsection("Summary")
if sampler == "nuts":
    print_clean_summary(samples)
else:
    print_nested_summary(samples, meta=meta)

# ---- Output ----
outdir = results_path(master_cfg["io"].get("root_output", "results/Maser"))
os.makedirs(outdir, exist_ok=True)
_out_name = "joint" if is_joint else galaxy

# ---- Automatic convergence check ----
_conv_cfg = master_cfg.get("convergence", {})
if _conv_cfg.get("auto_check", False) and not is_joint:
    from candel.model.maser_convergence import check_convergence, summarize
    fsection("Convergence check")
    try:
        _conv_res = check_convergence(model, samples, _conv_cfg)
        summarize(_conv_res)
        if _conv_cfg.get("save_json", True):
            import json
            _conv_path = os.path.join(
                outdir, f"{_out_name}_{suffix}_convergence.json")
            with open(_conv_path, "w") as _f:
                json.dump(_conv_res, _f, indent=2)
            fprint(f"Saved convergence report to {_conv_path}")
    except Exception as e:
        fprint(f"[WARN] Convergence check failed: {type(e).__name__}: {e}")
elif _conv_cfg.get("auto_check", False) and is_joint:
    fprint("Convergence auto-check skipped: joint mode not supported.")

# Spot classification plot (single-galaxy only)
if not is_joint:
    _cls = np.where(~data["is_highvel"], "systemic",
                    np.where(data["is_blue"], "blue HV", "red HV"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for cls, col in [("systemic", "forestgreen"), ("blue HV", "royalblue"),
                     ("red HV", "tomato")]:
        m = _cls == cls
        axes[0].scatter(data["x"][m], data["y"][m], c=col, s=12, alpha=0.7,
                        label=f"{cls} ({m.sum()})")
        axes[1].scatter(data["x"][m],
                        data["velocity"][m] - data["v_sys_obs"],
                        c=col, s=12, alpha=0.7)
    axes[0].set_xlabel(r"$\Delta x$ (mas)")
    axes[0].set_ylabel(r"$\Delta y$ (mas)")
    axes[0].legend(fontsize=8)
    axes[0].invert_xaxis()
    axes[1].set_xlabel(r"$\Delta x$ (mas)")
    axes[1].set_ylabel(r"$v - v_\mathrm{sys}$ (km s$^{-1}$)")
    fig.tight_layout()
    fname_spots = os.path.join(outdir, f"{_out_name}_{suffix}_spots.png")
    fig.savefig(fname_spots, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Spot classification plot saved to {fname_spots}", flush=True)

# Save HDF5
outpath = os.path.abspath(
    os.path.join(outdir, f"{_out_name}_{suffix}.hdf5"))
with H5File(outpath, 'w') as f:
    grp = f.create_group("samples", track_order=True)
    for k in sort_params(samples.keys()):
        v = samples[k]
        if k == 'r_ang':
            continue
        grp.create_dataset(k, data=np.asarray(v), dtype=np.float32)
    if not is_joint:
        f.create_dataset("D_A", data=D_A.astype(np.float32))
        f.create_dataset("M_BH", data=M_BH.astype(np.float32))
    if meta is not None:
        f.attrs["log_Z"] = meta['log_Z']
        f.attrs["log_Z_err"] = meta['log_Z_err']
        f.attrs["n_eff"] = meta['n_eff']
fprint(f"saved samples to {outpath}")

# Corner plots (skip for joint — too many params)
if not is_joint:
    if meta is not None and meta.get("n_eff", 2) < 2:
        fprint(f"skipping corner plots: n_eff={meta['n_eff']}")
    else:
        fname_corner = os.path.join(outdir, f"{_out_name}_{suffix}_corner.png")
        plot_corner(samples, show_fig=False, filename=fname_corner)
        fname_corner_raw = os.path.join(
            outdir, f"{_out_name}_{suffix}_corner_raw.png")
        plot_corner(samples, show_fig=False, filename=fname_corner_raw,
                    smooth=False)
