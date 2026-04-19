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

import argparse
import tempfile
import time

import tomli

# Check per-galaxy use_float64 before importing JAX (must be set pre-init)
with open("scripts/megamaser/config_maser.toml", "rb") as _f:
    _pre_cfg = tomli.load(_f)
_galaxy_arg = sys.argv[1] if len(sys.argv) > 1 else ""
_gal_cfg = _pre_cfg.get("model", {}).get("galaxies", {}).get(_galaxy_arg, {})
if _gal_cfg.get("use_float64", False):
    import jax
    jax.config.update("jax_enable_x64", True)
    print(f"float64 enabled for {_galaxy_arg}", flush=True)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tomli_w
from h5py import File as H5File
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median, init_to_value

from candel.inference.inference import print_clean_summary, _setup_dense_mass
from candel.inference.nested import print_nested_summary, run_nss
from candel.model.model_H0_maser import JointMaserModel, MaserDiskModel
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import fprint, fsection, get_nested, plot_corner

_devs = jax.devices()
_dev_names = ", ".join(d.device_kind for d in _devs)
_precision = "float64" if jax.config.jax_enable_x64 else "float32"
print(f"JAX platform: {jax.default_backend()}, devices: {_devs} "
      f"({_dev_names}), precision: {_precision}", flush=True)

# ---- Load master config ----
with open("scripts/megamaser/config_maser.toml", "rb") as f:
    master_cfg = tomli.load(f)

inf_cfg = master_cfg["inference"]

# >>> f-grid helpers
_FGRID_KEYS = (
    "n_phi_hv_high",
    "n_phi_hv_low",
    "n_phi_sys",
    "n_phi_hv_high_mode1",
    "n_phi_hv_low_mode1",
    "n_phi_sys_mode1",
    "n_r_local",
    "n_r_brute",
)


def _round_to_odd(n, f):
    """Scale n by f, round to the nearest odd int, floor at 3."""
    m = max(3, int(round(n * f)))
    if m % 2 == 0:
        m += 1
    return m
# <<< f-grid helpers

# ---- Parse args (CLI overrides config) ----
parser = argparse.ArgumentParser()
parser.add_argument("galaxy", type=str)
parser.add_argument("--sampler", type=str, default=None,
                    choices=["nuts", "nss"])
parser.add_argument("--seed", type=int, default=None)
# NUTS
parser.add_argument("--num-warmup", type=int, default=None)
parser.add_argument("--num-samples", type=int, default=None)
# NSS
parser.add_argument("--n-live", type=int, default=None)
parser.add_argument("--num-mcmc-steps", type=int, default=None)
parser.add_argument("--num-delete", type=int, default=None)
parser.add_argument("--termination", type=float, default=None)
parser.add_argument("--mode", type=str, default=None,
                    choices=["mode0", "mode1", "mode2"],
                    help="Sampling mode: mode0 samples r AND phi per spot, "
                         "mode1 samples r and marginalises phi, "
                         "mode2 marginalises both (default: from config)")
parser.add_argument("--grid-factor", type=float, default=1.0,
                    help="Multiply all grid sizes by this factor")
parser.add_argument("--map-only", action="store_true",
                    help="Run DE optimizer only, skip sampling")
parser.add_argument("--D-c-prior", type=str, default=None,
                    choices=["uniform", "volume"],
                    help="Override D_c prior (default: from config)")
parser.add_argument("--log2-N", type=int, default=None,
                    help="Override Sobol log2_N for DE/Sobol optimizer")
parser.add_argument("--init-method", type=str, default=None,
                    choices=["config", "sobol_adam", "median"],
                    help="NUTS init: config (globals from config, "
                         "r_ang from prior), sobol_adam, median")
parser.add_argument("--no-ecc", action="store_true",
                    help="Disable eccentricity model (override config)")
parser.add_argument("--no-quadratic-warp", action="store_true",
                    help="Disable quadratic warp (override config)")
args = parser.parse_args()

galaxy = args.galaxy
sampler = args.sampler or inf_cfg.get("sampler", "nss")
seed = args.seed or inf_cfg.get("seed", 42)

# Build descriptive output tag
_mcfg = master_cfg["model"]
if args.D_c_prior is not None:
    _mcfg["D_c_prior"] = args.D_c_prior
if args.log2_N is not None:
    master_cfg.setdefault("optimise", {})["log2_N"] = args.log2_N
_tags = []

# Distance prior
_dc_prior = _mcfg.get("D_c_prior", "uniform")
_tags.append("Dvol" if _dc_prior == "volume" else "Dflat")

# Mode tag (omit for default mode2).
if args.mode is not None and args.mode != "mode2":
    _tags.append(args.mode)

# Grid factor
if args.grid_factor != 1.0:
    _tags.append(f"gf{args.grid_factor:g}")

# Model feature overrides
if args.no_ecc:
    _tags.append("noecc")
if args.no_quadratic_warp:
    _tags.append("noqw")

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
        d = load_megamaser_spots("data/Megamaser", gname,
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
    data = load_megamaser_spots("data/Megamaser", galaxy, v_sys_obs=v_sys_obs)
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
        "num_warmup": args.num_warmup or inf_cfg.get("num_warmup", 1000),
        "num_samples": args.num_samples or inf_cfg.get("num_samples", 1000),
        "num_chains": inf_cfg.get("num_chains", 1),
        "chain_method": inf_cfg.get("chain_method", "sequential"),
        "seed": seed,
        "dense_mass_blocks": dense_mass_blocks,
        "init_maxiter": inf_cfg.get("init_maxiter", 0),
        "init_method": inf_cfg.get("init_method", "lbfgs"),
        "max_tree_depth": inf_cfg.get("max_tree_depth", 10),
    },
    "model": master_cfg["model"],
    "io": master_cfg["io"],
    "optimise": master_cfg.get("optimise", {}),
}
if args.mode is not None:
    config["model"]["mode"] = args.mode
if args.no_ecc or args.no_quadratic_warp:
    gals = config["model"].setdefault("galaxies", {})
    gcfg_ov = gals.setdefault(galaxy if not is_joint else "", {})
    if args.no_ecc:
        gcfg_ov["use_ecc"] = False
        fprint("CLI override: use_ecc = False")
    if args.no_quadratic_warp:
        gcfg_ov["use_quadratic_warp"] = False
        fprint("CLI override: use_quadratic_warp = False")

# Grid resolution multiplier
if args.grid_factor != 1.0:
    gf = args.grid_factor
    m = config["model"]
    m["G_phi_half"] = int(m.get("G_phi_half", 202) * gf)
    m["n_inner_sys"] = int(m.get("n_inner_sys", 202) * gf)
    m["n_wing_sys"] = int(m.get("n_wing_sys", 100) * gf)
    m["n_r"] = int(m.get("n_r", 251) * gf)
    fprint(f"grid-factor={gf:g}: G_phi_half={m['G_phi_half']}, "
           f"n_inner_sys={m['n_inner_sys']}, "
           f"n_wing_sys={m['n_wing_sys']}, n_r={m['n_r']}")

# Joint: apply prior overrides from [joint.priors] config section
if is_joint:
    joint_priors = master_cfg.get("joint", {}).get("priors", {})
    for pname, pval in joint_priors.items():
        config["model"]["priors"][pname] = pval
        fprint(f"joint override: {pname} -> {pval}")

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

# MAP-only mode: run DE optimizer and exit, regardless of sampler setting.
if args.map_only:
    if is_joint:
        print("ERROR: --map-only is not supported for joint mode.",
              flush=True)
        sys.exit(1)
    from candel.inference.optimise import find_MAP
    fsection(f"Running DE MAP ({galaxy}, {n_spots} spots)")
    init_params = find_MAP(model, model_kwargs={}, seed=seed)
    fprint("MAP-only run (--map-only), done.")
    sys.exit(0)

if sampler == "nuts":
    num_warmup = args.num_warmup or inf_cfg.get("num_warmup", 1000)
    num_samples = args.num_samples or inf_cfg.get("num_samples", 1000)

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
            init_strategy = init_to_median(num_samples=100)
            fprint("NUTS init: median (no config init found)")
    else:
        fsection(f"Running NUTS ({galaxy}, {n_spots} spots)")
        init_cfg = gcfg.get("init", {})
        init_method = args.init_method or inf_cfg.get("init_method", "config")

        _init_desc = {
            "sobol_adam": "Sobol+Adam MAP",
            "config": "config globals + r_ang from prior",
            "median": "numpyro median",
        }
        fprint(f"Initialisation: {init_method} "
               f"({_init_desc.get(init_method, '?')})")

        if init_method == "sobol_adam":
            from candel.inference.optimise import find_MAP
            init_params = find_MAP(model, model_kwargs={}, seed=seed)
            init_strategy = init_to_value(values=init_params)
            for k, v in init_params.items():
                if v.ndim == 0:
                    fprint(f"  {k:20s} = {float(v):12.4f}")
                else:
                    fprint(f"  {k:20s} = [{len(v)} values]")
        elif init_method == "config":
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

            # Mode 0/1: model samples r_ang, so init needs r_ang.
            # Mode 0 also samples phi_u (aux ∈ [0,1] mapped to phi).
            if model.mode in ("mode0", "mode1") and "r_ang" not in init_params:
                rng = np.random.default_rng(seed)
                if model._r_ang_init_dist is not None:
                    from scipy.stats import truncnorm
                    d = model._r_ang_init_dist
                    a = (d["low"] - d["loc"]) / d["scale"]
                    b = (d["high"] - d["loc"]) / d["scale"]
                    _r = truncnorm.rvs(
                        a, b, loc=d["loc"], scale=d["scale"],
                        size=model.n_spots, random_state=rng)
                    fprint(f"  r_ang: TruncatedNormal init, "
                           f"[{_r.min():.3f}, {_r.max():.3f}] mas")
                else:
                    _r = rng.uniform(
                        float(model._r_ang_lo),
                        float(model._r_ang_hi),
                        model.n_spots)
                    fprint(f"  r_ang: uniform random in r_ang, "
                           f"[{_r.min():.3f}, {_r.max():.3f}] mas")
                init_params["r_ang"] = jnp.asarray(_r)

            # Mode 0: also init phi_u (uniform aux mapped to spot-type
            # phi range). Default 0.5 places each spot at the centre
            # of its allowed region.
            if model.mode == "mode0" and "phi_u" not in init_params:
                init_params["phi_u"] = jnp.full(model.n_spots, 0.5)
                fprint("  phi_u: default 0.5 (centre of allowed phi)")

            init_strategy = init_to_value(values=init_params)
            for k, v in sorted(init_params.items()):
                v = jnp.asarray(v)
                if v.ndim == 0:
                    fprint(f"  {k:20s} = {float(v):12.4f}")
                else:
                    fprint(f"  {k:20s} = [{len(v)} values]")
        else:
            init_strategy = init_to_median(num_samples=20)
    # Dense mass matrix: joint uses full dense; single-galaxy uses
    # per-galaxy dense_mass_globals flag for a globals-only block.
    t0 = time.time()
    if is_joint:
        _dense_mass = True
        fprint("Dense mass: full dense (joint mode)")
    elif not is_joint and gcfg.get("dense_mass_globals", False):
        # Collect all scalar (non-r_ang) param names from init config,
        # excluding params that aren't sampled in single-galaxy mode
        # or that are disabled by CLI overrides.
        _skip = {"r_ang", "phi_u", "H0", "sigma_pec"}
        if not model.use_ecc:
            _skip |= {"e_x", "e_y", "ecc", "periapsis", "dperiapsis_dr"}
        if not model.use_quadratic_warp:
            _skip |= {"d2i_dr2", "d2Omega_dr2"}
        _global_names = [k for k in init_cfg if k not in _skip]
        _dense_mass = [tuple(_global_names)]
        fprint(f"Dense mass: 1 block with {len(_global_names)} globals: "
               f"{_global_names}")
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
        _tap = float(inf_cfg.get("target_accept_prob", 0.8))
    else:
        _tap = float(gcfg.get("target_accept_prob",
                              inf_cfg.get("target_accept_prob", 0.8)))
    nuts_kernel = NUTS(model,
                       max_tree_depth=inf_cfg.get("max_tree_depth", 10),
                       target_accept_prob=_tap,
                       dense_mass=_dense_mass,
                       init_strategy=init_strategy)

    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=1, progress_bar=True)
    mcmc.run(random.PRNGKey(seed))
    dt = time.time() - t0

    samples = mcmc.get_samples()
    n_div = int(mcmc.get_extra_fields()['diverging'].sum())
    print(f"\nWall time: {dt:.0f}s, Divergences: {n_div}", flush=True)
    suffix = f"nuts_{dist_tag}"
    meta = None

elif sampler == "nss":
    n_live = args.n_live or inf_cfg.get("n_live", 5000)
    num_mcmc_steps = args.num_mcmc_steps or inf_cfg.get("num_mcmc_steps", 0)
    num_delete = args.num_delete or inf_cfg.get("num_delete", 250)
    termination = args.termination or inf_cfg.get("termination", -3)

    if num_mcmc_steps == 0:
        num_mcmc_steps = None  # run_nss will use ndim

    _label = "joint" if is_joint else galaxy
    fsection(f"Running NSS ({_label}, {n_spots} spots)")
    fprint(f"n_live={n_live}, mcmc_steps={num_mcmc_steps}, "
           f"num_delete={num_delete}")
    t0 = time.time()
    samples = run_nss(
        model, model_kwargs={},
        n_live=n_live, num_mcmc_steps=num_mcmc_steps,
        num_delete=num_delete,
        termination=termination, seed=seed,
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
    H0_implied = v_sys_obs / D_c
    print(f"  H0 (v_sys/D_c) = {H0_implied.mean():.1f} "
          f"+/- {H0_implied.std():.1f} km/s/Mpc", flush=True)

# Summary table
fsection("Summary")
if sampler == "nuts":
    print_clean_summary(samples)
else:
    print_nested_summary(samples, meta=meta)

# ---- Output ----
outdir = os.path.abspath(master_cfg["io"].get("root_output", "results/Maser"))
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
    grp = f.create_group("samples")
    for k, v in samples.items():
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
    fname_corner = os.path.join(outdir, f"{_out_name}_{suffix}_corner.png")
    plot_corner(samples, show_fig=False, filename=fname_corner)
    fname_corner_raw = os.path.join(
        outdir, f"{_out_name}_{suffix}_corner_raw.png")
    plot_corner(samples, show_fig=False, filename=fname_corner_raw,
                smooth=False)
