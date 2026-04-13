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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tomli
import tomli_w
from h5py import File as H5File
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median, init_to_value

from candel.inference.inference import print_clean_summary
from candel.inference.nested import print_nested_summary, run_nss
from candel.model.model_H0_maser import JointMaserModel, MaserDiskModel
from candel.pvdata.megamaser_data import load_megamaser_spots
from candel.util import fprint, fsection, plot_corner

_devs = jax.devices()
_dev_names = ", ".join(d.device_kind for d in _devs)
print(f"JAX platform: {jax.default_backend()}, devices: {_devs} ({_dev_names})",
      flush=True)

# ---- Load master config ----
with open("scripts/megamaser/config_maser.toml", "rb") as f:
    master_cfg = tomli.load(f)

inf_cfg = master_cfg["inference"]

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
parser.add_argument("--sample-r", action="store_true",
                    help="Sample r_ang explicitly instead of marginalising")
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
                    help="Override NUTS init method")
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

# Clump weighting
if not _mcfg.get("use_clump_weight", True):
    _tags.append("noclump")

# Mode: marginalise r vs sample r
if args.sample_r:
    _tags.append("sampleR")

# Grid factor
if args.grid_factor != 1.0:
    _tags.append(f"gf{args.grid_factor:g}")

dist_tag = "_".join(_tags)

# ---- Validate galaxy ----
galaxies = master_cfg["model"]["galaxies"]
is_joint = (galaxy == "joint")

if not is_joint and galaxy not in galaxies:
    print(f"Unknown galaxy '{galaxy}'. "
          f"Available: {list(galaxies.keys())} + ['joint']", flush=True)
    sys.exit(1)

# ---- Load data ----
_clump_gals = _mcfg.get("clump_galaxies")

if is_joint:
    # Exclude NGC4258 (geometric parallax, not in the MCP sample)
    galaxy_names = [g for g in galaxies if g != "NGC4258"]
    fsection(f"Loading joint data ({len(galaxy_names)} galaxies)")
    data_list = []
    for gname in galaxy_names:
        gcfg_g = galaxies[gname]
        d = load_megamaser_spots("data/Megamaser", gname,
                                v_sys_obs=gcfg_g["v_sys_obs"],
                                clump_galaxies=_clump_gals)
        if "D_lo" in gcfg_g and "D_hi" in gcfg_g:
            d["D_lo"] = float(gcfg_g["D_lo"])
            d["D_hi"] = float(gcfg_g["D_hi"])
        data_list.append(d)
    n_spots = sum(d["n_spots"] for d in data_list)
else:
    gcfg = galaxies[galaxy]
    v_sys_obs = gcfg["v_sys_obs"]
    fsection(f"Loading {galaxy} data")
    data = load_megamaser_spots("data/Megamaser", galaxy, v_sys_obs=v_sys_obs,
                               clump_galaxies=_clump_gals)
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
if args.sample_r:
    config["model"]["marginalise_r"] = False

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
if sampler == "nss" and args.sample_r:
    print("ERROR: nested sampling requires marginalising r "
          "(remove --sample-r)", flush=True)
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
        if init_method == "sobol_adam":
            from candel.inference.optimise import find_MAP
            init_params = find_MAP(model, model_kwargs={}, seed=seed)
            init_strategy = init_to_value(values=init_params)
            fprint("NUTS init from Sobol+Adam MAP:")
            for k, v in init_params.items():
                if v.ndim == 0:
                    fprint(f"  {k:20s} = {float(v):12.4f}")
                else:
                    fprint(f"  {k:20s} = [{len(v)} values]")
        elif init_cfg and init_method == "config":
            init_params = {k: jnp.asarray(v) for k, v in init_cfg.items()}
            # Mode 1: compute log_r_ang init from acceleration
            if args.sample_r and "log_r_ang" not in init_params:
                from candel.model.model_H0_maser import C_v, C_a
                _D_A = float(init_params["D_c"]) / 1.002
                _M = 10.0**(float(init_params["eta"])
                            + np.log10(_D_A) - 7.0)
                _si = abs(np.sin(np.deg2rad(float(init_params["i0"]))))
                _sa2 = float(init_params["sigma_a_floor"])**2
                _sa_tot = np.sqrt(np.asarray(model._all_sigma_a)**2
                                  + _sa2)
                _snr = np.abs(np.asarray(model._all_a)) / (_sa_tot + 1e-30)
                _r_acc = np.sqrt(
                    C_a * _M * _si
                    / (_D_A**2
                       * (np.abs(np.asarray(model._all_a)) + 1e-30)))
                _r_mid = np.exp(0.5 * (np.log(float(model._r_ang_lo))
                                       + np.log(float(model._r_ang_hi))))
                _good = np.asarray(model._all_has_accel) & (_snr >= 2.0)
                _r_init = np.where(_good, _r_acc, _r_mid)
                _r_init = np.clip(_r_init, float(model._r_ang_lo) * 1.01,
                                  float(model._r_ang_hi) * 0.99)
                init_params["log_r_ang"] = jnp.asarray(np.log(_r_init))
                fprint(f"  log_r_ang init from accel: "
                       f"r=[{_r_init.min():.3f}, {_r_init.max():.3f}] mas")
            init_strategy = init_to_value(values=init_params)
            fprint(f"NUTS init from config ({len(init_params)} params):")
            for k, v in sorted(init_params.items()):
                v = jnp.asarray(v)
                if v.ndim == 0:
                    fprint(f"  {k:20s} = {float(v):12.4f}")
                else:
                    fprint(f"  {k:20s} = [{len(v)} values]")
        else:
            init_strategy = init_to_median(num_samples=20)
            fprint("NUTS init: median")
    t0 = time.time()
    nuts_kernel = NUTS(model,
                       max_tree_depth=inf_cfg.get("max_tree_depth", 10),
                       target_accept_prob=0.8,
                       dense_mass=True if is_joint else False,
                       init_strategy=init_strategy)

    # For Mode 1 (--sample-r) with adaptive phi: use HMCGibbs to
    # decouple the 358 log_r_ang from the ~20 global params. NUTS
    # handles globals; vectorized MH handles log_r_ang.
    if args.sample_r and model.adaptive_phi:
        from numpyro.infer import HMCGibbs

        # Proposal width for log_r_ang MH: position-derived sigma
        _sigma_pos = np.sqrt(np.asarray(model._all_sigma_x2)
                             + np.asarray(model._all_sigma_y2))
        # r_est from init
        _r_est = np.exp(np.asarray(init_params.get(
            "log_r_ang",
            np.log(np.ones(model.n_spots) * 4.0))))
        _mh_sigma = jnp.asarray(
            np.clip(_sigma_pos / (_r_est + 1e-30), 0.0001, 0.1))
        fprint(f"Gibbs MH sigma(log_r): [{float(_mh_sigma.min()):.5f}, "
               f"{float(_mh_sigma.max()):.5f}]")

        def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
            """Per-spot MH for log_r_ang conditioned on globals.

            Each spot's r_ang is updated independently with a Gaussian
            proposal in log-r. Accept/reject per spot using the
            per-spot phi-marginalised logL from adaptive phi.
            """
            log_r_cur = gibbs_sites["log_r_ang"]
            r_cur = jnp.exp(log_r_cur)

            # Propose: Gaussian in log-r, per spot
            key1, key2 = jax.random.split(rng_key)
            log_r_prop = log_r_cur + _mh_sigma * jax.random.normal(
                key1, shape=log_r_cur.shape)
            log_r_lo = jnp.log(model._r_ang_lo)
            log_r_hi = jnp.log(model._r_ang_hi)
            log_r_prop = jnp.clip(log_r_prop, log_r_lo, log_r_hi)
            r_prop = jnp.exp(log_r_prop)

            # Extract global params from hmc_sites
            D_c = hmc_sites["D_c"]
            h = hmc_sites["H0"] / 100.0
            z = model.distance2redshift(
                jnp.atleast_1d(D_c), h=h).squeeze()
            D_A = D_c / (1 + z)
            eta = hmc_sites["eta"]
            M_BH = 10.0**(eta + jnp.log10(D_A) - 7.0)
            x0 = hmc_sites["x0"] * 1e-3
            y0 = hmc_sites["y0"] * 1e-3
            v_sys = model.v_sys_obs + hmc_sites["dv_sys"]
            i0 = jnp.deg2rad(hmc_sites["i0"])
            di_dr = jnp.deg2rad(hmc_sites["di_dr"])
            Omega0 = jnp.deg2rad(hmc_sites["Omega0"])
            dOmega_dr = jnp.deg2rad(hmc_sites["dOmega_dr"])
            sx2 = (hmc_sites["sigma_x_floor"] * 1e-3)**2
            sy2 = (hmc_sites["sigma_y_floor"] * 1e-3)**2
            vvs = hmc_sites["sigma_v_sys"]**2
            vvh = hmc_sites["sigma_v_hv"]**2
            sa2 = hmc_sites["sigma_a_floor"]**2

            ecc_kw = {}
            if model.use_ecc:
                ecc_kw["ecc"] = hmc_sites["ecc"]
                ecc_kw["periapsis0"] = jnp.deg2rad(hmc_sites["periapsis"])
                ecc_kw["dperiapsis_dr"] = jnp.deg2rad(
                    hmc_sites["dperiapsis_dr"])
            quad_kw = {}
            if model.use_quadratic_warp:
                quad_kw["d2i_dr2"] = jnp.deg2rad(hmc_sites["d2i_dr2"])
                quad_kw["d2Omega_dr2"] = jnp.deg2rad(
                    hmc_sites["d2Omega_dr2"])

            common = (x0, y0, D_A, M_BH, v_sys,
                      model._r_ang_ref, i0, di_dr, Omega0, dOmega_dr,
                      sx2, sy2, vvs, vvh, sa2)

            # Per-spot logL at current and proposed r
            ll_cur = model._eval_adaptive_phi_mode1(
                r_cur, *common, **ecc_kw, **quad_kw)
            ll_prop = model._eval_adaptive_phi_mode1(
                r_prop, *common, **ecc_kw, **quad_kw)

            # Jacobian: uniform prior on r → factor(log_r)
            ll_cur = ll_cur + log_r_cur
            ll_prop = ll_prop + log_r_prop

            # Per-spot accept/reject
            log_alpha = ll_prop - ll_cur
            u = jnp.log(jax.random.uniform(key2, shape=log_r_cur.shape))
            accept = u < log_alpha
            log_r_new = jnp.where(accept, log_r_prop, log_r_cur)
            return {"log_r_ang": log_r_new}

        kernel = HMCGibbs(nuts_kernel, gibbs_fn=gibbs_fn,
                          gibbs_sites=["log_r_ang"])
        fprint("Using HMCGibbs: NUTS(globals) + MH(log_r_ang)")
    else:
        kernel = nuts_kernel

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
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
    z_cosmo = D_c * 73.0 / 299792.458
    D_A = D_c / (1 + z_cosmo)
    if 'log_MBH' in samples:
        log_MBH = np.asarray(samples['log_MBH'])
    else:
        log_MBH = np.asarray(samples['eta']) + np.log10(D_A)
    M_BH = 10**log_MBH
    print(f"\n  D_A = {D_A.mean():.1f} +/- {D_A.std():.1f} Mpc", flush=True)
    print(f"  M_BH = {M_BH.mean():.2e} +/- {M_BH.std():.2e} M_sun",
          flush=True)

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
