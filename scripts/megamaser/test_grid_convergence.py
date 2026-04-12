"""Grid convergence test for megamaser disk model.

Evaluates total log-likelihood at the MAP point while sweeping phi and r
grid resolutions. Compares NGC4258 (high-quality, nearby) vs NGC5765b
(typical distant megamaser) to quantify resolution sensitivity.
"""
import sys
import os
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
import tomli
import tomli_w

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from candel.model.model_H0_maser import MaserDiskModel
from candel.pvdata.megamaser_data import load_megamaser_spots

jax.config.update("jax_platform_name", "gpu")
print(f"JAX platform: {jax.default_backend()}", flush=True)

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
with open(CONFIG_PATH, "rb") as f:
    master_cfg = tomli.load(f)

galaxies_cfg = master_cfg["model"]["galaxies"]


def build_model(galaxy, G_phi_half=202, n_inner_sys=202, n_wing_sys=100,
                n_r=251):
    """Build a MaserDiskModel with specified grid sizes."""
    cfg = master_cfg.copy()
    cfg["model"] = master_cfg["model"].copy()
    cfg["model"]["G_phi_half"] = G_phi_half
    cfg["model"]["n_inner_sys"] = n_inner_sys
    cfg["model"]["n_wing_sys"] = n_wing_sys
    cfg["model"]["n_r"] = n_r

    gcfg = galaxies_cfg[galaxy]
    v_sys = gcfg["v_sys_obs"]
    data = load_megamaser_spots(
        master_cfg["io"]["maser_data"]["root"], galaxy=galaxy,
        v_sys_obs=v_sys)
    if "D_lo" in gcfg:
        data["D_lo"] = float(gcfg["D_lo"])
    if "D_hi" in gcfg:
        data["D_hi"] = float(gcfg["D_hi"])

    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(cfg, tmp)
    tmp.close()
    model = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)
    return model


def get_map_params(galaxy):
    """Get MAP init params from config."""
    gcfg = galaxies_cfg[galaxy]
    init = gcfg.get("init", {})
    return {k: jnp.asarray(v) for k, v in init.items()}


def eval_logp(model, params):
    """Evaluate log-density at given params."""
    from numpyro.infer.util import log_density
    from numpyro import handlers
    import jax.random as random
    # Seed the model so any un-supplied params get sampled from prior,
    # then substitute the params we do have.
    seeded = handlers.seed(handlers.substitute(model, data=params),
                           rng_seed=0)
    ld, _ = log_density(seeded, (), {}, params)
    return float(ld)


# ---- Test 1: Phi grid convergence (with r marginalised) ----
print("\n" + "=" * 70)
print("TEST 1: Phi grid convergence (r marginalised, varying G_phi_half)")
print("=" * 70)

phi_sizes = [50, 100, 150, 202, 300, 400, 600, 800]

for galaxy in ["NGC4258", "NGC5765b"]:
    print(f"\n--- {galaxy} ---")
    params = get_map_params(galaxy)
    if not params:
        print(f"  No init params for {galaxy}, skipping.")
        continue

    results = []
    for G in phi_sizes:
        # Scale systemic grid proportionally
        n_inner = G
        n_wing = max(20, G // 2)
        model = build_model(galaxy, G_phi_half=G, n_inner_sys=n_inner,
                            n_wing_sys=n_wing, n_r=251)
        lp = eval_logp(model, params)
        results.append((G, n_inner, n_wing, lp))
        print(f"  G_phi_half={G:4d}, n_inner={n_inner:4d}, "
              f"n_wing={n_wing:3d} -> logP = {lp:.4f}", flush=True)

    # Convergence: difference from finest grid
    lp_ref = results[-1][3]
    print(f"\n  Convergence (vs G={phi_sizes[-1]}):")
    for G, ni, nw, lp in results:
        print(f"    G={G:4d}: delta_logP = {lp - lp_ref:+.4f}")


# ---- Test 2: R grid convergence ----
print("\n" + "=" * 70)
print("TEST 2: R grid convergence (varying n_r, phi grids at default)")
print("=" * 70)

r_sizes = [51, 101, 151, 251, 401, 601]

for galaxy in ["NGC4258", "NGC5765b"]:
    print(f"\n--- {galaxy} ---")
    params = get_map_params(galaxy)
    if not params:
        print(f"  No init params for {galaxy}, skipping.")
        continue

    results = []
    for nr in r_sizes:
        model = build_model(galaxy, n_r=nr)
        lp = eval_logp(model, params)
        results.append((nr, lp))
        print(f"  n_r={nr:4d} -> logP = {lp:.4f}", flush=True)

    lp_ref = results[-1][1]
    print(f"\n  Convergence (vs n_r={r_sizes[-1]}):")
    for nr, lp in results:
        print(f"    n_r={nr:4d}: delta_logP = {lp - lp_ref:+.4f}")


# ---- Test 3: Joint phi+r scaling ----
print("\n" + "=" * 70)
print("TEST 3: Joint phi+r scaling (double/quadruple everything)")
print("=" * 70)

scale_factors = [0.5, 1.0, 2.0, 3.0, 4.0]

for galaxy in ["NGC4258", "NGC5765b"]:
    print(f"\n--- {galaxy} ---")
    params = get_map_params(galaxy)
    if not params:
        print(f"  No init params for {galaxy}, skipping.")
        continue

    results = []
    base_G, base_ni, base_nw, base_nr = 202, 202, 100, 251
    for sf in scale_factors:
        G = int(base_G * sf)
        ni = int(base_ni * sf)
        nw = max(20, int(base_nw * sf))
        nr = int(base_nr * sf)
        model = build_model(galaxy, G_phi_half=G, n_inner_sys=ni,
                            n_wing_sys=nw, n_r=nr)
        lp = eval_logp(model, params)
        results.append((sf, G, ni, nw, nr, lp))
        print(f"  scale={sf:.1f}x: G={G:4d}, n_inner={ni:4d}, "
              f"n_wing={nw:3d}, n_r={nr:4d} -> logP = {lp:.4f}",
              flush=True)

    lp_ref = results[-1][5]
    print(f"\n  Convergence (vs {scale_factors[-1]}x):")
    for sf, G, ni, nw, nr, lp in results:
        print(f"    {sf:.1f}x: delta_logP = {lp - lp_ref:+.4f}")

print("\nDone.", flush=True)
