"""Regression harness for the maser GPU refactor.

Usage:
  python scripts/megamaser/test_gpu_refactor.py --capture
  python scripts/megamaser/test_gpu_refactor.py --check
"""
import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import tomli

from candel.model.maser_convergence import (
    _production_ll_mode1, _production_ll_mode2, build_model,
)

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
BASELINE_PATH = "scripts/megamaser/baseline_gpu_refactor.npz"
GALAXIES = ["CGCG074-064", "NGC5765b", "NGC6264", "NGC6323",
            "UGC3789", "NGC4258"]
TOL_ABS = 1e-10


def compute_ll(galaxy, master_cfg):
    # NGC4258 forbids Mode 2 (forbid_marginalise_r=true); force Mode 1.
    overrides = {"mode": "mode1"} if galaxy == "NGC4258" else {}
    model = build_model(galaxy, master_cfg, **overrides)
    init = master_cfg["model"]["galaxies"][galaxy]["init"]
    sample = {k: np.asarray(v) for k, v in init.items()}
    phys_args, phys_kw, diag = model.phys_from_sample(sample)

    if model.mode == "mode2":
        return float(_production_ll_mode2(model, phys_args, phys_kw))
    if model.mode == "mode1":
        D_A = diag["D_A"]
        M_BH = diag["M_BH"]
        v_sys = diag["v_sys"]
        i0 = phys_args[8]
        var_v_hv = phys_args[15]
        sigma_a_floor2 = phys_args[16]
        r_est, _, _, _ = model._estimate_adaptive_r(
            D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
        return float(_production_ll_mode1(
            model, phys_args, phys_kw, jnp.asarray(r_est))["total"])
    raise ValueError(f"unsupported mode: {model.mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--tol", type=float, default=TOL_ABS)
    args = parser.parse_args()
    if not (args.capture or args.check):
        parser.error("pass --capture or --check")

    jax.config.update("jax_enable_x64", True)
    with open(CONFIG_PATH, "rb") as f:
        master_cfg = tomli.load(f)

    ll = {g: compute_ll(g, master_cfg) for g in GALAXIES}
    for g, v in ll.items():
        print(f"  {g:16s}  log L = {v:+.6f}")

    if args.capture:
        np.savez(BASELINE_PATH, **{g: np.asarray(v) for g, v in ll.items()})
        print(f"\nsaved baseline -> {BASELINE_PATH}")
        return

    ref = np.load(BASELINE_PATH)
    fail = False
    print()
    for g in GALAXIES:
        d = ll[g] - float(ref[g])
        status = "OK" if abs(d) <= args.tol else "FAIL"
        print(f"  {g:16s}  delta = {d:+.3e}   {status}")
        if abs(d) > args.tol:
            fail = True
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
