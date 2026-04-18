"""Time forward + gradient of ll_disk at the config init point.

Usage:
  python scripts/megamaser/bench_gpu_refactor.py --galaxies CGCG074-064 NGC4258
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import tomli

from candel.model.maser_convergence import (
    _production_ll_mode1, _production_ll_mode2, build_model,
)


def bench_one(galaxy, master_cfg, n_warmup=3, n_iter=20):
    overrides = {"mode": "mode1"} if galaxy == "NGC4258" else {}
    model = build_model(galaxy, master_cfg, **overrides)
    init = master_cfg["model"]["galaxies"][galaxy]["init"]
    sample = {k: np.asarray(v) for k, v in init.items()}
    phys_args, phys_kw, diag = model.phys_from_sample(sample)

    if model.mode == "mode2":
        D_A0 = jnp.asarray(diag["D_A"])

        def fwd(D_A):
            pa = list(phys_args)
            pa[2] = D_A
            groups = model._build_r_grids_mode2(
                D_A, phys_args[3], phys_args[4], phys_args[16],
                phys_args[8], phys_args[15])
            ll = model._eval_phi_marginal(groups, tuple(pa), phys_kw)
            return jnp.sum(ll)
        arg = D_A0
    else:
        D_A = diag["D_A"]
        M_BH = diag["M_BH"]
        v_sys = diag["v_sys"]
        i0 = phys_args[8]
        var_v_hv = phys_args[15]
        sigma_a_floor2 = phys_args[16]
        r_est, _, _, _ = model._estimate_adaptive_r(
            D_A, M_BH, v_sys, sigma_a_floor2, i0, var_v_hv)
        r0 = jnp.asarray(r_est)

        def fwd(r_ang_in):
            groups = []
            if model._n_sys > 0:
                groups.append(("sys", model._idx_sys,
                               r_ang_in[model._idx_sys], None))
            if model._n_red > 0:
                groups.append(("red", model._idx_red,
                               r_ang_in[model._idx_red], None))
            if model._n_blue > 0:
                groups.append(("blue", model._idx_blue,
                               r_ang_in[model._idx_blue], None))
            ll = model._eval_phi_marginal(groups, phys_args, phys_kw)
            return jnp.sum(ll)
        arg = r0

    jit_fwd = jax.jit(fwd)
    jit_grad = jax.jit(jax.grad(fwd))
    for _ in range(n_warmup):
        jit_fwd(arg).block_until_ready()
        jit_grad(arg).block_until_ready()
    t0 = time.time()
    for _ in range(n_iter):
        jit_fwd(arg).block_until_ready()
    t_fwd = (time.time() - t0) / n_iter * 1000
    t0 = time.time()
    for _ in range(n_iter):
        jit_grad(arg).block_until_ready()
    t_grad = (time.time() - t0) / n_iter * 1000
    return dict(mode=model.mode, fwd_ms=t_fwd, grad_ms=t_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxies", nargs="+",
                        default=["CGCG074-064", "NGC5765b", "NGC4258"])
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    print(f"JAX backend: {jax.default_backend()}")
    with open("scripts/megamaser/config_maser.toml", "rb") as f:
        master_cfg = tomli.load(f)
    print(f"\n{'galaxy':<16s}  mode    fwd [ms]   grad [ms]")
    for g in args.galaxies:
        r = bench_one(g, master_cfg, n_iter=args.n_iter)
        print(f"{g:<16s}  {r['mode']}   {r['fwd_ms']:8.2f}   "
              f"{r['grad_ms']:8.2f}")


if __name__ == "__main__":
    main()
