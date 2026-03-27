#!/usr/bin/env python
"""Batch mock closure tests for the full maser disk forward model."""
import argparse
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tomli_w
from scipy.stats import kstest, norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

import candel
from candel.mock.maser_disk_mock import (DEFAULT_TRUE_PARAMS,
                                          gen_maser_mock_like_cgcg074)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_CONFIG = os.path.join(REPO_ROOT, "scripts", "runs", "config_maser.toml")

TRACKED_PARAMS = ["H0", "D_c", "M_BH", "i0", "Omega0", "dOmega_dr",
                  "sigma_pec", "sigma_v_sys", "sigma_v_hv", "sigma_a_floor",
                  "A_thr", "sigma_det"]


def _write_tmp_config(config):
    f = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, f)
    f.close()
    return f.name


def run_one_mock(seed, num_warmup=1000, num_samples=500, quiet=True):
    from candel.model.model_H0_maser import MaserDiskModel

    data, tp = gen_maser_mock_like_cgcg074(
        seed=seed, verbose=not quiet)

    config = candel.load_config(BASE_CONFIG, replace_los_prior=False)
    config["inference"]["seed"] = seed
    config["inference"]["num_warmup"] = num_warmup
    config["inference"]["num_samples"] = num_samples
    config["inference"]["num_chains"] = 1
    # Set observed CMB velocity from mock truth
    v_cmb_obs = tp["v_sys"] + tp["v_helio_to_cmb"]
    config["model"]["v_cmb_obs"] = v_cmb_obs
    tmp = _write_tmp_config(config)

    # Init near truth for faster convergence.
    # D in mock is angular-diameter distance; convert to comoving.
    D_c_true = tp['D'] * (1 + tp['z_cosmo'])

    init_values = {
        'H0': jnp.array(tp['H0']),
        'sigma_pec': jnp.array(tp['sigma_pec']),
        'D_c': jnp.array(D_c_true),
        'M_BH': jnp.array(tp['M_BH']),
        'x0': jnp.array(tp['x0']),
        'y0': jnp.array(tp['y0']),
        'i0': jnp.array(tp['i0']),
        'Omega0': jnp.array(tp['Omega0']),
        'dOmega_dr': jnp.array(tp['dOmega_dr']),
        'di_dr': jnp.array(tp['di_dr']),
        'sigma_x_floor': jnp.array(tp['sigma_x_floor']),
        'sigma_y_floor': jnp.array(tp['sigma_y_floor']),
        'sigma_v_sys': jnp.array(tp['sigma_v_sys']),
        'sigma_v_hv': jnp.array(tp['sigma_v_hv']),
        'sigma_a_floor': jnp.array(tp['sigma_a_floor']),
        'A_thr': jnp.array(tp['A_thr']),
        'sigma_det': jnp.array(tp['sigma_det']),
    }

    try:
        if quiet:
            with open(os.devnull, "w") as _devnull, \
                    redirect_stdout(_devnull):
                model = MaserDiskModel(tmp, data)
                kernel = NUTS(model, max_tree_depth=8,
                              target_accept_prob=0.8,
                              init_strategy=init_to_value(values=init_values))
                mcmc = MCMC(kernel, num_warmup=num_warmup,
                            num_samples=num_samples,
                            num_chains=1, progress_bar=False)
                mcmc.run(random.PRNGKey(seed))
                samples = mcmc.get_samples()
                n_div = int(mcmc.get_extra_fields()['diverging'].sum())
        else:
            model = MaserDiskModel(tmp, data)
            kernel = NUTS(model, max_tree_depth=8,
                          target_accept_prob=0.8,
                          init_strategy=init_to_value(values=init_values))
            mcmc = MCMC(kernel, num_warmup=num_warmup,
                        num_samples=num_samples,
                        num_chains=1, progress_bar=True)
            mcmc.run(random.PRNGKey(seed))
            samples = mcmc.get_samples()
            mcmc.print_summary(exclude_deterministic=True)
            n_div = int(mcmc.get_extra_fields()['diverging'].sum())

        biases = {}
        for param in TRACKED_PARAMS:
            if param in samples:
                s = np.asarray(samples[param])
                if param == "D_c":
                    true_val = tp["D"] * (1 + tp["z_cosmo"])
                else:
                    true_val = tp[param]
                biases[param] = (s.mean() - true_val) / s.std()
    finally:
        os.unlink(tmp)

    return biases, n_div


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mocks", type=int, default=25)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--master-seed", type=int, default=12345)
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(REPO_ROOT, "results",
                                             "mocks_maser_disk"))
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.single:
        biases, n_div = run_one_mock(
            args.seed, num_warmup=args.num_warmup,
            num_samples=args.num_samples, quiet=False)
        print(f"\nDivergences: {n_div}")
        print("Standardised biases:")
        for p, b in biases.items():
            print(f"  {p}: {b:+.3f}")
        return

    rng = np.random.default_rng(args.master_seed)
    seeds = rng.integers(0, 2**31, size=args.n_mocks)
    all_biases = {p: [] for p in TRACKED_PARAMS}
    all_divs = []

    t0 = time.time()
    for i, seed in enumerate(seeds):
        t1 = time.time()
        try:
            biases, n_div = run_one_mock(
                int(seed), num_warmup=args.num_warmup,
                num_samples=args.num_samples)
            for p in TRACKED_PARAMS:
                if p in biases:
                    all_biases[p].append(biases[p])
            all_divs.append(n_div)
            dt = time.time() - t1
            elapsed = time.time() - t0
            b_str = "  ".join(f"{p}={biases.get(p, float('nan')):+.2f}"
                              for p in ["H0", "D", "M_BH"])
            print(f"[{i+1}/{args.n_mocks}] seed={seed} {dt:.0f}s "
                  f"(total {elapsed:.0f}s) div={n_div}  {b_str}",
                  flush=True)
        except Exception as e:
            print(f"[{i+1}/{args.n_mocks}] seed={seed} FAILED: {e}",
                  flush=True)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Final results ({args.n_mocks} mocks)")
    print(f"{'=' * 60}")
    print(f"{'Param':<15} {'N':>5} {'mean':>8} {'std':>8} {'KS p':>8}")
    print("-" * 50)
    for p in TRACKED_PARAMS:
        if all_biases[p]:
            b = np.array(all_biases[p])
            n = len(b)
            mean = np.mean(b)
            std = np.std(b)
            ks_stat, ks_p = kstest(b, 'norm')
            flag = " ***" if abs(mean) > 3 / np.sqrt(n) else ""
            print(f"{p:<15} {n:>5d} {mean:>+8.3f} {std:>8.3f} "
                  f"{ks_p:>8.3f}{flag}")
    print(f"\nMean divergences: {np.mean(all_divs):.0f}")

    # Save
    save_dict = {p: np.array(all_biases[p]) for p in TRACKED_PARAMS
                 if all_biases[p]}
    save_dict["n_divs"] = np.array(all_divs)
    outfile = os.path.join(args.outdir, "mock_maser_disk_biases.npz")
    np.savez(outfile, **save_dict)
    print(f"Saved to {outfile}")

    # Plot
    params = [p for p in ["H0", "D", "M_BH"] if all_biases[p]]
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4))
    if len(params) == 1:
        axes = [axes]
    x_grid = np.linspace(-4, 4, 200)
    for ax, p in zip(axes, params):
        b = np.array(all_biases[p])
        ax.hist(b, bins=15, density=True, alpha=0.7, edgecolor="black")
        ax.plot(x_grid, norm.pdf(x_grid), "r-", lw=2)
        ax.axvline(np.mean(b), color="blue", ls="--", lw=1.5,
                   label=f"mean={np.mean(b):.2f}")
        ax.set_xlabel(f"Standardised bias ({p})")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "mock_maser_disk_biases.png"),
                dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
