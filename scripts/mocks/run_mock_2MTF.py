#!/usr/bin/env python
"""Mock closure tests for EDD2MTFModel.

Usage:
    # Single mock (inspect output)
    python run_mock_2MTF.py --single --seed 42

    # Batch of 10 mocks, 5 in parallel
    python run_mock_2MTF.py --n-mocks 10 --n-procs 5

    # Fix H0 and infer zero-point
    python run_mock_2MTF.py --n-mocks 10 --n-procs 5 --fix-H0

    # Fix zero-point and infer H0
    python run_mock_2MTF.py --n-mocks 10 --n-procs 5 --fix-a-TFR

    # With reconstruction field
    python run_mock_2MTF.py --n-mocks 10 --n-procs 5 --fix-H0 --use-field
"""
import argparse
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import tomli_w
from scipy.stats import kstest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import candel  # noqa: E402
from candel.inference import radec_cartesian_to_galactic  # noqa: E402
from candel.mock import gen_EDD_2MTF_mock  # noqa: E402
from candel.mock.EDD_2MTF_mock import DEFAULT_TRUE_PARAMS  # noqa: E402

TRACKED_PARAMS = ["H0", "a_TFR", "b_TFR", "sigma_int", "sigma_v",
                   "Vext_mag", "Vext_ell", "Vext_b",
                   "eta_mean", "eta_std",
                   "beta", "b1"]

BASE_CONFIG = os.path.join(REPO_ROOT, "scripts/runs/config_EDD_2MTF.toml")


def _write_tmp_config(config):
    f = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, f)
    f.close()
    return f.name


def make_mock_config(seed, nsamples=500, num_warmup=500, num_samples=500,
                     fix_H0=False, fix_a_TFR=False, true_params=None,
                     use_field=False, rmax=150.0):
    """Build config dict for mock inference."""
    config = candel.load_config(BASE_CONFIG, replace_los_prior=False)
    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}

    config["model"]["use_reconstruction"] = use_field
    config["model"]["which_bias"] = "linear"
    config["model"]["marginalize_eta"] = True
    config["model"]["n_gauss_hermite"] = 5
    config["model"]["n_eta_sel_grid"] = 51
    config["model"]["r_limits_malmquist"] = [0.01, rmax]

    config["inference"]["seed"] = seed
    config["inference"]["num_warmup"] = num_warmup
    config["inference"]["num_samples"] = num_samples
    config["inference"]["num_chains"] = 1
    config["inference"]["chain_method"] = "sequential"

    priors = config["model"]["priors"]

    if fix_H0:
        priors["H0"] = {"dist": "delta", "value": tp["H0"]}
    if fix_a_TFR:
        priors["a_TFR"] = {"dist": "delta", "value": tp["a_TFR"]}

    # Always fix c_TFR for mocks
    priors["c_TFR"] = {"dist": "delta", "value": 0.0}

    config["io"]["load_host_los"] = use_field
    config["io"]["load_rand_los"] = False

    if not use_field:
        priors["beta"] = {"dist": "delta", "value": 0.0}
    else:
        priors["beta"] = {"dist": "uniform", "low": 0.0, "high": 2.0}
        priors["b1"] = {"dist": "uniform", "low": 0.1, "high": 5.0}

    return config


def run_one_mock(args):
    """Generate one mock, run inference, compute standardised biases."""
    (seed, true_params, mock_kwargs, fix_H0, fix_a_TFR,
     num_warmup, num_samples, use_field) = args

    try:
        tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
        data, tp_out, n_parent = gen_EDD_2MTF_mock(
            seed=seed, true_params=tp, verbose=False, **mock_kwargs)
        n_hosts = len(data["mag"])

        # Convert Vext to galactic for comparison with postprocessed
        Vx, Vy, Vz = tp_out["Vext_x"], tp_out["Vext_y"], tp_out["Vext_z"]
        Vmag, Vell, Vb = radec_cartesian_to_galactic(
            np.array([Vx]), np.array([Vy]), np.array([Vz]))
        tp_out["Vext_mag"] = float(Vmag)
        tp_out["Vext_ell"] = float(Vell)
        tp_out["Vext_b"] = float(Vb)

        config = make_mock_config(
            seed, num_warmup=num_warmup, num_samples=num_samples,
            fix_H0=fix_H0, fix_a_TFR=fix_a_TFR, true_params=tp,
            use_field=use_field,
            rmax=mock_kwargs.get("rmax", 150.0))
        tmp = _write_tmp_config(config)

        try:
            model = candel.model.EDD2MTFModel(tmp, data)
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
                samples = candel.run_H0_inference(
                    model, save_samples=False, print_summary=False)

            biases = {}
            for param in TRACKED_PARAMS:
                if param in samples:
                    s = np.asarray(samples[param])
                    biases[param] = (s.mean() - tp_out[param]) / s.std()
        finally:
            os.unlink(tmp)

        return biases, n_hosts, n_parent
    except Exception:
        return None


def _print_summary(biases_dict, n_skipped, n_mocks):
    print(f"\n{'='*60}")
    print("Summary: mean standardised bias +/- std")
    print("=" * 60)
    print(f"{'param':<18s}  {'bias':>20s}  {'KS p-value':>10s}")
    print("-" * 55)
    for p in TRACKED_PARAMS:
        if p in biases_dict and len(biases_dict[p]) > 0:
            b = np.array(biases_dict[p])
            pval = kstest(b, "norm").pvalue if len(b) >= 3 else float("nan")
            print(f"{p:<18s}  "
                  f"{f'{b.mean():+.3f} +/- {b.std():.3f}':>20s}  "
                  f"{pval:>10.3f}")
    if n_skipped:
        print(f"\n[WARN] {n_skipped}/{n_mocks} mocks failed")


def run_batch(n_mocks, n_procs, true_params, mock_kwargs,
              fix_H0, fix_a_TFR, num_warmup, num_samples, outdir,
              use_field):
    os.makedirs(outdir, exist_ok=True)

    mode = "fix_H0" if fix_H0 else ("fix_a_TFR" if fix_a_TFR else "free")
    print(f"[INFO] Running {n_mocks} 2MTF mocks ({mode}), "
          f"{n_procs} processes")
    print(f"[INFO] num_warmup={num_warmup}, num_samples={num_samples}")
    print(f"[INFO] use_field={use_field}")
    print(f"[INFO] True parameters:")
    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
    for p, v in tp.items():
        print(f"         {p:<15s} = {v}")

    t0 = time.time()

    args_list = [
        (seed, true_params, mock_kwargs, fix_H0, fix_a_TFR,
         num_warmup, num_samples, use_field)
        for seed in range(n_mocks)
    ]

    results = []
    n_skipped = 0

    if n_procs > 1:
        with Pool(n_procs) as pool:
            for i, result in enumerate(pool.imap_unordered(
                    run_one_mock, args_list)):
                now = datetime.now().strftime("%H:%M:%S")
                elapsed = time.time() - t0
                if result is None:
                    n_skipped += 1
                    print(f"[WARN {now}] {i+1}/{n_mocks} failed "
                          f"({elapsed:.0f}s)", flush=True)
                else:
                    results.append(result)
                    print(f"[INFO {now}] {i+1}/{n_mocks} done "
                          f"({elapsed:.0f}s)", flush=True)
    else:
        for i, args in enumerate(args_list):
            try:
                result = run_one_mock(args)
                results.append(result)
            except Exception as e:
                n_skipped += 1
                print(f"[WARN] Mock {i} failed: {e}")
            now = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - t0
            print(f"[INFO {now}] {i+1}/{n_mocks} done ({elapsed:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\n[INFO] Done in {elapsed:.0f}s "
          f"({elapsed / max(n_mocks, 1):.1f}s/mock)")

    # Collect biases
    biases = {p: [] for p in TRACKED_PARAMS}
    n_hosts_list = []
    for b, n_hosts, n_parent in results:
        n_hosts_list.append(n_hosts)
        for p in TRACKED_PARAMS:
            if p in b:
                biases[p].append(b[p])

    save_dict = {}
    for p in TRACKED_PARAMS:
        if biases[p]:
            save_dict[p] = np.array(biases[p])
    save_dict["n_mocks"] = np.array(n_mocks)
    save_dict["n_skipped"] = np.array(n_skipped)
    save_dict["n_hosts"] = np.array(n_hosts_list)

    tag = f"_field" if use_field else ""
    outfile = os.path.join(outdir, f"mock_2MTF_biases_{mode}{tag}.npz")
    np.savez(outfile, **save_dict)
    print(f"[INFO] Saved to {outfile}")

    _print_summary(biases, n_skipped, n_mocks)


def run_single(seed, true_params, mock_kwargs, fix_H0, fix_a_TFR,
               num_warmup, num_samples, use_field):
    """Generate one mock, run inference, print diagnostics."""
    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
    data, tp_out, n_parent = gen_EDD_2MTF_mock(
        seed=seed, true_params=tp, verbose=True, **mock_kwargs)

    # Convert Vext to galactic coordinates for comparison
    Vx, Vy, Vz = tp_out["Vext_x"], tp_out["Vext_y"], tp_out["Vext_z"]
    Vmag, Vell, Vb = radec_cartesian_to_galactic(
        np.array([Vx]), np.array([Vy]), np.array([Vz]))
    tp_out["Vext_mag"] = float(Vmag)
    tp_out["Vext_ell"] = float(Vell)
    tp_out["Vext_b"] = float(Vb)

    n = len(data["mag"])
    print(f"\nMock 2MTF catalog: {n} hosts (parent: {n_parent})")
    print(f"mag: [{data['mag'].min():.2f}, {data['mag'].max():.2f}]")
    print(f"eta: [{data['eta'].min():.3f}, {data['eta'].max():.3f}]")
    print(f"czcmb: [{data['czcmb'].min():.0f}, {data['czcmb'].max():.0f}]")
    if use_field:
        print(f"LOS density shape: {data['host_los_density'].shape}")

    config = make_mock_config(
        seed, num_warmup=num_warmup, num_samples=num_samples,
        fix_H0=fix_H0, fix_a_TFR=fix_a_TFR, true_params=tp,
        use_field=use_field,
        rmax=mock_kwargs.get("rmax", 150.0))
    tmp = _write_tmp_config(config)

    try:
        model = candel.model.EDD2MTFModel(tmp, data)
        samples = candel.run_H0_inference(
            model, save_samples=False, print_summary=True)

        print(f"\n{'='*60}")
        print("Standardised biases (posterior vs truth)")
        print("=" * 60)
        for param in TRACKED_PARAMS:
            if param in samples:
                s = np.asarray(samples[param])
                bias = (s.mean() - tp_out[param]) / s.std()
                print(f"  {param:<18s}: {bias:+.2f}σ  "
                      f"(posterior {s.mean():.3f} ± {s.std():.3f}, "
                      f"true {tp_out[param]:.3f})")
    finally:
        os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--single", action="store_true",
                        help="Run single mock with diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mocks", type=int, default=10)
    parser.add_argument("--n-procs", type=int, default=5)
    parser.add_argument("--nsamples", type=int, default=500,
                        help="Number of mock hosts")
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--outdir",
                        default=os.path.join(REPO_ROOT, "results/mocks_2MTF"))
    parser.add_argument("--fix-H0", action="store_true",
                        help="Fix H0 to true value, infer a_TFR")
    parser.add_argument("--fix-a-TFR", action="store_true",
                        help="Fix a_TFR to true value, infer H0")
    parser.add_argument("--rmax", type=float, default=150.0,
                        help="Maximum mock distance [Mpc]")

    # Field-based mock options
    parser.add_argument("--use-field", action="store_true",
                        help="Enable field-based distance sampling")
    parser.add_argument("--field-name", type=str, default="Carrick2015",
                        help="Reconstruction field name")
    parser.add_argument("--beta", type=float,
                        default=DEFAULT_TRUE_PARAMS["beta"])
    parser.add_argument("--b1", type=float,
                        default=DEFAULT_TRUE_PARAMS["b1"])

    args = parser.parse_args()

    true_params = None  # use defaults
    if args.beta != DEFAULT_TRUE_PARAMS["beta"] or \
       args.b1 != DEFAULT_TRUE_PARAMS["b1"]:
        true_params = {"beta": args.beta, "b1": args.b1}

    mock_kwargs = {
        "nsamples": args.nsamples,
        "rmax": args.rmax,
        "mag_lim": 11.25,
        "eta_min_sel": -0.1,
        "eta_max_sel": 0.2,
    }

    # Set up field loader if requested
    if args.use_field:
        from candel.field import name2field_loader
        config = candel.load_config(BASE_CONFIG, replace_los_prior=False)
        field_config = config["io"]["reconstruction_main"][args.field_name]
        loader_cls = name2field_loader(args.field_name)
        field_loader = loader_cls(**field_config)
        mock_kwargs["field_loader"] = field_loader

    if args.single:
        run_single(args.seed, true_params, mock_kwargs,
                   args.fix_H0, args.fix_a_TFR,
                   args.num_warmup, args.num_samples,
                   args.use_field)
    else:
        run_batch(args.n_mocks, args.n_procs, true_params, mock_kwargs,
                  args.fix_H0, args.fix_a_TFR,
                  args.num_warmup, args.num_samples, args.outdir,
                  args.use_field)


if __name__ == "__main__":
    main()
