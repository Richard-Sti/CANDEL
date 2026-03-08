#!/usr/bin/env python
"""Mock closure tests for combined TRGB + 2MTF model (two-group).

Usage:
    # Single mock (inspect output)
    python run_mock_TRGB_2MTF.py --single --seed 42

    # Batch of 10 mocks, 5 in parallel
    python run_mock_TRGB_2MTF.py --n-mocks 10 --n-procs 5

    # Fix H0 and infer a_TFR
    python run_mock_TRGB_2MTF.py --n-mocks 10 --n-procs 5 --fix-H0
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
from candel.mock.dev.TRGB_2MTF_mock import (DEFAULT_ANCHORS,  # noqa: E402
                                             DEFAULT_TRUE_PARAMS,
                                             gen_TRGB_2MTF_mock)

TRACKED_PARAMS = [
    "H0", "M_TRGB", "sigma_int_TRGB",
    "a_TFR", "b_TFR", "sigma_int_TFR",
    "sigma_v",
    "Vext_mag", "Vext_ell", "Vext_b",
    "eta_mean", "eta_std",
    "beta", "b1",
    "mu_LMC", "mu_N4258",
]

BASE_CONFIG = os.path.join(REPO_ROOT, "scripts/runs/dev/config_EDD_TRGB_2MTF.toml")


def _write_tmp_config(config):
    f = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, f)
    f.close()
    return f.name


def make_mock_config(seed, num_warmup=500, num_samples=500,
                     fix_H0=False, fix_a_TFR=False, true_params=None,
                     rmax_trgb=40.0, rmax_tfr=200.0,
                     mag_lim_TRGB=None, mag_lim_TRGB_width=None,
                     mag_lim_TFR=11.25,
                     eta_min_sel=None, eta_max_sel=None,
                     infer_selection=True, n_tfr=0):
    """Build config dict for mock inference."""
    config = candel.load_config(BASE_CONFIG, replace_los_prior=False)
    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}

    config["model"]["use_reconstruction"] = False
    config["model"]["which_bias"] = "linear"
    config["model"]["which_selection_trgb"] = "TRGB_magnitude"
    config["model"]["n_gauss_hermite"] = 5
    config["model"]["r_limits_malmquist"] = [
        0.01, max(rmax_trgb, rmax_tfr)]

    # TFR selection
    if n_tfr > 0:
        config["model"]["which_selection_tfr"] = "magnitude"
        config["model"]["mag_lim_TFR"] = mag_lim_TFR
        if eta_min_sel is not None:
            config["model"]["eta_min_sel"] = eta_min_sel
        if eta_max_sel is not None:
            config["model"]["eta_max_sel"] = eta_max_sel
    else:
        config["model"]["which_selection_tfr"] = None

    if infer_selection:
        config["model"]["mag_lim_TRGB"] = "infer"
        config["model"]["mag_lim_TRGB_width"] = "infer"
    else:
        if mag_lim_TRGB is not None:
            config["model"]["mag_lim_TRGB"] = mag_lim_TRGB
        if mag_lim_TRGB_width is not None:
            config["model"]["mag_lim_TRGB_width"] = mag_lim_TRGB_width

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

    priors["c_TFR"] = {"dist": "delta", "value": 0.0}

    priors["beta"] = {"dist": "delta", "value": 0.0}

    return config


def run_one_mock(args):
    """Generate one mock, run inference, compute standardised biases."""
    (seed, true_params, mock_kwargs, fix_H0, fix_a_TFR,
     num_warmup, num_samples, infer_selection) = args

    try:
        from candel.model.dev.model_H0_TRGB_2MTF import TRGB2MTFModel

        tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
        data_trgb, data_tfr, tp_out, n_parent = gen_TRGB_2MTF_mock(
            seed=seed, true_params=tp, verbose=False, **mock_kwargs)
        tp_out["mu_LMC"] = DEFAULT_ANCHORS["mu_LMC"]
        tp_out["mu_N4258"] = DEFAULT_ANCHORS["mu_N4258"]
        n_hosts_trgb = len(data_trgb["mag_obs"])
        n_hosts_tfr = len(data_tfr["mag"])

        Vx, Vy, Vz = tp_out["Vext_x"], tp_out["Vext_y"], tp_out["Vext_z"]
        Vmag, Vell, Vb = radec_cartesian_to_galactic(
            np.array([Vx]), np.array([Vy]), np.array([Vz]))
        tp_out["Vext_mag"] = float(Vmag)
        tp_out["Vext_ell"] = float(Vell)
        tp_out["Vext_b"] = float(Vb)

        config = make_mock_config(
            seed, num_warmup=num_warmup, num_samples=num_samples,
            fix_H0=fix_H0, fix_a_TFR=fix_a_TFR, true_params=tp,
            rmax_trgb=mock_kwargs.get("rmax_trgb", 40.0),
            rmax_tfr=mock_kwargs.get("rmax_tfr", 200.0),
            mag_lim_TRGB=mock_kwargs.get("mag_lim_TRGB"),
            mag_lim_TRGB_width=mock_kwargs.get("mag_lim_TRGB_width"),
            mag_lim_TFR=mock_kwargs.get("mag_lim_TFR", 11.25),
            eta_min_sel=mock_kwargs.get("eta_min_sel"),
            eta_max_sel=mock_kwargs.get("eta_max_sel"),
            infer_selection=infer_selection,
            n_tfr=n_hosts_tfr)
        tmp = _write_tmp_config(config)

        try:
            model = TRGB2MTFModel(tmp, data_trgb, data_tfr)
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

        return biases, n_hosts_trgb, n_hosts_tfr
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def _print_summary(biases_dict, n_skipped, n_mocks):
    print(f"\n{'='*60}")
    print("Summary: mean standardised bias +/- std")
    print("=" * 60)
    print(f"{'param':<20s}  {'bias':>20s}  {'KS p-value':>10s}")
    print("-" * 55)
    for p in TRACKED_PARAMS:
        if p in biases_dict and len(biases_dict[p]) > 0:
            b = np.array(biases_dict[p])
            pval = kstest(b, "norm").pvalue if len(b) >= 3 else float("nan")
            print(f"{p:<20s}  "
                  f"{f'{b.mean():+.3f} +/- {b.std():.3f}':>20s}  "
                  f"{pval:>10.3f}")
    if n_skipped:
        print(f"\n[WARN] {n_skipped}/{n_mocks} mocks failed")


def run_batch(n_mocks, n_procs, true_params, mock_kwargs,
              fix_H0, fix_a_TFR, num_warmup, num_samples, outdir,
              infer_selection):
    os.makedirs(outdir, exist_ok=True)

    mode = "fix_H0" if fix_H0 else ("fix_a_TFR" if fix_a_TFR else "free")
    print(f"[INFO] Running {n_mocks} TRGB+2MTF mocks ({mode}), "
          f"{n_procs} processes")
    print(f"[INFO] num_warmup={num_warmup}, num_samples={num_samples}")
    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
    for p, v in tp.items():
        print(f"         {p:<20s} = {v}")

    t0 = time.time()

    args_list = [
        (seed, true_params, mock_kwargs, fix_H0, fix_a_TFR,
         num_warmup, num_samples, infer_selection)
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
            result = run_one_mock(args)
            if result is None:
                n_skipped += 1
            else:
                results.append(result)
            now = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - t0
            print(f"[INFO {now}] {i+1}/{n_mocks} done ({elapsed:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\n[INFO] Done in {elapsed:.0f}s "
          f"({elapsed / max(n_mocks, 1):.1f}s/mock)")

    biases = {p: [] for p in TRACKED_PARAMS}
    n_hosts_trgb_list = []
    n_hosts_tfr_list = []
    for b, n_trgb, n_tfr in results:
        n_hosts_trgb_list.append(n_trgb)
        n_hosts_tfr_list.append(n_tfr)
        for p in TRACKED_PARAMS:
            if p in b:
                biases[p].append(b[p])

    save_dict = {}
    for p in TRACKED_PARAMS:
        if biases[p]:
            save_dict[p] = np.array(biases[p])
    save_dict["n_mocks"] = np.array(n_mocks)
    save_dict["n_skipped"] = np.array(n_skipped)
    save_dict["n_hosts_trgb"] = np.array(n_hosts_trgb_list)
    save_dict["n_hosts_tfr"] = np.array(n_hosts_tfr_list)

    outfile = os.path.join(outdir, f"mock_TRGB_2MTF_biases_{mode}.npz")
    np.savez(outfile, **save_dict)
    print(f"[INFO] Saved to {outfile}")

    _print_summary(biases, n_skipped, n_mocks)


def run_single(seed, true_params, mock_kwargs, fix_H0, fix_a_TFR,
               num_warmup, num_samples, infer_selection):
    """Generate one mock, run inference, print diagnostics."""
    from candel.model.dev.model_H0_TRGB_2MTF import TRGB2MTFModel

    tp = {**DEFAULT_TRUE_PARAMS, **(true_params or {})}
    data_trgb, data_tfr, tp_out, n_parent = gen_TRGB_2MTF_mock(
        seed=seed, true_params=tp, verbose=True, **mock_kwargs)
    tp_out["mu_LMC"] = DEFAULT_ANCHORS["mu_LMC"]
    tp_out["mu_N4258"] = DEFAULT_ANCHORS["mu_N4258"]

    Vx, Vy, Vz = tp_out["Vext_x"], tp_out["Vext_y"], tp_out["Vext_z"]
    Vmag, Vell, Vb = radec_cartesian_to_galactic(
        np.array([Vx]), np.array([Vy]), np.array([Vz]))
    tp_out["Vext_mag"] = float(Vmag)
    tp_out["Vext_ell"] = float(Vell)
    tp_out["Vext_b"] = float(Vb)

    n_trgb = len(data_trgb["mag_obs"])
    n_tfr = len(data_tfr["mag"])
    n_overlap = int(data_trgb["has_TFR"].sum())
    print(f"\nMock catalog: {n_trgb} TRGB hosts "
          f"({n_overlap} overlap), {n_tfr} TFR-only hosts")
    print(f"TRGB mag_obs: [{data_trgb['mag_obs'].min():.2f}, "
          f"{data_trgb['mag_obs'].max():.2f}]")
    if n_tfr > 0:
        print(f"TFR mag:      [{data_tfr['mag'].min():.2f}, "
              f"{data_tfr['mag'].max():.2f}]")
        print(f"TFR eta:      [{data_tfr['eta'].min():.3f}, "
              f"{data_tfr['eta'].max():.3f}]")

    config = make_mock_config(
        seed, num_warmup=num_warmup, num_samples=num_samples,
        fix_H0=fix_H0, fix_a_TFR=fix_a_TFR, true_params=tp,
        rmax_trgb=mock_kwargs.get("rmax_trgb", 40.0),
        rmax_tfr=mock_kwargs.get("rmax_tfr", 200.0),
        mag_lim_TRGB=mock_kwargs.get("mag_lim_TRGB"),
        mag_lim_TRGB_width=mock_kwargs.get("mag_lim_TRGB_width"),
        mag_lim_TFR=mock_kwargs.get("mag_lim_TFR", 11.25),
        eta_min_sel=mock_kwargs.get("eta_min_sel"),
        eta_max_sel=mock_kwargs.get("eta_max_sel"),
        infer_selection=infer_selection,
        n_tfr=n_tfr)
    tmp = _write_tmp_config(config)

    try:
        model = TRGB2MTFModel(tmp, data_trgb, data_tfr)
        samples = candel.run_H0_inference(
            model, save_samples=False, print_summary=True)

        print(f"\n{'='*60}")
        print("Standardised biases (posterior vs truth)")
        print("=" * 60)
        for param in TRACKED_PARAMS:
            if param in samples:
                s = np.asarray(samples[param])
                bias = (s.mean() - tp_out[param]) / s.std()
                print(f"  {param:<20s}: {bias:+.2f}σ  "
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
    parser.add_argument("--n-trgb", type=int, default=300,
                        help="Number of TRGB hosts")
    parser.add_argument("--n-tfr", type=int, default=2000,
                        help="Number of TFR-only hosts")
    parser.add_argument("--overlap-fraction", type=float, default=0.1,
                        help="Fraction of TRGB hosts with TFR overlap")
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--outdir",
                        default=os.path.join(
                            REPO_ROOT, "results/mocks_TRGB_2MTF"))
    parser.add_argument("--fix-H0", action="store_true")
    parser.add_argument("--fix-a-TFR", action="store_true")
    parser.add_argument("--rmax-trgb", type=float, default=40.0)
    parser.add_argument("--rmax-tfr", type=float, default=200.0)
    parser.add_argument("--mag-lim-TRGB", type=float, default=25.0)
    parser.add_argument("--mag-lim-TRGB-width", type=float, default=0.75)
    parser.add_argument("--mag-lim-TFR", type=float, default=11.25)
    parser.add_argument("--eta-min-sel", type=float, default=None)
    parser.add_argument("--eta-max-sel", type=float, default=None)
    parser.add_argument("--infer-selection", action="store_true")

    args = parser.parse_args()

    true_params = None
    mock_kwargs = {
        "n_trgb": args.n_trgb,
        "n_tfr": args.n_tfr,
        "overlap_fraction": args.overlap_fraction,
        "rmax_trgb": args.rmax_trgb,
        "rmax_tfr": args.rmax_tfr,
        "mag_lim_TRGB": args.mag_lim_TRGB,
        "mag_lim_TRGB_width": args.mag_lim_TRGB_width,
        "mag_lim_TFR": args.mag_lim_TFR,
        "eta_min_sel": args.eta_min_sel,
        "eta_max_sel": args.eta_max_sel,
    }

    if args.single:
        run_single(args.seed, true_params, mock_kwargs,
                   args.fix_H0, args.fix_a_TFR,
                   args.num_warmup, args.num_samples,
                   args.infer_selection)
    else:
        run_batch(args.n_mocks, args.n_procs, true_params, mock_kwargs,
                  args.fix_H0, args.fix_a_TFR,
                  args.num_warmup, args.num_samples, args.outdir,
                  args.infer_selection)


if __name__ == "__main__":
    main()
