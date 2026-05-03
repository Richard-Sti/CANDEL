#!/usr/bin/env python
"""Batch mock runner for TRGB H0 model closure tests (MPI version).

Generates synthetic TRGB data and runs TRGBModel inference,
checking that true parameters are recovered without bias.

Usage:
    # Single mock (no MPI, inspect the sample)
    python mock_TRGB.py --single --seed 42

    # MPI batch run
    mpirun -np 8 python mock_TRGB.py --n-mocks 100 --outdir results/mocks
"""
import argparse
import os
import signal
import sys
import tempfile
import time
from contextlib import redirect_stdout
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tomli_w
from scipy.stats import kstest

import candel
from candel.mock import gen_TRGB_mock
from candel.mock.TRGB_mock import DEFAULT_ANCHORS, DEFAULT_TRUE_PARAMS
from candel.util import results_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TRACKED_PARAMS = ["H0", "M_TRGB", "sigma_int", "sigma_v",
                  "Vext_mag", "Vext_phi", "Vext_cos_theta",
                  "beta", "b1", "mu_LMC", "mu_N4258",
                  "mag_lim_TRGB", "mag_lim_TRGB_width"]

PERIODIC_PARAMS = {"Vext_phi": 2 * np.pi}

TAG_WORK = 1
TAG_RESULT = 2
TAG_DONE = 3

_DENSITY_3D_CACHE = {}


def _safe_tag(value):
    """Return a filesystem-friendly tag component."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))


def _mode_tag(which_selection, use_field, field_name, infer_selection):
    """Tag output files by the mock/inference mode."""
    field = f"field_{_safe_tag(field_name)}" if use_field else "nofield"
    selection = "infersel" if infer_selection else "fixedsel"
    return "_".join([_safe_tag(which_selection), field, selection])


def _expected_mpi_tasks_from_env():
    """Return scheduler-advertised MPI tasks, or 1 when not allocated."""
    for name in ("SLURM_NTASKS", "SLURM_NPROCS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE"):
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    return 1


def _standardised_bias(samples, true_val, param):
    """Standardised bias, handling periodic parameters."""
    if param in PERIODIC_PARAMS:
        period = PERIODIC_PARAMS[param]
        delta = (samples - true_val + period / 2) % period - period / 2
        return delta.mean() / delta.std()
    return (samples.mean() - true_val) / samples.std()


def _rhat_warnings(diagnostics, threshold):
    """Return tracked parameters whose NumPyro R-hat exceeds threshold."""
    warnings = {}
    if not diagnostics:
        return warnings
    for param in TRACKED_PARAMS:
        stats = diagnostics.get(param)
        if not stats or "r_hat" not in stats:
            continue
        rhat = float(stats["r_hat"])
        if np.isfinite(rhat) and rhat > threshold:
            warnings[param] = rhat
    return warnings


def _format_rhat_warnings(warnings):
    """Format R-hat warnings for log output."""
    return ", ".join(f"{p}={v:.3f}" for p, v in sorted(warnings.items()))


def _write_tmp_config(config):
    f = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, f)
    f.close()
    return f.name


def make_mock_config(base_config_path, seed, num_warmup=500,
                     num_samples=500, which_selection="TRGB_magnitude",
                     mag_lim=None, mag_lim_width=None,
                     cz_lim=None, cz_lim_width=None,
                     infer_selection=True, use_field=False, rmax=40.0,
                     num_chains=1):
    """Build a config dict for mock inference."""
    config = candel.load_config(base_config_path, replace_los_prior=False)

    config["model"]["use_reconstruction"] = use_field
    config["model"]["which_selection"] = which_selection
    # Match integration range to mock's distance range to avoid
    # extrapolation artefacts in the LOSInterpolator.
    config["model"]["r_limits_malmquist"] = [0.01, rmax]
    config["model"]["num_points_malmquist"] = 1001

    if which_selection == "TRGB_magnitude":
        if infer_selection:
            config["model"]["mag_lim_TRGB"] = "infer"
            config["model"]["mag_lim_TRGB_width"] = "infer"
        else:
            if mag_lim is not None:
                config["model"]["mag_lim_TRGB"] = mag_lim
            if mag_lim_width is not None:
                config["model"]["mag_lim_TRGB_width"] = mag_lim_width
    elif which_selection == "redshift":
        if infer_selection:
            config["model"]["cz_lim_selection"] = "infer"
            config["model"]["cz_lim_selection_width"] = "infer"
        else:
            if cz_lim is not None:
                config["model"]["cz_lim_selection"] = cz_lim
            if cz_lim_width is not None:
                config["model"]["cz_lim_selection_width"] = cz_lim_width

    # When not using field, fix beta prior to delta(0)
    if not use_field:
        config["model"]["priors"]["beta"] = {"dist": "delta", "value": 0.0}

    config["inference"]["seed"] = seed
    config["inference"]["num_warmup"] = num_warmup
    config["inference"]["num_samples"] = num_samples
    config["inference"]["num_chains"] = num_chains
    config["inference"]["chain_method"] = "sequential"

    return config


def _load_density_3d_data(config, field_name):
    """Load cached 3D density data used by reconstruction integrals."""
    if not config["model"].get("use_reconstruction", False):
        return None
    if config["model"].get("which_selection") is None:
        return None
    if field_name is None:
        raise ValueError("`field_name` is required for field-based mocks.")

    from candel.pvdata.data import _load_volume_data_for_H0

    recon = config.get("io", {}).get("reconstruction_main", {})
    field_kwargs = recon.get(field_name)
    if field_kwargs is None:
        raise ValueError(
            f"No `io.reconstruction_main.{field_name}` configuration found.")

    which_selection = config["model"]["which_selection"]
    load_velocity = which_selection == "redshift"
    key = (
        field_name,
        repr(sorted(field_kwargs.items())),
        config["model"].get("which_bias", "linear"),
        config["model"].get("Om", config["model"].get("Om0", 0.3)),
        config["model"].get("selection_integral_grid_radius"),
        config["model"].get("density_3d_downsample", 1),
        config["model"].get("selection_integral_geometry", "sphere"),
        load_velocity,
    )
    if key not in _DENSITY_3D_CACHE:
        _DENSITY_3D_CACHE[key] = _load_volume_data_for_H0(
            field_name, field_kwargs, field_indices=[0],
            galaxy_bias=config["model"].get("which_bias", "linear"),
            Om0=config["model"].get("Om", config["model"].get("Om0", 0.3)),
            subcube_radius=config["model"].get(
                "selection_integral_grid_radius"),
            downsample=config["model"].get("density_3d_downsample", 1),
            load_velocity=load_velocity,
            geometry=config["model"].get(
                "selection_integral_geometry", "sphere"),
            cache_dir=config.get("io", {}).get("field_cache_dir"),
            cache_enabled=config["model"].get(
                "density_3d_cache_enabled", True))
    return _DENSITY_3D_CACHE[key]


def run_one_mock(seed, base_config_path, true_params, mock_kwargs,
                 num_warmup=500, num_samples=500,
                 which_selection="TRGB_magnitude",
                 infer_selection=True, use_field=False, field_name=None,
                 quiet=True, progress_bar=False, num_chains=1,
                 rhat_threshold=1.05):
    """Generate one mock, run inference, compute standardised biases."""
    config = make_mock_config(
        base_config_path, seed, num_warmup=num_warmup,
        num_samples=num_samples,
        which_selection=which_selection,
        mag_lim=mock_kwargs.get("mag_lim"),
        mag_lim_width=mock_kwargs.get("mag_lim_width"),
        cz_lim=mock_kwargs.get("cz_lim"),
        cz_lim_width=mock_kwargs.get("cz_lim_width"),
        infer_selection=infer_selection,
        use_field=use_field,
        rmax=mock_kwargs.get("rmax", 40.0),
        num_chains=num_chains)
    density_3d_data = _load_density_3d_data(
        config, field_name) if use_field else None
    data, tp, n_parent = gen_TRGB_mock(
        seed=seed, true_params=true_params, verbose=not quiet,
        density_3d_data=density_3d_data, **mock_kwargs)
    mock_diag = {
        "mag_obs": np.asarray(data["mag_obs"]),
        "czcmb": np.asarray(data["czcmb"]),
    }
    tp["mu_LMC"] = DEFAULT_ANCHORS["mu_LMC"]
    tp["mu_N4258"] = DEFAULT_ANCHORS["mu_N4258"]
    _ra, _dec = candel.galactic_to_radec(tp["Vext_ell"], tp["Vext_b"])
    tp["Vext_phi"] = np.deg2rad(_ra)
    tp["Vext_cos_theta"] = np.sin(np.deg2rad(_dec))
    if mock_kwargs.get("mag_lim") is not None:
        tp["mag_lim_TRGB"] = mock_kwargs["mag_lim"]
    if mock_kwargs.get("mag_lim_width") is not None:
        tp["mag_lim_TRGB_width"] = mock_kwargs["mag_lim_width"]
    n_hosts = len(data["mag_obs"])

    tmp = _write_tmp_config(config)

    try:
        if quiet:
            with open(os.devnull, "w") as _devnull, \
                    redirect_stdout(_devnull):
                model = candel.model.TRGBModel(tmp, data)
                samples, diagnostics = candel.run_H0_inference(
                    model, save_samples=False, print_summary=False,
                    progress_bar=progress_bar, return_diagnostics=True)
        else:
            model = candel.model.TRGBModel(tmp, data)
            samples, diagnostics = candel.run_H0_inference(
                model, save_samples=False, print_summary=True,
                return_diagnostics=True)

        # Reconstruct raw Vext params from postprocessed ell/b
        if "Vext_ell" in samples and "Vext_b" in samples:
            ra, dec = candel.galactic_to_radec(
                np.asarray(samples["Vext_ell"]),
                np.asarray(samples["Vext_b"]))
            samples["Vext_phi"] = np.deg2rad(ra)
            samples["Vext_cos_theta"] = np.sin(np.deg2rad(dec))

        biases = {}
        for param in TRACKED_PARAMS:
            if param in samples:
                s = np.asarray(samples[param])
                biases[param] = _standardised_bias(s, tp[param], param)
    finally:
        os.unlink(tmp)

    return biases, n_hosts, n_parent, mock_diag, _rhat_warnings(
        diagnostics, rhat_threshold)


# ---- MPI master/worker ---------------------------------------------------

def master(comm, n_workers, config_info):
    """Rank 0: distribute seeds to workers and collect results."""
    from mpi4py import MPI

    n_mocks = config_info["n_mocks"]
    master_seed = config_info["master_seed"]
    true_params = config_info["true_params"]
    outdir = config_info["outdir"]
    mode_tag = config_info["mode_tag"]

    t0 = time.time()
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31, size=n_mocks).tolist()
    results = []
    n_sent = 0
    n_done = 0
    n_ok = 0
    n_skipped = 0
    running_biases = {p: [] for p in TRACKED_PARAMS}
    active_jobs = {}
    status = MPI.Status()

    for rank in range(1, n_workers + 1):
        if n_sent < n_mocks:
            job_num = n_sent + 1
            comm.send(seeds[n_sent], dest=rank, tag=TAG_WORK)
            active_jobs[rank] = job_num
            n_sent += 1
        else:
            comm.send(None, dest=rank, tag=TAG_DONE)

    print(f"[INFO] Dispatched {n_sent} initial jobs to {n_workers} workers.",
          flush=True)

    while n_done < n_mocks:
        result = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT,
                           status=status)
        source = status.Get_source()
        job_num = active_jobs.pop(source, None)
        n_done += 1

        now = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - t0
        job_label = (f"job {job_num}/{n_mocks}"
                     if job_num is not None else "job ?")
        next_label = (f"rank {source} starting job {n_sent + 1}/{n_mocks}"
                      if n_sent < n_mocks
                      else f"rank {source} has no more jobs")
        if result is None:
            n_skipped += 1
            print(f"[WARN {now}] {n_done}/{n_mocks} done: rank {source} "
                  f"timed out on {job_label}; {next_label} "
                  f"({elapsed:.0f}s total)", flush=True)
        else:
            results.append(result)
            n_ok += 1
            b, _, _, _, rhat_warnings, dt = result
            for p in TRACKED_PARAMS:
                if p in b:
                    running_biases[p].append(b[p])
            print(f"[INFO {now}] {n_done}/{n_mocks} done: rank {source} "
                  f"finished {job_label}; {next_label} "
                  f"({dt:.0f}s worker, {elapsed:.0f}s total)",
                  flush=True)
            if rhat_warnings:
                print(f"[WARN {now}] rank {source} {job_label} has "
                      f"R-hat > {config_info['rhat_threshold']}: "
                      f"{_format_rhat_warnings(rhat_warnings)}",
                      flush=True)
            if n_ok % PROGRESS_INTERVAL == 0:
                _print_bias_table(
                    running_biases,
                    header=f"Running bias after {n_ok} mocks",
                    show_ks=True)

        if n_sent < n_mocks:
            active_jobs[source] = n_sent + 1
            comm.send(seeds[n_sent], dest=source, tag=TAG_WORK)
            n_sent += 1
        else:
            comm.send(None, dest=source, tag=TAG_DONE)

    elapsed = time.time() - t0
    print(f"\n[INFO] Done in {elapsed:.0f}s "
          f"({elapsed / max(n_mocks, 1):.1f}s/mock)")

    # Collect biases
    biases = {p: [] for p in TRACKED_PARAMS}
    n_hosts_list = []
    mock_mag_obs = []
    mock_czcmb = []
    for b, n_hosts, n_parent, mock_diag, _, _ in results:
        n_hosts_list.append(n_hosts)
        mock_mag_obs.append(mock_diag["mag_obs"])
        mock_czcmb.append(mock_diag["czcmb"])
        for p in TRACKED_PARAMS:
            if p in b:
                biases[p].append(b[p])

    # Save
    save_dict = {}
    for p in TRACKED_PARAMS:
        if biases[p]:
            save_dict[p] = np.array(biases[p])
    save_dict["n_mocks"] = np.array(n_mocks)
    save_dict["n_skipped"] = np.array(n_skipped)
    save_dict["n_hosts"] = np.array(n_hosts_list)
    save_dict["params"] = np.array(TRACKED_PARAMS)
    if mock_mag_obs:
        save_dict["mock_mag_obs"] = np.concatenate(mock_mag_obs)
        save_dict["mock_czcmb"] = np.concatenate(mock_czcmb)

    for p in TRACKED_PARAMS:
        val = true_params.get(p)
        if val is not None:
            save_dict[f"true_{p}"] = np.array(val)

    outfile = os.path.join(outdir, f"mock_TRGB_biases_{mode_tag}.npz")
    np.savez(outfile, **save_dict)
    print(f"[INFO] Saved to {outfile}")
    plotfile = _plot_bias_summary(save_dict, outfile)
    if plotfile is not None:
        print(f"[INFO] Saved plot to {plotfile}")

    _print_summary(save_dict, n_skipped, n_mocks)


def worker(comm, config_info):
    """Worker: receive seeds, run mocks, send results."""
    from mpi4py import MPI

    base_config = config_info["base_config"]
    true_params = config_info["true_params"]
    mock_kwargs = config_info["mock_kwargs"]
    timeout = config_info["timeout"]
    num_warmup = config_info["num_warmup"]
    num_samples = config_info["num_samples"]
    which_selection = config_info["which_selection"]
    infer_selection = config_info["infer_selection"]
    use_field = config_info.get("use_field", False)
    field_name = config_info.get("field_name")
    num_chains = config_info.get("num_chains", 1)
    rhat_threshold = config_info.get("rhat_threshold", 1.05)

    def _alarm_handler(signum, frame):
        raise TimeoutError

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    while True:
        status = MPI.Status()
        seed = comm.recv(source=0, status=status)

        if status.Get_tag() == TAG_DONE:
            break

        if timeout > 0:
            signal.alarm(timeout)
        t_start = time.time()
        try:
            result = run_one_mock(
                seed, base_config, true_params, mock_kwargs,
                num_warmup=num_warmup, num_samples=num_samples,
                which_selection=which_selection,
                infer_selection=infer_selection,
                use_field=use_field, field_name=field_name, quiet=True,
                num_chains=num_chains, rhat_threshold=rhat_threshold)
            result = (*result, time.time() - t_start)
        except TimeoutError:
            result = None
        except Exception:
            import traceback
            traceback.print_exc()
            result = None
        finally:
            signal.alarm(0)

        comm.send(result, dest=0, tag=TAG_RESULT)

    signal.signal(signal.SIGALRM, old_handler)


PROGRESS_INTERVAL = 15


def _print_bias_table(biases, header=None, show_ks=False):
    """Print a table of mean bias +/- std for accumulated biases."""
    w = 55 if not show_ks else 65
    if header:
        print(f"\n{'=' * w}")
        print(header)
        print("=" * w)
    if show_ks:
        print(f"{'param':<20s}  {'mean +/- std':>20s}  {'KS p-value':>10s}")
    else:
        print(f"{'param':<20s}  {'mean +/- std':>20s}")
    print("-" * w)
    for p in TRACKED_PARAMS:
        if p in biases and len(biases[p]) >= 2:
            b = np.array(biases[p])
            line = (f"{p:<20s}  "
                    f"{f'{b.mean():+.3f} +/- {b.std():.3f}':>20s}")
            if show_ks:
                pval = kstest(b, "norm").pvalue
                line += f"  {pval:>10.3f}"
            print(line)


def _print_summary(save_dict, n_skipped, n_mocks):
    """Print summary table of standardised biases."""
    biases = {p: list(save_dict[p])
              for p in TRACKED_PARAMS
              if p in save_dict and isinstance(save_dict[p], np.ndarray)}
    _print_bias_table(biases,
                      header="Summary: mean standardised bias +/- std",
                      show_ks=True)
    if n_skipped:
        print(f"\n[WARN] {n_skipped}/{n_mocks} mocks timed out")


def _plot_bias_summary(save_dict, outfile):
    """Save a diagnostic plot of standardised biases."""
    params = [p for p in TRACKED_PARAMS
              if p in save_dict and len(np.asarray(save_dict[p])) > 0]
    if not params:
        return None

    ncols = 3
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, param in zip(axes, params):
        vals = np.asarray(save_dict[param])
        ax.hist(vals, bins="auto", alpha=0.75, edgecolor="black",
                linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(
            f"{param}: {vals.mean():+.2f} +/- {vals.std():.2f}",
            fontsize=9)
        ax.set_xlabel("standardised bias")
        ax.set_ylabel("mocks")

    for ax in axes[len(params):]:
        ax.axis("off")

    fig.tight_layout()
    plotfile = os.path.splitext(outfile)[0] + ".png"
    fig.savefig(plotfile, dpi=150)
    plt.close(fig)
    return plotfile


# ---- Sequential mode (no MPI) ---------------------------------------------

def run_sequential(config_info):
    """Run mocks sequentially without MPI."""
    n_mocks = config_info["n_mocks"]
    outdir = config_info["outdir"]
    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Running {n_mocks} mocks sequentially (no MPI)")
    print(f"[INFO] Injected true parameters:")
    for p, v in config_info["true_params"].items():
        print(f"         {p:<15s} = {v}")
    print(f"[INFO] use_field = {config_info.get('use_field', False)}")
    t0 = time.time()
    results = []
    n_ok = 0
    n_skipped = 0
    running_biases = {p: [] for p in TRACKED_PARAMS}

    master_seed = config_info.get("master_seed", 0)
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31, size=n_mocks).tolist()

    for i, seed in enumerate(seeds):
        t_start = time.time()
        rhat_warnings = {}
        try:
            result = run_one_mock(
                seed, config_info["base_config"],
                config_info["true_params"], config_info["mock_kwargs"],
                num_warmup=config_info["num_warmup"],
                num_samples=config_info["num_samples"],
                which_selection=config_info["which_selection"],
                infer_selection=config_info["infer_selection"],
                use_field=config_info.get("use_field", False),
                field_name=config_info.get("field_name"),
                quiet=True,
                progress_bar=config_info.get("progress_bar", True),
                num_chains=config_info.get("num_chains", 1),
                rhat_threshold=config_info.get("rhat_threshold", 1.05))
            dt = time.time() - t_start
            result = (*result, dt)
            results.append(result)
            n_ok += 1
            b, _, _, _, rhat_warnings, _ = result
            for p in TRACKED_PARAMS:
                if p in b:
                    running_biases[p].append(b[p])
        except Exception:
            n_skipped += 1
            dt = time.time() - t_start

        elapsed = time.time() - t0
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[INFO {now}] {i + 1}/{n_mocks} done "
              f"({dt:.0f}s this mock, {elapsed:.0f}s total)",
              flush=True)
        if "rhat_warnings" in locals() and rhat_warnings:
            print(f"[WARN {now}] mock {i + 1}/{n_mocks} has "
                  f"R-hat > {config_info['rhat_threshold']}: "
                  f"{_format_rhat_warnings(rhat_warnings)}",
                  flush=True)
        if n_ok > 0 and n_ok % PROGRESS_INTERVAL == 0:
            _print_bias_table(
                running_biases,
                header=f"Running bias after {n_ok} mocks",
                show_ks=True)

    elapsed = time.time() - t0
    print(f"\n[INFO] Done in {elapsed:.0f}s "
          f"({elapsed / max(n_mocks, 1):.1f}s/mock)")

    # Collect and save (reuse master's logic)
    biases = {p: [] for p in TRACKED_PARAMS}
    n_hosts_list = []
    mock_mag_obs = []
    mock_czcmb = []
    for b, n_hosts, n_parent, mock_diag, _, _ in results:
        n_hosts_list.append(n_hosts)
        mock_mag_obs.append(mock_diag["mag_obs"])
        mock_czcmb.append(mock_diag["czcmb"])
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
    save_dict["params"] = np.array(TRACKED_PARAMS)
    if mock_mag_obs:
        save_dict["mock_mag_obs"] = np.concatenate(mock_mag_obs)
        save_dict["mock_czcmb"] = np.concatenate(mock_czcmb)

    for p in TRACKED_PARAMS:
        val = config_info["true_params"].get(p)
        if val is not None:
            save_dict[f"true_{p}"] = np.array(val)

    outfile = os.path.join(
        outdir, f"mock_TRGB_biases_{config_info['mode_tag']}.npz")
    np.savez(outfile, **save_dict)
    print(f"[INFO] Saved to {outfile}")
    plotfile = _plot_bias_summary(save_dict, outfile)
    if plotfile is not None:
        print(f"[INFO] Saved plot to {plotfile}")

    _print_summary(save_dict, n_skipped, n_mocks)


# ---- Single-run mode ------------------------------------------------------

def run_single(seed, true_params, mock_kwargs, config_path,
               num_warmup=500, num_samples=500,
               which_selection="TRGB_magnitude",
               infer_selection=True, use_field=False, field_name=None,
               outdir=None, plot_only=False):
    """Generate a single mock, optionally run inference, and plot."""
    print(f"{'='*60}")
    print("Mock configuration")
    print(f"{'='*60}")
    print(f"  seed            = {seed}")
    print(f"  nsamples        = {mock_kwargs.get('nsamples')}")
    print(f"  rmax            = {mock_kwargs.get('rmax')} Mpc")
    print(f"  selection       = {which_selection}")
    if which_selection == "TRGB_magnitude":
        print(f"  mag_lim         = {mock_kwargs.get('mag_lim')}")
        print(f"  mag_lim_width   = {mock_kwargs.get('mag_lim_width')}")
    elif which_selection == "redshift":
        print(f"  cz_lim          = {mock_kwargs.get('cz_lim')}")
        print(f"  cz_lim_width    = {mock_kwargs.get('cz_lim_width')}")
    print(f"  use_field       = {use_field}")
    if use_field:
        print(f"  field_loader    = {mock_kwargs.get('field_loader')}")
    print(f"  infer_selection = {infer_selection}")
    print(f"  plot_only       = {plot_only}")
    print(f"\nInjected parameters:")
    for p, v in true_params.items():
        print(f"  {p:<15s} = {v}")
    print()

    config = make_mock_config(
        config_path, seed, num_warmup=num_warmup,
        num_samples=num_samples,
        which_selection=which_selection,
        mag_lim=mock_kwargs.get("mag_lim"),
        mag_lim_width=mock_kwargs.get("mag_lim_width"),
        cz_lim=mock_kwargs.get("cz_lim"),
        cz_lim_width=mock_kwargs.get("cz_lim_width"),
        infer_selection=infer_selection,
        use_field=use_field,
        rmax=mock_kwargs.get("rmax", 40.0))
    density_3d_data = _load_density_3d_data(
        config, field_name) if use_field else None
    data, tp, n_parent = gen_TRGB_mock(
        seed=seed, true_params=true_params, verbose=True,
        density_3d_data=density_3d_data, **mock_kwargs)
    tp["mu_LMC"] = DEFAULT_ANCHORS["mu_LMC"]
    tp["mu_N4258"] = DEFAULT_ANCHORS["mu_N4258"]
    _ra, _dec = candel.galactic_to_radec(tp["Vext_ell"], tp["Vext_b"])
    tp["Vext_phi"] = np.deg2rad(_ra)
    tp["Vext_cos_theta"] = np.sin(np.deg2rad(_dec))
    n = len(data["mag_obs"])

    print(f"\n{'='*60}")
    print(f"Mock TRGB catalog: {n} hosts")
    print(f"Parent population: {n_parent}")
    print(f"{'='*60}")

    print("\nObservable summary:")
    mag = data["mag_obs"]
    cz = data["czcmb"]
    print(f"  mag_obs: {mag.mean():.2f} +/- {mag.std():.2f}  "
          f"[{mag.min():.2f}, {mag.max():.2f}]")
    print(f"  czcmb:   {cz.mean():.0f} +/- {cz.std():.0f}  "
          f"[{cz.min():.0f}, {cz.max():.0f}] km/s")

    print("\nAnchors:")
    mu_LMC_true = DEFAULT_ANCHORS["mu_LMC"]
    mu_N4258_true = DEFAULT_ANCHORS["mu_N4258"]
    M = tp["M_TRGB"]
    print(f"  mu_LMC     = {data['mu_LMC_anchor']:.4f} "
          f"(true: {mu_LMC_true:.4f})")
    print(f"  mag_LMC    = {data['mag_LMC_TRGB']:.4f} "
          f"(true: {M + mu_LMC_true:.4f})")
    print(f"  mu_N4258   = {data['mu_N4258_anchor']:.4f} "
          f"(true: {mu_N4258_true:.4f})")
    print(f"  mag_N4258  = {data['mag_N4258_TRGB']:.4f} "
          f"(true: {M + mu_N4258_true:.4f})")

    # Run inference
    samples = None
    if not plot_only:
        tmp = _write_tmp_config(config)
        try:
            model = candel.model.TRGBModel(tmp, data)
            samples = candel.run_H0_inference(
                model, save_samples=False, print_summary=True)
        finally:
            os.unlink(tmp)

        print(f"\n{'='*60}")
        print("Standardised biases (posterior vs truth)")
        print("=" * 60)
        for param in TRACKED_PARAMS:
            if param in samples:
                s = np.asarray(samples[param])
                bias = _standardised_bias(s, tp[param], param)
                print(f"  {param:<15s}: {bias:+.2f}σ  "
                      f"(posterior {s.mean():.2f} ± {s.std():.2f}, "
                      f"true {tp[param]:.2f})")

    # Load real data for comparison (suppress loader prints)
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        real = candel.pvdata.load_EDD_TRGB_from_config(config_path)
    mag_real = np.asarray(real["mag_obs"])
    cz_real = np.asarray(real["czcmb"])

    # Diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    kw = dict(bins=30, density=True, alpha=0.6, edgecolor="black",
              linewidth=0.5)

    axes[0].hist(mag, label="Mock", **kw)
    axes[0].hist(mag_real, label="EDD TRGB", **kw)
    axes[0].set_xlabel(r"$m_{\rm TRGB}$ [mag]")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(cz, label="Mock", **kw)
    axes[1].hist(cz_real, label="EDD TRGB", **kw)
    axes[1].set_xlabel(r"$cz_{\rm CMB}$ [km/s]")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    axes[2].scatter(cz, mag, s=3, alpha=0.4, label="Mock")
    axes[2].scatter(cz_real, mag_real, s=3, alpha=0.4, label="EDD TRGB")
    axes[2].set_xlabel(r"$cz_{\rm CMB}$ [km/s]")
    axes[2].set_ylabel(r"$m_{\rm TRGB}$ [mag]")
    axes[2].legend()

    fig.tight_layout()
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(
        outdir,
        "mock_TRGB_single_"
        f"{_mode_tag(which_selection, use_field, field_name, infer_selection)}.png")
    fig.savefig(fname, dpi=150)
    print(f"\nSaved plot to {fname}")
    plt.close(fig)

    return data, tp, samples


# ---- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--single", action="store_true",
                        help="Single-mock mode: generate, run inference, "
                        "and plot (no MPI needed)")
    parser.add_argument("--plot-only", action="store_true",
                        help="With --single: skip inference, only generate "
                        "mock and plot")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--master-seed", type=int, default=0,
                        help="Master seed for reproducible mock seed sequence")
    parser.add_argument("--n-mocks", type=int, default=50,
                        help="Number of mock catalogs")
    parser.add_argument("--nsamples", type=int, default=480,
                        help="Number of mock hosts per catalog")
    parser.add_argument("--config", type=str,
                        default=os.path.join(
                            REPO_ROOT, "scripts/runs/configs/config_EDD_TRGB.toml"),
                        help="Base config for inference settings")
    parser.add_argument("--outdir",
                        default=results_path("results/mocks_TRGB"),
                        help="Output directory")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-mock timeout in seconds (0=none)")
    parser.add_argument("--num-warmup", type=int, default=500,
                        help="NUTS warmup steps")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="NUTS posterior samples")
    parser.add_argument("--num-chains", type=int, default=1,
                        help="NUTS chains per mock")
    parser.add_argument("--rhat-threshold", type=float, default=1.05,
                        help="Warn when NumPyro R-hat exceeds this value")
    parser.add_argument("--no-progress-bar", action="store_false",
                        dest="progress_bar",
                        help="Disable NumPyro progress bars in sequential "
                        "batch mode")
    parser.set_defaults(progress_bar=True)

    # Selection options
    parser.add_argument("--which-selection", type=str,
                        default="TRGB_magnitude",
                        choices=["TRGB_magnitude", "redshift"],
                        help="Selection function type")
    parser.add_argument("--mag-lim", type=float, default=25.0,
                        help="TRGB magnitude selection limit")
    parser.add_argument("--mag-lim-width", type=float, default=0.75,
                        help="Sigmoid width for magnitude selection")
    parser.add_argument("--cz-lim", type=float, default=250.0,
                        help="cz selection limit [km/s]")
    parser.add_argument("--cz-lim-width", type=float, default=500.0,
                        help="Sigmoid width for cz selection")
    parser.add_argument("--rmax", type=float, default=40.0,
                        help="Maximum mock distance [Mpc]")
    parser.add_argument("--fix-selection", action="store_false",
                        dest="infer_selection",
                        help="Fix selection thresholds to true mock values "
                        "instead of inferring them (default: infer)")
    parser.set_defaults(infer_selection=True)

    # Field-based mock options
    parser.add_argument("--use-field", action="store_true",
                        help="Enable field-based distance sampling "
                        "(inhomogeneous Malmquist)")
    parser.add_argument("--field-name", type=str, default="Carrick2015",
                        help="Reconstruction field name")
    parser.add_argument("--beta", type=float,
                        default=DEFAULT_TRUE_PARAMS["beta"],
                        help="True beta (velocity bias parameter)")
    parser.add_argument("--b1", type=float,
                        default=DEFAULT_TRUE_PARAMS["b1"],
                        help="True b1 (linear galaxy bias)")

    # True parameter overrides (defaults from DEFAULT_TRUE_PARAMS)
    tp = DEFAULT_TRUE_PARAMS
    parser.add_argument("--H0", type=float, default=tp["H0"],
                        help="True H0")
    parser.add_argument("--M-TRGB", type=float, default=tp["M_TRGB"],
                        help="True M_TRGB")
    parser.add_argument("--sigma-int", type=float, default=tp["sigma_int"],
                        help="True sigma_int")
    parser.add_argument("--sigma-v", type=float, default=tp["sigma_v"],
                        help="True sigma_v")

    args = parser.parse_args()

    # Resolve relative config paths against repo root.
    if not os.path.isabs(args.config):
        args.config = os.path.join(REPO_ROOT, args.config)

    # Ensure CWD is repo root so relative data paths in config work.
    os.chdir(REPO_ROOT)

    true_params = {
        "H0": args.H0,
        "M_TRGB": args.M_TRGB,
        "sigma_int": args.sigma_int,
        "sigma_v": args.sigma_v,
        "beta": args.beta,
        "b1": args.b1,
    }

    # Only pass the selection parameters relevant to the chosen mode so the
    # mock generator uses the correct branch (mag_lim vs cz_lim).
    if args.which_selection == "redshift":
        mock_kwargs = {
            "nsamples": args.nsamples,
            "rmax": args.rmax,
            "mag_lim": None,
            "mag_lim_width": None,
            "cz_lim": args.cz_lim,
            "cz_lim_width": args.cz_lim_width,
        }
    else:
        mock_kwargs = {
            "nsamples": args.nsamples,
            "rmax": args.rmax,
            "mag_lim": args.mag_lim,
            "mag_lim_width": args.mag_lim_width,
            "cz_lim": None,
            "cz_lim_width": None,
        }

    # Set up field loader if requested
    if args.use_field:
        from candel.field import name2field_loader
        config = candel.load_config(args.config, replace_los_prior=False)
        field_config = config["io"]["reconstruction_main"][args.field_name]
        loader_cls = name2field_loader(args.field_name)
        field_loader = loader_cls(**field_config)
        mock_kwargs["field_loader"] = field_loader

    if args.single:
        run_single(args.seed, true_params, mock_kwargs, args.config,
                   num_warmup=args.num_warmup, num_samples=args.num_samples,
                   which_selection=args.which_selection,
                   infer_selection=args.infer_selection,
                   use_field=args.use_field,
                   field_name=args.field_name,
                   outdir=args.outdir,
                   plot_only=args.plot_only)
        return

    # Batch mode
    config_info = {
        "base_config": args.config,
        "true_params": true_params,
        "mock_kwargs": mock_kwargs,
        "n_mocks": args.n_mocks,
        "master_seed": args.master_seed,
        "outdir": args.outdir,
        "timeout": args.timeout,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
        "rhat_threshold": args.rhat_threshold,
        "which_selection": args.which_selection,
        "infer_selection": args.infer_selection,
        "use_field": args.use_field,
        "field_name": args.field_name,
        "mode_tag": _mode_tag(args.which_selection, args.use_field,
                              args.field_name, args.infer_selection),
        "progress_bar": args.progress_bar,
    }

    # Try MPI; fall back to sequential only when no multi-rank scheduler
    # allocation is present.
    expected_mpi_tasks = _expected_mpi_tasks_from_env()
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError as exc:
        if expected_mpi_tasks > 1:
            raise RuntimeError(
                "This job was started with multiple scheduler tasks "
                f"({expected_mpi_tasks}), but mpi4py is not importable by "
                f"{sys.executable}. Install mpi4py in that environment before "
                "submitting MPI mock batches.") from exc
        size = 1
        rank = 0

    if expected_mpi_tasks > 1 and size == 1:
        raise RuntimeError(
            "This job was started with multiple scheduler tasks "
            f"({expected_mpi_tasks}), but mpi4py sees COMM_WORLD size 1. "
            "Check that addqueue is starting an MPI environment compatible "
            f"with the mpi4py installation used by {sys.executable}.")

    if size > 1:
        n_workers = size - 1
        if rank == 0:
            print(f"[INFO] MPI size = {size} ({n_workers} workers)")
            print(f"[INFO] n_mocks = {args.n_mocks}, "
                  f"nsamples = {args.nsamples}")
            print(f"[INFO] timeout = {args.timeout}s"
                  if args.timeout > 0 else "[INFO] timeout = none")
            skip_params = {"beta", "b1"} if not args.use_field else set()
            print(f"[INFO] Injected true parameters:")
            for p, v in true_params.items():
                if p not in skip_params:
                    print(f"         {p:<15s} = {v}")
            print(f"[INFO] use_field = {args.use_field}")
            os.makedirs(args.outdir, exist_ok=True)

        bcast_info = config_info if rank == 0 else None
        config_info = comm.bcast(bcast_info, root=0)

        if rank == 0:
            master(comm, n_workers, config_info)
        else:
            worker(comm, config_info)
    else:
        run_sequential(config_info)


if __name__ == "__main__":
    main()
