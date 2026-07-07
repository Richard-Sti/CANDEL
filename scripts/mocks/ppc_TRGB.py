#!/usr/bin/env python
"""Run a posterior predictive check for an existing EDD TRGB H0 posterior."""
import argparse
import multiprocessing as mp
import os

import numpy as np

import candel
from candel import get_nested
from candel.mock import (generate_trgb_ppc, plot_trgb_ppc,
                         plot_trgb_ppc_distance, plot_trgb_ppc_sky,
                         plot_trgb_ppc_sky_exposure)
from candel.mock.ppc_trgb import _available_field_indices

_WORKER_SAMPLES = None
_WORKER_DATA = None
_WORKER_CONFIG = None
_WORKER_N_PER_FIELD = None
_WORKER_SEED = None


def _load_trgb_data(config_path):
    config = candel.load_config(config_path, replace_los_prior=False)
    which_run = get_nested(config, "model/which_run", "EDD_TRGB")
    if which_run == "EDD_TRGB":
        return candel.pvdata.load_EDD_TRGB_from_config(config_path)
    if which_run == "EDD_TRGB_grouped":
        return candel.pvdata.load_EDD_TRGB_grouped_from_config(config_path)
    raise ValueError(
        f"TRGB PPC expects model.which_run EDD_TRGB or EDD_TRGB_grouped, "
        f"got {which_run!r}.")


def _parse_field_indices(value, data):
    available = _available_field_indices(data)
    if value in (None, "random"):
        return None
    if value == "all":
        return [int(x) for x in available]

    selected = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            selected.extend(range(int(lo), int(hi) + 1))
        else:
            selected.append(int(part))
    return selected


def _split_counts(total, n_fields):
    q, r = divmod(total, n_fields)
    return [q + (i < r) for i in range(n_fields)]


def _init_worker(samples, data, config_path, n_per_field, seed):
    global _WORKER_SAMPLES, _WORKER_DATA, _WORKER_CONFIG
    global _WORKER_N_PER_FIELD, _WORKER_SEED
    _WORKER_SAMPLES = samples
    _WORKER_DATA = data
    _WORKER_CONFIG = config_path
    _WORKER_N_PER_FIELD = n_per_field
    _WORKER_SEED = seed


def _run_field(args):
    i, field_index = args
    n_ppc = _WORKER_N_PER_FIELD[i]
    ppc = generate_trgb_ppc(
        _WORKER_SAMPLES, _WORKER_DATA, _WORKER_CONFIG,
        n_ppc=n_ppc, seed=_WORKER_SEED + i, field_index=field_index,
        n_workers=1)
    ppc["field_index"] = int(field_index)
    return ppc


def _same_value(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_same_value(a[k], b[k]) for k in a)
    try:
        aa = np.asarray(a)
        bb = np.asarray(b)
    except (TypeError, ValueError):
        return a == b
    if aa.dtype.kind == "f" or bb.dtype.kind == "f":
        return np.allclose(aa, bb)
    return np.array_equal(aa, bb)


def _merge_ppc_chunks(chunks):
    keys = sorted(set().union(*(c.keys() for c in chunks)))
    out = {}
    for key in keys:
        if key == "field_index":
            continue
        values = [c[key] for c in chunks if key in c]
        if len(values) != len(chunks):
            continue
        if key.endswith("_sim"):
            out[key] = np.concatenate([np.asarray(v) for v in values])
        elif key.endswith("_obs"):
            out[key] = np.asarray(values[0])
        elif all(_same_value(values[0], v) for v in values[1:]):
            out[key] = values[0]
        elif key == "sky_exposure":
            print("PPC warning: sky_exposure differs between fields; "
                  "omitting aggregate exposure metadata.")
    out["field_indices"] = np.asarray(
        [c["field_index"] for c in chunks], dtype=np.int32)
    return out


def _npz_payload(ppc):
    payload = {}
    for key, value in ppc.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                payload[f"{key}_{subkey}"] = np.asarray(subvalue)
        else:
            payload[key] = np.asarray(value)
    return payload


def _generate_ppc(samples, data, config_path, n_ppc, seed, field_indices,
                  n_workers):
    if not field_indices:
        return generate_trgb_ppc(
            samples, data, config_path, n_ppc=n_ppc, seed=seed)

    if n_ppc is None:
        config = candel.load_config(config_path, replace_los_prior=False)
        n_ppc = get_nested(config, "model/ppc_factor",
                           10) * len(data["mag_obs"])

    counts = _split_counts(int(n_ppc), len(field_indices))
    n_workers = min(max(int(n_workers), 1), len(field_indices))
    print(f"running PPC over {len(field_indices)} fields "
          f"with {n_workers} worker(s).")
    for i, field_index in enumerate(field_indices):
        print(f"  field {field_index}: n_ppc={counts[i]}")

    if n_workers == 1:
        _init_worker(samples, data, config_path, counts, seed)
        chunks = [_run_field((i, field_index))
                  for i, field_index in enumerate(field_indices)]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
                processes=n_workers, initializer=_init_worker,
                initargs=(samples, data, config_path, counts, seed)) as pool:
            chunks = pool.map(
                _run_field, list(enumerate(field_indices)))

    return _merge_ppc_chunks(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a magnitude/redshift PPC plot for EDD TRGB H0.")
    parser.add_argument("--config", required=True,
                        help="TOML config used for the posterior.")
    parser.add_argument("--posterior", required=True,
                        help="HDF5 posterior samples file.")
    parser.add_argument("--output", default=None,
                        help="Output PNG. Defaults to <posterior>_ppc.png.")
    parser.add_argument("--n-ppc", type=int, default=None,
                        help="Number of simulated PPC hosts.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument(
        "--field-indices", default="random",
        help=("Reconstruction fields to use: random, all, comma list, or "
              "ranges such as 0-29. With all/list, fields are processed "
              "independently and concatenated."))
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel worker processes for field PPC chunks.")
    parser.add_argument("--save-npz", default=None,
                        help="Optional NPZ file for concatenated PPC samples.")
    args = parser.parse_args()

    posterior = os.path.abspath(args.posterior)
    config_path = os.path.abspath(args.config)
    output = args.output
    if output is None:
        output = posterior.rsplit(".", 1)[0] + "_ppc.png"
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    data = _load_trgb_data(config_path)
    samples = candel.read_samples("", posterior)
    field_indices = _parse_field_indices(args.field_indices, data)
    ppc = _generate_ppc(
        samples, data, config_path, args.n_ppc, args.seed,
        field_indices, args.n_workers)
    stats = plot_trgb_ppc(ppc, output)
    extra_outputs = []
    output_root, output_ext = os.path.splitext(output)
    output_ext = output_ext or ".png"
    if "r_sim" in ppc:
        distance_output = f"{output_root}_distance{output_ext}"
        plot_trgb_ppc_distance(ppc, distance_output)
        extra_outputs.append(distance_output)
    if all(k in ppc for k in ("ra_sim", "dec_sim", "ra_obs", "dec_obs")):
        sky_output = f"{output_root}_sky{output_ext}"
        plot_trgb_ppc_sky(ppc, sky_output)
        extra_outputs.append(sky_output)
    if "sky_exposure" in ppc:
        exposure_output = f"{output_root}_sky_exposure{output_ext}"
        plot_trgb_ppc_sky_exposure(ppc, exposure_output)
        extra_outputs.append(exposure_output)
    if args.save_npz is not None:
        npz_path = os.path.abspath(args.save_npz)
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, **_npz_payload(ppc))

    print("TRGB PPC summary")
    print("================")
    print(f"posterior: {posterior}")
    print(f"config:    {config_path}")
    print(f"output:    {output}")
    for extra_output in extra_outputs:
        print(f"output:    {extra_output}")
    print(f"n_obs:     {len(ppc['mag_obs'])}")
    print(f"n_ppc:     {len(ppc['mag_sim'])}")
    if "field_indices" in ppc:
        fields = ",".join(str(int(x)) for x in ppc["field_indices"])
        print(f"fields:    {fields}")
    if args.save_npz is not None:
        print(f"npz:       {os.path.abspath(args.save_npz)}")
    print(f"KS mag:    D={stats['ks_mag_statistic']:.3f}, "
          f"p={stats['ks_mag_pvalue']:.3f}")
    print(f"KS cz:     D={stats['ks_cz_statistic']:.3f}, "
          f"p={stats['ks_cz_pvalue']:.3f}")


if __name__ == "__main__":
    main()
