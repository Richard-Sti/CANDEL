#!/usr/bin/env python
"""Subsample convergence test for the CH0 3D selection integral.

For 1000 posterior samples, evaluate the CH0 reconstruction selection
normalizer

    log S(theta) = logsumexp_i(log P_sel,i + log n_i + log dV_i)

on one 3D field, then repeat after randomly keeping a voxel fraction f.  The
subsampled estimate is corrected by 1/f so it should converge to the full-field
value as f -> 1.  The plot shows the posterior-sample median and 16-84%
interval of ln(S) versus voxel fraction.
"""
from __future__ import annotations

import argparse
import csv
from functools import partial
import os
from pathlib import Path
import sys
import tomllib

import h5py


def ensure_gpu_ld_library_path() -> None:
    """Mirror megamaser GPU scripts: expose CUDA libs before importing JAX."""
    local_config = ROOT / "local_config.toml"
    if not local_config.exists():
        return
    with local_config.open("rb") as f:
        config = tomllib.load(f)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    needed = [p for p in config.get("gpu_ld_library_path", [])
              if p and p not in ld]
    if needed:
        os.environ["LD_LIBRARY_PATH"] = (
            ":".join(needed) + (f":{ld}" if ld else ""))
        os.execv(sys.executable, [sys.executable] + sys.argv)


ROOT = Path("/mnt/users/rstiskalek/CANDEL")
ensure_gpu_ld_library_path()

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax.scipy.special import logsumexp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from candel.cosmo.cosmography import Distance2Distmod, Distance2Redshift
from candel.model.utils import log_prob_integrand_sel, predict_cz


DEFAULT_POSTERIORS = {
    "SN_magnitude": ROOT / "results/CH0/CH0_sel-SN_magnitude_manticore_2MPP_MULTIBIN_N256_DES_V2.hdf5",
    "redshift": ROOT / "results/CH0/CH0_sel-redshift_manticore_2MPP_MULTIBIN_N256_DES_V2.hdf5",
}
DEFAULT_DENSITY_CACHE = (
    ROOT / "data/field_cache/h0_volume_data/"
    "v1__manticore_2MPP_MULTIBIN_N256_DES_V2__fields-0-29__sphere__r-100__ds-1__density.npz"
)
DEFAULT_VELOCITY_CACHE = (
    ROOT / "data/field_cache/h0_volume_data/"
    "v1__manticore_2MPP_MULTIBIN_N256_DES_V2__fields-0-29__sphere__r-100__ds-1__vel.npz"
)
POSTERIOR_SUBSAMPLE_SIZE = 1000
H0_REWEIGHT_PLOT_FRACTIONS = 3


def parse_fractions(text: str) -> np.ndarray:
    vals = np.array([float(x) for x in text.split(",")], dtype=np.float64)
    if vals.ndim != 1 or len(vals) == 0:
        raise argparse.ArgumentTypeError("at least one fraction is required")
    if np.any(vals <= 0) or np.any(vals > 1):
        raise argparse.ArgumentTypeError("fractions must be in (0, 1]")
    return np.unique(vals)


def choose_voxel_sample(n_voxels: int, fraction: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, float]:
    if np.isclose(fraction, 1.0):
        idx = np.arange(n_voxels, dtype=np.int64)
        return idx, 1.0
    n_keep = int(np.ceil(float(fraction) * n_voxels))
    idx = choose_indices(n_voxels, n_keep, rng)
    return idx, n_keep / n_voxels


def gpu_probe() -> None:
    devices = jax.devices()
    backend = jax.default_backend()
    print(f"JAX backend: {backend}")
    print("JAX devices:", ", ".join(str(d) for d in devices))
    if backend != "gpu":
        raise RuntimeError("JAX did not select a GPU backend.")


def read_samples(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {key: f["samples"][key][:] for key in f["samples"]}


def choose_indices(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    count = min(count, n)
    return np.sort(rng.choice(n, size=count, replace=False))


def describe_indices(indices: np.ndarray) -> str:
    if len(indices) <= 12:
        return ", ".join(str(int(i)) for i in indices)
    head = ", ".join(str(int(i)) for i in indices[:6])
    tail = ", ".join(str(int(i)) for i in indices[-6:])
    return f"{head}, ..., {tail}"


def infer_bias(samples: dict[str, np.ndarray], requested: str) -> str:
    if requested != "auto":
        return requested
    if {"alpha_low", "alpha_high", "log_rho_t"} <= samples.keys():
        return "double_powerlaw"
    if "b1" in samples:
        return "linear"
    return "unity"


def load_field_cache(path: Path, selection: str, field_index: int,
                     bias_model: str, max_voxels: int | None,
                     rng: np.random.Generator) -> dict[str, jax.Array]:
    print(f"loading cache: {path}")
    with np.load(path, allow_pickle=False) as f:
        rho = np.asarray(f["rho_3d_fields"][field_index], dtype=np.float32)
        r = np.asarray(f["r_3d"], dtype=np.float32)
        log_dV = float(f["log_dV_3d"])
        log_volume_weight = np.asarray(
            f["log_volume_weight_3d"], dtype=np.float32)

        out_np: dict[str, np.ndarray | float] = {
            "rho": rho,
            "r": r,
            "log_dV": log_dV,
            "log_volume_weight": log_volume_weight,
        }
        if selection == "redshift":
            out_np["vrad"] = np.asarray(
                f["vrad_3d_fields"][field_index], dtype=np.float32)
            for key in ("rhat_x_3d", "rhat_y_3d", "rhat_z_3d"):
                out_np[key] = np.asarray(f[key], dtype=np.float32)

    n = len(rho)
    if max_voxels is not None and max_voxels < n:
        idx = choose_indices(n, max_voxels, rng)
        for key, value in list(out_np.items()):
            if isinstance(value, np.ndarray) and value.shape == (n,):
                out_np[key] = value[idx]
        n = max_voxels
        print(f"restricted smoke grid to {n:,} voxels")

    rho_arr = out_np.pop("rho")
    if bias_model in ("powerlaw", "double_powerlaw"):
        density = np.log(np.clip(rho_arr, 1.0e-6, None))
        density_mode = "log_rho"
    else:
        density = rho_arr - 1.0
        density_mode = "delta"

    out: dict[str, jax.Array | float | str] = {
        "density": jnp.asarray(density),
        "density_mode": density_mode,
        "log_r": jnp.log(jnp.asarray(out_np.pop("r"))),
        "log_dV": out_np.pop("log_dV"),
        "log_volume_weight": jnp.asarray(out_np.pop("log_volume_weight")),
    }
    radius = jnp.exp(out["log_r"])
    out["mu_h1"] = Distance2Distmod(Om0=0.306)(radius, h=1.0)
    if selection == "redshift":
        out["zcosmo"] = Distance2Redshift(Om0=0.306)(radius, h=1.0)
    for key, value in out_np.items():
        out[key] = jnp.asarray(value)
    print(f"loaded field {field_index} with {n:,} voxels")
    return out


def sample_array(samples: dict[str, np.ndarray], key: str, idx: np.ndarray,
                 default: float | tuple[float, ...] | None = None) -> jax.Array:
    if key in samples:
        return jnp.asarray(np.asarray(samples[key])[idx])
    if default is None:
        raise KeyError(f"posterior is missing required sample '{key}'")
    arr = np.asarray(default, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.full(len(idx), float(arr), dtype=np.float32)
    else:
        arr = np.broadcast_to(arr, (len(idx),) + arr.shape).astype(np.float32)
    return jnp.asarray(arr)


def sample_batch(samples: dict[str, np.ndarray], idx: np.ndarray,
                 selection: str, bias_model: str) -> dict[str, jax.Array]:
    batch = {"H0": sample_array(samples, "H0", idx)}
    if selection == "SN_magnitude":
        batch["M_B"] = sample_array(samples, "M_B", idx)
        batch["mag_lim_SN"] = sample_array(
            samples, "mag_lim_SN", idx, 14.0)
        batch["mag_lim_SN_width"] = sample_array(
            samples, "mag_lim_SN_width", idx, 0.15)
    elif selection == "redshift":
        batch["Vext"] = sample_array(samples, "Vext", idx, (0.0, 0.0, 0.0))
        batch["beta"] = sample_array(samples, "beta", idx, 1.0)
        batch["sigma_v"] = sample_array(samples, "sigma_v", idx, 200.0)
        batch["cz_lim_selection"] = sample_array(
            samples, "cz_lim_selection", idx, 3300.0)
        batch["cz_lim_selection_width"] = sample_array(
            samples, "cz_lim_selection_width", idx, 300.0)
    else:
        raise ValueError(selection)

    if bias_model == "linear":
        batch["b1"] = sample_array(samples, "b1", idx, 1.0)
    elif bias_model == "double_powerlaw":
        batch["alpha_low"] = sample_array(samples, "alpha_low", idx)
        batch["alpha_high"] = sample_array(samples, "alpha_high", idx)
        batch["log_rho_t"] = sample_array(samples, "log_rho_t", idx)
    return batch


def slice_batch(batch: dict[str, jax.Array], start: int,
                stop: int) -> dict[str, jax.Array]:
    return {key: value[start:stop] for key, value in batch.items()}


def array_field(field: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {key: value for key, value in field.items()
            if not isinstance(value, str)}


def log_bias_batch(density: jax.Array, density_mode: str, bias_model: str,
                   batch: dict[str, jax.Array]) -> jax.Array | float:
    if bias_model == "unity":
        return 0.0
    if bias_model == "linear":
        nr = 1.0 + batch["b1"][:, None] * density[None, :]
        return jnp.log(0.5 * (nr + jnp.sqrt(nr * nr + 0.1**2)))
    if bias_model == "double_powerlaw":
        if density_mode != "log_rho":
            log_rho = jnp.log(jnp.clip(1.0 + density, a_min=1.0e-6))
        else:
            log_rho = density
        log_x = log_rho[None, :] - batch["log_rho_t"][:, None]
        return (batch["alpha_low"][:, None] * log_x
                + (batch["alpha_high"][:, None]
                   - batch["alpha_low"][:, None])
                * jnp.logaddexp(0.0, log_x))
    raise ValueError(f"unsupported bias model: {bias_model}")


@partial(jax.jit, static_argnames=("bias_model", "density_mode"))
def log_s_mag_batch(field: dict[str, jax.Array], batch: dict[str, jax.Array],
                    bias_model: str, density_mode: str,
                    e_mag: float) -> jax.Array:
    h = batch["H0"] / 100.0
    mu = field["mu_h1"][None, :] - 5.0 * jnp.log10(h)[:, None]
    log_p = log_prob_integrand_sel(
        mu + batch["M_B"][:, None], e_mag,
        batch["mag_lim_SN"][:, None],
        batch["mag_lim_SN_width"][:, None],
    )
    log_n = log_bias_batch(
        field["density"], density_mode, bias_model, batch)
    log_weight = field["log_dV"] - 3.0 * jnp.log(h)[:, None]
    log_weight = log_weight + field["log_volume_weight"][None, :]
    return logsumexp(log_p + log_n + log_weight, axis=1)


@partial(jax.jit, static_argnames=("bias_model", "density_mode"))
def log_s_redshift_batch(
        field: dict[str, jax.Array], batch: dict[str, jax.Array],
        bias_model: str, density_mode: str) -> jax.Array:
    h = batch["H0"] / 100.0
    vext = batch["Vext"]
    vext_rad = (
        vext[:, 0, None] * field["rhat_x_3d"][None, :]
        + vext[:, 1, None] * field["rhat_y_3d"][None, :]
        + vext[:, 2, None] * field["rhat_z_3d"][None, :]
    )
    cz_pred = predict_cz(
        field["zcosmo"][None, :],
        batch["beta"][:, None] * field["vrad"][None, :] + vext_rad)
    log_p = log_prob_integrand_sel(
        cz_pred,
        batch["sigma_v"][:, None],
        batch["cz_lim_selection"][:, None],
        batch["cz_lim_selection_width"][:, None],
    )
    log_n = log_bias_batch(
        field["density"], density_mode, bias_model, batch)
    log_weight = field["log_dV"] - 3.0 * jnp.log(h)[:, None]
    log_weight = log_weight + field["log_volume_weight"][None, :]
    return logsumexp(log_p + log_n + log_weight, axis=1)


def evaluate_selection_all(selection: str, field: dict[str, jax.Array],
                           batch: dict[str, jax.Array], bias_model: str,
                           e_mag: float, batch_size: int) -> np.ndarray:
    field_arr = array_field(field)
    density_mode = field["density_mode"]
    n = int(batch["H0"].shape[0])
    blocks = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        sub = slice_batch(batch, start, stop)
        if selection == "SN_magnitude":
            vals = log_s_mag_batch(
                field_arr, sub, bias_model, density_mode, e_mag)
        elif selection == "redshift":
            vals = log_s_redshift_batch(
                field_arr, sub, bias_model, density_mode)
        else:
            raise ValueError(selection)
        blocks.append(np.asarray(vals.block_until_ready()))
    return np.concatenate(blocks)


def subset_field(field: dict[str, jax.Array], idx: np.ndarray
                 ) -> dict[str, jax.Array]:
    idx_jax = jnp.asarray(idx)
    out = {}
    n = int(field["density"].shape[0])
    for key, value in field.items():
        if isinstance(value, str):
            out[key] = value
        elif np.ndim(value) == 1 and value.shape[0] == n:
            out[key] = jnp.take(value, idx_jax, axis=0)
        else:
            out[key] = value
    return out


def run_one(selection: str, posterior_path: Path, cache_path: Path,
            args: argparse.Namespace, rng: np.random.Generator) -> list[dict]:
    samples = read_samples(posterior_path)
    bias_model = infer_bias(samples, args.bias_model)
    sample_indices = choose_indices(
        len(next(iter(samples.values()))), POSTERIOR_SUBSAMPLE_SIZE, rng)
    print(f"{selection}: using {len(sample_indices)} posterior samples")
    print(
        f"{selection}: posterior index range "
        f"{int(sample_indices[0])}-{int(sample_indices[-1])} "
        f"(preview: {describe_indices(sample_indices)})")
    print(f"{selection}: bias model {bias_model}")
    posterior = sample_batch(samples, sample_indices, selection, bias_model)
    h0_values = np.asarray(posterior["H0"])

    field = load_field_cache(
        cache_path, selection, args.field_index, bias_model,
        args.max_voxels, rng)
    n_vox = int(field["density"].shape[0])
    full_log_s = evaluate_selection_all(
        selection, field, posterior, bias_model, args.sn_selection_mag_error,
        args.posterior_batch_size)
    print(
        f"{selection}: full-field ln(S) median="
        f"{np.median(full_log_s):.8f}, 16-84%="
        f"[{np.percentile(full_log_s, 16):.8f}, "
        f"{np.percentile(full_log_s, 84):.8f}]",
        flush=True)

    rows = []
    total_jobs = sum(1 if np.isclose(float(frac), 1.0)
                     else args.num_resamples for frac in args.fractions)
    job = 0
    for frac in args.fractions:
        n_rep = 1 if np.isclose(frac, 1.0) else args.num_resamples
        for rep in range(n_rep):
            job += 1
            idx, actual_frac = choose_voxel_sample(n_vox, float(frac), rng)
            n_keep = len(idx)
            print(
                f"{selection}: evaluating fraction {actual_frac:.6f} "
                f"({n_keep:,}/{n_vox:,} voxels), resample {rep} "
                f"[{job}/{total_jobs}]",
                flush=True)
            if n_keep == n_vox:
                naive_log_s = full_log_s
            else:
                sub_field = subset_field(field, idx)
                naive_log_s = evaluate_selection_all(
                    selection, sub_field, posterior, bias_model,
                    args.sn_selection_mag_error, args.posterior_batch_size)
                naive_log_s = naive_log_s - np.log(actual_frac)

            for i, sample_idx in enumerate(sample_indices):
                rows.append({
                    "selection": selection,
                    "posterior_path": str(posterior_path),
                    "posterior_index": int(sample_idx),
                    "bias_model": bias_model,
                    "field_index": int(args.field_index),
                    "H0": float(h0_values[i]),
                    "fraction": float(actual_frac),
                    "n_voxels": int(n_vox),
                    "n_keep": int(n_keep),
                    "resample": int(rep),
                    "logS": float(naive_log_s[i]),
                    "full_logS": float(full_log_s[i]),
                    "delta_logS": float(naive_log_s[i] - full_log_s[i]),
                })
    return rows


def normalized_importance_weights(delta_log_s: np.ndarray) -> np.ndarray:
    logw = -np.asarray(delta_log_s, dtype=np.float64)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    return w / np.sum(w)


def effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weights)))


def weighted_mean_std(x: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.sum(weights * x))
    var = float(np.sum(weights * np.square(x - mean)))
    return mean, float(np.sqrt(max(var, 0.0)))


def summarize_h0_reweighting(rows: list[dict]) -> list[dict]:
    summary = []
    selections = list(dict.fromkeys(row["selection"] for row in rows))
    for selection in selections:
        sel_rows = [row for row in rows if row["selection"] == selection]
        fractions = sorted({row["fraction"] for row in sel_rows})
        reference = {}
        start = len(summary)
        for frac in fractions:
            reps = sorted({row["resample"] for row in sel_rows
                           if row["fraction"] == frac})
            for rep in reps:
                one = [row for row in sel_rows
                       if row["fraction"] == frac
                       and row["resample"] == rep]
                h0 = np.array([row["H0"] for row in one], dtype=np.float64)
                delta = np.array(
                    [row["delta_logS"] for row in one], dtype=np.float64)
                weights = normalized_importance_weights(delta)
                mean, std = weighted_mean_std(h0, weights)
                ess = effective_sample_size(weights)
                row = {
                    "selection": selection,
                    "field_index": int(one[0]["field_index"]),
                    "fraction": float(frac),
                    "n_keep": int(one[0]["n_keep"]),
                    "resample": int(rep),
                    "H0_mean": mean,
                    "H0_std": std,
                    "ess": ess,
                    "ess_frac": ess / len(weights),
                }
                summary.append(row)
                if np.isclose(frac, 1.0):
                    reference[int(rep)] = (mean, std)

        default_ref = reference.get(0)
        if default_ref is None:
            continue
        for row in summary[start:]:
            mean_ref, std_ref = reference.get(row["resample"], default_ref)
            row["H0_mean_ratio"] = row["H0_mean"] / mean_ref
            row["H0_std_ratio"] = (
                row["H0_std"] / std_ref if std_ref != 0 else np.nan)
    return summary


def representative_fractions(fractions: list[float]) -> list[float]:
    subunit = [f for f in sorted(fractions) if not np.isclose(f, 1.0)]
    if len(subunit) <= H0_REWEIGHT_PLOT_FRACTIONS:
        return subunit
    targets = np.linspace(0, len(subunit) - 1, H0_REWEIGHT_PLOT_FRACTIONS)
    return [subunit[int(round(t))] for t in targets]


def summarize_rows(rows: list[dict]) -> list[dict]:
    summary = []
    selections = list(dict.fromkeys(row["selection"] for row in rows))
    for selection in selections:
        sel_rows = [row for row in rows if row["selection"] == selection]
        fractions = sorted({row["fraction"] for row in sel_rows})
        for frac in fractions:
            one = [row for row in sel_rows if row["fraction"] == frac]
            log_s = np.array([row["logS"] for row in one])
            delta = np.array([row["delta_logS"] for row in one])
            n_keep = sorted({row["n_keep"] for row in one})
            summary.append({
                "selection": selection,
                "field_index": int(one[0]["field_index"]),
                "fraction": float(frac),
                "n_keep": int(n_keep[0]),
                "n_values": int(len(one)),
                "logS_p16": float(np.percentile(log_s, 16.0)),
                "logS_p50": float(np.percentile(log_s, 50.0)),
                "logS_p84": float(np.percentile(log_s, 84.0)),
                "delta_logS_p16": float(np.percentile(delta, 16.0)),
                "delta_logS_p50": float(np.percentile(delta, 50.0)),
                "delta_logS_p84": float(np.percentile(delta, 84.0)),
            })
    return summary


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(header), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]
    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.rjust(widths[i]) for i, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def format_summary_tables(rows: list[dict]) -> str:
    lines = []
    ln_s_summary = summarize_rows(rows)
    h0_summary = summarize_h0_reweighting(rows)
    selections = list(dict.fromkeys(row["selection"] for row in rows))

    lines.append("selection_integral_subsample_summary")
    for selection in selections:
        sel_rows = [
            row for row in ln_s_summary if row["selection"] == selection]
        lines.append("")
        lines.append(f"[{selection}] field_index={sel_rows[0]['field_index']}")
        table_rows = [
            [
                f"{row['fraction']:.3f}",
                f"{row['n_keep']:,}",
                f"{row['n_values']:,}",
                f"{row['logS_p16']:.6f}",
                f"{row['logS_p50']:.6f}",
                f"{row['logS_p84']:.6f}",
                f"{row['delta_logS_p16']:+.6f}",
                f"{row['delta_logS_p50']:+.6f}",
                f"{row['delta_logS_p84']:+.6f}",
            ]
            for row in sel_rows
        ]
        lines.append(format_table(
            ["f", "n_keep", "N", "lnS16", "lnS50", "lnS84",
             "dlnS16", "dlnS50", "dlnS84"],
            table_rows))

    lines.append("")
    lines.append("reweighted_H0_summary")
    for selection in selections:
        sel_rows = [
            row for row in h0_summary if row["selection"] == selection]
        lines.append("")
        lines.append(f"[{selection}] field_index={sel_rows[0]['field_index']}")
        table_rows = [
            [
                f"{row['fraction']:.3f}",
                f"{row['n_keep']:,}",
                f"{row['resample']}",
                f"{row['H0_mean']:.5f}",
                f"{row['H0_std']:.5f}",
                f"{row['H0_mean_ratio']:.8f}",
                f"{row['H0_std_ratio']:.8f}",
                f"{row['ess']:.1f}",
                f"{row['ess_frac']:.4f}",
            ]
            for row in sel_rows
        ]
        lines.append(format_table(
            ["f", "n_keep", "rep", "H0_mean", "H0_std",
             "mean/f1", "std/f1", "ESS", "ESS/N"],
            table_rows))
    return "\n".join(lines)


def print_rows(rows: list[dict]) -> None:
    print()
    print(format_summary_tables(rows))


def plot_rows(rows: list[dict], path: Path) -> None:
    summary = summarize_rows(rows)
    selections = list(dict.fromkeys(row["selection"] for row in summary))
    fig, axes = plt.subplots(
        2, len(selections), figsize=(6.0 * len(selections), 7.0),
        squeeze=False, sharex="col")
    for col, selection in enumerate(selections):
        sel_rows = [row for row in summary if row["selection"] == selection]
        x = np.array([row["fraction"] for row in sel_rows])
        log_p16 = np.array([row["logS_p16"] for row in sel_rows])
        log_p50 = np.array([row["logS_p50"] for row in sel_rows])
        log_p84 = np.array([row["logS_p84"] for row in sel_rows])
        delta_p16 = np.array([row["delta_logS_p16"] for row in sel_rows])
        delta_p50 = np.array([row["delta_logS_p50"] for row in sel_rows])
        delta_p84 = np.array([row["delta_logS_p84"] for row in sel_rows])
        order = np.argsort(x)
        x = x[order]
        axes[0, col].fill_between(
            x, log_p16[order], log_p84[order], alpha=0.18, color="C0",
            label="16-84%")
        axes[0, col].plot(
            x, log_p50[order], "o-", ms=3, color="C0", label="median")
        axes[1, col].fill_between(
            x, delta_p16[order], delta_p84[order], alpha=0.18, color="C0",
            label="16-84%")
        axes[1, col].plot(
            x, delta_p50[order], "o-", ms=3, color="C0", label="median")
        axes[0, col].set_title(selection)
        axes[0, col].set_ylabel("ln S")
        axes[1, col].axhline(0.0, color="0.2", lw=1)
        axes[1, col].set_ylabel("ln S(f) - ln S(full)")
        axes[1, col].set_xlabel("voxel fraction f")
        axes[0, col].grid(alpha=0.25)
        axes[1, col].grid(alpha=0.25)
        axes[0, col].legend(fontsize=8)
        axes[1, col].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_reweighted_h0(rows: list[dict], path: Path) -> None:
    selections = list(dict.fromkeys(row["selection"] for row in rows))
    fig, axes = plt.subplots(
        1, len(selections), figsize=(6.0 * len(selections), 3.8),
        squeeze=False)
    for col, selection in enumerate(selections):
        ax = axes[0, col]
        sub = [row for row in rows if row["selection"] == selection]
        first_by_sample = {}
        for row in sub:
            first_by_sample.setdefault(row["posterior_index"], row)
        h0_base = np.array(
            [row["H0"] for row in first_by_sample.values()],
            dtype=np.float64)
        lo, hi = np.percentile(h0_base, [0.5, 99.5])
        pad = 0.15 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 45)
        ax.hist(h0_base, bins=bins, density=True, histtype="step",
                color="0.15", lw=1.6, label="original")

        fractions = sorted({row["fraction"] for row in sub})
        plot_fracs = representative_fractions(fractions)
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(plot_fracs)))
        for frac, color in zip(plot_fracs, colors):
            one = [row for row in sub
                   if row["fraction"] == frac and row["resample"] == 0]
            h0 = np.array([row["H0"] for row in one], dtype=np.float64)
            delta = np.array(
                [row["delta_logS"] for row in one], dtype=np.float64)
            weights = normalized_importance_weights(delta)
            ax.hist(h0, bins=bins, weights=weights, density=True,
                    histtype="step", lw=1.4, color=color,
                    label=f"f={frac:.3g}")

        ax.set_title(selection)
        ax.set_xlabel(r"$H_0$")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("H0 posterior reweighted by exp[-Delta ln S(f)]", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_h0_summary(rows: list[dict], path: Path) -> None:
    summary = summarize_h0_reweighting(rows)
    selections = list(dict.fromkeys(row["selection"] for row in summary))
    fig, axes = plt.subplots(
        2, len(selections), figsize=(6.0 * len(selections), 6.2),
        squeeze=False, sharex="col")
    for col, selection in enumerate(selections):
        axes[0, col].axhline(1.0, color="0.2", lw=1)
        axes[1, col].axhline(1.0, color="0.2", lw=1)
        sel_rows = [
            row for row in summary
            if row["selection"] == selection and row["resample"] == 0]
        x = np.array([row["fraction"] for row in sel_rows])
        mean_ratio = np.array([row["H0_mean_ratio"] for row in sel_rows])
        std_ratio = np.array([row["H0_std_ratio"] for row in sel_rows])
        order = np.argsort(x)
        axes[0, col].plot(
            x[order], mean_ratio[order], "o-", ms=3, color="C0")
        axes[1, col].plot(
            x[order], std_ratio[order], "o-", ms=3, color="C0")
        axes[0, col].set_title(selection)
        axes[0, col].set_ylabel(r"$\langle H_0\rangle_f / \langle H_0\rangle_{f=1}$")
        axes[1, col].set_ylabel(r"$\sigma(H_0)_f / \sigma(H_0)_{f=1}$")
        axes[1, col].set_xlabel("voxel fraction f")
        axes[0, col].grid(alpha=0.25)
        axes[1, col].grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_h0_summary_csv(rows: list[dict], path: Path) -> None:
    summary = summarize_h0_reweighting(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selection", "field_index", "fraction", "n_keep", "resample",
        "H0_mean", "H0_std", "H0_mean_ratio", "H0_std_ratio",
        "ess", "ess_frac",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row[key] for key in fieldnames})


def write_summary_txt(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_summary_tables(rows) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mag-posterior", type=Path,
                   default=DEFAULT_POSTERIORS["SN_magnitude"])
    p.add_argument("--redshift-posterior", type=Path,
                   default=DEFAULT_POSTERIORS["redshift"])
    p.add_argument("--density-cache", type=Path, default=DEFAULT_DENSITY_CACHE)
    p.add_argument("--velocity-cache", type=Path, default=DEFAULT_VELOCITY_CACHE)
    p.add_argument("--field-index", type=int, default=0)
    p.add_argument("--fractions", type=parse_fractions,
                   default=parse_fractions("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"))
    p.add_argument("--num-resamples", type=int, default=1)
    p.add_argument("--posterior-batch-size", type=int, default=32,
                   help="Number of posterior samples per JIT-compiled GPU batch.")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--bias-model",
                   choices=["auto", "unity", "linear", "double_powerlaw"],
                   default="auto")
    p.add_argument("--sn-selection-mag-error", type=float, default=0.13486865)
    p.add_argument("--max-voxels", type=int, default=None,
                   help="Optional smoke-test cap on voxels after loading field 0.")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "scripts/H0_convergence/outputs")
    p.add_argument("--gpu-probe", action="store_true",
                   help="Only check that JAX sees a GPU, then exit.")
    args = p.parse_args()
    if args.posterior_batch_size < 1:
        raise ValueError("--posterior-batch-size must be positive")
    if args.num_resamples < 1:
        raise ValueError("--num-resamples must be positive")
    if not np.any(np.isclose(args.fractions, 1.0)):
        raise ValueError("--fractions must include 1.0 for H0 ratio summaries")

    print(f"JAX backend: {jax.default_backend()}")
    print("JAX devices:", ", ".join(str(d) for d in jax.devices()))
    if args.gpu_probe:
        gpu_probe()
        return

    rng = np.random.default_rng(args.seed)

    rows = []
    rows.extend(run_one(
        "SN_magnitude", args.mag_posterior, args.density_cache, args, rng))
    rows.extend(run_one(
        "redshift", args.redshift_posterior, args.velocity_cache, args, rng))

    png_path = args.output_dir / "posterior_selection_integral_subsample.png"
    h0_png_path = args.output_dir / "posterior_selection_integral_reweighted_h0.png"
    h0_summary_path = args.output_dir / "posterior_selection_integral_reweighted_h0_summary.png"
    h0_csv_path = args.output_dir / "posterior_selection_integral_reweighted_h0_summary.csv"
    summary_txt_path = args.output_dir / "posterior_selection_integral_summary_tables.txt"
    print_rows(rows)
    plot_rows(rows, png_path)
    plot_reweighted_h0(rows, h0_png_path)
    plot_h0_summary(rows, h0_summary_path)
    write_h0_summary_csv(rows, h0_csv_path)
    write_summary_txt(rows, summary_txt_path)
    print(f"wrote {png_path}")
    print(f"wrote {h0_png_path}")
    print(f"wrote {h0_summary_path}")
    print(f"wrote {h0_csv_path}")
    print(f"wrote {summary_txt_path}")


if __name__ == "__main__":
    main()
