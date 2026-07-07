# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Checkpointed wrapper around NumPyro NUTS."""
from __future__ import annotations

import os
import pickle
import platform
import time
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import numpyro
from jax import random
from tqdm.auto import tqdm

from ..util import fprint

_CHECKPOINT_VERSION = 2


@dataclass
class CheckpointedNUTSResult:
    """Samples and diagnostics returned by :func:`run_checkpointed_nuts`."""
    samples: dict[str, Any]
    extra_fields: dict[str, Any]
    last_state: Any
    completed_warmup: int
    completed_samples: int
    checkpoint_path: str


def _to_host(tree):
    """Synchronise a JAX pytree and copy it to host memory."""
    for leaf in jax.tree_util.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    return jax.device_get(tree)


def _block_until_ready(tree):
    """Synchronise a JAX pytree without copying it to host."""
    for leaf in jax.tree_util.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _atomic_pickle(path, payload):
    """Write a pickle payload atomically via a same-directory temp file."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_checkpoint(path):
    """Load and version-check a NUTS checkpoint pickle."""
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    version = checkpoint.get("version")
    if version != _CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported NUTS checkpoint version {version!r} in {path}.")
    return checkpoint


def _save_checkpoint(path, *, samples_by_chain, extra_fields_by_chain,
                     last_state, completed_warmup, completed_samples,
                     metadata, complete):
    """Serialise the current NUTS run state and accumulated samples."""
    payload = {
        "version": _CHECKPOINT_VERSION,
        "metadata": dict(metadata),
        "samples_by_chain": _to_host(samples_by_chain),
        "extra_fields_by_chain": _to_host(extra_fields_by_chain),
        "last_state": _to_host(last_state),
        "completed_warmup": int(completed_warmup),
        "completed_samples": int(completed_samples),
        "complete": bool(complete),
        "saved_at_unix": time.time(),
    }
    _atomic_pickle(path, payload)
    status = "complete" if complete else "partial"
    fprint(f"NUTS checkpoint ({status}, warmup={completed_warmup}, "
           f"samples={completed_samples}/chain): {path}")


def _concatenate_chain_axis(chunks):
    """Concatenate checkpoint chunks along the per-chain sample axis."""
    chunks = [c for c in chunks if c is not None]
    if not chunks:
        return None

    def _concat(*parts):
        """Concatenate one pytree leaf from all retained chunks."""
        return np.concatenate([np.asarray(part) for part in parts], axis=1)

    return jax.tree_util.tree_map(_concat, *chunks)


def _flatten_chain_axis(tree):
    """Flatten leading ``(chain, draw)`` axes to NumPyro sample format."""
    if tree is None:
        return {}

    def _flatten(x):
        """Flatten a single checkpoint leaf if it has chain and draw axes."""
        x = np.asarray(x)
        if x.ndim < 2:
            return x
        return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

    return jax.tree_util.tree_map(_flatten, tree)


def _user_metadata_for_validation(user):
    """Return effective run options used for checkpoint validation."""
    user = dict(user)
    out = {}
    for key in (
            "galaxy", "mode", "dist_tag", "sampler", "seed",
            "use_ecc", "use_quadratic_warp", "chain_index"):
        if key in user:
            out[key] = user[key]

    nuts_settings = dict(user.get("nuts_settings", {}))
    for key in ("num_warmup", "num_samples", "num_chains", "chain_method"):
        if key in nuts_settings:
            out.setdefault("nuts_settings", {})[key] = nuts_settings[key]

    init = dict(user.get("init", {}))
    init = {k: init[k] for k in ("method", "r_ang_method") if k in init}
    if init:
        out["init"] = init
    return out


def _metadata_for_validation(metadata):
    """Return metadata with diagnostic/non-structural fields removed."""
    metadata = dict(metadata)
    if "user" in metadata:
        metadata["user"] = _user_metadata_for_validation(metadata["user"])
    return metadata


def _metadata_diff(old, new):
    """Return human-readable key differences between metadata mappings."""
    old = _metadata_for_validation(old)
    new = _metadata_for_validation(new)
    keys = sorted(set(old) | set(new))
    return [
        f"{k}: checkpoint={old.get(k)!r}, current={new.get(k)!r}"
        for k in keys if old.get(k) != new.get(k)
    ]


def _validate_checkpoint(checkpoint, metadata, checkpoint_path):
    """Reject checkpoints that are incompatible with the current run."""
    old = checkpoint.get("metadata", {})
    mismatches = _metadata_diff(old, metadata)
    if mismatches:
        details = "; ".join(mismatches[:8])
        if len(mismatches) > 8:
            details += f"; ... ({len(mismatches)} mismatches total)"
        raise ValueError(
            f"NUTS checkpoint {checkpoint_path} is incompatible ({details}).")
    completed = int(checkpoint.get("completed_samples", 0))
    if completed > metadata["num_samples"]:
        raise ValueError(
            f"NUTS checkpoint {checkpoint_path} already contains {completed} "
            f"samples/chain, more than the requested "
            f"{metadata['num_samples']}.")


def _prepare_rng_key(rng_key, num_chains, chain_method):
    """Prepare a NumPyro RNG key for single-chain or vectorised chains."""
    if num_chains == 1:
        return rng_key
    if chain_method != "vectorized":
        raise NotImplementedError(
            "checkpointed NUTS currently supports num_chains > 1 only with "
            "chain_method='vectorized'.")
    return random.split(rng_key, num_chains)


def _initialise_kernel(sampler, rng_key, *, num_warmup, num_chains,
                       chain_method, init_params, run_args, run_kwargs):
    """Initialise the NumPyro kernel and populate sampler-side caches."""
    rng_key = _prepare_rng_key(rng_key, num_chains, chain_method)
    return sampler.init(
        rng_key, num_warmup, init_params=init_params,
        model_args=run_args, model_kwargs=run_kwargs)


def _get_by_path(obj, path):
    """Read a dotted attribute/dict path from a sampler state object."""
    out = obj
    for part in path.split("."):
        if isinstance(out, dict):
            out = out[part]
        else:
            out = getattr(out, part)
    return out


def _unique_fields(fields):
    """Return fields in first-seen order with duplicates removed."""
    out = []
    for field in fields:
        if field not in out:
            out.append(field)
    return tuple(out)


def _one_step_chain_first(tree, num_chains):
    """Add ``(chain, draw)`` axes to one post-warmup transition."""
    def _convert(x):
        """Convert one transition leaf to checkpoint chunk layout."""
        x = np.asarray(x)
        if num_chains == 1:
            return x[np.newaxis, np.newaxis, ...]
        return x[:, np.newaxis, ...]

    return jax.tree_util.tree_map(_convert, _to_host(tree))


def _make_sample_step(sampler, run_args, run_kwargs):
    """Build a JIT-compiled one-transition sampler step."""
    @jax.jit
    def sample_step(state):
        """Advance the NumPyro sampler by one transition."""
        return sampler.sample(
            state, model_args=run_args, model_kwargs=run_kwargs)

    return sample_step


def _make_postprocess_fn(sampler, run_args, run_kwargs, num_chains,
                         chain_method):
    """Create the constrained-space postprocessor for sampled latent states."""
    postprocess_fn = sampler.postprocess_fn(run_args, run_kwargs)
    if num_chains > 1 and chain_method == "vectorized":
        postprocess_fn = jax.vmap(postprocess_fn)
    return postprocess_fn


def _format_progress_value(value, *, integer=False, scientific=False):
    """Format scalar or per-chain values for tqdm postfix output."""
    arr = np.asarray(jax.device_get(value))
    if arr.size == 1:
        val = arr.reshape(-1)[0]
        if integer:
            return str(int(val))
        if scientific:
            return f"{float(val):.2e}"
        return f"{float(val):.3g}"

    arr = arr.astype(float).reshape(-1)
    if np.all(arr == arr[0]):
        if integer:
            return str(int(arr[0]))
        if scientific:
            return f"{float(arr[0]):.2e}"
        return f"{float(arr[0]):.3g}"

    if integer:
        return f"{int(arr.min())}-{int(arr.max())}"
    if scientific:
        return f"{arr.min():.2e}-{arr.max():.2e}"
    return f"{arr.min():.3g}-{arr.max():.3g}"


def _set_nuts_postfix(bar, state):
    """Display step-size and tree-step diagnostics on a progress bar."""
    if bar.disable:
        return
    step_size = _format_progress_value(
        state.adapt_state.step_size, scientific=True)
    num_steps = _format_progress_value(state.num_steps, integer=True)
    bar.set_postfix_str(
        f"step_size={step_size}, num_steps={num_steps}", refresh=True)


def run_checkpointed_nuts(
        sampler, rng_key, *, num_warmup, num_samples, num_chains=1,
        chain_method="sequential", thinning=1, progress_bar=True,
        checkpoint_path, resume=False, checkpoint_interval_seconds=900,
        extra_fields=(), init_params=None, run_args=(), run_kwargs=None,
        checkpoint_metadata=None):
    """Run a NumPyro NUTS kernel with resumable checkpoints.

    This drives the kernel via ``sampler.init`` and ``sampler.sample`` so the
    same checkpoint path protects both warmup/adaptation and posterior
    sampling.  Checkpoint timing is inspected after each completed NUTS
    transition; leapfrog/tree-building internals are left entirely to NumPyro.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if thinning != 1:
        raise NotImplementedError(
            "checkpointed NUTS currently supports thinning=1 only.")
    if checkpoint_interval_seconds <= 0:
        raise ValueError("checkpoint_interval_seconds must be positive.")
    if run_kwargs is None:
        run_kwargs = {}
    if checkpoint_metadata is None:
        checkpoint_metadata = {}
    run_args = tuple(run_args)
    extra_fields = _unique_fields(("diverging",) + tuple(extra_fields))

    metadata = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "jax_version": jax.__version__,
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "num_chains": int(num_chains),
        "chain_method": chain_method,
        "thinning": int(thinning),
        "extra_fields": tuple(extra_fields),
        "numpyro_version": numpyro.__version__,
        "python_version": platform.python_version(),
        "user": dict(checkpoint_metadata),
    }

    samples_by_chain = None
    extra_fields_by_chain = None
    sample_chunks = []
    extra_field_chunks = []
    last_state = None
    completed_warmup = 0
    completed_samples = 0

    if resume and os.path.isfile(checkpoint_path):
        checkpoint = _load_checkpoint(checkpoint_path)
        _validate_checkpoint(checkpoint, metadata, checkpoint_path)
        samples_by_chain = checkpoint["samples_by_chain"]
        extra_fields_by_chain = checkpoint["extra_fields_by_chain"]
        if samples_by_chain is not None:
            sample_chunks.append(samples_by_chain)
        if extra_fields_by_chain is not None:
            extra_field_chunks.append(extra_fields_by_chain)
        last_state = checkpoint["last_state"]
        completed_warmup = int(checkpoint.get("completed_warmup", 0))
        completed_samples = int(checkpoint["completed_samples"])
        fprint(f"Resuming NUTS from {checkpoint_path} "
               f"(warmup={completed_warmup}/{num_warmup}, "
               f"samples={completed_samples}/{num_samples}/chain)")
    elif resume:
        fprint(f"--resume: no NUTS checkpoint found at {checkpoint_path}, "
               "starting fresh")

    # Rebuild NumPyro's cached sampling functions even when loading state.
    initial_state = _initialise_kernel(
        sampler, rng_key, num_warmup=num_warmup, num_chains=num_chains,
        chain_method=chain_method, init_params=init_params,
        run_args=run_args, run_kwargs=run_kwargs)
    if last_state is None:
        last_state = initial_state
    sample_step = _make_sample_step(sampler, run_args, run_kwargs)

    if completed_samples >= num_samples:
        return CheckpointedNUTSResult(
            samples=_flatten_chain_axis(samples_by_chain),
            extra_fields=_flatten_chain_axis(extra_fields_by_chain),
            last_state=last_state,
            completed_warmup=completed_warmup,
            completed_samples=completed_samples,
            checkpoint_path=checkpoint_path,
        )

    last_checkpoint = time.time()
    warmup_bar = tqdm(
        total=num_warmup, initial=completed_warmup, desc="warmup",
        disable=not progress_bar)
    try:
        while completed_warmup < num_warmup:
            last_state = sample_step(last_state)
            _block_until_ready(last_state)
            completed_warmup += 1
            _set_nuts_postfix(warmup_bar, last_state)
            warmup_bar.update(1)

            now = time.time()
            if (now - last_checkpoint >= checkpoint_interval_seconds
                    or completed_warmup == num_warmup):
                _save_checkpoint(
                    checkpoint_path, samples_by_chain=None,
                    extra_fields_by_chain=None, last_state=last_state,
                    completed_warmup=completed_warmup, completed_samples=0,
                    metadata=metadata, complete=False)
                last_checkpoint = now
    finally:
        warmup_bar.close()

    postprocess_fn = _make_postprocess_fn(
        sampler, run_args, run_kwargs, num_chains, chain_method)

    last_checkpoint = time.time()
    sample_bar = tqdm(
        total=num_samples, initial=completed_samples, desc="sample",
        disable=not progress_bar)
    try:
        while completed_samples < num_samples:
            last_state = sample_step(last_state)
            samples_step = postprocess_fn(last_state.z)
            extra_step = {
                field: _get_by_path(last_state, field)
                for field in extra_fields
            }
            _block_until_ready((last_state, samples_step, extra_step))

            sample_chunks.append(_one_step_chain_first(
                samples_step, num_chains))
            extra_field_chunks.append(_one_step_chain_first(
                extra_step, num_chains))
            completed_samples += 1
            _set_nuts_postfix(sample_bar, last_state)
            sample_bar.update(1)

            now = time.time()
            if (now - last_checkpoint >= checkpoint_interval_seconds
                    or completed_samples == num_samples):
                samples_by_chain = _concatenate_chain_axis(sample_chunks)
                extra_fields_by_chain = _concatenate_chain_axis(
                    extra_field_chunks)
                sample_chunks = [samples_by_chain]
                extra_field_chunks = [extra_fields_by_chain]
                _save_checkpoint(
                    checkpoint_path, samples_by_chain=samples_by_chain,
                    extra_fields_by_chain=extra_fields_by_chain,
                    last_state=last_state, completed_warmup=completed_warmup,
                    completed_samples=completed_samples, metadata=metadata,
                    complete=completed_samples >= num_samples)
                last_checkpoint = now
    finally:
        sample_bar.close()

    return CheckpointedNUTSResult(
        samples=_flatten_chain_axis(_concatenate_chain_axis(sample_chunks)),
        extra_fields=_flatten_chain_axis(
            _concatenate_chain_axis(extra_field_chunks)),
        last_state=last_state,
        completed_warmup=completed_warmup,
        completed_samples=completed_samples,
        checkpoint_path=checkpoint_path,
    )
