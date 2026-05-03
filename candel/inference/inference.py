# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Running the MCMC inference for the model and some postprocessing."""
import contextlib
from copy import deepcopy
from os import makedirs
from os.path import dirname, splitext

import jax
import numpy as np
from h5py import File
from jax import jit
from jax import numpy as jnp
from jax import vmap
from numpyro import handlers, set_platform
from numpyro.diagnostics import print_summary as print_summary_numpyro
from numpyro.diagnostics import summary as summary_numpyro
from numpyro.distributions.transforms import biject_to
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median, init_to_value
from numpyro.infer.util import log_density
from scipy.optimize import minimize as sp_minimize
from tqdm import trange

from ..util import (fprint, fsection, galactic_to_radec, plot_corner,
                    plot_radial_profiles, plot_Vext_moll, plot_Vext_rad_corner,
                    plot_Vext_radmag, radec_cartesian_to_galactic,
                    radec_to_cartesian, radec_to_galactic)
from .evidence import (BIC_AIC, dict_samples_to_array, harmonic_evidence,
                       laplace_evidence)


def _harmonic_available():
    """True iff the optional ``harmonic`` package can be imported."""
    import importlib.util
    return importlib.util.find_spec("harmonic") is not None


def _setup_platform():
    """Detect devices and set NumPyro platform."""
    devices = jax.devices()
    device_str = ", ".join(f"{d.device_kind}({d.platform})" for d in devices)
    fprint(f"running inference on devices: {device_str}")
    platform = "cuda" if any(d.platform == "gpu" for d in devices) else "cpu"
    set_platform(platform)
    fprint(f"using platform: {platform}")


def _parse_dense_mass(kwargs, site_names=None):
    """Parse the dense mass matrix config.

    Supports:
      - ``dense_mass = true/false`` (boolean). With True, builds one dense
        block over all sampled sites whose names do NOT contain
        ``dense_mass_exclude_pattern`` (default ``"_latent"``); this keeps
        per-host latents diagonal.
      - ``dense_mass_params = ["H0", "M_TRGB", ...]`` (single block)
      - ``dense_mass_blocks = [["H0", "D_c"], ["i0", "di_dr"], ...]``
        (multiple independent dense blocks)
      - ``dense_mass_exclude_pattern = "_latent"`` (substring, only applied
        on the ``dense_mass=True`` auto-exclusion path)

    If `site_names` is provided, entries in ``dense_mass_params`` /
    ``dense_mass_blocks`` that are not actual sample sites are dropped and
    the dropped names are logged.
    """
    exclude_pattern = kwargs.get("dense_mass_exclude_pattern", "_latent")

    blocks = kwargs.get("dense_mass_blocks", None)
    if blocks is not None:
        result = []
        for block in blocks:
            if site_names is not None:
                dropped = [p for p in block if p not in site_names]
                block = [p for p in block if p in site_names]
                if dropped:
                    fprint(f"dropped {len(dropped)} non-sampled sites from "
                           f"dense_mass_blocks entry: {dropped}")
            if len(block) >= 2:
                result.append(tuple(block))
                fprint(f"dense mass block ({len(block)}): {list(block)}")
            elif len(block) == 1:
                fprint(f"dropped single-site block: {list(block)}")
        return result if result else False

    dense_mass_params = kwargs.get("dense_mass_params", None)
    if dense_mass_params is not None:
        if site_names is not None:
            dropped = [p for p in dense_mass_params if p not in site_names]
            dense_mass_params = [p for p in dense_mass_params
                                 if p in site_names]
            if dropped:
                fprint(f"dropped {len(dropped)} non-sampled sites from "
                       f"dense_mass_params: {dropped}")

        fprint(f"using dense mass matrix for {len(dense_mass_params)} "
               f"parameters: {list(dense_mass_params)}")
        return [tuple(dense_mass_params)]

    if not bool(kwargs.get("dense_mass", True)):
        fprint("using diagonal mass matrix.")
        return False

    if site_names is None:
        # Cannot introspect sites; fall back to NumPyro's default.
        fprint("using dense mass matrix over all parameters (site names "
               f"unavailable — no *{exclude_pattern}* auto-exclusion).")
        return True

    excluded = sorted(s for s in site_names if exclude_pattern in s)
    kept = sorted(s for s in site_names if exclude_pattern not in s)
    if excluded:
        fprint(f"auto-excluding {len(excluded)} *{exclude_pattern}* site(s) "
               f"from dense mass matrix: {excluded}")
    if len(kept) < 2:
        fprint(f"using diagonal mass matrix ({len(kept)} non-excluded site).")
        return False
    fprint(f"using dense mass matrix for {len(kept)} parameters: {kept}")
    return [tuple(kept)]


def _get_sample_site_names(model, model_kwargs, seed=42):
    """Trace the model and return the set of non-observed sample site names."""
    substituted_model = handlers.substitute(
        handlers.seed(model, rng_seed=seed),
        substitute_fn=init_to_median(num_samples=15),
    )
    model_trace = handlers.trace(substituted_model).get_trace(**model_kwargs)
    return {k for k, v in model_trace.items()
            if v["type"] == "sample" and not v.get("is_observed", False)}


def _setup_dense_mass(kwargs, init_params, model, model_kwargs,
                      site_names=None):
    """Resolve sample-site names and delegate to :func:`_parse_dense_mass`.

    Resolution order for ``site_names``: caller-supplied > keys of
    ``init_params`` (e.g. from L-BFGS) > tracing the model. Tracing is
    skipped when it would not be used — i.e. ``dense_mass=False`` with no
    explicit ``dense_mass_params`` / ``dense_mass_blocks``.
    """
    if site_names is None:
        needs_sites = (
            kwargs.get("dense_mass_params") is not None
            or kwargs.get("dense_mass_blocks") is not None
            or bool(kwargs.get("dense_mass", True))
        )
        if needs_sites:
            if init_params is not None:
                site_names = set(init_params.keys())
            else:
                site_names = _get_sample_site_names(model, model_kwargs)
    return _parse_dense_mass(kwargs, site_names=site_names)


def find_initial_point(model, model_kwargs, maxiter=100, seed=42,
                       return_site_names=False):
    """Run L-BFGS to find a reasonable MCMC starting point.

    Traces the model from the prior, maps to unconstrained space, runs
    scipy L-BFGS-B with JAX autodiff gradients, then maps back to
    constrained space. Returns None if the optimisation fails.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    model_kwargs : dict
        Keyword arguments passed when tracing and evaluating the model.
    maxiter : int, optional
        Maximum number of L-BFGS-B iterations.
    seed : int, optional
        Random seed used for the initial trace.
    return_site_names : bool, optional
        If True, always returns a tuple ``(init_params, site_names)`` where
        ``site_names`` is the set of non-observed sample-site names
        populated from the trace that was performed regardless. Allows
        downstream callers to avoid re-tracing the model when the
        optimisation fails. Default False (original single-return API).

    Returns
    -------
    dict or None, or tuple
        Constrained initial parameters, or None if optimisation diverges.
        When ``return_site_names`` is True, returns
        ``(init_params_or_none, site_names)``.
    """
    # Trace at the prior median to get transforms and a reasonable start
    substituted_model = handlers.substitute(
        handlers.seed(model, rng_seed=seed),
        substitute_fn=init_to_median(num_samples=15),
    )
    model_trace = handlers.trace(substituted_model).get_trace(**model_kwargs)

    transforms = {}
    init_constrained = {}
    for k, v in model_trace.items():
        if v["type"] == "sample" and not v.get("is_observed", False):
            transforms[k] = biject_to(v["fn"].support)
            init_constrained[k] = v["value"]
    site_names = set(init_constrained.keys())

    # Sort keys for consistent flattening
    keys = sorted(init_constrained.keys())
    shapes = {k: init_constrained[k].shape for k in keys}
    sizes = {k: int(np.prod(s)) for k, s in shapes.items()}

    def flatten(params_dict):
        return np.concatenate(
            [np.asarray(params_dict[k]).ravel() for k in keys])

    def unflatten(x):
        out, offset = {}, 0
        for k in keys:
            out[k] = jnp.asarray(x[offset:offset + sizes[k]]).reshape(
                shapes[k])
            offset += sizes[k]
        return out

    # Negative log-density in unconstrained space (include Jacobian)
    def neg_log_density(unc_params):
        constrained = {k: transforms[k](v) for k, v in unc_params.items()}
        ld = log_density(model, (), model_kwargs, constrained)[0]
        for k, t in transforms.items():
            ld = ld + jnp.sum(t.log_abs_det_jacobian(
                unc_params[k], constrained[k]))
        return -ld

    jit_val_grad = jit(jax.value_and_grad(
        lambda x_flat: neg_log_density(unflatten(x_flat))))

    def val_and_grad_numpy(x):
        v, g = jit_val_grad(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    unc0 = {k: transforms[k].inv(v) for k, v in init_constrained.items()}
    x0 = flatten(unc0)
    loss0 = float(jit(neg_log_density)(unc0))

    fprint(f"finding initial point (L-BFGS, maxiter={maxiter}) ...")
    result = sp_minimize(
        val_and_grad_numpy, x0, method="L-BFGS-B", jac=True,
        options={"maxiter": maxiter, "disp": False})

    fprint(f"  -log p: {loss0:.2f} -> {result.fun:.2f} "
           f"({result.nit} iters, {result.message})")

    if not np.isfinite(result.fun):
        fprint("  optimisation diverged, falling back to prior sample.")
        if return_site_names:
            return None, site_names
        return None

    unc_opt = unflatten(result.x)
    constrained_opt = {k: transforms[k](v) for k, v in unc_opt.items()}

    # Re-trace at the optimised point to get live support bounds
    # (important for parameters with dynamic bounds, e.g. r_ang
    # whose Uniform bounds depend on the sampled D_A).
    opt_model = handlers.substitute(
        handlers.seed(model, rng_seed=seed), data=constrained_opt)
    opt_trace = handlers.trace(opt_model).get_trace(**model_kwargs)

    n_clamped = 0
    eps = 1e-6
    for k, v in constrained_opt.items():
        site = opt_trace.get(k)
        if site is None:
            continue
        support = site["fn"].support
        lo = getattr(support, "lower_bound", None)
        hi = getattr(support, "upper_bound", None)
        if lo is not None or hi is not None:
            v_new = v
            if lo is not None:
                v_new = jnp.maximum(v_new, lo + eps)
            if hi is not None:
                v_new = jnp.minimum(v_new, hi - eps)
            if not jnp.array_equal(v_new, v):
                n_clamped += int(jnp.sum(v_new != v))
                constrained_opt[k] = v_new
    if n_clamped > 0:
        fprint(f"  clamped {n_clamped} parameters to support bounds.")

    if return_site_names:
        return constrained_opt, site_names
    return constrained_opt


def print_evidence(bic, aic, lnZ_laplace, err_lnZ_laplace,
                   lnZ_harmonic, err_lnZ_harmonic):
    fprint(f"BIC:          {bic:.2f}")
    fprint(f"AIC:          {aic:.2f}")
    fprint(f"Laplace lnZ:  {lnZ_laplace:.2f} +- {err_lnZ_laplace:.2f}")
    fprint(f"Harmonic lnZ: {lnZ_harmonic:.2f} +- {err_lnZ_harmonic:.2f}")


def run_pv_inference(model, model_kwargs, print_summary=True,
                     save_samples=True, return_original_samples=False,
                     init_maxiter=None, progress_bar=True):
    """
    Run MCMC inference on the given PV model.

    This function sets up the NumPyro NUTS kernel, optionally finds a
    starting point via L-BFGS, runs the chains, and performs
    post-processing (including evidence calculation and plotting).

    Parameters
    ----------
    model : BasePVModel or JointPVModel
        The forward model to run.
    model_kwargs : dict
        Keyword arguments passed to the model (e.g., ``data``).
    print_summary : bool, optional
        Whether to print a summary of the posterior samples.
    save_samples : bool, optional
        Whether to save samples, summary, and plots to disk.
    return_original_samples : bool, optional
        Whether to return the raw samples (including deterministic sites).
    init_maxiter : int or None, optional
        Maximum L-BFGS iterations for finding a starting point. If `None`,
        read from config.
    progress_bar : bool, optional
        Whether to show a progress bar during sampling.

    Returns
    -------
    samples : dict
        Post-processed posterior samples.
    log_density : jnp.ndarray or None
        Log-density of the samples.
    original_samples : dict, optional
        Raw samples, only returned if `return_original_samples` is True.
    """
    fsection("Inference")
    _setup_platform()

    kwargs = model.config["inference"]

    # Validate data if model supports it (e.g. CSPModel)
    if hasattr(model, "submodels"):
        # JointPVModel: validate each submodel with its corresponding data
        for submodel, data_i in zip(model.submodels, model_kwargs["data"]):
            if hasattr(submodel, "validate_data"):
                submodel.validate_data(data_i)
    elif hasattr(model, "validate_data"):
        model.validate_data(model_kwargs["data"])

    if init_maxiter is None:
        init_maxiter = kwargs.get("init_maxiter", 1000)

    if init_maxiter > 0:
        init_params, site_names = find_initial_point(
            model, model_kwargs, maxiter=init_maxiter,
            seed=kwargs["seed"], return_site_names=True)
        if init_params is not None:
            fprint("initialising NUTS from L-BFGS solution.")
            init_strategy = init_to_value(values=init_params)
        else:
            fprint("L-BFGS failed, initialising NUTS from prior median.")
            init_strategy = init_to_median(num_samples=5000)
    else:
        init_params = None
        site_names = None
        fprint("initialising NUTS from prior median.")
        init_strategy = init_to_median(num_samples=5000)

    dense_mass = _setup_dense_mass(kwargs, init_params, model, model_kwargs,
                                   site_names=site_names)
    kernel = NUTS(model, init_strategy=init_strategy, dense_mass=dense_mass)
    mcmc = MCMC(
        kernel, num_warmup=kwargs["num_warmup"],
        num_samples=kwargs["num_samples"], num_chains=kwargs["num_chains"],
        chain_method=kwargs["chain_method"],
        progress_bar=progress_bar)
    mcmc.run(jax.random.key(kwargs["seed"]), **model_kwargs)

    samples = mcmc.get_samples()
    auxiliary = extract_auxiliary(samples, ["Vpec_host_skipZ"])
    log_density_per_sample = samples.pop("log_density_per_sample", None)

    compute_log_density = kwargs["compute_log_density"]
    compute_evidence = model.compute_evidence
    if (compute_log_density or compute_evidence) and not _harmonic_available():
        fprint("[WARN] `harmonic` not installed — disabling post-sampling "
               "log-density and evidence computation. Install with "
               "`pip install harmonic` to re-enable.")
        compute_log_density = False
        compute_evidence = False

    if compute_log_density:
        log_density = get_log_density(samples, model, model_kwargs)
    else:
        log_density = None

    if return_original_samples:
        original_samples = deepcopy(samples)

    samples = drop_deterministic(samples)

    if compute_evidence:
        ndata = len(model_kwargs["data"])
        bic, aic = BIC_AIC(samples, log_density, ndata)

        samples_arr, names = dict_samples_to_array(samples)
        fprint(f"computing harmonic evidence from {len(names)} "
               f"parameters: {names}")

        samples_arr = samples_arr.reshape(
            kwargs["num_chains_harmonic"], -1, len(names))
        log_density_arr = log_density.reshape(
            kwargs["num_chains_harmonic"], -1)
        lnZ_laplace, err_lnZ_laplace = laplace_evidence(
            samples_arr, log_density_arr)
        lnZ_harmonic, err_lnZ_harmonic = harmonic_evidence(
            samples_arr, log_density_arr, epochs_num=50,
            return_flow_samples=False)
        err_lnZ_harmonic = np.mean(np.abs(err_lnZ_harmonic))

        print_evidence(bic, aic, lnZ_laplace, err_lnZ_laplace,
                       lnZ_harmonic, err_lnZ_harmonic)

        gof = {"BIC": bic, "AIC": aic,
               "lnZ_laplace": lnZ_laplace,
               "err_lnZ_laplace": err_lnZ_laplace,
               "lnZ_harmonic": lnZ_harmonic,
               "err_lnZ_harmonic": err_lnZ_harmonic}
    else:
        gof = None

    samples = postprocess_samples(samples)

    if print_summary:
        print_clean_summary(samples)

    if save_samples:
        fname_out = model.config["io"]["fname_output"]
        fprint(f"output directory is {dirname(fname_out)}.")
        save_mcmc_samples(
            samples, log_density, log_density_per_sample, gof, fname_out,
            auxiliary=auxiliary)

        fname_plot = splitext(fname_out)[0] + ".png"
        plot_corner(samples, show_fig=False, filename=fname_plot,)

        fname_summary = splitext(fname_out)[0] + "_summary.txt"
        with open(fname_summary, "w") as f:
            with contextlib.redirect_stdout(f):
                print_clean_summary(samples)
                if compute_evidence:
                    print_evidence(
                        bic, aic, lnZ_laplace, err_lnZ_laplace,
                        lnZ_harmonic, err_lnZ_harmonic)
        fprint(f"saved summary to {fname_summary}")

        if model.which_Vext == "radial":
            fname_plot = splitext(fname_out)[0] + "_corner_Vext_rad.png"
            plot_Vext_rad_corner(samples, show_fig=False, filename=fname_plot)

            fname_plot = splitext(fname_out)[0] + "_profile_Vext_rad.png"
            plot_radial_profiles(samples, model, show_fig=False,
                                 filename=fname_plot)
        elif model.which_Vext == "radial_magnitude":
            fname_plot = splitext(fname_out)[0] + "_profile_Vext_radmag.png"
            plot_Vext_radmag(
                samples, model, show_fig=False, filename=fname_plot,)

        if model.which_Vext == "per_pix":
            npix = samples["Vext_pix"].shape[1]
            if npix > 50:
                fprint(f"Skipping corner plot of Vext_pix with {npix} pixels.")
            else:
                fname_plot = splitext(fname_out)[0] + "_corner_Vext_pix.png"
                samples_Vext = {
                    f"Vext_pix_{i}": samples["Vext_pix"][:, i]
                    for i in range(npix)}
                plot_corner(samples_Vext, show_fig=False, filename=fname_plot,)

            fname_plot = splitext(fname_out)[0] + "_moll_Vext_pix.png"
            plot_Vext_moll(samples["Vext_pix"], fname_plot,)

    if return_original_samples:
        return samples, log_density, original_samples

    return samples, log_density


def run_H0_inference(model, model_kwargs=None, print_summary=True,
                     save_samples=True, init_maxiter=None,
                     progress_bar=True, return_diagnostics=False):
    """
    Run MCMC inference on an H0 model.

    This function sets up the NumPyro NUTS kernel, optionally finds a
    starting point via L-BFGS, runs the chains, and performs
    post-processing (including plotting and saving).

    Parameters
    ----------
    model : ModelBase
        The H0 model instance.
    model_kwargs : dict, optional
        Keyword arguments passed to the model (e.g., ``data``).
    print_summary : bool, optional
        Whether to print a summary of the posterior samples.
    save_samples : bool, optional
        Whether to save samples, summary, and plots to disk.
    init_maxiter : int or None, optional
        Maximum L-BFGS iterations for finding a starting point. If `None`,
        read from config.
    progress_bar : bool, optional
        Whether to show a progress bar during sampling.
    return_diagnostics : bool, optional
        Whether to also return NumPyro summary diagnostics computed from
        chain-grouped samples.

    Returns
    -------
    samples : dict
        Post-processed posterior samples.
    """
    if model_kwargs is None:
        model_kwargs = {}

    fsection("Inference")
    _setup_platform()

    kwargs = model.config["inference"]

    init_method = kwargs.get("init_method", "lbfgs")

    site_names = None
    if init_method == "sobol_adam":
        from .optimise import _use_de, find_MAP
        init_params = find_MAP(model, model_kwargs, seed=kwargs["seed"])
        method = "DE" if _use_de(model) else "Sobol+Adam"
        fprint(f"initialising NUTS from {method} MAP.")
        init_strategy = init_to_value(values=init_params)
    else:
        if init_maxiter is None:
            init_maxiter = kwargs.get("init_maxiter", 1000)

        if init_maxiter > 0:
            init_params, site_names = find_initial_point(
                model, model_kwargs, maxiter=init_maxiter,
                seed=kwargs["seed"], return_site_names=True)
            if init_params is not None:
                fprint("initialising NUTS from L-BFGS solution.")
                init_strategy = init_to_value(values=init_params)
            else:
                init_params = None
                fprint("L-BFGS failed, initialising NUTS from prior median.")
                init_strategy = init_to_median(num_samples=5000)
        else:
            init_params = None
            fprint("initialising NUTS from prior median.")
            init_strategy = init_to_median(num_samples=5000)

    dense_mass = _setup_dense_mass(kwargs, init_params, model, model_kwargs,
                                   site_names=site_names)
    max_tree_depth = kwargs.get("max_tree_depth", 10)
    target_accept_prob = kwargs.get("target_accept_prob", 0.8)
    kernel = NUTS(model, init_strategy=init_strategy, dense_mass=dense_mass,
                  max_tree_depth=max_tree_depth,
                  target_accept_prob=target_accept_prob)
    fprint(f"NUTS: max_tree_depth={max_tree_depth}, "
           f"target_accept_prob={target_accept_prob}")
    mcmc = MCMC(
        kernel, num_warmup=kwargs["num_warmup"],
        num_samples=kwargs["num_samples"], num_chains=kwargs["num_chains"],
        chain_method=kwargs["chain_method"],
        progress_bar=progress_bar)
    mcmc.run(jax.random.key(kwargs["seed"]), **model_kwargs)

    samples = mcmc.get_samples()
    diagnostic_summary = None
    if return_diagnostics:
        samples_by_chain = mcmc.get_samples(group_by_chain=True)
        samples_by_chain = drop_deterministic(samples_by_chain.copy())
        samples_by_chain = postprocess_samples(samples_by_chain)
        diagnostic_summary = summary_numpyro(samples_by_chain,
                                             group_by_chain=True)

    auxiliary = extract_auxiliary(samples, ["Vpec_host_skipZ"])
    samples = drop_deterministic(samples)
    samples = postprocess_samples(samples)

    if print_summary:
        print_clean_summary(samples)

    if save_samples:
        fname_out = model.config["io"]["fname_output"]
        fprint(f"output directory is {dirname(fname_out)}.")
        save_mcmc_samples(samples, None, None, None, fname_out,
                          auxiliary=auxiliary)

        fname_plot = splitext(fname_out)[0] + ".png"
        plot_corner(samples, show_fig=False, filename=fname_plot,)

        fname_summary = splitext(fname_out)[0] + "_summary.txt"
        with open(fname_summary, "w") as f:
            with contextlib.redirect_stdout(f):
                print_clean_summary(samples)
        fprint(f"saved summary to {fname_summary}")

    if return_diagnostics:
        return samples, diagnostic_summary
    return samples


def get_log_density(samples, model, model_kwargs, batch_size=5):
    """
    Compute the NumPyro model log density for posterior samples.

    Samples are evaluated in batches to avoid exhausting device memory.
    """
    def f(sample):
        return log_density(model, (), model_kwargs, sample)[0]

    f_vmap = jit(vmap(f))

    samples = {k: jnp.array(v) for k, v in samples.items()}
    num_samples = len(samples[next(iter(samples))])

    chunks = []
    for i in trange(0, num_samples, batch_size, desc="Batched log densities"):
        batch = {k: v[i:i + batch_size] for k, v in samples.items()}
        chunks.append(f_vmap(batch))

    return jnp.concatenate(chunks)


def extract_auxiliary(samples, keys):
    """
    Extract and remove auxiliary deterministic samples.

    Parameters
    ----------
    samples : dict
        Dictionary returned by `mcmc.get_samples()`.
    keys : sequence of str
        Names to extract. If a key ends with `_skipZ`, the suffix is dropped
        in the returned dictionary.
    """
    aux = {}
    for key in keys:
        if key in samples:
            new_key = key.replace("_skipZ", "")
            aux[new_key] = samples.pop(key)
    return aux


def drop_deterministic(samples, check_all_equals=True):
    """Drop deterministic and latent variable samples."""
    for key in list(samples.keys()):
        # Remove unused, latent variables used for deterministic sampling
        if "skipZ" in key:
            samples.pop(key)
            continue

        if key == "obs":
            samples.pop(key)
            continue

        # Remove samples fixed to a single value (delta prior)
        x = samples[key]
        if check_all_equals and np.all(x.flatten()[0] == x):
            samples.pop(key)
            continue

    keys = list(samples.keys())
    # Remove the a_TFR_h samples if a_TFR is present or rename it to a_TFR
    if "a_TFR" in keys and "a_TFR_h" in keys:
        samples.pop("a_TFR_h")
    elif "a_TFR_h" in keys and "a_TFR" not in keys:
        samples["a_TFR"] = samples.pop("a_TFR_h")

    return samples


def postprocess_samples(samples):
    """Postprocess MCMC samples."""
    # Collect all unique model prefixes (e.g., "Foundation/", "LOSS/")
    model_prefixes = set()
    for key in samples.keys():
        if "/" in key:
            model_prefixes.add(key.split("/")[0] + "/")
    model_prefixes.add("")  # Also handle unprefixed keys

    for model_prefix in model_prefixes:
        for prefix in ["Vext_rad", "Vext_radmag", "Vext",
                       "zeropoint_dipole"]:
            full_prefix = f"{model_prefix}{prefix}"
            # Spherical form: phi + cos_theta (+ mag optional)
            phi_key = f"{full_prefix}_phi"
            cos_theta_key = f"{full_prefix}_cos_theta"
            if phi_key in samples and cos_theta_key in samples:
                phi = np.rad2deg(samples.pop(phi_key))
                theta = np.arccos(samples.pop(cos_theta_key))
                dec = np.rad2deg(0.5 * np.pi - theta)

                ell, b = radec_to_galactic(phi, dec)
                samples[f"{full_prefix}_ell"] = ell
                samples[f"{full_prefix}_b"] = b

                mag_key = f"{full_prefix}_mag"
                if mag_key in samples:
                    samples[mag_key] = samples.pop(mag_key)
                continue

            # Cartesian form: x, y, z
            if all(f"{full_prefix}_{c}" in samples for c in "xyz"):
                x = samples.pop(f"{full_prefix}_x")
                y = samples.pop(f"{full_prefix}_y")
                z = samples.pop(f"{full_prefix}_z")

                r, ell, b = radec_cartesian_to_galactic(x, y, z)
                samples[f"{full_prefix}_mag"] = r
                samples[f"{full_prefix}_ell"] = ell
                samples[f"{full_prefix}_b"] = b

    # Quadrupole postprocessing: convert q1/q2 from (phi, cos_theta) to
    # galactic coordinates (ell, b).
    for model_prefix in model_prefixes:
        for qn in ("q1", "q2"):
            phi_key = f"{model_prefix}Vext_quad_{qn}_phi"
            cos_theta_key = f"{model_prefix}Vext_quad_{qn}_cos_theta"
            if phi_key in samples and cos_theta_key in samples:
                ra = np.rad2deg(samples.pop(phi_key))
                theta = np.arccos(samples.pop(cos_theta_key))
                dec = np.rad2deg(0.5 * np.pi - theta)
                ell, b = radec_to_galactic(ra, dec)
                samples[f"{model_prefix}Vext_quad_{qn}_ell"] = ell
                samples[f"{model_prefix}Vext_quad_{qn}_b"] = b

    # Octupole postprocessing: convert q1/q2/q3 from (phi, cos_theta) to
    # galactic coordinates (ell, b).
    for model_prefix in model_prefixes:
        for qn in ("q1", "q2", "q3"):
            phi_key = f"{model_prefix}Vext_oct_{qn}_phi"
            cos_theta_key = f"{model_prefix}Vext_oct_{qn}_cos_theta"
            if phi_key in samples and cos_theta_key in samples:
                ra = np.rad2deg(samples.pop(phi_key))
                theta = np.arccos(samples.pop(cos_theta_key))
                dec = np.rad2deg(0.5 * np.pi - theta)
                ell, b = radec_to_galactic(ra, dec)
                samples[f"{model_prefix}Vext_oct_{qn}_ell"] = ell
                samples[f"{model_prefix}Vext_oct_{qn}_b"] = b

    return samples


def print_clean_summary(samples):
    """Wrapper around numpyro's `print_summary`."""
    samples_print = {}
    for key, x in samples.items():
        if "_latent" in key:
            continue

        samples_print[key] = x[None, ...]

    print_summary_numpyro(samples_print,)


def save_mcmc_samples(samples, log_density, log_density_per_sample, gof,
                      filename, auxiliary=None):
    """Save the MCMC samples to an HDF5 file."""
    out_dir = dirname(filename)
    if out_dir:
        makedirs(out_dir, exist_ok=True)
    with File(filename, 'w') as f:
        grp = f.create_group("samples")
        for key, x in samples.items():
            grp.create_dataset(key, data=x, dtype=np.float32)

        if auxiliary and "Vpec_host" in auxiliary:
            grp_aux = f.create_group("auxiliary")
            grp_aux.create_dataset(
                "Vpec_host", data=auxiliary["Vpec_host"],
                dtype=np.float32)

        if "Vext_ell" in samples:
            original_shape = samples["Vext_ell"].shape
            ra, dec = galactic_to_radec(
                samples["Vext_ell"].ravel(), samples["Vext_b"].ravel())
            Vext = (samples["Vext_mag"].ravel()[:, None]
                    * radec_to_cartesian(ra, dec))
            grp.create_dataset(
                "Vext", data=Vext.reshape(*original_shape, 3))

        if log_density is not None:
            f.create_dataset("log_density", data=log_density)

        if log_density_per_sample is not None:
            f.create_dataset(
                "log_density_per_sample", data=log_density_per_sample)

        if gof is not None:
            grp = f.create_group("gof")
            for key, x in gof.items():
                grp.create_dataset(key, data=x)

    fprint(f"saved samples to {filename}")
