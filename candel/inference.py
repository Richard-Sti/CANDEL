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
from os.path import dirname, splitext

import jax
import numpy as np
from h5py import File
from jax import jit
from jax import numpy as jnp
from jax import vmap
from numpyro import set_platform
from numpyro.diagnostics import print_summary as print_summary_numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
from numpyro.infer.util import log_density
from tqdm import trange

from .evidence import (BIC_AIC, dict_samples_to_array, harmonic_evidence,
                       laplace_evidence)
from .util import (fprint, galactic_to_radec, plot_corner,
                   plot_radial_profiles, plot_Vext_rad_corner,
                   plot_Vext_moll, radec_cartesian_to_galactic,
                   radec_to_cartesian, radec_to_galactic)


def print_evidence(bic, aic, lnZ_laplace, err_lnZ_laplace,
                   lnZ_harmonic, err_lnZ_harmonic):
    fprint(f"BIC:          {bic:.2f}")
    fprint(f"AIC:          {aic:.2f}")
    fprint(f"Laplace lnZ:  {lnZ_laplace:.2f} +- {err_lnZ_laplace:.2f}")
    fprint(f"Harmonic lnZ: {lnZ_harmonic:.2f} +- {err_lnZ_harmonic:.2f}")


def run_pv_inference(model, model_kwargs, print_summary=True,
                     save_samples=True, return_original_samples=False):
    """
    Run MCMC inference on the given PV model, post-process the samples,
    optionally compute the BIC, AIC, evidence and save the samples to an
    HDF5 file.
    """
    devices = jax.devices()
    device_str = ", ".join(f"{d.device_kind}({d.platform})" for d in devices)
    fprint(f"running inference on devices: {device_str}")

    if any(d.platform == "gpu" for d in devices):
        set_platform("gpu")
        fprint("using NumPyro platform: GPU")
    else:
        set_platform("cpu")
        fprint("using NumPyro platform: CPU")

    kwargs = model.config["inference"]

    kernel = NUTS(model, init_strategy=init_to_median(num_samples=5000))
    mcmc = MCMC(
        kernel, num_warmup=kwargs["num_warmup"],
        num_samples=kwargs["num_samples"], num_chains=kwargs["num_chains"],
        chain_method=kwargs["chain_method"])
    mcmc.run(jax.random.key(kwargs["seed"]), **model_kwargs)

    samples = mcmc.get_samples()
    log_density_per_sample = samples.pop("log_density_per_sample", None)

    if kwargs["compute_log_density"]:
        log_density = get_log_density(samples, model, model_kwargs)
    else:
        log_density = None

    if return_original_samples:
        original_samples = deepcopy(samples)

    samples = drop_deterministic(samples)

    compute_evidence = model.config["inference"]["compute_evidence"]
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
            samples, log_density, log_density_per_sample, gof, fname_out)

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


def run_SH0ES_inference(model, model_kwargs={}, print_summary=True,
                        save_samples=True):
    """
    Run MCMC inference on the SH0ES model, post-process the samples,
    plot the corner plot and optionally save the samples to an HDF5 file.
    """
    devices = jax.devices()
    device_str = ", ".join(f"{d.device_kind}({d.platform})" for d in devices)
    fprint(f"running inference on devices: {device_str}")

    if any(d.platform == "gpu" for d in devices):
        set_platform("gpu")
        fprint("using NumPyro platform: GPU")
    else:
        set_platform("cpu")
        fprint("using NumPyro platform: CPU")

    kwargs = model.config["inference"]

    kernel = NUTS(model, init_strategy=init_to_median(num_samples=5000))
    mcmc = MCMC(
        kernel, num_warmup=kwargs["num_warmup"],
        num_samples=kwargs["num_samples"], num_chains=kwargs["num_chains"],
        chain_method=kwargs["chain_method"])
    mcmc.run(jax.random.key(kwargs["seed"]), **model_kwargs)

    samples = mcmc.get_samples()
    samples = drop_deterministic(samples)
    samples = postprocess_samples(samples)

    if print_summary:
        print_clean_summary(samples)

    if save_samples:
        fname_out = model.config["io"]["fname_output"]
        fprint(f"output directory is {dirname(fname_out)}.")
        save_mcmc_samples(samples, None, None, fname_out)

        fname_plot = splitext(fname_out)[0] + ".png"
        plot_corner(samples, show_fig=False, filename=fname_plot,)

        fname_summary = splitext(fname_out)[0] + "_summary.txt"
        with open(fname_summary, "w") as f:
            with contextlib.redirect_stdout(f):
                print_clean_summary(samples)
        fprint(f"saved summary to {fname_summary}")

    return samples


def get_log_density(samples, model, model_kwargs, batch_size=5):
    """
    Compute the log density of NumPyro model. The batch size cannot be much
    larger to prevent exhausting the memory.
    """
    def f(sample):
        return log_density(model, (), model_kwargs, sample)[0]

    f_vmap = vmap(f)
    f_vmap = jit(f_vmap)

    samples = {k: jnp.array(v) for k, v in samples.items()}
    num_samples = len(samples[next(iter(samples))])
    log_densities = jnp.zeros((num_samples,))

    for i in trange(0, num_samples, batch_size, desc="Batched log densities"):
        batch = {k: v[i:i + batch_size] for k, v in samples.items()}
        batch_log_densities = f_vmap(batch)

        log_densities = log_densities.at[i:i+batch_size].set(
            batch_log_densities)

    return log_densities


def drop_deterministic(samples, check_all_equals=True):
    """Drop deterministic and latent variable samples."""
    for key in list(samples.keys()):
        # Remove unused, latent variables used for deterministic sampling
        if "skipZ" in key:
            samples.pop(key,)
            continue

        if key == "obs":
            samples.pop(key,)
            continue

        # Remove samples fixed to a single value (delta prior)
        x = samples[key]
        if check_all_equals and np.all(x.flatten()[0] == x):
            samples.pop(key,)
            continue

    keys = list(samples.keys())
    # Remove the a_TFR_h samples if a_TFR is present or rename it to a_TFR
    if "a_TFR" in keys and "a_TFR_h" in keys:
        samples.pop("a_TFR_h",)
    elif "a_TFR_h" in keys and "a_TFR" not in keys:
        samples["a_TRF"] = samples.pop("a_TFR_h",)

    return samples


def postprocess_samples(samples):
    """Postprocess MCMC samples."""
    for prefix in ["dH0", "Vext_rad", "Vext", "zeropoint_dipole"]:
        # Spherical form: phi + cos_theta (+ mag optional)
        if f"{prefix}_phi" in samples and f"{prefix}_cos_theta" in samples:
            phi = np.rad2deg(samples.pop(f"{prefix}_phi"))
            theta = np.arccos(samples.pop(f"{prefix}_cos_theta"))
            dec = np.rad2deg(0.5 * np.pi - theta)

            ell, b = radec_to_galactic(phi, dec)
            samples[f"{prefix}_ell"] = ell
            samples[f"{prefix}_b"] = b

            if f"{prefix}_mag" in samples:
                samples[f"{prefix}_mag"] = samples.pop(f"{prefix}_mag")
            continue

        # Cartesian form: x, y, z
        if all(f"{prefix}_{c}" in samples for c in "xyz"):
            x = samples.pop(f"{prefix}_x")
            y = samples.pop(f"{prefix}_y")
            z = samples.pop(f"{prefix}_z")

            r, ell, b = radec_cartesian_to_galactic(x, y, z)
            samples[f"{prefix}_mag"] = r
            samples[f"{prefix}_ell"] = ell
            samples[f"{prefix}_b"] = b

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
                      filename):
    """Save the MCMC samples to an HDF5 file."""
    with File(filename, 'w') as f:
        grp = f.create_group("samples")
        for key, x in samples.items():
            grp.create_dataset(key, data=x, dtype=np.float32)

        try:
            ndim = samples["Vext_ell"].ndim
            if ndim > 1:
                Vext_ell = samples["Vext_ell"].reshape(-1)
                Vext_b = samples["Vext_b"].reshape(-1)
                Vext_mag = samples["Vext_mag"].reshape(-1,)
                original_shape = samples["Vext_ell"].shape
            else:
                Vext_ell = samples["Vext_ell"]
                Vext_b = samples["Vext_b"]
                Vext_mag = samples["Vext_mag"]

            ra, dec = galactic_to_radec(Vext_ell, Vext_b)
            Vext = Vext_mag[:, None] * radec_to_cartesian(ra, dec)

            if ndim > 1:
                Vext = Vext.reshape(*original_shape, 3)
            grp.create_dataset("Vext", data=Vext)
        except KeyError:
            pass

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
