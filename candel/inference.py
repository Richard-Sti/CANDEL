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
from .util import (fprint, galactic_to_radec, plot_corner, radec_to_cartesian,
                   radec_to_galactic)


def run_pv_inference(model, model_args, print_summary=True, save_samples=True):
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
        num_samples=kwargs["num_samples"], num_chains=kwargs["num_chains"],)
    mcmc.run(jax.random.key(kwargs["seed"]), *model_args)

    samples = mcmc.get_samples()
    if kwargs["compute_log_density"]:
        log_density = get_log_density(samples, model, model_args)
        log_density = log_density.reshape(kwargs["num_chains"], -1)
    else:
        log_density = None

    samples = mcmc.get_samples(group_by_chain=True)
    samples = postprocess_samples(samples)

    if print_summary:
        print_clean_summary(samples)

    if model.config["inference"]["compute_evidence"]:
        bic, aic = BIC_AIC(samples, log_density)

        samples_arr, names = dict_samples_to_array(
            samples, stack_chains=True)
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

        fprint(f"BIC:          {bic:.2f}")
        fprint(f"AIC:          {aic:.2f}")
        fprint(f"Laplace lnZ:  {lnZ_laplace:.2f} +- {err_lnZ_laplace:.2f}")
        fprint(f"Harmonic lnZ: {lnZ_harmonic:.2f} +- {err_lnZ_harmonic:.2f}")

        gof = {"BIC": bic, "AIC": aic,
               "lnZ_laplace": lnZ_laplace,
               "err_lnZ_laplace": err_lnZ_laplace,
               "lnZ_harmonic": lnZ_harmonic,
               "err_lnZ_harmonic": err_lnZ_harmonic}
    else:
        gof = None

    if save_samples:
        fname_out = model.config["io"]["fname_output"]
        fprint(f"output directory is `{dirname(fname_out)}`.")
        save_mcmc_samples(samples, log_density, gof, fname_out)

        fname_plot = splitext(fname_out)[0] + ".png"
        plot_corner(samples, show_fig=False, filename=fname_plot,)

    return samples, log_density


def run_magsel_inference(model, model_args, num_warmup=1000, num_samples=5000,
                         seed=42, print_summary=True,):
    """Run MCMC inference on the given magnitude selection model."""
    kernel = NUTS(model, init_strategy=init_to_median(num_samples=5000))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, )
    mcmc.run(jax.random.key(seed), *model_args)

    if print_summary:
        mcmc.print_summary()

    return mcmc.get_samples()


def get_log_density(samples, model, model_args, batch_size=5):
    """
    Compute the log density of the peculiar velocity validation model. The
    batch size cannot be much larger to prevent exhausting the memory.
    """
    def f(sample):
        return log_density(model, model_args, {}, sample)[0]

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


def postprocess_samples(samples):
    """
    Postprocess the samples from the MCMC run. Removes unused latent variables,
    and converts Vext samples to galactic coordinates.
    """
    for key in list(samples.keys()):
        # Remove unused, latent variables used for deterministic sampling
        if "skipZ" in key:
            samples.pop(key,)
            continue

        # Remove samples fixed to a single value (delta prior)
        x = samples[key]
        if np.all(x.flatten()[0] == x):
            samples.pop(key,)
            continue

    keys = list(samples.keys())
    # Remove the a_TFR_h samples if a_TFR is present or rename it to a_TFR
    if "a_TFR" in keys and "a_TFR_h" in keys:
        samples.pop("a_TFR_h",)
    elif "a_TFR_h" in keys and "a_TFR" not in keys:
        samples["a_TRF"] = samples.pop("a_TFR_h",)

    # Convert the Vext samples to galactic coordinates
    if any("Vext" in key for key in samples.keys()):
        ell, b = radec_to_galactic(
            np.rad2deg(samples.pop("Vext_phi")),
            np.rad2deg(0.5 * np.pi - np.arccos(samples.pop("Vext_cos_theta"))),)  # noqa
        samples["Vext_mag"] = samples.pop("Vext_mag")
        samples["Vext_ell"] = ell
        samples["Vext_b"] = b

    # Convert aTFR dipole samples to galactic coordinates
    if any("a_TFR_dipole" in key for key in samples.keys()):
        ell, b = radec_to_galactic(
            np.rad2deg(samples.pop("a_TFR_dipole_phi")),
            np.rad2deg(0.5 * np.pi - np.arccos(samples.pop("a_TFR_dipole_cos_theta"))),)  # noqa
        samples["a_TFR_dipole_mag"] = samples.pop("a_TFR_dipole_mag")
        samples["a_TFR_dipole_ell"] = ell
        samples["a_TFR_dipole_b"] = b

    return samples


def print_clean_summary(samples):
    """Wrapper around numpyro's `print_summary`."""
    samples_print = {}
    for key, x in samples.items():
        if "_latent" in key:
            continue

        samples_print[key] = x

    print_summary_numpyro(samples_print,)


def save_mcmc_samples(samples, log_density, gof, filename):
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

        if gof is not None:
            grp = f.create_group("gof")
            for key, x in gof.items():
                grp.create_dataset(key, data=x)

    fprint(f"saved samples to `{filename}`.")
