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
import tomllib

import jax
import numpy as np
from numpyro.diagnostics import print_summary as print_summary_numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median

from .util import radec_to_galactic


def load_inference_config(config_file):
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return config["inference"]


def run_inference(model, model_args, config_path, print_summary=True, ):
    """Run MCMC inference on the given model."""
    kwargs = load_inference_config(config_path)

    kernel = NUTS(model, init_strategy=init_to_median(num_samples=5000))
    mcmc = MCMC(
        kernel, num_warmup=kwargs["num_warmup"],
        num_samples=kwargs["num_samples"], num_chains=kwargs["num_chains"],)
    mcmc.run(jax.random.key(kwargs["seed"]), *model_args)

    samples = mcmc.get_samples(group_by_chain=True)
    samples = postprocess_samples(samples)

    if print_summary:
        print_clean_summary(samples)

    return samples


def postprocess_samples(samples):
    """
    Postprocess the samples from the MCMC run. Removes unused latent variables,
    and converts Vext samples to galactic coordinates.
    """
    # Remove unused, latent variables used for deterministic sampling
    for key in list(samples.keys()):
        if "skipZ" in key:
            samples.pop(key,)

    # Convert the Vext samples to galactic coordinates
    if any("Vext" in key for key in samples.keys()):
        ell, b = radec_to_galactic(
            np.rad2deg(samples.pop("Vext_phi")),
            np.rad2deg(0.5 * np.pi - np.arccos(samples.pop("Vext_cos_theta"))),)  # noqa
        samples["Vext_mag"] = samples.pop("Vext_mag")
        samples["Vext_ell"] = ell
        samples["Vext_b"] = b

    return samples


def print_clean_summary(samples):
    """Wrapper around numpyro's `print_summary`."""
    samples_print = {}
    for key, x in samples.items():
        if "_latent" in key:
            continue

        samples_print[key] = x

    print_summary_numpyro(samples_print,)
