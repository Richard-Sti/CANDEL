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
"""Evidence and BIC/AIC calculation for MCMC samples."""
import numpy as np

from .util import fprint


def BIC_AIC(samples, log_density, stack_chains=True):
    """
    Get the BIC/AIC of HMC samples from a Numpyro model.

    Parameters
    ----------
    samples: dict
        Dictionary of samples from the Numpyro MCMC object.
    log_density: numpy array
        Log density of the samples.
    stack_chains: bool, optional
        Whether to stack the chains. Must be `True` if the samples come from
        multiple chains.

    Returns
    -------
    BIC, AIC: floats
    """
    if stack_chains and log_density.ndim == 1:
        stack_chains = False

    if stack_chains:
        log_density = log_density.reshape(-1, *log_density.shape[2:])

    kmax = np.argmax(log_density)

    # How many parameters?
    nparam = 0
    for x in samples.values():
        if stack_chains:
            x = x.reshape(-1, *x.shape[2:])

        if x.ndim == 1:
            nparam += 1
        else:
            # The first dimension is the number of steps.
            nparam += np.prod(x.shape[1:])

        ndata = x.shape[-1]

    fprint(f"found {nparam} parameters and {ndata} data points.")

    BIC = nparam * np.log(ndata) - 2 * log_density[kmax]
    AIC = 2 * nparam - 2 * log_density[kmax]

    return float(BIC), float(AIC)

def dict_samples_to_array(samples, stack_chains=True):
    """Convert a dictionary of samples to a 2-dimensional array."""
    data = []
    names = []

    for key, x in samples.items():
        if stack_chains:
            x = x.reshape(-1, *x.shape[2:])

        if x.ndim == 1:
            data.append(x)
            names.append(key)
        elif x.ndim == 2:
            for i in range(x.shape[-1]):
                data.append(x[:, i])
                names.append(f"{key}_{i}")
        elif x.ndim == 3:
            for i in range(x.shape[-1]):
                for j in range(x.shape[-2]):
                    data.append(x[:, j, i])
                    names.append(f"{key}_{j}_{i}")
        else:
            raise ValueError("Invalid dimensionality of samples to stack.")

    return np.vstack(data).T, names


def harmonic_evidence(samples_arr, log_density, temperature=0.8, epochs_num=20,
                      return_flow_samples=True, verbose=True):
    """
    Calculate the evidence using the `harmonic` package. The model has a few
    more hyperparameters that are set to defaults now.

    Parameters
    ----------
    samples_arr: 3-dimensional array
        MCMC samples of shape `(nchains, nsamples, ndim)`.
    log_density: 2-dimensional array
        Log posterior xs of shape `(nchains, nsamples)`.
    temperature: float, optional
        Temperature of the `harmonic` model.
    epochs_num: int, optional
        Number of epochs for training the model.
    return_flow_samples: bool, optional
        Whether to return the flow samples.
    verbose: bool, optional
        Whether to print progress.

    Returns
    -------
    ln_evidence, err_ln_inv_evidence: float and tuple of floats
        The log evidence and its error.
    flow_samples: 2-dimensional array, optional
        Flow samples of shape `(nsamples, ndim)`. To check their agreement
        with the input samples.
    """
    try:
        import harmonic as hm
    except ImportError as e:
        raise ImportError("The `harmonic` package is required to "
                          "calculate the evidence.") from e

    # Do some standard checks of inputs.
    if samples_arr.ndim != 3:
        raise ValueError("The samples_arr must be a 3-dimensional array of "
                         "shape `(nchains, nsamples_arr, ndim)`.")

    if log_density.ndim != 2 and log_density.shape[:2] != samples_arr.shape[:2]:  # noqa
        raise ValueError("The log posterior must be a 2-dimensional "
                         "array of shape `(nchains, nsamples_arr)`.")

    ndim = samples_arr.shape[-1]
    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples_arr, log_density)
    chains_train, chains_infer = hm.utils.split_data(
        chains, training_proportion=0.5)

    # This has a few more hyperparameters that are set to defaults now.
    model = hm.model.RQSplineModel(
        ndim, standardize=True, temperature=temperature)
    model.fit(chains_train.samples, epochs=epochs_num, verbose=verbose)

    ev = hm.Evidence(chains_infer.nchains, model)
    ev.add_chains(chains_infer)
    ln_evidence = -ev.ln_evidence_inv
    err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()

    if return_flow_samples:
        flow_samples = model.sample(samples_arr.shape[0])

        return ln_evidence, err_ln_inv_evidence, flow_samples

    return ln_evidence, err_ln_inv_evidence


def laplace_evidence(samples_array, log_density):
    """
    Calculate the evidence using the Laplace approximation. Calculates
    the mean and error of the log evidence estimated from the chains.

    Parameters
    ----------
    samples_array: 3-dimensional array
        MCMC samples of shape `(nchains, nsamples, ndim)`.
    log_density: 2-dimensional array
        Log posterior xs of shape `(nchains, nsamples)`.

    Returns
    -------
    mean_ln_inv_evidence, err_ln_inv_evidence: two floats
    """
    if samples_array.ndim != 3:
        raise ValueError("The samples_array must be a 3-dimensional array of "
                         "shape `(nchains, nsamples, ndim)`.")

    if log_density.ndim != 2:
        raise ValueError("The log_density must be a 2-dimensional array of "
                         "shape `(nchains, nsamples)`.")


    nchains = len(samples_array)
    ndim = samples_array.shape[-1]
    logZ = np.full(nchains, np.nan)

    for n in range(nchains):
        C = np.cov(samples_array[0], rowvar=False)
        lp_max = np.max(log_density[n])

        logZ[n] = (lp_max + 0.5 * (np.sum(np.log(np.linalg.eigvalsh(C)))
                                   + ndim * np.log(2 * np.pi)))

    return np.mean(logZ), np.std(logZ)

