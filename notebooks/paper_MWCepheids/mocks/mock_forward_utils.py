"""Thin wrapper: forward model mock data generation via CepheidData.

Uses the same generator and true parameters as the simplified model
(mock_utils.py), wrapping the output in CepheidData for MWCepheidModel.
"""
import numpy as np

from mock_utils import (
    DEFAULT_CONFIGS, EPSILON_OH, TRUE_PARAMS, generate_one_campaign)

# Forward-model campaign configs (references to DEFAULT_CONFIGS)
MOCK_CFG_MW = DEFAULT_CONFIGS["C22"]
MOCK_CFG_PI = DEFAULT_CONFIGS["C27"]


def generate_mock_forward(seed, true_params, mock_cfg, campaign):
    """Generate one mock dataset for the full forward model.

    Returns
    -------
    data : CepheidData
        From CepheidData.from_arrays(), ready for run_inference().
    n_parent : int
        Parent sample size before selection.
    n_selected : int
        Number of stars after selection.
    """
    from candel.pvdata.mwcepheids import CepheidData

    rng = np.random.default_rng(seed)
    result = generate_one_campaign(rng, mock_cfg, true_params)
    sel = result["sel"]

    n_selected = int(sel.sum())
    if n_selected == 0:
        raise RuntimeError(
            f"No stars survived selection (seed={seed}, campaign={campaign})")

    data = CepheidData.from_arrays(
        logP=result["logP"][sel],
        mW_H=result["m_obs"][sel],
        mW_H_err=result["sigma_m"][sel],
        OH=result["OH_obs"][sel],
        pi_EDR3=result["varpi_obs"][sel],
        pi_EDR3_err=result["sigma_varpi"][sel],
        ell=result["ell"][sel],
        b=result["b"][sel],
        campaign=campaign,
    )
    return data, mock_cfg["N_parent"], n_selected
