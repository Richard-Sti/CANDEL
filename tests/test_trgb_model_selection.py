import numpy as np
import pytest

from candel.model import TRGBModel


def _write_trgb_config(tmp_path, selection):
    config_path = tmp_path / f"config_{selection}.toml"
    config_path.write_text(
        f"""
root_main = "."
root_data = "."
root_results = "."

[inference]
compute_evidence = false

[model]
which_run = "EDD_TRGB"
which_selection = "{selection}"
use_reconstruction = false
use_TRGB_host_redshift = true
distmod_limits = [10.0, 38.0]
distmod_limits_LMC = [17.0, 20.0]
distmod_limits_N4258 = [28.0, 31.0]
r_limits_malmquist = [0.1, 10.0]
num_points_malmquist = 5
mag_min_TRGB = 22.1
mag_lim_TRGB = 26.0
mag_lim_TRGB_width = 0.5

[model.priors.H0]
dist = "uniform"
low = 40.0
high = 100.0

[model.priors.M_TRGB]
dist = "normal"
loc = -4.05
scale = 0.5

[model.priors.sigma_int]
dist = "maxwell"
scale = 0.0627

[model.priors.sigma_v]
dist = "maxwell"
scale = 187.997121

[model.priors.Vext]
dist = "delta"
value = [0.0, 0.0, 0.0]
""",
        encoding="utf-8")
    return config_path


def _minimal_trgb_data():
    return {
        "RA_host": np.array([0.0]),
        "dec_host": np.array([0.0]),
        "mag_obs": np.array([25.0]),
        "e_mag_obs": np.array([0.05]),
        "czcmb": np.array([500.0]),
        "e_czcmb": np.array([10.0]),
        "e_mag_median": 0.05,
        "mu_LMC_anchor": 18.477,
        "e_mu_LMC_anchor": 0.026,
        "mag_LMC_TRGB": 14.456,
        "e_mag_LMC_TRGB": 0.018,
        "mu_N4258_anchor": 29.398,
        "e_mu_N4258_anchor": 0.032,
        "mag_N4258_TRGB": 25.347,
        "e_mag_N4258_TRGB": 0.0443,
        "has_volume_density_3d": False,
    }


@pytest.mark.parametrize(
    "selection", ["redshift", "TRGB_magnitude_redshift"])
def test_trgb_model_rejects_redshift_selection(tmp_path, selection):
    config_path = _write_trgb_config(tmp_path, selection)

    with pytest.raises(ValueError, match="Unknown `which_selection`"):
        TRGBModel(config_path, _minimal_trgb_data())
