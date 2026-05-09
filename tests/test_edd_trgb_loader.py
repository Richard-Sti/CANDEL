from pathlib import Path

import numpy as np

from candel.pvdata.data import (
    SPEED_OF_LIGHT,
    load_EDD_TRGB,
    load_EDD_TRGB_grouped,
)


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "EDD_TRGB"


def test_edd_trgb_loader_matches_sample_summary():
    data = load_EDD_TRGB(str(DATA_ROOT))

    assert len(data["mag"]) == 445
    assert np.all(np.isfinite(data["mag"]))
    assert np.all(np.isfinite(data["colour_dered"]))
    assert np.all(np.abs(data["zcmb"] * SPEED_OF_LIGHT) < 9999)
    assert "M_TRGB" not in data
    assert "DM_tip" not in data

    # First retained row, UGC12894: T814 - A_814 = 25.83 - 0.166.
    # This differs from the EDD colour-corrected DM_tip - 4.06 value.
    assert np.isclose(data["mag"][0], 25.83 - 0.166)
    assert not np.isclose(data["mag"][0], 29.78 - 4.06)
    assert np.isclose(data["colour_dered"][0], 0.98)


def test_edd_trgb_grouped_loader_matches_sample_summary():
    data = load_EDD_TRGB_grouped(str(DATA_ROOT))

    assert len(data["mag"]) == 273
    assert len(data["czcmb_group"]) == 273
    assert np.all(np.isfinite(data["mag"]))
    assert np.all(np.isfinite(data["colour_dered"]))
    assert np.all(np.abs(data["zcmb"] * SPEED_OF_LIGHT) < 9999)
    assert np.all(np.isfinite(data["czcmb_group"]))
    assert "M_TRGB" not in data
    assert "DM_tip" not in data

    # First retained row, UGC12894: T814 - A_814 = 25.83 - 0.166.
    assert np.isclose(data["mag"][0], 25.83 - 0.166)
    assert np.isclose(data["colour_dered"][0], 0.98)
