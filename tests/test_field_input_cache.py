from scripts.preprocess.field_input_cache import (
    _h0_velocity_key, _variant_action)


def test_reconstructed_no_selection_ch0_still_warms_3d_cache():
    config = {
        "model": {
            "which_run": "CH0",
            "which_selection": None,
            "use_reconstruction": True,
        },
    }

    assert _variant_action(config) == "check/cache"


def test_reconstructed_no_selection_cchp_skips_3d_cache():
    config = {
        "model": {
            "which_run": "CCHP",
            "which_selection": None,
            "use_reconstruction": True,
        },
    }

    assert _variant_action(config) == "skip 3D"


def test_reconstructed_no_selection_trgb_string_none_warms_3d_cache():
    config = {
        "model": {
            "which_run": "EDD_TRGB",
            "which_selection": "none",
            "use_reconstruction": True,
        },
    }

    assert _variant_action(config) == "check/cache"


def test_edd_trgb_redshift_selection_does_not_request_velocity_cache():
    config = {
        "model": {
            "which_run": "EDD_TRGB",
            "which_selection": "redshift",
        },
    }

    assert _h0_velocity_key(config) == "density"
