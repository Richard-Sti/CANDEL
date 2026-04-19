"""Tests for the --f-grid scaling helper in run_maser_disk.py.

These are pure-Python tests — they import the helper and the key list
from the runner script by path, and exercise the rewrite logic on a
hand-built config dict. No JAX, no model instantiation.
"""
import os

_RUNNER = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "scripts", "megamaser", "run_maser_disk.py",
)


def _load_helpers():
    """Load _round_to_odd and _FGRID_KEYS without running the script.

    The runner is a top-level script that executes argparse and loads
    TOML at import. To unit-test its helpers we read the source and
    exec() only the two helper definitions in a fresh namespace.
    """
    with open(_RUNNER) as f:
        src = f.read()
    ns = {}
    start = src.index("# >>> f-grid helpers")
    end = src.index("# <<< f-grid helpers", start)
    exec(src[start:end], ns)
    return ns["_round_to_odd"], ns["_FGRID_KEYS"]


def test_round_to_odd_identity_at_one():
    r, _ = _load_helpers()
    for n in [3, 101, 251, 301, 501, 1001, 2001, 3001, 20001]:
        assert r(n, 1.0) == n


def test_round_to_odd_half():
    r, _ = _load_helpers()
    # 1001 * 0.5 = 500.5 -> round -> 500 (even) -> 501
    assert r(1001, 0.5) == 501
    # 301 * 0.5 = 150.5 -> round -> 150 (Python banker's) or 151,
    # either way the result must be odd and within 1 of 150.5
    out = r(301, 0.5)
    assert out % 2 == 1
    assert abs(out - 150.5) <= 1.5


def test_round_to_odd_double():
    r, _ = _load_helpers()
    # 251 * 2 = 502 (even) -> 503
    assert r(251, 2.0) == 503
    # 1001 * 2 = 2002 -> 2003
    assert r(1001, 2.0) == 2003


def test_round_to_odd_floor_three():
    r, _ = _load_helpers()
    assert r(5, 0.01) == 3
    assert r(3, 0.0) == 3
    assert r(1, 1.0) == 3


def test_round_to_odd_always_odd():
    r, _ = _load_helpers()
    for n in [3, 5, 101, 251, 301, 501, 1001, 2001, 3001, 20001]:
        for f in [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 3.1, 5.0]:
            out = r(n, f)
            assert out % 2 == 1, (n, f, out)
            assert out >= 3


def test_fgrid_keys_match_config():
    """The scaled keys must exactly match the 8 documented in the spec."""
    _, keys = _load_helpers()
    assert set(keys) == {
        "n_phi_hv_high",
        "n_phi_hv_low",
        "n_phi_sys",
        "n_phi_hv_high_mode1",
        "n_phi_hv_low_mode1",
        "n_phi_sys_mode1",
        "n_r_local",
        "n_r_brute",
    }


def _load_apply_fgrid():
    with open(_RUNNER) as f:
        src = f.read()
    ns = {}
    start = src.index("# >>> f-grid helpers")
    end = src.index("# <<< f-grid helpers", start)
    exec(src[start:end], ns)
    return ns["_apply_fgrid"]


def _sample_cfg():
    return {
        "model": {
            "Om": 0.315,
            "n_phi_hv_high": 1001,
            "n_phi_hv_low": 301,
            "n_phi_sys": 3001,
            "n_phi_hv_high_mode1": 2001,
            "n_phi_hv_low_mode1": 501,
            "n_phi_sys_mode1": 3001,
            "n_r_local": 251,
            "n_r_brute": 501,
            "K_sigma": 5.0,
            "galaxies": {
                "NGC4258": {
                    "ra": 184.74,
                    "n_phi_hv_high": 20001,
                    "n_phi_hv_low": 4001,
                    "n_phi_sys": 20001,
                    "init": {"D_c": 7.593},
                },
                "NGC5765b": {
                    "v_sys_obs": 8465,
                },
            },
        }
    }


def test_apply_fgrid_identity():
    apply = _load_apply_fgrid()
    cfg = _sample_cfg()
    apply(cfg, 1.0)
    assert cfg["model"]["n_phi_hv_high"] == 1001
    assert cfg["model"]["n_r_local"] == 251
    assert cfg["model"]["galaxies"]["NGC4258"]["n_phi_hv_high"] == 20001


def test_apply_fgrid_half():
    apply = _load_apply_fgrid()
    cfg = _sample_cfg()
    apply(cfg, 0.5)
    m = cfg["model"]
    assert m["n_phi_hv_high"] == 501      # 500.5 -> 500 -> 501
    assert m["n_phi_hv_low"] == 151       # 150.5 -> 150 or 151 -> 151
    assert m["n_phi_sys"] == 1501         # 1500.5 -> 1500 -> 1501
    assert m["n_phi_hv_high_mode1"] == 1001
    assert m["n_phi_hv_low_mode1"] == 251 # 250.5 -> 250 -> 251
    assert m["n_phi_sys_mode1"] == 1501
    assert m["n_r_local"] == 127          # 125.5 -> 126 -> 127
    assert m["n_r_brute"] == 251          # 250.5 -> 250 -> 251
    g = m["galaxies"]["NGC4258"]
    assert g["n_phi_hv_high"] == 10001    # 10000.5 -> 10000 -> 10001
    assert g["n_phi_hv_low"] == 2001      # 2000.5 -> 2000 -> 2001
    assert g["n_phi_sys"] == 10001
    assert m["Om"] == 0.315
    assert m["K_sigma"] == 5.0
    assert g["ra"] == 184.74
    assert g["init"] == {"D_c": 7.593}
    assert m["galaxies"]["NGC5765b"] == {"v_sys_obs": 8465}


def test_apply_fgrid_tiny_factor_floors_at_three():
    apply = _load_apply_fgrid()
    cfg = _sample_cfg()
    apply(cfg, 0.0001)
    m = cfg["model"]
    for k in [
        "n_phi_hv_high", "n_phi_hv_low", "n_phi_sys",
        "n_phi_hv_high_mode1", "n_phi_hv_low_mode1", "n_phi_sys_mode1",
        "n_r_local", "n_r_brute",
    ]:
        assert m[k] == 3, (k, m[k])
    assert m["galaxies"]["NGC4258"]["n_phi_hv_high"] == 3
