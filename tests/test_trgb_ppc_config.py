import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp
from numpyro import handlers

import candel.mock.ppc_trgb as ppc_trgb_module
from candel.inference import inference as inference_module
from candel.model.model_H0_TRGB import TRGBModel
from candel.mock.ppc_trgb import (_galactic_latitude_cut_from_config,
                                  _apply_selection_ppc,
                                  _bias_upper_bound,
                                  _field_name_config,
                                  generate_trgb_ppc,
                                  _merge_ppc_chunks,
                                  _runtime_cpu_count,
                                  fit_trgb_ppc_sky_exposure,
                                  sky_exposure_from_posterior_theta,
                                  _sky_exposure_settings_from_config)
from candel.mock._field_utils import (_density_max_within_radius,
                                      _sample_galactic_masked_xyz)


def _load_standalone_ppc_script():
    path = Path(__file__).resolve().parents[1] / "scripts/mocks/ppc_TRGB.py"
    spec = importlib.util.spec_from_file_location("ppc_TRGB_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trgb_ppc_sky_exposure_uses_model_config_by_default():
    config = {
        "model": {
            "TRGB_sky_exposure": {
                "enabled": True,
                "nside": 1,
                "kappa": 24.0,
            },
        },
    }

    assert _sky_exposure_settings_from_config(config) == (1, 24.0)


def test_trgb_ppc_sky_exposure_explicit_zero_disables_config():
    config = {
        "model": {
            "TRGB_sky_exposure": {
                "enabled": True,
                "nside": 1,
                "kappa": 24.0,
            },
        },
    }

    assert _sky_exposure_settings_from_config(config, nside=0) == (0, 48.0)


def test_trgb_ppc_sky_exposure_rejects_legacy_n_pix():
    config = {
        "model": {
            "TRGB_sky_exposure": {
                "enabled": True,
                "n_pix": 12,
                "kappa": 24.0,
            },
        },
    }

    with pytest.raises(ValueError, match="n_pix"):
        _sky_exposure_settings_from_config(config)


def test_trgb_ppc_b_min_uses_which_run_dataset_key():
    config = {
        "model": {"which_run": "EDD_TRGB_grouped"},
        "io": {
            "PV_main": {
                "EDD_TRGB": {"b_min": 5.0},
                "EDD_TRGB_grouped": {"b_min": 11.0},
            },
        },
    }

    assert _galactic_latitude_cut_from_config(config) == 11.0


def test_trgb_ppc_b_min_uses_model_fallback():
    config = {
        "model": {
            "which_run": "EDD_TRGB_grouped",
            "selection_integral_b_min": 7.5,
        },
        "io": {"PV_main": {}},
    }

    assert _galactic_latitude_cut_from_config(config) == 7.5


def test_trgb_ppc_field_config_uses_which_run_dataset_key():
    config = {
        "model": {"which_run": "EDD_TRGB_grouped"},
        "io": {
            "PV_main": {
                "EDD_TRGB": {"reconstruction": "WrongField"},
                "EDD_TRGB_grouped": {"reconstruction": "RightField"},
            },
            "reconstruction_main": {
                "WrongField": {"tag": "wrong"},
                "RightField": {"tag": "right"},
            },
        },
    }

    field_name, field_config = _field_name_config(config)

    assert field_name == "RightField"
    assert field_config == {"tag": "right"}


def test_trgb_ppc_b_min_without_which_run_uses_model_fallback():
    config = {
        "model": {"selection_integral_b_min": 7.5},
        "io": {
            "PV_main": {
                "EDD_TRGB": {"b_min": 5.0},
            },
        },
    }

    assert _galactic_latitude_cut_from_config(config) == 7.5


def test_field_pool_galactic_mask_sampler_obeys_b_min():
    gen = np.random.default_rng(123)

    xyz = _sample_galactic_masked_xyz(
        gen, r_sphere=50.0, pool_size=2000, b_min=10.0, rmin_h=0.1)

    r = np.linalg.norm(xyz, axis=1)
    b = np.rad2deg(np.arcsin(xyz[:, 2] / r))
    assert len(xyz) == 2000
    assert np.min(r) >= 0.1
    assert np.max(r) <= 50.0
    assert np.all(np.abs(b) >= 10.0 - 1.0e-6)


def test_ppc_density_envelope_uses_radius_limited_field_max():
    density = np.ones((4, 4, 4), dtype=float)
    density[0, 0, 0] = 100.0
    density[1, 1, 1] = 5.0

    rho_max = _density_max_within_radius(
        density, boxsize=4.0, observer_pos=np.array([2.0, 2.0, 2.0]),
        radius=1.0)

    assert rho_max == 5.0


def test_trgb_ppc_parallel_chunk_merge_preserves_diagnostics():
    chunks = [
        {
            "mag_sim": np.array([24.0]),
            "cz_sim": np.array([1000.0]),
            "r_sim": np.array([12.0]),
            "ra_obs": np.array([1.0, 2.0]),
            "sky_exposure": {"nside": 1, "theta": np.array([1.0])},
        },
        {
            "mag_sim": np.array([25.0]),
            "cz_sim": np.array([1100.0]),
            "r_sim": np.array([13.0]),
            "ra_obs": np.array([1.0, 2.0]),
            "sky_exposure": {"nside": 1, "theta": np.array([1.0])},
        },
    ]

    ppc = _merge_ppc_chunks(chunks)

    np.testing.assert_array_equal(ppc["mag_sim"], np.array([24.0, 25.0]))
    np.testing.assert_array_equal(ppc["cz_sim"], np.array([1000.0, 1100.0]))
    np.testing.assert_array_equal(ppc["r_sim"], np.array([12.0, 13.0]))
    np.testing.assert_array_equal(ppc["ra_obs"], np.array([1.0, 2.0]))
    assert "sky_exposure" in ppc


def test_trgb_ppc_worker_default_uses_runtime_cpu_env(monkeypatch):
    monkeypatch.setenv("CANDEL_PPC_N_WORKERS", "3")

    assert _runtime_cpu_count() == 3


def test_trgb_ppc_parallel_progress_reports_requested_galaxies(monkeypatch):
    class FakeTqdm:
        instances = []

        def __init__(self, total=None, **kwargs):
            self.total = total
            self.n = 0
            self.updates = []
            FakeTqdm.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n):
            self.updates.append(n)
            self.n += n

    monkeypatch.setattr(ppc_trgb_module, "tqdm", FakeTqdm)
    samples = {
        "H0": np.array([70.0, 71.0]),
        "M_TRGB": np.array([-4.0, -4.05]),
        "sigma_int": np.array([0.1, 0.1]),
        "sigma_v": np.array([100.0, 100.0]),
    }
    data = {
        "mag_obs": np.array([25.0, 25.1]),
        "czcmb": np.array([1000.0, 1100.0]),
        "e_mag_obs": np.array([0.05, 0.06]),
        "e_czcmb": np.array([30.0, 35.0]),
        "colour_dered": np.array([1.1, 1.2]),
    }
    config = {
        "model": {
            "use_reconstruction": False,
            "which_selection": None,
            "r_limits_malmquist": [0.1, 10.0],
            "Om": 0.3,
        },
    }

    ppc = generate_trgb_ppc(
        samples, data, config, n_ppc=8, n_workers=2, progress=True)

    assert len(ppc["mag_sim"]) == 8
    assert len(FakeTqdm.instances) == 1
    bar = FakeTqdm.instances[0]
    assert bar.total == 8
    assert bar.n == 8
    assert sum(bar.updates) == 8
    assert all(n > 0 for n in bar.updates)


def test_trgb_ppc_bias_upper_bound_includes_quadratic_peak():
    bias = {
        "which_bias": "quadratic",
        "b1": np.array([0.0]),
        "b2": np.array([-1.0]),
    }

    bound = _bias_upper_bound(
        np.array([0.0, 2.0]), bias, np.array([0]))

    assert bound[0] > 1.0


def test_trgb_ppc_bias_upper_bound_includes_degenerate_cubic_peak():
    bias = {
        "which_bias": "cubic",
        "b1": np.array([0.0]),
        "b2": np.array([-1.0]),
        "b3": np.array([0.0]),
    }

    bound = _bias_upper_bound(
        np.array([0.0, 2.0]), bias, np.array([0]))

    assert bound[0] > 1.0


def test_trgb_model_sky_exposure_applies_theta_to_hosts_and_volume():
    model = object.__new__(TRGBModel)
    model.TRGB_sky_exposure_n_pix = 4
    model.TRGB_sky_exposure_n_support = 3
    model.TRGB_sky_exposure_kappa = 9.0
    model._TRGB_sky_exposure_support_pix = jnp.array([0, 2, 3])
    model._TRGB_sky_exposure_host_pix = jnp.array([3, 0, 2])

    log_S_pix = jnp.array([[np.log(2.0), -jnp.inf,
                            np.log(5.0), np.log(7.0)]])
    theta_support = jnp.array([0.2, 0.3, 0.5])
    ratio_support = theta_support * 3
    theta_full = jnp.array([0.2, 0.0, 0.3, 0.5])
    fn = handlers.seed(
        handlers.substitute(
            model._sample_TRGB_sky_exposure,
            data={"TRGB_sky_exposure_ratio": ratio_support}),
        rng_seed=0)

    log_S, host_log_theta, log_theta = fn(log_S_pix)

    np.testing.assert_allclose(
        np.asarray(log_S),
        np.asarray(logsumexp(log_S_pix + jnp.log(theta_full)[None, :],
                             axis=-1)))
    np.testing.assert_allclose(
        np.asarray(host_log_theta),
        np.asarray(jnp.log(theta_full)[model._TRGB_sky_exposure_host_pix]))
    np.testing.assert_array_equal(np.isneginf(np.asarray(log_theta)),
                                  np.array([False, True, False, False]))


def test_trgb_model_sky_exposure_drops_zero_selection_pixels():
    model = object.__new__(TRGBModel)
    model.TRGB_sky_exposure_n_pix = 4
    model.TRGB_sky_exposure_n_support = 4
    model.TRGB_sky_exposure_kappa = 8.0
    model._TRGB_sky_exposure_support_pix = jnp.array([0, 1, 2, 3])
    model._TRGB_sky_exposure_host_pix = jnp.array([0, 2])

    log_S_pix = jnp.array([[np.log(2.0), -jnp.inf,
                            np.log(3.0), -jnp.inf]])
    theta_support = jnp.array([0.2, 0.3, 0.5, 0.1])
    ratio_support = theta_support * 2
    theta_full = jnp.array([0.2, 0.0, 0.5, 0.0])
    theta_full = theta_full / jnp.sum(theta_full)
    fn = handlers.seed(
        handlers.substitute(
            model._sample_TRGB_sky_exposure,
            data={"TRGB_sky_exposure_ratio": ratio_support}),
        rng_seed=0)

    log_S, host_log_theta, log_theta = fn(log_S_pix)

    np.testing.assert_allclose(
        np.asarray(log_S),
        np.asarray(logsumexp(log_S_pix + jnp.log(theta_full)[None, :],
                             axis=-1)))
    np.testing.assert_allclose(
        np.asarray(host_log_theta),
        np.asarray(jnp.log(theta_full)[model._TRGB_sky_exposure_host_pix]))
    np.testing.assert_array_equal(np.isneginf(np.asarray(log_theta)),
                                  np.array([False, True, False, True]))


def test_trgb_model_sky_exposure_uniform_theta_cancels_normalization():
    model = object.__new__(TRGBModel)
    model.TRGB_sky_exposure_n_pix = 4
    model.TRGB_sky_exposure_n_support = 2
    model.TRGB_sky_exposure_kappa = 8.0
    model._TRGB_sky_exposure_support_pix = jnp.array([1, 3])
    model._TRGB_sky_exposure_host_pix = jnp.array([1, 3, 3])

    log_S_pix = jnp.array([[-jnp.inf, np.log(4.0), -jnp.inf, np.log(6.0)]])
    theta_support = jnp.array([0.5, 0.5])
    ratio_support = theta_support * 2
    fn = handlers.seed(
        handlers.substitute(
            model._sample_TRGB_sky_exposure,
            data={"TRGB_sky_exposure_ratio": ratio_support}),
        rng_seed=0)

    log_S, host_log_theta, _ = fn(log_S_pix)

    log_S_no_sky = logsumexp(log_S_pix, axis=-1)
    np.testing.assert_allclose(
        np.asarray(log_S),
        np.asarray(log_S_no_sky - np.log(model.TRGB_sky_exposure_n_support)))
    np.testing.assert_allclose(
        np.asarray(host_log_theta),
        np.full(3, -np.log(model.TRGB_sky_exposure_n_support)))


def test_trgb_model_sky_exposure_marginalizes_over_field_realizations():
    """log_S_pix is (num_fields, n_pix): the sky-exposure baseline fraction
    must reflect all field realizations, not just the first, and the
    single-field case must reduce to the old field-0-only behaviour."""
    model = object.__new__(TRGBModel)
    model.TRGB_sky_exposure_n_pix = 4
    model.TRGB_sky_exposure_n_support = 3
    model.TRGB_sky_exposure_kappa = 9.0
    model._TRGB_sky_exposure_support_pix = jnp.array([0, 2, 3])
    model._TRGB_sky_exposure_host_pix = jnp.array([3, 0, 2])

    log_S_pix_multi = jnp.array([
        [np.log(2.0), -jnp.inf, np.log(5.0), np.log(7.0)],
        [np.log(100.0), -jnp.inf, np.log(1.0), np.log(1.0)],
    ])
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model._sample_TRGB_sky_exposure(log_S_pix_multi)
    q = np.asarray(tr["TRGB_sky_exposure_baseline_fraction"]["value"])

    log_S_marg = logsumexp(log_S_pix_multi, axis=0)
    q_expected = np.asarray(jnp.exp(log_S_marg - logsumexp(log_S_marg)))
    q_field0_only = np.asarray(
        jnp.exp(log_S_pix_multi[0] - logsumexp(log_S_pix_multi[0])))

    np.testing.assert_allclose(q, q_expected, atol=1e-6)
    assert not np.allclose(q, q_field0_only, atol=1e-3), (
        "baseline fraction must use both field realizations, not just "
        "field 0")

    # single-field case must be unchanged (regression safety)
    with handlers.trace() as tr_single, handlers.seed(rng_seed=0):
        model._sample_TRGB_sky_exposure(log_S_pix_multi[:1])
    q_single = np.asarray(
        tr_single["TRGB_sky_exposure_baseline_fraction"]["value"])
    np.testing.assert_allclose(q_single, q_field0_only, atol=1e-6)


def test_trgb_ppc_sky_exposure_uses_posterior_theta_samples():
    theta_samples = np.vstack([
        np.arange(1, 13, dtype=float),
        np.arange(12, 0, -1, dtype=float),
    ])
    theta_samples = theta_samples / np.sum(theta_samples, axis=1)[:, None]

    exposure = fit_trgb_ppc_sky_exposure(
        np.array([0.0, 180.0]), np.array([0.0, 0.0]),
        np.array([0.0, 180.0]), np.array([0.0, 0.0]),
        nside=1, kappa=12.0, theta_samples=theta_samples)

    support = exposure["baseline_fraction"] > 0.0
    assert exposure["n_support"] == int(np.sum(support))
    assert np.all(exposure["theta_samples"][:, ~support] == 0.0)
    np.testing.assert_allclose(
        np.sum(exposure["theta_samples"], axis=1),
        np.ones(theta_samples.shape[0]))
    np.testing.assert_allclose(
        exposure["theta"], np.mean(exposure["theta_samples"], axis=0))
    np.testing.assert_allclose(
        exposure["exposure_acceptance_norm_samples"],
        np.max(exposure["theta_samples"] * exposure["n_support"], axis=1))


def test_trgb_ppc_sky_exposure_from_posterior_theta_skips_pilot():
    theta_samples = np.zeros((2, 12), dtype=float)
    theta_samples[:, 0] = [0.25, 0.75]
    theta_samples[:, 3] = [0.75, 0.25]

    exposure = sky_exposure_from_posterior_theta(
        theta_samples, np.array([]), np.array([]), nside=1, kappa=12.0)

    assert exposure["n_support"] == 2
    assert np.sum(exposure["baseline_counts"]) == 0
    np.testing.assert_allclose(exposure["theta_samples"], theta_samples)
    np.testing.assert_allclose(
        exposure["exposure_acceptance_norm_samples"],
        np.max(theta_samples * exposure["n_support"], axis=1))


def test_trgb_ppc_sky_exposure_from_posterior_theta_uses_baseline_support():
    theta_samples = np.zeros((2, 12), dtype=float)
    theta_samples[:, 0] = [0.2, 0.7]
    theta_samples[:, 1] = [0.3, 0.2]
    theta_samples[:, 3] = [0.5, 0.1]
    baseline_fraction = np.zeros((2, 12), dtype=float)
    baseline_fraction[:, 0] = 0.25
    baseline_fraction[:, 3] = 0.75

    exposure = sky_exposure_from_posterior_theta(
        theta_samples, np.array([]), np.array([]), nside=1, kappa=12.0,
        baseline_fraction=baseline_fraction)

    assert exposure["n_support"] == 2
    assert exposure["baseline_fraction"][1] == 0.0
    assert np.all(exposure["theta_samples"][:, 1] == 0.0)
    np.testing.assert_allclose(
        np.sum(exposure["theta_samples"], axis=1),
        np.ones(theta_samples.shape[0]))


def test_trgb_ppc_selection_soft_lower_edge_is_not_hard_cut():
    class ZeroRandom:
        def random(self, n):
            return np.zeros(n)

    keep = _apply_selection_ppc(
        np.array([22.0]), "TRGB_magnitude", np.array([0]),
        22.1, None, 26.0, None, 0.5, ZeroRandom())

    assert keep[0]


def test_trgb_sky_exposure_corner_uses_variable_full_pixels(
        tmp_path, monkeypatch):
    calls = {}

    def fake_plot_corner(samples, show_fig, filename):
        calls["samples"] = samples
        calls["show_fig"] = show_fig
        calls["filename"] = filename

    monkeypatch.setattr(inference_module, "plot_corner", fake_plot_corner)
    model = object.__new__(TRGBModel)
    model.use_TRGB_sky_exposure = True
    samples = {
        "TRGB_sky_exposure_theta_full": np.array([
            [0.2, 0.0, 0.8],
            [0.3, 0.0, 0.7],
        ]),
    }

    paths = inference_module._plot_trgb_sky_exposure_outputs(
        model, samples, str(tmp_path / "run.hdf5"))

    expected = str(tmp_path / "run_corner_TRGB_sky_exposure.png")
    assert paths == [
        ("TRGB sky-exposure pixel-fraction corner plot", expected)]
    assert list(calls["samples"]) == ["sky_frac_pix_0", "sky_frac_pix_2"]
    assert calls["show_fig"] is False
    assert calls["filename"] == expected


def test_standalone_ppc_all_fields_keeps_diagnostic_arrays(monkeypatch):
    script = _load_standalone_ppc_script()

    def fake_generate_trgb_ppc(samples, data, config_path, n_ppc=None,
                               seed=42, field_index=None, **kwargs):
        field = int(field_index)
        exposure = {
            "nside": 1,
            "baseline_fraction": np.array([0.25, 0.75]),
            "exposure_ratio": np.array([1.0, 1.0]),
        }
        return {
            "mag_sim": np.full(n_ppc, field, dtype=float),
            "cz_sim": np.full(n_ppc, field + 100.0),
            "r_sim": np.full(n_ppc, field + 10.0),
            "ra_sim": np.full(n_ppc, field + 20.0),
            "dec_sim": np.full(n_ppc, field + 30.0),
            "colour_sim": np.full(n_ppc, field + 40.0),
            "mag_obs": np.array([24.0]),
            "cz_obs": np.array([1000.0]),
            "ra_obs": np.array([1.0]),
            "dec_obs": np.array([2.0]),
            "colour_obs": np.array([1.2]),
            "sky_exposure": exposure,
        }

    monkeypatch.setattr(
        script, "generate_trgb_ppc", fake_generate_trgb_ppc)

    ppc = script._generate_ppc(
        {}, {"mag_obs": np.array([24.0])}, "config.toml",
        n_ppc=4, seed=10, field_indices=[3, 7], n_workers=1)

    np.testing.assert_array_equal(ppc["field_indices"], np.array([3, 7]))
    for key in ("mag_sim", "cz_sim", "r_sim", "ra_sim", "dec_sim",
                "colour_sim"):
        assert len(ppc[key]) == 4
    for key in ("mag_obs", "cz_obs", "ra_obs", "dec_obs", "colour_obs"):
        assert key in ppc
    assert "sky_exposure" in ppc

    payload = script._npz_payload(ppc)
    assert "ra_sim" in payload
    assert "sky_exposure_exposure_ratio" in payload
