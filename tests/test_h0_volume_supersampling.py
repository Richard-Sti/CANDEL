from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.special import ndtr

import candel.pvdata.volume_density as volume_density_mod
from candel.cosmo.cosmography import Distance2Distmod
from candel.pvdata.field_cache import (
    _VOLUME_FIELD_CACHE_PREFIX,
    _field_cache_path,
    _volume_field_cache_filename,
)
from candel.pvdata.volume_density import (
    _load_volume_data_for_H0,
    _h0_volume_cache_supersampling_payload,
    _h0_volume_supersampling_cache_arrays,
    _h0_volume_apply_quadrature,
    _h0_volume_quadrature_geometry,
    _h0_volume_resolved_supersample_factor,
    _expected_h0_volume_grid_from_loader,
    _supersample_offsets_3d,
    _volume_density_geometry,
)


def _toy_spherical_volume(radius, dx):
    extract_radius = radius + 0.5 * np.sqrt(3.0) * dx
    n = int(np.ceil(2.0 * extract_radius / dx))
    shape = (n, n, n)
    observer = np.array([0.5 * n * dx] * 3, dtype=np.float32)
    log_r, log_dv = _volume_density_geometry(shape, observer, dx)

    axes = [
        (np.arange(shape[i], dtype=np.float32) + 0.5) * dx
        for i in range(3)
    ]
    disp = [
        axes[0][:, None, None] - observer[0],
        axes[1][None, :, None] - observer[1],
        axes[2][None, None, :] - observer[2],
    ]
    r_grid = np.sqrt(disp[0]**2 + disp[1]**2 + disp[2]**2)
    r_grid = np.maximum(r_grid, 0.25 * dx)
    return log_r, log_dv, disp, r_grid


def _radial_mag_selection_integral(radius, mag_lim):
    distance2distmod = Distance2Distmod(Om0=0.3)
    r = np.linspace(0.01, radius, 20_000)
    mu = np.asarray(distance2distmod(r, h=1.0))
    sigma = np.sqrt(0.05**2 + 0.15**2)
    p_sel = ndtr((mag_lim - (mu - 4.05)) / sigma)
    return simpson(4.0 * np.pi * r**2 * p_sel, x=r)


def _volume_mag_selection_integral(
        radius, dx, mag_lim, supersample_factor, supersample_radius):
    distance2distmod = Distance2Distmod(Om0=0.3)
    log_r, log_dv, disp, r_grid = _toy_spherical_volume(radius, dx)
    quad = _h0_volume_quadrature_geometry(
        log_r, disp, r_grid, dx, "sphere", radius,
        supersample_factor=supersample_factor,
        supersample_radius=supersample_radius)

    r = np.exp(np.asarray(quad["log_r_3d"]))
    mu = np.asarray(distance2distmod(r, h=1.0))
    sigma = np.sqrt(0.05**2 + 0.15**2)
    p_sel = ndtr((mag_lim - (mu - 4.05)) / sigma)
    weights = np.exp(float(log_dv) + quad["log_volume_weight_3d"])
    return np.sum(weights * p_sel)


def test_h0_volume_supersampling_is_generic_cache_payload():
    payload = {
        "kind": "volume_field_data",
        "product": "h0_volume",
        "loader_name": "toy_reconstruction",
        "field_indices": [0],
        "geometry": "sphere",
        "subcube_radius": 50.0,
        "max_radius": 50.0,
        "downsample": 1,
        "load_velocity": False,
    }
    payload.update(_h0_volume_cache_supersampling_payload(8, 15.0))

    filename = _volume_field_cache_filename(payload)

    assert filename.startswith("cache_sphere__")
    assert "h0-volume" not in filename
    assert "v1" not in filename
    assert "toy_reconstruction" not in filename
    assert "rmax-50" in filename
    assert "ss-f8-r15-linear" in filename
    assert filename.endswith("__density.npz")


def test_h0_volume_target_dx_resolves_nearest_integer_factor():
    assert _h0_volume_resolved_supersample_factor(
        681.1 / 256.0, 1, 0.325) == 8
    assert _h0_volume_resolved_supersample_factor(
        4.0, 1, 0.325) == 12


def test_h0_volume_supersampling_trilinearly_interpolates_subcells():
    dx = 1.0
    shape = (9, 9, 9)
    observer = np.array([4.5, 4.5, 4.5], dtype=np.float32)
    log_r, _ = _volume_density_geometry(shape, observer, dx)
    axes = [
        (np.arange(shape[i], dtype=np.float32) + 0.5) * dx
        for i in range(3)
    ]
    disp = [
        axes[0][:, None, None] - observer[0],
        axes[1][None, :, None] - observer[1],
        axes[2][None, None, :] - observer[2],
    ]
    r_grid = np.maximum(
        np.sqrt(disp[0]**2 + disp[1]**2 + disp[2]**2), 0.25 * dx)
    i, j, k = np.indices(shape, dtype=np.float32)
    field = 2.0 * i - 3.0 * j + 0.5 * k + 7.0

    factor = 4
    quad = _h0_volume_quadrature_geometry(
        log_r, disp, r_grid, dx, "cube", None,
        supersample_factor=factor, supersample_radius=1.5)
    got = _h0_volume_apply_quadrature(field, quad)

    expected = [field.reshape(-1)[quad["unsup_flat"]]]
    parents = np.array(np.unravel_index(quad["sup_flat"], shape)).T
    offsets = _supersample_offsets_3d(factor, dx) / dx
    xyz = parents[:, None, :] + offsets[None, :, :]
    expected.append((
        2.0 * xyz[..., 0] - 3.0 * xyz[..., 1]
        + 0.5 * xyz[..., 2] + 7.0).reshape(-1))
    expected = np.concatenate(expected).astype(np.float32)

    assert np.max(np.abs(got - expected)) < 2e-6
    assert not np.allclose(
        got[-len(expected) + len(quad["unsup_flat"]):],
        np.repeat(
            field.reshape(-1)[quad["sup_flat"]], quad["n_subcells"]))


def test_pv_volume_cache_filename_keeps_requested_subsample_fraction():
    payload = {
        "kind": "volume_field_data",
        "product": "pv_volume_density",
        "loader_name": "ManticoreLocalCOLA",
        "field_indices": [0],
        "subcube_radius": 150.0,
        "max_radius": 150.0,
        "pad_subcube_boundary": True,
        "downsample": 1,
        "voxel_subsample_fraction": 0.5,
        "voxel_subsample_seed": 42,
        "store_rhat_3d": False,
    }

    filename = _volume_field_cache_filename(payload)

    assert "pv-volume-density" in filename
    assert "ManticoreLocalCOLA" not in filename
    assert "rmax-150" in filename
    assert "sub-0p5-seed-42" in filename
    assert "sub-0p1-seed-42" not in filename


def test_volume_cache_filenames_include_field_smoothing():
    h0_payload = {
        "kind": "volume_field_data",
        "product": "h0_volume",
        "loader_name": "toy_reconstruction",
        "field_indices": [0],
        "geometry": "sphere",
        "subcube_radius": 50.0,
        "downsample": 1,
        "field_smoothing_scale": 3.0,
        "load_velocity": True,
    }
    pv_payload = {
        "kind": "volume_field_data",
        "product": "pv_volume_density",
        "loader_name": "toy_reconstruction",
        "field_indices": [0],
        "subcube_radius": 150.0,
        "pad_subcube_boundary": True,
        "downsample": 1,
        "voxel_subsample_fraction": 0.5,
        "voxel_subsample_seed": 42,
        "store_rhat_3d": False,
        "field_smoothing_scale": 3.0,
    }

    assert "field-smooth-R3" in _volume_field_cache_filename(h0_payload)
    assert "field-smooth-R3" in _volume_field_cache_filename(pv_payload)


def test_h0_field_smoothing_applies_to_density_and_velocity(monkeypatch):
    class FakeLoader:
        def __init__(self, nsim, **kwargs):
            self.nsim = nsim
            self.boxsize = 12.0
            self.ngrid = 4
            self.coordinate_frame = "icrs"
            self.observer_pos = np.array([6.0, 6.0, 6.0],
                                         dtype=np.float32)

        def load_density(self):
            return np.ones((4, 4, 4), dtype=np.float32)

        def load_velocity_component(self, component):
            return np.full((4, 4, 4), component + 1, dtype=np.float32)

    calls = []

    def fake_smooth(field, smooth_scale, boxsize, make_copy=False):
        calls.append((field.shape, smooth_scale, boxsize))
        return np.array(field, copy=True)

    monkeypatch.setattr(
        volume_density_mod, "name2field_loader", lambda name: FakeLoader)
    monkeypatch.setattr(
        volume_density_mod, "apply_gaussian_smoothing", fake_smooth)

    _load_volume_data_for_H0(
        "fake", {"Om0": 0.3}, [0], "linear", 0.3,
        load_velocity=True, geometry="cube", cache_enabled=False,
        return_cache_fields=True, field_smoothing_scale=4.0)

    assert len(calls) == 4


def test_h0_volume_supersampling_matches_homogeneous_radial_integral():
    radius = 50.0
    dx = 5.0
    mag_lim = 24.0

    radial = _radial_mag_selection_integral(radius, mag_lim)
    coarse = _volume_mag_selection_integral(
        radius, dx, mag_lim,
        supersample_factor=1, supersample_radius=0.0)
    supersampled = _volume_mag_selection_integral(
        radius, dx, mag_lim,
        supersample_factor=8, supersample_radius=15.0)

    coarse_relerr = abs(coarse - radial) / radial
    supersampled_relerr = abs(supersampled - radial) / radial

    assert supersampled_relerr < 5e-4
    assert supersampled_relerr < coarse_relerr / 100.0


def test_h0_volume_supersampling_handles_bright_homogeneous_limit():
    radius = 50.0
    dx = 681.1 / 256.0
    mag_lim = 22.0

    radial = _radial_mag_selection_integral(radius, mag_lim)
    coarse = _volume_mag_selection_integral(
        radius, dx, mag_lim,
        supersample_factor=1, supersample_radius=0.0)
    supersampled = _volume_mag_selection_integral(
        radius, dx, mag_lim,
        supersample_factor=8, supersample_radius=15.0)

    coarse_relerr = abs(coarse - radial) / radial
    supersampled_relerr = abs(supersampled - radial) / radial

    assert coarse_relerr > 0.99
    assert supersampled_relerr < 2e-3


def test_h0_volume_component_loader_cache_is_cleared():
    class FakeLoader:
        clear_calls = 0

        def __init__(self, nsim, **kwargs):
            self.nsim = nsim
            self.boxsize = 3.0
            self.coordinate_frame = "icrs"
            self.observer_pos = np.array([1.5, 1.5, 1.5],
                                         dtype=np.float32)

        def load_density(self):
            return np.ones((3, 3, 3), dtype=np.float32)

        def load_velocity_component(self, component):
            return np.full((3, 3, 3), component + 1, dtype=np.float32)

        def clear_velocity_cache(self):
            type(self).clear_calls += 1

    original = volume_density_mod.name2field_loader
    try:
        volume_density_mod.name2field_loader = lambda name: FakeLoader
        _load_volume_data_for_H0(
            "fake", {"Om0": 0.3}, [0], "linear", 0.3,
            load_velocity=True, geometry="cube", cache_enabled=False,
            return_cache_fields=True)
    finally:
        volume_density_mod.name2field_loader = original

    assert FakeLoader.clear_calls == 1


def test_h0_volume_uses_per_field_warmed_cache(
        tmp_path, monkeypatch):
    class FakeLoader:
        def __init__(self, nsim, **kwargs):
            self.nsim = nsim
            self.boxsize = 16.0
            self.ngrid = 4
            self.coordinate_frame = "icrs"
            self.observer_pos = np.array([8.0, 8.0, 8.0],
                                         dtype=np.float32)

        def load_density(self):
            raise AssertionError("raw field should not be loaded")

    payload = {
        "kind": "volume_field_data",
        "product": "h0_volume",
        "loader_name": "fake_manticore",
        "field_indices": [1],
        "subcube_radius": 50.0,
        "geometry": "sphere",
        "sources": [],
        "downsample": 1,
        "supersample": {
            "factor": 4,
            "radius": 15.0,
            "method": "linear",
        },
        "load_velocity": False,
    }
    cache_path = Path(_field_cache_path(
        tmp_path, _VOLUME_FIELD_CACHE_PREFIX, payload))
    assert cache_path.parent.name == "fake_manticore"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    r_3d, _ = _expected_h0_volume_grid_from_loader(
        FakeLoader(1), "sphere", 50.0, 4, 15.0)
    np.savez(
        cache_path,
        rho_3d_fields=np.full((1, len(r_3d)), 20.0, dtype=np.float32),
        r_3d=r_3d,
        log_dV_3d=np.asarray(0.0, dtype=np.float32),
        log_volume_weight_3d=np.zeros(len(r_3d), dtype=np.float32),
        **_h0_volume_supersampling_cache_arrays(4, 15.0))

    monkeypatch.setattr(
        volume_density_mod, "name2field_loader", lambda name: FakeLoader)
    loaded = _load_volume_data_for_H0(
        "fake_manticore", {"Om0": 0.3}, [1], "linear", 0.3,
        subcube_radius=50.0, geometry="sphere", cache_dir=tmp_path,
        cache_enabled=True, supersample_radius=15.0,
        supersample_target_dx=1.0)

    np.testing.assert_allclose(
        np.asarray(loaded["density_3d_fields"]),
        np.full((1, len(r_3d)), 19.0, dtype=np.float32))
    assert cache_path.exists()


def test_h0_volume_target_dx_warmed_superset_matches_resolved_factor(
        tmp_path, monkeypatch):
    class FakeLoader:
        def __init__(self, nsim, **kwargs):
            self.nsim = nsim
            self.boxsize = 16.0
            self.ngrid = 4
            self.coordinate_frame = "icrs"
            self.observer_pos = np.array([8.0, 8.0, 8.0],
                                         dtype=np.float32)

        def load_density(self):
            raise AssertionError("raw field should not be loaded")

    payload = {
        "kind": "volume_field_data",
        "product": "h0_volume",
        "loader_name": "fake_manticore",
        "field_indices": [1],
        "subcube_radius": 50.0,
        "geometry": "sphere",
        "sources": [],
        "downsample": 1,
        "load_velocity": False,
    }
    r_3d, _ = _expected_h0_volume_grid_from_loader(
        FakeLoader(1), "sphere", 50.0, 4, 15.0)
    for factor, value in ((4, 40.0), (8, 80.0)):
        cache_payload = {
            **payload,
            "supersample": {
                "factor": factor,
                "radius": 15.0,
                "method": "linear",
            },
        }
        cache_path = Path(_field_cache_path(
            tmp_path, _VOLUME_FIELD_CACHE_PREFIX, cache_payload))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        factor_r_3d = (
            r_3d if factor == 4 else
            _expected_h0_volume_grid_from_loader(
                FakeLoader(1), "sphere", 50.0, factor, 15.0)[0])
        np.savez(
            cache_path,
            rho_3d_fields=np.full(
                (1, len(factor_r_3d)), value, dtype=np.float32),
            r_3d=factor_r_3d,
            log_dV_3d=np.asarray(0.0, dtype=np.float32),
            log_volume_weight_3d=np.zeros(
                len(factor_r_3d), dtype=np.float32),
            **_h0_volume_supersampling_cache_arrays(factor, 15.0))

    monkeypatch.setattr(
        volume_density_mod, "name2field_loader", lambda name: FakeLoader)
    loaded = _load_volume_data_for_H0(
        "fake_manticore", {"Om0": 0.3}, [1], "linear", 0.3,
        subcube_radius=50.0, geometry="sphere", cache_dir=tmp_path,
        cache_enabled=True, supersample_radius=15.0,
        supersample_target_dx=1.0)

    np.testing.assert_allclose(
        np.asarray(loaded["density_3d_fields"]),
        np.full((1, len(r_3d)), 39.0, dtype=np.float32))


def test_h0_volume_missing_warmed_cache_errors_before_raw_load(
        tmp_path, monkeypatch):
    class FakeLoader:
        def __init__(self, nsim, **kwargs):
            self.nsim = nsim
            self.boxsize = 16.0
            self.ngrid = 4
            self.coordinate_frame = "icrs"
            self.observer_pos = np.array([8.0, 8.0, 8.0],
                                         dtype=np.float32)

        def load_density(self):
            raise AssertionError("raw field should not be loaded")

    monkeypatch.setattr(
        volume_density_mod, "name2field_loader", lambda name: FakeLoader)
    monkeypatch.delenv("CANDEL_FIELD_CACHE_WARMUP", raising=False)

    try:
        _load_volume_data_for_H0(
            "fake_manticore", {"Om0": 0.3}, [1], "linear", 0.3,
            subcube_radius=50.0, geometry="sphere", cache_dir=tmp_path,
            cache_enabled=True, supersample_radius=15.0,
            supersample_target_dx=1.0)
    except RuntimeError as exc:
        msg = str(exc)
    else:
        raise AssertionError("expected missing warmed-cache error")

    assert "missing required warmed H0 3D volume data cache" in msg
    assert "cache_sphere__field-1" in msg


def test_h0_volume_raw_readable_field_builds_cache_on_miss(
        tmp_path, monkeypatch):
    class FakeLoader:
        def __init__(self, nsim, **kwargs):
            self.nsim = nsim
            self.boxsize = 16.0
            self.ngrid = 4
            self.coordinate_frame = "icrs"
            self.observer_pos = np.array([8.0, 8.0, 8.0],
                                         dtype=np.float32)

        def load_density(self):
            return np.full((4, 4, 4), 2.0 + self.nsim, dtype=np.float32)

    monkeypatch.setattr(
        volume_density_mod, "name2field_loader", lambda name: FakeLoader)

    loaded = _load_volume_data_for_H0(
        "ManticoreLocalCOLA", {"Om0": 0.306}, [0], "linear", 0.306,
        geometry="cube", cache_dir=tmp_path, cache_enabled=True)

    np.testing.assert_allclose(
        np.asarray(loaded["density_3d_fields"]), np.ones((1, 4, 4, 4)))
    assert list((tmp_path / "ManticoreLocalCOLA").glob("*.npz"))
