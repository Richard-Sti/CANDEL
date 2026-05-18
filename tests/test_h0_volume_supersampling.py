import numpy as np
from scipy.integrate import simpson
from scipy.special import ndtr

from candel.cosmo.cosmography import Distance2Distmod
from candel.pvdata.field_cache import (
    _h0_volume_cache_filename,
    _pv_volume_density_cache_filename,
)
from candel.pvdata.volume_density import (
    _h0_volume_cache_supersampling_payload,
    _h0_volume_apply_quadrature,
    _h0_volume_quadrature_geometry,
    _h0_volume_resolved_supersample_factor,
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
        "kind": "h0_volume_data",
        "field_name": "toy_reconstruction",
        "field_indices": [0, 1],
        "geometry": "sphere",
        "subcube_radius": 50.0,
        "downsample": 1,
        "load_velocity": False,
    }
    payload.update(_h0_volume_cache_supersampling_payload(8, 15.0))

    filename = _h0_volume_cache_filename(payload)

    assert "toy_reconstruction" in filename
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
        "kind": "pv_volume_density_3d",
        "loader_name": "COLA_manticore_2MPP_MULTIBIN_N256_DES_V2",
        "field_indices": [0, 1],
        "subcube_radius": 150.0,
        "pad_subcube_boundary": True,
        "downsample": 1,
        "voxel_subsample_fraction": 0.5,
        "voxel_subsample_seed": 42,
        "store_rhat_3d": False,
    }

    filename = _pv_volume_density_cache_filename(payload)

    assert "sub-0p5-seed-42" in filename
    assert "sub-0p1-seed-42" not in filename


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
