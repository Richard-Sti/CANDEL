#!/usr/bin/env python
"""Test maser peak refactor against saved reference values."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np

from candel.mock.maser_disk_mock import gen_maser_mock_like_cgcg074
from candel.model.model_H0_maser import (
    MaserDiskModel, build_grid_config, marginalise_spots,
    _phi_bounds,
)

data, tp = gen_maser_mock_like_cgcg074(seed=42, verbose=False)

gc = build_grid_config()

i0 = jnp.deg2rad(tp["i0"])
Omega0 = jnp.deg2rad(tp["Omega0"])
dOmega_dr = jnp.deg2rad(tp["dOmega_dr"])
di_dr = jnp.array(0.0)
phi_lo, phi_hi = _phi_bounds(data["spot_type"], data["n_spots"])

ll, r_star, phi_star = marginalise_spots(
    jnp.asarray(data["x"]), jnp.asarray(data["sigma_x"]),
    jnp.asarray(data["y"]), jnp.asarray(data["sigma_y"]),
    jnp.asarray(data["velocity"]),
    jnp.asarray(data["a"]), jnp.asarray(data["sigma_a"]),
    jnp.asarray(data["accel_measured"]),
    phi_lo, phi_hi,
    tp["x0"], tp["y0"], tp["D"], tp["M_BH"], tp["v_sys"],
    i0, di_dr, Omega0, dOmega_dr,
    tp["sigma_x_floor"], tp["sigma_y_floor"],
    tp["sigma_v_sys"], tp["sigma_v_hv"],
    tp["sigma_a_floor"], tp["A_thr"], tp["sigma_det"],
    gc["dr_offsets"], gc["dphi_offsets"], gc["log_wr"], gc["log_wphi"],
)

print(f"ll_total = {float(ll):.4f}")

ref = np.load(os.path.join(os.path.dirname(__file__),
                           "maser_peak_reference.npz"))
print(f"Reference ll = {ref['ll']:.4f}")
print(f"Delta ll = {float(ll) - ref['ll']:.4f}")
