import unittest

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro import handlers

from candel.model.base_pv import BasePVModel
from candel.model.pv_utils import (
    RHO_CRIT_H2,
    convert_cartesian_frame,
    gaussian_missing_mass_delta,
    gaussian_missing_mass_velocity,
    missing_mass_los_delta_velocity,
    spherical_rhat,
)
from candel.pvdata.data import PVDataFrame
from candel.util import galactic_to_radec_cartesian


class DummyPVModel(BasePVModel):
    def __call__(self):
        raise NotImplementedError


class MissingMassPhysicsTest(unittest.TestCase):
    def test_model_samples_Mmiss_in_log_mass(self):
        model = object.__new__(DummyPVModel)
        model.use_Mmiss = True
        model.Mmiss_coordinate_frame = "galactic"
        model.priors = {
            "logM_miss": dist.Uniform(14.0, 16.5),
            "Mmiss_distance": dist.Uniform(20.0, 150.0),
            "Mmiss_ell": dist.Uniform(0.0, 360.0),
            "Mmiss_b": dist.Uniform(-10.0, 10.0),
        }

        trace = handlers.trace(
            handlers.seed(model._sample_Mmiss, rng_seed=0)
        ).get_trace()

        sample_sites = {
            name for name, site in trace.items() if site["type"] == "sample"
        }
        self.assertIn("logM_miss", sample_sites)
        self.assertIn("Mmiss_distance", sample_sites)
        self.assertNotIn("Mmiss", trace)
        self.assertNotIn("Mmiss_rhat_icrs", trace)
        self.assertNotIn("missing_mass_rhat_icrs", trace)

    def test_galactic_frame_conversion_matches_astropy_helper(self):
        got = np.asarray(convert_cartesian_frame(
            spherical_rhat(0.0, 0.0), "galactic", "icrs"))
        expected = galactic_to_radec_cartesian(0.0, 0.0)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_gaussian_density_central_amplitude(self):
        mass = 1.0e15
        sigma = 4.0
        Om = 0.3
        pos = jnp.array([[0.0, 0.0, 0.0]])
        delta = gaussian_missing_mass_delta(
            pos, jnp.zeros(3), mass, sigma, Om)

        expected = mass / (
            RHO_CRIT_H2 * Om * (2.0 * np.pi)**1.5 * sigma**3)
        self.assertAlmostEqual(float(delta[0]), expected, places=6)

    def test_far_field_velocity_matches_point_mass_limit(self):
        mass = 1.0e15
        sigma = 2.0
        Om = 0.3
        s = 60.0
        pos = jnp.array([[s, 0.0, 0.0]])
        rhat = jnp.array([[1.0, 0.0, 0.0]])
        vlos = gaussian_missing_mass_velocity(
            pos, rhat, jnp.zeros(3), mass, sigma, Om)

        expected = (
            -100.0 * Om**0.55 * mass
            / (4.0 * np.pi * RHO_CRIT_H2 * Om * s**2)
        )
        self.assertLess(float(vlos[0]), 0.0)
        self.assertAlmostEqual(float(vlos[0]), expected, delta=abs(expected) * 1e-3)

    def test_los_velocity_changes_sign_across_cluster(self):
        r_grid = jnp.array([10.0, 30.0])
        rhat = jnp.array([[1.0, 0.0, 0.0]])
        cluster_r = 20.0
        cluster_rhat = jnp.array([1.0, 0.0, 0.0])

        delta, vlos = missing_mass_los_delta_velocity(
            r_grid, rhat, cluster_r, cluster_rhat,
            mass=5.0e15, sigma=3.0, Om=0.3)

        self.assertEqual(delta.shape, (1, 2))
        self.assertEqual(vlos.shape, (1, 2))
        self.assertGreater(float(vlos[0, 0]), 0.0)
        self.assertLess(float(vlos[0, 1]), 0.0)


class MissingMassDataGeometryTest(unittest.TestCase):
    def test_volume_density_can_store_voxel_directions(self):
        frame = PVDataFrame({
            "RA": np.array([0.0]),
            "dec": np.array([0.0]),
            "zcmb": np.array([0.01]),
        })
        rho = np.ones((4, 4, 4), dtype=np.float32)
        frame.attach_volume_density_3d_fields(
            [(rho, np.array([2.0, 2.0, 2.0], dtype=np.float32), 1.0,
              "galactic")],
            galaxy_bias="linear",
            geometry="cube",
            store_rhat_3d=True,
        )

        self.assertEqual(frame.coordinate_frame_3d, "galactic")
        for key in ("rhat_x_3d", "rhat_y_3d", "rhat_z_3d"):
            self.assertIn(key, frame.keys())

        rhat = np.stack(
            [np.asarray(frame[key]) for key in (
                "rhat_x_3d", "rhat_y_3d", "rhat_z_3d")],
            axis=-1)
        norms = np.linalg.norm(rhat, axis=-1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
