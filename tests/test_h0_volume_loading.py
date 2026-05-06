import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from candel.pvdata.data import _load_h0_volume_data_from_config


class H0VolumeDataLoadingTest(unittest.TestCase):
    def test_no_selection_reconstruction_loads_finite_volume_normalizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            los_path = Path(tmpdir) / "los.hdf5"
            with h5py.File(los_path, "w") as f:
                f.create_dataset("field_indices", data=np.array([0, 2]))

            config = {
                "model": {
                    "which_run": "CH0",
                    "which_selection": None,
                    "use_reconstruction": True,
                    "selection_integral_grid_radius": 100.0,
                    "density_3d_subsample_fraction": 1.0,
                    "which_bias": "linear",
                    "Om": 0.3,
                },
                "io": {
                    "field_cache_enabled": False,
                    "reconstruction_main": {
                        "Carrick2015": {"path_density": "density.npy"},
                    },
                },
            }

            with patch(
                    "candel.pvdata.data._load_volume_data_for_H0",
                    return_value={"density_3d_fields": object()}) as loader:
                out = _load_h0_volume_data_from_config(
                    config, str(los_path), "Carrick2015", "SH0ES",
                    velocity_selections=("redshift",))

        self.assertIn("density_3d_fields", out)
        loader.assert_called_once()
        args, kwargs = loader.call_args
        self.assertEqual(args[0], "Carrick2015")
        np.testing.assert_array_equal(args[2], np.array([0, 2]))
        self.assertEqual(kwargs["subcube_radius"], 100.0)
        self.assertFalse(kwargs["load_velocity"])

    def test_no_selection_cchp_does_not_load_unused_volume_normalizer(self):
        config = {
            "model": {
                "which_run": "CCHP",
                "which_selection": None,
                "use_reconstruction": True,
            },
        }

        with patch("candel.pvdata.data._load_volume_data_for_H0") as loader:
            out = _load_h0_volume_data_from_config(
                config, "unused.hdf5", "Carrick2015", "CCHP",
                velocity_selections=("redshift",))

        self.assertIsNone(out)
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
