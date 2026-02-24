"""Quick MCMC test for the FP model with analytical MNR marginalisation.

Runs a short NUTS chain on 500 subsampled SDSS_FP galaxies with Carrick.
"""
import sys
sys.path.insert(0, "/Users/rstiskalek/Projects/CANDEL")

import candel
from candel.util import plot_corner

CONFIG_PATH = "results_test/precomputed_los_Carrick2015_SDSS_FP_noMNR_.toml"
CORNER_PATH = "/Users/rstiskalek/Downloads/FP_MNR_corner.png"

data = candel.pvdata.load_PV_dataframes(CONFIG_PATH)

model = candel.model.FPModel(CONFIG_PATH)

print(f"n_gal={len(data)}")

samples, _ = candel.run_pv_inference(
    model, {"data": data}, save_samples=False)

plot_corner(samples, show_fig=False, filename=CORNER_PATH)
print(f"\nCorner plot saved to {CORNER_PATH}")
