#!/usr/bin/env python
"""Measure peak memory for each task config (runs 1 sample only)."""
import resource
import sys
from pathlib import Path

# Must set before importing JAX/NumPyro
import numpyro
numpyro.set_host_device_count(1)

import candel
from candel import fprint, get_nested


def measure_config(config_path):
    """Run minimal inference and return peak memory in MB."""
    config = candel.load_config(config_path)

    # Override to minimal run
    config["inference"]["num_warmup"] = 1
    config["inference"]["num_samples"] = 1
    config["inference"]["num_chains"] = 1
    config["inference"]["compute_evidence"] = False
    config["inference"]["compute_log_density"] = False

    data = candel.pvdata.load_PV_dataframes(config_path)

    model_name = config["inference"]["model"]
    shared_param = get_nested(config, "inference/shared_params", None)

    # Temporarily override config in model
    model = candel.model.name2model(model_name, shared_param, config_path)
    model.config = config

    model_kwargs = {"data": data}
    candel.run_pv_inference(model, model_kwargs, print_summary=False, save_samples=False)

    # Peak memory in MB (Linux)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak_mb = peak_kb / 1024 / 1024  # macOS returns bytes
    else:
        peak_mb = peak_kb / 1024  # Linux returns KB

    return peak_mb


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure_memory.py <config.toml>")
        sys.exit(1)

    config_path = sys.argv[1]
    peak = measure_config(config_path)
    print(f"PEAK_MEMORY_MB: {peak:.0f}")
