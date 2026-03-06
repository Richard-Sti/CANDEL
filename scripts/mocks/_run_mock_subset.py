#!/usr/bin/env python
"""Run a subset of TRGB mocks (for parallel execution)."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from run_mock_TRGB import run_one_mock, TRACKED_PARAMS
from candel.mock.TRGB_mock import DEFAULT_ANCHORS, DEFAULT_TRUE_PARAMS
from candel.field import name2field_loader
import candel

seeds = json.loads(sys.argv[1])
outfile = sys.argv[2]

base_config = os.path.join(
    os.path.dirname(__file__), "..", "runs", "config_EDD_TRGB.toml")
base_config = os.path.abspath(base_config)

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

config = candel.load_config(base_config, replace_los_prior=False)
field_config = config["io"]["reconstruction_main"]["Carrick2015"]
loader_cls = name2field_loader("Carrick2015")
field_loader = loader_cls(**field_config)

true_params = {
    "H0": DEFAULT_TRUE_PARAMS["H0"],
    "M_TRGB": DEFAULT_TRUE_PARAMS["M_TRGB"],
    "sigma_int": DEFAULT_TRUE_PARAMS["sigma_int"],
    "sigma_v": DEFAULT_TRUE_PARAMS["sigma_v"],
    "beta": DEFAULT_TRUE_PARAMS["beta"],
    "b1": DEFAULT_TRUE_PARAMS["b1"],
}

mock_kwargs = {
    "nsamples": 480,
    "rmax": 40.0,
    "mag_lim": 25.0,
    "mag_lim_width": 0.75,
    "cz_lim": None,
    "cz_lim_width": None,
    "field_loader": field_loader,
}

results = []
for seed in seeds:
    print(f"[Mock seed={seed}] Starting...", flush=True)
    try:
        biases, n_hosts, n_parent = run_one_mock(
            seed, base_config, true_params, mock_kwargs,
            num_warmup=500, num_samples=500,
            which_selection="TRGB_magnitude",
            infer_selection=True, use_field=True, quiet=True)
        results.append({"seed": seed, "biases": biases,
                         "n_hosts": n_hosts, "n_parent": n_parent})
        print(f"[Mock seed={seed}] Done. Biases:", flush=True)
        for p in TRACKED_PARAMS:
            if p in biases:
                print(f"  {p}: {biases[p]:+.2f}σ", flush=True)
    except Exception as e:
        print(f"[Mock seed={seed}] FAILED: {e}", flush=True)
        results.append({"seed": seed, "biases": None})

np.save(outfile, results)
print(f"Saved to {outfile}")
