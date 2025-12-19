"""
Generate two cluster mocks with no zeropoint dipole injected and a linear bulk flow:
1) One using Carrick2015 reconstruction
2) One with no reconstruction

Then run four ready-to-go inference configs (dipA and radmag for each mock).
Configs are expected to exist in results/dipole_mock:
    - mock_0_carrick_dipA.toml
    - mock_0_carrick_radmag.toml
    - mock_1_norecon_dipA.toml
    - mock_1_norecon_radmag.toml

Usage (from repo root):
    python scripts/cluster_runs/run_single_dipole_mock.py
"""
from os import makedirs
from os.path import join, exists

import numpy as np
from h5py import File

import candel


def generate_mock(seed=1234, nsamples=276, field_loader=None):
    # Use fixed, reasonable mock truths (inside priors) to avoid sampling
    # values that fall outside prior support when initializing inference.
    b1 = 0.0
    beta = 0.44
    sigma_int = 0.14          # sigma_YT in the summary
    sigma_int_LT = 0.15       # keep a simple LT scatter
    sigma_v = 405.0

    # Y-T relation
    A_YT = 2.10
    B_YT = 2.20

    # L-T relation (kept simple)
    A_LT = 0.0
    B_LT = 2.5

    # Zeropoint dipole: 15% injected dipole
    zeropoint_dipole_mag = 0.15
    zeropoint_dipole_ell = 72.0
    zeropoint_dipole_b = 48.0

    # Bulk flow (set to zero here)
    Vext_mag = 0.0
    Vext_ell = 0.0
    Vext_b = 0.0

    # Distance prior (empirical)
    R = 265.0
    p = 1.96
    n = 2.84

    kwargs = {
        "r_grid": np.linspace(0.1, 2001, 2001),
        "Vext_mag": Vext_mag,
        "Vext_ell": Vext_ell,
        "Vext_b": Vext_b,
        "sigma_v": sigma_v,
        "beta": beta,
        "b1": b1,
        "A_YT": A_YT,
        "B_YT": B_YT,
        "sigma_int": sigma_int,
        "A_LT": A_LT,
        "B_LT": B_LT,
        "sigma_int_LT": sigma_int_LT,
        "zeropoint_dipole_mag": zeropoint_dipole_mag,
        "zeropoint_dipole_ell": zeropoint_dipole_ell,
        "zeropoint_dipole_b": zeropoint_dipole_b,
        "h": 1.0,
        "logT_prior_mean": 0.0,
        "logT_prior_std": 0.2,
        "e_logT": 0.03,
        "e_logY": 0.09,
        "e_logF": 0.05,
        "b_min": 20.0,
        "zcmb_max": 0.45,
        "R_dist_emp": R,
        "p_dist_emp": p,
        "n_dist_emp": n,
        "field_loader": field_loader,
        "r2distmod": candel.Distance2Distmod(),
        "r2z": candel.Distance2Redshift(),
        "Om": 0.3,
        # No linear bulk flow injected
        "linear_Vext_slope": 0.0,
        "linear_Vext_ell": None,
        "linear_Vext_b": None,
        # Stretch Carrick LOS fields for zeropoint dipole mocks
        "rescale_carrick_fields": zeropoint_dipole_mag is not None,
    }

    mock = candel.mock.gen_Clusters_mock(nsamples, seed=seed, **kwargs)
    kwargs["seed"] = seed
    return mock, kwargs


def save_mock(mock, kwargs, out_dir, mock_name="mock_0"):
    makedirs(out_dir, exist_ok=True)
    mock_path = join(out_dir, f"{mock_name}.hdf5")
    with File(mock_path, "w") as f:
        grp = f.create_group("mock")
        for key, value in mock.items():
            grp.create_dataset(key, data=value, dtype=np.float32)

        # Save scalar params as attributes
        for key, value in kwargs.items():
            if isinstance(value, (float, int, bool)):
                grp.attrs[key] = value
        grp.attrs["seed"] = kwargs.get("seed", 1234)
        grp.attrs["nsamples"] = len(mock["zcmb"])
        grp.attrs["mock_name"] = mock_name
    return mock_path


def run_inference_config(config_path: str):
    """Run a single inference config (serial, no MPI).

    If this is a zeropoint-dipole run (filename contains 'dipA'), we
    explicitly enable LOS stretching on the model to match the mock
    generation when a zeropoint dipole is injected.
    """
    cfg = candel.load_config(config_path)
    data = candel.pvdata.load_PV_dataframes(config_path)
    # Drop mock truth hints to avoid invalid init if outside prior support.
    def _strip_truth(d):
        if hasattr(d, "data"):
            d.data.pop("_mock_truths", None)
        elif isinstance(d, dict):
            d.pop("_mock_truths", None)
    if isinstance(data, list):
        for d in data:
            _strip_truth(d)
    else:
        _strip_truth(data)

    model_name = cfg["inference"]["model"]
    shared_param = cfg["inference"].get("shared_params", None)
    model = candel.model.name2model(model_name, shared_param, config_path)

    # Enable LOS stretching for zeropoint-dipole runs
    if "dipA" in config_path and hasattr(model, "stretch_los_with_zeropoint"):
        model.stretch_los_with_zeropoint = True

    if isinstance(data, list):
        if not isinstance(model, candel.model.JointPVModel):
            raise TypeError("Multiple datasets provided but model is not JointPVModel.")
        if len(data) != len(model.submodels):
            raise ValueError(f"Datasets ({len(data)}) != submodels ({len(model.submodels)})")

    candel.run_pv_inference(model, {"data": data})


def main():
    out_dir = "results/dipole_mock"
    density_path = join("data", "fields", "carrick2015_twompp_density.npy")
    velocity_path = join("data", "fields", "carrick2015_twompp_velocity.npy")
    field_loader = candel.field.name2field_loader("Carrick2015")(
        path_density=density_path,
        path_velocity=velocity_path,
    )

    rng = np.random.default_rng()
    seeds = rng.integers(0, 2**32 - 1, size=2, dtype=np.uint32)

    # Generate Carrick2015 mock (index 0)
    mock0, kwargs0 = generate_mock(seed=int(seeds[0]), field_loader=field_loader)
    mock0_path = save_mock(mock0, kwargs0, out_dir, mock_name="mock_0")
    print(f"Carrick2015 mock saved to: {mock0_path} (seed={kwargs0['seed']})")

    # Generate no-reconstruction mock (index 1)
    mock1, kwargs1 = generate_mock(seed=int(seeds[1]), field_loader=None)
    mock1_path = save_mock(mock1, kwargs1, out_dir, mock_name="mock_1")
    print(f"No-recon mock saved to: {mock1_path} (seed={kwargs1['seed']})")

    # Static configs expected to exist
    carrick_dipA_cfg = join(out_dir, "mock_0_carrick_dipA.toml")
    carrick_radmag_cfg = join(out_dir, "mock_0_carrick_radmag.toml")
    norecon_dipA_cfg = join(out_dir, "mock_1_norecon_dipA.toml")
    norecon_radmag_cfg = join(out_dir, "mock_1_norecon_radmag.toml")

    for cfg in [carrick_dipA_cfg, carrick_radmag_cfg, norecon_dipA_cfg, norecon_radmag_cfg]:
        if not exists(cfg):
            raise FileNotFoundError(f"Config not found: {cfg}")

    print("Running Carrick2015 dipA inference...")
    run_inference_config(carrick_dipA_cfg)
    print("Running Carrick2015 radmag inference...")
    run_inference_config(carrick_radmag_cfg)
    print("Running no-recon dipA inference...")
    run_inference_config(norecon_dipA_cfg)
    print("Running no-recon radmag inference...")
    run_inference_config(norecon_radmag_cfg)
    print("All inferences completed.")


if __name__ == "__main__":
    main()
