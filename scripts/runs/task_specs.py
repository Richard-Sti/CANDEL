"""Named task-list specifications for generate_tasks.py.

Keep this file focused on sweep definitions. Generator behavior belongs in
generate_tasks.py.
"""


FOUNDATION_SMOKE_COMMON = {
    "inference/model": "SNModel",
    "inference/num_chains": 1,
    "inference/num_warmup": 500,
    "inference/num_samples": 1000,
    "inference/chain_method": "sequential",
    "inference/compute_evidence": False,
    "inference/skip_if_exists": True,
    "pv_model/kind": "precomputed_los_Carrick2015",
    "pv_model/galaxy_bias": "linear_from_beta",
    "pv_model/which_distance_prior": "empirical",
    "model/cz_likelihood": "student_t",
    "io/catalogue_name": "Foundation",
    "io/root_output": "results/test_foundation_carrick2015_student_t",
}


TASK_SPECS = {
    "0": {
        "description": "Smoke/test Foundation SN run with Carrick2015 PVs.",
        "config_path": "configs/config.toml",
        "tag": "student_t",
        "common": FOUNDATION_SMOKE_COMMON,
        "datasets": [{}],
        "expected_tasks": 1,
    },
    "test_foundation_carrick2015_student_t": {
        "description": "Named alias of task 0 for the Foundation smoke run.",
        "config_path": "configs/config.toml",
        "tag": "student_t",
        "common": FOUNDATION_SMOKE_COMMON,
        "datasets": [{}],
        "expected_tasks": 1,
    },
    "ch0": {
        "description": "SH0ES Cepheid H0 run using config_shoes defaults.",
        "config_path": "configs/config_shoes.toml",
        "tag": "default",
        "common": {
            "io/root_output": "results/CH0",
        },
        "datasets": [{}],
        "expected_tasks": 1,
    },
    "ch0_mag": {
        "description": "SH0ES Cepheid H0 run with SN-magnitude selection.",
        "config_path": "configs/config_shoes.toml",
        "tag": "default",
        "common": {
            "io/root_output": "results/CH0",
            "model/which_selection": "SN_magnitude",
        },
        "datasets": [{}],
        "expected_tasks": 1,
    },
    "S8_FP_student_t": {
        "description": "S8 from FP PVs: 2 catalogues x 2 galaxy biases.",
        "config_path": "configs/config.toml",
        "tag": "student_t",
        "common": {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": ["linear", "quadratic"],
            "pv_model/density_3d_downsample": 1,
            "model/priors/beta": {"dist": "uniform", "low": 0.0, "high": 2.0},
            "model/cz_likelihood": "student_t",
            "inference/num_chains": 1,
            "inference/num_warmup": 2000,
            "inference/num_samples": 10000,
            "io/root_output": "results/S8",
        },
        "datasets": [
            {"inference/model": "FPModel", "io/catalogue_name": "6dF_FP"},
            {"inference/model": "FPModel", "io/catalogue_name": "SDSS_FP"},
        ],
        "expected_tasks": 4,
    },
}
