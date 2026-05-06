"""Named task-list specifications for generate_tasks.py.

Keep this file focused on sweep definitions. Generator behavior belongs in
generate_tasks.py.
"""


CH0_PAPER_ROOT = "results/CH0_paper"
CH0_MANTICORE_LOS = "manticore_2MPP_MULTIBIN_N256_DES_V2"
CH0_MANTICORE_BIAS = "double_powerlaw"
TRGBH0_ROOT = "results/TRGBH0"
TRGBH0_MANTICORE_LOS = "manticore_2MPP_MULTIBIN_N256_DES_V2"
TRGBH0_MANTICORE_BIAS = "double_powerlaw"

CH0_PAPER_COMMON = {
    "inference/num_chains": 12,
    "inference/chain_method": "sequential",
    "inference/num_warmup": 1000,
    "inference/num_samples": 6000,
    "model/use_uniform_mu_host_priors": False,
    "model/selection_integral_geometry": "sphere",
    "model/selection_integral_grid_radius": 100.0,
    "model/density_3d_subsample_fraction": 1.0,
    "model/priors/M_B": {"dist": "uniform", "low": -22.0, "high": -18.0},
    "model/priors/Vext": {
        "dist": "vector_uniform_fixed",
        "low": 0.0,
        "high": 1000.0,
    },
}

TRGBH0_COMMON = {
    "inference/num_chains": 1,
    "inference/chain_method": "sequential",
    "inference/num_warmup": 1000,
    "inference/num_samples": 5000,
    "model/selection_integral_geometry": "sphere",
    "model/selection_integral_grid_radius": 75.0,
    "model/density_3d_subsample_fraction": 1.0,
    "model/priors/Vext": {
        "dist": "vector_uniform_fixed",
        "low": 0.0,
        "high": 1000.0,
    },
}


def _delta(value):
    return {"dist": "delta", "value": value}


def _normal(loc, scale):
    return {"dist": "normal", "loc": loc, "scale": scale}


def _ch0_selection(selection):
    return {"model/which_selection": selection}


def _trgbh0_selection(selection):
    return {"model/which_selection": selection}


def _with_root(root_output):
    return {"io/root_output": root_output}


def _ch0_main_datasets():
    selections = ("none", "SN_magnitude", "redshift")
    pv_models = [
        {
            "model/use_reconstruction": False,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/priors/Vext": _delta([0.0, 0.0, 0.0]),
        },
        {
            "model/use_reconstruction": False,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
        },
        {
            "model/use_reconstruction": False,
            "model/use_fiducial_Cepheid_host_PV_covariance": True,
            "model/use_PV_covmat_scaling": False,
            "model/priors/Vext": _delta([0.0, 0.0, 0.0]),
        },
        {
            "model/use_reconstruction": False,
            "model/use_fiducial_Cepheid_host_PV_covariance": True,
            "model/use_PV_covmat_scaling": True,
            "model/priors/Vext": _delta([0.0, 0.0, 0.0]),
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/which_host_los": "Carrick2015",
            "model/priors/beta": _normal(0.43, 0.02),
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/which_host_los": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": True,
            "io/SH0ES/which_host_los": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/which_host_los": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
            "model/priors/beta": _normal(1.0, 0.5),
        },
    ]

    datasets = []
    for pv_model in pv_models:
        for selection in selections:
            dataset = {
                **pv_model,
                **_ch0_selection(selection),
            }
            if dataset.get("model/use_fiducial_Cepheid_host_PV_covariance") \
                    and selection == "redshift":
                dataset["model/weight_selection_by_covmat_Neff"] = True
            else:
                dataset.setdefault(
                    "model/weight_selection_by_covmat_Neff", False)
            datasets.append(dataset)
    return datasets


def _ch0_distance_only_datasets():
    base = {
        "model/use_Cepheid_host_redshift": False,
        "model/use_reconstruction": False,
        "model/use_fiducial_Cepheid_host_PV_covariance": False,
        "model/use_PV_covmat_scaling": False,
        "model/weight_selection_by_covmat_Neff": False,
        "model/priors/Vext": _delta([0.0, 0.0, 0.0]),
        **_with_root(f"{CH0_PAPER_ROOT}/distances"),
    }
    return [
        {
            **base,
            **_ch0_selection("none"),
            "model/use_uniform_mu_host_priors": True,
        },
        {
            **base,
            **_ch0_selection("none"),
            "model/use_uniform_mu_host_priors": False,
        },
        {
            **base,
            **_ch0_selection("SN_magnitude"),
            "model/use_uniform_mu_host_priors": False,
        },
    ]


def _ch0_mixed_selection_datasets():
    return [
        {
            "model/which_selection": "SN_magnitude_or_redshift_Nmag",
            "model/num_hosts_selection_mag": n_mag,
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/which_host_los": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
        }
        for n_mag in range(36)
    ]


def _trgbh0_selection_datasets(pv_models, selections=("TRGB_magnitude",
                                                      "redshift")):
    return [
        {
            **pv_model,
            **_trgbh0_selection(selection),
        }
        for pv_model in pv_models
        for selection in selections
    ]


def _trgbh0_main_datasets():
    selections = ("TRGB_magnitude", "redshift")
    main_pv_models = [
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "io/PV_main/EDD_TRGB/which_host_los": "Carrick2015",
            "model/priors/beta": _normal(0.43, 0.02),
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "io/PV_main/EDD_TRGB/which_host_los": TRGBH0_MANTICORE_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": False,
        },
    ]
    extra_pv_models = [
        {
            "model/use_reconstruction": False,
            "model/priors/Vext": _delta([0.0, 0.0, 0.0]),
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "io/PV_main/EDD_TRGB/which_host_los": "Carrick2015",
            "model/priors/beta": _normal(1.0, 0.5),
        },
    ]
    grouped_pv_models = [
        {
            "model/which_run": "EDD_TRGB_grouped",
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "io/PV_main/EDD_TRGB_grouped/which_host_los": "Carrick2015",
            "model/priors/beta": _normal(0.43, 0.02),
        },
        {
            "model/which_run": "EDD_TRGB_grouped",
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "io/PV_main/EDD_TRGB_grouped/which_host_los": TRGBH0_MANTICORE_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
        },
        {
            "model/which_run": "EDD_TRGB_grouped",
            "model/use_reconstruction": False,
        },
    ]

    return (
        _trgbh0_selection_datasets(main_pv_models, selections)
        + _trgbh0_selection_datasets(extra_pv_models, selections)
        + _trgbh0_selection_datasets(grouped_pv_models, selections)
    )


TASK_SPECS = {
    "CH0_main": {
        "description": "CH0 paper H0 grid plus redshift-free distance runs.",
        "config_path": "configs/config_CH0.toml",
        "tag": "paper",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
            **_with_root(f"{CH0_PAPER_ROOT}/table"),
        },
        "datasets": _ch0_main_datasets() + _ch0_distance_only_datasets(),
        "expected_tasks": 27,
    },
    "CH0_mixed_selection": {
        "description": "CH0 paper mixed SN-magnitude/redshift split.",
        "config_path": "configs/config_CH0.toml",
        "tag": "paper_mixed",
        "common": {
            **CH0_PAPER_COMMON,
            **_with_root(f"{CH0_PAPER_ROOT}/mixed_selection"),
            "model/density_3d_subsample_fraction": 0.25,
        },
        "datasets": _ch0_mixed_selection_datasets(),
        "expected_tasks": 36,
    },
    "CH0_test": {
        "description": "CH0 Manticore SN/redshift selection test with 25% voxel subsampling.",
        "config_path": "configs/config_CH0.toml",
        "tag": "test",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/num_chains": 1,
            **_with_root("results/CH0_test"),
            "model/density_3d_subsample_fraction": 0.25,
        },
        "datasets": [
            {
                "model/use_reconstruction": True,
                "model/use_fiducial_Cepheid_host_PV_covariance": False,
                "model/use_PV_covmat_scaling": False,
                "model/weight_selection_by_covmat_Neff": False,
                "model/use_density_dependent_sigma_v": False,
                "io/SH0ES/which_host_los": CH0_MANTICORE_LOS,
                "model/which_bias": "linear",
                **_ch0_selection(selection),
            }
            for selection in ("SN_magnitude", "redshift")
        ],
        "expected_tasks": 2,
    },
    "TRGBH0_main": {
        "description": "TRGB H0 grid: PV-field, Vext-only, no-Vext, and grouped selections.",
        "config_path": "configs/config_EDD_TRGB.toml",
        "tag": "main",
        "common": {
            **TRGBH0_COMMON,
            **_with_root(f"{TRGBH0_ROOT}/table"),
        },
        "datasets": _trgbh0_main_datasets(),
        "expected_tasks": 16,
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
            "model/priors/nu_cz": {
                "dist": "truncated_normal",
                "low": 2.0,
                "high": 100.0,
                "mean": 30.0,
                "scale": 10.0,
            },
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
