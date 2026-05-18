"""Named task-list specifications for generate_tasks.py.

Keep this file focused on sweep definitions. Generator behavior belongs in
generate_tasks.py.
"""


CH0_PAPER_ROOT = "results/CH0_paper"
CH0_MANTICORE_LOS = "manticore_2MPP_MULTIBIN_N256_DES_V2"
CH0_MANTICORE_BIAS = "double_powerlaw"
TRGBH0_ROOT = "results/TRGBH0_paper"
TRGBH0_MANTICORE_LOS = "manticore_2MPP_MULTIBIN_N256_DES_V2"
TRGBH0_MANTICORE_COLA_LOS = "COLA_manticore_2MPP_MULTIBIN_N256_DES_V2"
TRGBH0_MANTICORE_BIAS = "double_powerlaw"
TRGBH0_CARRICK_BETA_LOC = 0.461
TRGBH0_CARRICK_BETA_SCALE = 0.013
TRGBH0_EDD_MAG_MIN = 22.1
TRGBH0_EDD_MAG_LIM_LOW = 22.101
TRGBH0_EDD_MAG_LIM_HIGH = 29.0
TRGBH0_SELECTION_SUPERSAMPLE_RADIUS = 15.0
TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX = 0.325
TRGBH0_CCHP_SELECTION_SUPERSAMPLE_RADIUS = 30.0
TRGBH0_CCHP_SELECTION_SUPERSAMPLE_TARGET_DX = 0.3
S8_ROOT = "results/S8"
S8_PV_KIND = "precomputed_los_Carrick2015"
S8_BIAS_MODELS = ["linear", "quadratic", "double_powerlaw"]
VEXT_RAD_ROOT = "results/Vext_rad"
VEXT_RAD_SDSS_FP_VEXT_KNOTS = [0, 100, 200, 300, 400]
VEXT_RAD_SDSS_FP_CARRICK_KNOTS = [0, 20, 40, 60, 80, 100, 120, 140]
VEXT_RAD_SDSS_FP_VEXT_PRIOR = {
    "dist": "vector_radial_uniform",
    "low": 0.0,
    "high": 500,
    "rknot": VEXT_RAD_SDSS_FP_VEXT_KNOTS,
    "method": "cubic",
}
VEXT_RADMAG_SDSS_FP_VEXT_PRIOR = {
    "dist": "vector_radialmag_uniform",
    "low": 0.0,
    "high": 500,
    "rknot": VEXT_RAD_SDSS_FP_VEXT_KNOTS,
    "method": "cubic",
}
VEXT_RAD_SDSS_FP_CARRICK_PRIOR = {
    "dist": "vector_radial_uniform",
    "low": 0.0,
    "high": 500,
    "rknot": VEXT_RAD_SDSS_FP_CARRICK_KNOTS,
    "method": "cubic",
}
VEXT_RADMAG_SDSS_FP_CARRICK_PRIOR = {
    "dist": "vector_radialmag_uniform",
    "low": 0.0,
    "high": 500,
    "rknot": VEXT_RAD_SDSS_FP_CARRICK_KNOTS,
    "method": "cubic",
}
VFO_ROOT = "results/VFO"
VFO_MANTICORE_LOS = "manticore_2MPP_MULTIBIN_N256_DES_V2"
VFO_MANTICORE_COLA_LOS = "COLA_manticore_2MPP_MULTIBIN_N256_DES_V2"

CH0_PAPER_COMMON = {
    "inference/compute_log_density": False,
    "inference/compute_evidence": False,
    "inference/num_chains": 12,
    "inference/chain_method": "sequential",
    "inference/num_warmup": 1000,
    "inference/num_samples": 6000,
    "model/use_uniform_mu_host_priors": False,
    "model/selection_integral_geometry": "sphere",
    "model/selection_integral_grid_radius": 60.0,
    "model/density_3d_subsample_fraction": 1.0,
    "model/priors/M_B": {"dist": "uniform", "low": -22.0, "high": -18.0},
    "model/priors/Vext": {
        "dist": "vector_uniform_fixed",
        "low": 0.0,
        "high": 1000.0,
    },
}

TRGBH0_COMMON = {
    "inference/compute_log_density": True,
    "inference/compute_evidence": True,
    "inference/num_chains": 1,
    "inference/chain_method": "sequential",
    "inference/num_warmup": 1000,
    "inference/num_samples": 1000,
    "model/selection_integral_geometry": "sphere",
    "model/selection_integral_grid_radius": 50.0,
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


def _nu_cz_student_t_prior():
    return {
        "dist": "truncated_normal",
        "low": 1.0,
        "high": 100.0,
        "mean": 30.0,
        "scale": 10.0,
    }


def _trgbh0_carrick_beta_prior():
    return _normal(TRGBH0_CARRICK_BETA_LOC, TRGBH0_CARRICK_BETA_SCALE)


def _trgbh0_edd_mag_lim_prior():
    return {
        "dist": "uniform",
        "low": TRGBH0_EDD_MAG_LIM_LOW,
        "high": TRGBH0_EDD_MAG_LIM_HIGH,
    }


def _trgbh0_cchp_config():
    return {
        "config_path": "configs/config_CCHP_TRGB.toml",
        "model/selection_integral_supersample_radius": (
            TRGBH0_CCHP_SELECTION_SUPERSAMPLE_RADIUS),
        "model/selection_integral_supersample_target_dx": (
            TRGBH0_CCHP_SELECTION_SUPERSAMPLE_TARGET_DX),
    }


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


def _trgbh0_selection_datasets(pv_models, selections=("TRGB_magnitude",)):
    return [
        {
            **pv_model,
            **_trgbh0_selection(selection),
        }
        for pv_model in pv_models
        for selection in selections
    ]


def _trgbh0_main_datasets():
    selections = ("TRGB_magnitude",)
    main_pv_models = [
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "inference/init_maxiter": 0,
            "io/PV_main/EDD_TRGB/which_host_los": "Carrick2015",
            "model/priors/beta": _trgbh0_carrick_beta_prior(),
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "io/PV_main/EDD_TRGB/which_host_los": TRGBH0_MANTICORE_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "io/PV_main/EDD_TRGB/which_host_los": TRGBH0_MANTICORE_COLA_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
        },
    ]
    extra_pv_models = [
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "student_t",
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "model/priors/nu_cz": _nu_cz_student_t_prior(),
            "io/PV_main/EDD_TRGB/which_host_los": "Carrick2015",
            "model/priors/beta": _trgbh0_carrick_beta_prior(),
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "student_t",
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "model/priors/nu_cz": _nu_cz_student_t_prior(),
            "io/PV_main/EDD_TRGB/which_host_los": TRGBH0_MANTICORE_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "gaussian",
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "io/PV_main/EDD_TRGB/which_host_los": TRGBH0_MANTICORE_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 2.0,
            },
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "student_t",
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "model/priors/nu_cz": _nu_cz_student_t_prior(),
            "io/PV_main/EDD_TRGB/which_host_los": TRGBH0_MANTICORE_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 2.0,
            },
        },
        {
            "model/use_reconstruction": False,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "student_t",
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "model/priors/nu_cz": _nu_cz_student_t_prior(),
        },
    ]
    return (
        _trgbh0_selection_datasets(main_pv_models, selections)
        + _trgbh0_selection_datasets(extra_pv_models, selections)
        + _trgbh0_cchp_subset_datasets()
        + _trgbh0_distance_only_datasets()
    )


def _trgbh0_distance_only_datasets():
    return [
        {
            "model/use_TRGB_host_redshift": False,
            "model/use_reconstruction": False,
            "model/use_density_dependent_sigma_v": False,
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "model/priors/H0": _delta(73.04),
            "model/priors/Vext": _delta([0.0, 0.0, 0.0]),
            "model/priors/sigma_v": _delta(100.0),
            **_trgbh0_selection("TRGB_magnitude"),
            **_with_root(f"{TRGBH0_ROOT}/distances"),
        },
    ]


def _trgbh0_manticore_field_datasets():
    datasets = []
    for los, n_fields in (
            (TRGBH0_MANTICORE_LOS, 30),
            (TRGBH0_MANTICORE_COLA_LOS, 50),
    ):
        for field in range(n_fields):
            datasets.append({
                "model/use_reconstruction": True,
                "model/use_density_dependent_sigma_v": False,
                "model/cz_likelihood": "gaussian",
                "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
                "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
                "io/PV_main/EDD_TRGB/which_host_los": los,
                "model/which_bias": TRGBH0_MANTICORE_BIAS,
                "io/field_indices": field,
                **_trgbh0_selection("TRGB_magnitude"),
            })
    return datasets


def _trgbh0_cchp_subset_datasets():
    main_models = [
        {
            **_trgbh0_cchp_config(),
            "model/use_reconstruction": False,
            "model/use_density_dependent_sigma_v": False,
        },
        {
            **_trgbh0_cchp_config(),
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "io/which_host_los": "Carrick2015",
            "model/priors/beta": _trgbh0_carrick_beta_prior(),
        },
        {
            **_trgbh0_cchp_config(),
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "io/which_host_los": TRGBH0_MANTICORE_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
        },
    ]
    student_t_models = [
        {
            **_trgbh0_cchp_config(),
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "student_t",
            "model/priors/nu_cz": _nu_cz_student_t_prior(),
            "io/which_host_los": "Carrick2015",
            "model/priors/beta": _trgbh0_carrick_beta_prior(),
        },
        {
            **_trgbh0_cchp_config(),
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "student_t",
            "model/priors/nu_cz": _nu_cz_student_t_prior(),
            "io/which_host_los": TRGBH0_MANTICORE_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
        },
    ]
    return (
        _trgbh0_selection_datasets(
            main_models, selections=("redshift", "TRGB_magnitude"))
        + _trgbh0_selection_datasets(
            student_t_models, selections=("TRGB_magnitude",))
    )


def _s8_production_datasets():
    return [
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_W1",
        },
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_i",
        },
        {
            "inference/model": "FPModel",
            "io/catalogue_name": "6dF_FP",
            "pv_model/galaxy_bias": [*S8_BIAS_MODELS, "cubic"],
        },
        {
            "inference/model": "FPModel",
            "io/catalogue_name": "SDSS_FP",
            "pv_model/galaxy_bias": [*S8_BIAS_MODELS, "cubic"],
        },
        {
            "inference/model": "PantheonPlusModel",
            "io/catalogue_name": "PantheonPlus",
            "inference/init_maxiter": 0,
        },
        {
            "inference/model": [
                "TFRModel", "TFRModel", "PantheonPlusModel"],
            "io/catalogue_name": ["CF4_i", "CF4_W1", "PantheonPlus"],
            "inference/shared_params": "beta,sigma_v",
            "inference/init_maxiter": 0,
        },
    ]


def _vfo_datasets():
    catalogues = [
        {
            "inference/model": "SNModel",
            "io/catalogue_name": "LOSS",
        },
        {
            "inference/model": "SNModel",
            "io/catalogue_name": "Foundation",
        },
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_W1",
        },
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_i",
        },
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "2MTF",
        },
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "SFI",
        },
    ]
    pv_models = [
        {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "linear",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 1.0,
            },
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": _delta(1.0),
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": _delta(1.0),
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "manticore_stdp",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": _delta(1.0),
        },
    ]
    manticore_linear_models = [
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "linear",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": _delta(1.0),
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "quadratic",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": _delta(1.0),
        },
    ]
    manticore_beta_free_models = [
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": _normal(1.0, 0.1),
        },
    ]
    carrick_double_powerlaw_models = [
        {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 1.0,
            },
        },
    ]
    fp_catalogues = [
        {
            "inference/model": "FPModel",
            "io/catalogue_name": "SDSS_FP",
        },
        {
            "inference/model": "FPModel",
            "io/catalogue_name": "6dF_FP",
        },
    ]
    fp_pv_models = [
        {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "linear",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 1.0,
            },
        },
        {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 1.0,
            },
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": _delta(1.0),
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 2.0,
            },
        },
    ]
    student_t_pv_models = [
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": _delta(1.0),
            "model/cz_likelihood": "student_t",
            "model/priors/nu_cz": _nu_cz_student_t_prior(),
        },
        {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "linear",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 1.0,
            },
            "model/cz_likelihood": "student_t",
            "model/priors/nu_cz": _nu_cz_student_t_prior(),
        },
    ]
    cola_manticore_pv_models = [
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_COLA_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": _delta(1.0),
            "model/cz_likelihood": "gaussian",
        },
    ]
    return [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues
        for pv_model in pv_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues
        for pv_model in manticore_linear_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues
        for pv_model in manticore_beta_free_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues
        for pv_model in carrick_double_powerlaw_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in fp_catalogues
        for pv_model in fp_pv_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues + fp_catalogues
        for pv_model in student_t_pv_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues + fp_catalogues
        for pv_model in cola_manticore_pv_models
    ]


def _vfo_single_datasets():
    datasets = []
    for los, n_fields, subsample_fraction in (
            (VFO_MANTICORE_LOS, 30, 0.1),
            (VFO_MANTICORE_COLA_LOS, 50, 0.5),
    ):
        for field in range(n_fields):
            dataset = {
                "pv_model/kind": f"precomputed_los_{los}",
                "io/field_indices": field,
            }
            if subsample_fraction != 0.1:
                dataset[
                    "pv_model/density_3d_subsample_fraction"
                ] = subsample_fraction
            datasets.append(dataset)
    return datasets


TASK_SPECS = {
    "test": {
        "description": "CF4 TFR W1 Mmiss run with a single NUTS chain.",
        "config_path": "configs/config.toml",
        "tag": "Mmiss_single_chain",
        "common": {
            "inference/model": "TFRModel",
            "inference/num_chains": 1,
            "inference/chain_method": "sequential",
            "inference/num_warmup": 500,
            "inference/num_samples": 500,
            "inference/compute_evidence": False,
            "inference/compute_log_density": False,
            "inference/target_accept_prob": 0.9,
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "linear",
            "pv_model/use_Mmiss": True,
            "pv_model/Mmiss_sigma": 5.0,
            "pv_model/Mmiss_coordinate_frame": "galactic",
            "pv_model/dr_malmquist": 1.0,
            "pv_model/density_3d_geometry": "sphere",
            "pv_model/density_3d_radius": 200.0,
            "pv_model/density_3d_downsample": 1,
            "model/priors/Mmiss_distance/high": [
                50.0, 100.0, 150.0, 200.0, 250.0],
            "model/priors/Vext": _delta([0.0, 0.0, 0.0]),
            "model/priors/b1": {
                "dist": "truncated_normal",
                "low": 0.1,
                "mean": 1.2,
                "scale": 0.4,
            },
            "io/root_output": "results/test",
        },
        "datasets": [
            {
                "io/catalogue_name": "CF4_W1",
            },
        ],
        "expected_tasks": 5,
    },
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
            "inference/num_chains": 1,
            **_with_root(f"{CH0_PAPER_ROOT}/mixed_selection"),
            "model/density_3d_subsample_fraction": 0.5,
        },
        "datasets": _ch0_mixed_selection_datasets(),
        "expected_tasks": 36,
    },
    "Vext_rad": {
        "description": (
            "CF4 W1 and SDSS FP radial Vext comparison."),
        "config_path": "configs/config.toml",
        "common": {
            "inference/model": "TFRModel",
            "inference/num_chains": 1,
            "inference/chain_method": "sequential",
            "inference/compute_log_density": False,
            "inference/compute_evidence": False,
            "pv_model/which_Vext": [
                "constant", "radial", "radial_magnitude"],
            "io/root_output": VEXT_RAD_ROOT,
        },
        "datasets": [
            {
                "io/catalogue_name": "CF4_W1",
                "pv_model/kind": "Vext",
            },
            {
                "io/catalogue_name": "CF4_W1",
                "pv_model/kind": "precomputed_los_Carrick2015",
                "pv_model/galaxy_bias": "linear",
                "model/priors/beta": {
                    "dist": "uniform",
                    "low": 0.0,
                    "high": 2.0,
                },
            },
            {
                "inference/model": "FPModel",
                "io/catalogue_name": "SDSS_FP",
                "io/SDSS_FP/zcmb_max": 0.1,
                "pv_model/kind": "Vext",
                "model/priors/Vext_radial": VEXT_RAD_SDSS_FP_VEXT_PRIOR,
                "model/priors/Vext_radial_magnitude": (
                    VEXT_RADMAG_SDSS_FP_VEXT_PRIOR),
            },
            {
                "inference/model": "FPModel",
                "io/catalogue_name": "SDSS_FP",
                "pv_model/kind": "precomputed_los_Carrick2015",
                "pv_model/galaxy_bias": "linear",
                "model/priors/Vext_radial": VEXT_RAD_SDSS_FP_CARRICK_PRIOR,
                "model/priors/Vext_radial_magnitude": (
                    VEXT_RADMAG_SDSS_FP_CARRICK_PRIOR),
                "model/priors/beta": {
                    "dist": "uniform",
                    "low": 0.0,
                    "high": 2.0,
                },
            },
        ],
        "expected_tasks": 12,
    },
    "TRGBH0_main": {
        "description": "TRGB H0 grid plus redshift-free distance run.",
        "config_path": "configs/config_EDD_TRGB.toml",
        "tag": "main",
        "common": {
            **TRGBH0_COMMON,
            "inference/num_chains_harmonic": 10,
            "model/selection_integral_supersample_radius": (
                TRGBH0_SELECTION_SUPERSAMPLE_RADIUS),
            "model/selection_integral_supersample_target_dx": (
                TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX),
            **_with_root(f"{TRGBH0_ROOT}/table"),
        },
        "datasets": _trgbh0_main_datasets(),
        "expected_tasks": 17,
    },
    "TRGBH0_manticore_fields_const_sigv": {
        "description": (
            "TRGB H0 one-Manticore-field runs with Gaussian constant "
            "sigma_v."),
        "config_path": "configs/config_EDD_TRGB.toml",
        "tag": "manticore_field_const_sigv",
        "common": {
            **TRGBH0_COMMON,
            "inference/num_warmup": 1000,
            "inference/num_samples": 1000,
            "model/priors/H0/low": 40,
            "model/priors/H0/high": 100,
            "model/selection_integral_supersample_radius": (
                TRGBH0_SELECTION_SUPERSAMPLE_RADIUS),
            "model/selection_integral_supersample_target_dx": (
                TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX),
            "model/priors/mag_lim_TRGB_width/low": 0.15,
            "model/priors/sigma_int": {
                "dist": "delta",
                "value": 0.1,
            },
            **_with_root(f"{TRGBH0_ROOT}/manticore_fields_const_sigv"),
        },
        "datasets": _trgbh0_manticore_field_datasets(),
        "expected_tasks": 80,
    },
    "S8_production": {
        "description": (
            "S8 production PV sweep for individual and joint catalogues."),
        "config_path": "configs/config.toml",
        "tag": "default",
        "common": {
            "pv_model/kind": S8_PV_KIND,
            "pv_model/galaxy_bias": S8_BIAS_MODELS,
            "pv_model/density_3d_downsample": 1,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 2.0,
            },
            "inference/num_chains": 1,
            "inference/num_warmup": 2000,
            "inference/num_samples": 10000,
            "io/root_output": S8_ROOT,
        },
        "datasets": _s8_production_datasets(),
        "expected_tasks": 20,
    },
    "VFO": {
        "description": (
            "VFO PV catalogue Carrick2015/Manticore/COLA comparison."),
        "config_path": "configs/config.toml",
        "tag": "paper",
        "common": {
            "pv_model/density_3d_geometry": "sphere",
            "pv_model/density_3d_radius": 150.0,
            "pv_model/density_3d_downsample": 1,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
            "io/root_output": VFO_ROOT,
        },
        "datasets": _vfo_datasets(),
        "expected_tasks": 80,
    },
    "VFO_single": {
        "description": (
            "CF4 W1 TFR one-field Manticore/COLA runs for the VFO setup."),
        "config_path": "configs/config.toml",
        "tag": "single",
        "common": {
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_W1",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "pv_model/density_3d_geometry": "sphere",
            "pv_model/density_3d_radius": 150.0,
            "pv_model/density_3d_downsample": 1,
            "model/priors/beta": _delta(1.0),
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
            "io/root_output": f"{VFO_ROOT}/single_fields",
        },
        "datasets": _vfo_single_datasets(),
        "expected_tasks": 80,
    },
}
