"""Named task-list specifications for generate_tasks.py.

Keep this file focused on sweep definitions. Generator behavior belongs in
generate_tasks.py.
"""


CH0_PAPER_ROOT = "results/CH0_paper"
CH0_MANTICORE_LOS = "ManticoreLocalSWIFT"
CH0_MANTICORE_COLA_LOS = "ManticoreLocalCOLA"
CH0_MANTICORE_BIAS = "double_powerlaw"
TRGBH0_ROOT = "results/TRGBH0_paper"
TRGBH0_MANTICORE_LOS = "ManticoreLocalSWIFT"
TRGBH0_MANTICORE_COLA_LOS = "ManticoreLocalCOLA"
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
VFO_MANTICORE_LOS = "ManticoreLocalSWIFT"
VFO_MANTICORE_COLA_LOS = "ManticoreLocalCOLA"

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
    "inference/num_chains_harmonic": 10,
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


CH0_FIXED_BIAS_PRIORS = {
    "model/priors/alpha_low": _delta(1.835),
    "model/priors/alpha_high": _delta(0.343),
    "model/priors/log_rho_t": _delta(0.313),
    "model/priors/log_rho_width": _delta(0.879),
}

CH0_SWIFT_FIXED_BIAS_PRIORS = {
    "model/priors/alpha_low": _delta(1.542),
    "model/priors/alpha_high": _delta(0.286),
    "model/priors/log_rho_t": _delta(-0.027),
    "model/priors/log_rho_width": _delta(0.954),
}


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


def _trgbh0_edd_mag_lim_uninformative_prior():
    return {
        "dist": "uniform",
        "low": TRGBH0_EDD_MAG_LIM_LOW,
        "high": TRGBH0_EDD_MAG_LIM_HIGH,
    }


def _trgbh0_edd_mag_lim_informative_prior():
    return {
        "dist": "truncated_normal",
        "mean": 24.1,
        "scale": 0.5,
        "low": TRGBH0_EDD_MAG_LIM_LOW,
    }


def _trgbh0_edd_mag_lim_prior():
    prior = _trgbh0_edd_mag_lim_informative_prior()
    # prior = _trgbh0_edd_mag_lim_uninformative_prior()
    return prior


def _trgbh0_sigma_int_uninformative_prior():
    return {
        "dist": "maxwell",
        "scale": 0.0627,
    }


def _trgbh0_sigma_int_informative_prior():
    return {
        "dist": "truncated_normal",
        "mean": 0.1,
        "scale": 0.01,
        "low": 0.01,
    }


def _trgbh0_sigma_int_prior():
    prior = _trgbh0_sigma_int_informative_prior()
    # prior = _trgbh0_sigma_int_uninformative_prior()
    return prior


TRGBH0_COMMON["model/priors/sigma_int"] = _trgbh0_sigma_int_prior()
TRGBH0_COMMON["model/priors/alpha_c"] = _delta(-0.2)
TRGBH0_COMMON["model/priors/alpha_high/low"] = 0.0


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
            "io/SH0ES/reconstruction": "Carrick2015",
            "model/priors/beta": _normal(0.43, 0.02),
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": True,
            "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
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
            "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
        }
        for n_mag in range(36)
    ]


def _ch0_manticore_field_datasets():
    datasets = []
    for field in range(30):
        datasets.append({
            "model/which_selection": "SN_magnitude",
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
            "io/field_indices": field,
        })

    for mas in ("CIC", "PCS", "SPH"):
        for field in range(80):
            datasets.append({
                "model/which_selection": "SN_magnitude",
                "model/use_reconstruction": True,
                "model/use_fiducial_Cepheid_host_PV_covariance": False,
                "model/use_PV_covmat_scaling": False,
                "model/weight_selection_by_covmat_Neff": False,
                "model/use_density_dependent_sigma_v": False,
                "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
                "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": mas,
                "model/which_bias": CH0_MANTICORE_BIAS,
                "io/field_indices": field,
            })

    return datasets


def _ch0_manticore_cola_cic_field_datasets():
    return [
        {
            "model/which_selection": "SN_magnitude",
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
            "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "CIC",
            "model/which_bias": CH0_MANTICORE_BIAS,
            "io/field_indices": field,
        }
        for field in range(80)
    ]


def _ch0_manticore_cola_cic_fixed_bias_datasets():
    return [
        {
            **dataset,
            "model/which_bias": CH0_MANTICORE_BIAS,
            **CH0_FIXED_BIAS_PRIORS,
        }
        for dataset in _ch0_manticore_cola_cic_field_datasets()
    ]


def _ch0_manticore_swift_fixed_bias_datasets():
    return [
        {
            "model/which_selection": "SN_magnitude",
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
            "io/field_indices": field,
            **CH0_SWIFT_FIXED_BIAS_PRIORS,
        }
        for field in range(30)
    ]


def _ch0_manticore_swift_cola_cic_uniform_bias_datasets():
    swift_datasets = [
        {
            "model/which_selection": "SN_magnitude",
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
            "model/which_bias": "uniform",
            "model/field_3d_smoothing_scale": 0.0,
            "io/field_indices": field,
        }
        for field in range(30)
    ]
    cola_datasets = [
        {
            **dataset,
            "model/which_bias": "uniform",
            "model/field_3d_smoothing_scale": 0.0,
        }
        for dataset in _ch0_manticore_cola_cic_field_datasets()
    ]
    return swift_datasets + cola_datasets


def _ch0_leaveoneout_datasets():
    return [{
        "model/which_selection": "SN_magnitude",
        "model/use_reconstruction": True,
        "model/use_fiducial_Cepheid_host_PV_covariance": False,
        "model/use_PV_covmat_scaling": False,
        "model/weight_selection_by_covmat_Neff": False,
        "model/use_density_dependent_sigma_v": False,
        "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
        "model/which_bias": CH0_MANTICORE_BIAS,
        "io/field_indices": 21,
        "io/SH0ES/drop_observation": list(range(35)),
    }]


def _ch0_angular_scatter_datasets():
    return [{
        "model/which_selection": "SN_magnitude",
        "model/use_reconstruction": True,
        "model/use_fiducial_Cepheid_host_PV_covariance": False,
        "model/use_PV_covmat_scaling": False,
        "model/weight_selection_by_covmat_Neff": False,
        "model/use_density_dependent_sigma_v": False,
        "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
        "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "CIC",
        "model/which_bias": CH0_MANTICORE_BIAS,
        "io/field_indices": list(range(30)),
        "io/angular_position_scatter_deg": [2.0, 4.0, 8.0, 16.0],
        "io/angular_position_scatter_seed": 42,
    }]


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
            "io/PV_main/EDD_TRGB/reconstruction": "Carrick2015",
            "model/priors/beta": _trgbh0_carrick_beta_prior(),
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "io/PV_main/EDD_TRGB/reconstruction": TRGBH0_MANTICORE_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "io/PV_main/EDD_TRGB/reconstruction": TRGBH0_MANTICORE_COLA_LOS,
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
            "io/PV_main/EDD_TRGB/reconstruction": "Carrick2015",
            "model/priors/beta": _trgbh0_carrick_beta_prior(),
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "student_t",
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "model/priors/nu_cz": _nu_cz_student_t_prior(),
            "io/PV_main/EDD_TRGB/reconstruction": TRGBH0_MANTICORE_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "gaussian",
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
            "io/PV_main/EDD_TRGB/reconstruction": TRGBH0_MANTICORE_LOS,
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
            "io/PV_main/EDD_TRGB/reconstruction": TRGBH0_MANTICORE_LOS,
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
    for field in range(50):
        datasets.append({
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "gaussian",
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": (
                _trgbh0_edd_mag_lim_uninformative_prior()),
            "io/PV_main/EDD_TRGB/reconstruction": TRGBH0_MANTICORE_COLA_LOS,
            "model/which_bias": TRGBH0_MANTICORE_BIAS,
            "io/field_indices": field,
            **_trgbh0_selection("TRGB_magnitude"),
        })
    return datasets


def _trgbh0_manticore_cola_mas_field_datasets(mas_values):
    datasets = []
    for mas in mas_values:
        for field in range(80):
            datasets.append({
                "model/use_reconstruction": True,
                "model/use_density_dependent_sigma_v": False,
                "model/cz_likelihood": "gaussian",
                "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
                "model/priors/mag_lim_TRGB": (
                    _trgbh0_edd_mag_lim_uninformative_prior()),
                "io/PV_main/EDD_TRGB/reconstruction": (
                    TRGBH0_MANTICORE_COLA_LOS),
                "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": mas,
                "model/which_bias": TRGBH0_MANTICORE_BIAS,
                "io/field_indices": field,
                **_trgbh0_selection("TRGB_magnitude"),
            })
    return datasets


def _trgbh0_manticore_cola_single_datasets():
    return _trgbh0_manticore_cola_mas_field_datasets(("CIC", "PCS", "SPH"))


def _trgbh0_manticore_cola_pcs_field_datasets():
    return _trgbh0_manticore_cola_mas_field_datasets(("PCS",))


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
            "io/CCHP/reconstruction": "Carrick2015",
            "model/priors/beta": _trgbh0_carrick_beta_prior(),
        },
        {
            **_trgbh0_cchp_config(),
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "io/CCHP/reconstruction": TRGBH0_MANTICORE_LOS,
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
            "io/CCHP/reconstruction": "Carrick2015",
            "model/priors/beta": _trgbh0_carrick_beta_prior(),
        },
        {
            **_trgbh0_cchp_config(),
            "model/use_reconstruction": True,
            "model/use_density_dependent_sigma_v": False,
            "model/cz_likelihood": "student_t",
            "model/priors/nu_cz": _nu_cz_student_t_prior(),
            "io/CCHP/reconstruction": TRGBH0_MANTICORE_LOS,
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
        "description": (
            "Foundation SN simple Carrick/COLA field test with optional "
            "density smoothing."),
        "config_path": "configs/config.toml",
        "tag": "foundation_simple",
        "common": {
            "inference/model": "SNModel",
            "io/catalogue_name": "Foundation",
            "inference/num_chains": 1,
            "inference/chain_method": "sequential",
            "inference/num_warmup": 500,
            "inference/num_samples": 500,
            "inference/compute_evidence": False,
            "inference/compute_log_density": False,
            "inference/target_accept_prob": 0.9,
            "pv_model/density_3d_geometry": "sphere",
            "pv_model/density_3d_radius": 150.0,
            "pv_model/density_3d_downsample": 1,
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/field_3d_smoothing_scale": [0.0, 8.0],
            "io/root_output": "results/test",
        },
        "datasets": [
            {
                "pv_model/kind": "precomputed_los_Carrick2015",
                "pv_model/galaxy_bias": "linear",
                "model/priors/beta": {
                    "dist": "uniform",
                    "low": 0.0,
                    "high": 1.0,
                },
            },
            {
                "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_COLA_LOS}",
                "pv_model/galaxy_bias": "double_powerlaw",
                "model/priors/beta": _delta(1.0),
                "model/cz_likelihood": "gaussian",
                "io/field_indices": 0,
            },
            {
                "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_COLA_LOS}",
                "pv_model/galaxy_bias": "double_powerlaw",
                "model/priors/beta": _delta(1.0),
                "model/cz_likelihood": "gaussian",
            },
        ],
        "expected_tasks": 6,
    },
    "CH0_main": {
        "description": "CH0 paper H0 grid plus redshift-free distance runs.",
        "config_path": "configs/config_CH0.toml",
        "tag": "paper",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
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
    "CH0_single": {
        "description": (
            "CH0 SN-magnitude one-Manticore-field runs with evidence."),
        "config_path": "configs/config_CH0.toml",
        "tag": "single",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            "inference/save_log_likelihood_per_galaxy": True,
            **_with_root(f"{CH0_PAPER_ROOT}/single_fields"),
        },
        "datasets": _ch0_manticore_field_datasets(),
        "expected_tasks": 270,
    },
    "CH0_single_smoothed": {
        "description": (
            "CH0 CIC COLA one-field runs with density-field smoothing."),
        "config_path": "configs/config_CH0.toml",
        "tag": "single_smoothed",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            "inference/save_log_likelihood_per_galaxy": True,
            "model/field_3d_smoothing_scale": [4.0, 8.0, 16.0, 32.0],
            "model/velocity_3d_smoothing_scale": 0.0,
            **_with_root(f"{CH0_PAPER_ROOT}/single_fields_smoothed"),
        },
        "datasets": (
            _ch0_manticore_cola_cic_field_datasets()
            + _ch0_manticore_swift_cola_cic_uniform_bias_datasets()
        ),
        "expected_tasks": 430,
    },
    "CH0_single_fixed_bias": {
        "description": (
            "CH0 CIC COLA and SWIFT one-field runs with fixed "
            "double-power-law bias."),
        "config_path": "configs/config_CH0.toml",
        "tag": "single_fixed_bias",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            "inference/save_log_likelihood_per_galaxy": True,
            **_with_root(f"{CH0_PAPER_ROOT}/single_fields_fixed_bias"),
        },
        "datasets": (
            _ch0_manticore_cola_cic_fixed_bias_datasets()
            + _ch0_manticore_swift_fixed_bias_datasets()
        ),
        "expected_tasks": 110,
    },
    "CH0_leaveoneout": {
        "description": (
            "CH0 SN-magnitude leave-one-out runs for ManticoreLocalSWIFT "
            "field 21."),
        "config_path": "configs/config_CH0.toml",
        "tag": "leaveoneout",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            "inference/save_log_likelihood_per_galaxy": True,
            **_with_root(f"{CH0_PAPER_ROOT}/leaveoneout"),
        },
        "datasets": _ch0_leaveoneout_datasets(),
        "expected_tasks": 35,
    },
    "CH0_angular_scatter": {
        "description": (
            "CH0 SN-magnitude angular-position scatter runs for CIC "
            "ManticoreLocalCOLA fields 0-29."),
        "config_path": "configs/config_CH0.toml",
        "tag": "angular_scatter",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            "inference/save_log_likelihood_per_galaxy": True,
            **_with_root(f"{CH0_PAPER_ROOT}/angular_scatter"),
        },
        "datasets": _ch0_angular_scatter_datasets(),
        "expected_tasks": 120,
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
            "model/selection_integral_supersample_radius": (
                TRGBH0_SELECTION_SUPERSAMPLE_RADIUS),
            "model/selection_integral_supersample_target_dx": (
                TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX),
            "model/priors/mag_lim_TRGB": _trgbh0_edd_mag_lim_prior(),
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
            "inference/save_log_likelihood_per_galaxy": True,
            "model/priors/H0/low": 40,
            "model/priors/H0/high": 100,
            "model/selection_integral_supersample_radius": (
                TRGBH0_SELECTION_SUPERSAMPLE_RADIUS),
            "model/selection_integral_supersample_target_dx": (
                TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX),
            "model/priors/mag_lim_TRGB_width/low": 0.15,
            **_with_root(f"{TRGBH0_ROOT}/manticore_fields_const_sigv"),
        },
        "datasets": _trgbh0_manticore_field_datasets(),
        "expected_tasks": 50,
    },
    "TRGBH0_single": {
        "description": (
            "TRGB H0 COLA one-field runs for CIC, PCS, and SPH MAS."),
        "config_path": "configs/config_EDD_TRGB.toml",
        "tag": "single",
        "common": {
            **TRGBH0_COMMON,
            "inference/num_warmup": 1000,
            "inference/num_samples": 1000,
            "inference/save_log_likelihood_per_galaxy": True,
            "model/priors/H0/low": 40,
            "model/priors/H0/high": 100,
            "model/selection_integral_supersample_radius": (
                TRGBH0_SELECTION_SUPERSAMPLE_RADIUS),
            "model/selection_integral_supersample_target_dx": (
                TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX),
            "model/priors/mag_lim_TRGB_width/low": 0.15,
            **_with_root(f"{TRGBH0_ROOT}/single_fields"),
        },
        "datasets": _trgbh0_manticore_cola_single_datasets(),
        "expected_tasks": 240,
    },
    "TRGBH0_single_smoothed": {
        "description": (
            "TRGB H0 PCS COLA one-field runs with density-field smoothing."),
        "config_path": "configs/config_EDD_TRGB.toml",
        "tag": "single_smoothed",
        "common": {
            **TRGBH0_COMMON,
            "inference/num_warmup": 1000,
            "inference/num_samples": 1000,
            "inference/save_log_likelihood_per_galaxy": True,
            "model/priors/H0/low": 40,
            "model/priors/H0/high": 100,
            "model/selection_integral_supersample_radius": (
                TRGBH0_SELECTION_SUPERSAMPLE_RADIUS),
            "model/selection_integral_supersample_target_dx": (
                TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX),
            "model/priors/mag_lim_TRGB_width/low": 0.15,
            "model/field_3d_smoothing_scale": [4.0, 8.0],
            "model/velocity_3d_smoothing_scale": 0.0,
            **_with_root(f"{TRGBH0_ROOT}/single_fields_smoothed"),
        },
        "datasets": _trgbh0_manticore_cola_pcs_field_datasets(),
        "expected_tasks": 160,
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
