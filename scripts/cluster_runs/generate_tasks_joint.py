# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Prepare a director structure for new runs, including copying and overwriting
the default configuration file.
"""

from argparse import ArgumentParser
from copy import deepcopy
from itertools import product
from os import makedirs
from os.path import exists, join, splitext

import numpy as np
import tomli_w

from candel import fprint, load_config, replace_prior_with_delta
from candel.pvdata.data import load_clusters

# Hardcoded flags for task generation.
scaling_relations = [ "LTYT"]  # Set to None to run all
reconstructions = ["zspace"] #"zspace", "Carrick2015",
include_quad = False
include_pairs = False
include_pix = False
include_radmag_fine = False    # Radmag with finer knot spacing
include_radmag_finest = True  # Radmag with finest knot spacing
radmag_smoothness_threshold = 4000  # Flat region (km/s), no penalty within this
radmag_smoothness_scale = 200  # Gaussian scale (km/s) beyond threshold, 0 or None to disable
radmag_sample_galactic = True  # Sample direction in galactic coords (ell, b) instead of ICRS
radmag_half_sky = True  # Restrict ell to [0, 180°] to break sign degeneracy with magnitude
include_rad = False    # Radial Vext (direction free, magnitude varies with r)
include_radmag = False  # Radial magnitude Vext (direction fixed, magnitude varies with r)
# Base model flags (split from old include_base)
include_base = False  # No flow/H0 model (both Vext and zeropoint are delta)
include_dipH0 = False  # H0_dipole varies (H0 anisotropy, affects z→r conversion)
include_dipA = False   # zeropoint_dipole varies (calibration only, no z→r effect)
include_dipVext = False  # Vext dipole only
include_A = False  # Master switch for all A runs (dipA, quadA, pixA, pairs with A)
include_bias = False  # Double power law bias model tests
include_fixed_sigma = False
# Z-space mode is auto-detected by the model based on H0 or Vext priors.
n_zspace_iterations = 2  # Iterations to refine z->r mapping for H0/Vext models
output_root = "results/radtest"
num_chains = 1
chain_method = "sequential"
LTYT_joint = True
split_tasks_two_to_one = False
split_tasks_by_kind = False
overwrite_existing = True  # If False, sets skip_if_exists=True in configs


RECONSTRUCTION_KIND_MAP = {
    "Vext": "Vext",
    "Carrick2015": "precomputed_los_Carrick2015",
    "manticore": "precomputed_los_manticore",
    "zspace": "precomputed_los_2mpp_zspace_galaxies",
}

# Malmquist grid settings matched to LOS data resolution.
# These ensure the Malmquist integration grid perfectly matches the precomputed
# LOS grid at low radius, then continues with the same spacing beyond.
# Carrick2015: 251 pts from 0.001-201 Mpc, spacing 0.803996 Mpc
# manticore: 501 pts from 0.001-330 Mpc, spacing 0.659998 Mpc
MALMQUIST_GRID_SETTINGS = {
    "carrick": {
        "r_limits_malmquist": [0.1, 1401],
        "num_points_malmquist": 1401,
    },
    "manticore": {
        "r_limits_malmquist": [0.1, 1401],
        "num_points_malmquist": 1401,
    },
    "vext": {  # Same as Carrick2015 (no reconstruction)
        "r_limits_malmquist": [0.1, 1401],
        "num_points_malmquist": 1401,
    },
}

# Double power law bias model prior configurations
BIAS_PRIORS = {
    "DPLunif": {
        "alpha_low": {"dist": "uniform", "low": 0.0, "high": 10.0},
        "alpha_high": {"dist": "uniform", "low": 0.0, "high": 4.0},
        "log_rho_t": {"dist": "uniform", "low": -10.0, "high": 10.0},
    },
    "DPLnorm": {
        "alpha_low": {"dist": "truncated_normal", "low": 0.0, "mean": 1.0, "scale": 1.0},
        "alpha_high": {"dist": "truncated_normal", "low": 0.0, "mean": 0.5, "scale": 1.0},
        "log_rho_t": {"dist": "normal", "loc": 0.0, "scale": 2.0},
    },
}

PAIR_RUNS = {
    f"{output_root}/Carrick2015_YT_noMNR_dipA_dipVext_hasY.toml",
    f"{output_root}/Carrick2015_YT_noMNR_dipH0_dipVext_hasY.toml",
}

def overwrite_config(config, key, value):
    """Return a new config dict with a nested key overwritten."""
    new_config = deepcopy(config)
    keys = key.split("/")
    d = new_config
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]

    fprint(f"overwriting config['{'/'.join(keys)}'] = {value}")
    d[keys[-1]] = value
    return new_config


def overwrite_subtree(config, key_path, subtree):
    """
    Overwrite a nested subtree (dict) at a slash-separated key path.
    """
    new_config = deepcopy(config)
    keys = key_path.split("/")
    d = new_config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = subtree
    fprint(f"overwriting subtree config['{'/'.join(keys)}'] = {subtree}")
    return new_config


def get_nested(config, key_path, default=None):
    """Recursively access a nested value using a slash-separated key."""
    keys = key_path.split("/")
    current = config
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return default
        current = current[k]
    return current


def _prior_is_varying(config, prior_name):
    """Helper to check if a prior is varying (not delta)."""
    prior = get_nested(config, f"model/priors/{prior_name}", {})
    if not isinstance(prior, dict):
        return False
    dist = prior.get("dist", "delta")
    return dist != "delta"


def generate_dynamic_tag(config, scenario_label):
    """Generate a concise tag string for the output filename."""
    parts = []

    if scenario_label and scenario_label != "default":
        parts.append(scenario_label)

    # MNR flag (always include for requested naming pattern)
    use_mnr = get_nested(config, "pv_model/use_MNR", False)
    parts.append("MNR" if use_mnr else "noMNR")

    # Check H0 and zeropoint dipole priors
    h0_dip_varying = _prior_is_varying(config, "H0_dipole")
    h0_quad_varying = _prior_is_varying(config, "H0_quad")
    zp_dip_varying = _prior_is_varying(config, "zeropoint_dipole")
    zp_quad_varying = _prior_is_varying(config, "zeropoint_quad")

    # Determine dipole/quadrupole tags
    if not h0_quad_varying and not zp_quad_varying:
        # Dipole-only case
        if h0_dip_varying:
            parts.append("dipH0")
        elif zp_dip_varying:
            parts.append("dipA")

    catalogue_value = get_nested(config, "io/catalogue_name", None)
    if isinstance(catalogue_value, list):
        catalogue_names = catalogue_value
    elif isinstance(catalogue_value, str) and catalogue_value:
        catalogue_names = [catalogue_value]
    else:
        catalogue_names = ["Clusters"]

    # Vext configuration - only add non-default cases
    which_vext = get_nested(config, "pv_model/which_Vext", "constant")
    if which_vext == "per_pix":
        parts.append("pixVext")
    elif which_vext == "radial":
        parts.append("radVext")
    elif which_vext == "radial_magnitude":
        variant = get_nested(config, "pv_model/radmag_variant", "default")
        label = "radmagVext" if variant in ("", "default") else f"radmagVext-{variant}"
        parts.append(label)
    else:
        # Check for separate Vext_quad component first
        Vext_quad_prior = get_nested(config, "model/priors/Vext_quad", {})
        vext_quad_dist = Vext_quad_prior.get("dist") if isinstance(Vext_quad_prior, dict) else None
        if vext_quad_dist is not None and vext_quad_dist != "delta":
            parts.append("quadVext")  # Quadrupole implicitly includes dipole
        else:
            # Check regular Vext prior
            Vext_prior = get_nested(config, "model/priors/Vext", {})
            if isinstance(Vext_prior, dict):
                vext_dist = Vext_prior.get("dist", "")
                if vext_dist == "vector_uniform_fixed":
                    parts.append("dipVext")
                elif vext_dist == "quadrupole_uniform_fixed":
                    parts.append("quadVext")  # Main Vext is quadrupole
                # delta case is default, don't add anything

    # Per-pixel anisotropy configuration
    which_zp = get_nested(config, "pv_model/which_zeropoint", "constant")
    which_h0 = get_nested(config, "pv_model/which_H0", "constant")

    if which_h0 == "per_pix":
        parts.append("pixH0")
    elif which_zp == "per_pix":
        parts.append("pixA")

    # Quadrupole case
    if h0_quad_varying:
        parts.append("quadH0")
    elif zp_quad_varying:
        parts.append("quadA")

    # Flag if sampling the dust prior
    dust_flags = []
    for name in catalogue_names:
        dust_model = get_nested(config, f"io/{name}/dust_model", None)
        if isinstance(dust_model, str) and dust_model.lower() != "none":
            label = f"dust-{dust_model}"
            if label not in dust_flags:
                dust_flags.append(label)
    parts.extend(dust_flags)

    # if remove_noY is true then label tag with hasY:
    has_y = False
    for name in catalogue_names:
        if get_nested(config, f"io/{name}/remove_noY", False):
            has_y = True
            break
    if has_y:
        parts.append("hasY")

    # Double power law bias variant
    bias_variant = get_nested(config, "pv_model/bias_variant", None)
    if bias_variant:
        parts.append(bias_variant)

    # Fixed sigma_v variant
    sigmav_variant = get_nested(config, "pv_model/sigmav_variant", None)
    if sigmav_variant:
        parts.append(sigmav_variant)

    return "_".join(parts)


def expand_override_grid(overrides):
    """
    Convert a dictionary with lists of override values into a list of flat
    key-value combinations.
    """
    keys, values = zip(*[
        (k, v if isinstance(v, list) else [v])
        for k, v in overrides.items()
    ])
    return [dict(zip(keys, combo)) for combo in product(*values)]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "tasks_index", type=int, nargs="?", default=0,
        help="Index of the task to run (default: 0)")
    parser.add_argument(
        "--include-fixed-sigma", action="store_true",
        help="Include fixed sigma_v variation runs.")
    args = parser.parse_args()

    config_path = "scripts/cluster_runs/config_clusters.toml"
    config = load_config(
        config_path, replace_none=False, replace_los_prior=False)
    # Note: num_points_malmquist and r_limits_malmquist are set per-reconstruction
    # later in the script to match the LOS grid resolution exactly.
    config = overwrite_config(config, "io/reconstruction_main/num_steps", 1001)
    config = overwrite_config(config, "inference/num_chains", num_chains)
    config = overwrite_config(config, "inference/chain_method", chain_method)
    if not overwrite_existing:
        config = overwrite_config(config, "inference/skip_if_exists", True)

    tasks_index = args.tasks_index
    include_fixed_sigma = args.include_fixed_sigma

    task_file = f"tasks_{tasks_index}.txt"
    log_dir = f"logs_{tasks_index}"

    unknown_recon = [name for name in reconstructions if name not in RECONSTRUCTION_KIND_MAP]
    if unknown_recon:
        raise ValueError(f"Unknown reconstructions: {unknown_recon}")
    reconstruction_kinds = [RECONSTRUCTION_KIND_MAP[name] for name in reconstructions]

    base = {
        "pv_model/kind": reconstruction_kinds,
        "pv_model/which_Vext": ["constant"],
        "io/root_output": output_root,
        "model/priors/Vext": [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]}],
        "model/priors/zeropoint_dipole": [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]}],
    }

    dipole_settings = deepcopy(base)

    # Dipoles and permutations
    dipole_settings["model/priors/Vext"] = [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]}, 
            {"dist": "vector_uniform_fixed", "low": 0.0, "high": 5000.0},
        ]
    dipole_settings["model/priors/zeropoint_dipole"] = [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]},
            {"dist": "vector_uniform_fixed", "low": 0.0, "high": 0.2},
        ]

    dipole_combinations = expand_override_grid(dipole_settings)

    def classify_run(override_set):
        """Classify a run based on its Vext and zeropoint priors.

        Returns one of: 'base', 'dipVext', 'dipZP', 'dipVext_dipZP'
        - 'base': both Vext and zeropoint are delta (no flow/H0)
        - 'dipVext': Vext dipole only
        - 'dipZP': zeropoint dipole only (will be split into dipA/dipH0 later)
        - 'dipVext_dipZP': both Vext and zeropoint dipoles
        """
        vext_prior = override_set.get("model/priors/Vext", {})
        zp_prior = override_set.get("model/priors/zeropoint_dipole", {})
        vext_dist = vext_prior.get("dist", "")
        zp_dist = zp_prior.get("dist", "")

        vext_is_dipole = vext_dist == "vector_uniform_fixed"
        zp_is_dipole = zp_dist == "vector_uniform_fixed"

        if vext_is_dipole and zp_is_dipole:
            return "dipVext_dipZP"
        elif vext_is_dipole:
            return "dipVext"
        elif zp_is_dipole:
            return "dipZP"
        else:
            return "base"

    def should_include_run(override_set):
        """Check if a run should be included based on the include_* flags."""
        run_type = classify_run(override_set)
        if run_type == "base":
            return include_base
        elif run_type == "dipVext":
            return include_dipVext
        elif run_type == "dipZP":
            # dipZP runs are split into dipA and dipH0 later via stretch_variants
            # Include if either (dipA and include_A) or dipH0 is enabled
            return (include_dipA and include_A) or include_dipH0
        elif run_type == "dipVext_dipZP":
            # Both dipoles - include if pairs are enabled
            return include_pairs
        return True

    dipole_combinations = [
        combo for combo in dipole_combinations if should_include_run(combo)
    ]
    
    # Per-pixel Vext
    pixelVext_settings = deepcopy(base)
    pixelVext_settings["pv_model/which_Vext"] = ["per_pix"]

    pixelVext_combinations = expand_override_grid(pixelVext_settings)

    # Per-pixel zeropoint (calibration uncertainty, no z→r effect)
    pixelA_settings = deepcopy(base)
    pixelA_settings["pv_model/which_zeropoint"] = ["per_pix"]

    pixelA_combinations = expand_override_grid(pixelA_settings)

    # Per-pixel H0 (spatially-varying Hubble constant, affects z→r)
    pixelH0_settings = deepcopy(base)
    pixelH0_settings["pv_model/which_H0"] = ["per_pix"]

    pixelH0_combinations = expand_override_grid(pixelH0_settings)

    # Dipole and quadrupole Vext
    quadVext_settings = deepcopy(base)

    quadVext_settings["model/priors/Vext"] = [
        {"dist": "vector_uniform_fixed", "low": 0.0, "high": 5000.0},
    ]
    quadVext_settings["model/priors/Vext_quad"] = [
        {"dist": "quadrupole_uniform_fixed", "low": 0.0, "high": 5000.0},
    ]

    quadVext_combinations = expand_override_grid(quadVext_settings)

    # Dipole and quadrupole zeropoint
    quad_zeropoint_settings = deepcopy(base)
    quad_zeropoint_settings["model/priors/zeropoint_dipole"] = [
        {"dist": "vector_uniform_fixed", "low": 0.0, "high": 0.3},
    ]
    quad_zeropoint_settings["model/priors/zeropoint_quad"] = [
        {"dist": "quadrupole_uniform_fixed", "low": 0.0, "high": 0.3},
    ]

    quad_zeropoint_combinations = expand_override_grid(quad_zeropoint_settings)
    
    # Radial Vext
    radialVext_settings = deepcopy(base)
    radialVext_settings["pv_model/which_Vext"] = ["radial"]

    radialVext_combinations = expand_override_grid(radialVext_settings)

    # Radial magnitude-only Vext (direction fixed, magnitude varies)
    radmag_prior_template = get_nested(
        config, "model/priors/Vext_radmag", {})

    def radmag_prior_with_knots(knots):
        prior = deepcopy(radmag_prior_template)
        prior["rknot"] = knots
        # Remove max_modulus if it doesn't match the new knot count.
        # h0_dipole_percent is safe to keep since it scales with rknot.
        if "max_modulus" in prior and len(prior["max_modulus"]) != len(knots):
            del prior["max_modulus"]
        # Override smoothness prior from the global flags
        if radmag_smoothness_scale:
            prior["smoothness_scale"] = radmag_smoothness_scale
            prior["smoothness_threshold"] = radmag_smoothness_threshold
        else:
            # Disable smoothness prior
            if "smoothness_scale" in prior:
                del prior["smoothness_scale"]
            if "smoothness_threshold" in prior:
                del prior["smoothness_threshold"]
        # Override galactic sampling options from the global flags
        prior["sample_galactic"] = radmag_sample_galactic
        prior["half_sky"] = radmag_half_sky
        return prior

    def build_radmag_combinations(knots, variant_label):
        settings = deepcopy(base)
        settings["pv_model/which_Vext"] = ["radial_magnitude"]
        settings["model/priors/Vext_radmag"] = [
            radmag_prior_with_knots(knots)]
        if variant_label not in ("", "default"):
            settings["pv_model/radmag_variant"] = [variant_label]
        return expand_override_grid(settings)

    radialMagVext_combinations = build_radmag_combinations(
        [0, 250, 500, 750, 1000], "default"
    )
    if not include_radmag:
        radialMagVext_combinations = []
    radmag_fine_combinations = []
    radmag_finest_combinations = []
    if include_radmag_fine:
        radmag_fine_combinations = build_radmag_combinations(
            [0, 125, 250, 500, 750, 1000], "fine"
        )
    if include_radmag_finest:
        radmag_finest_combinations = build_radmag_combinations(
            [0, 62.5, 125, 187.5, 250, 500, 750, 1000], "finest"
        )

    # Double power law bias model combinations (manticore + LTYT + dipH0 only)
    # Now uses explicit H0_dipole prior instead of stretch_los_with_zeropoint
    bias_combinations = []
    if include_bias:
        for bias_label, bias_priors in BIAS_PRIORS.items():
            bias_settings = {
                "pv_model/kind": ["precomputed_los_manticore"],
                "pv_model/which_Vext": ["constant"],
                "pv_model/bias_variant": [bias_label],
                "io/root_output": output_root,
                "model/priors/Vext": [{"dist": "delta", "value": [0.0, 0.0, 0.0]}],
                # Use new H0_dipole prior (fractional δH) for dipH0 runs
                "model/priors/H0_dipole": [
                    {"dist": "vector_uniform_fixed", "low": 0.0, "high": 0.15}],
                "model/priors/zeropoint_dipole": [
                    {"dist": "delta", "value": [0.0, 0.0, 0.0]}],
                "model/priors/alpha_low": [bias_priors["alpha_low"]],
                "model/priors/alpha_high": [bias_priors["alpha_high"]],
                "model/priors/log_rho_t": [bias_priors["log_rho_t"]],
            }
            bias_combinations.extend(expand_override_grid(bias_settings))

    # Fixed sigma_v runs (Vext + LT + dipVext/dipH0 + sigma_v=100)
    fixed_sigmav_settings = {
        "pv_model/kind": ["Vext"],
        "pv_model/which_Vext": ["constant"],
        "pv_model/sigmav_variant": ["sigv100"],
        "io/root_output": output_root,
        "model/priors/Vext": [
            {"dist": "vector_uniform_fixed", "low": 0.0, "high": 5000.0}],
        "model/priors/zeropoint_dipole": [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]}],
        "model/priors/H0_dipole": [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]}],
        "model/priors/sigma_v": [{"dist": "delta", "value": 100.0}],
    }
    # Fixed sigma_v + dipH0 runs using new H0_dipole prior
    fixed_sigmav_diph0_settings = {
        "pv_model/kind": ["Vext"],
        "pv_model/which_Vext": ["constant"],
        "pv_model/sigmav_variant": ["sigv100"],
        "io/root_output": output_root,
        "model/priors/Vext": [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]}],
        "model/priors/zeropoint_dipole": [
            {"dist": "delta", "value": [0.0, 0.0, 0.0]}],
        # Use new H0_dipole prior (fractional δH) for dipH0 runs
        "model/priors/H0_dipole": [
            {"dist": "vector_uniform_fixed", "low": 0.0, "high": 0.15}],
        "model/priors/sigma_v": [{"dist": "delta", "value": 100.0}],
    }
    fixed_sigmav_combinations = expand_override_grid(
        fixed_sigmav_settings) if include_fixed_sigma else []
    fixed_sigmav_diph0_combinations = expand_override_grid(
        fixed_sigmav_diph0_settings) if include_fixed_sigma else []

    override_groups = [
        ("all_other_runs", dipole_combinations
         + fixed_sigmav_combinations + fixed_sigmav_diph0_combinations),
        ("pix", (pixelA_combinations if include_A else []) + pixelH0_combinations + pixelVext_combinations if include_pix else []),
        ("quad", quadVext_combinations + quad_zeropoint_combinations if include_quad else []),
        ("radmag_fine", radmag_fine_combinations),
        ("radmag_finest", radmag_finest_combinations),
        ("rad", radialVext_combinations if include_rad else []),
        ("radmag", radialMagVext_combinations),
        ("bias", bias_combinations if include_bias else []),
    ]

    base_clusters_section = deepcopy(config["io"].get("Clusters", {}))
    if not base_clusters_section:
        raise ValueError("Base config must define [io.Clusters] for template settings.")

    shared_logT_mean = None
    if LTYT_joint:
        base_clusters_full = deepcopy(base_clusters_section)
        root = base_clusters_full.pop("root", None)
        if root is None:
            raise ValueError("Base Clusters config must include `root`.")
        base_clusters_full["finite_logY"] = False
        base_clusters_full["remove_noY"] = False
        base_clusters_full["only_missing_Y"] = False
        full_data = load_clusters(root, subtract_logT_mean=False, **base_clusters_full)
        shared_logT_mean = float(np.mean(full_data["logT"]))
        fprint(f"using shared logT_mean={shared_logT_mean:.6g} for joint LTYT runs.")

    def build_cluster_section(**updates):
        section = deepcopy(base_clusters_section)
        section.setdefault("remove_noY", False)
        section.setdefault("only_missing_Y", False)
        section.update(updates)
        return section

    if LTYT_joint:
        ltyt_overrides = {
            "inference/model": ["ClustersModel", "ClustersModel"],
            "io/catalogue_name": ["Clusters_hasY", "Clusters_LTtail"],
            "io/Clusters_hasY": build_cluster_section(
                which_relation="LTYT",
                finite_logY=True,
                remove_noY=True,
                only_missing_Y=False,
                logT_mean=shared_logT_mean,
            ),
            "io/Clusters_LTtail": build_cluster_section(
                which_relation="LT",
                finite_logY=False,
                remove_noY=False,
                only_missing_Y=True,
                logT_mean=shared_logT_mean,
            ),
        }
        ltyt_shared_params = [
            "A_LT",
            "B_LT",
            "A_YT",
            "B_YT",
            "sigma_LT",
            "sigma_YT",
            "Vext",
            "Vext_quad",
            "zeropoint_dipole",
            "zeropoint_quad",
            "zeropoint_pix",
            "H0_dipole",
            "H0_quad",
            "H0_pix",
            "sigma_v",
            "beta",
            "b1",
            "R_dist_emp",
            "p_dist_emp",
            "n_dist_emp",
        ]
        ltyt_scenario = {
            "label": "LTYT",
            "overrides": ltyt_overrides,
            "shared_params_base": ltyt_shared_params,
            "share_flow": True,
        }
    else:
        ltyt_scenario = {
            "label": "LTYT",
            "overrides": {
                "inference/model": "ClustersModel",
                "inference/shared_params": "none",
                "io/catalogue_name": "Clusters",
                "io/Clusters": build_cluster_section(
                    which_relation="LTYT",
                    finite_logY=True,
                    remove_noY=True,
                    only_missing_Y=False,
                ),
            },
        }

    scenarios = [
        ltyt_scenario,
        {
            "label": "LT",
            "overrides": {
                "inference/model": "ClustersModel",
                "inference/shared_params": "none",
                "io/catalogue_name": "Clusters",
                "io/Clusters": build_cluster_section(
                    which_relation="LT",
                    finite_logY=False,
                    remove_noY=False,
                    only_missing_Y=False,
                ),
            },
        },
        {
            "label": "YT",
            "overrides": {
                "inference/model": "ClustersModel",
                "inference/shared_params": "none",
                "io/catalogue_name": "Clusters",
                "io/Clusters": build_cluster_section(
                    which_relation="YT",
                    finite_logY=True,
                    remove_noY=True,
                    only_missing_Y=False,
                ),
            },
        },
    ]

    if scaling_relations:
        scenarios = [sc for sc in scenarios if sc.get("label") in scaling_relations]

    def flow_shared_params(which_vext):
        mapping = {
            "constant": ["Vext", "Vext_quad"],
            "radial": ["Vext_radial"],
            "radial_magnitude": ["Vext_radmag"],
            "per_pix": ["Vext_pix"],
        }
        return mapping.get(which_vext, [])

    total_overrides = sum(len(group[1]) for group in override_groups)
    print(f"Total override combinations per scenario: {total_overrides}")

    task_counter = 0
    tasks_by_group = {
        "all_other_runs": [],
        "pix": [],
        "quad": [],
        "pairs": [],
        "radmag_fine": [],
        "radmag_finest": [],
        "rad": [],
        "radmag": [],
        "bias": [],
    }
    for scenario in scenarios:
        scenario_overrides = scenario["overrides"]
        scenario_label = scenario["label"]
        for group_label, override_sets in override_groups:
            if not override_sets:
                continue
            # Bias runs are only for LTYT scenario
            if group_label == "bias" and scenario_label != "LTYT":
                continue
            for override_set in override_sets:
                # Fixed sigma_v runs are only for LT scenario
                if override_set.get("pv_model/sigmav_variant") and scenario_label != "LT":
                    continue

                local_config = deepcopy(config)

                for key, value in scenario_overrides.items():
                    if isinstance(value, dict):
                        local_config = overwrite_subtree(local_config, key, value)
                    else:
                        local_config = overwrite_config(local_config, key, value)

                for key, value in override_set.items():
                    if isinstance(value, dict):
                        local_config = overwrite_subtree(local_config, key, value)
                    else:
                        local_config = overwrite_config(local_config, key, value)

                shared_base = scenario.get("shared_params_base", None)
                if shared_base and scenario_label == "LTYT" and scenario.get("share_flow", False):
                    which_vext = get_nested(
                        local_config, "pv_model/which_Vext", "constant")
                    if which_vext == "per_pix":
                        nside = get_nested(
                            local_config, "pv_model/Vext_per_pix_nside", 1)
                        npix = 12 * nside**2
                        local_config = overwrite_subtree(
                            local_config,
                            "model/priors/Vext_pix",
                            {"dist": "array_uniform", "low": -10000.0, "high": 10000.0, "nval": npix},
                        )
                    which_zp = get_nested(
                        local_config, "pv_model/which_zeropoint", "constant")
                    if which_zp == "per_pix":
                        nside = get_nested(
                            local_config, "pv_model/anisotropy_per_pix_nside", 1)
                        npix = 12 * nside**2
                        local_config = overwrite_subtree(
                            local_config,
                            "model/priors/zeropoint_pix",
                            {"dist": "array_uniform", "low": -0.1, "high": 0.1, "nval": npix},
                        )

                if shared_base:
                    shared_list = list(shared_base)
                    if scenario.get("share_flow", False):
                        which_vext = get_nested(
                            local_config, "pv_model/which_Vext", "constant")
                        shared_list.extend(flow_shared_params(which_vext))
                        which_zp = get_nested(
                            local_config, "pv_model/which_zeropoint", "constant")
                        if which_zp == "per_pix":
                            shared_list.append("zeropoint_pix")
                    shared_list = list(dict.fromkeys(shared_list))
                    shared_str = ",".join(shared_list)
                    local_config = overwrite_config(
                        local_config, "inference/shared_params", shared_str)

                if scenario_label == "Joint":
                    dipole_prior = get_nested(
                        local_config, "model/priors/zeropoint_dipole", {})
                    if not (
                        isinstance(dipole_prior, dict)
                        and dipole_prior.get("dist") == "vector_uniform_fixed"
                    ):
                        continue

                # Check for varying H0_dipole or zeropoint_dipole
                h0_dipole_prior = get_nested(
                    local_config, "model/priors/H0_dipole", {})
                zp_dipole_prior = get_nested(
                    local_config, "model/priors/zeropoint_dipole", {})

                h0_dip_varying = (
                    isinstance(h0_dipole_prior, dict)
                    and h0_dipole_prior.get("dist") not in (None, "delta"))
                zp_dip_varying = (
                    isinstance(zp_dipole_prior, dict)
                    and zp_dipole_prior.get("dist") == "vector_uniform_fixed")

                # For bias runs, H0_dipole is already configured, don't iterate
                bias_variant = get_nested(local_config, "pv_model/bias_variant", None)
                if bias_variant:
                    # Bias runs already have H0_dipole set, just run once
                    h0_variants = [None]
                elif h0_dip_varying:
                    # H0_dipole already varying, just run once
                    h0_variants = [None]
                elif zp_dip_varying:
                    # zeropoint_dipole varying: generate dipA and/or dipH0 runs
                    h0_variants = []
                    if include_dipA and include_A:
                        h0_variants.append("dipA")
                    if include_dipH0:
                        h0_variants.append("dipH0")
                    if not h0_variants:
                        # Neither dipA nor dipH0 enabled, skip this run
                        continue
                else:
                    h0_variants = [None]

                for h0_variant in h0_variants:
                    run_config = deepcopy(local_config)

                    # Configure priors based on variant
                    if h0_variant == "dipA":
                        # dipA run: zeropoint varies, H0 fixed
                        run_config = overwrite_subtree(
                            run_config,
                            "model/priors/H0_dipole",
                            {"dist": "delta", "value": [0.0, 0.0, 0.0]},
                        )
                    elif h0_variant == "dipH0":
                        # dipH0 run: H0 varies, zeropoint fixed
                        zp_high = zp_dipole_prior.get("high", 0.2)
                        h0_high = min(0.15, zp_high)  # Fractional δH
                        run_config = overwrite_subtree(
                            run_config,
                            "model/priors/H0_dipole",
                            {"dist": "vector_uniform_fixed", "low": 0.0, "high": h0_high},
                        )
                        run_config = overwrite_subtree(
                            run_config,
                            "model/priors/zeropoint_dipole",
                            {"dist": "delta", "value": [0.0, 0.0, 0.0]},
                        )

                    # Determine run type flags
                    is_H0_run = (h0_variant == "dipH0") or h0_dip_varying
                    is_A_run = (h0_variant == "dipA")
                    is_vext_or_base_run = (h0_variant is None and not h0_dip_varying)

                    # Check if Vext model is active
                    vext_prior = override_set.get("model/priors/Vext", {})
                    has_vext_dipole = (
                        isinstance(vext_prior, dict)
                        and vext_prior.get("dist") == "vector_uniform_fixed"
                    )
                    which_vext = override_set.get("pv_model/which_Vext", "constant")
                    has_radial_vext = which_vext in ("radial", "radial_magnitude")
                    has_vext_model = has_vext_dipole or has_radial_vext

                    # Set n_zspace_iterations for models that need it
                    if is_H0_run or has_vext_model:
                        run_config = overwrite_config(
                            run_config, "pv_model/n_zspace_iterations",
                            n_zspace_iterations)

                    if not include_pairs:
                        vext_prior = get_nested(
                            run_config, "model/priors/Vext", {})
                        zp_dipole_prior = get_nested(
                            run_config, "model/priors/zeropoint_dipole", {})
                        if (
                            isinstance(vext_prior, dict)
                            and vext_prior.get("dist") == "vector_uniform_fixed"
                            and isinstance(zp_dipole_prior, dict)
                            and zp_dipole_prior.get("dist") == "vector_uniform_fixed"
                        ):
                            continue

                    fdir_out = join(
                        run_config["root_main"], run_config["io"]["root_output"])
                    if not exists(fdir_out):
                        fprint(f"creating output directory `{fdir_out}`")
                        makedirs(fdir_out, exist_ok=True)

                    kind = get_nested(run_config, "pv_model/kind", "unknown")
                    kind_lower = str(kind).lower()

                    run_config = overwrite_config(
                        run_config, "inference/num_warmup", 500)
                    run_config = overwrite_config(
                        run_config, "inference/num_samples", 500)

                    which_vext = get_nested(
                        run_config, "pv_model/which_Vext", "constant")
                    if which_vext in ("radial", "radial_magnitude"):
                        run_config = overwrite_config(
                            run_config, "inference/num_warmup", 1000)
                        run_config = overwrite_config(
                            run_config, "inference/num_samples", 3000)

                    if kind.startswith("precomputed_los_"):
                        if "manticore" in kind_lower:
                            beta_prior = {"dist": "normal", "loc": 1.0, "scale": 0.05}
                            run_config = overwrite_subtree(
                                run_config, "model/priors/beta", beta_prior)
                            fprint("set beta prior to Normal(1.0, 0.02) for manticore reconstruction")

                            run_config = overwrite_config(
                                run_config, "pv_model/galaxy_bias", "powerlaw")
                            fprint("set galaxy_bias to 'powerlaw' for manticore reconstruction")

                            bias_variant = get_nested(
                                run_config, "pv_model/bias_variant", None)
                            if bias_variant in BIAS_PRIORS:
                                run_config = overwrite_config(
                                    run_config, "pv_model/galaxy_bias",
                                    "double_powerlaw")
                                fprint(
                                    "set galaxy_bias to 'double_powerlaw' "
                                    "for DPL bias runs")

                            # Set Malmquist grid to match manticore LOS resolution
                            grid_settings = MALMQUIST_GRID_SETTINGS["manticore"]
                            run_config = overwrite_config(
                                run_config, "pv_model/r_limits_malmquist",
                                grid_settings["r_limits_malmquist"])
                            run_config = overwrite_config(
                                run_config, "pv_model/num_points_malmquist",
                                grid_settings["num_points_malmquist"])
                            fprint(f"set Malmquist grid to {grid_settings['num_points_malmquist']} points for manticore")

                        elif "carrick" in kind_lower:
                            beta_prior = {"dist": "normal", "loc": 0.43, "scale": 0.02}
                            run_config = overwrite_subtree(
                                run_config, "model/priors/beta", beta_prior)
                            fprint("set beta prior to Normal(0.43, 0.02) for Carrick2015 reconstruction")

                            run_config = overwrite_config(
                                run_config, "pv_model/galaxy_bias", "linear")
                            fprint("set galaxy_bias to 'linear' for Carrick2015 reconstruction")

                            # Set Malmquist grid to match Carrick2015 LOS resolution
                            grid_settings = MALMQUIST_GRID_SETTINGS["carrick"]
                            run_config = overwrite_config(
                                run_config, "pv_model/r_limits_malmquist",
                                grid_settings["r_limits_malmquist"])
                            run_config = overwrite_config(
                                run_config, "pv_model/num_points_malmquist",
                                grid_settings["num_points_malmquist"])
                            fprint(f"set Malmquist grid to {grid_settings['num_points_malmquist']} points for Carrick2015")

                    elif kind_lower == "vext":
                        # Set Malmquist grid for Vext (no reconstruction, same as Carrick2015)
                        grid_settings = MALMQUIST_GRID_SETTINGS["vext"]
                        run_config = overwrite_config(
                            run_config, "pv_model/r_limits_malmquist",
                            grid_settings["r_limits_malmquist"])
                        run_config = overwrite_config(
                            run_config, "pv_model/num_points_malmquist",
                            grid_settings["num_points_malmquist"])
                        fprint(f"set Malmquist grid to {grid_settings['num_points_malmquist']} points for Vext")

                    dynamic_tag = generate_dynamic_tag(run_config, scenario_label)

                    kind_for_filename = kind.replace("precomputed_los_", "")

                    fname_out = join(
                        run_config["io"]["root_output"],
                        f"{kind_for_filename}_{dynamic_tag}.hdf5"
                    )
                    run_config = overwrite_config(
                        run_config, "io/fname_output", fname_out)

                    toml_out = join(
                        run_config["root_main"],
                        splitext(fname_out)[0] + ".toml"
                    )
                    rel_toml = splitext(fname_out)[0] + ".toml"
                    if not include_pairs and rel_toml in PAIR_RUNS:
                        continue
                    fprint(f"writing the configuration file to `{toml_out}`")
                    with open(toml_out, "wb") as f:
                        tomli_w.dump(run_config, f)

                    entry = (toml_out, kind_lower)
                    if rel_toml in PAIR_RUNS:
                        tasks_by_group["pairs"].append(entry)
                    else:
                        tasks_by_group[group_label].append(entry)

    ordered_groups = [
        "all_other_runs",
        "radmag",
        "radmag_fine",
        "radmag_finest",
        "quad",
        "rad",
        "pix",
        "pairs",
        "bias",
    ]

    if split_tasks_two_to_one:
        task_files = {
            "tasks_0": "tasks_0.txt",
            "tasks_1": "tasks_1.txt",
        }
        task_handles = {
            key: open(path, "w") for key, path in task_files.items()
        }
        try:
            for group_name in ordered_groups:
                for toml_out, _ in tasks_by_group[group_name]:
                    target = "tasks_1" if task_counter % 3 == 2 else "tasks_0"
                    task_handles[target].write(f"{task_counter} {toml_out}\n")
                    task_counter += 1
        finally:
            for fh in task_handles.values():
                fh.close()
        fprint(
            "wrote task lists to "
            f"`{task_files['tasks_0']}` and `{task_files['tasks_1']}`"
        )
    elif split_tasks_by_kind:
        task_files = {
            "manticore": "tasks_0.txt",
            "other": "tasks_1.txt",
        }
        task_handles = {
            key: open(path, "w") for key, path in task_files.items()
        }
        try:
            for group_name in ordered_groups:
                for toml_out, kind_lower in tasks_by_group[group_name]:
                    bucket = "manticore" if "manticore" in kind_lower else "other"
                    task_handles[bucket].write(f"{task_counter} {toml_out}\n")
                    task_counter += 1
        finally:
            for fh in task_handles.values():
                fh.close()
        fprint(
            "wrote task lists to "
            f"`{task_files['manticore']}` and `{task_files['other']}`"
        )
    else:
        with open(task_file, "w") as task_fh:
            for group_name in ordered_groups:
                for toml_out, _ in tasks_by_group[group_name]:
                    task_fh.write(f"{task_counter} {toml_out}\n")
                    task_counter += 1
        fprint(f"wrote task list to `{task_file}`")
