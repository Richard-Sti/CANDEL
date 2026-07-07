#!/usr/bin/env python
"""Make EDD TRGB Gaussian PPC plots."""
import argparse
import copy
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

import matplotlib  # noqa: E402
import tomli_w  # noqa: E402
from trgbh0_plot_style import (OUTPUT_DIR, ROOT,  # noqa: E402
                               TRGBH0_TABLE_RESULTS)

import candel  # noqa: E402
from candel.mock import (generate_trgb_ppc, plot_trgb_ppc,  # noqa: E402
                         plot_trgb_ppc_distance, plot_trgb_ppc_sky,
                         plot_trgb_ppc_sky_exposure)

matplotlib.use("Agg")


CONFIG = ROOT / "scripts/runs/configs/config_EDD_TRGB.toml"
SINGLE_FIELD_RESULTS = TRGBH0_TABLE_RESULTS.parent / "single_fields"
DEFAULT_MANTICORE_ROOT = (
    ROOT / "data/MANTICORE/2MPP_MULTIBIN_N256_DES_V2/"
           "sph_fields_new_feb/sph_fields"
)
CARRICK_POSTERIOR = (
    TRGBH0_TABLE_RESULTS
    / "EDD_TRGB_sel-TRGB_magnitude_Carrick2015_main.hdf5"
)
DEFAULT_POSTERIORS = {
    "carrick": CARRICK_POSTERIOR,
    "none": CARRICK_POSTERIOR,
    "manticore": (
        TRGBH0_TABLE_RESULTS
        / "EDD_TRGB_rhoSmoothR4_MAS-PCS_sel-TRGB_magnitude_"
          "ManticoreLocalCOLA_main.hdf5"
    ),
}
DEFAULT_OUTPUTS = {
    "carrick": OUTPUT_DIR / "trgbh0_edd_trgb_carrick_gaussian_ppc.pdf",
    "none": OUTPUT_DIR / "trgbh0_edd_trgb_nofield_gaussian_ppc.pdf",
    "manticore": OUTPUT_DIR / "trgbh0_edd_trgb_manticore_gaussian_ppc.pdf",
}

SEED = 42
DEFAULT_FIELD_INDEX = 0
RULE_WIDTH = 72


def print_rule(char="="):
    """Print a terminal separator."""
    print(char * RULE_WIDTH, flush=True)


def print_section(title):
    """Print a visually separated terminal section title."""
    print()
    print_rule()
    print(title, flush=True)
    print_rule()


def _set_nested(config, path, value):
    """Set a slash-delimited config key in-place."""
    node = config
    parts = path.split("/")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _manticore_single_field_posterior(field_index):
    return (
        SINGLE_FIELD_RESULTS
        / "EDD_TRGB_rhoSmoothR4_MAS-PCS_sel-TRGB_magnitude_"
          f"ManticoreLocalCOLA_field{field_index:02d}_single.hdf5"
    )


def default_posterior(mode, field_index):
    """Return the default posterior path for a PPC mode."""
    if mode == "manticore" and field_index is not None:
        single = _manticore_single_field_posterior(field_index)
        if single.exists():
            return single
    return DEFAULT_POSTERIORS[mode]


def default_output(mode, field_index):
    """Return the default output path for a PPC mode."""
    if mode == "manticore" and field_index is not None:
        return (
            OUTPUT_DIR
            / f"trgbh0_edd_trgb_manticore_field{field_index:02d}_"
            "gaussian_ppc.pdf"
        )
    return DEFAULT_OUTPUTS[mode]


def configure_mode(config, mode, manticore_root=None):
    """Mutate a loaded EDD TRGB config for the requested PPC mode."""
    if mode == "none":
        _set_nested(config, "model/use_reconstruction", False)
        _set_nested(config, "model/which_bias", "uniform")
    elif mode == "carrick":
        _set_nested(config, "model/use_reconstruction", True)
        _set_nested(config, "model/which_bias", "linear")
        _set_nested(config, "io/PV_main/EDD_TRGB/reconstruction",
                    "Carrick2015")
    elif mode == "manticore":
        _set_nested(config, "model/use_reconstruction", True)
        _set_nested(config, "model/which_bias", "double_powerlaw")
        _set_nested(config, "model/field_3d_smoothing_scale", 4.0)
        _set_nested(config, "model/velocity_3d_smoothing_scale", 0.0)
        _set_nested(config, "model/priors/beta",
                    {"dist": "delta", "value": 1.0})
        _set_nested(config, "io/PV_main/EDD_TRGB/reconstruction",
                    "ManticoreLocalCOLA")
        _set_nested(config,
                    "io/reconstruction_main/ManticoreLocalCOLA/which_MAS",
                    "PCS")
        field_config = config["io"]["reconstruction_main"][
            "ManticoreLocalCOLA"]
        if manticore_root is not None:
            field_config["fpath_root"] = str(manticore_root)
        elif "fpath_root" not in field_config:
            raise ValueError(
                "Manticore mode requires "
                "`io.reconstruction_main.ManticoreLocalCOLA.fpath_root` "
                "or --manticore-root.")
    else:
        raise ValueError(f"Unknown PPC mode: {mode}")


def load_observed_data(config):
    """Load only catalogue observables, without reconstruction products."""
    data_config = copy.deepcopy(config)
    _set_nested(data_config, "model/use_reconstruction", False)
    _set_nested(data_config, "io/load_host_los", False)
    _set_nested(data_config, "io/load_rand_los", False)

    with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".toml", delete=False) as handle:
        tmp_path = Path(handle.name)
        tomli_w.dump(data_config, handle)
    try:
        return candel.pvdata.load_EDD_TRGB_from_config(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("carrick", "none", "manticore"),
        default="carrick",
        help="PPC reconstruction mode.")
    parser.add_argument(
        "--field-index", type=int, default=None,
        help=("Manticore realisation to draw from. Carrick defaults to field "
              "0; no-field ignores this."))
    parser.add_argument(
        "--posterior", type=Path, default=None,
        help=("Posterior HDF5 file. Defaults to the mode-matched paper "
              "result where available, otherwise the Carrick fiducial."))
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output figure path.")
    parser.add_argument(
        "--n-ppc", type=int, default=None,
        help="Number of PPC galaxies. Defaults to config ppc_factor.")
    parser.add_argument(
        "--ppc-factor", type=int, default=None,
        help="Override config model.ppc_factor when --n-ppc is omitted.")
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Parallel PPC worker processes. Defaults to runtime CPU env.")
    parser.add_argument(
        "--manticore-root", type=Path, default=None,
        help="Root containing ManticoreLocalCOLA MAS subdirectories.")
    parser.add_argument(
        "--b-min", type=float, default=None,
        help=("Apply the same lower |Galactic latitude| cut, in degrees, "
              "to the observed EDD TRGB hosts and PPC sky positions."))
    parser.add_argument(
        "--vmono", action="store_true",
        help="Include the constant Vext monopole term in the PPC.")
    parser.add_argument(
        "--voct", action="store_true",
        help="Include the Vext octupole term in the PPC.")
    parser.add_argument(
        "--sky-mask-nside", type=int, default=None,
        help=("Override the PPC HEALPix sky-mask nside. Defaults to "
              "model.TRGB_sky_exposure; use 0 to disable."))
    parser.add_argument(
        "--sky-mask-kappa", type=float, default=None,
        help=("Override the Dirichlet prior concentration for the angular "
              "exposure. Defaults to model.TRGB_sky_exposure.kappa."))
    parser.add_argument(
        "--sky-mask-posterior-mean", action="store_true",
        help=("Use the posterior-mean angular exposure instead of drawing "
              "one Dirichlet realization."))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the PPC tqdm progress bar.")
    return parser.parse_args()


def main():
    args = parse_args()
    field_index = args.field_index
    if args.mode == "carrick" and field_index is None:
        field_index = DEFAULT_FIELD_INDEX
    if args.mode == "none":
        field_index = None

    config = candel.load_config(str(CONFIG), replace_los_prior=False)
    manticore_root = args.manticore_root
    if (manticore_root is None and args.mode == "manticore"
            and DEFAULT_MANTICORE_ROOT.exists()):
        manticore_root = DEFAULT_MANTICORE_ROOT
    try:
        configure_mode(config, args.mode, manticore_root=manticore_root)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.ppc_factor is not None:
        _set_nested(config, "model/ppc_factor", args.ppc_factor)
    if args.b_min is not None:
        _set_nested(config, "io/PV_main/EDD_TRGB/b_min", args.b_min)
    if args.vmono:
        _set_nested(config, "model/which_Vext_monopole", "constant")
    if args.voct:
        _set_nested(config, "model/use_Vext_octupole", True)

    data = load_observed_data(config)
    posterior = args.posterior or default_posterior(args.mode, field_index)
    output = args.output or default_output(args.mode, field_index)
    print_section("EDD TRGB PPC")
    print(f"mode         : {args.mode}", flush=True)
    print(f"field_index  : {field_index}", flush=True)
    print(f"b_min        : {args.b_min}", flush=True)
    print(f"posterior    : {posterior}", flush=True)
    print(f"output       : {output}", flush=True)
    print_rule("-")
    print(f"Loading posterior samples from: {posterior}", flush=True)
    samples = candel.read_samples("", str(posterior))
    if "nu_cz" in samples:
        _set_nested(config, "model/cz_likelihood", "student_t")
        print("cz_likelihood: student_t (detected nu_cz samples)",
              flush=True)

    ppc = generate_trgb_ppc(
        samples,
        data,
        config,
        n_ppc=args.n_ppc,
        seed=args.seed,
        field_index=field_index,
        progress=not args.no_progress,
        sky_exposure_nside=args.sky_mask_nside,
        sky_exposure_kappa=args.sky_mask_kappa,
        sky_exposure_posterior_mean=args.sky_mask_posterior_mean,
        n_workers=args.n_workers,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    stats = plot_trgb_ppc(ppc, str(output), mnras=True)
    distance_output = output.with_name(
        f"{output.stem}_distance_distribution{output.suffix}")
    distance_stats = plot_trgb_ppc_distance(
        ppc, str(distance_output), mnras=True)
    sky_output = output.with_name(
        f"{output.stem}_sky_distribution{output.suffix}")
    sky_stats = plot_trgb_ppc_sky(ppc, str(sky_output), mnras=True)
    exposure_output = None
    exposure_stats = None
    if "sky_exposure" in ppc:
        exposure_output = output.with_name(
            f"{output.stem}_sky_exposure{output.suffix}")
        exposure_stats = plot_trgb_ppc_sky_exposure(
            ppc, str(exposure_output), mnras=True)

    print_section("EDD TRGB PPC result")
    print(f"mode                      : {args.mode}")
    print(f"field_index               : {field_index}")
    print(f"posterior                 : {posterior}")
    print(f"wrote                     : {output}")
    print(f"wrote                     : {distance_output}")
    print(f"wrote                     : {sky_output}")
    if exposure_output is not None:
        print(f"wrote                     : {exposure_output}")
    print(f"n_obs                     : {len(ppc['mag_obs'])}")
    print(f"n_ppc                     : {len(ppc['mag_sim'])}")
    print(f"sky n_obs                 : {sky_stats['n_obs']}")
    print(f"sky n_ppc                 : {sky_stats['n_ppc']}")
    if "sky_exposure" in ppc:
        exposure = ppc["sky_exposure"]
        print(f"sky mask nside            : {exposure['nside']}")
        print(f"sky mask n_pix            : {exposure['n_pix']}")
        print(f"sky mask kappa            : {exposure['kappa']:.3g}")
        print("sky mask relative range   : "
              f"{exposure_stats['ratio_min']:.3g}--"
              f"{exposure_stats['ratio_max']:.3g}")
        print("sky mask accept norm      : "
              f"{exposure['exposure_acceptance_norm']:.3g}")
    print("retained distance median  : "
          f"{distance_stats['r_median']:.3g} Mpc "
          f"[{distance_stats['r_p16']:.3g}, "
          f"{distance_stats['r_p84']:.3g}]")
    print(f"KS mag p                  : {stats['ks_mag_pvalue']:.4g}")
    print(f"KS cz p                   : {stats['ks_cz_pvalue']:.4g}")
    print_rule()


if __name__ == "__main__":
    main()
