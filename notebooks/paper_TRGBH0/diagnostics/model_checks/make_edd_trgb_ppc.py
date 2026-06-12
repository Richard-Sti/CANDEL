#!/usr/bin/env python
"""Make EDD TRGB Gaussian PPC plots."""
import argparse
import copy
from pathlib import Path
import sys
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

import matplotlib
matplotlib.use("Agg")

import candel
import tomli_w
from candel.mock import generate_trgb_ppc, plot_trgb_ppc
from trgbh0_plot_style import OUTPUT_DIR, ROOT, TRGBH0_TABLE_RESULTS


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
    "carrick": OUTPUT_DIR / "trgbh0_edd_trgb_carrick_gaussian_ppc.png",
    "none": OUTPUT_DIR / "trgbh0_edd_trgb_nofield_gaussian_ppc.png",
    "manticore": OUTPUT_DIR / "trgbh0_edd_trgb_manticore_gaussian_ppc.png",
}

SEED = 42
DEFAULT_FIELD_INDEX = 0


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
              "gaussian_ppc.png"
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
        help="Output PNG path.")
    parser.add_argument(
        "--n-ppc", type=int, default=None,
        help="Number of PPC galaxies. Defaults to config ppc_factor.")
    parser.add_argument(
        "--ppc-factor", type=int, default=None,
        help="Override config model.ppc_factor when --n-ppc is omitted.")
    parser.add_argument(
        "--manticore-root", type=Path, default=None,
        help="Root containing ManticoreLocalCOLA MAS subdirectories.")
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

    data = load_observed_data(config)
    posterior = args.posterior or default_posterior(args.mode, field_index)
    output = args.output or default_output(args.mode, field_index)
    samples = candel.read_samples("", str(posterior))

    ppc = generate_trgb_ppc(
        samples,
        data,
        config,
        n_ppc=args.n_ppc,
        seed=args.seed,
        field_index=field_index,
        progress=not args.no_progress,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    stats = plot_trgb_ppc(ppc, str(output))

    print(f"mode={args.mode}")
    print(f"field_index={field_index}")
    print(f"posterior={posterior}")
    print(f"Wrote {output}")
    print(f"n_obs={len(ppc['mag_obs'])}, n_ppc={len(ppc['mag_sim'])}")
    print(f"KS mag p={stats['ks_mag_pvalue']:.4g}")
    print(f"KS cz p={stats['ks_cz_pvalue']:.4g}")


if __name__ == "__main__":
    main()
