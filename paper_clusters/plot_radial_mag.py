"""Replot radial Vext magnitude profiles for LT, YT, and LTYT."""

from h5py import File

from config import get_results_path, get_figure_path, setup_style, CANDEL_ROOT
from candel.inference import postprocess_samples
from candel.model.model import ClustersModel
from candel.pvdata.data import load_PV_dataframes
from candel.util import fprint, plot_Vext_radmag


def load_samples(h5_path):
    """Load the samples group from a saved HDF5 file into a dict of arrays."""
    samples = {}
    with File(h5_path, "r") as f:
        grp = f["samples"]
        for key in grp.keys():
            samples[key] = grp[key][()]
    return samples


def plot_radmag_run(label, radmag_stem, diph0_stem):
    """Replot a single radial-magnitude run with its dipH0 reference."""
    config_path = get_results_path(f"{radmag_stem}.toml")
    results_path = get_results_path(f"{radmag_stem}.hdf5")
    diph0_path = get_results_path(f"{diph0_stem}.hdf5")

    out_path = get_figure_path(f"{radmag_stem}_profile_Vext_radmag.png")

    fprint(f"[{label}] loading model from {config_path}")
    model = ClustersModel(str(config_path))

    fprint(f"[{label}] loading data from {config_path}")
    data = load_PV_dataframes(str(config_path), local_root=str(CANDEL_ROOT))

    fprint(f"[{label}] loading samples from {results_path}")
    samples = postprocess_samples(load_samples(str(results_path)))

    fprint(f"[{label}] loading dipH0 samples from {diph0_path}")
    h0_samples = postprocess_samples(
        load_samples(str(diph0_path)), convert_zeropoint_to_dH=True)

    fprint(f"[{label}] plotting to {out_path}")
    plot_Vext_radmag(
        samples,
        model,
        show_fig=False,
        filename=str(out_path),
        data=data,
        h0_samples=h0_samples,
    )


def main():
    setup_style()
    runs = [
        ("LT", "Carrick2015_LT_noMNR_radmagVext", "Carrick2015_LT_noMNR_dipH0"),
        ("YT", "Carrick2015_YT_noMNR_radmagVext_hasY", "Carrick2015_YT_noMNR_dipH0_hasY"),
        ("LTYT", "Carrick2015_LTYT_noMNR_radmagVext_hasY", "Carrick2015_LTYT_noMNR_dipH0_hasY"),
    ]
    for label, radmag_stem, diph0_stem in runs:
        plot_radmag_run(label, radmag_stem, diph0_stem)


if __name__ == "__main__":
    main()
