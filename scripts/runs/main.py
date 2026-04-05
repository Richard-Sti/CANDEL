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
A short script to run inference on a PV model. It loads the model and data
from the configuration file and runs the inference.

This script is expected to be run either from the command line or from a shell
submission script.
"""
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser
from os.path import exists

# ---- Pre-parse device args BEFORE importing anything that pulls JAX/NumPyro
_pre = ArgumentParser(add_help=False)
_pre.add_argument(
    "--host-devices", type=int,
    help="Set NumPyro host device count before importing candel."
)
_pre_args, _ = _pre.parse_known_args()

if _pre_args.host_devices:
    import numpyro  # safe to import here; must be before candel/JAX use
    if _pre_args.host_devices:
        numpyro.set_host_device_count(_pre_args.host_devices)

# Only now import candel (which may import jax/numpyro internally)
import candel  # noqa
from candel import fprint, get_nested  # noqa


class GPUMonitor:
    """Poll nvidia-smi in a background thread and report a summary on stop."""

    def __init__(self, interval=10):
        self.interval = interval
        self._util, self._mem_used, self._mem_total = [], [], []
        self._times = []
        self._t0 = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _query(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL).decode()
            u, mu, mt = out.strip().split(",")
            self._util.append(float(u))
            self._mem_used.append(float(mu))
            self._mem_total.append(float(mt))
            self._times.append(time.time() - self._t0)
        except Exception:
            pass

    def _run(self):
        while not self._stop.wait(self.interval):
            self._query()

    def start(self):
        self._t0 = time.time()
        self._query()
        self._thread.start()

    def stop(self):
        self._query()
        self._stop.set()
        self._thread.join()
        if not self._util:
            return
        n = len(self._util)
        mem_total = self._mem_total[0]
        mem_pct = [100 * m / mem_total for m in self._mem_used]
        mean_util = sum(self._util) / n
        mean_mem = sum(mem_pct) / n
        peak_util = max(self._util)
        peak_mem = max(mem_pct)
        width = min(60, max(n, 2))

        def _chart(values, vmin, vmax, label):
            if vmax == vmin:
                vmax = vmin + 1
            height = 5
            lines = [f"   {label}"]
            for row in range(height):
                threshold = vmax - (row / (height - 1)) * (vmax - vmin)
                if n >= width:
                    step = n / width
                    idxs = [int(i * step) for i in range(width)]
                else:
                    idxs = [int(i * (n - 1) / max(width - 1, 1))
                            for i in range(width)]
                bar = "".join(
                    "█" if values[min(i, n-1)] >= threshold else " "
                    for i in idxs)
                lines.append(f"   {threshold:4.0f}% |{bar}|")
            t_total = int(self._times[-1])
            t_min, t_sec = divmod(t_total, 60)
            t_label = f"{t_min}m{t_sec:02d}s" if t_min else f"{t_sec}s"
            lines.append(f"         +{'-' * width}+")
            lines.append(f"         0{t_label:>{width}}")
            return "\n".join(lines)

        fprint("── GPU usage summary ────────────────────────────────────────")
        fprint(f"   utilization : mean {mean_util:.0f}%,  peak {peak_util:.0f}%")
        fprint(f"   memory      : mean {mean_mem:.1f}%,  peak {peak_mem:.1f}%"
               f"  ({max(self._mem_used):.0f} / {mem_total:.0f} MiB)")
        fprint(_chart(self._util, 0, 100, "GPU utilization (%)"))
        fprint(_chart(mem_pct, 0, 100, "GPU memory (%)"))


def insert_comment_at_top(path: str, label: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    comment = f"# Job {label} at: {timestamp}\n"
    with open(path, "r") as f:
        original = f.readlines()
    with open(path, "w") as f:
        f.write(comment)
        f.writelines(original)


_SPOT_ARRAY_KEYS = ("velocity", "x", "sigma_x", "y", "sigma_y",
                     "a", "sigma_a", "accel_measured", "is_highvel",
                     "phi_lo", "phi_hi")


def downsample_spots(data, max_spots, seed=42):
    """Randomly downsample maser spot data to at most `max_spots`."""
    n = data["n_spots"]
    if max_spots >= n:
        return data
    rng = __import__("numpy").random.default_rng(seed)
    idx = rng.choice(n, max_spots, replace=False)
    idx.sort()
    data = dict(data)
    for key in _SPOT_ARRAY_KEYS:
        if key in data:
            data[key] = data[key][idx]
    data["n_spots"] = max_spots
    fprint(f"downsampled to {max_spots}/{n} spots.")
    return data


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run inference on a PV model."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    parser.add_argument("--max-spots", type=int, default=None,
                        help="Randomly downsample maser spots to this many.")
    # Re-expose the pre-parsed options so they show up in --help
    parser.add_argument("--host-devices", type=int,
                        help="NumPyro host device count (handled pre-import).")
    args = parser.parse_args()

    insert_comment_at_top(args.config, "started")

    gpu_monitor = GPUMonitor(interval=10)
    gpu_monitor.start()

    config = candel.load_config(args.config, replace_los_prior=False)

    fname_out = get_nested(config, "io/fname_output")
    skip_if_exists = get_nested(config, "inference/skip_if_exists", False)
    if skip_if_exists and exists(fname_out):
        fprint(f"Output file `{fname_out}` already exists. "
               "Skipping inference.")
        insert_comment_at_top(args.config, "skipped")
        gpu_monitor.stop()
        sys.exit(0)

    try:
        which_run = get_nested(config, "model/which_run", None)
        if which_run == "CH0":
            fprint("selected `CH0` model.")
            data = candel.pvdata.load_SH0ES_from_config(args.config, )
            model = candel.model.CH0Model(args.config, data)
            candel.run_H0_inference(model, )
        elif which_run == "CCHP":
            fprint("selected `CCHP` model (TRGB with SN data).")
            data = candel.pvdata.load_CCHP_from_config(args.config)
            model = candel.model.TRGBModel(args.config, data)
            candel.run_H0_inference(model, )
        elif which_run in ("EDD_TRGB", "EDD_TRGB_grouped"):
            fprint(f"selected `{which_run}` model.")
            if which_run == "EDD_TRGB_grouped":
                data = candel.pvdata.load_EDD_TRGB_grouped_from_config(
                    args.config)
            else:
                data = candel.pvdata.load_EDD_TRGB_from_config(args.config)
            model = candel.model.TRGBModel(args.config, data)
            candel.run_H0_inference(model, )

            # Posterior predictive check
            if get_nested(config, "model/run_ppc", True):
                from candel.mock.ppc_trgb import generate_trgb_ppc, plot_trgb_ppc

                fprint("running posterior predictive check...")
                samples = candel.read_samples("", fname_out)
                ppc = generate_trgb_ppc(samples, data, args.config)
                ppc_fname = fname_out.rsplit(".", 1)[0] + "_ppc.png"
                plot_trgb_ppc(ppc, ppc_fname)
        elif which_run == "CCHP_CSP":
            fprint("selected `CCHP_CSP` joint TRGB-CSP model.")
            trgb_data = candel.pvdata.load_CCHP_from_config(args.config)
            csp_data = candel.pvdata.load_CSP_from_config(args.config)
            model = candel.model.JointTRGBCSPModel(
                args.config, trgb_data, csp_data)
            candel.run_H0_inference(model, )
        elif which_run == "maser_disk":
            import tempfile
            import tomli_w

            maser_cfg = get_nested(config, "io/maser_data", {})
            root = maser_cfg.get("root", "data/Megamaser")
            galaxy = get_nested(config, "model/galaxy", "CGCG074-064")
            all_galaxies = get_nested(config, "model/galaxies", {})

            if galaxy == "joint":
                galaxy_names = list(all_galaxies.keys())
                fprint(f"selected joint maser disk model "
                       f"({len(galaxy_names)} galaxies: "
                       f"{', '.join(galaxy_names)}).")
                data_list = [
                    candel.pvdata.load_megamaser_spots(
                        root, g,
                        v_sys_obs=all_galaxies[g]["v_sys_obs"])
                    for g in galaxy_names]
                if args.max_spots is not None:
                    data_list = [downsample_spots(d, args.max_spots)
                                 for d in data_list]
                config["io"]["fname_output"] = "results/Maser/joint.hdf5"

                tmp = tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".toml", delete=False)
                tomli_w.dump(config, tmp)
                tmp.close()
                model = candel.model.JointMaserModel(
                    tmp.name, data_list)
            else:
                fprint(f"selected maser disk model for {galaxy}.")
                gal_cfg = all_galaxies.get(galaxy, {})
                for key in ("fit_di_dr", "sample_accel_det", "use_selection"):
                    if key in gal_cfg:
                        config["model"][key] = gal_cfg[key]
                gal_priors = gal_cfg.get("priors", {})
                for pname, pval in gal_priors.items():
                    config["model"]["priors"][pname] = pval

                # Per-galaxy init_values override global init settings.
                gal_init = gal_cfg.get("init_values", None)
                if gal_init is not None:
                    config["inference"]["init_values"] = gal_init
                    fprint(f"using per-galaxy init_values for {galaxy}.")

                config["io"]["fname_output"] = (
                    f"results/Maser/{galaxy}.hdf5")

                tmp = tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".toml", delete=False)
                tomli_w.dump(config, tmp)
                tmp.close()
                data = candel.pvdata.load_megamaser_spots(
                    root, galaxy,
                    v_sys_obs=gal_cfg["v_sys_obs"])
                if args.max_spots is not None:
                    data = downsample_spots(data, args.max_spots)
                model = candel.model.MaserDiskModel(tmp.name, data)

            candel.run_H0_inference(model)
        else:
            data = candel.pvdata.load_PV_dataframes(args.config)

            model_name = config["inference"]["model"]
            data_name = config["io"]["catalogue_name"]
            fprint(f"Loading model `{model_name}` from `{args.config}` "
                   f"for data `{data_name}`")

            shared_param = get_nested(config, "inference/shared_params", None)
            model = candel.model.name2model(model_name, shared_param, args.config)

            if isinstance(data, list):
                if not isinstance(model, candel.model.JointPVModel):
                    raise TypeError(
                        "You provided multiple datasets, but the selected model "
                        f"`{model.__class__.__name__}` is not JointPVModel."
                    )
                if len(data) != len(model.submodels):
                    raise ValueError(
                        f"Number of datasets ({len(data)}) does not match "
                        f"number of submodels ({len(model.submodels)}) in the "
                        "joint model."
                    )

            model_kwargs = {"data": data}
            candel.run_pv_inference(model, model_kwargs)

            insert_comment_at_top(args.config, "finished")
    finally:
        gpu_monitor.stop()
