"""Benchmark HDF5 gzip compression vs uncompressed for LOS-sized arrays."""
import time
import tempfile
import os

import numpy as np
from h5py import File


def main():
    # Representative LOS array: 30 sims x 500 galaxies x 301 radial steps
    # Use smooth correlated data (closer to real density/velocity fields)
    rng = np.random.default_rng(42)
    r = np.linspace(0.1, 250, 301).astype(np.float32)
    # Smooth profiles: each galaxy has a slowly varying density + noise
    base = 1.0 + 0.3 * np.sin(r[None, :] / 50)
    data = np.tile(base, (30, 500, 1)).astype(np.float32)
    data += rng.standard_normal(data.shape).astype(np.float32) * 0.01
    print(f"Array shape: {data.shape}, dtype: {data.dtype}, "
          f"size: {data.nbytes / 1e6:.1f} MB")

    tmpdir = tempfile.mkdtemp()
    f_raw = os.path.join(tmpdir, "raw.hdf5")
    f_gz = os.path.join(tmpdir, "gz.hdf5")

    # Write both files
    with File(f_raw, "w") as f:
        f.create_dataset("data", data=data)
    with File(f_gz, "w") as f:
        f.create_dataset("data", data=data, chunks=True,
                         compression="gzip", compression_opts=4)

    raw_size = os.path.getsize(f_raw)
    gz_size = os.path.getsize(f_gz)
    print(f"\nFile sizes:  raw={raw_size/1e6:.1f} MB, "
          f"gzip={gz_size/1e6:.1f} MB, "
          f"ratio={raw_size/gz_size:.2f}x")

    # Benchmark reads
    n_iter = 20
    for label, path in [("raw", f_raw), ("gzip", f_gz)]:
        # Warm cache
        with File(path, "r") as f:
            _ = f["data"][:]

        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            with File(path, "r") as f:
                _ = f["data"][:]
            times.append(time.perf_counter() - t0)

        times = np.array(times)
        print(f"  {label:5s}: {times.mean()*1e3:.1f} ± {times.std()*1e3:.1f} ms "
              f"(min {times.min()*1e3:.1f} ms)")

    os.remove(f_raw)
    os.remove(f_gz)
    os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
