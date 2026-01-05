"""
Example: Loading zcosmo posterior data from HDF5.

The file contains cosmological redshift posteriors for each galaxy,
computed on a common redshift grid.

Ordering
--------
The ordering in the HDF5 file matches the row order in the original input
text file exactly. Galaxy at row `i` in the text file corresponds to index
`i` in all HDF5 arrays (RA[i], dec[i], log_posterior[i, :], etc.).
"""
import numpy as np
from h5py import File

fname = "/Users/rstiskalek/Projects/CANDEL/data/BTFR_Andreea/master_sample_bTFr_redshifts_genform_zcosmo_posterior.hdf5"  # noqa

with File(fname, 'r') as f:
    # Input galaxy data (1D arrays, length = number of galaxies)
    RA = f["RA"][:]          # Right ascension [deg]
    dec = f["dec"][:]        # Declination [deg]
    Vcmb = f["Vcmb"][:]      # CMB-frame velocity [km/s]

    # Redshift grid (1D array, length = number of grid points)
    z_grid = f["z_grid"][:]

    # Log-posterior: shape (n_galaxies, n_grid)
    # Each row is the log-posterior p(z_cosmo | data) for one galaxy
    log_posterior = f["log_posterior"][:]

    # Summary statistics (1D arrays, length = n_galaxies)
    z_p16 = f["zcosmo_p16"][:]  # 16th percentile
    z_p50 = f["zcosmo_p50"][:]  # 50th percentile (median)
    z_p84 = f["zcosmo_p84"][:]  # 84th percentile

ngal, nz = log_posterior.shape
print(f"Number of galaxies: {ngal}")
print(f"Redshift grid points: {nz}")
print(f"Redshift range: [{z_grid[0]:.4f}, {z_grid[-1]:.4f}]")

# Convert Vcmb to redshift
zcmb = Vcmb / 299792.458

# Print 20 randomly chosen galaxies
rng = np.random.default_rng(42)
indices = rng.choice(ngal, size=20, replace=False)

for i in indices:
    # Posterior: exp(log_posterior) integrates to 1 over z_grid
    p = np.exp(log_posterior[i])

    print(f"Galaxy {i}: RA={RA[i]:.3f}, dec={dec[i]:.3f}, zcmb={zcmb[i]:.4f}")
    print(f"  z_cosmo = {z_p50[i]:.4f} "
          f"(+{z_p84[i]-z_p50[i]:.4f} / -{z_p50[i]-z_p16[i]:.4f})")
