"""Compare JAX reconstruction against Carrick dField_43Runs and vField_45Runs."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import sys

sys.path.insert(0, '/Users/yasin/code/CANDEL')

# =============================================================================
# Parameters
# =============================================================================
BOXSIZE = 400.0  # Mpc/h
BETA = 0.43

# vField grid params
N_VFIELD = 257
CELL_VFIELD = BOXSIZE / N_VFIELD

# =============================================================================
# Step 1: Load JAX reconstruction via ClustersAnisModel
# =============================================================================
print("Loading ClustersAnisModel with 43Runs config...")
from candel.model.model import ClustersAnisModel
from candel.field.jax_reconstruction import compute_los_profiles_jax
import jax.numpy as jnp

config_path = "scripts/runs/config_jax_43runs_test.toml"
model = ClustersAnisModel(config_path)

print(f"  Grid: N={model.N_grid}, BOX_SIDE={model.BOX_SIDE}")
print(f"  Schechter: M*={model.Mstar_LF}, alpha={model.alpha_LF}")
print(f"  use_gaussian_splat: {model.precomputed.use_gaussian_splat}")
print(f"  splat_sigma_mpc: {model.precomputed.splat_sigma_mpc}")
print(f"  n_galaxies: {len(model.precomputed.gal_z_obs)}")
print(f"  n_clusters: {len(model.precomputed.cluster_rhat)}")

# Compute LOS profiles (isotropic, H0_dipole=0)
print("\nComputing JAX LOS profiles (H0_dipole=0)...")
H0_dipole = jnp.zeros(3)
los_rho_raw_jax, los_density_jax, los_velocity_jax = compute_los_profiles_jax(H0_dipole, model.precomputed)
los_rho_raw_jax = np.array(los_rho_raw_jax)
los_density_jax = np.array(los_density_jax)
los_velocity_jax = np.array(los_velocity_jax)
los_r = np.array(model.precomputed.los_r)
print(f"  JAX LOS shapes: rho_raw={los_rho_raw_jax.shape}, density={los_density_jax.shape}, velocity={los_velocity_jax.shape}")
print(f"  r range: [{los_r.min():.1f}, {los_r.max():.1f}] Mpc/h")

 # =============================================================================
# Step 2: Load vField_45Runs and interpolate along cluster LOS
 # =============================================================================
print("\nLoading vField_45Runs...")
vfield = np.load('data/Carrick_reconstruction_2015/vField_45Runs.npy')
# vField files are stored as v/beta
print(f"  Shape: {vfield.shape}")

# Build interpolators
coords_v = np.linspace(0, BOXSIZE, N_VFIELD+1)[:-1] + CELL_VFIELD/2
interp_vx = RegularGridInterpolator((coords_v, coords_v, coords_v), vfield[0],
                                     bounds_error=False, fill_value=0.0)
interp_vy = RegularGridInterpolator((coords_v, coords_v, coords_v), vfield[1],
                                     bounds_error=False, fill_value=0.0)
interp_vz = RegularGridInterpolator((coords_v, coords_v, coords_v), vfield[2],
                                     bounds_error=False, fill_value=0.0)

# Get cluster positions from precomputed data
cluster_rhat = np.array(model.precomputed.cluster_rhat)  # (n_cl, 3) Galactic Cartesian
n_clusters = len(cluster_rhat)

# Observer at center of box
obs = np.array([BOXSIZE/2, BOXSIZE/2, BOXSIZE/2])

print(f"Interpolating vField_45Runs along {n_clusters} cluster LOS...")
los_velocity_vfield = np.zeros((n_clusters, len(los_r)))
for i in range(n_clusters):
    # Positions along LOS
    pos = obs + los_r[:, None] * cluster_rhat[i]
    # Interpolate velocity
    vx = interp_vx(pos)
    vy = interp_vy(pos)
    vz = interp_vz(pos)
    # Project onto LOS
    los_velocity_vfield[i] = (vx * cluster_rhat[i, 0] +
                              vy * cluster_rhat[i, 1] +
                              vz * cluster_rhat[i, 2])

# =============================================================================
# Step 2b: Load dField_43Runs and interpolate along cluster LOS
# =============================================================================
print("\nLoading dField_43Runs...")
dfield = np.load('data/Carrick_reconstruction_2015/dField_43Runs.npy')
print(f"  Shape: {dfield.shape}")
interp_den = RegularGridInterpolator((coords_v, coords_v, coords_v), dfield,
                                     bounds_error=False, fill_value=0.0)
los_density_dfield = np.zeros((n_clusters, len(los_r)))
for i in range(n_clusters):
    pos = obs + los_r[:, None] * cluster_rhat[i]
    los_density_dfield[i] = interp_den(pos)
# dField is delta; convert to 1+delta for comparison with LOS density
los_density_dfield_1p = 1.0 + los_density_dfield

# =============================================================================
# Step 3: Compare statistics
# =============================================================================
print("\n" + "="*60)
print("VELOCITY STATISTICS COMPARISON (vField_45Runs)")
print("="*60)
print(f"{'Source':<25} {'Mean (km/s)':<15} {'Std (km/s)':<15}")
print("-"*60)
print(f"{'JAX reconstruction':<25} {los_velocity_jax.mean():>12.1f} {los_velocity_jax.std():>12.1f}")
print(f"{'vField_45Runs interp':<25} {los_velocity_vfield.mean():>12.1f} {los_velocity_vfield.std():>12.1f}")
print("-"*60)

# Correlation
mask = np.isfinite(los_velocity_jax) & np.isfinite(los_velocity_vfield)
corr = np.corrcoef(los_velocity_jax[mask].ravel(), los_velocity_vfield[mask].ravel())[0, 1]
print(f"Correlation coefficient: {corr:.4f}")

print("\nDENSITY STATISTICS COMPARISON (dField_43Runs)")
print(f"{'Source':<25} {'Mean':<15} {'Std':<15}")
print("-"*60)
print(f"{'JAX reconstruction':<25} {los_density_jax.mean():>12.3f} {los_density_jax.std():>12.3f}")
print(f"{'dField_43Runs interp':<25} {los_density_dfield_1p.mean():>12.3f} {los_density_dfield_1p.std():>12.3f}")
mask_den = np.isfinite(los_density_jax) & np.isfinite(los_density_dfield_1p)
corr_den_field = np.corrcoef(los_density_jax[mask_den].ravel(),
                             los_density_dfield_1p[mask_den].ravel())[0, 1]
print(f"Correlation coefficient (density): {corr_den_field:.4f}")

# =============================================================================
# Step 4: Plot comparison (12 clusters)
# =============================================================================
# Multiply velocities by beta before plotting
los_velocity_jax_scaled = los_velocity_jax * BETA
los_velocity_vfield_scaled = los_velocity_vfield * BETA

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

np.random.seed(42)
sample_idx = np.random.choice(n_clusters, 12, replace=False)

# Get cluster Galactic coords for labels
cluster_l = np.rad2deg(np.arctan2(cluster_rhat[:, 1], cluster_rhat[:, 0]))
cluster_b = np.rad2deg(np.arcsin(cluster_rhat[:, 2]))

for i, idx in enumerate(sample_idx):
    ax = axes[i]
    ax.plot(los_r, los_velocity_vfield_scaled[idx], 'b-', lw=2, label='vField_45Runs interp')
    ax.plot(los_r, los_velocity_jax_scaled[idx], 'r--', lw=2, label='JAX reconstruction')
    ax.axhline(0, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel(r'$v_{los}$ [km/s]')
    ax.set_title(f'Cl {idx}: l={cluster_l[idx]:.0f}, b={cluster_b[idx]:.0f}')
    ax.set_xlim(0, 250)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle(f'JAX vs vField_45Runs (N=257, beta={BETA}) | corr={corr:.3f}', fontsize=12)
plt.tight_layout()
plt.savefig('jax_vs_vfield45_comparison_43runs.png', dpi=150)
print(f"\nSaved: jax_vs_vfield45_comparison_43runs.png")
plt.close()

# =============================================================================
# Step 5: Compare against Carrick LOS file (beta=1 convention)
# =============================================================================
import h5py
from scipy.interpolate import interp1d

raw_los_file = 'data/Clusters/los_Clusters_Carrick2015.hdf5'
print(f"\nLoading Carrick LOS file: {raw_los_file}")

with h5py.File(raw_los_file, 'r') as f:
    los_r_raw = f['r'][:]
    los_density_carrick = f['los_density'][0]  # 1+delta with psi+smoothing
    los_velocity_carrick = f['los_velocity'][0]  # v/beta from processed delta
    print(f"  Keys: {list(f.keys())}")
    print(f"  los_density: {los_density_carrick.shape}")
    print(f"  los_velocity: {los_velocity_carrick.shape}")
    print(f"  r range: [{los_r_raw.min():.1f}, {los_r_raw.max():.1f}] Mpc/h")

# Scale velocity by beta
los_velocity_carrick_scaled = los_velocity_carrick * BETA

# Check if r grids match
r_grids_match = (len(los_r) == len(los_r_raw)) and np.allclose(los_r, los_r_raw)
r_grids_differ = not r_grids_match
if r_grids_differ:
    print(f"  Note: r grids differ ({len(los_r)} vs {len(los_r_raw)} points), interpolating JAX to raw grid")
    los_r_plot = los_r_raw
else:
    los_r_plot = los_r

# Statistics comparison
print("\n" + "="*60)
print("CARRICK LOS COMPARISON")
print("="*60)

print(f"\n1+delta (processed):")
print(f"{'Source':<25} {'Mean':<15} {'Std':<15}")
print("-"*60)
print(f"{'JAX':<25} {los_density_jax.mean():>12.3f} {los_density_jax.std():>12.3f}")
print(f"{'carrick':<25} {los_density_carrick.mean():>12.3f} {los_density_carrick.std():>12.3f}")

print(f"\nVelocity (km/s):")
print(f"{'Source':<25} {'Mean':<15} {'Std':<15}")
print("-"*60)
print(f"{'JAX':<25} {los_velocity_jax_scaled.mean():>12.1f} {los_velocity_jax_scaled.std():>12.1f}")
print(f"{'carrick':<25} {los_velocity_carrick_scaled.mean():>12.1f} {los_velocity_carrick_scaled.std():>12.1f}")

# =============================================================================
# Step 6: Plot comparison for 20 clusters - three separate figures
# =============================================================================
np.random.seed(42)
sample_idx_20 = np.random.choice(n_clusters, 20, replace=False)

# Use native grids for plotting when they differ
los_r_jax = los_r
los_r_carrick = los_r_raw

# --- Figure 1: 1+delta (processed) ---
print("\nPlotting 1+delta comparison...")
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for i, idx in enumerate(sample_idx_20):
    ax = axes[i]
    ax.plot(los_r_carrick, los_density_carrick[idx], 'g-', lw=2, label='Carrick LOS')
    ax.plot(los_r_jax, los_density_dfield_1p[idx], 'b-', lw=2, label='dField_43Runs')
    ax.plot(los_r_jax, los_density_jax[idx], 'r--', lw=2, label='JAX')
    ax.axhline(1, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel(r'$1 + \delta$')
    ax.set_title(f'Cl {idx}: l={cluster_l[idx]:.0f}, b={cluster_b[idx]:.0f}')
    ax.set_xlim(0, 250)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

# Compute correlation for density on overlapping r-range
n_overlap = min(los_density_jax.shape[1], los_density_carrick.shape[1])
corr_den = np.corrcoef(
    los_density_jax[:, :n_overlap].ravel(),
    los_density_carrick[:, :n_overlap].ravel()
)[0, 1]

plt.suptitle(f'1+delta: JAX vs Carrick LOS vs dField_43Runs | corr={corr_den:.3f}', fontsize=14)
plt.tight_layout()
plt.savefig('los_density_comparison_43runs.png', dpi=150)
print(f"Saved: los_density_comparison_43runs.png")
plt.close()

# --- Figure 2: Velocity ---
print("\nPlotting velocity comparison...")
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for i, idx in enumerate(sample_idx_20):
    ax = axes[i]
    ax.plot(los_r_carrick, los_velocity_carrick_scaled[idx], 'g-', lw=2, label='Carrick LOS')
    ax.plot(los_r_jax, los_velocity_vfield_scaled[idx], 'b-', lw=2, label='vField_45Runs')
    ax.plot(los_r_jax, los_velocity_jax_scaled[idx], 'r--', lw=2, label='JAX')
    ax.axhline(0, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('r [Mpc/h]')
    ax.set_ylabel(r'$v_{los}$ [km/s]')
    ax.set_title(f'Cl {idx}: l={cluster_l[idx]:.0f}, b={cluster_b[idx]:.0f}')
    ax.set_xlim(0, 250)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

# Compute correlation for velocity on overlapping r-range
n_overlap_v = min(los_velocity_jax_scaled.shape[1], los_velocity_carrick_scaled.shape[1])
corr_vel = np.corrcoef(
    los_velocity_jax_scaled[:, :n_overlap_v].ravel(),
    los_velocity_carrick_scaled[:, :n_overlap_v].ravel()
)[0, 1]

plt.suptitle(f'Velocity: JAX vs Carrick LOS vs vField_45Runs (beta={BETA}) | corr={corr_vel:.3f}', fontsize=14)
plt.tight_layout()
plt.savefig('los_velocity_comparison_43runs.png', dpi=150)
print(f"Saved: los_velocity_comparison_43runs.png")
plt.close()

print(f"\nCorrelations:")
print(f"  Density:  {corr_den:.4f}")
print(f"  Velocity: {corr_vel:.4f}")

# =============================================================================
# Step 7: Compare galaxy catalogues (JAX vs carrick_iterated)
# =============================================================================
print("\n" + "="*60)
print("GALAXY CATALOGUE COMPARISON (JAX vs carrick_iterated)")
print("="*60)

from candel.field.jax_reconstruction import compute_galaxy_catalogue_jax

# Compute JAX catalogue
print("\nComputing JAX galaxy catalogue...")
z_obs_jax, r_mpc_jax, lumWeight_jax, L_over_Lstar_jax, l_deg_jax, b_deg_jax = \
    compute_galaxy_catalogue_jax(H0_dipole, model.precomputed)

# Convert to numpy
z_obs_jax = np.array(z_obs_jax)
r_mpc_jax = np.array(r_mpc_jax)
lumWeight_jax = np.array(lumWeight_jax)
L_over_Lstar_jax = np.array(L_over_Lstar_jax)
l_deg_jax = np.array(l_deg_jax)
b_deg_jax = np.array(b_deg_jax)

print(f"  n_galaxies: {len(z_obs_jax)}")
print(f"  r_mpc: [{r_mpc_jax.min():.1f}, {r_mpc_jax.max():.1f}] Mpc/h")
print(f"  lumWeight: [{lumWeight_jax.min():.3f}, {lumWeight_jax.max():.3f}]")
print(f"  L/L*: [{L_over_Lstar_jax.min():.3f}, {L_over_Lstar_jax.max():.3f}]")

# Load carrick_iterated catalogue
carrick_cat_file = 'carrick_iterated_catalogue_iso_iter000.npz'
print(f"\nLoading carrick catalogue: {carrick_cat_file}")
cat_carrick = np.load(carrick_cat_file)

z_obs_carrick = cat_carrick['z_obs']
r_mpc_carrick = cat_carrick['r_mpc']
lumWeight_carrick = cat_carrick['lumWeight']
L_over_Lstar_carrick = cat_carrick['L_over_Lstar']
l_deg_carrick = cat_carrick['l_deg']
b_deg_carrick = cat_carrick['b_deg']

print(f"  n_galaxies: {len(z_obs_carrick)}")
print(f"  r_mpc: [{r_mpc_carrick.min():.1f}, {r_mpc_carrick.max():.1f}] Mpc/h")
print(f"  lumWeight: [{lumWeight_carrick.min():.3f}, {lumWeight_carrick.max():.3f}]")
print(f"  L/L*: [{L_over_Lstar_carrick.min():.3f}, {L_over_Lstar_carrick.max():.3f}]")

# Match galaxies by coordinates (in case filtering differs slightly)
if len(z_obs_jax) != len(z_obs_carrick):
    print(f"\nNote: Different galaxy counts ({len(z_obs_jax)} vs {len(z_obs_carrick)})")
    print("Matching by coordinates...")

    from scipy.spatial import cKDTree
    # Build KD-tree from carrick coordinates
    coords_carrick = np.column_stack([l_deg_carrick, b_deg_carrick])
    coords_jax = np.column_stack([l_deg_jax, b_deg_jax])
    tree = cKDTree(coords_carrick)

    # Find nearest match for each JAX galaxy
    dists, indices = tree.query(coords_jax, k=1)

    # Keep only close matches (within 0.01 degrees)
    good_match = dists < 0.01
    print(f"  Matched {good_match.sum()} of {len(z_obs_jax)} galaxies (dist < 0.01 deg)")

    # Extract matched pairs
    r_mpc_jax_matched = r_mpc_jax[good_match]
    r_mpc_carrick_matched = r_mpc_carrick[indices[good_match]]
    lumWeight_jax_matched = lumWeight_jax[good_match]
    lumWeight_carrick_matched = lumWeight_carrick[indices[good_match]]
    L_over_Lstar_jax_matched = L_over_Lstar_jax[good_match]
    L_over_Lstar_carrick_matched = L_over_Lstar_carrick[indices[good_match]]

    # Rename for plotting code below
    r_mpc_jax = r_mpc_jax_matched
    r_mpc_carrick = r_mpc_carrick_matched
    lumWeight_jax = lumWeight_jax_matched
    lumWeight_carrick = lumWeight_carrick_matched
    L_over_Lstar_jax = L_over_Lstar_jax_matched
    L_over_Lstar_carrick = L_over_Lstar_carrick_matched

if len(r_mpc_jax) > 0:
    # Compute correlations
    corr_r = np.corrcoef(r_mpc_jax, r_mpc_carrick)[0, 1]
    corr_lw = np.corrcoef(lumWeight_jax, lumWeight_carrick)[0, 1]
    corr_L = np.corrcoef(L_over_Lstar_jax, L_over_Lstar_carrick)[0, 1]

    print(f"\nCatalogue correlations:")
    print(f"  r_mpc:      {corr_r:.6f}")
    print(f"  lumWeight:  {corr_lw:.6f}")
    print(f"  L/L*:       {corr_L:.6f}")

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: 1-to-1 comparisons
    ax = axes[0, 0]
    ax.scatter(r_mpc_carrick, r_mpc_jax, alpha=0.1, s=1)
    ax.plot([0, 200], [0, 200], 'r--', lw=2)
    ax.set_xlabel('r_mpc (carrick)')
    ax.set_ylabel('r_mpc (JAX)')
    ax.set_title(f'Distance | corr={corr_r:.4f}')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_aspect('equal')

    ax = axes[0, 1]
    ax.scatter(lumWeight_carrick, lumWeight_jax, alpha=0.1, s=1)
    lw_max = min(lumWeight_carrick.max(), lumWeight_jax.max(), 100)
    ax.plot([0, lw_max], [0, lw_max], 'r--', lw=2)
    ax.set_xlabel('lumWeight (carrick)')
    ax.set_ylabel('lumWeight (JAX)')
    ax.set_title(f'lumWeight | corr={corr_lw:.4f}')
    ax.set_xlim(0, lw_max)
    ax.set_ylim(0, lw_max)

    ax = axes[0, 2]
    ax.scatter(L_over_Lstar_carrick, L_over_Lstar_jax, alpha=0.1, s=1)
    L_max = min(L_over_Lstar_carrick.max(), L_over_Lstar_jax.max(), 20)
    ax.plot([0, L_max], [0, L_max], 'r--', lw=2)
    ax.set_xlabel('L/L* (carrick)')
    ax.set_ylabel('L/L* (JAX)')
    ax.set_title(f'L/L* | corr={corr_L:.4f}')
    ax.set_xlim(0, L_max)
    ax.set_ylim(0, L_max)

    # Row 2: Histograms
    ax = axes[1, 0]
    ax.hist(r_mpc_carrick, bins=50, alpha=0.5, label='carrick', density=True)
    ax.hist(r_mpc_jax, bins=50, alpha=0.5, label='JAX', density=True)
    ax.set_xlabel('r_mpc [Mpc/h]')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title('Distance distribution')

    ax = axes[1, 1]
    ax.hist(lumWeight_carrick, bins=50, alpha=0.5, label='carrick', density=True, range=(0, 50))
    ax.hist(lumWeight_jax, bins=50, alpha=0.5, label='JAX', density=True, range=(0, 50))
    ax.set_xlabel('lumWeight')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title('lumWeight distribution')

    ax = axes[1, 2]
    ax.hist(L_over_Lstar_carrick, bins=50, alpha=0.5, label='carrick', density=True, range=(0, 10))
    ax.hist(L_over_Lstar_jax, bins=50, alpha=0.5, label='JAX', density=True, range=(0, 10))
    ax.set_xlabel('L/L*')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title('L/L* distribution')

    plt.suptitle('Galaxy Catalogue: JAX vs carrick_iterated', fontsize=14)
    plt.tight_layout()
    plt.savefig('catalogue_comparison_43runs.png', dpi=150)
    print(f"\nSaved: catalogue_comparison_43runs.png")
    plt.close()
