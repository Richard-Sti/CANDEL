"""
MPI script for mock cluster analysis with iterative cluster removal.

Each MPI rank:
1. Generates a mock dataset with a different seed
2. Runs inference with and without dipole
3. Iteratively removes clusters preferring dipole (based on log-likelihood)
4. Saves results for each removal iteration

Usage:
    mpiexec -n 4 python analyze_mocks_with_cluster_removal.py --config config.toml --nmocks 10 --n_remove_iterations 10
"""

import sys
import os
from os.path import join, exists
import argparse
import numpy as np
from h5py import File
import copy
import re
from candel.util import radec_to_galactic

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    HAS_MPI = True
except ImportError:
    rank = 0
    size = 1
    HAS_MPI = False
    print("Warning: mpi4py not available, running in serial mode")

# Add CANDEL to path
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import candel


def fprint(msg):
    """Print message with rank prefix."""
    print(f"[Rank {rank}] {msg}", flush=True)


def sample_uniform(n=1, low=0.0, high=10.0, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=n)


def generate_mock(nsamples, seed, field_loader, output_dir, mock_id=0):
    """
    Generate a mock cluster dataset.
    
    Parameters
    ----------
    nsamples : int
        Number of clusters to generate
    seed : int
        Random seed
    field_loader : FieldLoader
        Field reconstruction loader
    output_dir : str
        Directory to save the mock HDF5 file
    mock_id : int
        Mock identifier for naming
    
    Returns
    -------
    mock : dict
        Mock data dictionary
    mock_path : str
        Path to saved mock file
    """
    
    rng = np.random.default_rng(seed)

    # fixed values
    b1 = rng.uniform(1.0, 6.0)
    beta = rng.normal(0.43, 0.02)

    # Sample sigma_int from Jeffreys prior between 0.005 and 0.2
    log_sigma_int = rng.uniform(np.log(0.01), np.log(0.2))
    sigma_int = np.exp(log_sigma_int)
    log_sigma_v = rng.uniform(np.log(200.0), np.log(700.0))
    sigma_v = np.exp(log_sigma_v)

    # draw ALL random params from the SAME rng
    A_CL = rng.uniform(1.0, 3.0)
    B_CL = rng.uniform(2.0, 3.0)

    zeropoint_dipole_mag = rng.uniform(0.0, 0.1)

    # isotropic direction: phi ~ U[0,2π), cosθ ~ U[-1,1]
    phi = rng.uniform(0.0, 2*np.pi)
    cos_theta = rng.uniform(-1.0, 1.0)

    # convert to (RA, dec)s
    theta = np.arccos(cos_theta)                 # [0, π]
    ra = np.rad2deg(phi)                         # [0°, 360°)
    dec = np.rad2deg(0.5*np.pi - theta)          # [-90°, 90°]

    # to galactic (ℓ, b)
    zeropoint_dipole_ell, zeropoint_dipole_b = radec_to_galactic(ra, dec)

    # Sample distance model parameters to match toml priors
    R = rng.uniform(50.0, 200.0)  # uniform [50, 200]
    # p: truncated normal (mean=2.0, scale=0.1, low=0.0) - approximate with normal clipped
    p = np.clip(rng.normal(2.0, 0.1), 0.0, None)
    n = rng.uniform(0.5, 1.5)  # uniform [0.5, 1.5]
    
    fprint(f"Generating mock with {nsamples} clusters, seed={seed}, b1={b1:.3f}, sigma_int={sigma_int:.3f}, sigma_v={sigma_v:.1f}, R={R:.1f}, p={p:.2f}, n={n:.2f}")

    # Mock generation parameters (same as make_Clusters_mocks.ipynb)
    kwargs = {
        'r_grid': np.linspace(0.1, 1001, 1001),
        'Vext_mag': 0.00,
        'Vext_ell': 0.0,
        'Vext_b': 0.0,
        'sigma_v': sigma_v,
        'beta': beta,
        'b1': b1,
        'A_CL': A_CL,
        'B_CL': B_CL,
        'sigma_int': sigma_int,
        'A_CL_LT': 0.0,
        'B_CL_LT': 2.5,
        'sigma_int_LT': 0.15,
        'zeropoint_dipole_mag': zeropoint_dipole_mag,
        'zeropoint_dipole_ell': zeropoint_dipole_ell,
        'zeropoint_dipole_b': zeropoint_dipole_b,
        'h': 1.0,
        'logT_prior_mean': 0.0,
        'logT_prior_std': 0.2,
        'e_logT': 0.03,
        'e_logY': 0.09,
        'e_logF': 0.05,
        'b_min': 20.0,  #20.0,
        'zcmb_max': 0.45,
        'R_dist_emp': R,
        'p_dist_emp': p,
        'n_dist_emp': n,
        'field_loader': field_loader,
        'r2distmod': candel.Distance2Distmod(),
        'r2z': candel.Distance2Redshift(),
        'Om': 0.3,
    }
    
    mock = candel.mock.gen_Clusters_mock(nsamples, seed=seed, **kwargs)
    
    # Save mock to HDF5
    mock_path = join(output_dir, f"mock_{mock_id:04d}.hdf5")
    fprint(f"Saving mock to {mock_path}")
    
    with File(mock_path, 'w') as f:
        grp = f.create_group("mock")
        for key, value in mock.items():
            grp.create_dataset(key, data=value, dtype=np.float32)
        
        # Save parameters as attributes
        for key, value in kwargs.items():
            if isinstance(value, (float, int, bool)):
                grp.attrs[key] = value
        grp.attrs["seed"] = seed
        grp.attrs["nsamples"] = nsamples
        grp.attrs["mock_id"] = mock_id
    
    return mock, mock_path


def run_inference_on_mock(config_path, mock_dir, mock_id, output_dir, 
                          field_density, field_velocity, num_samples=None,
                          temp_suffix="", verbose_name="", mock_name=None):
    """
    Run inference on mock data with a given config.
    
    Parameters
    ----------
    config_path : str
        Config file path
    mock_dir : str
        Directory containing mock files
    mock_id : int
        Mock identifier
    output_dir : str
        Directory to save temp config
    field_density : str
        Path to field density file
    field_velocity : str
        Path to field velocity file
    temp_suffix : str
        Suffix for temp config file
    verbose_name : str
        Name for logging
    mock_name : str, optional
        Name of the mock to load (e.g., "Clusters_mock_0000" or "Clusters_mock_0000_filtered_iter01")
        If None, uses "Clusters_mock_{mock_id:04d}"
    
    Returns
    -------
    lp : array
        Log density per sample
    output : tuple
        Full inference output
    data : PVDataFrame
        Loaded data
    """
    if mock_name is None:
        mock_name = f"Clusters_mock_{mock_id:04d}"

    print('num_samples:', num_samples)
    
    # Create a temporary config by modifying the original via text replacement
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    # Replace catalogue_name and root path using simple text substitution
    # This avoids TOML parsing issues with nested arrays
    config_text = config_text.replace(
        'catalogue_name = "Clusters_mock_0"',
        f'catalogue_name = "{mock_name}"'
    )
    config_text = config_text.replace(
        'root = "PLACEHOLDER_MOCK_DIR"',
        f'root = "{mock_dir}"'
    )
    
    # Replace field paths
    config_text = config_text.replace(
        'path_density = "PLACEHOLDER_DENSITY"',
        f'path_density = "{field_density}"'
    )
    config_text = config_text.replace(
        'path_velocity = "PLACEHOLDER_VELOCITY"',
        f'path_velocity = "{field_velocity}"'
    )
    
    # Set unique output file for this mock and iteration
    # temp_suffix contains info about which run this is (e.g., "_nodipole" or "_dipole_iter0")
    output_file = join(output_dir, f"mock_{mock_id:04d}{temp_suffix}.hdf5")
    config_text = config_text.replace(
        'fname_output = "results/mocks/nodipole.hdf5"',
        f'fname_output = "{output_file}"'
    )
    config_text = config_text.replace(
        'fname_output = "results/mocks/dipole.hdf5"',
        f'fname_output = "{output_file}"'
    )

    config_text = config_text.replace(
        'num_samples = 1000', f'num_samples = {num_samples}'
    )
    config_text = config_text.replace(
        'num_warmup = 1000', f'num_warmup = {num_samples}'
    )

    # Replace whatever number appears after '='

    # config_text = re.sub(r'n_warmup\s*=\s*\d+', f'n_warmup = {num_samples}', config_text)
    # config_text = re.sub(r'n_samples\s*=\s*\d+', f'n_samples = {num_samples}', config_text)

    # Save temp config
    temp_config = join(output_dir, f"temp_config_mock{mock_id:04d}{temp_suffix}.toml")
    with open(temp_config, 'w') as f:
        f.write(config_text)
    
    # Load data and run inference
    data = candel.pvdata.load_PV_dataframes(temp_config)
    fprint(f"  {verbose_name}: {len(data['RA'])} clusters")
    
    model = candel.model.ClustersModel(temp_config)
    samples, log_density = candel.run_pv_inference(
        model, {'data': data}, save_samples=True, print_summary=False
    )
    
    # Read log_density_per_sample from saved file
    config = candel.load_config(temp_config)
    output_file = config['io']['fname_output']
    
    with File(output_file, 'r') as f:
        lp = f['log_density_per_sample'][...]
    
    os.remove(temp_config)
    
    return lp, (samples, log_density), data


def remove_clusters_iteratively(data, mean_dlog, n_remove=1, verbose=True):
    """
    Remove clusters with highest mean_dlog values from data dictionary.
    
    Parameters
    ----------
    data : PVDataFrame
        The data object containing cluster information
    mean_dlog : array_like
        Array of length n_clusters with mean log density difference values
        (positive means cluster prefers dipole model)
    n_remove : int
        Number of clusters to remove (removes highest mean_dlog values)
    verbose : bool
        Print information about removed clusters
    
    Returns
    -------
    data_filtered : dict
        New data dictionary with clusters removed
    mask : array
        Boolean mask where True = kept, False = removed
    removed_indices : array
        Indices of removed clusters
    """
    n_clusters = len(mean_dlog)
    
    # Get indices sorted by mean_dlog (highest first - these prefer dipole most)
    sorted_indices = np.argsort(mean_dlog)[::-1]
    
    # Indices to remove (n_remove highest values)
    remove_indices = sorted_indices[:n_remove]
    
    # Create boolean mask (True = keep, False = remove)
    mask = np.ones(n_clusters, dtype=bool)
    mask[remove_indices] = False
    
    if verbose:
        fprint(f"  Removing {n_remove} cluster(s) with highest dipole preference:")
        for idx in remove_indices:
            redshift = data['zcmb'][idx] if 'zcmb' in data.data else idx
            fprint(f"    Index {idx}: mean_dlog = {mean_dlog[idx]:.4f}, z = {redshift:.4f}")
        fprint(f"  Remaining clusters: {np.sum(mask)} / {n_clusters}")
    
    # Create new data dictionary
    data_filtered = {}
    
    # Iterate through all keys in the data
    for key in data.data.keys():
        value = data.data[key]
        
        # Skip scalar values or special keys
        if not hasattr(value, 'shape') or len(value.shape) == 0:
            data_filtered[key] = value
            continue
        
        # Filter based on which axis has cluster dimension
        if value.shape[0] == n_clusters:
            # First dimension is clusters
            data_filtered[key] = value[mask]
        elif len(value.shape) >= 2 and value.shape[1] == n_clusters:
            # Second dimension is clusters
            data_filtered[key] = value[:, mask]
        elif len(value.shape) >= 3 and value.shape[2] == n_clusters:
            # Third dimension is clusters
            data_filtered[key] = value[:, :, mask]
        else:
            # No cluster dimension, keep as is
            data_filtered[key] = value
    
    return data_filtered, mask, remove_indices


def save_filtered_mock(data_filtered, mock_path_original, mock_path_new, 
                       removed_indices, iteration):
    """
    Save filtered mock data to a new HDF5 file.
    
    Parameters
    ----------
    data_filtered : dict
        Filtered data dictionary
    mock_path_original : str
        Path to original mock file (to copy attributes)
    mock_path_new : str
        Path to save new filtered mock
    removed_indices : array
        Indices of removed clusters
    iteration : int
        Iteration number
    """
    fprint(f"  Saving filtered mock to {mock_path_new}")
    
    with File(mock_path_new, 'w') as f_new, File(mock_path_original, 'r') as f_orig:
        grp = f_new.create_group("mock")
        
        # Copy filtered data
        for key, value in data_filtered.items():
            if isinstance(value, np.ndarray):
                grp.create_dataset(key, data=value, dtype=np.float32)
        
        # Copy and update attributes
        for key, value in f_orig['mock'].attrs.items():
            grp.attrs[key] = value
        
        grp.attrs['iteration'] = iteration
        grp.attrs['n_clusters_removed'] = len(removed_indices)
        grp.attrs['removed_indices'] = removed_indices


def main():
    parser = argparse.ArgumentParser(
        description='MPI-based mock cluster analysis with iterative cluster removal'
    )
    parser.add_argument('--config_nodipole', type=str, default='scripts/cluster_runs/mock_cluster_nodipole.toml',
                        help='Path to config file for model WITHOUT dipole (default: scripts/cluster_runs/mock_cluster_nodipole.toml)')
    parser.add_argument('--config_dipole', type=str, default='scripts/cluster_runs/mock_cluster_dipole.toml',
                        help='Path to config file for model WITH dipole (default: scripts/cluster_runs/mock_cluster_dipole.toml)')
    parser.add_argument('--field_density', type=str, default=os.path.expanduser('~/code/CANDEL/data/fields/carrick2015_twompp_density.npy'),
                        help='Path to field density file (default: ~/code/CANDEL/data/fields/carrick2015_twompp_density.npy)')
    parser.add_argument('--field_velocity', type=str, default=os.path.expanduser('~/code/CANDEL/data/fields/carrick2015_twompp_velocity.npy'),
                        help='Path to field velocity file (default: ~/code/CANDEL/data/fields/carrick2015_twompp_velocity.npy)')
    parser.add_argument('--nclusters', type=int, default=275,
                        help='Number of clusters per mock (default: 275)')
    parser.add_argument('--n_mocks_total', type=int, default=None,
                        help='Total number of mocks to generate (distributed across ranks). If None, generates 1 mock per rank.')
    parser.add_argument('--n_remove_iterations', type=int, default=10,
                        help='Number of cluster removal iterations (default: 10)')
    parser.add_argument('--n_remove_per_iteration', type=int, default=1,
                        help='Number of clusters to remove per iteration (default: 1)')
    parser.add_argument('--output_dir', type=str, 
                        default='results/mock_cluster_removal',
                        help='Output directory for results (default: results/mock_cluster_removal)')
    parser.add_argument('--seed_offset', type=int, default=1000,
                        help='Seed offset (seed = offset + mock_id, default: 1000)')
    parser.add_argument('--dipole_only', action='store_true',
                        help='Only run dipole inference on full mock (skip no-dipole and removal)')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples for inference runs (default: 500)')
    
    args = parser.parse_args()
    
    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        fprint(f"Results will be saved to: {args.output_dir}")
        fprint(f"Running with {size} MPI ranks")
    
    if HAS_MPI:
        comm.Barrier()
    
    # Determine which mocks this rank will generate
    if args.n_mocks_total is not None:
        # Distribute n_mocks_total across ranks
        mocks_per_rank = args.n_mocks_total // size
        remainder = args.n_mocks_total % size
        
        # Start and end mock IDs for this rank
        if rank < remainder:
            # First 'remainder' ranks get one extra mock
            start_mock_id = rank * (mocks_per_rank + 1)
            n_mocks_this_rank = mocks_per_rank + 1
        else:
            start_mock_id = rank * mocks_per_rank + remainder
            n_mocks_this_rank = mocks_per_rank
        
        mock_ids = list(range(start_mock_id, start_mock_id + n_mocks_this_rank))
        fprint(f"DEBUG: n_mocks_total={args.n_mocks_total}, size={size}, rank={rank}, mocks_per_rank={mocks_per_rank}, remainder={remainder}, start_mock_id={start_mock_id}, n_mocks_this_rank={n_mocks_this_rank}, mock_ids={mock_ids}")
        if mock_ids:
            fprint(f"Generating {n_mocks_this_rank} mocks: IDs {mock_ids[0]}-{mock_ids[-1]}")
        else:
            fprint("No mocks assigned to this rank.")
    else:
        # Default: 1 mock per rank, ID = rank
        mock_ids = [rank]
        fprint(f"Generating 1 mock: ID {rank}")
    
    # Load field reconstruction (Carrick2015)
    fprint(f"Loading Carrick2015 field reconstruction...")
    fprint(f"  Density: {args.field_density}")
    fprint(f"  Velocity: {args.field_velocity}")
    field_loader = candel.field.name2field_loader("Carrick2015")(
        path_density=args.field_density,
        path_velocity=args.field_velocity
    )
    
    # Create mock directory (shared across all mocks, no rank-specific dir)
    mock_dir = os.path.abspath(join(args.output_dir, "mocks"))
    os.makedirs(mock_dir, exist_ok=True)
    
    # Loop over mocks assigned to this rank
    for mock_id in mock_ids:
        fprint(f"\n{'='*60}")
        fprint(f"MOCK {mock_id} - Starting analysis")
        fprint(f"{'='*60}")
        
        # Generate mock with unique seed
        seed = args.seed_offset + mock_id
        mock, mock_path = generate_mock(args.nclusters, seed, field_loader, mock_dir, mock_id=mock_id)
    
        fprint(f"Starting cluster removal analysis...")
        fprint(f"  Initial clusters: {args.nclusters}")
        fprint(f"  Removal iterations: {args.n_remove_iterations}")
        fprint(f"  Clusters per iteration: {args.n_remove_per_iteration}")
        
        # If dipole_only flag is set, just run dipole inference and exit
        if args.dipole_only:
            fprint(f"\n{'='*60}")
            fprint(f"DIPOLE ONLY MODE: Running dipole inference on full catalogue")
            fprint(f"{'='*60}")
            
            lp_dipole_full, output_dipole, data_dipole = run_inference_on_mock(
                args.config_dipole, mock_dir, mock_id, args.output_dir,
                args.field_density, args.field_velocity,
                temp_suffix="_dipole_full", verbose_name="Dipole (full)",
                num_samples=args.num_samples
            )
            
            fprint(f"\n{'='*60}")
            fprint(f"MOCK {mock_id} - Dipole-only analysis completed!")
            fprint(f"{'='*60}")
            continue  # Move to next mock
        
        # ===================================================================
        # STEP 1: Run no-dipole inference ONCE on full mock
        # ===================================================================
        fprint(f"\n{'='*60}")
        fprint(f"INITIAL: Running no-dipole inference on full catalogue")
        fprint(f"{'='*60}")
        
        lp_nodipole_full, output_nodipole, data_full = run_inference_on_mock(
            args.config_nodipole, mock_dir, mock_id, args.output_dir,
            args.field_density, args.field_velocity,
            temp_suffix="_nodipole", verbose_name="No-dipole",
            num_samples=args.num_samples
        )
        
        # Save initial no-dipole results
        result_file_nodipole = join(args.output_dir, f"mock_{mock_id:04d}_nodipole_full.hdf5")
        fprint(f"  Saving no-dipole results to {result_file_nodipole}")
        
        with File(result_file_nodipole, 'w') as f:
            f.create_dataset('lp_nodipole', data=lp_nodipole_full)
            f.attrs['n_clusters'] = len(data_full['RA'])
            f.attrs['mock_id'] = mock_id
            
            # Save posterior samples
            grp_nodipole = f.create_group('posterior_nodipole')
            for key, value in output_nodipole[0].items():
                if isinstance(value, np.ndarray):
                    grp_nodipole.create_dataset(key, data=value)
        
        # ===================================================================
        # STEP 2: Run dipole inference ONCE on full mock
        # ===================================================================
        fprint(f"\n{'='*60}")
        fprint(f"Running dipole inference on full catalogue")
        fprint(f"{'='*60}")
        
        lp_dipole_full, output_dipole, data_dipole = run_inference_on_mock(
            args.config_dipole, mock_dir, mock_id, args.output_dir,
            args.field_density, args.field_velocity,
            temp_suffix="_dipole_full", verbose_name="Dipole (full)",
            num_samples=args.num_samples
        )
        
        # ===================================================================
        # STEP 3: Compute mean_dlog ONCE from full samples
        # ===================================================================
        fprint(f"\n{'='*60}")
        fprint(f"Computing log density statistics on full sample")
        fprint(f"{'='*60}")
        fprint(f"  lp_dipole_full shape: {lp_dipole_full.shape}")
        fprint(f"  lp_nodipole_full shape: {lp_nodipole_full.shape}")
        
        mean_dlog, std_dlog = candel.get_dlog_density_stats(lp_dipole_full, lp_nodipole_full)
        fprint(f"  mean_dlog range: [{mean_dlog.min():.4f}, {mean_dlog.max():.4f}]")
        fprint(f"  mean_dlog mean: {mean_dlog.mean():.4f} ± {mean_dlog.std():.4f}")
        
        # Get sorted indices (highest mean_dlog first - these prefer dipole most)
        sorted_indices = np.argsort(mean_dlog)[::-1]
        
        # ===================================================================
        # STEP 4: Iteratively create filtered mocks and run dipole inference
        # ===================================================================
        
        for iteration in range(1, args.n_remove_iterations + 1):
            fprint(f"\n{'='*60}")
            fprint(f"ITERATION {iteration} - Removing top {iteration * args.n_remove_per_iteration} cluster(s)")
            fprint(f"{'='*60}")
            
            # Total clusters to remove in this iteration
            n_remove_total = iteration * args.n_remove_per_iteration
            remove_indices = sorted_indices[:n_remove_total]
            
            # Create boolean mask (True = keep, False = remove)
            n_clusters = len(mean_dlog)
            mask = np.ones(n_clusters, dtype=bool)
            mask[remove_indices] = False
            
            fprint(f"  Removing {n_remove_total} cluster(s) with highest dipole preference")
            fprint(f"  Top removed indices: {remove_indices[:min(5, len(remove_indices))]}")
            fprint(f"  Remaining clusters: {np.sum(mask)} / {n_clusters}")
            
            # Filter the original mock dict
            # Read the original mock file
            with File(mock_path, 'r') as f_orig:
                mock_filtered = {}
                for key in f_orig['mock'].keys():
                    data = f_orig['mock'][key][...]
                    # Filter if this is a per-cluster array
                    # Check which axis has the cluster dimension
                    if len(data.shape) == 0:
                        # Scalar
                        mock_filtered[key] = data
                    elif data.shape[0] == n_clusters:
                        # First dimension is clusters
                        mock_filtered[key] = data[mask]
                    elif len(data.shape) >= 2 and data.shape[1] == n_clusters:
                        # Second dimension is clusters (e.g., los_density: (1, n_clusters, n_points))
                        mock_filtered[key] = data[:, mask]
                    elif len(data.shape) >= 3 and data.shape[2] == n_clusters:
                        # Third dimension is clusters
                        mock_filtered[key] = data[:, :, mask]
                    else:
                        # No cluster dimension, keep as is
                        mock_filtered[key] = data
            
            # Save filtered mock (use format without underscore before iter so loader parses correctly)
            filtered_mock_path = join(mock_dir, f"mock_{mock_id:04d}iter{iteration:02d}.hdf5")
            fprint(f"  Saving filtered mock to {filtered_mock_path}")
            
            with File(filtered_mock_path, 'w') as f_new:
                grp = f_new.create_group("mock")
                for key, value in mock_filtered.items():
                    if isinstance(value, np.ndarray):
                        grp.create_dataset(key, data=value, dtype=np.float32)
                
                # Copy attributes from original
                with File(mock_path, 'r') as f_orig:
                    for key, value in f_orig['mock'].attrs.items():
                        grp.attrs[key] = value
                
                grp.attrs['iteration'] = iteration
                grp.attrs['n_clusters_removed'] = n_remove_total
                grp.attrs['removed_indices'] = remove_indices
            
            # Update config to point to this filtered mock
            # The loader expects name.split("_")[-1] as index, so use format without underscore
            # Clusters_mock_0000iter01 -> index = "0000iter01" -> mock_0000iter01.hdf5
            filtered_mock_name = f"Clusters_mock_{mock_id:04d}iter{iteration:02d}"
            filtered_mock_file = join(mock_dir, f"mock_{mock_id:04d}iter{iteration:02d}.hdf5")
            
            # Run dipole inference on filtered mock
            lp_dipole_filtered, output_dipole_filtered, data_filtered_loaded = run_inference_on_mock(
                args.config_dipole, mock_dir, mock_id, args.output_dir,
                args.field_density, args.field_velocity,
                temp_suffix=f"_dipole_iter{iteration:02d}",
                verbose_name=f"Dipole (iter {iteration})",
                mock_name=filtered_mock_name,
                num_samples=args.num_samples
            )
            
            # Save summary results for this iteration
            result_file = join(args.output_dir, f"mock_{mock_id:04d}_iter_{iteration:02d}.hdf5")
            fprint(f"  Saving summary results to {result_file}")
            
            with File(result_file, 'w') as f:
                f.create_dataset('mean_dlog_full', data=mean_dlog)
                f.create_dataset('removed_indices', data=remove_indices)
                f.attrs['iteration'] = iteration
                f.attrs['n_clusters_removed'] = n_remove_total
                f.attrs['n_clusters_remaining'] = np.sum(mask)
                f.attrs['mock_id'] = mock_id
        
        fprint(f"\n{'='*60}")
        fprint(f"MOCK {mock_id} - Analysis complete")
        fprint(f"{'='*60}")
    
    if HAS_MPI:
        comm.Barrier()
    
    if rank == 0:
        fprint(f"\nAll ranks completed successfully!")
        fprint(f"Results from {size} mocks saved to {args.output_dir}")


if __name__ == "__main__":
    main()
