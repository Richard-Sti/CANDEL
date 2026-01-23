#!/usr/bin/env python3
"""
Compare per-cluster likelihood components between two CANDEL runs.

Uses saved likelihood decomposition arrays to compute per-cluster
delta statistics at the MAP point.

Usage:
    python scripts/compare_cluster_likelihood.py \
        --baseline results/radtest/Carrick2015_LT_noMNR.hdf5 \
        --test results/radtest/Carrick2015_LT_noMNR_radmagVext-finest.hdf5 \
        --output results/radtest/comparison_LT.png

Outputs:
    - 4 delta-LL plots vs zcmb (total, lp_dist, ll_cluster, ll_cz)
    - 2 n_sigma plots vs zcmb (redshift and cluster, both models)
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_cluster_zcmb(data_path="data/Clusters"):
    """Load cluster zcmb from the data files."""
    import pandas as pd
    from os.path import join

    # Load the main cluster catalogue (whitespace-separated, skip header)
    fpath = join(data_path, "ClustersData.txt")
    df = pd.read_csv(fpath, sep=r'\s+', skiprows=1, header=None)
    # Column 1 (index 1) is redshift z
    return df.iloc[:, 1].values


def load_likelihood_components(fname):
    """Load all per-cluster likelihood components from HDF5."""
    data = {}
    with h5py.File(fname, 'r') as f:
        # Log density for finding MAP
        if 'log_density' in f:
            data['log_density'] = f['log_density'][:]

        # Per-sample log density
        if 'log_density_per_sample' in f:
            data['log_density_per_sample'] = f['log_density_per_sample'][:]

        # Check samples group for skipZ arrays
        if 'samples' in f:
            samples = f['samples']
            for key in ['r_map_skipZ', 'r_mean_skipZ', 'r_std_skipZ',
                        'lp_dist_skipZ', 'll_cz_skipZ', 'cz_nsigma_skipZ',
                        'll_cluster_skipZ', 'cluster_nsigma_skipZ']:
                if key in samples:
                    data[key] = samples[key][:]

    return data


def get_map_index(data):
    """Get index of MAP sample (highest log_density)."""
    if 'log_density' in data:
        return np.argmax(data['log_density'])
    elif 'log_density_per_sample' in data:
        # Sum over clusters to get total log density per sample
        return np.argmax(np.sum(data['log_density_per_sample'], axis=1))
    else:
        return 0  # Default to first sample


def extract_at_map_distance(arr, r_map_idx):
    """
    Extract values at MAP distance for the MAP sample.

    arr: shape (n_field, n_gal, n_rbin) - already extracted MAP sample
    r_map_idx: shape (n_field, n_gal) - indices into n_rbin

    Returns: shape (n_gal,) after averaging over fields
    """
    n_field, n_gal, n_rbin = arr.shape

    result = np.zeros((n_field, n_gal))
    for j in range(n_field):
        for k in range(n_gal):
            result[j, k] = arr[j, k, r_map_idx[j, k]]

    # Average over fields
    return np.mean(result, axis=0)


def compute_map_statistics(data_A, data_B, label_A="Baseline", label_B="Test"):
    """
    Compute per-cluster statistics at MAP point.

    Returns dict with values for each component.
    Convention: delta = B - A (test - baseline)
    """
    results = {}

    # Get MAP indices
    map_idx_A = get_map_index(data_A)
    map_idx_B = get_map_index(data_B)
    print(f"MAP sample indices: {label_A}={map_idx_A}, {label_B}={map_idx_B}")

    # 1. Total log-density per cluster at MAP
    if 'log_density_per_sample' in data_A and 'log_density_per_sample' in data_B:
        lp_A = data_A['log_density_per_sample'][map_idx_A]  # (n_gal,)
        lp_B = data_B['log_density_per_sample'][map_idx_B]
        delta = lp_B - lp_A  # test - baseline
        results['total'] = {'A': lp_A, 'B': lp_B, 'delta': delta}
        print(f"\nTotal log-density (per cluster at MAP):")
        print(f"  Δ = {np.sum(delta):.2f} (sum), {np.mean(delta):.4f} (mean)")

    # For component arrays, extract at MAP sample and MAP distance
    # Get r_map indices for extracting at correct distance
    r_map_idx_A = None
    r_map_idx_B = None

    if 'lp_dist_skipZ' in data_A:
        # Compute r_map index from total ll = lp_dist + ll_cz + ll_cluster
        lp_dist_A = data_A['lp_dist_skipZ'][map_idx_A]  # (n_field, n_gal, n_rbin)
        ll_cz_A = data_A['ll_cz_skipZ'][map_idx_A]
        ll_cluster_A = data_A['ll_cluster_skipZ'][map_idx_A]
        ll_total_A = lp_dist_A + ll_cz_A + ll_cluster_A
        r_map_idx_A = np.argmax(ll_total_A, axis=-1)  # (n_field, n_gal)

        lp_dist_B = data_B['lp_dist_skipZ'][map_idx_B]
        ll_cz_B = data_B['ll_cz_skipZ'][map_idx_B]
        ll_cluster_B = data_B['ll_cluster_skipZ'][map_idx_B]
        ll_total_B = lp_dist_B + ll_cz_B + ll_cluster_B
        r_map_idx_B = np.argmax(ll_total_B, axis=-1)

    # Extract components at MAP distance
    for key, short_name in [('lp_dist_skipZ', 'lp_dist'),
                             ('ll_cz_skipZ', 'll_cz'),
                             ('ll_cluster_skipZ', 'll_cluster')]:
        if key in data_A and key in data_B and r_map_idx_A is not None:
            arr_A = data_A[key][map_idx_A]  # (n_field, n_gal, n_rbin)
            arr_B = data_B[key][map_idx_B]

            # Extract at MAP distance
            val_A = extract_at_map_distance(arr_A, r_map_idx_A)  # (n_gal,)
            val_B = extract_at_map_distance(arr_B, r_map_idx_B)

            delta = val_B - val_A  # test - baseline
            results[short_name] = {'A': val_A, 'B': val_B, 'delta': delta}
            print(f"\n{short_name}:")
            print(f"  Δ = {np.sum(delta):.2f} (sum), {np.mean(delta):.4f} (mean)")

    # N_sigma at MAP distance
    for key, short_name in [('cz_nsigma_skipZ', 'cz_nsigma'),
                             ('cluster_nsigma_skipZ', 'cluster_nsigma')]:
        if key in data_A and key in data_B and r_map_idx_A is not None:
            arr_A = data_A[key][map_idx_A]  # (n_field, n_gal, n_rbin)
            arr_B = data_B[key][map_idx_B]

            # Extract at MAP distance
            val_A = extract_at_map_distance(arr_A, r_map_idx_A)  # (n_gal,)
            val_B = extract_at_map_distance(arr_B, r_map_idx_B)

            results[short_name] = {'A': val_A, 'B': val_B}
            print(f"\n{short_name}:")
            print(f"  {label_A}: mean n_sigma = {np.mean(val_A):.3f}, rms = {np.sqrt(np.mean(val_A**2)):.3f}")
            print(f"  {label_B}: mean n_sigma = {np.mean(val_B):.3f}, rms = {np.sqrt(np.mean(val_B**2)):.3f}")

    return results


def plot_comparison(results, zcmb, output_path, label_A="Baseline", label_B="Test"):
    """Create 6-panel comparison visualization vs zcmb."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Sort by zcmb for cleaner plots
    sort_idx = np.argsort(zcmb)
    zcmb_sorted = zcmb[sort_idx]

    # Panel 1: Delta total log-density vs zcmb
    ax = axes[0, 0]
    if 'total' in results:
        delta = results['total']['delta'][sort_idx]
        ax.scatter(zcmb_sorted, delta, s=15, alpha=0.7)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('$z_{\\rm CMB}$')
        ax.set_ylabel(f'$\\Delta\\log L$ ({label_B} $-$ {label_A})')
        ax.set_title(f'Total log-density: $\\Sigma$={np.sum(delta):.1f}')
    else:
        ax.text(0.5, 0.5, 'No total log-density data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel 2: Delta lp_dist vs zcmb
    ax = axes[0, 1]
    if 'lp_dist' in results:
        delta = results['lp_dist']['delta'][sort_idx]
        ax.scatter(zcmb_sorted, delta, s=15, alpha=0.7, color='green')
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('$z_{\\rm CMB}$')
        ax.set_ylabel(f'$\\Delta\\log L_{{\\rm dist}}$ ({label_B} $-$ {label_A})')
        ax.set_title(f'Distance prior: $\\Sigma$={np.sum(delta):.1f}')
    else:
        ax.text(0.5, 0.5, 'No lp_dist data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel 3: Delta ll_cluster vs zcmb
    ax = axes[0, 2]
    if 'll_cluster' in results:
        delta = results['ll_cluster']['delta'][sort_idx]
        ax.scatter(zcmb_sorted, delta, s=15, alpha=0.7, color='orange')
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('$z_{\\rm CMB}$')
        ax.set_ylabel(f'$\\Delta\\log L_{{\\rm cluster}}$ ({label_B} $-$ {label_A})')
        ax.set_title(f'Scaling relation: $\\Sigma$={np.sum(delta):.1f}')
    else:
        ax.text(0.5, 0.5, 'No ll_cluster data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel 4: Delta ll_cz vs zcmb
    ax = axes[1, 0]
    if 'll_cz' in results:
        delta = results['ll_cz']['delta'][sort_idx]
        ax.scatter(zcmb_sorted, delta, s=15, alpha=0.7, color='purple')
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('$z_{\\rm CMB}$')
        ax.set_ylabel(f'$\\Delta\\log L_{{cz}}$ ({label_B} $-$ {label_A})')
        ax.set_title(f'Redshift likelihood: $\\Sigma$={np.sum(delta):.1f}')
    else:
        ax.text(0.5, 0.5, 'No ll_cz data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel 5: Redshift n_sigma vs zcmb (both models)
    ax = axes[1, 1]
    if 'cz_nsigma' in results:
        nsig_A = results['cz_nsigma']['A'][sort_idx]
        nsig_B = results['cz_nsigma']['B'][sort_idx]

        ax.scatter(zcmb_sorted, nsig_A, s=15, alpha=0.6, label=label_A, color='blue')
        ax.scatter(zcmb_sorted, nsig_B, s=15, alpha=0.6, label=label_B, color='red')
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('$z_{\\rm CMB}$')
        ax.set_ylabel('$n_\\sigma$ (redshift)')
        ax.set_title(f'Redshift $n_\\sigma$: rms={np.sqrt(np.mean(nsig_A**2)):.1f} vs {np.sqrt(np.mean(nsig_B**2)):.1f}')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No cz_nsigma data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel 6: Cluster n_sigma vs zcmb (both models)
    ax = axes[1, 2]
    if 'cluster_nsigma' in results:
        nsig_A = results['cluster_nsigma']['A'][sort_idx]
        nsig_B = results['cluster_nsigma']['B'][sort_idx]

        ax.scatter(zcmb_sorted, nsig_A, s=15, alpha=0.6, label=label_A, color='blue')
        ax.scatter(zcmb_sorted, nsig_B, s=15, alpha=0.6, label=label_B, color='red')
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('$z_{\\rm CMB}$')
        ax.set_ylabel('$n_\\sigma$ (cluster)')
        ax.set_title(f'Cluster $n_\\sigma$: rms={np.sqrt(np.mean(nsig_A**2)):.1f} vs {np.sqrt(np.mean(nsig_B**2)):.1f}')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No cluster_nsigma data', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare per-cluster likelihood between two CANDEL runs')
    parser.add_argument('--baseline', required=True, help='Baseline HDF5 file')
    parser.add_argument('--test', required=True, help='Test HDF5 file')
    parser.add_argument('--output', required=True, help='Output plot path')
    parser.add_argument('--label-baseline', default='Baseline', help='Label for baseline model')
    parser.add_argument('--label-test', default='Test', help='Label for test model')
    parser.add_argument('--data-path', default='data/Clusters', help='Path to cluster data')
    args = parser.parse_args()

    # Load cluster zcmb
    print(f"Loading cluster zcmb from {args.data_path}")
    zcmb = load_cluster_zcmb(args.data_path)
    print(f"  Found {len(zcmb)} clusters, zcmb range: [{zcmb.min():.4f}, {zcmb.max():.4f}]")

    print(f"\nLoading baseline: {args.baseline}")
    data_A = load_likelihood_components(args.baseline)
    print(f"  Keys: {list(data_A.keys())}")

    print(f"\nLoading test: {args.test}")
    data_B = load_likelihood_components(args.test)
    print(f"  Keys: {list(data_B.keys())}")

    # Verify n_gal matches
    n_gal = len(zcmb)

    print(f"\nComputing MAP statistics for {n_gal} clusters...")
    results = compute_map_statistics(data_A, data_B,
                                      label_A=args.label_baseline,
                                      label_B=args.label_test)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if 'total' in results:
        delta = results['total']['delta']
        print(f"Clusters preferring {args.label_baseline} (Δ < 0): {np.sum(delta < 0)}")
        print(f"Clusters preferring {args.label_test} (Δ > 0): {np.sum(delta > 0)}")

        # Find most significant differences
        top_idx = np.argsort(np.abs(delta))[-5:][::-1]
        print(f"\nTop 5 largest |Δ| (cluster indices, zcmb, Δ):")
        for i, idx in enumerate(top_idx):
            print(f"  {i+1}. Cluster {idx}: zcmb={zcmb[idx]:.4f}, Δ = {delta[idx]:+.3f}")

    plot_comparison(results, zcmb, args.output,
                    label_A=args.label_baseline,
                    label_B=args.label_test)


if __name__ == "__main__":
    main()
