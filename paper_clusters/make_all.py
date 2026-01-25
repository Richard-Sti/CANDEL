"""Run all plots and tables for the paper."""
import sys
import io
import traceback

import plot_redshift_distribution
import plot_scaling_relations
import plot_whole_posterior
import plot_dipoles
import plot_relation_comparison
import plot_sigma_v
import plot_migkas_comparison
# import plot_mnr_comparison  # MNR results not available
import plot_radial_mag
import plot_reconstruction
import tables
# import plot_dipoles_all
# import plot_base_all


def run_silent(name, func):
    """Run a function, suppressing output unless it fails."""
    print(f"Generating {name}...", end=" ", flush=True)

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_out = io.StringIO()
    sys.stderr = captured_err = io.StringIO()

    try:
        func()
        print("done")
    except Exception:
        print("FAILED")
        # Restore streams before printing
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        # Print captured output
        out = captured_out.getvalue()
        err = captured_err.getvalue()
        if out:
            print(out)
        if err:
            print(err, file=sys.stderr)
        traceback.print_exc()
        return False
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return True


def main():
    tasks = [
        ("redshift distribution plot", plot_redshift_distribution.main),
        ("scaling relations plot", plot_scaling_relations.main),
        ("whole posterior plot", plot_whole_posterior.main),
        ("dipoles reconstructions plots", plot_dipoles.main),
        ("scaling comparison plot", plot_relation_comparison.main),
        ("sigma_v plot", plot_sigma_v.main),
        ("Migkas comparison plot", plot_migkas_comparison.main),
        ("radial magnitude Vext plot", plot_radial_mag.main),
        ("reconstruction plot", plot_reconstruction.main),
        ("results tables", tables.main),
        # ("MNR comparison plot", plot_mnr_comparison.main),
        # ("dipoles all plot", plot_dipoles_all.main),
        # ("base all plot", plot_base_all.main),
    ]

    failed = []
    for name, func in tasks:
        if not run_silent(name, func):
            failed.append(name)

    print()
    if failed:
        print(f"Failed: {', '.join(failed)}")
    else:
        print("All done!")


if __name__ == "__main__":
    main()
