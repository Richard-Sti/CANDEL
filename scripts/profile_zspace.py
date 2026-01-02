"""Profile the z-space likelihood to identify bottlenecks."""
import time
import jax
import jax.numpy as jnp
from jax import random

# Ensure JAX uses 64-bit floats
from jax import config
config.update("jax_enable_x64", True)

from candel import load_config
from candel.model.model import ClustersModel, compute_Vext_radial, compute_los_zspace_to_rspace
from candel.pvdata.data import load_PV_dataframes
from candel.cosmography import Redshift2Distance

# Print JAX device info
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")


def profile_component(name, fn, *args, n_runs=10, **kwargs):
    """Profile a function and return timing stats."""
    # Warmup / JIT compile
    result = fn(*args, **kwargs)
    jax.block_until_ready(result)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)

    avg = sum(times) / len(times)
    print(f"{name}: {avg*1000:.2f} ms (avg of {n_runs} runs)")
    return result, avg


def main():
    config_path = "results/radtest/2mpp_zspace_galaxies_LTYT_noMNR_radmagVext-finest_hasY.toml"

    print("Loading config and model...")
    cfg = load_config(config_path, replace_none=False, replace_los_prior=False)

    # Override paths for local
    local_root = "/Users/yasin/code/CANDEL"

    print("Loading data...")
    data_list = load_PV_dataframes(config_path, local_root=local_root)
    data = data_list[0]  # Use first catalogue (Clusters_hasY)

    print(f"Number of galaxies: {len(data)}")
    print(f"LOS grid size: {data['los_r'].shape}")

    print("\nCreating model...")
    model = ClustersModel(config_path)

    # Get model settings
    which_Vext = model.which_Vext
    kwargs_Vext = model.kwargs_Vext
    n_iterations = model.n_zspace_iterations

    print(f"which_Vext: {which_Vext}")
    print(f"n_zspace_iterations: {n_iterations}")
    print(f"rknot: {kwargs_Vext.get('rknot', 'N/A')}")

    # Create test inputs
    los_r = data["los_r"]
    z_grid = data["los_z"]
    n_gal = len(data)
    n_los = len(los_r)

    # Sample random Vext parameters (8 knots for finest)
    key = random.PRNGKey(42)
    n_knots = len(kwargs_Vext["rknot"])
    Vext_mag = random.uniform(key, (n_knots,), minval=-1000, maxval=1000)

    # Random direction
    key, subkey = random.split(key)
    direction = random.normal(subkey, (3,))
    direction = direction / jnp.linalg.norm(direction)

    Vext = (Vext_mag, direction)

    # Create redshift2distance
    redshift2distance = Redshift2Distance(Om0=0.3)
    h = 1.0

    print("\n--- Profiling Components ---\n")

    # 1. Profile compute_Vext_radial with 1D r_grid
    print("1. compute_Vext_radial (1D r_grid):")
    _, t1 = profile_component(
        "   1D r_grid",
        lambda: compute_Vext_radial(data, los_r, Vext, which_Vext, **kwargs_Vext)
    )

    # 2. Profile compute_Vext_radial with 2D r_grid (simulating z-space iteration)
    print("\n2. compute_Vext_radial (2D r_grid - n_gal x n_los):")
    r_grid_2d = jnp.broadcast_to(los_r[None, :], (n_gal, n_los))
    _, t2 = profile_component(
        "   2D r_grid",
        lambda: compute_Vext_radial(data, r_grid_2d, Vext, which_Vext, **kwargs_Vext)
    )

    # 3. Profile full z-space iteration
    print("\n3. Full z-space iteration (compute_los_zspace_to_rspace):")
    _, t3 = profile_component(
        f"   {n_iterations} iterations",
        lambda: compute_los_zspace_to_rspace(
            data, los_r, z_grid, Vext, which_Vext, kwargs_Vext,
            redshift2distance, h, n_iterations=n_iterations
        )
    )

    # 4. Profile just 0 iterations (baseline)
    print("\n4. Z-space with 0 iterations (baseline):")
    _, t4 = profile_component(
        "   0 iterations",
        lambda: compute_los_zspace_to_rspace(
            data, los_r, z_grid, Vext, which_Vext, kwargs_Vext,
            redshift2distance, h, n_iterations=0
        )
    )

    # 5. Profile the interpolation step
    print("\n5. LOS interpolation (vmap over galaxies):")
    r_cosmo = compute_los_zspace_to_rspace(
        data, los_r, z_grid, Vext, which_Vext, kwargs_Vext,
        redshift2distance, h, n_iterations=n_iterations
    )
    r_cosmo_2d = r_cosmo[0]  # (n_gal, n_los)

    # Get original LOS values
    los_delta_orig = data.f_los_delta.interp_many_steps_per_galaxy(los_r)

    r_grid = jnp.linspace(0.1, 1401, 1001)  # model.r_grid equivalent

    def _interp_to_rgrid(los_values, r_cosmo_line):
        return jnp.interp(r_grid, r_cosmo_line, los_values)

    def _interp_field(los_field):
        return jax.vmap(_interp_to_rgrid, in_axes=(0, 0))(los_field, r_cosmo_2d)

    _, t5 = profile_component(
        "   Interp one field",
        lambda: jax.vmap(_interp_field)(los_delta_orig)
    )

    # 6. Profile redshift2distance
    print("\n6. Redshift2Distance conversion:")
    z_test = jnp.linspace(0.001, 0.4, n_los)
    _, t6 = profile_component(
        f"   {n_los} redshifts",
        lambda: redshift2distance(z_test, h=1.0)
    )

    # Full z_cosmo array
    z_cosmo_full = jnp.broadcast_to(z_test[None, None, :], (1, n_gal, n_los))
    _, t7 = profile_component(
        f"   Full array ({1}x{n_gal}x{n_los})",
        lambda: redshift2distance(z_cosmo_full.ravel(), h=1.0).reshape(z_cosmo_full.shape)
    )

    print("\n--- Summary ---")
    print(f"Galaxies: {n_gal}, LOS points: {n_los}, Knots: {n_knots}")
    print(f"Estimated per-iteration cost: {(t3 - t4) / n_iterations * 1000:.2f} ms")
    print(f"Overhead from 2D r_grid: {(t2 - t1) * 1000:.2f} ms")

    # 7. Test effect of reducing iterations
    print("\n7. Effect of reducing iterations:")
    for n_iter in [1, 2, 3, 4]:
        _, t_iter = profile_component(
            f"   {n_iter} iteration(s)",
            lambda n=n_iter: compute_los_zspace_to_rspace(
                data, los_r, z_grid, Vext, which_Vext, kwargs_Vext,
                redshift2distance, h, n_iterations=n
            ),
            n_runs=5
        )

    # 8. Test effect of Malmquist grid size
    print("\n8. ln_simpson integration cost by grid size:")
    from candel.model.simpson import ln_simpson

    for n_points in [251, 501, 1001]:
        r_test = jnp.linspace(0.1, 1401, n_points)
        ll_test = jnp.ones((1, n_gal, n_points))  # (n_field, n_gal, n_rbin)
        _, t_simp = profile_component(
            f"   {n_points} points",
            lambda ll=ll_test, r=r_test: ln_simpson(ll, x=r[None, None, :], axis=-1),
            n_runs=10
        )

    print("\n--- Recommendations ---")
    print("1. Try reducing n_zspace_iterations to 2 (should be sufficient with smoothness prior)")
    print("2. Try reducing num_points_malmquist to 501 (matches LOS grid)")
    print("3. On GPU, JAX XLA will fuse many ops - actual bottleneck may differ")

    # 9. Profile full model likelihood
    print("\n9. Full model likelihood profiling:")
    from numpyro import handlers
    from numpyro.infer.util import log_density

    # Profile on a single dataset for simplicity (avoids shared_params complexity)
    # The joint model overhead is approximately 2x this (for 2 datasets)
    print("   Profiling on single dataset (first catalogue)")

    # Get a trace to extract param values
    def get_trace():
        with handlers.seed(rng_seed=42):
            with handlers.trace() as trace:
                model(data, shared_params=None)
        return trace

    trace = get_trace()

    # Extract parameter values from trace
    param_values = {}
    for name, site in trace.items():
        if site['type'] == 'sample' and not site.get('is_observed', False):
            param_values[name] = site['value']

    print(f"   Number of parameters: {len(param_values)}")

    # 10. Profile gradient computation (this is what HMC actually does)
    print("\n10. Gradient computation (potential energy + grad):")

    def compute_log_density(params):
        def model_fn():
            with handlers.seed(rng_seed=42):
                model(data, shared_params=None)
        ld, _ = log_density(model_fn, (), {}, params)
        return -ld  # potential energy is negative log density

    # JIT compile
    jit_pe = jax.jit(compute_log_density)
    jit_grad_pe = jax.jit(jax.grad(compute_log_density))

    # Warmup
    print("   JIT compiling potential energy...")
    pe = jit_pe(param_values)
    jax.block_until_ready(pe)
    print("   JIT compiling gradient...")
    grad_pe = jit_grad_pe(param_values)
    jax.block_until_ready(grad_pe)

    # Time potential energy (forward pass)
    times_pe = []
    for _ in range(10):
        start = time.perf_counter()
        pe = jit_pe(param_values)
        jax.block_until_ready(pe)
        times_pe.append(time.perf_counter() - start)

    avg_pe = sum(times_pe) / len(times_pe)
    print(f"   Potential energy (forward): {avg_pe*1000:.2f} ms (avg of 10 runs)")

    # Time gradient (what HMC leapfrog does)
    times_grad = []
    for _ in range(10):
        start = time.perf_counter()
        grad_pe = jit_grad_pe(param_values)
        jax.block_until_ready(grad_pe)
        times_grad.append(time.perf_counter() - start)

    avg_grad = sum(times_grad) / len(times_grad)
    print(f"   Gradient computation: {avg_grad*1000:.2f} ms (avg of 10 runs)")

    # 11. Estimate HMC step time
    print("\n11. HMC step time estimate:")
    num_leapfrog = 63  # from your output
    estimated_hmc_step = num_leapfrog * avg_grad
    print(f"   With {num_leapfrog} leapfrog steps: {estimated_hmc_step*1000:.0f} ms per sample")
    print(f"   Expected throughput: {1/estimated_hmc_step:.2f} samples/sec")

    # 12. Breakdown estimate
    print("\n12. Time breakdown estimate:")
    zspace_fraction = (t3 / avg_pe) * 100 if avg_pe > 0 else 0
    print(f"   Z-space iteration: ~{t3*1000:.1f} ms ({zspace_fraction:.0f}% of forward pass)")
    print(f"   Rest of likelihood: ~{(avg_pe - t3)*1000:.1f} ms ({100-zspace_fraction:.0f}% of forward pass)")


if __name__ == "__main__":
    main()
