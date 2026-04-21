"""
NGC4258 Mode 1 NUTS: sample global params + per-spot r_ang,
marginalize phi via dense 100k uniform grid on GPU.

No eccentricity, no quadratic warp (for now).

Usage:
    python scripts/megamaser/run_n4258_mode1.py [--num-warmup 2000] [--num-samples 2000]
"""
import argparse
import sys
import os
import time
import tempfile
import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = ""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, init_to_value
import tomli
import tomli_w

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from candel.model.model_H0_maser import MaserDiskModel, PC_PER_MAS_MPC
from candel.pvdata.megamaser_data import load_megamaser_spots
import optax
from numpyro.infer.util import initialize_model, log_density

print(f"JAX platform: {jax.default_backend()}", flush=True)
print(f"JAX devices: {jax.devices()}", flush=True)
numpyro.set_host_device_count(1)

CONFIG_PATH = "scripts/megamaser/config_maser.toml"
GALAXY = "NGC4258"

# Physics constants
C_v = 2978.8656
C_a = 1.872e3
C_g = 1.974e-4
SPEED_OF_LIGHT = 299792.458
LOG_2PI = float(np.log(2 * np.pi))

# Phi grid (module-level constant for JIT)
N_PHI = 100001
PHI_GRID = jnp.linspace(0.0, 2 * jnp.pi, N_PHI)
DPHI = float(PHI_GRID[1] - PHI_GRID[0])
_log_w = np.full(N_PHI, np.log(DPHI))
_log_w[0] = np.log(DPHI / 2)
_log_w[-1] = np.log(DPHI / 2)
LOG_W_PHI = jnp.array(_log_w)

SIN_PHI = jnp.sin(PHI_GRID)
COS_PHI = jnp.cos(PHI_GRID)


# -----------------------------------------------------------------------
# Phi-marginalized logL (vectorized over all spots)
# -----------------------------------------------------------------------

def phi_marginal(r_ang, x_obs, y_obs, var_x, var_y,
                 v_obs, var_v, a_obs, var_a, has_accel,
                 x0, y0, D_A, M_BH, v_sys,
                 r_ang_ref, i0_rad, di_dr_rad, Omega0_rad, dOmega_dr_rad):
    """Phi-marginalized logL for all spots. Returns (n_spots,)."""
    dr = r_ang - r_ang_ref
    i_r = i0_rad + di_dr_rad * dr
    Om_r = Omega0_rad + dOmega_dr_rad * dr

    sin_i = jnp.sin(i_r)[:, None]
    cos_i = jnp.cos(i_r)[:, None]
    sin_O = jnp.sin(Om_r)[:, None]
    cos_O = jnp.cos(Om_r)[:, None]
    r = r_ang[:, None]

    sp = SIN_PHI[None, :]
    cp = COS_PHI[None, :]

    X = x0 + r * (sp * sin_O - cp * cos_O * cos_i)
    Y = y0 + r * (sp * cos_O + cp * sin_O * cos_i)

    rD = r * D_A
    v_kep = C_v * jnp.sqrt(M_BH / rD)
    beta = v_kep / SPEED_OF_LIGHT
    gamma = 1.0 / jnp.sqrt(1.0 - beta * beta)
    zpg = 1.0 / jnp.sqrt(1.0 - C_g * M_BH / rD)
    v_z = v_kep * sp * sin_i
    V = SPEED_OF_LIGHT * (
        gamma * (1.0 + v_z / SPEED_OF_LIGHT) * zpg
        * (1.0 + v_sys / SPEED_OF_LIGHT) - 1.0)

    chi2 = ((x_obs[:, None] - X) ** 2 / var_x[:, None]
            + (y_obs[:, None] - Y) ** 2 / var_y[:, None]
            + (v_obs[:, None] - V) ** 2 / var_v[:, None])

    # Acceleration: masked by has_accel (1 for spots with a measurement)
    A = C_a * M_BH / (r ** 2 * D_A ** 2) * cp * sin_i
    chi2 = chi2 + ((a_obs[:, None] - A) ** 2 / var_a[:, None]
                   * has_accel[:, None])

    lnorm = -0.5 * (3 * LOG_2PI + jnp.log(var_x) + jnp.log(var_y)
                     + jnp.log(var_v))
    lnorm = lnorm - 0.5 * (LOG_2PI + jnp.log(var_a)) * has_accel

    log_integrand = lnorm[:, None] - 0.5 * chi2
    return logsumexp(log_integrand + LOG_W_PHI[None, :], axis=1)


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_data():
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomli.load(f)
    gcfg = cfg["model"]["galaxies"][GALAXY]

    data = load_megamaser_spots(
        cfg["io"]["maser_data"]["root"], galaxy=GALAXY,
        v_sys_obs=gcfg["v_sys_obs"])

    # Build model just to extract arrays
    cfg_copy = cfg.copy()
    cfg_copy["model"] = cfg["model"].copy()
    cfg_copy["model"]["galaxy"] = GALAXY
    for key in ("D_lo", "D_hi"):
        if key in gcfg:
            data[key] = float(gcfg[key])
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(cfg_copy, tmp)
    tmp.close()
    model = MaserDiskModel(tmp.name, data)
    os.unlink(tmp.name)

    # Extract arrays
    has_accel = np.asarray(model._all_has_accel)

    sigma_a2 = np.array(model._all_sigma_a2, dtype=np.float64, copy=True)
    sigma_a2[~has_accel] = 1e30  # huge so chi2_a ≈ 0

    conv_DA_est = float(gcfg.get("v_sys_obs", 667.0)) / 73.0
    r_min = model._R_phys_lo / (conv_DA_est * PC_PER_MAS_MPC)
    r_max = model._R_phys_hi / (conv_DA_est * PC_PER_MAS_MPC)

    d = dict(
        x_obs=jnp.array(model._all_x),
        y_obs=jnp.array(model._all_y),
        sigma_x2=jnp.array(model._all_sigma_x2),
        sigma_y2=jnp.array(model._all_sigma_y2),
        v_obs=jnp.array(model._all_v),
        sigma_v2=jnp.array(model._all_sigma_v2),
        a_obs=jnp.array(model._all_a),
        sigma_a2=jnp.array(sigma_a2),
        has_accel=jnp.array(has_accel),
        is_hv=jnp.array(model.is_highvel),
        n_spots=model.n_spots,
        v_sys_obs=float(gcfg["v_sys_obs"]),
        r_ang_ref=float(gcfg.get("r_ang_ref", 0.0)),
        r_min=float(r_min),
        r_max=float(r_max),
        D_lo=float(gcfg["D_lo"]),
        D_hi=float(gcfg["D_hi"]),
        init=gcfg["init"],
    )
    print(f"Loaded {d['n_spots']} spots, r_ang in [{r_min:.3f}, {r_max:.3f}] mas",
          flush=True)
    return d


# -----------------------------------------------------------------------
# NumPyro model
# -----------------------------------------------------------------------

def maser_model(data):
    n = data["n_spots"]

    # --- Global parameters ---
    D_c = numpyro.sample("D_c", dist.Uniform(data["D_lo"], data["D_hi"]))
    numpyro.factor("D_c_vol", 2 * jnp.log(D_c))  # volume prior

    eta = numpyro.sample("eta", dist.Uniform(3.0, 9.0))
    dv_sys = numpyro.sample("dv_sys", dist.Normal(0.0, 300.0))
    x0 = numpyro.sample("x0", dist.Normal(0.0, 500.0))  # μas
    y0 = numpyro.sample("y0", dist.Normal(0.0, 500.0))
    i0 = numpyro.sample("i0", dist.Uniform(60.0, 110.0))
    di_dr = numpyro.sample("di_dr", dist.Uniform(-30.0, 30.0))
    Omega0 = numpyro.sample("Omega0", dist.Uniform(0.0, 360.0))
    dOmega_dr = numpyro.sample("dOmega_dr", dist.Uniform(-30.0, 30.0))
    sigma_x_floor = numpyro.sample(
        "sigma_x_floor", dist.TruncatedNormal(10.0, 5.0, low=0.0, high=100.0))
    sigma_y_floor = numpyro.sample(
        "sigma_y_floor", dist.TruncatedNormal(10.0, 5.0, low=0.0, high=100.0))
    sigma_v_sys = numpyro.sample(
        "sigma_v_sys", dist.TruncatedNormal(2.0, 1.0, low=0.0, high=100.0))
    sigma_v_hv = numpyro.sample(
        "sigma_v_hv", dist.TruncatedNormal(2.0, 1.0, low=0.0, high=100.0))
    sigma_a_floor = numpyro.sample(
        "sigma_a_floor", dist.TruncatedNormal(0.3, 0.15, low=0.0, high=0.75))

    # --- Per-spot r_ang (uniform prior) ---
    r_ang = numpyro.sample(
        "r_ang", dist.Uniform(data["r_min"], data["r_max"]).expand([n]))

    # --- Derived quantities ---
    D_A = D_c  # approx for z ~ 0.002
    M_BH = jnp.power(10.0, eta + jnp.log10(D_A) - 7.0)
    v_sys = data["v_sys_obs"] + dv_sys

    # Unit conversions
    i0_rad = jnp.deg2rad(i0)
    di_dr_rad = jnp.deg2rad(di_dr)
    Omega0_rad = jnp.deg2rad(Omega0)
    dOmega_dr_rad = jnp.deg2rad(dOmega_dr)
    sigma_x_floor2 = sigma_x_floor ** 2
    sigma_y_floor2 = sigma_y_floor ** 2

    # Variances per spot
    var_x = data["sigma_x2"] + sigma_x_floor2
    var_y = data["sigma_y2"] + sigma_y_floor2
    var_v = data["sigma_v2"] + jnp.where(
        data["is_hv"], sigma_v_hv ** 2, sigma_v_sys ** 2)
    var_a = data["sigma_a2"] + sigma_a_floor ** 2

    # --- Phi-marginalized likelihood ---
    ll = phi_marginal(
        r_ang, data["x_obs"], data["y_obs"], var_x, var_y,
        data["v_obs"], var_v, data["a_obs"], var_a, data["has_accel"],
        x0, y0, D_A, M_BH, v_sys,
        data["r_ang_ref"], i0_rad, di_dr_rad, Omega0_rad, dOmega_dr_rad)

    numpyro.factor("logL", jnp.sum(ll))


# -----------------------------------------------------------------------
# Initial r_ang estimates
# -----------------------------------------------------------------------

def estimate_r_ang(data):
    """Estimate r_ang per spot from velocity (HV) or position (sys)."""
    init = data["init"]
    D_A = float(init["D_c"])  # approx
    M_BH = 10.0**(float(init["eta"]) + np.log10(D_A) - 7.0)
    v_sys = data["v_sys_obs"] + float(init["dv_sys"])
    sin_i = np.sin(np.deg2rad(float(init["i0"])))
    x0 = float(init["x0"])
    y0 = float(init["y0"])

    v_obs = np.asarray(data["v_obs"])
    is_hv = np.asarray(data["is_hv"])
    x_obs = np.asarray(data["x_obs"])
    y_obs = np.asarray(data["y_obs"])

    n = data["n_spots"]
    r_est = np.full(n, np.sqrt(data["r_min"] * data["r_max"]))  # fallback

    # HV spots: r ≈ C_v² M sin²i / (D_A dv²)
    dv = np.abs(v_obs - v_sys)
    dv = np.maximum(dv, 1.0)  # avoid div by zero
    r_hv = C_v**2 * M_BH * sin_i**2 / (dv**2 * D_A)
    r_est[is_hv] = r_hv[is_hv]

    # Systemic spots: r ≈ |position offset|
    r_pos = np.sqrt((x_obs - x0)**2 + (y_obs - y0)**2)
    r_pos = np.maximum(r_pos, data["r_min"])
    r_est[~is_hv] = r_pos[~is_hv]

    # Clip to bounds
    r_est = np.clip(r_est, data["r_min"] * 1.01, data["r_max"] * 0.99)
    return r_est


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-warmup", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-map", action="store_true",
                        help="Skip Sobol+Adam, use config init directly")
    parser.add_argument("--map-only", action="store_true")
    args = parser.parse_args()

    data = load_data()
    init = data["init"]
    n = data["n_spots"]

    # --- Initial values from config ---
    r_est = estimate_r_ang(data)
    init_values = dict(
        D_c=float(init["D_c"]),
        eta=float(init["eta"]),
        dv_sys=float(init["dv_sys"]),
        x0=float(init["x0"]),
        y0=float(init["y0"]),
        i0=float(init["i0"]),
        di_dr=float(init["di_dr"]),
        Omega0=float(init["Omega0"]),
        dOmega_dr=float(init["dOmega_dr"]),
        sigma_x_floor=float(init["sigma_x_floor"]),
        sigma_y_floor=float(init["sigma_y_floor"]),
        sigma_v_sys=float(init["sigma_v_sys"]),
        sigma_v_hv=float(init["sigma_v_hv"]),
        sigma_a_floor=float(init["sigma_a_floor"]),
        r_ang=r_est,
    )

    print(f"\nInit values (global):", flush=True)
    for k, v in init_values.items():
        if k != "r_ang":
            print(f"  {k}: {v}", flush=True)
    print(f"  r_ang: median={np.median(r_est):.3f}, "
          f"range=[{r_est.min():.3f}, {r_est.max():.3f}]", flush=True)

    # --- Adam MAP finding from config init ---
    if not args.skip_map:
        print(f"\n--- Adam MAP optimization from config init ---", flush=True)

        # Initialize model to get unconstrained params and potential_fn
        model_info = initialize_model(
            jax.random.PRNGKey(args.seed), maser_model,
            model_args=(data,),
            init_strategy=init_to_value(values=init_values))
        init_params_unc = model_info.param_info.z
        potential_fn = model_info.potential_fn

        # Evaluate initial potential
        pe_init = float(potential_fn(init_params_unc))
        print(f"  Init potential energy: {pe_init:.2f} "
              f"(logp = {-pe_init:.2f})", flush=True)

        # Adam optimization in unconstrained space
        n_steps = 5000
        lr_schedule = optax.cosine_onecycle_schedule(
            transition_steps=n_steps, peak_value=0.05,
            pct_start=0.1, div_factor=10, final_div_factor=100)
        optimizer = optax.adam(lr_schedule)

        @jax.jit
        def adam_step(params, opt_state):
            loss, grads = jax.value_and_grad(potential_fn)(params)
            updates, new_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state, loss

        opt_state = optimizer.init(init_params_unc)
        params = init_params_unc

        t0 = time.perf_counter()
        best_loss = pe_init
        best_params_unc = params

        for step in range(n_steps):
            params, opt_state, loss = adam_step(params, opt_state)
            loss = float(loss)
            if loss < best_loss:
                best_loss = loss
                best_params_unc = params
            if step % 500 == 0 or step == n_steps - 1:
                print(f"  step {step:>5d}: PE = {loss:.2f} "
                      f"(best = {best_loss:.2f})", flush=True)

        t_adam = time.perf_counter() - t0
        print(f"  Adam completed in {t_adam:.1f}s", flush=True)
        print(f"  Best logp = {-best_loss:.2f}", flush=True)

        # Extract constrained params from best unconstrained
        from numpyro.infer.util import constrain_fn
        transforms = model_info.param_info.transforms if hasattr(model_info.param_info, 'transforms') else None

        # Use the best unconstrained params for NUTS init
        init_params_unc = best_params_unc

        if args.map_only:
            print("\n--map-only: stopping after MAP.", flush=True)
            return

    # --- NUTS ---
    print(f"\n--- NUTS ({args.num_warmup} warmup, {args.num_samples} samples) ---",
          flush=True)

    init_strategy = init_to_value(values=init_values)

    kernel = NUTS(
        maser_model,
        init_strategy=init_strategy,
        dense_mass=[
            ("D_c", "eta", "dv_sys", "i0", "di_dr",
             "Omega0", "dOmega_dr", "x0", "y0"),
            ("sigma_x_floor", "sigma_y_floor", "sigma_v_sys",
             "sigma_v_hv", "sigma_a_floor"),
        ],
        max_tree_depth=10,
    )

    mcmc = MCMC(kernel, num_warmup=args.num_warmup,
                num_samples=args.num_samples, num_chains=1)

    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(args.seed), data)
    t_nuts = time.perf_counter() - t0

    print(f"\nNUTS completed in {t_nuts:.1f}s "
          f"({t_nuts / args.num_samples:.2f}s/sample)", flush=True)

    # --- Summary ---
    mcmc.print_summary(exclude_deterministic=True)

    samples = mcmc.get_samples()

    # Key result: D_c posterior
    D_c = np.asarray(samples["D_c"])
    print(f"\n--- Key result ---", flush=True)
    print(f"D_c = {D_c.mean():.3f} ± {D_c.std():.3f} Mpc", flush=True)
    print(f"D_c median = {np.median(D_c):.3f} Mpc", flush=True)

    # Save
    outdir = "results/Maser/NGC4258_mode1"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "samples.npz")
    np.savez(outpath, **{k: np.asarray(v) for k, v in samples.items()})
    print(f"Saved samples to {outpath}", flush=True)


if __name__ == "__main__":
    main()
