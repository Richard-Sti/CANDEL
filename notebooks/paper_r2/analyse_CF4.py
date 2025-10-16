import numpy as np
import warnings
import sys
import os
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pandas as pd
import matplotlib.pyplot as plt
# Suppress noisy FutureWarnings emitted within seaborn's internal pandas usage
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"seaborn\._oldcore"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*use_inf_as_na option is deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*length-1 list-like.*length-1 tuple.*"
)
import seaborn as sns

# Required CLI flags (positional):
#   qj_model   (0: no q0/j0; 1: fixed q0=-0.595,j0=1; 2: infer q0, j0 fixed 1; 3: infer q0 & j0)
#   lcdm_flag  (0: original q0/j0 scheme; 1: LCDM Omega parameterization)
# Internal (set in code below):
#   tf_only (True: use TF distances only; False: use all distances)
#   no_selection (False: apply Vcmb > 4000 cut; True: skip cut)

if len(sys.argv) < 3:
    print("Usage: python CF4_bias_3.py <qj_model:0|1|2|3> <lcdm_flag:0|1>")
    sys.exit(1)
try:
    qj_model = int(sys.argv[1])
except Exception:
    print("ERROR: qj_model argument invalid")
    sys.exit(1)
try:
    lcdm_flag = int(sys.argv[2])
except Exception:
    print("ERROR: lcdm_flag argument invalid")
    sys.exit(1)
if qj_model not in (0,1,2,3):
    print("ERROR: qj_model must be 0,1,2,3")
    sys.exit(1)
if lcdm_flag not in (0,1):
    print("ERROR: lcdm_flag must be 0 or 1")
    sys.exit(1)

# Hard-coded analysis toggles (adjust here, not via CLI)
tf_only = True       # set False to use all distance indicators
no_selection = False # set True to keep all Vcmb (no >4000 cut)

qj_desc = {
    0: "No q0/j0 (pure H0)",
    1: "Fixed q0=-0.595, j0=1", 
    2: "Infer q0, fix j0=1", 
    3: "Infer q0 and j0"
}[qj_model]
print(f"qj_model = {qj_model}: {qj_desc}")
print(f"lcdm_flag = {lcdm_flag} (1=use LCDM Omega_m/Omega_L parameterization)")
print(f"tf_only (hard-coded)   = {int(tf_only)} (1=TF only; 0=all distances)")
print(f"no_selection (hard-coded) = {int(no_selection)} (1=NO Vcmb>4000 cut; 0=apply cut)")
qj_suffix = f"_qj{qj_model}"
all_suffix = "" if tf_only else "_all"
sel_suffix = "_noselection" if no_selection else ""
lcdm_suffix = "_lcdm" if lcdm_flag else ""

PGC, VCMB, D, eD, D_TF, eD_TF = np.genfromtxt("CF4_distances.csv", delimiter=',', unpack=True)

# Make a Pandas dataframe of this
df = pd.DataFrame({
    'Vcmb': VCMB,
    'D': D,
    'eD': eD,
    'D_TF': D_TF,
    'eD_TF': eD_TF
})

print("Original number of objects:", len(df))


# Ensure output directory exists before saving any plots
os.makedirs('Plots', exist_ok=True)
# Apply or skip the Vcmb > 4000 selection based on flag
if not no_selection:
    df = df[df['Vcmb'] > 4000]
df = df.reset_index(drop=True)

# Save the TF-only subset (as would be kept under tf_only) to an .npy file
tf_df = df[(df['eD_TF'] > 0) & (df['Vcmb'] > 0)].reset_index(drop=True)
tf_subset = tf_df[['Vcmb', 'D_TF', 'eD_TF']].to_numpy()
tf_out = f"CF4_TF_subset{sel_suffix}.npy"
np.save(tf_out, tf_subset)
print(f"Saved TF subset to {tf_out} with shape {tf_subset.shape}")

# Save the full-distance subset (all indicators) with quality cuts to an .npy file
full_df = df[(df['eD'] > 0) & (df['D'] > 0) & (df['Vcmb'] > 0)].reset_index(drop=True)
full_subset = full_df[['Vcmb', 'D', 'eD']].to_numpy()
full_out = f"CF4_full_subset{sel_suffix}.npy"
np.save(full_out, full_subset)
print(f"Saved FULL subset to {full_out} with shape {full_subset.shape}")

if tf_only:
    filtered_df = df[df['eD_TF'] > 0]
    # Drop non-positive Vcmb to avoid log10 invalids in model
    filtered_df = filtered_df[filtered_df['Vcmb'] > 0]
    Vcmb = filtered_df['Vcmb'].values
    D_obs_np = filtered_df['D_TF'].values
    eD_obs_np = filtered_df['eD_TF'].values
else:
    filtered_df = df[df['eD'] > 0]
    # Drop non-positive Vcmb to avoid log10 invalids in model
    filtered_df = filtered_df[filtered_df['Vcmb'] > 0]
    Vcmb = filtered_df['Vcmb'].values
    D_obs_np = filtered_df['D'].values
    eD_obs_np = filtered_df['eD'].values

z = Vcmb / 299792.458  # Convert Vcmb to redshift

print("Final number of objects:", len(filtered_df))

try:
    os.makedirs('Plots', exist_ok=True)
except Exception:
    pass


# Likelihood function, starting with the fiducial analysis that reproduces the results from CF4

import importlib
###############################
# Model Definitions (Two Variants)
###############################

def _cosmography(Vcmb, h0, z_local):
    """Return distance modulus model.

    Two parameterizations:
      - Original (lcdm_flag=0): sample/ fix q0, j0 directly per qj_model.
      - LCDM (lcdm_flag=1): sample Omegas; derive q0, j0 assuming FLRW with matter, Lambda, curvature.

    LCDM relations (a=1 today):
      q0 = 0.5 * Omega_m - Omega_L
      j0 = 1 + (Omega_k)  where Omega_k = 1 - Omega_m - Omega_L
      (For flat case: Omega_k=0 -> j0=1.)
    For qj_model==2 under LCDM: flat assumed -> sample Omega_m, set Omega_L = 1 - Omega_m.
    For qj_model==3 under LCDM: allow curvature -> sample Omega_m, Omega_L independently (Uniform priors) and clip small negatives minimally via transform enforcement (kept simple here by broad uniform in [0,1.5]).
    """
    if not lcdm_flag:
        # Original q0/j0 scheme
        if qj_model == 0:
            return 5*jnp.log10(Vcmb / h0) + 25
        elif qj_model == 1:
            q0 = -0.595; j0 = 1.0
        elif qj_model == 2:
            q0 = numpyro.sample("q0", dist.Uniform(-10.0, 10.0)); j0 = 1.0
        elif qj_model == 3:
            q0 = numpyro.sample("q0", dist.Uniform(-10.0, 10.0))
            j0 = numpyro.sample("j0", dist.Uniform(-10.0, 10.0))
        else:
            raise ValueError("Invalid qj_model")
    else:
        # LCDM-based parameterization
        if qj_model == 0:
            return 5*jnp.log10(Vcmb / h0) + 25
        elif qj_model == 1:
            # fixed baseline (use Planck-like ~0.315 for Omega_m, flat)
            Omega_m = 0.315
            Omega_L = 1.0 - Omega_m
        elif qj_model == 2:  # flat LCDM: infer Omega_m only
            Omega_m = numpyro.sample("Omega_m", dist.Uniform(-10, 10))
            Omega_L = 1.0 - Omega_m
        elif qj_model == 3:  # allow curvature: infer Omega_m & Omega_L
            Omega_m = numpyro.sample("Omega_m", dist.Uniform(-10, 10))
            Omega_L = numpyro.sample("Omega_L", dist.Uniform(-10, 10))
        else:
            raise ValueError("Invalid qj_model")
        Omega_k = 1.0 - Omega_m - Omega_L
        if qj_model == 3:
            numpyro.deterministic("Omega_k", Omega_k)
        q0 = 0.5*Omega_m - Omega_L
        j0 = 1.0 + Omega_k

    if qj_model > 0:  # use series expansion including q0, j0
        f = 1. + 0.5*(1.-q0)*z_local - 1./6.*(1 - q0 - 3*q0**2 + j0)*z_local**2
        return 5*jnp.log10(Vcmb / h0 * f) + 25
    else:
        return 5*jnp.log10(Vcmb / h0) + 25

def model_fiducial(Vcmb, D_obs, eD_obs):
    h0 = numpyro.sample("h0", dist.Uniform(50.0, 100.0))
    D_theory = _cosmography(Vcmb, h0, z)
    numpyro.sample("obs", dist.Normal(D_theory, eD_obs), obs=D_obs)

def model_comoving(Vcmb, D_obs, eD_obs):
    h0 = numpyro.sample("h0", dist.Uniform(50.0, 100.0))
    D_theory = _cosmography(Vcmb, h0, z)
    numpyro.sample("obs", dist.Normal(D_theory, eD_obs), obs=D_obs)
    # extra volume prior factor
    extra_term = 0.6 * jnp.log(10) * D_theory
    numpyro.factor("extra_term", extra_term)

###############################
# Run Both Models
###############################
rng_key = random.PRNGKey(0)
key1, key2 = random.split(rng_key)

def run_chain(model_fn, key):
    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=3000)
    mcmc.run(key, Vcmb=jnp.array(Vcmb), D_obs=jnp.array(D_obs_np), eD_obs=jnp.array(eD_obs_np))
    return mcmc

mcmc_fid = run_chain(model_fiducial, key1)
mcmc_com = run_chain(model_comoving, key2)

print("\n===== Fiducial model MCMC summary =====")
mcmc_fid.print_summary()
print("\n===== Comoving-volume model MCMC summary =====")
mcmc_com.print_summary()

samples_fid = mcmc_fid.get_samples()
samples_com = mcmc_com.get_samples()

###############################
# Build param sample dicts for plotting
###############################
def collect_params(samples_dict):
    ps = {"h0": np.array(samples_dict["h0"]) }
    if lcdm_flag:
        # Under LCDM we may have Omega parameters instead of / in addition to q0,j0
        if "Omega_m" in samples_dict:
            ps["Omega_m"] = np.array(samples_dict["Omega_m"])
        if "Omega_L" in samples_dict:
            ps["Omega_L"] = np.array(samples_dict["Omega_L"])
        if "Omega_k" in samples_dict:
            ps["Omega_k"] = np.array(samples_dict["Omega_k"])
    else:
        if qj_model in (2,3) and "q0" in samples_dict:
            ps["q0"] = np.array(samples_dict["q0"])
        if qj_model == 3 and "j0" in samples_dict:
            ps["j0"] = np.array(samples_dict["j0"]) 
    return ps

param_fid = collect_params(samples_fid)
param_com = collect_params(samples_com)

###############################
# Plotting helpers
###############################
def plot_param_set(param_dict, tag):
    if len(param_dict) == 1:
        plt.figure(figsize=(8,6))
        plt.hist(param_dict["h0"], bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f"Posterior $H_0$ ({tag})")
        plt.xlabel("$H_0$")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = f"Plots/H0_posterior_CF4_{tag}{qj_suffix}{lcdm_suffix}{all_suffix}{sel_suffix}.png"
        plt.savefig(out_path, dpi=300)
    else:
        have_corner = importlib.util.find_spec("corner") is not None
        data = np.vstack([param_dict[k] for k in param_dict.keys()]).T
        def _label(k):
            if k == "h0":
                return r"$H_0$"
            if lcdm_flag:
                if k == "Omega_m": return r"$\Omega_m$"
                if k == "Omega_L": return r"$\Omega_\Lambda$"
                if k == "Omega_k": return r"$\Omega_k$"
            return r"$q_0$" if k == "q0" else (r"$j_0$" if k == "j0" else k)
        labels = [_label(k) for k in param_dict.keys()]
        out_path = f"Plots/posterior_corner_CF4_{tag}{qj_suffix}{lcdm_suffix}{all_suffix}{sel_suffix}.png"
        if have_corner:
            import corner
            fig = corner.corner(data, labels=labels, show_titles=True, quantiles=[0.16,0.5,0.84], title_fmt=".2f")
            fig.savefig(out_path, dpi=300)
            print(f"Corner plot produced with corner: {out_path}")
        else:
            import pandas as _pd
            _df = _pd.DataFrame(data, columns=[k.upper() for k in param_dict.keys()])
            sns.pairplot(_df, corner=True)
            plt.savefig(out_path, dpi=300)
            print(f"Corner-style pairplot produced (seaborn): {out_path}")
    print(f"Plot saved to {out_path}")

plot_param_set(param_fid, "fiducial")
plot_param_set(param_com, "comoving_volume")
