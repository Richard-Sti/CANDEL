# GW170817 Parameter Estimation with Bilby

Goal: replicate the `high_spin_PhenomPNRT` analysis from Abbott et al. 2019
(Phys. Rev. X 9, 011001; arXiv:1805.11579) using Bilby + dynesty, then extend
with a jet afterglow viewing angle constraint and an inhomogeneous Malmquist
bias distance prior.

---

## Original analysis configuration

| Setting | Value |
|---|---|
| Waveform | `IMRPhenomPv2_NRTidal` (IMRPhenomPv2 + NRTidal + self-spin) |
| Sampler | LALInference MCMC |
| Frequency range | 23 -- 2048 Hz |
| Segment length | 128 s |
| Sampling rate | 4096 Hz |
| PSD | BayesWave median (spline + Lorentzians, from on-source data) |
| Detectors | H1, L1, V1 |
| Sky position | Fixed to NGC 4993 (RA=197.45°, dec=-23.38°) |
| Calibration | Marginalized (spline envelope, 10 nodes in log f) |
| Reference frequency | 100 Hz |

### Priors

| Parameter | Prior |
|---|---|
| $m_1, m_2$ (detector frame) | Uniform in [0.5, 7.7] $M_\odot$, $m_1 \geq m_2$ |
| Spin magnitudes | Uniform in [0, 0.89] |
| Spin orientations | Isotropic |
| $D_L$ | $\propto D_L^2$ (uniform in comoving volume, source-frame rate) |
| $\cos\theta_{JN}$ | Uniform in [-1, 1] |
| $\Lambda_1, \Lambda_2$ | Uniform in [0, 5000], independent |
| Redshift | Fixed $z = 0.0099$ |

### Note on redshift and detector-frame masses

The GW waveform constrains **detector-frame masses** $m_i^\text{det}$. The
source-frame masses are $m_i = m_i^\text{det} / (1 + z_\text{obs})$. The
conversion always uses the **observed** CMB-frame redshift
$z_\text{obs} = 0.0099$ (which includes the peculiar velocity of NGC 4993),
because the GW frequency redshift is set by the total relative motion, not
just the Hubble flow. This is a fixed measured number — it does not depend on
$H_0$ or the cosmological model.

---

## Part 1: Vanilla Bilby PE

### Step 0: Install dependencies

```bash
pip install bilby lalsuite dynesty
```

LALSuite installs as a pre-built wheel (waveform generation only; no need to
compile LALInference). Verify the waveform is available:

```python
import lalsimulation
lalsimulation.GetApproximantFromString("IMRPhenomPv2_NRTidal")
```

### Step 1: Load GWOSC strain data

Bilby does **not** have a single `get_event_data()` function. Load each
interferometer individually using gwpy:

```python
import bilby
from gwpy.timeseries import TimeSeries

trigger_time = 1187008882.43
duration = 128
sampling_frequency = 4096
post_trigger_duration = 2
start_time = trigger_time - duration + post_trigger_duration
end_time = start_time + duration
psd_duration = 32 * duration  # longer segment for PSD estimation

ifos = bilby.gw.detector.InterferometerList([])
for det_name in ["H1", "L1", "V1"]:
    ifo = bilby.gw.detector.get_empty_interferometer(det_name)

    # Fetch strain from GWOSC via gwpy
    data = TimeSeries.fetch_open_data(det_name, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    # Estimate PSD from a longer off-source segment
    psd_data = TimeSeries.fetch_open_data(
        det_name, start_time - psd_duration, start_time,
    )
    psd = psd_data.psd(fftlength=duration, method="median")
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value,
    )

    ifo.minimum_frequency = 23.0
    ifo.maximum_frequency = 2048.0
    ifos.append(ifo)
```

The PSD is estimated from off-source data via Welch's method, which differs
from the BayesWave PSD used in the original analysis — this is one known
source of minor differences.

### Step 2: Set up the waveform generator

```python
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=(
        bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
    ),
    waveform_arguments={
        "waveform_approximant": "IMRPhenomPv2_NRTidal",
        "reference_frequency": 100.0,
        "minimum_frequency": 23.0,
    },
)
```

### Step 3: Configure priors

Match the original high-spin priors as closely as possible:

```python
import numpy as np

priors = bilby.gw.prior.BNSPriorDict(aligned_spin=False)

# Masses (detector frame)
priors["mass_1"] = bilby.prior.Uniform(0.5, 7.7, name="mass_1")
priors["mass_2"] = bilby.prior.Uniform(0.5, 7.7, name="mass_2")
priors["mass_ratio"] = bilby.prior.Constraint(0.125, 1.0)

# Spins (high-spin: |chi| < 0.89, isotropic orientations)
priors["a_1"] = bilby.prior.Uniform(0, 0.89, name="a_1")
priors["a_2"] = bilby.prior.Uniform(0, 0.89, name="a_2")
priors["tilt_1"] = bilby.prior.Sine(name="tilt_1")
priors["tilt_2"] = bilby.prior.Sine(name="tilt_2")
priors["phi_12"] = bilby.prior.Uniform(0, 2 * np.pi, name="phi_12")
priors["phi_jl"] = bilby.prior.Uniform(0, 2 * np.pi, name="phi_jl")

# Distance (uniform in comoving volume, source-frame rate correction)
priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
    minimum=1, maximum=100, name="luminosity_distance",
)

# Sky position (fixed to NGC 4993)
priors["ra"] = bilby.prior.DeltaFunction(3.44616, name="ra")
priors["dec"] = bilby.prior.DeltaFunction(-0.4085, name="dec")

# Inclination (isotropic)
priors["theta_jn"] = bilby.prior.Sine(name="theta_jn")

# Phase and polarisation
priors["phase"] = bilby.prior.Uniform(0, 2 * np.pi, name="phase")
priors["psi"] = bilby.prior.Uniform(0, np.pi, name="psi")

# Coalescence time
priors["geocent_time"] = bilby.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name="geocent_time",
)

# Tidal deformability (uniform, independent)
priors["lambda_1"] = bilby.prior.Uniform(0, 5000, name="lambda_1")
priors["lambda_2"] = bilby.prior.Uniform(0, 5000, name="lambda_2")
```

### Step 4: Set up likelihood

```python
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    time_marginalization=False,
    phase_marginalization=False,
    distance_marginalization=False,
)
```

Cannot analytically marginalize time or phase for a precessing waveform.
Could enable `distance_marginalization=True` to speed up convergence — this
importance-reweights the distance posterior after sampling.

### Step 5: Run dynesty

```python
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=1000,
    npool=N_CPUS,
    walks=100,
    nact=50,
    maxmcmc=5000,
    outdir=outdir,
    label="GW170817_high_spin_PhenomPNRT",
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
)
```

### Step 6: Validate against published samples

Compare marginal posteriors (especially $\mathcal{M}_c$, $q$, $D_L$,
$\theta_{JN}$, $\tilde{\Lambda}$) against the LIGO-released
`high_spin_PhenomPNRT_posterior_samples.dat`. Use a PP plot or KS test.
Small differences are expected from:

- PSD: GWOSC vs BayesWave
- Sampler: dynesty vs LALInference MCMC
- Calibration: Bilby's implementation vs LALInference's

---

## Compute estimates

| Setup | Cores | Wall time |
|---|---|---|
| Single node, `npool` | 16--32 | ~12--24 hrs |
| `parallel_bilby` + MPI, 4 nodes | 128 | ~3 hrs |
| `parallel_bilby` + MPI, 8 nodes | 256 | ~1.5 hrs |

For MPI cluster runs, use `parallel_bilby`:

```bash
pip install parallel-bilby
parallel_bilby_generation config.ini
mpirun -n 128 parallel_bilby_analysis outdir/config_complete.ini
```

---

## Known differences from the original

1. **PSD**: GWOSC PSD vs BayesWave median — expect small shifts in posteriors.
2. **Sampler**: dynesty nested sampling vs LALInference MCMC — statistically
   equivalent but not identical; different convergence diagnostics.
3. **Calibration**: Bilby supports spline calibration marginalisation but the
   exact node placement and envelope widths may differ from LALInference.
4. **Phase marginalisation**: the original could not marginalise phase for
   precessing waveforms either, so this is consistent.

---

## Key outputs from Part 1

- $D_L$ posterior samples — the main product for H$_0$ inference
- $\theta_{JN}$ posterior — important for the $D_L$--inclination degeneracy
- $\tilde{\Lambda}$ posterior — constrains the neutron star EOS
- Full corner plots for validation

---

## Part 2: JetFit afterglow viewing angle constraint

Following Palmese et al. 2024 (PRD 109, 063508; arXiv:2305.19914).

### Background

Palmese et al. fit the multiwavelength afterglow light curve (Chandra X-ray,
HST optical, VLA/MeerKAT radio; ~3.5 years of data) using the JetFit code,
which models a structured relativistic jet via boosted fireball hydrodynamics.
The fit yields a 2D posterior over $(D_L, \theta_\text{obs})$ where
$\theta_\text{obs} = \min(\iota, 180° - \iota)$ is the viewing angle.

Their result: $\theta_\text{obs} = 0.53^{+0.05}_{-0.03}$ rad
($30.4^{+2.9}_{-1.7}$ deg).

### How they combine GW + EM

Joint Bayesian inference:

$$
p(H_0 \mid x_\text{GW}, x_\text{EM}, v_r)
\propto p(H_0) \int d D_L\, d\cos\iota\;
p(x_\text{GW} \mid D_L, \cos\iota) \;
p(x_\text{EM} \mid D_L, \cos\iota) \;
p(v_r \mid D_L, H_0)
$$

They fit a **12-component Gaussian Mixture Model** to the 2D JetFit samples
to get an evaluable density for the EM term.

**Angle folding**: GW uses $\iota \in [0°, 180°]$. EM uses
$\theta_\text{obs} = \min(\iota, 180° - \iota) \in [0°, 90°]$. Must convert
explicitly when evaluating the EM term at GW sample locations.

### Step 2.0: Install JetFit

```bash
git clone https://github.com/NYU-CAL/JetFit.git
pip install emcee h5py pandas
```

The repo contains everything needed:

| File | Contents |
|---|---|
| `GW170817.csv` | Multiwavelength afterglow data (time, flux, freq) |
| `Table.h5` | Pre-computed characteristic spectral functions |
| `Example_Fitter.py` | Working example that fits GW170817 |
| `JetFit/` | Package (Interpolator, FluxGenerator, Fitter classes) |

### Step 2.1: Understand the JetFit model

JetFit uses the "boosted fireball" structured jet model (Duffell & MacFadyen
2013; Wu & MacFadyen 2018). The fitted parameters are:

| Parameter | Description | Default bounds |
|---|---|---|
| `E` | Jet isotropic-equivalent energy [erg] | $[10^{-6}, 10^3]$ (in units of $10^{50}$ erg) |
| `n` | ISM number density [cm$^{-3}$] | $[10^{-6}, 10^3]$ |
| `Eta0` | Initial Lorentz factor profile index | [2, 10] |
| `GammaB` | Bulk Lorentz factor of jet core | [1, 12] |
| `theta_obs` | Observer viewing angle [rad] | [0, 1] |
| `epse` | Electron energy fraction | $[10^{-6}, 1]$ |
| `epsb` | Magnetic energy fraction | $[10^{-6}, 1]$ |
| `p` | Electron power-law index | [2, 4] |

Fixed parameters: $d_L = 0.012188$ (in units of $10^{28}$ cm; NGC 4993 at
$\sim$40 Mpc gives $40 \times 3.086 \times 10^{24} / 10^{28} \approx 0.0123$),
$z = 0.00973$, $\xi_N = 1$.

The example fits `Eta0`, `GammaB`, and `theta_obs` while keeping the others
fixed. For a full run matching Palmese et al., all 8 parameters should be
sampled.

### Step 2.2: Run JetFit MCMC

The `Example_Fitter.py` provides the template. For a production run:

```python
from JetFit import FitterClass

# Use parallel-tempered emcee for multimodal posterior
SamplerType = "ParallelTempered"
NTemps = 10
NWalkers = 100
Threads = 8
BurnLength = 10000
RunLength = 10000
```

To also fit $D_L$: add `'dL'` to the `Info['Fit']` array and set appropriate
bounds. This yields joint $(\theta_\text{obs}, D_L, \ldots)$ posterior
samples.

Compute: ~24 hours on 8 cores (per the JetFit README).

### Step 2.3: Build the GMM for the EM constraint

```python
from sklearn.mixture import GaussianMixture

# Extract (D_L, theta_obs) from JetFit chain
jetfit_samples = np.column_stack([dL_chain, theta_obs_chain])

gmm = GaussianMixture(n_components=12, covariance_type="full")
gmm.fit(jetfit_samples)
```

### Step 2.4: Combine with GW posterior

**Option A (post-hoc reweighting of Part 1 samples):**

```python
# For each GW sample, fold iota to theta_obs
theta_obs = np.minimum(iota_samples, np.pi - iota_samples)

# Evaluate EM log-density
log_w_em = gmm.score_samples(np.c_[DL_samples, theta_obs])

# Divide out the original D_L prior to avoid double-counting
log_w_prior = -np.log(prior_DL.prob(DL_samples))

log_weights = log_w_em + log_w_prior
weights = np.exp(log_weights - log_weights.max())

# Resample
idx = np.random.choice(len(DL_samples), size=N, p=weights / weights.sum())
DL_combined = DL_samples[idx]
```

**Option B (joint likelihood in Bilby):** Subclass `bilby.Likelihood` and add
the GMM log-density to the GW log-likelihood:

```python
class GWplusEMLikelihood(bilby.gw.GravitationalWaveTransient):
    def __init__(self, *args, gmm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gmm = gmm

    def log_likelihood(self):
        log_l_gw = super().log_likelihood()
        theta_jn = self.parameters["theta_jn"]
        dl = self.parameters["luminosity_distance"]
        theta_obs = min(theta_jn, np.pi - theta_jn)
        log_l_em = self.gmm.score_samples([[dl, theta_obs]])[0]
        return log_l_gw + log_l_em
```

---

## Part 3: Inhomogeneous Malmquist bias distance prior

### Motivation

The standard GW PE distance prior is $p(D_L) \propto D_L^2$ (uniform in
comoving volume). If you have a reconstructed density field $n(D_L, \hat{r})$
along the line of sight from a peculiar velocity survey (e.g., Cosmicflows-4),
you can replace this with:

$$
p(D_L \mid \hat{r}) \propto D_L^2 \; n(D_L, \hat{r})
$$

where $\hat{r}$ is the sky direction. This upweights distances where the
galaxy density is higher, reflecting the prior probability that the host
galaxy lives in an overdense region.

### Implementation in Bilby (on the fly, within the sampler)

Since RA/dec are **fixed** for GW170817 (sky position is known from the EM
counterpart), the density profile $n(D_L)$ along that sightline is a fixed 1D
function. This means the modified prior is just a 1D tabulated function that
replaces `UniformSourceFrame` directly — no conditional prior machinery needed.

**Use `bilby.core.prior.Interped`:**

```python
# DL_grid: array of luminosity distances [Mpc]
# n_LOS: reconstructed galaxy density n(DL) along the NGC 4993 sightline
#         from the CANDEL/CF4 density field (1 + delta, normalized so
#         the mean over random sightlines is 1)

p_DL = DL_grid**2 * n_LOS  # unnormalized; Interped normalizes internally

priors["luminosity_distance"] = bilby.core.prior.Interped(
    xx=DL_grid, yy=p_DL,
    minimum=1.0, maximum=100.0,
    name="luminosity_distance",
    latex_label=r"$D_L$", unit="Mpc",
)
```

This directly replaces `UniformSourceFrame` in Step 3. The sampler (dynesty)
then draws $D_L$ proposals from this modified prior via the inverse CDF that
`Interped` builds internally. The prior evaluation in the likelihood is also
automatic — Bilby calls `prior.ln_prob(D_L)` which returns the log of the
tabulated density.

`Interped` internally:
- Normalizes `yy` via trapezoidal integration
- Builds `scipy.interpolate.interp1d` for the PDF, CDF, and inverse CDF
- `rescale(u)` maps uniform [0,1] samples to $D_L$ via inverse CDF — this is
  how dynesty proposes new live points
- `prob(D_L)` evaluates the normalized PDF at any $D_L$

**No changes needed anywhere else in the pipeline.** The waveform generator,
likelihood, and sampler configuration remain identical. The only modification
is swapping the distance prior in Step 3.

### Where to get $n(D_L, \hat{r})$

From the CANDEL codebase: the line-of-sight density field is stored in the
data dictionaries as `los_density` evaluated on a radial grid `los_r`. For
NGC 4993 at (RA, dec) = (197.45°, -23.38°), extract the density profile from
the CF4 reconstruction and interpolate onto the $D_L$ grid.

### Caveats

1. **Density field resolution**: the CF4 density reconstruction has finite
   smoothing ($\sim$few Mpc). At $D_L \sim 40$ Mpc (NGC 4993), this is a
   $\sim$10% fractional distance, so the correction could be significant.

2. **Density field normalisation**: ensure $n(D_L)$ is normalised such that
   the mean over random sightlines gives 1 (i.e., it is a relative
   overdensity $1 + \delta$, not an absolute number density).

3. **Consistency**: if you also model the peculiar velocity of NGC 4993 using
   the CF4 velocity field, the density and velocity fields must come from the
   same reconstruction to avoid double-counting information.

---

## Future extension: Joint PE + H$_0$ inference

The two-stage approach (PE with fixed $z_\text{obs}$ $\to$ extract $D_L$
$\to$ infer $H_0$) is standard. Note that the detector-frame to source-frame
mass conversion **always** uses the observed CMB-frame redshift
$z_\text{obs} = 0.0099$, regardless of whether $H_0$ is inferred jointly or
separately. The GW frequency redshift is caused by the total relative motion
(Hubble flow + peculiar velocity), so $z_\text{obs}$ is the correct quantity
for the mass conversion and it is a fixed measured number.

What changes in a joint inference is the **distance--velocity relation**:

$$
D_L = D_L(z_\text{cos}, H_0), \qquad
z_\text{cos} = z_\text{obs} - v_\text{pec}/c
$$

A single-step approach would sample $H_0$ (and optionally $v_\text{pec}$) as
parameters, and evaluate $p(v_r \mid D_L, H_0)$ at each likelihood call.
The mass posteriors are unaffected since $z_\text{obs}$ is fixed.

For a first pass, the two-stage approach is sufficient — the $D_L$ posterior
is only weakly correlated with non-geometric GW parameters once $\iota$ is
marginalised out.

---

## References

- Abbott et al. 2019, Phys. Rev. X 9, 011001 (arXiv:1805.11579) — GW170817 PE
- Palmese et al. 2024, PRD 109, 063508 (arXiv:2305.19914) — afterglow H$_0$
- Hotokezaka et al. 2019, Nat. Astron. 3, 940 (arXiv:1806.10596) — VLBI H$_0$
- Mooley et al. 2018, Nature 561, 355 (arXiv:1806.09693) — superluminal motion
- Wu & MacFadyen 2018, ApJ 869, 55 — boosted fireball jet model (JetFit basis)
- Ryan et al. 2020, ApJ 896, 166 (arXiv:1909.11691) — afterglowpy
- Veitch et al. 2015, PRD 91, 042003 (arXiv:1409.7215) — LALInference
- Dietrich et al. 2017, PRD 96, 121501 (arXiv:1706.02969) — NRTidal
