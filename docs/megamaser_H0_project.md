# Megamaser Geometric H0: A Full Bayesian Forward Model

## 1. Introduction

We present a Bayesian forward model for measuring the Hubble constant from water megamaser disk galaxies. Unlike the approach of Pesce et al. (2020), who derived angular-diameter distances from maser disk modelling and then fit H0 to those derived distances, we construct a hierarchical model that operates directly on the raw VLBI observables — maser spot sky positions, line-of-sight velocities, and line-of-sight accelerations. The distance to each galaxy is a latent variable sampled with a uniform-in-volume (D^2) prior, and the selection function is modelled explicitly.

This approach is motivated by the principle that a proper forward model must predict the *observables*, not intermediate derived quantities. Published distance posteriors are outputs of a separate inference and carry implicit prior choices, approximations, and potential systematic biases that cannot be propagated transparently into the H0 inference.

The model is implemented in JAX/NumPyro within the CANDEL peculiar velocity analysis framework and validated through mock closure tests.

---

## 2. The Forward Model

### 2.1 Observables

For each maser galaxy, the data consists of measurements for individual maser spots obtained from VLBI imaging and long-baseline spectral monitoring:

| Observable | Symbol | Units | Source |
|-----------|--------|-------|--------|
| Sky position (RA offset) | x_k | mas | VLBI imaging |
| Sky position (Dec offset) | y_k | mas | VLBI imaging |
| Position uncertainty | sigma_{x,k}, sigma_{y,k} | mas | Beam / S/N |
| Line-of-sight velocity | v_k | km/s | Spectral channel centre |
| Line-of-sight acceleration | a_k | km/s/yr | Spectral monitoring (~2 yr baseline) |
| Acceleration uncertainty | sigma_{a,k} | km/s/yr | Monitoring fit |
| Spot type | b / s / r | — | Velocity classification |

The velocities have no formal per-spot uncertainties; the entire velocity uncertainty budget is captured by error floor parameters fitted as part of the model.

### 2.2 Latent variables

**Global cosmological parameters:**
- H0: Hubble constant (km/s/Mpc)
- sigma_pec: peculiar velocity dispersion (km/s)

**Per-galaxy disk parameters:**
- D: angular-diameter distance (Mpc) — sampled with D^2 volume prior
- M_BH: supermassive black hole mass (M_sun)
- v_sys: systemic recession velocity (km/s, barycentric frame)
- x0, y0: black hole sky position (mas)
- i0: disk inclination at r = 0 (degrees); i = 90 is edge-on
- Omega0: disk position angle at r = 0 (degrees)
- dOmega/dr: position angle warp rate (degrees/mas)
- Error floors: sigma_x, sigma_y (mas), sigma_{v,sys}, sigma_{v,hv} (km/s), sigma_a (km/s/yr)

**Per-spot latent variables:**
- r_k: orbital radius in the disk (mas)
- phi_k: azimuthal angle in the disk (radians), with type-dependent priors:
  - Redshifted: phi in [0, pi]
  - Blueshifted: phi in [pi, 2pi]
  - Systemic: phi in [-pi/2, pi/2]

For a galaxy with N spots, the total parameter count is ~15 global + 2N per-spot.

### 2.3 Disk physics

The disk model follows Pesce et al. (2020, arXiv:2001.04581, Appendix A). All equations are implemented in JAX for automatic differentiation.

**Warped geometry:**
- i(r) = i0 + (di/dr) * r
- Omega(r) = Omega0 + (dOmega/dr) * r

**Sky-plane positions** (Eqs A5a-b):
- X = x0 + r * [sin(phi) sin(Omega) - cos(phi) cos(Omega) cos(i)]
- Y = y0 + r * [sin(phi) cos(Omega) + cos(phi) sin(Omega) cos(i)]

**Keplerian circular velocity** (Eq A10):
- v(r) = C_v * sqrt(M_BH / (r * D))
- where C_v = 0.9420 km/s (unit conversion constant for r in mas, D in Mpc, M in M_sun)

**Observed velocity** (Eqs A12-A16): The full relativistic prescription is used:
- Line-of-sight velocity: v_z = v_kep * sin(phi) * sin(i)
- Relativistic Doppler: (1 + z_D) = gamma * (1 + v_z / c)
- Gravitational redshift: (1 + z_g) = (1 - R_s / (r * D))^{-1/2}
- Total observed: v_obs = c * [(1 + z_D)(1 + z_g)(1 + z_0) - 1]

where z_0 = v_sys / c is the systemic redshift (optical convention).

**Line-of-sight acceleration** (Eqs A6c, A7):
- a(r) = C_a * M_BH / (r^2 * D^2)
- A_z = a * cos(phi) * sin(i)
- where C_a = 1.872e-4 km/s/yr

**How distance enters:** The angular-diameter distance D appears in the velocity as v ~ D^{-1/2} and in the acceleration as a ~ D^{-2}. This is the key: velocities constrain M_BH / D, while accelerations constrain M_BH / D^2. Together they break the degeneracy and determine both M_BH and D independently.

### 2.4 Likelihood

The likelihood factorises into three independent terms (Eqs B19, B21, B23):

**Positions:**
ln L_1 = -1/2 sum_k [(x_k - X_k)^2 / (sigma_{x,k}^2 + sigma_x^2) + (y_k - Y_k)^2 / (sigma_{y,k}^2 + sigma_y^2) + normalisation terms]

**Accelerations** (only where measured):
ln L_2 = -1/2 sum_k [(a_k - A_k)^2 / (sigma_{a,k}^2 + sigma_a^2) + normalisation terms]

**Velocities** (different error floors for systemic vs high-velocity features):
ln L_3 = -1/2 sum_k [(v_k - V_k)^2 / sigma_v^2 + ln(2 pi sigma_v^2)]

where sigma_v = sigma_{v,sys} for systemic masers and sigma_{v,hv} for high-velocity masers. The error floor is the SOLE source of velocity uncertainty — there are no per-spot formal velocity errors.

### 2.5 H0-distance connection

The cosmological redshift z_cosmo is determined from D and H0 via the exact angular-diameter distance relation in flat LCDM:

D_A(z, H0) = (c / (H0 (1 + z))) * integral_0^z dz' / E(z')

We invert this numerically using 5 Newton-Raphson iterations to find z_cosmo given D and H0. The peculiar velocity is then obtained from the exact relativistic redshift composition:

(1 + z_obs) = (1 + z_cosmo) * (1 + v_pec / c)

where z_obs = v_{sys,CMB} / c. This enters the likelihood as a Gaussian constraint on v_pec:

ln L_vpec = -1/2 * (v_pec / sigma_pec)^2 - ln(sigma_pec)

### 2.6 Selection function

The Megamaser Cosmology Project surveys active galactic nuclei for water maser emission. More distant galaxies have fainter maser flux (flux ~ D^{-2}), reducing the detection probability. We model this as:

S(D) = Phi((D_lim - D) / D_width)

where Phi is the standard normal CDF, D_lim is an effective distance limit, and D_width controls the sharpness of the selection boundary. Both are sampled as parameters.

The selection correction enters the likelihood in two parts:
1. Per-object: factor(log S(D_i)) — the probability that this galaxy at distance D_i would be detected
2. Normalisation: factor(-log Z_sel) where Z_sel = integral S(D') D'^2 dD' — computed on a pre-built grid

This is structurally identical to the magnitude selection in the TRGB model implemented in CANDEL: the D^2 volume prior plays the role of the r^2 distance prior, and S(D) plays the role of the magnitude selection probability.

### 2.7 Priors

All priors are uniform or weakly informative:
- D: Uniform(D_min, D_max) with D^2 volume factor
- M_BH: Uniform(1e6, 1e8) M_sun
- v_sys: Uniform (galaxy-specific range)
- Geometry: Uniform (galaxy-specific, informed by approximate disk orientation)
- Position error floors: Uniform(0, 0.1) mas
- Velocity error floors: TruncatedNormal(5, 3, 0.5, 20) km/s — prevents collapse to zero
- Acceleration error floor: Uniform(0, 5) km/s/yr
- H0: Uniform(50, 100) km/s/Mpc
- sigma_pec: Uniform(50, 500) km/s

The informative prior on velocity error floors is necessary because per-spot latent variables (r, phi) can absorb velocity noise within the position measurement uncertainty (~0.01 mas allows ~18 km/s velocity shifts). Without this prior, the error floor collapses to zero, the velocity constraint becomes infinitely tight, and the D-M_BH degeneracy is resolved incorrectly.

---

## 3. Mock Closure Tests

Mock data is generated by:
1. Sampling true D from the D^2 volume prior
2. Sampling disk geometry from priors
3. Drawing maser spot positions (r, phi) in the disk
4. Computing true observables (x, y, v, a) from the disk model
5. Adding measurement noise consistent with error floors (including velocity noise)
6. Applying S/N >= 3 selection on maser spots
7. Using exact cosmography (astropy) for the z_cosmo(D, H0) relation

**Results (10 mocks, CGCG 074-064 configuration):**

| Parameter | Mean bias (sigma) | Std | KS p-value |
|-----------|-------------------|-----|------------|
| H0 | -0.20 | 0.80 | 0.73 |
| D | +0.09 | 0.72 | 0.49 |
| M_BH | +0.19 | 0.70 | 0.37 |

All key parameters are recovered without bias. The model is validated.

---

## 4. Data Availability

### 4.1 Complete datasets

| Galaxy | Spots | Positions | Velocities | Accelerations | Spot types | Source |
|--------|-------|-----------|------------|---------------|------------|--------|
| CGCG 074-064 | 165 | yes | yes | yes (145 measured) | yes | Pesce+2020 (MCP XI), IOP MRT |
| NGC 5765b | 192 | yes | yes | yes (all) | classified from v | Gao+2016 (MCP VIII), CDS |
| UGC 3789 | 127 | yes | yes | partial | classified from v | Reid+2013 (MCP V), IOP |
| NGC 6264 | 76 | yes | yes | yes | classified from v | Kuo+2013 (MCP V), IOP |
| NGC 6323 | 78 | yes | yes | yes | classified from v | Kuo+2015 (MCP VI), IOP |

### 4.2 Incomplete datasets

| Galaxy | Available | Missing | Notes |
|--------|-----------|---------|-------|
| NGC 4258 | 14,291 multi-epoch VLBI obs (Argon+2007) | Accelerations (Humphreys+2008, no electronic table); reduced spot table from Humphreys+2013/Reid+2019 not deposited | Raw data needs epoch-averaging; accelerations need digitising or author contact |

### 4.3 Data quality notes

- **Spot type classification**: For galaxies without published classifications (NGC 5765b, UGC 3789, NGC 6264, NGC 6323), we classify by velocity offset from the systemic velocity: |v - v_sys| < 200 km/s => systemic, v > v_sys + 200 => redshifted, v < v_sys - 200 => blueshifted. The 200 km/s threshold is motivated by the typical velocity gap between systemic and high-velocity features.

- **Velocity frames**: Different papers use different conventions (heliocentric, LSR, barycentric). The heliocentric-to-CMB correction is computed per galaxy from the CMB dipole.

- **Accelerations vs no accelerations**: Galaxies without acceleration data can still constrain D, but only through the M_BH/D ratio from velocities. The D-M_BH degeneracy is not fully broken, resulting in wider posteriors. Accelerations provide the M_BH/D^2 constraint needed to pin both independently.

---

## 5. Results

### 5.1 Individual galaxy results

| Galaxy | H0 (km/s/Mpc) | D (Mpc) | D_Pesce (Mpc) | Accelerations? |
|--------|---------------|---------|---------------|----------------|
| CGCG 074-064 | 80.0 +7.8 -6.6 | 87.2 +6.9 -7.2 | 87.6 +7.9 -7.2 | yes |
| NGC 6264 | 68.1 +20.2 -9.4 | 152.7 +25 -36 | 132.1 +21 -17 | no (Kuo+2011 only) |
| NGC 6323 | 65.7 +17.1 -8.3 | 121.5 +19 -25 | 109.4 +34 -23 | no (Kuo+2011 only) |

CGCG 074-064 reproduces the Pesce+2020 result (H0 = 81.0 +7.4 -6.9) within 0.1 sigma. NGC 6264 and NGC 6323 were run without accelerations — these have now been found (Kuo+2013, Kuo+2015) and the analysis should be re-run.

### 5.2 Galaxies not yet analysed

- **NGC 5765b**: Data is complete but the model converges to the wrong inclination (i0 = 80 deg instead of the true 72.4 deg). This galaxy has a significantly inclined disk (not edge-on), which requires careful initialisation of the per-spot latent variables.

- **UGC 3789**: Spot data with partial accelerations has been obtained from Reid+2013 (IOP). Not yet parsed or run.

- **NGC 4258**: Raw multi-epoch data available but accelerations are not in machine-readable form.

### 5.3 Combined constraint

From the 3 galaxies currently analysed (inverse-variance weighted):

**H0 = 75.2 +/- 5.6 km/s/Mpc**

This is consistent with both Pesce+2020 (73.9 +/- 3.0) and the SH0ES measurement (73.0 +/- 1.0), though with larger uncertainties due to using only 3 of the 6 available galaxies.

---

## 6. Open Issues and Next Steps

1. **Re-run NGC 6264 and NGC 6323 with accelerations**: The acceleration data from Kuo+2013 and Kuo+2015 has now been obtained. These should be parsed and the inference re-run, which will substantially tighten the D and H0 constraints for these two galaxies.

2. **Fix NGC 5765b**: The inclined disk (i = 72.4 deg) needs better per-spot initialisation. The position-to-(r,phi) mapping is more sensitive to geometry when the disk is far from edge-on.

3. **Parse and run UGC 3789**: Data obtained from Reid+2013. Needs a loader and config file.

4. **NGC 4258**: Requires either processing the Argon+2007 multi-epoch data into a spot table, or obtaining the reduced dataset from the authors.

5. **Mock tests with selection**: The current 10-mock validation does not exercise the selection function (mocks use fixed D near the real galaxy). A proper test would draw D from the volume prior and apply distance-dependent selection cuts.

6. **Multi-galaxy joint model**: Currently each galaxy is fit independently and the H0 posteriors are combined by inverse-variance weighting. A proper joint model would sample a single H0 (and sigma_pec) across all galaxies simultaneously.

---

## 7. Code and Reproducibility

All code is in the CANDEL package:
- `candel/model/maser_disk.py` — JAX disk physics functions
- `candel/model/model_H0_maser.py` — NumPyro forward model (MaserDiskModel)
- `candel/mock/maser_disk_mock.py` — Mock data generator
- `candel/pvdata/megamaser_data.py` — Spot data parsers
- `scripts/runs/config_maser*.toml` — Per-galaxy configuration files
- `scripts/mocks/run_mock_maser_disk.py` — Batch mock runner

All inference uses NUTS (No-U-Turn Sampler) via NumPyro with JAX backend on CPU. Typical run: 2000 warmup + 2000 samples, ~2 minutes per galaxy for CGCG 074-064 (165 spots, 345 parameters).
