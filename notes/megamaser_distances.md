# H$_2$O Disk Megamaser Distances: Data, Inference, and Limitations

**Reference:** Pesce et al. 2020, ApJ 891, L1 (MCP XIII) — the primary source for the CF4 maser distance table.

---

## 1. Raw Data

### What the data are

For each megamaser galaxy, the raw observables come from two independent observing programmes:

**From VLBI (VLBA, sub-mas imaging):** sky-plane positions of individual maser spots,
$$
(x_k,\, y_k) \quad [\text{mas}], \quad \text{with per-spot uncertainties } \sigma_{x,k},\, \sigma_{y,k}.
$$
Uncertainties are set by the synthesised beam FWHM divided by twice the SNR. Only spots with SNR $\geq$ 3 are retained.

**From single-dish monitoring (GBT/VLA, multi-year spectral campaigns):** line-of-sight (LOS) velocities and centripetal accelerations,
$$
v_k \quad [\text{km\,s}^{-1}], \qquad a_k \quad [\text{km\,s}^{-1}\,\text{yr}^{-1}], \quad \text{with uncertainty } \sigma_{a,k}.
$$
Velocity uncertainties are not measured directly; they are parameterised as free error floors in the model (see Section 2). Accelerations are measured from the secular drift of spectral line centroids across monitoring epochs spanning 2–10 years.

### Physical origin

The maser emission is 22 GHz H$_2$O stimulated by a thin, nearly edge-on Keplerian accretion disk around a supermassive black hole. Three groups of maser features are present:

- **Systemic masers** — near the galaxy recession velocity; sit in front of the BH along the LOS; have measurable centripetal accelerations ($\sim$few km\,s$^{-1}$\,yr$^{-1}$) and near-zero angular offset from the disk centre.
- **High-velocity (redshifted/blueshifted) masers** — on the near and far sides of the disk midline; LOS velocity offset $\sim v_\text{rot}$; near-zero measurable accelerations; angular offset $\sim\theta_\text{HV}$ from the disk centre.

### Source availability

Raw maser spot tables (positions, velocities, accelerations) are published as supplementary VizieR tables for the individual galaxy papers:

| Galaxy | Paper | VizieR |
|---|---|---|
| NGC 5765b | Gao et al. 2016, ApJ 817, 128 | `2016yCat..18170128G` |
| CGCG 074-064 | Pesce et al. 2020, ApJ 890, 118 | published in paper |
| UGC 3789 | Reid et al. 2013, ApJ 767, 154 | published in paper |
| NGC 6264 | Kuo et al. 2013, ApJ 767, 155 | published in paper |
| NGC 6323 | Kuo et al. 2015, ApJ 800, 26 | published in paper |
| NGC 4258 | Reid, Pesce & Riess 2019, ApJL 886, L27 | published in paper |

The CF4 maser table (`DM`, `eDM`, `Vhel`, sky coordinates, group properties) is the compressed output of these fits — it does **not** contain the raw maser spot data.

---

## 2. Inference Framework

### 2.1 Disk model (per galaxy)

The geometric forward model is a **warped Keplerian disk** with a linear radial warp in all three orientation angles:
$$
i(r) = i_0 + \frac{di}{dr}\,r, \qquad
\Omega(r) = \Omega_0 + \frac{d\Omega}{dr}\,r, \qquad
\omega(r) = \omega_0 + \frac{d\omega}{dr}\,r,
$$
where $i$ is inclination, $\Omega$ is position angle, and $\omega$ is the periapsis angle ($r$ in angular units, mas).

For a maser spot at disk coordinates $(r, \varphi)$, the sky-plane position, LOS velocity, and LOS centripetal acceleration are:
$$
x_k = x_0 + r\bigl[\sin\varphi\sin\Omega - \cos\varphi\cos\Omega\cos i\bigr],
$$
$$
y_k = y_0 + r\bigl[\sin\varphi\cos\Omega + \cos\varphi\sin\Omega\cos i\bigr],
$$
$$
v_k = cz_0 + v_\text{rot}(r)\,\sin\varphi\,\sin i, \qquad v_\text{rot}(r) = \sqrt{\frac{GM_\text{BH}}{r\,D}},
$$
$$
a_k = \frac{GM_\text{BH}}{r^2 D^2}\,\cos\varphi\,\sin i.
$$

Here $D$ is the angular diameter distance (Mpc) and $r$ is in angular units (mas), so $r\,D$ gives physical radius. The angular diameter distance **enters in two places with different power**:

- Velocity: $v_\text{rot}^2 \propto M_\text{BH}/(rD)$ — constrains $M_\text{BH}/D$.
- Acceleration: $a \propto M_\text{BH}/(rD)^2$ — constrains $M_\text{BH}/D^2$.
- Position: $r$ angular directly observed.

The combination of all three breaks the $D$–$M_\text{BH}$ degeneracy geometrically.

### 2.2 Likelihood

The three data classes contribute independently:

$$
\ln\mathcal{L} = \ln\mathcal{L}_\text{pos} + \ln\mathcal{L}_\text{acc} + \ln\mathcal{L}_\text{vel},
$$

$$
\ln\mathcal{L}_\text{pos} = -\frac{1}{2}\sum_k\left[\frac{(x_k-X_k)^2}{\sigma_{x,k}^2+\sigma_x^2} + \frac{(y_k-Y_k)^2}{\sigma_{y,k}^2+\sigma_y^2} + \text{log terms}\right],
$$

$$
\ln\mathcal{L}_\text{acc} = -\frac{1}{2}\sum_k\frac{(a_k-A_k)^2}{\sigma_{a,k}^2+\sigma_a^2} + \text{log terms},
$$

$$
\ln\mathcal{L}_\text{vel} = -\frac{1}{2}\sum_k\frac{(v_k-V_k)^2}{\sigma_{v,\text{type}}^2} + \text{log terms},
$$

where $X_k, Y_k, V_k, A_k$ are the model-predicted values and $\sigma_x, \sigma_y, \sigma_a, \sigma_{v,\text{sys}}, \sigma_{v,\text{hv}}$ are **free error floor parameters** (the key advance of Pesce 2020 over earlier MCP papers, which fixed these externally).

### 2.3 Parameters

**Global parameters (16):**
$D$, $M_\text{BH}$, $z_0$, $x_0$, $y_0$, $i_0$, $di/dr$, $\Omega_0$, $d\Omega/dr$, $\omega_0$, $d\omega/dr$, $\sigma_x$, $\sigma_y$, $\sigma_a$, $\sigma_{v,\text{sys}}$, $\sigma_{v,\text{hv}}$.

**Per-spot nuisance parameters:** orbital coordinates $(r_k, \varphi_k)$ for each maser spot. For CGCG 074-064 this yields 348 free parameters against 604 data constraints.

**Priors:** Uniform in $D$: $\mathcal{U}(10, 150)$ Mpc. Uniform in all other global parameters over physically motivated ranges. Uniform or Gaussian for error floors (choice shifts $D$ by $\sim$1%).

### 2.4 Sampling

Pesce et al. 2020 use **PyMC3 with HMC/NUTS**. Earlier MCP papers (Reid 2013, Kuo 2013, Gao 2016) used a custom Metropolis–Hastings code with fixed error floors. No public code exists; the complete mathematical specification is in Appendices A–B of arXiv:2001.04581.

### 2.5 H$_0$ inference (two-stage)

The H$_0$ inference is entirely separate from the disk fitting. Inputs are:

- The marginal posterior $P(\hat{D}_i \mid D_i)$ for each of the 6 galaxies, taken from the individual disk fits.
- The SMBH CMB-frame redshift $z_{0,i}$ from the disk fit (uncertainty $\sim$1 km\,s$^{-1}$, negligible).

The H$_0$ likelihood is:
$$
\mathcal{L}_{H_0} = \prod_i P(\hat{D}_i \mid D_i(H_0, z_i)) \times \prod_i \mathcal{N}\!\left(\hat{v}_i \mid v_i,\, \sqrt{\sigma_{v,i}^2 + \sigma_\text{pec}^2}\right),
$$
where $D_i(H_0, z_i) = (c/H_0)\int_0^{z_i} dz / E(z)$ and $\sigma_\text{pec} = 250$ km\,s$^{-1}$ is added in quadrature as a fixed peculiar velocity uncertainty. Sampling uses **dynesty** (nested sampling). The prior on $H_0$ is flat.

---

## 3. Limitations

### 3.1 Distance prior not volume-weighted

The prior on $D$ is flat in $D$ (uniform in Mpc), not flat in volume ($\propto D^2$). For a survey with a well-defined selection volume, the correct uninformative prior for a distance is the volume prior $p(D) \propto D^2$. Using a flat-in-$D$ prior biases individual distance posteriors toward smaller $D$, which in turn biases $H_0$ high. For $\sim$10% per-galaxy distance uncertainties this effect is small but non-negligible when combining 6 galaxies.

### 3.2 No treatment of Malmquist bias

The GBT survey from which MCP targets are drawn is flux-limited. Galaxies with intrinsically more luminous (or geometrically more favourable) maser systems are preferentially included. No correction for this selection bias is made, nor is it discussed. The word "Malmquist" does not appear in either Pesce 2020 paper. Whether this biases $D$ or $H_0$ has never been studied.

### 3.3 No treatment of geometric selection bias

Disk megamaser distances require a nearly perfectly edge-on disk ($i \gtrsim 85°$), a bright systemic group with measurable accelerations, and a well-ordered Keplerian rotation curve. The fraction of AGN meeting all these criteria is small ($\lesssim 3\%$ of surveyed Seyfert 2s). Whether the subsample that passes these cuts is statistically representative in terms of distance or peculiar velocity is unknown and uncorrected.

### 3.4 Peculiar velocities treated as noise, not modelled

Peculiar velocities are not drawn from a physical prior (e.g., a reconstructed velocity field). They are instead absorbed into a fixed additive uncertainty $\sigma_\text{pec} = 250$ km\,s$^{-1}$ added in quadrature to each galaxy's velocity. This is conservative but discards all information from peculiar velocity surveys. With only 6 galaxies at $50$–$150$ Mpc, a $250$ km\,s$^{-1}$ peculiar velocity is a $\sim$0.5–1% fractional velocity perturbation, making this the dominant uncertainty term for each galaxy.

### 3.5 No Lauer-type bias discussion

No correction is made for the tendency of flux-limited samples to preferentially include galaxies with anomalously high luminosities or favourable orientations (the Lauer et al. 2007 bias). This is distinct from the Malmquist bias but similarly unaddressed.

### 3.6 Tiny sample size

The full MCP sample used in Pesce 2020 is only **6 galaxies** (including NGC 4258 as the near-field anchor). The statistical power of the combined $H_0$ measurement is therefore limited and the result is sensitive to outliers. The paper checks this only with a jackknife (remove one galaxy at a time), finding no single galaxy dominates, but this does not constitute a bias analysis.

### 3.7 No public code

There is no public code for the disk-fitting pipeline. The complete mathematical specification of the model is available in Appendices A–B of arXiv:2001.04581, which is sufficient to re-implement from scratch, but no reference implementation exists.

---

## References

```bibtex
@ARTICLE{2020ApJ...891L...1P,
  author  = {{Pesce}, D.~W. and {Braatz}, J.~A. and {Reid}, M.~J. and {Riess}, A.~G.
             and {Scolnic}, D. and {Condon}, J.~J. and {Gao}, F. and {Henkel}, C.
             and {Impellizzeri}, C.~M.~V. and {Kuo}, C.~Y. and {Lo}, K.~Y.},
  title   = "{The Megamaser Cosmology Project. XIII. Combined Hubble Constant Constraints}",
  journal = {\apjl},
  year    = 2020, volume = {891}, pages = {L1},
  doi     = {10.3847/2041-8213/ab75f0}, eprint = {2001.09213}
}

@ARTICLE{2020ApJ...890..118P,
  author  = {{Pesce}, D.~W. and {Braatz}, J.~A. and {Reid}, M.~J. and {Condon}, J.~J.
             and {Gao}, F. and {Henkel}, C. and {Kuo}, C.~Y. and {Lo}, K.~Y. and {Zhao}, W.},
  title   = "{The Megamaser Cosmology Project. XI. A Geometric Distance to CGCG 074-064}",
  journal = {\apj},
  year    = 2020, volume = {890}, pages = {118},
  doi     = {10.3847/1538-4357/ab6bcd}, eprint = {2001.04581}
}

@ARTICLE{2016ApJ...817..128G,
  author  = {{Gao}, F. and {Braatz}, J.~A. and {Reid}, M.~J. and {Lo}, K.~Y. and others},
  title   = "{The Megamaser Cosmology Project. VIII. A Geometric Distance to NGC 5765b}",
  journal = {\apj},
  year    = 2016, volume = {817}, pages = {128},
  doi     = {10.3847/0004-637X/817/2/128}, eprint = {1511.08311}
}

@ARTICLE{2013ApJ...767..154R,
  author  = {{Reid}, M.~J. and {Braatz}, J.~A. and {Condon}, J.~J. and {Lo}, K.~Y. and others},
  title   = "{The Megamaser Cosmology Project. IV. A Direct Measurement of the Hubble Constant from UGC 3789}",
  journal = {\apj},
  year    = 2013, volume = {767}, pages = {154},
  doi     = {10.1088/0004-637X/767/2/154}, eprint = {1207.7292}
}

@ARTICLE{2019ApJ...886L..27R,
  author  = {{Reid}, M.~J. and {Pesce}, D.~W. and {Riess}, A.~G.},
  title   = "{An Improved Distance to NGC 4258 and Its Implications for the Hubble Constant}",
  journal = {\apjl},
  year    = 2019, volume = {886}, pages = {L27},
  doi     = {10.3847/2041-8213/ab552d}, eprint = {1908.05625}
}

@ARTICLE{2023ApJ...944...94T,
  author  = {{Tully}, R.~B. and {Kourkchi}, E. and {Courtois}, H.~M. and others},
  title   = "{Cosmicflows-4}",
  journal = {\apj},
  year    = 2023, volume = {944}, pages = {94},
  doi     = {10.3847/1538-4357/ac94d8}, eprint = {2209.11238}
}
```
