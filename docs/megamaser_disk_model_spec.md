# Megamaser Disk Model: Mathematical Specification

Reference: Pesce et al. (2020), "The Megamaser Cosmology Project. XI. A geometric distance to CGCG 074-064" (arXiv:2001.04581).

This document contains the complete mathematical specification of the warped Keplerian disk model used in the Megamaser Cosmology Project (MCP) for geometric distance measurements. It is intended as the reference for implementing the forward model in JAX/NumPyro.

---

## 1. Coordinate System

### 1.1 Disk frame

Each maser spot is assigned a location $(r, \phi)$ in the disk, where:
- $r$ is the orbital radius measured from the SMBH (in mas on the sky, convertible to physical units via the angular-diameter distance $D$),
- $\phi$ is the azimuthal angle measured from the line of sight, oriented such that:
  - systemic features are at $\phi \approx 0^\circ$,
  - redshifted features are at $\phi \approx 90^\circ$,
  - blueshifted features are at $\phi \approx -90^\circ$ (equivalently $\approx 270^\circ$).

### 1.2 Sky frame

The sky-plane position is $(x, y)$ with:
- $x$-axis aligned with right ascension (positive to the east),
- $y$-axis aligned with declination (positive to the north),
- $z$-axis directed along the line of sight (positive $z$ points away from us).

### 1.3 Angles

- **Inclination** $i$: the angle the disk normal makes with respect to the line of sight. $i = 90^\circ$ corresponds to perfectly edge-on.
- **Position angle** $\Omega$: the angle that the receding portion of the disk midplane makes east of north (i.e., clockwise down from the $y$-axis).

Both $i$ and $\Omega$ are functions of $r$ (see Section 2).

### 1.4 Rotation matrices: disk frame to sky frame

The transformation from disk-frame coordinates to sky-frame coordinates is a product of two rotations: first by $i$ about the $x$-axis, then by $\Omega$ about the $z$-axis.

The disk-frame position vector (before rotation) assumes a pre-rotation disk orientation of $i = \Omega = 90^\circ$:

$$\begin{pmatrix} r\sin(\phi) \\ 0 \\ -r\cos(\phi) \end{pmatrix}$$

The full transformation is:

$$\begin{pmatrix} x \\ y \\ z \end{pmatrix} = \underbrace{\begin{pmatrix} \sin(\Omega) & -\cos(\Omega) & 0 \\ \cos(\Omega) & \sin(\Omega) & 0 \\ 0 & 0 & 1 \end{pmatrix}}_{R_z(\Omega)} \underbrace{\begin{pmatrix} 1 & 0 & 0 \\ 0 & \sin(i) & -\cos(i) \\ 0 & \cos(i) & \sin(i) \end{pmatrix}}_{R_x(i)} \begin{pmatrix} r\sin(\phi) \\ 0 \\ -r\cos(\phi) \end{pmatrix} \tag{A4}$$

### 1.5 Sky-frame position (expanded)

After accounting for the BH location $(x_0, y_0)$:

$$x = x_0 + r\bigl[\sin(\phi)\sin(\Omega) - \cos(\phi)\cos(\Omega)\cos(i)\bigr] \tag{A5a}$$

$$y = y_0 + r\bigl[\sin(\phi)\cos(\Omega) + \cos(\phi)\sin(\Omega)\cos(i)\bigr] \tag{A5b}$$

$$z = -r\cos(\phi)\sin(i) \tag{A5c}$$

Note: the $z$-coordinate of the BH itself is fixed at $z = 0$ by construction.

---

## 2. Warped Geometry

The disk geometry is parameterized as linear functions of orbital radius $r$:

$$i(r) = i_0 + \frac{di}{dr}\,r \tag{A1}$$

$$\Omega(r) = \Omega_0 + \frac{d\Omega}{dr}\,r \tag{A2}$$

$$\omega(r) = \omega_0 + \frac{d\omega}{dr}\,r \tag{A3}$$

where $\omega$ is the argument of periapsis (relevant only for eccentric orbits).

The modeled geometric parameters are: $i_0$, $\frac{di}{dr}$, $\Omega_0$, $\frac{d\Omega}{dr}$, $\omega_0$, $\frac{d\omega}{dr}$.

---

## 3. Keplerian Velocity

### 3.1 Circular velocity

The Keplerian circular velocity at radius $r$ is:

$$v(r) = \sqrt{\frac{G M_\text{BH}}{r D}} \tag{A10}$$

where $r$ is in angular units (mas) and $D$ is the angular-diameter distance (Mpc). The product $rD$ gives the physical radius.

### 3.2 Eccentric orbit decomposition

For a maser spot at $(r, \psi)$ where $\psi = \phi - \omega$ is the true anomaly and $e$ is the orbital eccentricity:

**Radial component:**

$$v_r = v(r)\,\frac{e\sin(\psi)}{\sqrt{1 + e\cos(\psi)}} \tag{A8}$$

**Azimuthal component:**

$$v_\phi = v(r)\,\sqrt{1 + e\cos(\psi)} \tag{A9}$$

For circular orbits ($e = 0$): $v_r = 0$ and $v_\phi = v(r)$.

---

## 4. Velocity Projection to Sky Frame

### 4.1 Sky-frame velocity components

$$v_x = v_\phi\bigl[\cos(\phi)\sin(\Omega) + \sin(\phi)\cos(\Omega)\cos(i)\bigr] + v_r\bigl[\sin(\phi)\sin(\Omega) - \cos(\phi)\cos(\Omega)\cos(i)\bigr] \tag{A11a}$$

$$v_y = v_\phi\bigl[\cos(\phi)\cos(\Omega) - \sin(\phi)\sin(\Omega)\cos(i)\bigr] + v_r\bigl[\sin(\phi)\cos(\Omega) + \cos(\phi)\sin(\Omega)\cos(i)\bigr] \tag{A11b}$$

$$v_z = v_\phi\sin(\phi)\sin(i) - v_r\cos(\phi)\sin(i) \tag{A11c}$$

### 4.2 Relativistic Doppler redshift

The redshift imparted by the relativistic Doppler effect is:

$$1 + z_D = \gamma\left(1 - \frac{v}{c}\cos(\theta)\right) \tag{A12}$$

where $\gamma = (1 - v^2/c^2)^{-1/2}$ is the Lorentz factor and $\theta$ is the angle between the velocity vector $\vec{v}$ and the line of sight $(-\hat{z})$.

Taking $\cos(\theta) = \vec{v}\cdot(-\hat{z})/v$, this evaluates to:

$$1 + z_D = \gamma\left(1 + \frac{1}{c}\bigl[v_\phi\sin(\phi)\sin(i) - v_r\cos(\phi)\sin(i)\bigr]\right) \tag{A13}$$

The term in square brackets is $v_z$ (the line-of-sight velocity component, with positive meaning recession).

### 4.3 Gravitational redshift

In Schwarzschild spacetime, the gravitational redshift of a photon emitted at radius $r$ (angular) from a SMBH and received at infinity is:

$$1 + z_g = \left(1 - \frac{R_s}{rD}\right)^{-1/2} \tag{A14}$$

where $R_s = 2GM_\text{BH}/c^2$ is the Schwarzschild radius. Note $rD$ is the physical radius.

### 4.4 Total observed redshift

The observed redshift of the maser spot combines the Doppler shift, gravitational redshift, and the cosmological redshift of the SMBH ($z_0$, measured in the CMB frame):

$$1 + z = (1 + z_D)(1 + z_g)(1 + z_0) \tag{A15}$$

### 4.5 Observed velocity (optical convention)

$$v_\text{obs} = cz \tag{A16}$$

---

## 5. Centripetal Acceleration

### 5.1 Acceleration magnitude

The centripetal acceleration at orbital radius $r$ (angular) is:

$$a(r) = \frac{GM_\text{BH}}{r^2 D^2} \tag{A7}$$

where $r$ is in angular units and $D$ is the angular-diameter distance, so $r^2 D^2$ gives the physical radius squared.

### 5.2 Sky-frame acceleration components

$$a_x = a\bigl[-\sin(\phi)\sin(\Omega) + \cos(\phi)\cos(\Omega)\cos(i)\bigr] \tag{A6a}$$

$$a_y = a\bigl[-\sin(\phi)\cos(\Omega) - \cos(\phi)\sin(\Omega)\cos(i)\bigr] \tag{A6b}$$

$$a_z = a\cos(\phi)\sin(i) \tag{A6c}$$

Note: the acceleration vector points radially inward (toward the BH), which is the negative of the position offset direction (compare signs with Eq. A5a--c).

The measured line-of-sight acceleration is $a_z$.

---

## 6. Likelihood Function

For each data point $k$, we have measurements of:
- on-sky position $(x_k, y_k)$,
- line-of-sight velocity $v_k$,
- line-of-sight acceleration $a_k$.

Each measurement class is treated independently. Lowercase letters denote measured values; uppercase letters denote modeled values.

### 6.1 Position likelihood $\mathcal{L}_1$

Individual spot likelihoods are Gaussian with error floors $\sigma_x$, $\sigma_y$ added in quadrature:

$$\ell_{x,k} = \frac{1}{\sqrt{2\pi(\sigma_{x,k}^2 + \sigma_x^2)}}\exp\left[-\frac{1}{2}\frac{(x_k - X_k)^2}{\sigma_{x,k}^2 + \sigma_x^2}\right] \tag{B17}$$

$$\ell_{y,k} = \frac{1}{\sqrt{2\pi(\sigma_{y,k}^2 + \sigma_y^2)}}\exp\left[-\frac{1}{2}\frac{(y_k - Y_k)^2}{\sigma_{y,k}^2 + \sigma_y^2}\right] \tag{B18}$$

Joint log-likelihood:

$$\ln(\mathcal{L}_1) = -\frac{1}{2}\sum_k\left[\frac{(x_k - X_k)^2}{\sigma_{x,k}^2 + \sigma_x^2} + \frac{(y_k - Y_k)^2}{\sigma_{y,k}^2 + \sigma_y^2} + \ln\bigl[2\pi(\sigma_{x,k}^2 + \sigma_x^2)\bigr] + \ln\bigl[2\pi(\sigma_{y,k}^2 + \sigma_y^2)\bigr]\right] \tag{B19}$$

where $\sigma_{x,k}$, $\sigma_{y,k}$ are per-spot measurement uncertainties (from beam fitting, Eq. 1 of the paper) and $\sigma_x$, $\sigma_y$ are global error floor parameters.

### 6.2 Acceleration likelihood $\mathcal{L}_2$

$$\ell_{a,k} = \frac{1}{\sqrt{2\pi(\sigma_{a,k}^2 + \sigma_a^2)}}\exp\left[-\frac{1}{2}\frac{(a_k - A_k)^2}{\sigma_{a,k}^2 + \sigma_a^2}\right] \tag{B20}$$

Joint log-likelihood:

$$\ln(\mathcal{L}_2) = -\frac{1}{2}\sum_k\left[\frac{(a_k - A_k)^2}{\sigma_{a,k}^2 + \sigma_a^2} + \ln\bigl[2\pi(\sigma_{a,k}^2 + \sigma_a^2)\bigr]\right] \tag{B21}$$

The sum runs only over maser spots with measured accelerations. For spots with unmeasured accelerations ($N_a = 20$ such spots for CGCG 074-064), the acceleration measurement is excluded from the likelihood entirely, and the acceleration becomes a fitted nuisance parameter.

### 6.3 Velocity likelihood $\mathcal{L}_3$

Two different error floors are used depending on the maser type:

$$\ell_{v,k} = \begin{cases} \frac{1}{\sqrt{2\pi\sigma_{v,\text{sys}}^2}}\exp\left(-\frac{(v_k - V_k)^2}{2\sigma_{v,\text{sys}}^2}\right) & \text{for systemic features} \\[6pt] \frac{1}{\sqrt{2\pi\sigma_{v,\text{hv}}^2}}\exp\left(-\frac{(v_k - V_k)^2}{2\sigma_{v,\text{hv}}^2}\right) & \text{for high-velocity features} \end{cases} \tag{B22}$$

Joint log-likelihood:

$$\ln(\mathcal{L}_3) = \begin{cases} -\frac{1}{2}\sum_k\left[\frac{(v_k - V_k)^2}{\sigma_{v,\text{sys}}^2} + \ln(2\pi\sigma_{v,\text{sys}}^2)\right] & \text{for systemic features} \\[6pt] -\frac{1}{2}\sum_k\left[\frac{(v_k - V_k)^2}{\sigma_{v,\text{hv}}^2} + \ln(2\pi\sigma_{v,\text{hv}}^2)\right] & \text{for high-velocity features} \end{cases} \tag{B23}$$

Note: there are no per-spot measurement uncertainties for velocities --- the velocity of each spot is determined by the spectral channel center, so the entire uncertainty budget is captured by the error floor parameters $\sigma_{v,\text{sys}}$ and $\sigma_{v,\text{hv}}$.

### 6.4 Total likelihood

$$\ln(\mathcal{L}) = \ln(\mathcal{L}_1) + \ln(\mathcal{L}_2) + \ln(\mathcal{L}_3) \tag{B24}$$

---

## 7. Unit Conventions and Conversion Constants

### 7.1 Working units

| Quantity | Unit |
|---|---|
| Orbital radius $r$ | mas (milliarcseconds) |
| Angular-diameter distance $D$ | Mpc |
| SMBH mass $M_\text{BH}$ | $M_\odot$ |
| Velocity $v$ | km/s |
| Acceleration $a$ | km/s/yr |
| Position $(x, y)$ | mas |
| Position angle $\Omega$, inclination $i$ | degrees |
| Warp rates $di/dr$, $d\Omega/dr$ | degrees/mas |
| Error floors $\sigma_x$, $\sigma_y$ | mas |
| Error floors $\sigma_{v,\text{sys}}$, $\sigma_{v,\text{hv}}$ | km/s |
| Error floor $\sigma_a$ | km/s/yr |

### 7.2 Physical radius

The physical radius of a maser spot is:

$$r_\text{phys} = r \times D$$

where $r$ is in mas and $D$ in Mpc. To get SI:

$$r_\text{phys}\;\text{[m]} = r\;\text{[rad]} \times D\;\text{[m]} = r \times \frac{\pi}{180 \times 3600 \times 1000} \times D \times 3.0857 \times 10^{22}$$

### 7.3 Velocity: $v(r) = \sqrt{GM_\text{BH}/(rD)}$

We need consistent units. With $r$ in mas, $D$ in Mpc, $M_\text{BH}$ in $M_\odot$, and $v$ in km/s:

$$v(r) = \sqrt{\frac{GM_\text{BH}}{rD}} \quad\text{[km/s]}$$

where the product $rD$ must be converted to metres:

$$rD\;\text{[m]} = r\;\text{[mas]} \times \frac{\pi}{180 \times 3600 \times 1000} \times D\;\text{[Mpc]} \times 3.08568 \times 10^{22}\;\text{[m/Mpc]}$$

$$= r \times D \times 1.49598 \times 10^{14}\;\text{[m]}$$

(using $\pi/(180 \times 3.6 \times 10^6) \times 3.08568 \times 10^{22} = 4.84814 \times 10^{-9} \times 3.08568 \times 10^{22} = 1.49598 \times 10^{14}$).

Then:

$$v(r) = \sqrt{\frac{G M_\text{BH}\;\text{[kg]}}{rD\;\text{[m]}}} \times 10^{-3}\;\text{[km/s per m/s]}$$

With $G = 6.674 \times 10^{-11}$ m$^3$ kg$^{-1}$ s$^{-2}$ and $M_\odot = 1.989 \times 10^{30}$ kg:

$$v(r) = 10^{-3}\sqrt{\frac{6.674 \times 10^{-11} \times M_\text{BH} \times 1.989 \times 10^{30}}{r \times D \times 1.49598 \times 10^{14}}}$$

$$= 10^{-3}\sqrt{\frac{1.327 \times 10^{20} \times M_\text{BH}}{1.49598 \times 10^{14} \times r \times D}}$$

$$= 10^{-3}\sqrt{\frac{8.872 \times 10^{5} \times M_\text{BH}}{r \times D}}$$

$$= \sqrt{\frac{8.872 \times 10^{-1} \times M_\text{BH}}{r \times D}}\;\text{[km/s]}$$

**Convenient form:** define the velocity constant

$$C_v \equiv \sqrt{\frac{GM_\odot}{\text{1 mas} \times \text{1 Mpc}}} = \sqrt{\frac{6.674 \times 10^{-11} \times 1.989 \times 10^{30}}{1.49598 \times 10^{14}}} = 9.4196 \times 10^{2}\;\text{m/s}$$

Then:

$$v(r) = C_v \times \sqrt{\frac{M_\text{BH}}{r \times D}} \times 10^{-3}\;\text{[km/s]} = 0.94196\,\sqrt{\frac{M_\text{BH}}{r \times D}}\;\text{[km/s]}$$

where $M_\text{BH}$ is in $M_\odot$, $r$ in mas, $D$ in Mpc.

**Sanity check:** For CGCG 074-064, $M_\text{BH} = 2.42 \times 10^7\,M_\odot$, $D = 87.6$ Mpc, $r = 0.3$ mas:

$$v \approx 0.942 \times \sqrt{\frac{2.42 \times 10^7}{0.3 \times 87.6}} \approx 0.942 \times \sqrt{9.21 \times 10^5} \approx 0.942 \times 960 \approx 904\;\text{km/s} \quad\checkmark$$

(consistent with observed orbital velocities of 400--1000 km/s).

### 7.4 Acceleration: $a(r) = GM_\text{BH}/(r^2 D^2)$

With $r$ in mas, $D$ in Mpc, $M_\text{BH}$ in $M_\odot$, and $a$ in km/s/yr:

$$a(r) = \frac{GM_\text{BH}\;\text{[m}^3/\text{s}^2\text{]}}{(rD)^2\;\text{[m}^2\text{]}} \times \frac{1\;\text{yr}}{10^3\;\text{m/s per km/s}} = \frac{GM_\odot \times M_\text{BH}}{(rD\;\text{[m]})^2} \times \frac{3.156 \times 10^7}{10^3}$$

$$= \frac{6.674 \times 10^{-11} \times 1.989 \times 10^{30} \times M_\text{BH}}{(1.49598 \times 10^{14})^2 \times r^2 \times D^2} \times 3.156 \times 10^4$$

$$= \frac{1.327 \times 10^{20} \times M_\text{BH} \times 3.156 \times 10^4}{2.238 \times 10^{28} \times r^2 \times D^2}$$

$$= \frac{1.872 \times 10^{-4} \times M_\text{BH}}{r^2 \times D^2}\;\text{[km/s/yr]}$$

**Convenient form:** define the acceleration constant

$$C_a = \frac{GM_\odot \times (1\;\text{yr})}{\text{(1 mas} \times \text{1 Mpc)}^2 \times 10^3} = 1.872 \times 10^{-4}\;\text{[km/s/yr]}$$

Then:

$$a(r) = C_a \times \frac{M_\text{BH}}{r^2 \times D^2}\;\text{[km/s/yr]}$$

where $M_\text{BH}$ is in $M_\odot$, $r$ in mas, $D$ in Mpc.

### 7.5 Schwarzschild radius

$$R_s = \frac{2GM_\text{BH}}{c^2}$$

In metres: $R_s = 2 \times 6.674 \times 10^{-11} \times M_\text{BH} \times 1.989 \times 10^{30} / (3 \times 10^8)^2 = 2.953 \times 10^3 \times M_\text{BH}$ [m] with $M_\text{BH}$ in $M_\odot$.

The ratio $R_s/(rD)$ appearing in the gravitational redshift is dimensionless:

$$\frac{R_s}{rD} = \frac{2GM_\text{BH}}{c^2 \times rD\;\text{[m]}} = \frac{2.953 \times 10^3 \times M_\text{BH}}{1.49598 \times 10^{14} \times r \times D} = \frac{1.974 \times 10^{-11} \times M_\text{BH}}{r \times D}$$

with $M_\text{BH}$ in $M_\odot$, $r$ in mas, $D$ in Mpc.

### 7.6 Summary of conversion constants

| Constant | Value | Formula |
|---|---|---|
| 1 mas $\times$ 1 Mpc in metres | $1.49598 \times 10^{14}$ m | $\frac{\pi}{180 \times 3.6 \times 10^6} \times 3.08568 \times 10^{22}$ |
| $C_v = \sqrt{GM_\odot / (1\;\text{mas}\cdot 1\;\text{Mpc})}$ | 942.0 m/s = 0.9420 km/s | velocity prefactor |
| $C_a = GM_\odot \cdot (1\;\text{yr}) / [(1\;\text{mas}\cdot 1\;\text{Mpc})^2 \cdot 10^3]$ | $1.872 \times 10^{-4}$ km/s/yr | acceleration prefactor |
| $C_g = 2GM_\odot / [c^2 \cdot (1\;\text{mas}\cdot 1\;\text{Mpc})]$ | $1.974 \times 10^{-11}$ | $R_s/(rD)$ prefactor |

---

## 8. Parameter List and Priors

### 8.1 Global disk parameters (16 total)

| Parameter | Symbol | Units | Prior | Description |
|---|---|---|---|---|
| Angular-diameter distance | $D$ | Mpc | $\mathcal{U}(10, 150)$ | Distance to galaxy |
| SMBH mass | $M_\text{BH}$ | $10^7\,M_\odot$ | $\mathcal{U}(0.1, 10.0)$ | Black hole mass |
| Systemic velocity | $v_0$ | km/s | $\mathcal{U}(6500, 7500)$ | LOS velocity of SMBH (barycentric); note $v_0 = cz_0 - 263.3$ km/s for CMB frame |
| BH x-position | $x_0$ | mas | $\mathcal{U}(-0.5, 0.5)$ | RA offset of BH |
| BH y-position | $y_0$ | mas | $\mathcal{U}(-0.5, 0.5)$ | Dec offset of BH |
| Inclination at $r=0$ | $i_0$ | degree | $\mathcal{U}(70, 110)$ | Disk inclination |
| Position angle at $r=0$ | $\Omega_0$ | degree | $\mathcal{U}(0, 180)$ | Disk position angle |
| PA warp rate | $d\Omega/dr$ | degree/mas | $\mathcal{U}(-100, 100)$ | Position angle gradient |
| Periapsis angle at $r=0$ | $\omega_0$ | --- | (fitted if eccentric) | Argument of periapsis |
| Periapsis warp rate | $d\omega/dr$ | --- | (fitted if eccentric) | Periapsis gradient |
| Inclination warp rate | $di/dr$ | --- | (fitted if eccentric/warped) | Inclination gradient |
| Eccentricity | $e$ | --- | (fitted if eccentric) | Orbital eccentricity |

Note: for the fiducial model in Pesce+2020 (CGCG 074-064), the warp in inclination direction ($di/dr$) is set to zero, eccentricity $e = 0$, and the periapsis parameters $\omega_0$, $d\omega/dr$ are not used.

### 8.2 Error floor parameters (5 total)

| Parameter | Symbol | Units | Prior | Description |
|---|---|---|---|---|
| x-position error floor | $\sigma_x$ | mas | $\mathcal{U}(0.0, 0.1)$ | Additional position scatter in RA |
| y-position error floor | $\sigma_y$ | mas | $\mathcal{U}(0.0, 0.1)$ | Additional position scatter in Dec |
| Systemic velocity error floor | $\sigma_{v,\text{sys}}$ | km/s | $\mathcal{U}(0, 20)$ | Velocity uncertainty for systemic masers |
| High-velocity error floor | $\sigma_{v,\text{hv}}$ | km/s | $\mathcal{U}(0, 20)$ | Velocity uncertainty for red/blueshifted masers |
| Acceleration error floor | $\sigma_a$ | km/s/yr | $\mathcal{U}(0, 20)$ | Additional acceleration scatter |

### 8.3 Per-spot nuisance parameters

| Parameter | Symbol | Units | Prior | Count |
|---|---|---|---|---|
| Orbital radius | $r_k$ | mas | $\mathcal{U}(0.1, 1.5)$ | $N_r + N_b + N_s$ |
| Azimuthal angle | $\phi_k$ | rad | see below | $N_r + N_b + N_s$ |
| Acceleration (unmeasured) | $a_k$ | km/s/yr | fitted | $N_a$ (spots without measured acceleration) |

**Azimuthal angle priors by maser type:**

| Maser type | $\phi$ range |
|---|---|
| Redshifted | $[0, \pi]$ |
| Blueshifted | $[\pi, 2\pi]$ |
| Systemic | $[-\pi/2, \pi/2]$ |

### 8.4 Total parameter count

For $N_r$ redshifted, $N_b$ blueshifted, and $N_s$ systemic spots, and $N_a$ spots with unmeasured accelerations:

- Global parameters: 16 (or fewer if warping/eccentricity disabled)
- Per-spot parameters: $2(N_r + N_b + N_s) + N_a$
- Total measurements: $4(N_r + N_b + N_s) - N_a$ (each spot has $x, y, v, a$ but $N_a$ accelerations are missing)
- Degrees of freedom: $4(N_r + N_b + N_s) - N_a - N_\text{params}$

For CGCG 074-064: $N_r = 71$, $N_b = 50$, $N_s = 45$, $N_a = 20$, giving 348 parameters, 604 constraints, and 256 degrees of freedom.

### 8.5 Prior definitions

Uniform prior:

$$\mathcal{U}(a,b) = \begin{cases} \frac{1}{b-a} & a \leq \Theta \leq b \\ 0 & \text{otherwise} \end{cases} \tag{B25}$$

Normal (Gaussian) prior:

$$\mathcal{N}(\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{1}{2}\left(\frac{\Theta - \mu}{\sigma}\right)^2\right] \tag{B26}$$

---

## 9. H0 Connection

### 9.1 Disk model outputs

The disk model directly constrains:
- the angular-diameter distance $D$ (in Mpc),
- the SMBH redshift $z_0$ (in the CMB frame).

The model parameterizes $z_0$ via the systemic velocity $v_0$ in the barycentric frame. The conversion for CGCG 074-064 is:

$$v_\text{CMB} = v_\text{bary} + 263.3\;\text{km/s}$$

and the CMB-frame redshift is (optical convention):

$$z_0 = v_\text{CMB} / c$$

### 9.2 H0 from D and z_0

Assuming a flat $\Lambda$CDM cosmology with matter density parameter $\Omega_m$:

$$H_0 = \frac{c}{D(1 + z_0)} \int_0^{z_0} \frac{dz}{\sqrt{\Omega_m(1+z)^3 + (1-\Omega_m)}} \tag{Eq.~7}$$

This uses the angular-diameter distance relation:

$$D_A(z_0) = \frac{c}{H_0(1+z_0)} \int_0^{z_0} \frac{dz}{\sqrt{\Omega_m(1+z)^3 + (1-\Omega_m)}}$$

so that $H_0 = c \int_0^{z_0} [\ldots] dz / [D(1+z_0)]$.

Pesce+2020 use $\Omega_m = 0.315$ (Planck 2018). For CGCG 074-064, applying this formula yields $H_0$ about 2.7% lower than the simple estimate $H_0 = cz_0/D$.

### 9.3 Peculiar velocity correction

To account for the peculiar motion of the host galaxy, $z_0$ in Eq. 7 is replaced by the flow-corrected redshift $z_\text{flow}$ from the Cosmicflows-3 database:

$$v_\text{flow} = cz_\text{flow}$$

For CGCG 074-064: $v_\text{flow} = 7308 \pm 150$ km/s, implying a peculiar velocity of ~136 km/s.

### 9.4 Redshift decomposition

The observed redshift of each maser spot is:

$$1 + z = (1 + z_D)(1 + z_g)(1 + z_0) \tag{A15}$$

where:
- $z_D$ is the relativistic Doppler redshift from orbital motion (Eq. A13),
- $z_g$ is the gravitational redshift from the SMBH potential (Eq. A14),
- $z_0$ is the cosmological redshift of the SMBH in the CMB frame.

---

## 10. Maser Spot Types

### 10.1 Classification

Maser spots are classified into three groups based on their line-of-sight velocity relative to the systemic velocity of the galaxy:

| Type | Label | Velocity range (CGCG 074-064) | Physical location |
|---|---|---|---|
| Systemic | s | 6880--6975 km/s | Near the disk midline, viewed along the line of sight ($\phi \approx 0$) |
| Blueshifted | b | 6120--6405 km/s | Approaching side of disk ($\phi \approx 270^\circ$, i.e. $\phi \in [\pi, 2\pi]$) |
| Redshifted | r | 7580--7780 km/s | Receding side of disk ($\phi \approx 90^\circ$, i.e. $\phi \in [0, \pi]$) |

### 10.2 Azimuthal angle constraints

The $\phi$ prior ranges enforce the physical location constraint:

- **Redshifted**: $\phi \in [0, \pi]$ -- the receding half of the disk
- **Blueshifted**: $\phi \in [\pi, 2\pi]$ -- the approaching half of the disk
- **Systemic**: $\phi \in [-\pi/2, \pi/2]$ -- near the midline where the velocity is close to systemic

### 10.3 Velocity error floor differences

The velocity error floors differ between systemic and high-velocity (red + blue) features:

- **Systemic features** use $\sigma_{v,\text{sys}}$: these masers amplify background continuum emission, producing narrow, well-defined spectral features, so the effective velocity uncertainty may differ from the high-velocity features.
- **High-velocity features** use $\sigma_{v,\text{hv}}$: these masers are thermal (not amplifying background), and their spectral properties may differ.

Both error floors capture uncertainties from: (1) spectral channel discretization, (2) maser lines spanning multiple channels, and (3) velocity drift over the ~4-month observing baseline.

### 10.4 Acceleration characteristics

- **Systemic features**: show roughly constant acceleration of ~4.4 km/s/yr (RMS 0.66 km/s/yr), as expected for masers near the disk midline where the acceleration is directed along the line of sight.
- **Redshifted features**: mean acceleration ~0.06 km/s/yr (RMS 0.65 km/s/yr), consistent with zero.
- **Blueshifted features**: mean acceleration ~-0.35 km/s/yr (RMS 1.11 km/s/yr), consistent with zero.
- Both high-velocity feature groups have accelerations consistent with zero, as expected for masers located near the midplane of the disk (where the line-of-sight component of centripetal acceleration vanishes).

### 10.5 Spot counts (CGCG 074-064)

| Type | Count | Measured accelerations |
|---|---|---|
| Redshifted | $N_r = 71$ | most |
| Blueshifted | $N_b = 50$ | most |
| Systemic | $N_s = 45$ | most |
| Unmeasured accel. | $N_a = 20$ | 0 (fitted as nuisance) |
| **Total** | **166** | **146** |

---

## 11. Forward Model Summary (for implementation)

Given global parameters $\{D, M_\text{BH}, v_0, x_0, y_0, i_0, \Omega_0, d\Omega/dr, \sigma_x, \sigma_y, \sigma_{v,\text{sys}}, \sigma_{v,\text{hv}}, \sigma_a\}$ and per-spot parameters $\{r_k, \phi_k\}$ (and optionally $\{e, \omega_0, d\omega/dr, di/dr, a_k\}$):

**For each maser spot $k$:**

1. Compute warped geometry: $i_k = i(r_k)$, $\Omega_k = \Omega(r_k)$, $\omega_k = \omega(r_k)$.
2. Compute sky-frame position: $(X_k, Y_k)$ from Eqs. (A5a, A5b).
3. Compute Keplerian velocity: $v(r_k) = C_v \sqrt{M_\text{BH}/(r_k D)}$.
4. Compute velocity components: $v_r$, $v_\phi$ from Eqs. (A8, A9) with $\psi_k = \phi_k - \omega_k$.
5. Project to sky frame: $(v_x, v_y, v_z)$ from Eqs. (A11a--c).
6. Compute Doppler redshift: $1 + z_D$ from Eq. (A13).
7. Compute gravitational redshift: $1 + z_g$ from Eq. (A14).
8. Compute total redshift: $1 + z = (1+z_D)(1+z_g)(1+z_0)$ from Eq. (A15).
9. Compute observed velocity: $V_k = cz$ from Eq. (A16).
10. Compute acceleration magnitude: $a(r_k) = C_a M_\text{BH}/(r_k^2 D^2)$.
11. Compute LOS acceleration: $A_k = a(r_k)\cos(\phi_k)\sin(i_k)$ from Eq. (A6c).
12. Evaluate log-likelihood contributions from Eqs. (B19, B21, B23).

**Posterior:**

$$\ln P(\Theta|\mathbf{D}) = \ln\mathcal{L}(\Theta) + \ln\pi(\Theta) + \text{const} \tag{Eq.~3}$$

---

## 12. Position Uncertainty Model

The per-spot position uncertainties $(\sigma_{x,k}, \sigma_{y,k})$ are derived from the restoring beam dimensions and the signal-to-noise ratio:

$$\sigma_x \approx \frac{1}{2}\frac{\Delta_x}{\text{S/N}} \tag{Eq.~1}$$

where $\Delta_x$ is the FWHM of the restoring beam along the relevant axis. Only spots with measured S/N $\geq 3$ are retained.
