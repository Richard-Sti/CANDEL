"""Diagnostic: compare log-likelihood at published vs fitted parameters."""
import sys
sys.path.insert(0, "/mnt/users/rstiskalek/CANDEL")

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from candel.model.model_H0_maser import (
    C_v, C_a, C_g, predict_position, predict_velocity_los,
    predict_acceleration_los, warp_geometry, SPEED_OF_LIGHT)

# ---- Load data ----
root = "data/Megamaser"
fname = os.path.join(root, "NGC5765b_Gao2016_table6.dat")

with open(fname) as f:
    lines = f.readlines()

vels, xs, exs, ys, eys, accs, eaccs = [], [], [], [], [], [], []
for line in lines:
    parts = line.split()
    vels.append(float(parts[0]))
    xs.append(float(parts[1]))
    exs.append(float(parts[2]))
    ys.append(float(parts[3]))
    eys.append(float(parts[4]))
    accs.append(float(parts[5]))
    eaccs.append(float(parts[6]))

v = np.array(vels)
x = np.array(xs)
ex = np.array(exs)
y = np.array(ys)
ey = np.array(eys)
a = np.array(accs)
ea = np.array(eaccs)

# Flag unmeasured accelerations
unmeasured = (a == 0.0) & (np.abs(ea - 0.2) < 0.01)
measured_mask = ~unmeasured
print(f"Loaded {len(v)} spots, {measured_mask.sum()} with measured accel")

# Classify
v_sys_approx = 8327.6
v_gap = 200
sys_mask = np.abs(v - v_sys_approx) < v_gap
blue_mask = v < v_sys_approx - v_gap
red_mask = v > v_sys_approx + v_gap
print(f"Systemic: {sys_mask.sum()}, Blue: {blue_mask.sum()}, Red: {red_mask.sum()}")

# ---- Check: what does Gao+2016 report? ----
# Gao+2016 Table 7 best-fit parameters
print("\n=== Gao+2016 published parameters ===")
print("D = 126.3 +/- 11.6 Mpc (angular diameter distance)")
print("M_BH = 4.55 +/- 0.40 x 10^7 Msun")
print("v_sys = 8327.6 +/- 0.7 km/s (heliocentric)")
print("x0 = -0.044 +/- 0.008 mas")
print("y0 = -0.100 +/- 0.033 mas")
print("i0 = 72.4 +/- 0.6 deg")
print("Omega0 = 149.7 +/- 0.5 deg")
print("dOmega/dr = -3.2 +/- 1.2 deg/mas")

# ---- Quick forward model at published params ----
D_A_pub = 126.3
M_BH_pub = 4.55e7
v_sys_pub = 8327.6
x0_pub = -0.044
y0_pub = -0.100
i0_pub = np.deg2rad(72.4)
Omega0_pub = np.deg2rad(149.7)
dOmega_dr_pub = np.deg2rad(-3.2)
di_dr_pub = np.deg2rad(0.0)

# Also Pesce+2020 revised distance
D_A_pesce = 112.2

# Our fitted values
D_A_ours = 160.4
M_BH_ours = 6.04e7
v_sys_ours = 8304.9
x0_ours = -0.014
y0_ours = -0.167
i0_ours = np.deg2rad(79.8)
Omega0_ours = np.deg2rad(152.6)
dOmega_dr_ours = np.deg2rad(-6.3)

# For each spot, we need r and phi (these are nuisance params)
# We can estimate r from the acceleration for systemic spots
# A_z = C_a * M_BH / (r^2 * D^2) * cos(phi) * sin(i)
# For phi ~ 0: A_z ~ C_a * M_BH / (r^2 * D^2) * sin(i)

# Let's just check the predicted vs observed for a rough r estimate
# Use the position to estimate r: for HV spots at phi ~ pi/2 or 3*pi/2,
# r ~ sqrt(x^2 + y^2) (roughly, ignoring projection effects)

print("\n=== Spot position ranges ===")
for label, mask in [("Sys", sys_mask), ("Blue", blue_mask), ("Red", red_mask)]:
    print(f"{label}: x=[{x[mask].min():.4f}, {x[mask].max():.4f}], "
          f"y=[{y[mask].min():.4f}, {y[mask].max():.4f}]")
    r_approx = np.sqrt(x[mask]**2 + y[mask]**2)
    print(f"  r_approx: [{r_approx.min():.4f}, {r_approx.max():.4f}] mas")

# The main check: for systemic masers at phi~0, acceleration constrains D
# a_obs = C_a * M_BH * sin(i) / (r^2 * D^2)  (at phi=0)
# v_obs = v_sys + C_v * sqrt(M_BH/(r*D)) * sin(phi) * sin(i)
# At phi~0, the velocity contribution is ~0

# But the position gives:
# x ~ x0 + r*[sin(phi)*sin(Omega) - cos(phi)*cos(Omega)*cos(i)]
# y ~ y0 + r*[sin(phi)*cos(Omega) + cos(phi)*sin(Omega)*cos(i)]
# At phi~0: x ~ x0 - r*cos(Omega)*cos(i), y ~ y0 + r*sin(Omega)*cos(i)

# So for systemic spots, position constrains r (given i0 and Omega0)
# And then acceleration constrains M_BH/D^2

print("\n=== Predicted positions for systemic spots (phi=0) ===")
for label, D_A, M_BH, x0, y0, i0, Omega0 in [
    ("Gao+2016", D_A_pub, M_BH_pub, x0_pub, y0_pub, i0_pub, Omega0_pub),
    ("Ours", D_A_ours, M_BH_ours, x0_ours, y0_ours, i0_ours, Omega0_ours)]:

    cos_i = np.cos(i0)
    cos_O = np.cos(Omega0)
    sin_O = np.sin(Omega0)

    # At phi=0: x_pred = x0 - r*cos(Omega)*cos(i)
    #           y_pred = y0 + r*sin(Omega)*cos(i)
    # So r = (y - y0) / (sin(Omega) * cos(i))
    # Or r = -(x - x0) / (cos(Omega) * cos(i))

    y_sys = y[sys_mask]
    x_sys = x[sys_mask]

    r_from_y = (y_sys - y0) / (sin_O * cos_i)
    r_from_x = -(x_sys - x0) / (cos_O * cos_i)

    print(f"\n{label}:")
    print(f"  cos(i)*sin(Omega) = {cos_i * sin_O:.4f}")
    print(f"  cos(i)*cos(Omega) = {cos_i * cos_O:.4f}")
    print(f"  r from y (phi=0): [{r_from_y.min():.4f}, {r_from_y.max():.4f}]")
    print(f"  r from x (phi=0): [{r_from_x.min():.4f}, {r_from_x.max():.4f}]")

    # Use average r to predict acceleration
    r_avg = np.abs(r_from_y).mean()
    a_pred = C_a * M_BH / (r_avg**2 * D_A**2) * np.sin(i0)
    print(f"  avg |r| = {r_avg:.4f} mas")
    print(f"  pred accel at avg r: {a_pred:.3f} km/s/yr (obs mean: {a[sys_mask].mean():.3f})")

    # Predicted v_kep at avg r
    v_kep = C_v * np.sqrt(M_BH / (r_avg * D_A))
    print(f"  v_kep at avg r: {v_kep:.0f} km/s")

    # Velocity offset for HV spots at phi=pi/2 (redshifted)
    v_offset = v_kep * np.sin(i0)
    print(f"  HV velocity offset (phi=pi/2): +/- {v_offset:.0f} km/s")
    print(f"  Expected red range: {v_sys_pub + v_offset:.0f}")
    print(f"  Observed red max: {v[red_mask].max():.0f}")

# Let me also check: what M_BH and r would match both velocity and
# acceleration simultaneously?
print("\n\n=== Joint velocity + acceleration diagnostic ===")
# For a systemic maser at phi=0, r~r0:
# a_obs = C_a * M_BH * sin(i) / (r^2 * D^2)   ... (1)
# v_obs = v_sys  (since sin(0)=0, no velocity shift)
# Position: y = y0 + r * sin(Omega) * cos(i)    ... (2)
# From (2): r is determined by position alone (given geometry)
# From (1): M_BH/D^2 = a_obs * r^2 / (C_a * sin(i))

# For HV masers at phi=pi/2 (red):
# v_obs = v_sys + C_v * sqrt(M_BH / (r*D)) * sin(i)   ... (3)
# From (3): M_BH/D = (v_obs - v_sys)^2 * r / (C_v^2 * sin^2(i))

# Combining: D = [M_BH/D] / [M_BH/D^2]
# That's the key constraint!

# Let me check what happens with both parameter sets
for label, i0 in [("Gao i=72.4", np.deg2rad(72.4)),
                   ("Ours i=79.8", np.deg2rad(79.8))]:
    sin_i = np.sin(i0)

    # Use observed acceleration mean for systemic spots
    a_mean = a[sys_mask].mean()

    # Typical radius of systemic spots (from position)
    r_sys = 0.8  # mas (rough estimate)

    # M_BH / D^2 from acceleration
    MBH_over_D2 = a_mean * r_sys**2 / (C_a * sin_i)

    # For HV spots, typical velocity offset ~ 750 km/s
    v_offset = 750  # km/s (rough: max red - v_sys ~ 9081-8328 = 753)

    # At a typical HV radius of r_hv ~ 0.5 mas
    r_hv = 0.5  # mas
    MBH_over_D = v_offset**2 * r_hv / (C_v**2 * sin_i**2)

    D_pred = MBH_over_D / MBH_over_D2
    MBH_pred = MBH_over_D * D_pred

    print(f"\n{label}:")
    print(f"  sin(i) = {sin_i:.4f}")
    print(f"  M_BH/D^2 (from a_sys) = {MBH_over_D2:.0f}")
    print(f"  M_BH/D (from v_hv) = {MBH_over_D:.0f}")
    print(f"  Implied D = {D_pred:.1f} Mpc")
    print(f"  Implied M_BH = {MBH_pred:.2e} Msun")
