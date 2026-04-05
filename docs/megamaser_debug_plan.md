# Megamaser Distance Reproduction: Debug Plan

## Status: RESOLVED

## Goal
Reproduce NGC5765b distance from Pesce+2020 (MCP XIII): D = 112.2 +5.4/-5.1 Mpc.
Previous result: D = 160.4 ± 8.3 Mpc (H0 = 52.5 — wrong).
**Fixed result: D_A = 121.7 ± 8.2 Mpc** (consistent with published 112-126 range).

## Published Reference Values (Pesce+2020 MCP XIII Table 1)
| Galaxy | D (Mpc) | v_CMB (km/s) |
|--------|---------|--------------|
| NGC 5765b | 112.2 +5.4/-5.1 | 8525.7 ± 0.7 |
| CGCG 074-064 | 87.6 +7.9/-7.2 | 7172.2 ± 1.9 |
| UGC 3789 | 51.5 +4.5/-4.0 | 3319.9 ± 0.8 |
| NGC 6264 | 132.1 +21/-17 | 10192.6 ± 0.8 |
| NGC 6323 | 109.4 +34/-23 | 7801.5 ± 1.5 |

Original Gao+2016: D = 126.3 ± 11.6 Mpc, M_BH = 4.55 ± 0.40 × 10^7 M_sun

## Identified Issues (Priority Order)

### CRITICAL: Published parameters were wrong in config
Gao+2016 Table 7 actual values:
- i0 = 94.5° ± 0.25° (NOT 72.4° as previously assumed)
- di/dr = -10.6 ± 1.13 deg/mas (LARGE warp!)
- Omega0 = 146.7° ± 0.125°
- dOmega/dr = -3.46 ± 0.43 deg/mas

The 72.4° value comes from Pesce+2020 re-analysis with a different warp
parameterization (di/dr = +12.5 deg/mas). Both agree at maser radii:
- Gao: i(1 mas) = 94.5 - 10.6 = 83.9°
- Pesce: i(1 mas) = 72.4 + 12.5 = 84.9°

The disk axis ratio from the data independently gives i ≈ 83.5° at
characteristic maser radii, confirming this.

### Issue: Previous fit did NOT include di/dr
The saved NGC5765b_real_samples.npz does NOT contain di_dr, meaning
the inclination warp was not fitted. Without it, the model is forced
to use a single inclination for all radii, which cannot fit the warped
disk properly. This causes:
- Wrong inclination (79.8° instead of 94.5° + warp)
- Inflated error floors (sigma_x=92µas, sigma_v_hv=16 km/s)
- Wrong distance (160 Mpc instead of 126 Mpc)

### Issue 2: Data file name mismatch (FIXED)
Created symlink: NGC5765b_Gao2016_table6_tex.dat → NGC5765b_Gao2016_table6.dat

### Issue 3: v_sys_obs is in correct frame
v_sys_obs = 8327.6 km/s is heliocentric, matching the spot velocities.
The CMB correction (~207 km/s) only matters for the H0 inference, not
for the single-galaxy disk fit. So this is NOT an issue for D.

### Issue 4: Acceleration treatment is correct
123/192 spots have unmeasured accelerations (A=0, sigma_a=0.2 sentinel).
Code correctly sets sigma_a → 1e6, excluding them from L2.
40 systemic spots all have measured accelerations — these drive the
distance measurement.

## Action Plan

### Round 1: Fix v_sys_obs and filename
1. Fix data filename: rename or create symlink
2. Determine correct v_sys_obs for NGC5765b
3. Compute heliocentric-to-CMB correction for NGC5765b
4. Run a quick test fit

### Round 2: Verify acceleration handling
1. Check which spots truly have measured accelerations
2. Compare with Gao+2016 paper description
3. Verify the sentinel detection logic in load_NGC5765b_spots

### Round 3: Verify data integrity
1. Compare loaded data against CDS table (VizieR)
2. Check velocity frame (helio vs CMB)
3. Verify spot classification (blue/sys/red)

### Round 4: Run inference and compare
1. Run NUTS with corrected settings
2. Compare D, M_BH, i0, Omega0 with published values
3. Check for convergence issues

### Round 5-10: Iterate on remaining issues
- Adjust priors if needed
- Try other galaxies for cross-validation
- Consider warp model differences
- Check if Pesce+2020 used updated data

## Key Diagnostic
If M_BH/D ≈ const (velocity degeneracy direction), but M_BH/D^2 is wrong,
then accelerations aren't constraining properly. With correct v_sys_obs,
the v_sys parameter should be closer to the true CMB velocity, potentially
shifting the entire fit.
