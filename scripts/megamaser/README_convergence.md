# Maser disk model convergence tests

## Mode 2: five galaxies (NGC5765b, UGC3789, CGCG074-064, NGC6264, NGC6323)

Tests the production integration (Simpson HV + two-cluster systemic +
adaptive r) against a 10001² brute-force reference.

### Run

```bash
addqueue -q gpulong -s -m 16 --gpus 1 \
    /mnt/users/$USER/CANDEL/venv_gpu_candel/bin/python -u \
    scripts/megamaser/convergence_grids.py \
    --galaxies NGC5765b UGC3789 CGCG074-064 NGC6264 NGC6323
```

### What it tests
1. **r-sweep** at default phi: adaptive r at n_local = 51, 101, 151, 251
2. **phi-sweep** at adaptive r=151: G_phi = 51, 101, 201, 401, 801

### Expected results (2026-04-13)
All within 0.08 nats of reference. The 10001² reference is adequate for
these galaxies (position errors ~0.03 mas → ~50 pts per peak).

---

## Mode 1: NGC4258

NGC4258 has ~10× tighter position errors (~0.003 mas), creating razor-thin
peaks in (r, φ). Mode 2 cannot resolve them. Mode 1 (sample r, adaptive
phi) handles this by centering the phi grid at the sampled r where the 2×2
solve is exact.

### Test 1: Per-spot phi accuracy at fixed r

Compare 51-pt adaptive phi vs 100K uniform phi at the MAP r of each spot.
This tests whether the adaptive phi grid resolves the peak when r is correct.

```bash
addqueue -q gpulong -s -m 16 --gpus 1 \
    /mnt/users/$USER/CANDEL/venv_gpu_candel/bin/python -u \
    scripts/megamaser/test_mode1_adaptive_phi.py
```

The MAP r is found by sweeping 2001 r × 10K phi per spot (~10 min on GPU).
Then evaluates adaptive (51 pts) and reference (100K pts) at that r.

**Note:** the MAP r search grid (2001 pts) is coarser than the position
peak (σ_logr ~ 0.0008). For spots where the found r is slightly off, both
adaptive and reference give poor logL, but their DIFFERENCE reflects the phi
accuracy. Focus on spots with reference logL near 0 (well-fit at found r).

### Test 2: Full brute-force reference (100K²)

Pre-computed per-spot reference for all 358 spots. Saved to avoid
recomputing (~40 min on GPU).

```bash
# Compute and save (first time only):
addqueue -q gpulong -s -m 16 --gpus 1 \
    /mnt/users/$USER/CANDEL/venv_gpu_candel/bin/python -u \
    scripts/megamaser/diagnose_residuals_n4258.py \
    --save-ref results/Maser/NGC4258_ll_ref_100k.npy

# Load saved reference for diagnostics:
addqueue -q gpulong -s -m 16 --gpus 1 \
    /mnt/users/$USER/CANDEL/venv_gpu_candel/bin/python -u \
    scripts/megamaser/diagnose_residuals_n4258.py \
    --load-ref results/Maser/NGC4258_ll_ref_100k.npy
```

**Note:** this reference tests Mode 2 (marginalised r+φ) against the
brute-force. For NGC4258, Mode 2 has a ~1000 nat floor from r-grid
resolution. This is why Mode 1 is needed.

### Test 3: Per-spot residual diagnostics (all galaxies)

Per-spot breakdown of model vs reference, identifying which spots and
types (sys+acc, sys, red HV, blue HV) contribute to the error.

```bash
addqueue -q gpulong -s -m 16 --gpus 1 \
    /mnt/users/$USER/CANDEL/venv_gpu_candel/bin/python -u \
    scripts/megamaser/diagnose_residuals.py \
    --galaxies NGC5765b CGCG074-064 UGC3789 NGC6264 NGC6323
```

### Key findings from development (2026-04-13)

**Mode 2 (five galaxies):** all within 0.08 nats after Simpson HV +
two-cluster systemic.

**NGC4258 Mode 2:** -4218 nats error (confirmed with 100K² reference).
- Systemic spots: -4206 nats (99.7%)
- HV spots: -12 nats (negligible)
- Root cause: position peaks σ_φ ~ 0.001 rad, grid spacing ~0.005 rad.
  Per-(spot,r) adaptive phi reduces to -1000 nats but saturates due to
  r-grid ridge-tracking error.

**NGC4258 Mode 1:** per-spot adaptive phi at sampled r is exact (2×2 solve
gives s*²+c*² = 1). Phi is unimodal (front/back solutions 133σ apart).
51 pts suffice. NUTS handles the r exploration via gradients.
