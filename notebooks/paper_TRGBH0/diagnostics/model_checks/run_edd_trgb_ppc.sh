#!/bin/bash -l
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUTDIR="$ROOT/notebooks/paper_TRGBH0/output/model_checks"
PYTHON="${CANDEL_PYTHON:-$ROOT/venv_candel/bin/python}"
# PPC_FACTOR="${CANDEL_PPC_FACTOR:-10}"
PPC_FACTOR=10
# MANTICORE_FIELD_INDEX="${CANDEL_MANTICORE_FIELD_INDEX:-10}"
MANTICORE_FIELD_INDEX=68
printf -v MANTICORE_FIELD_TAG "field%02d" "$MANTICORE_FIELD_INDEX"
B_MIN="${CANDEL_B_MIN:-10}"
B_MIN_TAG="${B_MIN//./p}"
PPC_N_WORKERS="${CANDEL_PPC_N_WORKERS:-${SLURM_CPUS_PER_TASK:-${PBS_NP:-${NSLOTS:-4}}}}"
RESOURCE_TRACKER_WARNING="ignore:resource_tracker:UserWarning:multiprocessing.resource_tracker"
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}$RESOURCE_TRACKER_WARNING"

mkdir -p "$OUTDIR"
echo "[INFO] PPC workers: $PPC_N_WORKERS"

# No reconstruction: Vext-only Gaussian cz likelihood.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode none \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --output "$OUTDIR/trgbh0_edd_trgb_vext_only_bmin${B_MIN_TAG}_gaussian_ppc.pdf"

# # Carrick2015 reconstruction: Gaussian cz likelihood.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode carrick \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --output "$OUTDIR/trgbh0_edd_trgb_carrick_bmin${B_MIN_TAG}_gaussian_ppc.pdf"

# Carrick2015 reconstruction: Gaussian cz likelihood with sky exposure.
"$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
    --mode carrick \
    --posterior "$ROOT/results/TRGBH0_paper/table/EDD_TRGB_sel-TRGB_magnitude_bmin10_skyhp_nside2_k192_Carrick2015_main.hdf5" \
    --b-min "$B_MIN" \
    --ppc-factor "$PPC_FACTOR" \
    --n-workers "$PPC_N_WORKERS" \
    --sky-mask-nside 2 \
    --sky-mask-kappa 192 \
    --output "$OUTDIR/trgbh0_edd_trgb_carrick_bmin${B_MIN_TAG}_gaussian_skyhp_nside2_k192_ppc.pdf"

# Carrick2015 reconstruction: Gaussian cz likelihood with sky exposure.
"$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
    --mode carrick \
    --posterior "$ROOT/results/TRGBH0_paper/table/EDD_TRGB_sel-TRGB_magnitude_bmin10_skyhp_nside1_k48_Carrick2015_main.hdf5" \
    --b-min "$B_MIN" \
    --ppc-factor "$PPC_FACTOR" \
    --n-workers "$PPC_N_WORKERS" \
    --sky-mask-nside 1 \
    --sky-mask-kappa 48 \
    --output "$OUTDIR/trgbh0_edd_trgb_carrick_bmin${B_MIN_TAG}_gaussian_skyhp_nside1_k48_ppc.pdf"

# Carrick2015 reconstruction: Vmono Student-t cz likelihood posterior.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode carrick \
#     --vmono \
#     --posterior "$ROOT/results/TRGBH0_paper/table/EDD_TRGB_Vmono_cz-student_t_sel-TRGB_magnitude_bmin10_Carrick2015_main.hdf5" \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --output "$OUTDIR/trgbh0_edd_trgb_carrick_vmono_bmin${B_MIN_TAG}_student_t_ppc.pdf"

# Carrick2015 reconstruction: Vmono+Voct Student-t cz likelihood posterior.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode carrick \
#     --vmono \
#     --voct \
#     --posterior "$ROOT/results/TRGBH0_paper/table/EDD_TRGB_Vmono_Voct_cz-student_t_sel-TRGB_magnitude_bmin10_Carrick2015_main.hdf5" \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --sky-mask-nside 1 \
#     --sky-mask-kappa 48 \
#     --output "$OUTDIR/trgbh0_edd_trgb_carrick_vmono_voct_bmin${B_MIN_TAG}_student_t_skyhp_nside1_k48_ppc.pdf"

# Carrick2015 reconstruction: Student-t cz likelihood.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode carrick \
#     --posterior "$ROOT/results/TRGBH0_paper/table/EDD_TRGB_cz-student_t_sel-TRGB_magnitude_Carrick2015_main.hdf5" \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --output "$OUTDIR/trgbh0_edd_trgb_carrick_bmin${B_MIN_TAG}_student_t_ppc.pdf"

# ManticoreLocalCOLA field reconstruction: Student-t cz likelihood.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode manticore \
#     --field-index "$MANTICORE_FIELD_INDEX" \
#     --posterior "$ROOT/results/TRGBH0_paper/single_fields/EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_ManticoreLocalCOLA_${MANTICORE_FIELD_TAG}_single.hdf5" \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --output "$OUTDIR/trgbh0_edd_trgb_manticore_${MANTICORE_FIELD_TAG}_bmin${B_MIN_TAG}_student_t_ppc.pdf"

# ManticoreLocalCOLA field reconstruction: Gaussian cz likelihood.
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode manticore \
#     --field-index "$MANTICORE_FIELD_INDEX" \
#     --b-min "$B_MIN" \
#     --ppc-factor "$PPC_FACTOR" \
#     --output "$OUTDIR/trgbh0_edd_trgb_manticore_${MANTICORE_FIELD_TAG}_bmin${B_MIN_TAG}_gaussian_ppc.pdf"
