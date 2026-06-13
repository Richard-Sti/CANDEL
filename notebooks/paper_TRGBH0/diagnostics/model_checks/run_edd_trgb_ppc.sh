#!/bin/bash -l
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUTDIR="$ROOT/notebooks/paper_TRGBH0/output/model_checks"
PYTHON="${CANDEL_PYTHON:-python}"

mkdir -p "$OUTDIR"

"$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
    --mode none \
    --output "$OUTDIR/trgbh0_edd_trgb_vext_only_gaussian_ppc.pdf"

# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode carrick \
#     --output "$OUTDIR/trgbh0_edd_trgb_carrick_gaussian_ppc.pdf"
#
# "$PYTHON" "$SCRIPT_DIR/make_edd_trgb_ppc.py" \
#     --mode manticore \
#     --field-index 0 \
#     --output "$OUTDIR/trgbh0_edd_trgb_manticore_field00_gaussian_ppc.pdf"
