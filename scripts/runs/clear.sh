#!/bin/bash

set -e

echo "[INFO] Removing .out/.err files from job arrays..."
rm -v *_%A_%a.out *_%A_%a.err 2>/dev/null || true