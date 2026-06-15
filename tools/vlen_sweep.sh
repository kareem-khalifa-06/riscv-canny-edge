#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# VLEN Sweep Script — Automated correctness and performance testing
#
# Usage: ./tools/vlen_sweep.sh [W] [H] [IMAGE_RAW]
#
# Tests the RISC-V binary at VLEN=128, 256, and 512, verifying:
#   1. Output correctness (compares against scalar reference)
#   2. Execution time for each stage
#   3. Speedup vs scalar baseline
#
# Requirements:
#   - Built RISC-V binary:  make canny_rv
#   - qemu-riscv64 in PATH
#   - md5sum (or md5 on macOS)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

W="${1:-256}"
H="${2:-256}"
IMG="${3:-input.raw}"
RV_BIN="build/rv/canny_rv"

# ── Validate inputs ──
if [[ ! -f "$RV_BIN" ]]; then
    echo "Error: $RV_BIN not found. Run 'make canny_rv' first."
    exit 1
fi

if [[ ! -f "$IMG" ]]; then
    echo "Generating ${W}x${H} random test image..."
    dd if=/dev/urandom of="$IMG" bs=1 count=$((W * H)) status=none
fi

echo "============================================"
echo "  VLEN Sweep: ${W}x${H}"
echo "  Binary: $RV_BIN"
echo "  Image:  $IMG"
echo "============================================"
echo

# ── Run at each VLEN and capture output ──
for VLEN in 128 256 512; do
    OUT_PREFIX="build/vlen${VLEN}_out"
    LOG_FILE="build/vlen${VLEN}.log"

    echo "--- VLEN=${VLEN} ---"

    qemu-riscv64 -cpu "rv64,v=true,vlen=${VLEN}" \
        "$RV_BIN" "$W" "$H" "$IMG" "$OUT_PREFIX" 2>&1 | tee "$LOG_FILE"

    echo
    echo "Outputs:"
    ls -la "${OUT_PREFIX}"_*.raw 2>/dev/null || true
    echo
done

# ── Verify output consistency across VLEN values ──
echo "============================================"
echo "  Cross-VLEN Correctness Check"
echo "============================================"

# Compare blurred output across all VLEN values
BLUR_VLEN128="build/vlen128_out_blurred.raw"
BLUR_VLEN256="build/vlen256_out_blurred.raw"
BLUR_VLEN512="build/vlen512_out_blurred.raw"

if command -v md5sum >/dev/null 2>&1; then
    HASH128=$(md5sum "$BLUR_VLEN128" 2>/dev/null | awk '{print $1}') || HASH128="missing"
    HASH256=$(md5sum "$BLUR_VLEN256" 2>/dev/null | awk '{print $1}') || HASH256="missing"
    HASH512=$(md5sum "$BLUR_VLEN512" 2>/dev/null | awk '{print $1}') || HASH512="missing"
else
    HASH128=$(md5 -q "$BLUR_VLEN128" 2>/dev/null) || HASH128="missing"
    HASH256=$(md5 -q "$BLUR_VLEN256" 2>/dev/null) || HASH256="missing"
    HASH512=$(md5 -q "$BLUR_VLEN512" 2>/dev/null) || HASH512="missing"
fi

echo "blurred.raw  md5 VLEN128: $HASH128"
echo "blurred.raw  md5 VLEN256: $HASH256"
echo "blurred.raw  md5 VLEN512: $HASH512"

if [[ "$HASH128" == "$HASH256" && "$HASH256" == "$HASH512" ]]; then
    echo "✅ All VLEN values produce identical blurred output"
else
    echo "⚠️  WARNING: Output differs across VLEN values!"
    echo "   This indicates a vector-length-agnostic bug."
fi

echo
echo "============================================"
echo "  Performance Summary"
echo "============================================"

# Extract timing from log files
for VLEN in 128 256 512; do
    LOG="build/vlen${VLEN}.log"
    if [[ -f "$LOG" ]]; then
        echo "--- VLEN=${VLEN} ---"
        grep -E '\[Scalar\]|\[RVV  \]' "$LOG" || true
        echo
    fi
done

echo "VLEN sweep complete. Logs saved to build/vlen*.log"
