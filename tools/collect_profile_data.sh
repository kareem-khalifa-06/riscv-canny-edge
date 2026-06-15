#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Profiling Data Collection Script
#
# Collects per-stage timing data for the optimization report.
# Runs the RISC-V binary at a given VLEN and extracts structured timing info.
#
# Usage: ./tools/collect_profile_data.sh [VLEN] [W] [H] [IMAGE_RAW]
# Output: build/profile_vlen{N}.json  (structured timing data)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VLEN="${1:-256}"
W="${2:-256}"
H="${3:-256}"
IMG="${4:-input.raw}"
RV_BIN="build/rv/canny_rv"
OUT_DIR="build"

mkdir -p "$OUT_DIR"

if [[ ! -f "$RV_BIN" ]]; then
    echo "Error: $RV_BIN not found. Run 'make canny_rv' first."
    exit 1
fi

if [[ ! -f "$IMG" ]]; then
    echo "Generating ${W}x${H} test image..."
    dd if=/dev/urandom of="$IMG" bs=1 count=$((W * H)) status=none
fi

OUT_PREFIX="${OUT_DIR}/profile_vlen${VLEN}"
LOG="${OUT_PREFIX}.log"
JSON="${OUT_PREFIX}.json"

echo "Collecting profile data at VLEN=${VLEN}..."

qemu-riscv64 -cpu "rv64,v=true,vlen=${VLEN}" \
    "$RV_BIN" "$W" "$H" "$IMG" "$OUT_PREFIX" > "$LOG" 2>&1

# Parse timing data from log
echo "Parsing timing data..."

# Helper: extract ms value from a log line
extract_ms() {
    local label="$1"
    grep "$label" "$LOG" | sed -E 's/.*:\s+([0-9.]+)\s+ms.*/\1/' | head -1
}

gauss_scalar=$(extract_ms "\[Scalar\] Gaussian Blur")
sobel_scalar=$(extract_ms "\[Scalar\] Sobel Gradient")
mag_l1_scalar=$(extract_ms "\[Scalar\] Magnitude L1")
mag_l2_scalar=$(extract_ms "\[Scalar\] Magnitude L2")
dir_scalar=$(extract_ms "\[Scalar\] Direction")
total_scalar=$(extract_ms "\[Scalar\] Total pipeline")

gauss_rvv=$(extract_ms "\[RVV  \] Gaussian Blur")
sobel_rvv=$(extract_ms "\[RVV  \] Sobel Gradient")
mag_rvv=$(extract_ms "\[RVV  \] Magnitude L1")
total_rvv=$(extract_ms "\[RVV  \] Total pipeline")

# Write JSON
cat > "$JSON" << EOF
{
  "vlen": ${VLEN},
  "image_width": ${W},
  "image_height": ${H},
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "scalar": {
    "gaussian_ms": ${gauss_scalar:-0},
    "sobel_ms": ${sobel_scalar:-0},
    "magnitude_l1_ms": ${mag_l1_scalar:-0},
    "magnitude_l2_ms": ${mag_l2_scalar:-0},
    "direction_ms": ${dir_scalar:-0},
    "total_ms": ${total_scalar:-0}
  },
  "rvv": {
    "gaussian_ms": ${gauss_rvv:-0},
    "sobel_ms": ${sobel_rvv:-0},
    "magnitude_l1_ms": ${mag_rvv:-0},
    "total_ms": ${total_rvv:-0}
  }
}
EOF

echo "Profile data saved to $JSON"
cat "$JSON"
