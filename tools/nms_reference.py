#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Python Reference Implementation: NMS + Thresholding
#
# This is a clean, readable reference for the bonus Canny stages.
# Use it to verify your C++ output against a known-good implementation.
#
# Usage:
#   python3 nms_reference.py <mag.raw> <dir.raw> <w> <h> <prefix> [low] [high]
#
# Dependencies: numpy (pip install numpy)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import numpy as np

DIRECTION_VECTORS = {
    0: ( 1,  0),   # horizontal gradient → check left/right
    1: ( 1,  1),   # 45° → check (x+1,y+1) and (x-1,y-1)
    2: ( 0,  1),   # vertical gradient → check top/bottom
    3: (-1,  1),   # 135° → check (x-1,y+1) and (x+1,y-1)
}

def non_maximum_suppression(mag, dir_map, w, h):
    """Scalar reference NMS matching the C++ algorithm exactly."""
    out = np.zeros((h, w), dtype=np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            d = dir_map[y, x]
            dx, dy = DIRECTION_VECTORS[d]

            center = mag[y, x]
            n1 = mag[y - dy, x - dx]
            n2 = mag[y + dy, x + dx]

            if center > n1 and center > n2:
                out[y, x] = center
            else:
                out[y, x] = 0

    return out


def double_threshold(mag, w, h, low, high):
    """Classify pixels as STRONG, WEAK, or NONE."""
    out = np.zeros((h, w), dtype=np.uint8)
    out[mag >= high] = 255   # EDGE_STRONG
    out[(mag >= low) & (mag < high)] = 128  # EDGE_WEAK
    return out


def hysteresis(img, w, h):
    """Propagate strong connectivity through weak pixels."""
    changed = True
    iterations = 0

    while changed:
        changed = False
        iterations += 1
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 128:  # only weak pixels can be promoted
                    continue

                # Check 8-connected neighbours
                neighbourhood = img[y-1:y+2, x-1:x+2]
                if 255 in neighbourhood:  # connected to strong
                    img[y, x] = 255
                    changed = True

    # Demote remaining weak pixels
    img[img == 128] = 0
    return img


def threshold_and_hysteresis(mag, w, h, low, high):
    """Full thresholding pipeline."""
    classified = double_threshold(mag, w, h, low, high)
    return hysteresis(classified.copy(), w, h)


def main():
    if len(sys.argv) < 6:
        print(f"Usage: {sys.argv[0]} <mag.raw> <dir.raw> <w> <h> <prefix> [low] [high]")
        sys.exit(1)

    mag_path = sys.argv[1]
    dir_path = sys.argv[2]
    w = int(sys.argv[3])
    h = int(sys.argv[4])
    prefix = sys.argv[5]

    mag = np.fromfile(mag_path, dtype=np.uint8).reshape(h, w)
    dir_map = np.fromfile(dir_path, dtype=np.uint8).reshape(h, w)

    # Auto thresholds or user-specified
    if len(sys.argv) >= 8:
        low = int(sys.argv[6])
        high = int(sys.argv[7])
    else:
        max_mag = mag.max()
        high = max(int(max_mag * 0.15), 10)
        low = max(int(max_mag * 0.05), 5)
        print(f"Auto thresholds: low={low}, high={high} (max_mag={max_mag})")

    # NMS
    nms = non_maximum_suppression(mag, dir_map, w, h)
    nms_path = f"{prefix}_nms_ref.raw"
    nms.tofile(nms_path)
    print(f"NMS output:  {nms_path}")

    # Thresholding
    edges = threshold_and_hysteresis(mag, w, h, low, high)
    edges_path = f"{prefix}_edges_ref.raw"
    edges.tofile(edges_path)
    print(f"Edges output: {edges_path}")

    # Stats
    strong = np.sum(edges == 255)
    total = w * h
    print(f"\nEdge pixels: {strong} / {total} ({100*strong/total:.2f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
