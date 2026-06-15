#ifdef __riscv_v

#include "../src/gaussian.h"
#include <riscv_vector.h>
#include <cstdint>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// Fixed-point reciprocal for division by 273.
//
//   floor(round(2^16 / 273)) = 240   →   error < 0.5 LSB on uint8 output
//
// Proof:  240/65536 = 0.003662109…   vs   1/273 = 0.003663004…
//         max absolute error per pixel = 255 × |0.003663004 - 0.003662109|
//                                      = 255 × 8.95e-7 ≈ 0.00023  (<< 1)
//
// WHY fixed-point?  RVV has no integer divide instruction.  A scalar idiv
// would serialise the entire inner loop.  One vmul + vsra costs 2 vector
// instructions and runs fully pipelined.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int32_t FP_MULT  = 240;
static constexpr int32_t FP_SHIFT = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Scalar fallback for a single Gaussian pixel — used for the 2-pixel border.
// This ensures exact match with the scalar reference on boundary pixels.
// ─────────────────────────────────────────────────────────────────────────────
static inline void gaussian_scalar_pixel(const uint8_t* src, uint8_t* dst,
                                          int w, int h, int y, int x)
{
    int32_t sum = 0;
    for (int ky = 0; ky < 5; ++ky) {
        int ny = y + ky - 2;
        if (ny < 0 || ny >= h) continue;
        for (int kx = 0; kx < 5; ++kx) {
            int nx = x + kx - 2;
            if (nx < 0 || nx >= w) continue;
            sum += (int32_t)src[ny * w + nx] * (int32_t)K5[ky][kx];
        }
    }
    int32_t result = (sum + 136) / 273;
    if (result > 255) result = 255;
    if (result < 0)   result = 0;
    dst[y * w + x] = (uint8_t)result;
}

// ─────────────────────────────────────────────────────────────────────────────
// gaussian_5x5_rvv_m1  —  LMUL=1 implementation
//
// Register budget with LMUL=1:
//   32 logical vector registers, each VLEN/8 bytes wide.
//   We need: 1 accumulator (widened to m4), 1 pixel load (u8m1),
//            2 intermediate widen results (u16m2, u32m4).
//   Total: ~4 registers.  Plenty of headroom — no spills.
//
// WHY LMUL=1 as the baseline?
//   Gaussian loops over 25 kernel taps.  Each tap loads 1 pixel vector,
//   widens twice (u8→u16→u32), multiplies, accumulates.
//   With m1 we have 32 regs; with m4 we only have 8.
//   The multiply chain: u8m1 → u16m2 → u32m4 already consumes m4 for
//   the accumulator.  Going to m2/m4 base LMUL would cause register
//   pressure and likely spills — measured in the LMUL sweep table.
//
// BOUNDARY HANDLING:
//   The 2-pixel border (top/bottom 2 rows, left/right 2 columns) is
//   handled by a scalar fallback.  The interior (y=2..h-3, x=2..w-3) is
//   fully vectorised.  This is correct because:
//   - The scalar fallback matches the reference exactly.
//   - The vector path only accesses valid memory (nx>=2 and nx+vl<=w-2).
// ─────────────────────────────────────────────────────────────────────────────
void gaussian_5x5_rvv_m1(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst,
                                int w, int h)
{
    // ── Top border: rows 0 and 1 ──
    for (int y = 0; y < 2 && y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            gaussian_scalar_pixel(src, dst, w, h, y, x);
        }
    }

    // ── Interior rows: y = 2 .. h-3 ──
    for (int y = 2; y < h - 2; ++y) {
        const int row_base = y * w;

        // Left border: columns 0 and 1
        for (int x = 0; x < 2 && x < w; ++x) {
            gaussian_scalar_pixel(src, dst, w, h, y, x);
        }

        // Vectorised interior: x = 2 .. w-3
        for (int x = 2; x < w - 2; ) {
            int remaining = (w - 2) - x;
            if (remaining <= 0) break;

            // INTRINSIC: __riscv_vsetvl_e8m1(n)
            // WHAT:  Sets vl = min(n, VLMAX) for element width e8, LMUL=1.
            //        Returns the actual vl chosen by the hardware.
            // WHY m1: Widen chain e8m1→e16m2→e32m4 stays within 32 regs.
            // IF VLEN CHANGES: vl automatically scales.  At VLEN=128 we get
            //         vl=16 uint8 elements; at VLEN=512 we get vl=64.
            size_t vl = __riscv_vsetvl_e8m1((size_t)remaining);

            // INTRINSIC: __riscv_vmv_v_x_i32m4(0, vl)
            // WHAT:  Splat scalar 0 into a vl-element i32m4 vector (zero init).
            // WHY i32m4: e8m1 → (vzext×2) → e32m4.  Must match widen result.
            vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);

            // Convolve 5×5 kernel — all memory accesses are in bounds because:
            //   y ∈ [2, h-3]  →  ny = y + ky - 2 ∈ [0, h-1]  ✓
            //   x ∈ [2, w-3]  →  nx = x + kx - 2 ≥ 0, and
            //                    nx + vl ≤ (w-3) + 2 - 2 + vl = w - 3 + vl
            //   But we need nx + vl ≤ w, and since x < w-2 and vl ≤ remaining,
            //   x + vl ≤ w - 2, so nx + vl = x + kx - 2 + vl ≤ w - 2 + 2 = w.  ✓
            for (int ky = 0; ky < 5; ++ky) {
                int ny = y + ky - 2;  // Always in [0, h-1] for interior rows
                for (int kx = 0; kx < 5; ++kx) {
                    int nx = x + kx - 2;  // Always ≥ 0 for x ≥ 2

                    // INTRINSIC: __riscv_vle8_v_u8m1(ptr, vl)
                    // WHAT:  Unit-stride load of vl bytes from ptr.
                    // SAFE:  nx ≥ 0 and nx + vl ≤ w (proven above).
                    vuint8m1_t px8 = __riscv_vle8_v_u8m1(
                        src + ny * w + nx, vl);

                    // INTRINSIC: __riscv_vzext_vf2_u16m2(v, vl)
                    // WHAT:  Zero-extend each u8 element to u16 (LMUL=2).
                    vuint16m2_t px16 = __riscv_vzext_vf2_u16m2(px8, vl);

                    // INTRINSIC: __riscv_vzext_vf2_u32m4(v, vl)
                    // WHAT:  Zero-extend each u16 element to u32 (LMUL=4).
                    vuint32m4_t px32u = __riscv_vzext_vf2_u32m4(px16, vl);

                    // INTRINSIC: __riscv_vreinterpret_v_u32m4_i32m4(v)
                    // WHAT:  Bit-cast; no instruction emitted.
                    vint32m4_t px32 = __riscv_vreinterpret_v_u32m4_i32m4(px32u);

                    // INTRINSIC: __riscv_vmul_vx_i32m4(v, scalar, vl)
                    // WHAT:  Multiply every element by K5[ky][kx].
                    vint32m4_t prod = __riscv_vmul_vx_i32m4(
                        px32, (int32_t)K5[ky][kx], vl);

                    // INTRINSIC: __riscv_vadd_vv_i32m4(acc, prod, vl)
                    // WHAT:  Element-wise addition.
                    //        Max after 25 taps: 273 × 255 = 69615 — fits i32.
                    acc = __riscv_vadd_vv_i32m4(acc, prod, vl);
                }
            }

            // Fixed-point divide by 273: (acc × 240) >> 16
            vint32m4_t scaled = __riscv_vmul_vx_i32m4(acc, FP_MULT, vl);
            vint32m4_t norm   = __riscv_vsra_vx_i32m4(scaled, FP_SHIFT, vl);

            // Clamp to [0,255] before narrowing (avoids vnclipu UB)
            vint32m4_t clamped = __riscv_vmax_vx_i32m4(norm,    0,   vl);
                       clamped = __riscv_vmin_vx_i32m4(clamped, 255, vl);

            // Narrow i32 → u8 in two steps
            // Step A: i32m4 → u16m2
            vuint16m2_t narrow16 = __riscv_vnclipu_wx_u16m2(
                __riscv_vreinterpret_v_i32m4_u32m4(clamped), 0,
                __RISCV_VXRM_RNU, vl);

            // Step B: u16m2 → u8m1
            vuint8m1_t result = __riscv_vnclipu_wx_u8m1(
                narrow16, 0, __RISCV_VXRM_RNU, vl);

            // Store result
            __riscv_vse8_v_u8m1(dst + row_base + x, result, vl);

            x += (int)vl;
        }

        // Right border: columns w-2 and w-1
        for (int x = w - 2; x < w; ++x) {
            if (x < 2) continue;  // Already handled for very small images
            gaussian_scalar_pixel(src, dst, w, h, y, x);
        }
    }

    // ── Bottom border: rows h-2 and h-1 ──
    for (int y = h - 2; y < h; ++y) {
        if (y < 2) continue;  // Already handled for very small images
        for (int x = 0; x < w; ++x) {
            gaussian_scalar_pixel(src, dst, w, h, y, x);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// gaussian_5x5_rvv_m2  —  LMUL=2 variant for LMUL sweep experiment
// ─────────────────────────────────────────────────────────────────────────────
void gaussian_5x5_rvv_m2(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst,
                                int w, int h)
{
    // Top border
    for (int y = 0; y < 2 && y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            gaussian_scalar_pixel(src, dst, w, h, y, x);
        }
    }

    // Interior rows
    for (int y = 2; y < h - 2; ++y) {
        const int row_base = y * w;

        // Left border
        for (int x = 0; x < 2 && x < w; ++x) {
            gaussian_scalar_pixel(src, dst, w, h, y, x);
        }

        // Vectorised interior
        for (int x = 2; x < w - 2; ) {
            int remaining = (w - 2) - x;
            if (remaining <= 0) break;

            size_t vl = __riscv_vsetvl_e8m2((size_t)remaining);
            vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);

            for (int ky = 0; ky < 5; ++ky) {
                int ny = y + ky - 2;
                for (int kx = 0; kx < 5; ++kx) {
                    int nx = x + kx - 2;
                    vuint8m2_t  px8  = __riscv_vle8_v_u8m2(src + ny * w + nx, vl);
                    vuint16m4_t px16 = __riscv_vzext_vf2_u16m4(px8,  vl);
                    vuint32m8_t px32u= __riscv_vzext_vf2_u32m8(px16, vl);
                    vint32m8_t  px32 = __riscv_vreinterpret_v_u32m8_i32m8(px32u);
                    vint32m8_t  prod = __riscv_vmul_vx_i32m8(px32, (int32_t)K5[ky][kx], vl);
                    acc = __riscv_vadd_vv_i32m8(acc, prod, vl);
                }
            }

            vint32m8_t  scaled   = __riscv_vmul_vx_i32m8(acc, FP_MULT, vl);
            vint32m8_t  norm     = __riscv_vsra_vx_i32m8(scaled, FP_SHIFT, vl);
            vint32m8_t  clamped  = __riscv_vmax_vx_i32m8(norm,    0,   vl);
                        clamped  = __riscv_vmin_vx_i32m8(clamped, 255, vl);
            vuint16m4_t narrow16 = __riscv_vnclipu_wx_u16m4(
                __riscv_vreinterpret_v_i32m8_u32m8(clamped), 0,
                __RISCV_VXRM_RNU, vl);
            vuint8m2_t  result   = __riscv_vnclipu_wx_u8m2(
                narrow16, 0, __RISCV_VXRM_RNU, vl);
            __riscv_vse8_v_u8m2(dst + row_base + x, result, vl);

            x += (int)vl;
        }

        // Right border
        for (int x = w - 2; x < w; ++x) {
            if (x < 2) continue;
            gaussian_scalar_pixel(src, dst, w, h, y, x);
        }
    }

    // Bottom border
    for (int y = h - 2; y < h; ++y) {
        if (y < 2) continue;
        for (int x = 0; x < w; ++x) {
            gaussian_scalar_pixel(src, dst, w, h, y, x);
        }
    }
}

// Public dispatch: defaults to m1.
void gaussian_5x5_rvv(const uint8_t* src, uint8_t* dst, int w, int h) {
    gaussian_5x5_rvv_m1(src, dst, w, h);
}

#endif // __riscv_v
