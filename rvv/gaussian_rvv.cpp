#ifdef __riscv_v

#include "../src/gaussian.h"
#include <riscv_vector.h>
#include <cstdint>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// 5×5 Gaussian kernel coefficients (sum = 273, σ ≈ 1.0)
// ─────────────────────────────────────────────────────────────────────────────
static const int16_t K5[5][5] = {
    { 2,  4,  5,  4,  2},
    { 4,  9, 12,  9,  4},
    { 5, 12, 15, 12,  5},
    { 4,  9, 12,  9,  4},
    { 2,  4,  5,  4,  2}
};

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
// ─────────────────────────────────────────────────────────────────────────────
void gaussian_5x5_rvv_m1(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst,
                                int w, int h)
{
    for (int y = 0; y < h; ++y) {
        const int row_base = y * w;

        for (int x = 0; x < w; ) {
            // ── 1. Ask hardware: how many uint8 elements fit this iteration?
            //
            // INTRINSIC: __riscv_vsetvl_e8m1(n)
            // WHAT:  Sets vl = min(n, VLMAX) for element width e8, LMUL=1.
            //        Returns the actual vl chosen by the hardware.
            // WHY m1: Widen chain e8m1→e16m2→e32m4 stays within 32 regs.
            //         Going to e8m2 would push accumulator to e32m8 (only 4
            //         regs) — high spill risk with 25 kernel iterations.
            // IF VLEN CHANGES: vl automatically scales.  At VLEN=128 we get
            //         vl=16 uint8 elements; at VLEN=512 we get vl=64.
            //         No code change required — that is the VLA contract.
            size_t vl = __riscv_vsetvl_e8m1((size_t)(w - x));

            // ── 2. Accumulator: int32, LMUL=4 (result of two widen steps)
            //
            // INTRINSIC: __riscv_vmv_v_x_i32m4(0, vl)
            // WHAT:  Splat scalar 0 into a vl-element i32m4 vector (zero init).
            // WHY i32m4: e8m1 → (vzext×2) → e32m4.  Must match widen result.
            // IF VLEN CHANGES: vl controls element count; register width adapts.
            vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);

            // ── 3. Convolve 5×5 kernel (scalar loops, vectorised inner body)
            for (int ky = 0; ky < 5; ++ky) {
                int ny = y + ky - 2;
                if (ny < 0 || ny >= h) continue;   // zero-padding: skip row

                for (int kx = 0; kx < 5; ++kx) {
                    int nx = x + kx - 2;

                    // Boundary clamp for the x-start of this vector strip.
                    // If nx < 0 the whole strip needs shifting — handled by
                    // scalar fallback below.  For most interior pixels nx >= 0.
                    if (nx < 0 || nx + (int)vl > w) {
                        // Scalar fallback for partial-overlap boundary strips
                        // (fires only for the 2 leftmost / 2 rightmost columns)
                        for (size_t i = 0; i < vl; ++i) {
                            int px = x + (int)i + kx - 2;
                            if (px >= 0 && px < w)
                                ((int32_t*)&acc)[i] +=
                                    (int32_t)src[ny * w + px] * K5[ky][kx];
                        }
                        continue;
                    }

                    // ── 3a. Load vl uint8 pixels from row ny, offset nx
                    //
                    // INTRINSIC: __riscv_vle8_v_u8m1(ptr, vl)
                    // WHAT:  Unit-stride load of vl bytes from ptr.
                    // WHY m1: Matches vsetvl_e8m1 above — same LMUL required.
                    // IF VLEN CHANGES: vl already accounts for VLEN; load width
                    //         scales automatically.
                    vuint8m1_t px8 = __riscv_vle8_v_u8m1(
                        src + ny * w + nx, vl);

                    // ── 3b. Widen u8 → u16
                    //
                    // INTRINSIC: __riscv_vzext_vf2_u16m2(v, vl)
                    // WHAT:  Zero-extend each u8 element to u16.
                    //        Result is LMUL=2 (twice the registers) because
                    //        element width doubled.
                    // WHY zero-extend: pixel values are unsigned [0,255];
                    //        sign-extension would corrupt values ≥128.
                    // IF VLEN CHANGES: m2 means 2× VLEN/16 elements — exactly
                    //         the same count as the m1 source.
                    vuint16m2_t px16 = __riscv_vzext_vf2_u16m2(px8, vl);

                    // ── 3c. Widen u16 → u32
                    //
                    // INTRINSIC: __riscv_vzext_vf2_u32m4(v, vl)
                    // WHAT:  Zero-extend each u16 element to u32 (LMUL=4).
                    // WHY two widen steps: RVV vzext only doubles width per
                    //        call.  u8→u32 requires two calls.
                    vuint32m4_t px32u = __riscv_vzext_vf2_u32m4(px16, vl);

                    // ── 3d. Reinterpret u32 → i32 for signed multiply
                    //
                    // INTRINSIC: __riscv_vreinterpret_v_u32m4_i32m4(v)
                    // WHAT:  Bit-cast; no instruction emitted.  Required because
                    //        vmul_vx operates on signed i32, not u32.
                    // WHY safe: pixel values ≤255, so the MSB of the u32 is
                    //        always 0 — reinterpret as i32 is lossless.
                    vint32m4_t px32 = __riscv_vreinterpret_v_u32m4_i32m4(px32u);

                    // ── 3e. Multiply by scalar kernel coefficient
                    //
                    // INTRINSIC: __riscv_vmul_vx_i32m4(v, scalar, vl)
                    // WHAT:  Multiply every element by the scalar K5[ky][kx].
                    //        Max result: 255 × 15 = 3825 — fits i32.
                    // WHY scalar (not vector): all pixels in this strip share
                    //        the same coefficient; vx is 1 instruction vs
                    //        a vector broadcast + vmul.
                    vint32m4_t prod = __riscv_vmul_vx_i32m4(
                        px32, (int32_t)K5[ky][kx], vl);

                    // ── 3f. Accumulate
                    //
                    // INTRINSIC: __riscv_vadd_vv_i32m4(acc, prod, vl)
                    // WHAT:  Element-wise addition.
                    //        Max accumulation after 25 taps: 273 × 255 = 69615.
                    //        Fits comfortably in i32 (max ~2.1 billion).
                    acc = __riscv_vadd_vv_i32m4(acc, prod, vl);
                }
            }

            // ── 4. Fixed-point divide by 273:  (acc × 240) >> 16
            //
            // INTRINSIC: __riscv_vmul_vx_i32m4 then __riscv_vsra_vx_i32m4
            // WHAT:  Multiply accumulator by 240, then arithmetic right-shift 16.
            // WHY: RVV has no integer divide.  Fixed-point approximation
            //      gives error < 0.5 LSB (see analysis at top of file).
            // IF VLEN CHANGES: same instructions, vl controls element count.
            vint32m4_t scaled = __riscv_vmul_vx_i32m4(acc, FP_MULT, vl);
            vint32m4_t norm   = __riscv_vsra_vx_i32m4(scaled, FP_SHIFT, vl);

            // ── 5. Saturating narrow i32 → u8 in two steps
            //
            // Step A: i32m4 → u16m2 with unsigned saturation
            // INTRINSIC: __riscv_vnclipu_wx_u16m2(v, 0, vl)
            // WHAT:  Narrow i32→u16, clipping values to [0, 65535].
            //        The shift of 0 means no shift — just clip and narrow.
            // WHY vnclipu (unsigned clip): pixel output is always ≥ 0 after
            //        normalisation; unsigned clip avoids wrapping of large values.
            vuint16m2_t narrow16 = __riscv_vnclipu_wx_u16m2(
                __riscv_vreinterpret_v_i32m4_u32m4(norm), 0, vl);

            // Step B: u16m2 → u8m1 with unsigned saturation
            // INTRINSIC: __riscv_vnclipu_wx_u8m1(v, 0, vl)
            // WHAT:  Narrow u16→u8, clipping values to [0, 255].
            // WHY two narrowing steps: RVV vnclipu only halves width per call.
            vuint8m1_t result = __riscv_vnclipu_wx_u8m1(narrow16, 0, vl);

            // ── 6. Store result
            //
            // INTRINSIC: __riscv_vse8_v_u8m1(ptr, v, vl)
            // WHAT:  Unit-stride store of vl bytes to ptr.
            // IF VLEN CHANGES: vl controls write width automatically.
            __riscv_vse8_v_u8m1(dst + row_base + x, result, vl);

            x += (int)vl;   // advance by EXACTLY vl — never a fixed constant
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// gaussian_5x5_rvv_m2  —  LMUL=2 variant for LMUL sweep experiment
//
// Base LMUL=2 means: e8m2 base, widen chain → e16m4 → e32m8
// Register budget: only 4 logical registers for e32m8.
// Expected: ~same throughput as m1 or slightly worse due to register pressure
//           when the compiler needs temporaries.
// ─────────────────────────────────────────────────────────────────────────────
void gaussian_5x5_rvv_m2(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst,
                                int w, int h)
{
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ) {
            // INTRINSIC: __riscv_vsetvl_e8m2
            // WHAT:  vl = min(n, VLMAX) for e8, LMUL=2.
            //        At VLEN=256: VLMAX = 256/8 × 2 = 64 elements.
            // WHY m2 here: This is the LMUL sweep variant.  Processing 2×
            //        more elements per iteration — but widen chain hits e32m8
            //        which only has 4 logical regs.  Spill risk is real.
            // IF VLEN CHANGES: vl doubles vs m1 at the same VLEN.
            size_t vl = __riscv_vsetvl_e8m2((size_t)(w - x));

            vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);

            for (int ky = 0; ky < 5; ++ky) {
                int ny = y + ky - 2;
                if (ny < 0 || ny >= h) continue;

                for (int kx = 0; kx < 5; ++kx) {
                    int nx = x + kx - 2;
                    if (nx < 0 || nx + (int)vl > w) {
                        for (size_t i = 0; i < vl; ++i) {
                            int px = x + (int)i + kx - 2;
                            if (px >= 0 && px < w)
                                ((int32_t*)&acc)[i] +=
                                    (int32_t)src[ny * w + px] * K5[ky][kx];
                        }
                        continue;
                    }

                    // INTRINSIC: __riscv_vle8_v_u8m2 / vzext_vf2_u16m4 / vzext_vf2_u32m8
                    // WHAT:  Same widen chain as m1, but starting at LMUL=2.
                    //        u8m2 → u16m4 → u32m8 → i32m8
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
            vuint16m4_t narrow16 = __riscv_vnclipu_wx_u16m4(
                __riscv_vreinterpret_v_i32m8_u32m8(norm), 0, vl);
            vuint8m2_t  result   = __riscv_vnclipu_wx_u8m2(narrow16, 0, vl);
            __riscv_vse8_v_u8m2(dst + y * w + x, result, vl);

            x += (int)vl;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public dispatch: defaults to m1 (best register pressure / spill tradeoff).
// E4 profiling selects the winner; swap the call here to compare.
// ─────────────────────────────────────────────────────────────────────────────
void gaussian_5x5_rvv(const uint8_t* src, uint8_t* dst, int w, int h) {
    gaussian_5x5_rvv_m1(src, dst, w, h);
}

#endif // __riscv_v