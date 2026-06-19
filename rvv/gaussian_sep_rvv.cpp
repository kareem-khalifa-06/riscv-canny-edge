#ifdef __riscv_v

#include "../src/gaussian_separable.h"
#include <riscv_vector.h>
#include <cstdint>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────────────────────
// gaussian_5x5_sep_rvv
//
// RVV separable 5×5 Gaussian using k1d = [1,4,7,4,1], divide by 289 (= 17²).
//
// TWO-PASS DESIGN
// ───────────────
// Pass 1 (horizontal): each output element = dot(src[y, x-2..x+2], k1d)
//   Output stored as int16_t (max = 255×17 = 4335 < 32767).
//   Vectorised: for each row, load 5 overlapping windows (offsets -2..+2),
//   widen u8→i16, multiply by scalar coefficient, accumulate into i16m2.
//   Border pixels (x < 2 or x > w-3) handled by scalar fallback.
//
// Pass 2 (vertical): each output element = dot(tmp[y-2..y+2, x], k1d) / 289
//   Vectorised: for each interior row, load 5 row-chunks from tmp as i16m1,
//   multiply by scalar coefficient into i32m2 accumulator, then divide by 289,
//   clamp, narrow to u8.
//   Border rows (y < 2 or y > h-3) handled by scalar fallback.
//
// LMUL CHOICES
//   Pass 1: u8m1 loads → widen to i16m2 accumulators. m2 = 16 logical regs,
//           enough for 1 acc + 5 pixel temporaries without spills.
//   Pass 2: i16m1 loads → widen to i32m2 accumulators. m2 = 16 logical regs,
//           enough for 1 acc + 5 row temporaries.
//
// VLEN AGNOSTIC: vsetvl called fresh every iteration, correct at any VLEN.
// ─────────────────────────────────────────────────────────────────────────────

// ── Scalar helpers for border pixels ─────────────────────────────────────────
static inline int16_t pass1_scalar(const uint8_t* src_row, int x, int w)
{
    // Horizontal dot product at (row, x), zero-padding out-of-bounds
    static const int16_t k[5] = {1, 4, 7, 4, 1};
    int32_t sum = 0;
    for (int kx = -2; kx <= 2; ++kx) {
        int ix = x + kx;
        if (ix >= 0 && ix < w)
            sum += (int32_t)src_row[ix] * k[kx + 2];
    }
    return (int16_t)sum;
}

static inline uint8_t pass2_scalar(const int16_t* tmp, int x, int y, int w, int h)
{
    // Vertical dot product at (y, x), zero-padding out-of-bounds
    static const int16_t k[5] = {1, 4, 7, 4, 1};
    int32_t acc = 0;
    for (int ky = -2; ky <= 2; ++ky) {
        int iy = y + ky;
        if (iy >= 0 && iy < h)
            acc += (int32_t)tmp[iy * w + x] * k[ky + 2];
    }
    int32_t result = (acc + 144) / 289;  // round-to-nearest, divide by 17²
    if (result > 255) result = 255;
    if (result < 0)   result = 0;
    return (uint8_t)result;
}

// ─────────────────────────────────────────────────────────────────────────────
extern "C"
void gaussian_5x5_sep_rvv(const uint8_t* __restrict__ src,
                           uint8_t*       __restrict__ dst,
                           int w, int h)
{
    // ── Temp buffer: int16_t pass-1 results ───────────────────────────────────
    int16_t* tmp = static_cast<int16_t*>(
        aligned_alloc(64, ((size_t)w * (size_t)h * sizeof(int16_t) + 63) & ~63ULL));
    if (!tmp) return;

    // ═════════════════════════════════════════════════════════════════════════
    // PASS 1 — horizontal convolution, store into tmp
    // ═════════════════════════════════════════════════════════════════════════
    // k1d coefficients as scalars (used with vmul_vx)
    // k = [1, 4, 7, 4, 1]  — offsets [-2, -1, 0, +1, +2]
    for (int y = 0; y < h; ++y) {
        const uint8_t* src_row = src + y * w;
        int16_t*       tmp_row = tmp + y * w;

        // Left border: x = 0, 1 (need left neighbours that don't exist)
        for (int x = 0; x < 2 && x < w; ++x)
            tmp_row[x] = pass1_scalar(src_row, x, w);

        // Interior: x = 2 .. w-3 — fully vectorised, all 5 neighbours valid
        for (int x = 2; x < w - 2; ) {
            int remaining = (w - 2) - x;
            if (remaining <= 0) break;

            // vsetvl for u8m1: vl = number of u8 elements this HW can process
            size_t vl = __riscv_vsetvl_e8m1((size_t)remaining);

            // Load 5 windows at offsets -2,-1,0,+1,+2
            // All loads are in-bounds for interior x (x-2 >= 0, x+2 <= w-1)
            vuint8m1_t v_m2 = __riscv_vle8_v_u8m1(src_row + x - 2, vl);
            vuint8m1_t v_m1 = __riscv_vle8_v_u8m1(src_row + x - 1, vl);
            vuint8m1_t v_0  = __riscv_vle8_v_u8m1(src_row + x,     vl);
            vuint8m1_t v_p1 = __riscv_vle8_v_u8m1(src_row + x + 1, vl);
            vuint8m1_t v_p2 = __riscv_vle8_v_u8m1(src_row + x + 2, vl);

            // Zero-extend u8m1 → i16m2 (LMUL doubles: m1 → m2)
            vint16m2_t w_m2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(v_m2, vl));
            vint16m2_t w_m1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(v_m1, vl));
            vint16m2_t w_0  = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(v_0,  vl));
            vint16m2_t w_p1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(v_p1, vl));
            vint16m2_t w_p2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(v_p2, vl));

            // acc = 1×w_m2 + 4×w_m1 + 7×w_0 + 4×w_p1 + 1×w_p2
            // Start from the centre (coefficient 7) to minimise adds
            vint16m2_t acc = __riscv_vmul_vx_i16m2(w_0,  7, vl);
            acc = __riscv_vadd_vv_i16m2(acc, __riscv_vmul_vx_i16m2(w_m1, 4, vl), vl);
            acc = __riscv_vadd_vv_i16m2(acc, __riscv_vmul_vx_i16m2(w_p1, 4, vl), vl);
            acc = __riscv_vadd_vv_i16m2(acc, w_m2, vl);   // coeff 1 → just add
            acc = __riscv_vadd_vv_i16m2(acc, w_p2, vl);   // coeff 1 → just add

            // Store i16m2 result into tmp (max value = 4335, fits int16_t)
            __riscv_vse16_v_i16m2(tmp_row + x, acc, vl);

            x += (int)vl;
        }

        // Right border: x = w-2, w-1
        for (int x = w - 2; x < w; ++x) {
            if (x < 2) continue;  // guard for tiny images
            tmp_row[x] = pass1_scalar(src_row, x, w);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // PASS 2 — vertical convolution + divide by 289
    // ═════════════════════════════════════════════════════════════════════════
    // For interior rows (y=2..h-3), all 5 vertical neighbours are valid.
    // We vectorise across x: load 5 row-chunks from tmp, multiply by k1d,
    // accumulate in i32m2, divide by 289, narrow to u8.

    // Top border: rows 0, 1
    for (int y = 0; y < 2 && y < h; ++y)
        for (int x = 0; x < w; ++x)
            dst[y * w + x] = pass2_scalar(tmp, x, y, w, h);

    // Interior rows
    for (int y = 2; y < h - 2; ++y) {
        const int row_base = y * w;

        // Pointers to the 5 rows of tmp needed for vertical kernel
        const int16_t* row_m2 = tmp + (y - 2) * w;
        const int16_t* row_m1 = tmp + (y - 1) * w;
        const int16_t* row_0  = tmp + y * w;
        const int16_t* row_p1 = tmp + (y + 1) * w;
        const int16_t* row_p2 = tmp + (y + 2) * w;

        for (int x = 0; x < w; ) {
            int remaining = w - x;

            // vsetvl for i16m1 loads: vl matches the i16 load and i32m2 widening
            size_t vl = __riscv_vsetvl_e16m1((size_t)remaining);

            // Load 5 rows from tmp as i16m1
            vint16m1_t r_m2 = __riscv_vle16_v_i16m1(row_m2 + x, vl);
            vint16m1_t r_m1 = __riscv_vle16_v_i16m1(row_m1 + x, vl);
            vint16m1_t r_0  = __riscv_vle16_v_i16m1(row_0  + x, vl);
            vint16m1_t r_p1 = __riscv_vle16_v_i16m1(row_p1 + x, vl);
            vint16m1_t r_p2 = __riscv_vle16_v_i16m1(row_p2 + x, vl);

            // Widen i16m1 → i32m2 before multiply to avoid i16 overflow
            // (max i16 value 4335 × coeff 7 = 30345, would overflow i16 max 32767
            //  at the accumulation step if we stayed in 16-bit)
            vint32m2_t w_m2 = __riscv_vsext_vf2_i32m2(r_m2, vl);
            vint32m2_t w_m1 = __riscv_vsext_vf2_i32m2(r_m1, vl);
            vint32m2_t w_0  = __riscv_vsext_vf2_i32m2(r_0,  vl);
            vint32m2_t w_p1 = __riscv_vsext_vf2_i32m2(r_p1, vl);
            vint32m2_t w_p2 = __riscv_vsext_vf2_i32m2(r_p2, vl);

            // acc = 1×r_m2 + 4×r_m1 + 7×r_0 + 4×r_p1 + 1×r_p2
            vint32m2_t acc = __riscv_vmul_vx_i32m2(w_0,  7, vl);
            acc = __riscv_vadd_vv_i32m2(acc, __riscv_vmul_vx_i32m2(w_m1, 4, vl), vl);
            acc = __riscv_vadd_vv_i32m2(acc, __riscv_vmul_vx_i32m2(w_p1, 4, vl), vl);
            acc = __riscv_vadd_vv_i32m2(acc, w_m2, vl);  // coeff 1
            acc = __riscv_vadd_vv_i32m2(acc, w_p2, vl);  // coeff 1

            // Add rounding bias (144 = floor(289/2)), then divide by 289
            acc = __riscv_vadd_vx_i32m2(acc, 144, vl);
            acc = __riscv_vdiv_vx_i32m2(acc, 289, vl);   // signed div (values ≥ 0)

            // Clamp to [0, 255]
            acc = __riscv_vmax_vx_i32m2(acc,   0, vl);
            acc = __riscv_vmin_vx_i32m2(acc, 255, vl);

            // Narrow i32m2 → u16m1 → u8mf2
            vuint16m1_t narrow16 = __riscv_vnclipu_wx_u16m1(
                __riscv_vreinterpret_v_i32m2_u32m2(acc), 0, __RISCV_VXRM_RDN, vl);
            vuint8mf2_t result   = __riscv_vnclipu_wx_u8mf2(
                narrow16, 0, __RISCV_VXRM_RDN, vl);

            __riscv_vse8_v_u8mf2(dst + row_base + x, result, vl);

            x += (int)vl;
        }
    }

    // Bottom border: rows h-2, h-1
    for (int y = h - 2; y < h; ++y) {
        if (y < 2) continue;
        for (int x = 0; x < w; ++x)
            dst[y * w + x] = pass2_scalar(tmp, x, y, w, h);
    }

    free(tmp);
}

#endif // __riscv_v