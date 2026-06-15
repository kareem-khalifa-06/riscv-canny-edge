#ifdef __riscv_v

#include "../src/sobel.h"
#include <riscv_vector.h>
#include <cstdint>

static const int8_t KX[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
static const int8_t KY[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

// ─────────────────────────────────────────────────────────────────────────────
// Scalar fallback for a single Sobel pixel — used for column boundaries.
// Ensures exact match with the scalar reference on edge pixels.
// ─────────────────────────────────────────────────────────────────────────────
static inline void sobel_scalar_pixel(const uint8_t* src,
                                      int16_t* Gx, int16_t* Gy,
                                      int w, int h, int y, int x)
{
    int sum_x = 0, sum_y = 0;
    for (int ky = -1; ky <= 1; ++ky) {
        int py = y + ky;
        if (py < 0 || py >= h) continue;
        for (int kx = -1; kx <= 1; ++kx) {
            int px = x + kx;
            if (px < 0 || px >= w) continue;
            uint8_t pix = src[py * w + px];
            sum_x += pix * KX[ky + 1][kx + 1];
            sum_y += pix * KY[ky + 1][kx + 1];
        }
    }
    Gx[y * w + x] = static_cast<int16_t>(sum_x);
    Gy[y * w + x] = static_cast<int16_t>(sum_y);
}

// ─────────────────────────────────────────────────────────────────────────────
// sobel_rvv  —  LMUL=2 vectorised Sobel Gx/Gy
//
// DESIGN: strip-mine interior columns, loading 3 contiguous vectors per
// kernel row (left/center/right at offsets -1/0/+1).  This avoids the
// expensive vslidedown that the previous implementation used — on QEMU
// vslidedown is emulated element-by-element and dominates runtime.
//
// 3 separate vle8 loads are cheaper than 1 vle8(vl+2) + 2 vslidedown.
// Each vector element does 9 multiply-accumulates (3x3 kernel).
//
// LMUL=2 (e16m2) chosen because:
//   - We need 2 accumulators (Gx, Gy) + 3 pixel vectors per kernel row
//   - Each pixel vector widens from u8m1 → i16m2
//   - m2 gives 16 logical registers — enough for temporaries without spills.
// ─────────────────────────────────────────────────────────────────────────────
void sobel_rvv(const uint8_t* __restrict__ src,
               int16_t* __restrict__ Gx,
               int16_t* __restrict__ Gy,
               int w, int h)
{
    for (int y = 0; y < h; ++y) {
        const int row_base = y * w;

        // Leftmost column: scalar fallback (neighbour x-1 is out of bounds)
        sobel_scalar_pixel(src, Gx, Gy, w, h, y, 0);

        // Interior columns: vectorised strip-mining
        for (int x = 1; x < w - 1; ) {
            int remaining = (w - 1) - x;
            if (remaining <= 0) break;

            size_t vl = __riscv_vsetvl_e16m2((size_t)remaining);

            vint16m2_t acc_gx = __riscv_vmv_v_x_i16m2(0, vl);
            vint16m2_t acc_gy = __riscv_vmv_v_x_i16m2(0, vl);

            for (int ky = -1; ky <= 1; ++ky) {
                int py = y + ky;
                if (py < 0 || py >= h) continue;

                const uint8_t* row_ptr = src + py * w;
                const int krow = ky + 1;

                // ── 3 direct loads at offsets -1, 0, +1 ──
                // NO vslidedown — each load is a simple contiguous vle8.
                vuint8m1_t v8_l = __riscv_vle8_v_u8m1(row_ptr + x - 1, vl);
                vuint8m1_t v8_c = __riscv_vle8_v_u8m1(row_ptr + x,     vl);
                vuint8m1_t v8_r = __riscv_vle8_v_u8m1(row_ptr + x + 1, vl);

                // Zero-extend u8 → i16 (widens LMUL: m1 → m2)
                vint16m2_t vl16 = __riscv_vreinterpret_v_u16m2_i16m2(
                    __riscv_vzext_vf2_u16m2(v8_l, vl));
                vint16m2_t vc16 = __riscv_vreinterpret_v_u16m2_i16m2(
                    __riscv_vzext_vf2_u16m2(v8_c, vl));
                vint16m2_t vr16 = __riscv_vreinterpret_v_u16m2_i16m2(
                    __riscv_vzext_vf2_u16m2(v8_r, vl));

                int8_t kxl = KX[krow][0], kxc = KX[krow][1], kxr = KX[krow][2];
                int8_t kyl = KY[krow][0], kyc = KY[krow][1], kyr = KY[krow][2];

                // Gx accumulator
                if (kxl != 0) {
                    vint16m2_t prod = __riscv_vmul_vx_i16m2(vl16, (int16_t)kxl, vl);
                    acc_gx = __riscv_vadd_vv_i16m2(acc_gx, prod, vl);
                }
                if (kxc != 0) {
                    vint16m2_t prod = __riscv_vmul_vx_i16m2(vc16, (int16_t)kxc, vl);
                    acc_gx = __riscv_vadd_vv_i16m2(acc_gx, prod, vl);
                }
                if (kxr != 0) {
                    vint16m2_t prod = __riscv_vmul_vx_i16m2(vr16, (int16_t)kxr, vl);
                    acc_gx = __riscv_vadd_vv_i16m2(acc_gx, prod, vl);
                }

                // Gy accumulator
                if (kyl != 0) {
                    vint16m2_t prod = __riscv_vmul_vx_i16m2(vl16, (int16_t)kyl, vl);
                    acc_gy = __riscv_vadd_vv_i16m2(acc_gy, prod, vl);
                }
                if (kyc != 0) {
                    vint16m2_t prod = __riscv_vmul_vx_i16m2(vc16, (int16_t)kyc, vl);
                    acc_gy = __riscv_vadd_vv_i16m2(acc_gy, prod, vl);
                }
                if (kyr != 0) {
                    vint16m2_t prod = __riscv_vmul_vx_i16m2(vr16, (int16_t)kyr, vl);
                    acc_gy = __riscv_vadd_vv_i16m2(acc_gy, prod, vl);
                }
            }

            __riscv_vse16_v_i16m2(Gx + row_base + x, acc_gx, vl);
            __riscv_vse16_v_i16m2(Gy + row_base + x, acc_gy, vl);

            x += (int)vl;
        }

        // Rightmost column: scalar fallback (neighbour x+1 is out of bounds)
        if (w > 1) {
            sobel_scalar_pixel(src, Gx, Gy, w, h, y, w - 1);
        }
    }
}

#endif // __riscv_v
