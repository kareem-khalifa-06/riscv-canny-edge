#ifdef __riscv_v

#include "../src/sobel.h"
#include <riscv_vector.h>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// sobel_rvv  —  vectorised Sobel Gx/Gy computation
//
// DESIGN:
//   For each output row y, we iterate across columns in strip-mined chunks.
//   Each output pixel is a 3×3 convolution with the Sobel-X and Sobel-Y kernels.
//   We compute Gx and Gy simultaneously since they share the same memory access
//   pattern (same 3×3 neighbourhood loaded once, used for both kernels).
//
// KERNELS:
//   Gx (vertical-edge detector):   [-1  0  1]    Gy (horizontal-edge detector):  [-1 -2 -1]
//                                  [-2  0  2]                                   [ 0  0  0]
//                                  [-1  0  1]                                   [ 1  2  1]
//
// LMUL CHOICE:
//   We use LMUL=1 (e8m1 → e16m2 after widen).  The accumulator stays at i16m2.
//   32 registers available — we need ~6 for pixel loads + 2 accumulators (Gx, Gy).
//   Plenty of headroom, no spill risk.
//
// CORRECTNESS:
//   The scalar sobel() uses zero-padding boundary handling.  Our RVV version
//   matches this exactly by skipping out-of-bounds rows/columns (equivalent to
//   multiplying by 0).  Interior pixels are fully vectorised; the 1-pixel border
//   falls back to the scalar reference — a clean and correct approach.
//
// OUTPUT RANGE:
//   Max |Gx| = |−1×0 + 0×0 + 1×255 −2×0 + 0×0 + 2×255 −1×0 + 0×0 + 1×255| = 1020
//   Fits int16_t (max 32767) with comfortable margin.
// ─────────────────────────────────────────────────────────────────────────────

// Sobel X kernel coefficients (flat array for indexed access)
static const int8_t KX[3][3] = {
    {-1,  0,  1},
    {-2,  0,  2},
    {-1,  0,  1}
};

// Sobel Y kernel coefficients
static const int8_t KY[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

// Fallback scalar Sobel for a single pixel — handles the 1-pixel border where
// the vector path would need gather/scatter or complex masking.
static inline void sobel_scalar_pixel(const uint8_t* src,
                                       int16_t* Gx, int16_t* Gy,
                                       int w, int h, int y, int x)
{
    int sum_x = 0;
    int sum_y = 0;
    for (int ky = -1; ky <= 1; ++ky) {
        int py = y + ky;
        if (py < 0 || py >= h) continue;  // zero-padding: skip out-of-bounds row
        for (int kx = -1; kx <= 1; ++kx) {
            int px = x + kx;
            if (px < 0 || px >= w) continue;  // zero-padding: skip out-of-bounds col
            uint8_t pix = src[py * w + px];
            sum_x += pix * KX[ky + 1][kx + 1];
            sum_y += pix * KY[ky + 1][kx + 1];
        }
    }
    Gx[y * w + x] = static_cast<int16_t>(sum_x);
    Gy[y * w + x] = static_cast<int16_t>(sum_y);
}

// ─────────────────────────────────────────────────────────────────────────────
// sobel_rvv  —  main vectorised entry point
// ─────────────────────────────────────────────────────────────────────────────
void sobel_rvv(const uint8_t* __restrict__ src,
               int16_t* __restrict__ Gx,
               int16_t* __restrict__ Gy,
               int w, int h)
{
    // Process each row
    for (int y = 0; y < h; ++y) {
        const int row_base = y * w;

        // ── Handle left border (x = 0) with scalar fallback ──────────────────
        // The vector path needs x-1 to be valid; at x=0 this would underflow.
        sobel_scalar_pixel(src, Gx, Gy, w, h, y, 0);

        // ── Vectorised interior: x = 1 .. w-2 ────────────────────────────────
        // We process in strip-mined chunks starting at x=1.
        // For each chunk we load vl pixels from columns [x-1, x+vl] —
        // the extra pixel at each end is the kernel radius overlap.
        for (int x = 1; x < w - 1; ) {
            // Elements to process in this strip (not including right border)
            int remaining = (w - 1) - x;   // exclusive of w-1
            if (remaining <= 0) break;

            // vl = number of OUTPUT pixels we can produce this iteration
            // For each output we need to load 3 input pixels (x-1, x, x+1).
            // The limiting factor is the output count, not the load count.
            size_t vl = __riscv_vsetvl_e16m1((size_t)remaining);

            // Accumulators for Gx and Gy — int16, LMUL=1
            vint16m1_t acc_gx = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t acc_gy = __riscv_vmv_v_x_i16m1(0, vl);

            // Convolve 3×3 neighbourhood
            for (int ky = -1; ky <= 1; ++ky) {
                int py = y + ky;
                if (py < 0 || py >= h) continue;  // zero-padding: entire row is 0

                const uint8_t* row_ptr = src + py * w;
                const int krow = ky + 1;  // kernel row index [0,2]

                // Load vl+2 pixels starting from x-1
                // We need pixels at [x-1, x+vl] inclusive → vl+2 pixels
                // Use a unit-stride load of vl pixels, then handle the edge
                // with a masked load or by ensuring we have headroom.
                // Simpler approach: load vl pixels from x, then shift+load edges.
                // Even simpler: load contiguous vl+2 bytes — safe because x≥1
                // and x+vl < w-1, so x+vl+1 ≤ w-1 which is < w.
                // Actually x+vl could equal w-1, so x+vl+1 = w which is out of bounds.
                // We need to be careful: the rightmost pixel we need is at x+vl.
                // Since x+vl < w-1 (because vl ≤ remaining = w-1-x), x+vl ≤ w-2.
                // So x+vl+1 ≤ w-1, which is the last valid column. We need vl+1 pixels
                // starting from x-1 (indices x-1 to x+vl-1) plus one more at x+vl.
                // Total: vl+2 pixels from x-1 to x+vl inclusive.
                // x+vl ≤ w-2, so x+vl < w-1 < w. Safe to load vl+2 bytes.

                // Load the base segment of vl pixels starting at x-1
                vuint8m1_t px_base = __riscv_vle8_v_u8m1(row_ptr + x - 1, vl + 2);

                // Now we need three shifted views: left (x-1), center (x), right (x+1)
                // Extract them using vslidedown

                // Left column: offset 0 within the loaded window
                vuint8m1_t px_left  = px_base;  // offset 0 — no slide needed for first vl

                // Center column: offset 1 — slide down by 1
                vuint8m1_t px_center = __riscv_vslidedown_vx_u8m1(px_base, 1, vl + 2);

                // Right column: offset 2 — slide down by 2
                vuint8m1_t px_right  = __riscv_vslidedown_vx_u8m1(px_base, 2, vl + 2);

                // NOTE: After slide, only the first vl elements of each vector are valid.
                // The vl+2 load ensures we have enough data; the slides shift valid data
                // into position.  We process vl output pixels, so we use vl as the vector
                // length for all operations after this point.

                // Widen u8 → i16 for arithmetic
                vint16m1_t pxl = __riscv_vreinterpret_v_i16m1(
                    __riscv_vzext_vf2_u16m1(px_left,   vl));
                vint16m1_t pxc = __riscv_vreinterpret_v_i16m1(
                    __riscv_vzext_vf2_u16m1(px_center, vl));
                vint16m1_t pxr = __riscv_vreinterpret_v_i16m1(
                    __riscv_vzext_vf2_u16m1(px_right,  vl));

                // Apply Sobel-X kernel for this row
                // KX[krow][0], KX[krow][1], KX[krow][2]
                int8_t kxl = KX[krow][0];
                int8_t kxc = KX[krow][1];
                int8_t kxr = KX[krow][2];

                if (kxl != 0) {
                    vint16m1_t prod = __riscv_vmul_vx_i16m1(pxl, (int16_t)kxl, vl);
                    acc_gx = __riscv_vadd_vv_i16m1(acc_gx, prod, vl);
                }
                if (kxc != 0) {
                    vint16m1_t prod = __riscv_vmul_vx_i16m1(pxc, (int16_t)kxc, vl);
                    acc_gx = __riscv_vadd_vv_i16m1(acc_gx, prod, vl);
                }
                if (kxr != 0) {
                    vint16m1_t prod = __riscv_vmul_vx_i16m1(pxr, (int16_t)kxr, vl);
                    acc_gx = __riscv_vadd_vv_i16m1(acc_gx, prod, vl);
                }

                // Apply Sobel-Y kernel for this row
                int8_t kyl = KY[krow][0];
                int8_t kyc = KY[krow][1];
                int8_t kyr = KY[krow][2];

                if (kyl != 0) {
                    vint16m1_t prod = __riscv_vmul_vx_i16m1(pxl, (int16_t)kyl, vl);
                    acc_gy = __riscv_vadd_vv_i16m1(acc_gy, prod, vl);
                }
                if (kyc != 0) {
                    vint16m1_t prod = __riscv_vmul_vx_i16m1(pxc, (int16_t)kyc, vl);
                    acc_gy = __riscv_vadd_vv_i16m1(acc_gy, prod, vl);
                }
                if (kyr != 0) {
                    vint16m1_t prod = __riscv_vmul_vx_i16m1(pxr, (int16_t)kyr, vl);
                    acc_gy = __riscv_vadd_vv_i16m1(acc_gy, prod, vl);
                }
            }

            // Store results
            __riscv_vse16_v_i16m1(Gx + row_base + x, acc_gx, vl);
            __riscv_vse16_v_i16m1(Gy + row_base + x, acc_gy, vl);

            x += (int)vl;
        }

        // ── Handle right border (x = w-1) with scalar fallback ───────────────
        if (w > 1) {
            sobel_scalar_pixel(src, Gx, Gy, w, h, y, w - 1);
        }
    }
}

#endif // __riscv_v
