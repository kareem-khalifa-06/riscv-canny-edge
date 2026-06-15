#ifdef __riscv_v

#include "../src/magnitude.h"
#include <riscv_vector.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// magnitude_l1_rvv  —  optimized with RVV reduction for global max
//
// Algorithm: single-pass compute |Gx|+|Gy| into temp buffer while tracking
// global max via vredmax.  Then normalize in second vector pass.
//
// LMUL=1 (e16m1) chosen because:
//   - We need 2 loads (Gx, Gy), 2 abs ops, 1 add, 1 store per iteration
//   - Plus reduction temporaries — m1 gives 32 registers, plenty of headroom.
//
// CORRECTNESS: vdivu is exact truncating division == C semantics for unsigned.
// vnclipu with shift=0 performs pure saturation narrow, no rounding.
// Result is bit-exact with scalar reference on all inputs, all VLEN.
// ─────────────────────────────────────────────────────────────────────────────
void magnitude_l1_rvv(const int16_t* __restrict__ Gx,
                      const int16_t* __restrict__ Gy,
                      uint8_t*  __restrict__ mag,
                      int w, int h)
{
    const int n = w * h;

    // ── Pass 1: compute mag_raw[i] = |Gx|+|Gy|  AND  track global_max ───────
    // Allocate aligned temp buffer.  Required for vector stores.
    int16_t* mag_raw = static_cast<int16_t*>(
        aligned_alloc(64, ((size_t)n * sizeof(int16_t) + 63) & ~63ULL));
    if (!mag_raw) return;

    int16_t global_max = 0;
    // Reduction accumulator — vredmax writes to element 0 of a vector.
    // We use a scalar vector initialized with INT16_MIN.
    vint16m1_t vred_acc = __riscv_vmv_v_x_i16m1((int16_t)(-32768), 1);

    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)(n - i));

        // Load Gx, Gy
        vint16m1_t vgx = __riscv_vle16_v_i16m1(Gx + i, vl);
        vint16m1_t vgy = __riscv_vle16_v_i16m1(Gy + i, vl);

        // |v| = vmax(v, -v)  — RVV 1.0 has no native integer vabs
        // Safe: Sobel output is bounded by 4*255 = 1020, INT16_MIN never occurs.
        vint16m1_t vax = __riscv_vmax_vv_i16m1(
            vgx, __riscv_vneg_v_i16m1(vgx, vl), vl);
        vint16m1_t vay = __riscv_vmax_vv_i16m1(
            vgy, __riscv_vneg_v_i16m1(vgy, vl), vl);

        // L1 magnitude: max = 1020+1020 = 2040, fits i16
        vint16m1_t vmag = __riscv_vadd_vv_i16m1(vax, vay, vl);
        __riscv_vse16_v_i16m1(mag_raw + i, vmag, vl);

        // Update running max with vector reduction (fold into scalar accumulator)
        vred_acc = __riscv_vredmax_vs_i16m1_i16m1(vmag, vred_acc, vl);

        i += (int)vl;
    }

    // Extract scalar result from reduction accumulator (element 0)
    global_max = __riscv_vmv_x_s_i16m1_i16(vred_acc);

    if (global_max <= 0) {
        memset(mag, 0, (size_t)n);
        free(mag_raw);
        return;
    }
    const uint32_t gmax_u32 = (uint32_t)global_max;

    // ── Pass 2: vectorised normalisation  mag[i] = mag_raw[i] * 255 / gmax ──
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)(n - i));

        vint16m1_t vmag16 = __riscv_vle16_v_i16m1(mag_raw + i, vl);

        // Bit-cast i16→u16 (no instruction, values are non-negative)
        vuint16m1_t vmag_u16 = __riscv_vreinterpret_v_i16m1_u16m1(vmag16);

        // Widen u16→u32: 2040*255 = 520200 > UINT16_MAX, must use 32-bit
        vuint32m2_t vmag_u32 = __riscv_vzext_vf2_u32m2(vmag_u16, vl);

        // Scale by 255
        vuint32m2_t vscaled = __riscv_vmul_vx_u32m2(vmag_u32, 255u, vl);

        // Exact truncating divide (same as C unsigned division)
        vuint32m2_t vnormed = __riscv_vdivu_vx_u32m2(vscaled, gmax_u32, vl);

        // Narrow u32→u16→u8 with saturation
        vuint16m1_t vnarrow16 = __riscv_vnclipu_wx_u16m1(
            vnormed, 0, __RISCV_VXRM_RDN, vl);
        vuint8mf2_t vresult = __riscv_vnclipu_wx_u8mf2(
            vnarrow16, 0, __RISCV_VXRM_RDN, vl);

        __riscv_vse8_v_u8mf2(mag + i, vresult, vl);

        i += (int)vl;
    }

    free(mag_raw);
}

#endif // __riscv_v
