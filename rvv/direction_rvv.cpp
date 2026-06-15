
#ifdef __riscv_v

#include "../src/direction.h"
#include <riscv_vector.h>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// direction_rvv  —  vectorised gradient-direction quantisation
//
// Quantises each pixel's gradient direction to one of 4 bins:
//   0 = ~0 deg   (horizontal gradient, |Gy| << |Gx|)
//   1 = ~45 deg  (diagonal, same sign Gx*Gy)
//   2 = ~90 deg  (vertical gradient, |Gy| >> |Gx|)
//   3 = ~135 deg (diagonal, opposite sign Gx*Gy)
//
// Uses integer cross-multiplication instead of atan2():
//   tan(22.5) ~ 2/5,  tan(67.5) ~ 12/5
//   |Gy|*5 < |Gx|*2   → 0
//   |Gy|*5 > |Gx|*12  → 2
//   else               → 1 or 3 (based on sign of Gx*Gy)
//
// LMUL=1 (e16m1) — plenty of registers for this simple kernel.
// ─────────────────────────────────────────────────────────────────────────────
void direction_rvv(const int16_t* __restrict__ Gx,
                   const int16_t* __restrict__ Gy,
                   uint8_t* __restrict__ dir,
                   int w, int h)
{
    const int n = w * h;

    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e16m1((size_t)(n - i));

        // Load Gx and Gy
        vint16m1_t vgx = __riscv_vle16_v_i16m1(Gx + i, vl);
        vint16m1_t vgy = __riscv_vle16_v_i16m1(Gy + i, vl);

        // |Gx|, |Gy| using vmax(v, -v)  — no native i16 vabs in RVV 1.0
        vint16m1_t vax = __riscv_vmax_vv_i16m1(
            vgx, __riscv_vneg_v_i16m1(vgx, vl), vl);
        vint16m1_t vay = __riscv_vmax_vv_i16m1(
            vgy, __riscv_vneg_v_i16m1(vgy, vl), vl);

        // Cross-multiplication: ay*5 vs ax*2, ay*5 vs ax*12
        // We use vint16m1_t for these, max value = 1020*12 = 12240 < INT16_MAX
        vint16m1_t vay5 = __riscv_vmul_vx_i16m1(vay, 5, vl);
        vint16m1_t vax2  = __riscv_vmul_vx_i16m1(vax, 2, vl);
        vint16m1_t vax12 = __riscv_vmul_vx_i16m1(vax, 12, vl);

        // Boolean masks (true = all-ones, false = all-zeroes)
        vbool16_t m_lt_22_5  = __riscv_vmslt_vv_i16m1_b16(vay5, vax2,  vl);   // ~0 deg
        vbool16_t m_gt_67_5  = __riscv_vmsgt_vv_i16m1_b16(vay5, vax12, vl);   // ~90 deg
        vbool16_t m_diag     = __riscv_vmnot_m_b16(
            __riscv_vmor_mm_b16(m_lt_22_5, m_gt_67_5, vl), vl);              // diagonal region

        // For diagonal: determine sign of Gx*Gy using xor of sign bits
        // Gx>=0 && Gy>=0  OR  Gx<<0 && Gy<<0  → 45 deg (1)
        // else → 135 deg (3)
        vbool16_t m_gx_ge0 = __riscv_vmsgt_vx_i16m1_b16(vgx, -1, vl);  // Gx >= 0
        vbool16_t m_gy_ge0 = __riscv_vmsgt_vx_i16m1_b16(vgy, -1, vl);  // Gy >= 0
        vbool16_t m_same_sign = __riscv_vmnot_m_b16(
            __riscv_vmxor_mm_b16(m_gx_ge0, m_gy_ge0, vl), vl);  // same sign

        // Merge: 45 deg only where diagonal AND same_sign
        vbool16_t m_45 = __riscv_vmand_mm_b16(m_diag, m_same_sign, vl);
        vbool16_t m_135 = __riscv_vmand_mm_b16(m_diag,
            __riscv_vmnot_m_b16(m_same_sign, vl), vl);

        // Build result vector as 16-bit first (masks are vbool16_t)
        vuint16m1_t vres16 = __riscv_vmv_v_x_u16m1(0, vl);  // default: 0 deg

        // Where m_gt_67_5 is true → 2 (90 deg)
        vres16 = __riscv_vmerge_vxm_u16m1(vres16, 2, m_gt_67_5, vl);

        // Where m_45 is true → 1 (45 deg)
        vres16 = __riscv_vmerge_vxm_u16m1(vres16, 1, m_45, vl);

        // Where m_135 is true → 3 (135 deg)
        vres16 = __riscv_vmerge_vxm_u16m1(vres16, 3, m_135, vl);

        // Narrow 16-bit result to 8-bit for storage
                              vuint8m1_t vresult = __riscv_vncvt_x_x_w_u8m1(__riscv_vlmul_ext_v_u16m1_u16m2(vres16), vl);

        __riscv_vse8_v_u8m1(dir + i, vresult, vl);

        i += (int)vl;
    }
}

#endif // __riscv_v